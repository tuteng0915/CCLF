"""
EXP-07 Phase 1: Collect x̂_t backbone states at each t for linear probe training.

For each t in t_grid:
  1. Load validation sequences from openwebtext-t5
  2. Embed with T5 encoder → x_clean (B, L, 512), normalize by latent_std=0.2
  3. Sample z_t = t*x_clean + (1-t)*eps
  4. Run ELF backbone → x̂_t
  5. Save (x̂_t, y_tokens) pairs to disk

Output: results/exp07/states/states_t{t:.2f}.pt per t value

Usage (from ELF-torch root):
  CUDA_VISIBLE_DEVICES=6 python experiments/probe_elf/collect_probe_states.py \
    --checkpoint converted/elf_b-owt-baseline_torch.pt \
    --n_seqs 512 --output_dir results/exp07/states
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_model(checkpoint_path: str, device: torch.device):
    from configs.config import load_config_from_yaml
    from modules.model import ELF_models
    from utils.checkpoint_utils import load_checkpoint
    from utils.train_utils import TrainState, get_optimizer

    cfg_path = os.path.join(os.path.dirname(__file__),
                            "../../src/configs/training_configs/eval_exp13.yml")
    config = load_config_from_yaml(cfg_path)

    # Load T5 encoder to get vocab_size and d_model
    from modules.t5_encoder import T5EncoderConfig
    encoder_config = T5EncoderConfig.from_pretrained(config.encoder_model_name)

    # Infer vocab_size from checkpoint (may differ from encoder config's default)
    _ckpt_tmp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _ckpt_params = _ckpt_tmp.get("params", _ckpt_tmp)
    vocab_size_ckpt = _ckpt_params["unembed_kernel"].shape[1]
    del _ckpt_tmp, _ckpt_params

    model = ELF_models[getattr(config, "model", "ELF-B")](
        text_encoder_dim=encoder_config.d_model,
        max_length=getattr(config, "max_length", 1024),
        attn_drop=0.0, proj_drop=0.0,
        num_time_tokens=getattr(config, "num_time_tokens", 4),
        num_self_cond_cfg_tokens=getattr(config, "num_self_cond_cfg_tokens", 4),
        vocab_size=vocab_size_ckpt,
        num_model_mode_tokens=getattr(config, "num_model_mode_tokens", 4),
        bottleneck_dim=getattr(config, "bottleneck_dim", 128),
    ).to(device).eval()

    optimizer = get_optimizer(model, config, lr=1e-4)
    g = torch.Generator(device="cpu").manual_seed(42)
    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=None,
        ema_params1=TrainState.init_ema(model), step=0, epoch=0,
        dropout_generator=g,
    )
    state, _ = load_checkpoint(checkpoint_path, state)

    return state.model, encoder_config, config


def load_encoder(encoder_model_name: str, device: torch.device):
    from modules.t5_encoder import get_encoder
    encoder_config, encoder = get_encoder(encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device).eval()
    return encoder, encoder_config


def load_data(n_seqs: int, max_length: int):
    """Return (input_ids, attn_masks) as numpy arrays, truncated/padded to max_length."""
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset("embedded-language-flows/openwebtext-t5",
                         split="train", streaming=True)
    ids_list, mask_list = [], []
    for i, ex in enumerate(ds):
        if i >= n_seqs:
            break
        ids = ex["input_ids"][:max_length]
        L = len(ids)
        pad = max_length - L
        ids_padded = ids + [0] * pad
        mask = [1] * L + [0] * pad
        ids_list.append(ids_padded)
        mask_list.append(mask)
    return np.array(ids_list, dtype=np.int64), np.array(mask_list, dtype=np.float32)


@torch.no_grad()
def collect_states(model, encoder, config, input_ids_np, attn_mask_np,
                   t_values, batch_size, latent_std, device, output_dir):
    """Collect and save x̂_t states for each t in t_values."""
    os.makedirs(output_dir, exist_ok=True)
    n_seqs = input_ids_np.shape[0]
    n_batches = (n_seqs + batch_size - 1) // batch_size

    # Step 1: encode all sequences with T5 to get x_clean
    print(f"[EXP-07] Encoding {n_seqs} sequences with T5...")
    x_clean_list, y_tokens_list = [], []
    for b in range(n_batches):
        sl = slice(b * batch_size, min((b + 1) * batch_size, n_seqs))
        ids = torch.tensor(input_ids_np[sl], dtype=torch.long, device=device)
        mask = torch.tensor(attn_mask_np[sl], dtype=torch.float32, device=device)
        latents = encoder(input_ids=ids, attention_mask=mask, deterministic=True)
        x_clean = latents / latent_std  # normalize (latent_mean=0)
        x_clean_list.append(x_clean.cpu())
        y_tokens_list.append(ids.cpu())
        if b % 5 == 0:
            print(f"  batch {b+1}/{n_batches}")

    x_clean_all = torch.cat(x_clean_list, dim=0).float()  # (N, L, 512)
    y_tokens_all = torch.cat(y_tokens_list, dim=0)         # (N, L)
    attn_mask_all = torch.tensor(attn_mask_np, dtype=torch.float32)
    print(f"  x_clean shape: {x_clean_all.shape}")

    # Step 2: for each t, create z_t and run backbone
    for t_val in t_values:
        print(f"\n[EXP-07] t={t_val:.3f} ...")
        x_hat_list = []
        for b in range(n_batches):
            sl = slice(b * batch_size, min((b + 1) * batch_size, n_seqs))
            x_c = x_clean_all[sl].to(device)        # (B, L, d)
            attn = attn_mask_all[sl].to(device)
            B, L, d = x_c.shape

            eps = torch.randn_like(x_c)
            z_t = t_val * x_c + (1.0 - t_val) * eps  # oracle forward noise

            zeros_sc = torch.zeros_like(z_t)
            z_in = torch.cat([z_t, zeros_sc], dim=-1)  # (B, L, 2d)

            t_batch = torch.full((B,), t_val, dtype=torch.float32, device=device)
            sc_scale = torch.ones(B, dtype=torch.float32, device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                x_hat, _, _ = model(z_in, t_batch, decoder_step_active=None,
                                    self_cond_cfg_scale=sc_scale,
                                    attention_mask=attn)
            x_hat_list.append(x_hat.float().cpu())

        x_hat_all = torch.cat(x_hat_list, dim=0)  # (N, L, d)
        out_path = os.path.join(output_dir, f"states_t{t_val:.3f}.pt")
        torch.save({
            "t": t_val,
            "x_hat": x_hat_all,
            "y_tokens": y_tokens_all,
            "attn_mask": attn_mask_all,
        }, out_path)
        print(f"  Saved {out_path}  x_hat shape={x_hat_all.shape}")


def main():
    parser = argparse.ArgumentParser(description="EXP-07: collect backbone states for linear probe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_seqs", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--latent_std", type=float, default=0.2)
    parser.add_argument("--n_t_steps", type=int, default=20,
                        help="Number of equally-spaced t values from 0.05 to 1.00")
    parser.add_argument("--output_dir", default="results/exp07/states")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-07] Device: {device}")

    print("[EXP-07] Loading model ...")
    model, encoder_config, config = load_model(args.checkpoint, device)

    print("[EXP-07] Loading encoder ...")
    encoder, _ = load_encoder(getattr(config, "encoder_model_name", "t5-small"), device)

    print(f"[EXP-07] Loading {args.n_seqs} sequences ...")
    input_ids_np, attn_mask_np = load_data(args.n_seqs, args.max_length)
    print(f"  input_ids shape: {input_ids_np.shape}")

    t_values = np.linspace(0.05, 1.00, args.n_t_steps).tolist()
    print(f"[EXP-07] t_values: {[f'{t:.2f}' for t in t_values]}")

    collect_states(
        model=model, encoder=encoder, config=config,
        input_ids_np=input_ids_np, attn_mask_np=attn_mask_np,
        t_values=t_values, batch_size=args.batch_size,
        latent_std=args.latent_std, device=device,
        output_dir=args.output_dir,
    )
    print("\n[EXP-07] State collection complete.")


if __name__ == "__main__":
    main()
