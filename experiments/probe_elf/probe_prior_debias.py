"""
EXP-05/06: Prior estimation and subtraction.

EXP-05: Estimate q_t(v) via batch-shuffle — break per-instance signal while preserving statistics.
EXP-06: r_t = log p_t - log q_t → G_debiased(t).

Usage:
  CUDA_VISIBLE_DEVICES=7 python experiments/probe_elf/probe_prior_debias.py \
    --checkpoint converted/elf_b-owt-kd-cr_torch.pt \
    --n_seqs 128 --n_t_steps 20 --n_shuffle 8 \
    --output_dir results/exp05_kd_cr
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_model_and_encoder(checkpoint_path, device):
    from configs.config import load_config_from_yaml
    from modules.model import ELF_models
    from modules.t5_encoder import T5EncoderConfig, get_encoder
    from utils.checkpoint_utils import load_checkpoint
    from utils.train_utils import TrainState, get_optimizer

    cfg_path = os.path.join(os.path.dirname(__file__),
                            "../../src/configs/training_configs/eval_exp13.yml")
    config = load_config_from_yaml(cfg_path)
    encoder_config = T5EncoderConfig.from_pretrained(config.encoder_model_name)

    _ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _params = _ckpt.get("params", _ckpt)
    vocab_size = _params["unembed_kernel"].shape[1]
    del _ckpt, _params

    model = ELF_models[getattr(config, "model", "ELF-B")](
        text_encoder_dim=encoder_config.d_model,
        max_length=getattr(config, "max_length", 1024),
        attn_drop=0.0, proj_drop=0.0,
        num_time_tokens=getattr(config, "num_time_tokens", 4),
        num_self_cond_cfg_tokens=getattr(config, "num_self_cond_cfg_tokens", 4),
        vocab_size=vocab_size,
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

    _, encoder = get_encoder(config.encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device).eval()

    return state.model, encoder, config


def load_data(n_seqs, max_length=1024):
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset("embedded-language-flows/openwebtext-t5", split="train", streaming=True)
    ids_list, mask_list = [], []
    for i, ex in enumerate(ds):
        if i >= n_seqs:
            break
        ids = ex["input_ids"][:max_length]
        L = len(ids)
        pad = max_length - L
        ids_list.append(ids + [0] * pad)
        mask_list.append([1] * L + [0] * pad)
    return (np.array(ids_list, dtype=np.int64),
            np.array(mask_list, dtype=np.float32))


@torch.no_grad()
def get_decoder_logprobs(model, x_clean, t_val, device):
    """Oracle pass using decoder_step_active=True. Returns log-softmax (B, L, V)."""
    B, L, d = x_clean.shape
    sc_scale = torch.ones(B, dtype=torch.float32, device=device)
    t_batch = torch.full((B,), t_val, dtype=torch.float32, device=device)
    eps = torch.randn_like(x_clean)
    z = t_val * x_clean + (1.0 - t_val) * eps
    zeros_sc = torch.zeros_like(z)
    z_in = torch.cat([z, zeros_sc], dim=-1)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, decoder_logits, _ = model(z_in, t_batch, decoder_step_active=True,
                                     self_cond_cfg_scale=sc_scale)
    return F.log_softmax(decoder_logits.float(), dim=-1)  # (B, L, V)


def compute_accuracy(log_probs, y_tokens, attn_mask):
    """Masked accuracy of argmax(log_probs) vs y_tokens."""
    pred = log_probs.argmax(dim=-1)  # (B, L)
    correct = (pred == y_tokens) & attn_mask.bool()
    return correct.float().sum().item() / attn_mask.bool().float().sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="converted/elf_b-owt-kd-cr_torch.pt")
    parser.add_argument("--n_seqs", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_t_steps", type=int, default=20)
    parser.add_argument("--n_shuffle", type=int, default=8)
    parser.add_argument("--latent_std", type=float, default=0.2)
    parser.add_argument("--output_dir", default="results/exp05_kd_cr")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-05/06] Device: {device}, n_seqs={args.n_seqs}, n_shuffle={args.n_shuffle}")

    print("[EXP-05/06] Loading model + encoder ...")
    model, encoder, config = load_model_and_encoder(args.checkpoint, device)

    print(f"[EXP-05/06] Loading {args.n_seqs} sequences ...")
    ids_np, mask_np = load_data(args.n_seqs, args.max_length)

    print("[EXP-05/06] Encoding with T5 ...")
    x_clean_list, y_list = [], []
    n_batches = (args.n_seqs + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for b in range(n_batches):
            sl = slice(b * args.batch_size, min((b + 1) * args.batch_size, args.n_seqs))
            ids_t = torch.tensor(ids_np[sl], dtype=torch.long, device=device)
            mask_t = torch.tensor(mask_np[sl], dtype=torch.float32, device=device)
            latents = encoder(input_ids=ids_t, attention_mask=mask_t, deterministic=True)
            x_clean_list.append(latents.float() / args.latent_std)
            y_list.append(ids_t.cpu())

    x_clean_all = torch.cat(x_clean_list, dim=0).cpu()  # (N, L, 512) on CPU
    y_all = torch.cat(y_list, dim=0).cpu()               # (N, L)
    attn_all = torch.tensor(mask_np, dtype=torch.float32)  # (N, L)
    N, L, d = x_clean_all.shape
    print(f"  x_clean: {x_clean_all.shape}")

    t_values = np.linspace(0.05, 1.00, args.n_t_steps).tolist()
    results = []

    for t_val in t_values:
        print(f"\n[EXP-05/06] t={t_val:.4f} ...")

        # Oracle log-probs (p_t)
        log_p_list = []
        for b in range(n_batches):
            sl = slice(b * args.batch_size, min((b + 1) * args.batch_size, N))
            xc = x_clean_all[sl].to(device)
            log_p_list.append(get_decoder_logprobs(model, xc, t_val, device).cpu())
        log_p = torch.cat(log_p_list, dim=0)  # (N, L, V)

        # Shuffle log-probs (q_t)
        log_q_sum = torch.zeros_like(log_p)
        for k in range(args.n_shuffle):
            torch.manual_seed(k * 997 + int(t_val * 1000))
            perm = torch.randperm(N)
            x_shuf = x_clean_all[perm]
            log_q_k_list = []
            for b in range(n_batches):
                sl = slice(b * args.batch_size, min((b + 1) * args.batch_size, N))
                xc = x_shuf[sl].to(device)
                log_q_k_list.append(get_decoder_logprobs(model, xc, t_val, device).cpu())
            log_q_sum += torch.cat(log_q_k_list, dim=0)
        log_q = log_q_sum / args.n_shuffle  # (N, L, V)

        # Debiased score
        r_t = log_p - log_q

        G_oracle   = compute_accuracy(log_p, y_all, attn_all)
        G_prior    = compute_accuracy(log_q, y_all, attn_all)
        G_debiased = compute_accuracy(r_t, y_all, attn_all)

        print(f"  G_oracle={G_oracle:.4f}  G_prior={G_prior:.4f}  G_debiased={G_debiased:.4f}")
        results.append({
            "t": t_val, "G_oracle": G_oracle,
            "G_prior": G_prior, "G_debiased": G_debiased,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "prior_debias.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXP-05/06] Saved to {out_path}")

    print("\n=== Summary ===")
    print(f"{'t':>6}  {'G_oracle':>10}  {'G_prior':>10}  {'G_debiased':>12}  {'debias_delta':>14}")
    for r in results:
        delta = r["G_debiased"] - r["G_oracle"]
        print(f"  {r['t']:.4f}  {r['G_oracle']:>10.4f}  {r['G_prior']:>10.4f}"
              f"  {r['G_debiased']:>12.4f}  {delta:>+14.4f}")


if __name__ == "__main__":
    main()
