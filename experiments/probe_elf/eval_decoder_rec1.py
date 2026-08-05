"""
EXP-07 Supplement: Compute actual ELF decoder Rec@1 at each t.

The decoder path is:  x (768-dim backbone output)
                      → proj_kernel + GELU
                      → unembed_kernel  →  logits (V=32100)

This is DIFFERENT from x_hat = final_layer(x) (512-dim), which is what
train_linear_probe.py probes. To get true decoder accuracy, we must call
model(..., decoder_step_active=True) and read the returned decoder_logits.

Comparison: probe_acc(t) vs decoder_rec1(t)
  probe_acc > decoder_rec1 → Story A: x_hat encodes tokens beyond what
  the native decoder exposes at time t.

Usage (from ELF-torch root):
  CUDA_VISIBLE_DEVICES=4 python experiments/probe_elf/eval_decoder_rec1.py \
    --states_dir results/exp07/states \
    --checkpoint converted/elf_b-owt-baseline_torch.pt \
    --probe_json results/exp07/probe_accuracies.json \
    --output_dir results/exp07
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

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

    _ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _params = _ckpt.get("params", _ckpt)
    vocab_size_ckpt = _params["unembed_kernel"].shape[1]
    del _ckpt, _params

    encoder_config = T5EncoderConfig.from_pretrained(config.encoder_model_name)
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

    _, encoder = get_encoder(config.encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device).eval()

    return state.model, encoder, config


@torch.no_grad()
def eval_decoder_rec1_at_t(model, encoder, y_tokens, attn_mask, t_val,
                            batch_size, latent_std, device):
    """
    For each sequence in y_tokens:
      1. Encode with T5 → x_clean
      2. Sample z_t = t*x_clean + (1-t)*eps
      3. Call model(z_t, t, decoder_step_active=True) → decoder_logits
      4. Compute Rec@1 = fraction of valid positions where argmax == y_token
    Returns Rec@1 (float).
    """
    N, L = y_tokens.shape
    n_batches = (N + batch_size - 1) // batch_size
    correct = 0
    total = 0

    for b in range(n_batches):
        sl = slice(b * batch_size, min((b + 1) * batch_size, N))
        ids = y_tokens[sl].to(device)           # (B, L)
        mask = attn_mask[sl].to(device)         # (B, L) float

        latents = encoder(input_ids=ids, attention_mask=mask.long(),
                          deterministic=True)
        x_clean = (latents / latent_std).float()  # (B, L, d)
        B, _, d = x_clean.shape

        eps = torch.randn_like(x_clean)
        z_t = t_val * x_clean + (1.0 - t_val) * eps

        zeros_sc = torch.zeros_like(z_t)
        z_in = torch.cat([z_t, zeros_sc], dim=-1)

        t_batch = torch.full((B,), t_val, dtype=torch.float32, device=device)
        sc_scale = torch.ones(B, dtype=torch.float32, device=device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            _, decoder_logits, _ = model(
                z_in, t_batch,
                decoder_step_active=True,
                self_cond_cfg_scale=sc_scale,
                attention_mask=mask.float(),
            )

        if decoder_logits is None:
            raise RuntimeError("decoder_logits is None — check decoder_step_active")

        # (B, L, V) → argmax
        preds = decoder_logits.float().argmax(dim=-1)  # (B, L)
        valid = mask.bool()
        correct += (preds[valid] == ids[valid]).sum().item()
        total += valid.sum().item()

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states_dir", default="results/exp07/states")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe_json", default=None,
                        help="Optional path to probe_accuracies.json for side-by-side table")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--latent_std", type=float, default=0.2)
    parser.add_argument("--output_dir", default="results/exp07")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-07 decoder] Device: {device}")

    print("[EXP-07 decoder] Loading model + encoder ...")
    model, encoder, config = load_model_and_encoder(args.checkpoint, device)

    state_files = sorted(
        f for f in os.listdir(args.states_dir) if f.startswith("states_t") and f.endswith(".pt")
    )
    print(f"[EXP-07 decoder] Found {len(state_files)} state files")

    results = {}
    for fname in state_files:
        data = torch.load(os.path.join(args.states_dir, fname),
                          map_location="cpu", weights_only=False)
        t_val = float(data["t"])
        y_tokens = data["y_tokens"]       # (N, L)
        attn_mask = data["attn_mask"]     # (N, L)

        print(f"  t={t_val:.3f}  N={y_tokens.shape[0]} ...", end="", flush=True)
        rec1 = eval_decoder_rec1_at_t(
            model=model, encoder=encoder,
            y_tokens=y_tokens, attn_mask=attn_mask,
            t_val=t_val, batch_size=args.batch_size,
            latent_std=args.latent_std, device=device,
        )
        results[f"{t_val:.3f}"] = rec1
        print(f"  decoder_rec1={rec1:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "decoder_rec1.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXP-07 decoder] Saved to {out_path}")

    # Side-by-side table
    probe_data = {}
    if args.probe_json and os.path.exists(args.probe_json):
        with open(args.probe_json) as f:
            probe_data = json.load(f)

    print("\n=== EXP-07 decoder Rec@1 vs probe ===")
    print(f"{'t':>6}  {'decoder_rec1':>14}  {'probe_acc':>10}  {'gap(probe-dec)':>16}")
    for t_str, rec1 in sorted(results.items()):
        probe_acc = probe_data.get(t_str, {}).get("probe_acc", float("nan"))
        gap = probe_acc - rec1 if not np.isnan(probe_acc) else float("nan")
        print(f"{float(t_str):6.3f}  {rec1:14.4f}  {probe_acc:10.4f}  {gap:+16.4f}")


if __name__ == "__main__":
    main()
