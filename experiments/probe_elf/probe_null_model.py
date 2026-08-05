"""
EXP-04: Decoder geometry null model.

Feed isotropic Gaussian z ~ N(0, 1) to the ELF backbone at each t and compute:
  - Mode fraction: fraction of positions that predict the single most common token
  - Top-5 token coverage: fraction of positions whose prediction is one of 5 most-predicted tokens
  - Mean cosine similarity to nearest token centroid (max cosine sim)
  - Entropy of token belief distribution

These characterize any geometric biases in the output head independent of signal.
Compare with oracle-probe G(t) to bound geometry's contribution to G.

Usage (from ELF-torch root):
  CUDA_VISIBLE_DEVICES=4 python experiments/probe_elf/probe_null_model.py \
    --checkpoint converted/elf_b-owt-baseline_torch.pt \
    --output_dir results/exp04 \
    --n_seqs 128 --seq_len 1024 --n_t_steps 20
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_model_and_embeddings(checkpoint_path: str, device: torch.device):
    """Load ELF model and extract output projection matrix for G(t) computation."""
    from configs.config import load_config_from_yaml
    from modules.model import ELF_models
    from transformers import AutoConfig

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = ckpt.get("params", ckpt)

    # Infer text_encoder_dim from unembed_kernel: (text_encoder_dim, vocab_size)
    unembed_kernel = None
    for k in ["unembed_kernel", "model.unembed_kernel"]:
        if k in params:
            unembed_kernel = params[k]
            break
    if unembed_kernel is None:
        raise KeyError("Cannot find unembed_kernel in checkpoint. Run with --list_keys.")

    text_encoder_dim, vocab_size = unembed_kernel.shape
    print(f"  unembed_kernel: ({text_encoder_dim}, {vocab_size})")

    # Use eval_exp13 config for model hyperparameters (ELF-B standard)
    base_cfg_path = os.path.join(
        os.path.dirname(__file__),
        "../../src/configs/training_configs/eval_exp13.yml"
    )
    config = load_config_from_yaml(base_cfg_path)

    # Build model using factory (same pattern as eval.py)
    model = ELF_models[getattr(config, "model", "ELF-B")](
        text_encoder_dim=text_encoder_dim,
        max_length=getattr(config, "max_length", 1024),
        attn_drop=0.0,
        proj_drop=0.0,
        num_time_tokens=getattr(config, "num_time_tokens", 4),
        num_self_cond_cfg_tokens=getattr(config, "num_self_cond_cfg_tokens", 4),
        vocab_size=vocab_size,
        num_model_mode_tokens=getattr(config, "num_model_mode_tokens", 4),
        bottleneck_dim=getattr(config, "bottleneck_dim", 128),
    )

    # Load weights (ignore missing lin_branch)
    missing, _ = model.load_state_dict(params, strict=False)
    non_trivial_missing = [k for k in missing if "lin_branch" not in k]
    if non_trivial_missing:
        print(f"  WARNING: missing keys: {non_trivial_missing[:5]}")

    model = model.to(device).eval()

    # E: (vocab_size, text_encoder_dim) — rows are token "centroids" in embedding space
    E = unembed_kernel.T.float().to(device)  # (vocab_size, text_encoder_dim)
    E_norm = E / (E.norm(dim=-1, keepdim=True) + 1e-8)

    return model, E, E_norm, text_encoder_dim, vocab_size, config


@torch.no_grad()
def probe_null_at_t(model, t_val: float, n_seqs: int, seq_len: int,
                    text_encoder_dim: int, E_norm: torch.Tensor, device: torch.device):
    """
    Feed n_seqs × seq_len of pure Gaussian noise to backbone at time t_val.
    Returns dict of metrics.
    """
    z = torch.randn(n_seqs, seq_len, text_encoder_dim, device=device)
    zeros_sc = torch.zeros_like(z)
    z_in = torch.cat([z, zeros_sc], dim=-1)  # (B, L, 2d) — zero self-cond

    t_batch = torch.full((n_seqs,), t_val, dtype=torch.float32, device=device)
    # Pass self_cond_cfg_scale=1.0 to match training/eval forward (adds sc_cfg tokens so RoPE aligns).
    sc_scale = torch.ones(n_seqs, dtype=torch.float32, device=device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        x_hat, _, _ = model(z_in, t_batch, decoder_step_active=None,
                            self_cond_cfg_scale=sc_scale)  # (B, L, d)

    x_hat = x_hat.float()  # (B, L, d)
    x_hat_flat = x_hat.reshape(-1, text_encoder_dim)  # (B*L, d)

    # Cosine nearest-token
    x_n = x_hat_flat / (x_hat_flat.norm(dim=-1, keepdim=True) + 1e-8)
    cosine_sims = x_n @ E_norm.T  # (B*L, V)
    max_sims, pred_tokens = cosine_sims.max(dim=-1)  # (B*L,)

    # Mode fraction
    pred_np = pred_tokens.cpu().numpy()
    token_counts = np.bincount(pred_np, minlength=E_norm.shape[0])
    mode_count = token_counts.max()
    mode_token = int(token_counts.argmax())
    total = len(pred_np)
    mode_fraction = mode_count / total

    # Top-5 coverage
    top5_counts = np.sort(token_counts)[-5:]
    top5_coverage = top5_counts.sum() / total

    # Unique tokens predicted
    n_unique = int((token_counts > 0).sum())

    # Mean max cosine sim
    mean_max_cos = float(max_sims.mean().cpu())

    return {
        "t": t_val,
        "mode_fraction": float(mode_fraction),
        "mode_token": mode_token,
        "top5_coverage": float(top5_coverage),
        "n_unique_tokens_predicted": n_unique,
        "mean_max_cosine_sim": mean_max_cos,
        "vocab_size": E_norm.shape[0],
        "chance_fraction": 1.0 / E_norm.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-04: null model geometry probe")
    parser.add_argument("--checkpoint", required=True, help="ELF checkpoint path")
    parser.add_argument("--output_dir", default="results/exp04", help="Output directory")
    parser.add_argument("--n_seqs", type=int, default=128, help="Number of sequences (batch size)")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--n_t_steps", type=int, default=20, help="Number of t values to probe")
    parser.add_argument("--list_keys", action="store_true", help="List checkpoint keys and exit")
    args = parser.parse_args()

    if args.list_keys:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        params = ckpt.get("params", ckpt)
        for k, v in sorted(params.items()):
            print(f"  {k}: {getattr(v, 'shape', type(v))}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-04] Device: {device}")
    print(f"[EXP-04] Loading model from {args.checkpoint} ...")
    model, E, E_norm, text_encoder_dim, vocab_size, config = load_model_and_embeddings(
        args.checkpoint, device
    )
    print(f"  Vocab size: {vocab_size}, text_encoder_dim: {text_encoder_dim}")
    print(f"  Chance fraction: {1.0/vocab_size:.6f}")

    t_values = np.linspace(0.05, 1.00, args.n_t_steps).tolist()
    results = []

    print(f"\n[EXP-04] Probing {args.n_t_steps} t values with {args.n_seqs} seqs × {args.seq_len} positions ...")
    for t_val in t_values:
        r = probe_null_at_t(model, t_val, args.n_seqs, args.seq_len,
                            text_encoder_dim, E_norm, device)
        results.append(r)
        print(f"  t={t_val:.3f}: mode_frac={r['mode_fraction']:.4f}  "
              f"top5_cov={r['top5_coverage']:.4f}  "
              f"n_unique={r['n_unique_tokens_predicted']}  "
              f"max_cos={r['mean_max_cosine_sim']:.4f}  "
              f"(chance={r['chance_fraction']:.6f})")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "null_model_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXP-04] Saved results to {out_path}")

    # Summary: decision rule
    mean_mode = np.mean([r["mode_fraction"] for r in results])
    chance = 1.0 / vocab_size
    print(f"\n=== EXP-04 Decision ===")
    print(f"  Mean mode fraction: {mean_mode:.4f}  (chance={chance:.6f})")
    if mean_mode > 0.10:
        print("  => HIGH geometry bias. G(t) values are inflated by output head geometry.")
        print("     Use baseline-corrected G: G_corrected(t) = G(t) - G_null(t).")
    elif mean_mode > 0.02:
        print("  => MODERATE geometry bias. Report G_null alongside G(t) as a baseline.")
    else:
        print("  => LOW geometry bias. G(t) reflects genuine representation quality.")


if __name__ == "__main__":
    main()
