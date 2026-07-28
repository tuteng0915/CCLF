"""
probe_traj_stability_v2.py — EXP-24v2: LangFlow trajectory with per-step entropy

Extends EXP-24 by recording per-step:
  - top-1 probability
  - entropy H(p_t)
  - top-1 / top-2 margin
  - KL divergence between consecutive posteriors

Also corrects the ELF comparison baseline (uses EXP-14v2 numbers, not EXP-14).

Usage:
  CUDA_VISIBLE_DEVICES=6 conda run -n elf python \
    experiments/probe_langflow/probe_traj_stability_v2.py \
    --checkpoint Continuous-Rivals-Discrete/langflow-owt \
    --n_samples 64 --seq_len 128 --num_steps 32 \
    --out_dir results/exp24v2_langflow
"""

import argparse, json, sys, os, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROBE_DIR = Path(__file__).parent
_LF_SRC = _PROBE_DIR.parents[1] / "models" / "LangFlow"
sys.path.insert(0, str(_LF_SRC))
sys.path.insert(0, str(_PROBE_DIR))

from probe_langflow import load_langflow, encode_with_langflow, load_owt_texts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_trajectory_with_entropy(model, num_samples, seq_length, num_steps):
    """
    Run LangFlow Euler-EDM, recording per-step:
      - argmax token IDs
      - top-1 probability
      - entropy
      - top-1 margin (p[0] - p[1])
      - KL from previous step's distribution
    """
    embed_dim = model.config.hidden_size
    eps = 1e-5
    z = torch.randn(num_samples, seq_length, embed_dim, device=device)
    t_schedule = torch.linspace(1.0 - eps, eps, num_steps, device=device)
    gamma = model.proposal(t_schedule)

    x_self_cond = None

    # Per-step records
    traj_argmax = []   # list of [N, L] int tensors
    traj_top1p  = []   # list of [N, L] float (top-1 prob)
    traj_entropy = []  # list of [N, L] float (entropy, nats)
    traj_margin  = []  # list of [N, L] float (top1-top2)
    traj_kl      = []  # list of float (mean KL from prev step)

    prev_probs = None

    with torch.no_grad():
        for i in range(len(gamma)):
            is_last = (i == len(gamma) - 1)
            if is_last:
                gamma_t = gamma[i]
                gamma_expanded = gamma_t.unsqueeze(0).expand(num_samples)
                logits = model(noisy_embeds=z, timesteps=gamma_expanded,
                               x_self_cond=x_self_cond, return_dict=False)
            else:
                gamma_t = gamma[i]
                gamma_s = gamma[i + 1]
                gamma_expanded = gamma_t.unsqueeze(0).expand(num_samples)
                logits = model(noisy_embeds=z, timesteps=gamma_expanded,
                               x_self_cond=x_self_cond, return_dict=False)

            probs = F.softmax(logits.float(), dim=-1)  # [N, L, V]

            # Argmax
            argmax_ids = probs.argmax(dim=-1)  # [N, L]

            # Top-1 probability
            top1_prob = probs.max(dim=-1).values  # [N, L]

            # Entropy (nats)
            log_probs = torch.log(probs.clamp(min=1e-10))
            entropy = -(probs * log_probs).sum(dim=-1)  # [N, L]

            # Top-1 / Top-2 margin
            top2_vals = probs.topk(2, dim=-1).values  # [N, L, 2]
            margin = top2_vals[..., 0] - top2_vals[..., 1]  # [N, L]

            # KL from previous step
            if prev_probs is not None:
                # KL(prev || curr) per position, then mean
                kl = (prev_probs * (torch.log(prev_probs.clamp(1e-10)) -
                                    torch.log(probs.clamp(1e-10)))).sum(-1)
                mean_kl = float(kl.mean().item())
            else:
                mean_kl = float("nan")

            # Record (move to CPU to avoid OOM)
            traj_argmax.append(argmax_ids.cpu())
            traj_top1p.append(top1_prob.cpu().float())
            traj_entropy.append(entropy.cpu().float())
            traj_margin.append(margin.cpu().float())
            traj_kl.append(mean_kl)

            # Keep prev_probs for next iteration
            prev_probs = probs.clone()

            # Step forward (except final step)
            if not is_last:
                x_pred = model._embed_tokens(probs)
                if model.config.self_conditioning:
                    x_self_cond = x_pred
                z = model._euler_edm_step(z, x_pred, gamma_t, gamma_s)

    return (
        torch.stack(traj_argmax,  dim=0),   # [T, N, L] int
        torch.stack(traj_top1p,   dim=0),   # [T, N, L] float
        torch.stack(traj_entropy, dim=0),   # [T, N, L] float
        torch.stack(traj_margin,  dim=0),   # [T, N, L] float
        traj_kl,                            # list of T floats
    )


def analyze_trajectory(traj_argmax, traj_top1p, traj_entropy, traj_margin, traj_kl):
    """Full analysis matching EXP-14v2 format + new entropy/probability metrics."""
    T, N, L = traj_argmax.shape

    # Flip statistics (argmax stability)
    flips = (traj_argmax[1:] != traj_argmax[:-1])   # [T-1, N, L]
    flip_count = flips.long().sum(0)                   # [N, L]

    last_flip = torch.zeros(N, L, dtype=torch.long)
    for step in range(T - 1):
        mask = flips[step]
        last_flip[mask] = step + 1

    # Per-step mean statistics
    steps = list(range(T))
    per_step = []
    for i in steps:
        per_step.append({
            "step": i,
            "mean_top1_prob": float(traj_top1p[i].mean().item()),
            "p10_top1_prob":  float(traj_top1p[i].flatten().quantile(0.10).item()),
            "mean_entropy":   float(traj_entropy[i].mean().item()),
            "p90_entropy":    float(traj_entropy[i].flatten().quantile(0.90).item()),
            "mean_margin":    float(traj_margin[i].mean().item()),
            "mean_kl_from_prev": traj_kl[i],
        })

    # When does top-1 prob first exceed threshold?
    high_conf_steps = {}
    for thresh in [0.5, 0.9, 0.99]:
        first_high = (traj_top1p > thresh).float().argmax(dim=0).float()
        # Mask positions that never reach threshold
        never = ~(traj_top1p > thresh).any(dim=0)
        first_high[never] = T  # sentinel
        high_conf_steps[f"first_step_p>{thresh}"] = {
            "mean":   float(first_high.mean().item()),
            "frac_never": float(never.float().mean().item()),
        }

    return {
        # Argmax flip stats
        "n_flips_geq1":   float((flip_count >= 1).float().mean().item()),
        "n_flips_geq5":   float((flip_count >= 5).float().mean().item()),
        "n_flips_geq10":  float((flip_count >= 10).float().mean().item()),
        "n_flips_0":      float((flip_count == 0).float().mean().item()),
        "mean_flip_count": float(flip_count.float().mean().item()),
        "mean_last_flip_frac": float(last_flip.float().mean().item()) / (T - 1),

        # Per-step entropy/probability
        "per_step": per_step,

        # High-confidence thresholds
        "high_conf_thresholds": high_conf_steps,

        # Summary: entropy at representative steps
        "entropy_step0":   float(traj_entropy[0].mean().item()),
        "entropy_step8":   float(traj_entropy[min(8, T-1)].mean().item()),
        "entropy_step16":  float(traj_entropy[min(16, T-1)].mean().item()),
        "entropy_step24":  float(traj_entropy[min(24, T-1)].mean().item()),
        "entropy_step31":  float(traj_entropy[min(31, T-1)].mean().item()),
        "top1p_step0":     float(traj_top1p[0].mean().item()),
        "top1p_step8":     float(traj_top1p[min(8, T-1)].mean().item()),
        "top1p_step16":    float(traj_top1p[min(16, T-1)].mean().item()),
        "top1p_step24":    float(traj_top1p[min(24, T-1)].mean().item()),
        "top1p_step31":    float(traj_top1p[min(31, T-1)].mean().item()),

        # flip count distribution
        "flip_count_dist": {str(k): float((flip_count == k).float().mean().item())
                            for k in range(min(15, T))},
        "n_steps": T,
        "n_positions": N * L,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--n_samples",  type=int, default=64)
    p.add_argument("--seq_len",    type=int, default=128)
    p.add_argument("--num_steps",  type=int, default=32)
    p.add_argument("--out_dir",    default="results/exp24v2_langflow")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] LangFlow from {args.checkpoint}")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    print(f"  self_conditioning = {model.config.self_conditioning}  ← CONFIRMED")

    print(f"[traj] Generating {args.num_steps}-step trajectories for {args.n_samples} samples...")
    traj_argmax, traj_top1p, traj_entropy, traj_margin, traj_kl = \
        generate_trajectory_with_entropy(model, args.n_samples, args.seq_len, args.num_steps)

    print("[analyze] Computing stability statistics + entropy profile...")
    stats = analyze_trajectory(traj_argmax, traj_top1p, traj_entropy, traj_margin, traj_kl)

    out_json = out_dir / "traj_entropy_stability.json"
    with open(out_json, "w") as f:
        json.dump({"stats": stats, "args": vars(args)}, f, indent=2)
    print(f"[saved] {out_json}")

    # Print summary
    print("\n── EXP-24v2 Summary ──────────────────────────────────────────────────")
    print(f"n_positions: {stats['n_positions']}  n_steps: {stats['n_steps']}")
    print(f"\nArgmax flip statistics (corrected ELF comparison: EXP-14v2):")
    print(f"  LangFlow flip≥1:  {stats['n_flips_geq1']:.1%}")
    print(f"  LangFlow flip≥5:  {stats['n_flips_geq5']:.1%}   ELF baseline: 67.6%  ELF kd_cr: 48.4%")
    print(f"  mean flip count:  {stats['mean_flip_count']:.2f}  ELF baseline: 6.08  ELF kd_cr: 4.66")
    print(f"  mean last_flip:   {stats['mean_last_flip_frac']:.1%}  ELF baseline: 66%   ELF kd_cr: 60%")

    print(f"\nPer-step entropy profile (nats):")
    print(f"  step 0:  entropy={stats['entropy_step0']:.3f}  top1p={stats['top1p_step0']:.4f}")
    print(f"  step 8:  entropy={stats['entropy_step8']:.3f}  top1p={stats['top1p_step8']:.4f}")
    print(f"  step 16: entropy={stats['entropy_step16']:.3f}  top1p={stats['top1p_step16']:.4f}")
    print(f"  step 24: entropy={stats['entropy_step24']:.3f}  top1p={stats['top1p_step24']:.4f}")
    print(f"  step 31: entropy={stats['entropy_step31']:.3f}  top1p={stats['top1p_step31']:.4f}")

    print(f"\nHigh-confidence thresholds (early commitment diagnostic):")
    for k, v in stats["high_conf_thresholds"].items():
        print(f"  {k}: mean step = {v['mean']:.1f}/{stats['n_steps']}  frac_never = {v['frac_never']:.1%}")

    print("\nNote: self_conditioning=True confirmed — mechanistic explanation in EXP-24 spec invalid.")


if __name__ == "__main__":
    main()
