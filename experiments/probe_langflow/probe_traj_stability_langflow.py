"""
EXP-24: LangFlow Trajectory Stability
LangFlow analog of ELF EXP-14.

Runs LangFlow Euler-EDM sampling for num_steps steps, records argmax(logits)
at each step, then computes position-flip statistics.

Comparison with ELF EXP-14:
  ELF-baseline: 83.4% of positions flip ≥5 times, mean last-flip step = 27/32
  ELF-kd_cr:   ~75% flip ≥5 times, mean last-flip step = ~22/32

Usage (from CCLF root):
  CUDA_VISIBLE_DEVICES=2 conda run -n elf python \
    experiments/probe_langflow/probe_traj_stability_langflow.py \
    --checkpoint Continuous-Rivals-Discrete/langflow-owt \
    --n_samples 64 --seq_len 128 --num_steps 32 \
    --out_dir results/exp24_langflow
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROBE_DIR = Path(__file__).parent
_LF_SRC = _PROBE_DIR.parents[1] / "models" / "LangFlow"
sys.path.insert(0, str(_LF_SRC))
sys.path.insert(0, str(_PROBE_DIR))

from probe_langflow import load_langflow


def generate_trajectory(model, num_samples: int, seq_length: int,
                        num_steps: int, device: torch.device):
    """Run LangFlow Euler-EDM, record argmax(logits) at each of num_steps+1 calls."""
    embed_dim = model.config.hidden_size
    eps = 1e-5
    z = torch.randn(num_samples, seq_length, embed_dim, device=device)
    t_schedule = torch.linspace(1.0 - eps, eps, num_steps, device=device)
    gamma = model.proposal(t_schedule)

    x_self_cond = None
    trajectory = []  # each entry: [N, L] argmax at that step

    with torch.no_grad():
        for i in range(len(gamma) - 1):
            gamma_t = gamma[i]
            gamma_s = gamma[i + 1]
            gamma_expanded = gamma_t.unsqueeze(0).expand(num_samples)

            logits = model(noisy_embeds=z, timesteps=gamma_expanded,
                           x_self_cond=x_self_cond, return_dict=False)

            probs = F.softmax(logits.float(), dim=-1)
            x_pred = model._embed_tokens(probs)

            if model.config.self_conditioning:
                x_self_cond = x_pred

            trajectory.append(logits.argmax(dim=-1).cpu())  # [N, L]
            z = model._euler_edm_step(z, x_pred, gamma_t, gamma_s)

        # Final step
        gamma_expanded = gamma[-1].unsqueeze(0).expand(num_samples)
        logits = model(noisy_embeds=z, timesteps=gamma_expanded,
                       x_self_cond=x_self_cond, return_dict=False)
        trajectory.append(logits.argmax(dim=-1).cpu())

    return torch.stack(trajectory, dim=0)  # [num_steps, N, L]


def analyze_stability(trajectory: torch.Tensor) -> dict:
    """
    trajectory: [T, N, L] int tensor.
    Returns flip statistics matching EXP-14 format.
    """
    T, N, L = trajectory.shape
    flips = (trajectory[1:] != trajectory[:-1]).float()  # [T-1, N, L]
    flip_count = flips.sum(0)  # [N, L]

    # Last flip step (1-indexed)
    last_flip = torch.zeros(N, L, dtype=torch.float32)
    for step in range(T - 1):
        mask = flips[step].bool()
        last_flip[mask] = float(step + 1)

    total_steps = T - 1  # number of flip opportunities

    # Distribution of flip counts
    max_k = min(20, total_steps + 1)
    flip_dist = {
        str(k): float((flip_count == k).float().mean())
        for k in range(max_k)
    }
    flip_dist[f"{max_k}+"] = float((flip_count >= max_k).float().mean())

    return {
        "n_positions": N * L,
        "num_steps": T,
        "n_flips_geq1": float((flip_count >= 1).float().mean()),
        "n_flips_geq5": float((flip_count >= 5).float().mean()),
        "n_flips_geq10": float((flip_count >= 10).float().mean()),
        "n_flips_0": float((flip_count == 0).float().mean()),
        "mean_flip_count": float(flip_count.float().mean()),
        "median_flip_count": float(flip_count.float().median()),
        "mean_last_flip_step": float(last_flip.mean()),
        "mean_last_flip_frac": float(last_flip.mean()) / total_steps,
        "flip_count_distribution": flip_dist,
    }


def main():
    p = argparse.ArgumentParser(description="EXP-24: LangFlow trajectory stability")
    p.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--num_steps", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Samples per forward batch (memory limit)")
    p.add_argument("--out_dir", default="results/exp24_langflow")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] Loading LangFlow from {args.checkpoint}")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    model = model.to(device).eval()

    print(f"[run] n_samples={args.n_samples}, seq_len={args.seq_len}, "
          f"num_steps={args.num_steps}, batch_size={args.batch_size}")

    all_trajectories = []
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size

    for bi in range(n_batches):
        start = bi * args.batch_size
        end = min(start + args.batch_size, args.n_samples)
        n = end - start
        print(f"  batch {bi+1}/{n_batches} (samples {start}–{end})")
        traj = generate_trajectory(model, n, args.seq_len, args.num_steps, device)
        all_trajectories.append(traj)  # [T, n, L]

    # Concatenate along sample dimension: [T, N, L]
    full_traj = torch.cat(all_trajectories, dim=1)
    print(f"[traj] shape={tuple(full_traj.shape)}: [steps={full_traj.shape[0]}, "
          f"samples={full_traj.shape[1]}, len={full_traj.shape[2]}]")

    stats = analyze_stability(full_traj)

    print("\n── EXP-24 LangFlow Trajectory Stability ─────────────────────")
    print(f"  num_steps={args.num_steps}, n_samples={args.n_samples}, seq_len={args.seq_len}")
    print(f"  flip ≥ 1:  {stats['n_flips_geq1']*100:.1f}%  (ELF-baseline: ~95%)")
    print(f"  flip ≥ 5:  {stats['n_flips_geq5']*100:.1f}%  (ELF-baseline: 83.4%, kd_cr: ~75%)")
    print(f"  flip ≥ 10: {stats['n_flips_geq10']*100:.1f}%")
    print(f"  flip = 0:  {stats['n_flips_0']*100:.1f}%")
    print(f"  mean flips: {stats['mean_flip_count']:.2f}")
    print(f"  mean last-flip step: {stats['mean_last_flip_step']:.1f}/{args.num_steps}"
          f"  ({stats['mean_last_flip_frac']*100:.1f}%)")
    print(f"  ELF-baseline last-flip: 27.0/{args.num_steps} (84.4%)")

    out_path = os.path.join(args.out_dir, "traj_stability.json")
    with open(out_path, "w") as f:
        json.dump({"stats": stats, "args": vars(args)}, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
