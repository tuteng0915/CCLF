#!/usr/bin/env python3
"""
EXP-53: LangFlow T_stable / Never-Commit Rate (EXP-16v2 analogue)

Fixed-ε oracle probe with K=3 consecutive stable correct predictions.
Direct comparison with ELF EXP-16v2:
  ELF baseline:  never-stably-commits = 25.1%
  ELF kd_cr:    never-stably-commits = 0.53%
  ELF kd2:      never-stably-commits = 0.98%

LangFlow prediction: EXP-25 frac_never_committed (T_first) ≈ 1.37%,
T_stable (K=3) likely similar or slightly higher.

Protocol: same as EXP-16v2 (ELF):
  - Fixed ε (same noise draw for all t values, per-sequence)
  - Dense t grid (51 values from 0.03 to 1.0)
  - Oracle probe: z_t = α(t)·x_clean + σ(t)·ε
  - Commit criterion: top1(z_t) == gt_id for K=3 consecutive t values (ONCE first triggered)
  - T_stable = t value at which position first achieves K consecutive correct
  - Never-stable: position never achieves K=3 consecutive correct oracle
  - T_first: t at which position first gets oracle correct (K=1)

Usage (from CCLF root):
    CUDA_VISIBLE_DEVICES=X conda run -n elf python \
        experiments/probe_langflow/tstable_langflow_exp53.py \
        --device cuda:0 \
        --out_dir results/exp53_langflow_tstable
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PROBE_DIR = Path(__file__).parent
_LF_SRC    = _PROBE_DIR.parents[1] / "models" / "LangFlow"
for _p in [str(_LF_SRC)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_PROBE_DIR))

from probe_langflow import (
    load_langflow, encode_with_langflow, load_owt_texts,
    gamma_from_t, softmax_np,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def probe_tstable_single(model, sample, t_grid, gamma_grid, seed, K=3):
    """
    Fixed-ε oracle probe with K-consecutive stable commit detection.

    Returns:
        t_first: [L] float array, t at which each position first gets top1==gt (-1 if never)
        t_stable: [L] float array, t at which first K-consecutive run achieved (-1 if never)
        top1_at_t: [T, L] bool array, oracle top-1 correctness at each t
    """
    gt_ids, clean_emb, attn_mask = sample
    L, d = clean_emb.shape
    T = len(t_grid)

    # Fixed noise draw per position (same ε across all t)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((L, d)).astype(np.float32)
    eps_t = torch.from_numpy(eps).to(device)
    x_clean = torch.from_numpy(clean_emb).to(device, dtype=torch.float32)

    gt_tensor = torch.from_numpy(gt_ids).to(device)
    sc = model.config.self_conditioning

    # [T, L] bool
    correct = np.zeros((T, L), dtype=bool)

    for ti, (t_val, gamma) in enumerate(zip(t_grid, gamma_grid)):
        alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma)).item())
        sigma = math.sqrt(torch.sigmoid(torch.tensor( gamma)).item())
        z_t   = (alpha * x_clean + sigma * eps_t)[None]  # [1, L, d]
        gamma_t = torch.full((1,), float(gamma), device=device, dtype=torch.float32)
        sc_in = torch.zeros_like(z_t) if sc else None

        with torch.no_grad():
            out = model(
                noisy_embeds=z_t,
                timesteps=gamma_t,
                x_self_cond=sc_in,
                return_dict=False,
            )
        logits_np = (out[0] if isinstance(out, (tuple, list)) else out)[0].cpu().float().numpy()
        top1 = np.argmax(logits_np, axis=-1)  # [L]
        correct[ti] = (top1 == gt_ids)

    # Compute T_first (first correct t) and T_stable (first K-consecutive)
    t_first  = np.full(L, -1.0)
    t_stable = np.full(L, -1.0)
    consec   = np.zeros(L, dtype=int)  # consecutive run counter

    for ti in range(T):
        # Update first-correct
        newly_first = correct[ti] & (t_first < 0)
        t_first[newly_first] = t_grid[ti]

        # Update consecutive run
        consec[correct[ti]]  += 1
        consec[~correct[ti]] = 0

        # Check stable: first time consec >= K
        stable_now = (consec >= K) & (t_stable < 0)
        if stable_now.any():
            t_stable[stable_now] = t_grid[ti]

    return t_first, t_stable, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_t_steps", type=int, default=51)
    parser.add_argument("--K", type=int, default=3, help="Consecutive correct for T_stable")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results/exp53_langflow_tstable")
    args = parser.parse_args()

    global device
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # t grid: uniform from 0.03 to 1.0 (MUST reach 1.0 — LangFlow cliff is at t≈0.83-0.93)
    # EXP-25 LangFlow also goes to t=1.0 for comparable results
    t_grid = np.linspace(0.03, 1.0, args.n_t_steps)
    print(f"t grid: {len(t_grid)} values from {t_grid[0]:.3f} to {t_grid[-1]:.3f}")

    print(f"Loading LangFlow from {args.checkpoint}")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    model = model.to(device).eval()
    gamma_grid = gamma_from_t(t_grid, gamma_min, gamma_max)

    print(f"Loading {args.n_samples} OWT samples (seq_len={args.seq_len})")
    texts = load_owt_texts(args.n_samples)
    samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)

    all_t_first  = []
    all_t_stable = []
    all_correct  = []  # [N_samples, T, L]

    for si, sample in enumerate(samples):
        if si % 16 == 0:
            print(f"  sample {si+1}/{len(samples)}")
        t1, ts, corr = probe_tstable_single(
            model, sample, t_grid, gamma_grid, seed=si * 1337, K=args.K)
        all_t_first.append(t1)
        all_t_stable.append(ts)
        all_correct.append(corr)

    # Concatenate across samples: [N*L] positions
    t_first_all  = np.concatenate(all_t_first)
    t_stable_all = np.concatenate(all_t_stable)

    n_total = len(t_first_all)

    # Summary statistics
    never_first  = (t_first_all < 0).mean()
    never_stable = (t_stable_all < 0).mean()

    valid_first  = t_first_all[t_first_all >= 0]
    valid_stable = t_stable_all[t_stable_all >= 0]

    print(f"\n=== EXP-53 Results (LangFlow T_stable, K={args.K}) ===")
    print(f"Total positions: {n_total}")
    print(f"Never T_first:  {never_first:.4f}  ({never_first*100:.2f}%)")
    print(f"Never T_stable: {never_stable:.4f}  ({never_stable*100:.2f}%)  ← compare ELF baseline 25.1%, kd_cr 0.53%")
    if len(valid_first) > 0:
        print(f"Mean T_first (committed): {valid_first.mean():.4f}")
        print(f"Median T_first:           {np.median(valid_first):.4f}")
    if len(valid_stable) > 0:
        print(f"Mean T_stable (committed): {valid_stable.mean():.4f}")
        print(f"Median T_stable:           {np.median(valid_stable):.4f}")

    # G_oracle(t): fraction of positions with oracle correct top-1 at each t
    all_correct_arr = np.concatenate(all_correct, axis=1)  # [T, N*L]
    g_oracle = all_correct_arr.mean(axis=1)  # [T]
    print("\nG_oracle(t) — oracle top-1 accuracy at each t:")
    for ti, (t_val, g) in enumerate(zip(t_grid, g_oracle)):
        if ti % 5 == 0 or t_val >= 0.8:
            print(f"  t={t_val:.3f}: {g:.4f}")

    # Commit fraction: G_stable(t) = fraction with T_stable <= t
    g_stable = np.array([(t_stable_all >= 0) & (t_stable_all <= t) for t in t_grid]).mean(axis=1)
    print("\nG_stable(t) — cumulative fraction with T_stable <= t:")
    for ti, (t_val, g) in enumerate(zip(t_grid, g_stable)):
        if ti % 5 == 0 or t_val >= 0.8:
            print(f"  t={t_val:.3f}: {g:.4f}")

    # ELF comparison
    print("\n=== ELF EXP-16v2 comparison ===")
    print(f"{'':20} {'baseline':>10} {'kd_cr':>8} {'kd2':>8} {'LangFlow':>10}")
    print(f"{'never_stable':20} {'25.1%':>10} {'0.53%':>8} {'0.98%':>8} {never_stable*100:>9.2f}%")

    results = {
        "n_total": int(n_total),
        "K": args.K,
        "never_first": float(never_first),
        "never_stable": float(never_stable),
        "mean_t_first": float(valid_first.mean()) if len(valid_first) else -1,
        "mean_t_stable": float(valid_stable.mean()) if len(valid_stable) else -1,
        "median_t_first": float(np.median(valid_first)) if len(valid_first) else -1,
        "median_t_stable": float(np.median(valid_stable)) if len(valid_stable) else -1,
        "t_grid": t_grid.tolist(),
        "g_oracle_by_t": g_oracle.tolist(),
        "g_stable_by_t": g_stable.tolist(),
        "args": vars(args),
    }

    out_path = out_dir / "tstable_results.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved → {out_path}")

    # Save raw arrays for downstream analysis
    np.save(out_dir / "t_first.npy",  t_first_all)
    np.save(out_dir / "t_stable.npy", t_stable_all)


if __name__ == "__main__":
    main()
