"""Rigor retrofit: sequence-level bootstrap CIs for EXP-PT5's headline claims,
using the per-position arrays newly saved by intervene_decoder_bias.py
(decoder_bias_raw_<label>.npz) -- same free-win pattern as bootstrap_pt1_pt2.py
/ bootstrap_pt3.py (no new GPU compute needed, just requires having rerun
intervene_decoder_bias.py once after the npz-saving change).

Usage:
    conda run -n elf python experiments/phase_transition/bootstrap_pt5.py \\
        --npz results/phase_transition/elf/baseline/decoder_bias_raw_full.npz \\
        --out_dir results/phase_transition/elf/baseline --label full
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from bootstrap_utils import bootstrap_ci  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.npz)
    tau_e = d["tau_e"]  # (N,L)
    tau_b0 = d["tau_b_lambda0"]  # (N,L)
    tau_b1 = d["tau_b_lambda1"]  # (N,L)
    correct_beta = d["correct_beta"]  # (n_betas, T, N, L)
    wrong_raw = d["wrong_raw"]  # (T, N, L)
    betas = d["betas"]
    N, L = tau_e.shape
    print(f"[bootstrap_PT5] {args.npz}: N={N} sequences, L={L} positions")

    results = {}

    # 1. mean_shift_vs_lambda0 at lambda=1 (headline "does prior-debiasing move
    # tau_b earlier or later" claim), per-sequence mean over positions where
    # both tau_b0 and tau_b1 are finite.
    both_finite = np.isfinite(tau_b0) & np.isfinite(tau_b1)
    shift = np.where(both_finite, tau_b0 - tau_b1, np.nan)  # positive = earlier
    per_seq_shift = np.nanmean(shift, axis=1)  # (N,) -- nan if a seq has 0 finite positions
    valid_seqs = np.isfinite(per_seq_shift)
    point, lo, hi, std = bootstrap_ci(per_seq_shift[valid_seqs], args.n_boot, seed=args.seed)
    results["tau_b_shift_lambda1_ci"] = {"point": point, "lo": lo, "hi": hi, "std": std,
                                          "n_valid_seqs": int(valid_seqs.sum())}
    print(f"  mean_shift(lambda=1 vs raw) = {point:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"({'earlier' if point > 0 else 'LATER'})")

    # 2. frac_positions_boundary_explained, per sequence.
    finite_e = np.isfinite(tau_e)
    boundary_explained = (tau_b0 > tau_e) & finite_e & (tau_b1 <= tau_e + 1e-9)
    per_seq_frac = boundary_explained.mean(axis=1)  # (N,)
    point, lo, hi, std = bootstrap_ci(per_seq_frac, args.n_boot, seed=args.seed)
    results["frac_boundary_explained_ci"] = {"point": point, "lo": lo, "hi": hi, "std": std}
    print(f"  frac_boundary_explained = {point:.4f} [{lo:.4f}, {hi:.4f}]")

    # 3. beta flip rate, per sequence, for each beta: sum(newly_correct) /
    # sum(wrong_raw) pooled over (t, position) within that sequence.
    for bi, beta in enumerate(betas.tolist()):
        newly_correct = correct_beta[bi] & wrong_raw  # (T,N,L)
        num_per_seq = newly_correct.sum(axis=(0, 2)).astype(np.float64)  # (N,)
        den_per_seq = wrong_raw.sum(axis=(0, 2)).astype(np.float64)  # (N,)
        rng = np.random.default_rng(args.seed)
        boot_ratios = np.empty(args.n_boot)
        for b in range(args.n_boot):
            idx = rng.integers(0, N, size=N)
            num_s, den_s = num_per_seq[idx].sum(), den_per_seq[idx].sum()
            boot_ratios[b] = num_s / den_s if den_s > 0 else np.nan
        boot_ratios = boot_ratios[np.isfinite(boot_ratios)]
        point_ratio = num_per_seq.sum() / max(1.0, den_per_seq.sum())
        lo_r, hi_r = np.percentile(boot_ratios, [2.5, 97.5])
        results[f"beta_{beta}_flip_rate_ci"] = {"point": float(point_ratio), "lo": float(lo_r), "hi": float(hi_r)}
        print(f"  beta={beta}: flip_rate = {point_ratio:.4f} [{lo_r:.4f}, {hi_r:.4f}]")

    json_path = out_dir / f"bootstrap_pt5_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bootstrap_PT5] Saved {json_path}")


if __name__ == "__main__":
    main()
