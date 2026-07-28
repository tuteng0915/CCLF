"""Rigor retrofit: sequence-level bootstrap CIs for EXP-PT9's headline claims
(diag_mean, upper_tri_mean, lower_tri_mean), using the per-sequence accuracy
matrix newly saved by probe_cross_time_transfer.py
(cross_time_transfer_raw_<label>.npz) -- same pattern as bootstrap_pt3.py /
bootstrap_pt5.py (requires having rerun probe_cross_time_transfer.py once
after the npz-saving change; no change to probe training itself).

Usage:
    conda run -n elf python experiments/phase_transition/bootstrap_pt9.py \\
        --npz results/phase_transition/elf/baseline/cross_time_transfer_raw_full.npz \\
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
    acc = d["acc_per_seq_matrix"]  # (T, T, n_val)
    T, _, n_val = acc.shape
    print(f"[bootstrap_PT9] {args.npz}: T={T} t-points, n_val={n_val} held-out sequences")

    triu_i, triu_j = np.triu_indices(T, k=1)
    tril_i, tril_j = np.tril_indices(T, k=-1)
    diag_i = np.arange(T)

    rng = np.random.default_rng(args.seed)

    def stat(idx):
        seq_sample = acc[:, :, idx]  # (T,T,len(idx)) -- reduce over val seqs then over the cell selection
        M = seq_sample.mean(axis=2)  # (T,T)
        return {
            "diag_mean": float(np.diag(M).mean()),
            "upper_tri_mean": float(M[triu_i, triu_j].mean()),
            "lower_tri_mean": float(M[tril_i, tril_j].mean()),
        }

    point = stat(np.arange(n_val))
    boot = {k: np.empty(args.n_boot) for k in point}
    for b in range(args.n_boot):
        idx = rng.integers(0, n_val, size=n_val)
        s = stat(idx)
        for k, v in s.items():
            boot[k][b] = v

    results = {}
    for k in point:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        results[f"{k}_ci"] = {"point": point[k], "lo": float(lo), "hi": float(hi), "std": float(boot[k].std())}
        print(f"  {k} = {point[k]:.4f} [{lo:.4f}, {hi:.4f}]")

    # upper vs lower: is the asymmetry (upper > lower) itself robust?
    diff_boot = boot["upper_tri_mean"] - boot["lower_tri_mean"]
    lo, hi = np.percentile(diff_boot, [2.5, 97.5])
    results["upper_minus_lower_ci"] = {
        "point": point["upper_tri_mean"] - point["lower_tri_mean"], "lo": float(lo), "hi": float(hi),
    }
    print(f"  upper_tri_mean - lower_tri_mean = {results['upper_minus_lower_ci']['point']:.4f} "
          f"[{lo:.4f}, {hi:.4f}] ({'asymmetry confirmed (CI excludes 0)' if lo > 0 else 'CI includes 0'})")

    json_path = out_dir / f"bootstrap_pt9_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bootstrap_PT9] Saved {json_path}")


if __name__ == "__main__":
    main()
