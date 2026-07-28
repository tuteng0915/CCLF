"""Rigor retrofit: sequence-level bootstrap CI for EXP-PT10's headline
val_accuracy vs majority-class-baseline comparison, using the per-position
val raw arrays newly saved by analyze_failure_predictors.py
(failure_predictors_raw_<label>.npz). No refit needed -- resamples which
held-out VAL sequences contribute to the accuracy average (same "free win"
pattern as bootstrap_pt4.py/bootstrap_pt9.py).

Usage:
    conda run -n elf python experiments/phase_transition/bootstrap_pt10.py \
        --npz results/phase_transition/elf/baseline/failure_predictors_raw_full.npz \
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
    val_seq_idx = d["val_seq_idx"]
    val_correct = d["val_correct"]
    val_is_majority = d["val_is_majority"]
    val_seqs = np.unique(val_seq_idx)
    n_val = len(val_seqs)
    print(f"[bootstrap_PT10] {args.npz}: {n_val} held-out sequences, {len(val_correct)} positions")

    # Per-sequence mean accuracy / majority-baseline-accuracy (unequal
    # position counts per sequence are fine -- each sequence contributes
    # its own within-sequence mean, matching the resampling unit used by
    # every other PT bootstrap script).
    per_seq_acc = np.array([val_correct[val_seq_idx == s].mean() for s in val_seqs])
    per_seq_majority = np.array([val_is_majority[val_seq_idx == s].mean() for s in val_seqs])

    rng = np.random.default_rng(args.seed)
    boot_acc = np.empty(args.n_boot)
    boot_majority = np.empty(args.n_boot)
    boot_diff = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng.integers(0, n_val, size=n_val)
        boot_acc[b] = per_seq_acc[idx].mean()
        boot_majority[b] = per_seq_majority[idx].mean()
        boot_diff[b] = boot_acc[b] - boot_majority[b]

    def ci(point, boot):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"point": float(point), "lo": float(lo), "hi": float(hi)}

    results = {
        "val_accuracy_ci": ci(per_seq_acc.mean(), boot_acc),
        "majority_baseline_ci": ci(per_seq_majority.mean(), boot_majority),
        "improvement_over_majority_ci": ci(per_seq_acc.mean() - per_seq_majority.mean(), boot_diff),
        "prob_improvement_positive": float((boot_diff > 0).mean()),
    }
    print(f"  val_accuracy = {results['val_accuracy_ci']['point']:.4f} "
          f"[{results['val_accuracy_ci']['lo']:.4f}, {results['val_accuracy_ci']['hi']:.4f}]")
    print(f"  majority_baseline = {results['majority_baseline_ci']['point']:.4f} "
          f"[{results['majority_baseline_ci']['lo']:.4f}, {results['majority_baseline_ci']['hi']:.4f}]")
    print(f"  improvement_over_majority = {results['improvement_over_majority_ci']['point']:+.4f} "
          f"[{results['improvement_over_majority_ci']['lo']:+.4f}, {results['improvement_over_majority_ci']['hi']:+.4f}] "
          f"P(>0)={results['prob_improvement_positive']:.3f}")

    json_path = out_dir / f"bootstrap_pt10_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bootstrap_PT10] Saved {json_path}")


if __name__ == "__main__":
    main()
