"""Rigor retrofit: sequence-level bootstrap CIs for EXP-PT4's headline
"local-window sufficiency/necessity" claims, using the acc_per_seq arrays
intervene_context.py now saves (added in the rigor-audit pass).

Usage:
    conda run -n elf python experiments/phase_transition/bootstrap_pt4.py \\
        --pt4_json results/phase_transition/elf/baseline/context_ablation_full.json \\
        --out_dir results/phase_transition/elf/baseline --label full
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from bootstrap_utils import bootstrap_ci  # noqa: E402

HEADLINE_CONDITIONS = ["full_context", "local_window_r0", "local_window_r1", "local_window_r2",
                        "global_only_r0", "global_only_r1", "within_sequence_shuffle", "cross_sequence_swap"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt4_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pt4_json) as f:
        d = json.load(f)
    t_grid = d["t"]
    t_idx = len(t_grid) - 1  # final t (most informative endpoint)
    N = d["n_samples"]

    if "acc_per_seq" not in d["conditions"]["full_context"]:
        print("[bootstrap_PT4] ERROR: this context_ablation json predates the rigor-audit fix "
              "(no acc_per_seq saved) -- rerun intervene_context.py first.")
        return

    results = {"t": t_grid[t_idx], "n_samples": N, "conditions": {}}
    print(f"[bootstrap_PT4] {args.pt4_json}: N={N} sequences, t={t_grid[t_idx]:.3f}")
    for cond in HEADLINE_CONDITIONS:
        acc_per_seq = np.array(d["conditions"][cond]["acc_per_seq"][t_idx])
        point, lo, hi, std = bootstrap_ci(acc_per_seq, args.n_boot, seed=args.seed)
        results["conditions"][cond] = {"point": point, "lo": lo, "hi": hi, "std": std}
        print(f"  {cond:>24}: {point:.4f} [{lo:.4f}, {hi:.4f}]")

    # Headline comparison: does local_window_r1's CI overlap full_context's?
    # (non-overlap would support "radius 1 already achieves full-context
    # accuracy" only weakly -- really want the DIFFERENCE's CI to include 0.)
    full = np.array(d["conditions"]["full_context"]["acc_per_seq"][t_idx])
    r1 = np.array(d["conditions"]["local_window_r1"]["acc_per_seq"][t_idx])
    diff = r1 - full
    point, lo, hi, std = bootstrap_ci(diff, args.n_boot, seed=args.seed)
    results["diff_local_r1_minus_full_context"] = {"point": point, "lo": lo, "hi": hi, "std": std}
    print(f"  diff(local_r1 - full_context) = {point:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"({'CI includes 0 -> consistent with full sufficiency' if lo <= 0 <= hi else 'CI excludes 0 -> radius 1 is NOT statistically equivalent to full context'})")

    json_path = out_dir / f"bootstrap_pt4_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bootstrap_PT4] Saved {json_path}")


if __name__ == "__main__":
    main()
