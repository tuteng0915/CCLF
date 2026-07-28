"""Rigor retrofit: per-UID bootstrap CIs for EXP-PT8's "grammatical pull"
ranking, using the per-pair raw arrays newly saved by probe_minimal_pairs.py
(minimal_pairs_raw_<label>.npz). Resampling unit is the PAIR (each BLiMP pair
is an independent sentence, not a shared sequence like the other PT scripts),
resampled WITHIN each UID (since the by-UID ranking is the headline claim).

Usage:
    conda run -n elf python experiments/phase_transition/bootstrap_pt8.py \
        --npz results/phase_transition/elf/baseline/minimal_pairs_raw_full.npz \
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

    d = np.load(args.npz, allow_pickle=True)
    uids = d["uids"]
    rank_good = d["rank_good_in_bad_traj"]
    rank_bad = d["rank_bad_in_bad_traj"]
    N = len(uids)
    unique_uids = sorted(set(uids.tolist()))
    print(f"[bootstrap_PT8] {args.npz}: N={N} pairs, {len(unique_uids)} UIDs")

    rng = np.random.default_rng(args.seed)
    results = {}
    rg_cis = {}
    for uid in unique_uids:
        mask = uids == uid
        rg, rb = rank_good[mask], rank_bad[mask]
        n = mask.sum()

        def stat(idx):
            rg_s, rb_s = rg[idx], rb[idx]
            # NOTE: the good/bad RATIO is degenerate for heavily-committed
            # checkpoints where rank_bad is exactly 0 for most/all pairs in
            # a UID (observed on kd_cr/kd2 -- see EXP-PT8-spec.md rigor
            # section) -- bootstrap resamples then divide by ~0 and produce
            # meaningless huge ratios. The actual ranking criterion used
            # throughout EXP-PT8-spec.md is rank_good alone (bigger =
            # weaker grammatical pull), which has no such degeneracy, so
            # that's what the CI/disjointness test below uses.
            return rg_s.mean(), rb_s.mean()

        point_rg, point_rb = stat(np.arange(n))
        boot = np.empty((args.n_boot, 2))
        for b in range(args.n_boot):
            idx = rng.integers(0, n, size=n)
            boot[b] = stat(idx)
        rg_lo, rg_hi = np.percentile(boot[:, 0], [2.5, 97.5])
        rb_lo, rb_hi = np.percentile(boot[:, 1], [2.5, 97.5])
        point_ratio = point_rg / max(1e-9, point_rb)
        results[uid] = {
            "n": int(n),
            "rank_good_ci": {"point": float(point_rg), "lo": float(rg_lo), "hi": float(rg_hi)},
            "rank_bad_ci": {"point": float(point_rb), "lo": float(rb_lo), "hi": float(rb_hi)},
            "ratio_point_estimate_only": float(point_ratio),
        }
        rg_cis[uid] = (float(rg_lo), float(rg_hi), float(point_rg))
        print(f"  {uid} (n={n}): rank_good={point_rg:.1f} [{rg_lo:.1f}, {rg_hi:.1f}]  "
              f"rank_bad={point_rb:.2f}  ratio(point-est only)={point_ratio:.1f}")

    # Is the weakest UID's rank_good CI disjoint from every other UID's CI?
    # (directly tests the cross-4-model "existential_there is always weakest"
    # claim with actual statistics instead of eyeballing point estimates.
    # Uses rank_good, not the good/bad ratio -- see note above.)
    weakest_uid = max(rg_cis, key=lambda u: rg_cis[u][2])
    w_lo, w_hi, w_point = rg_cis[weakest_uid]
    disjoint_from_all = all(
        (w_lo > rg_cis[u][1]) for u in rg_cis if u != weakest_uid
    )
    results["weakest_uid"] = weakest_uid
    results["weakest_uid_disjoint_from_all_others"] = disjoint_from_all
    print(f"[bootstrap_PT8] weakest UID (by rank_good) = {weakest_uid} (rank_good={w_point:.1f}); "
          f"CI disjoint from every other UID's CI: {disjoint_from_all}")

    json_path = out_dir / f"bootstrap_pt8_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bootstrap_PT8] Saved {json_path}")


if __name__ == "__main__":
    main()
