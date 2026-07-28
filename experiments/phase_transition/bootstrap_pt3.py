"""Rigor retrofit: sequence-level bootstrap CIs for EXP-PT3's headline
claims, using the already-saved velocity_alignment_raw_<label>.npz
(no new GPU compute needed -- same free-win pattern as bootstrap_pt1_pt2.py).

Usage:
    conda run -n elf python experiments/phase_transition/bootstrap_pt3.py \\
        --npz results/phase_transition/elf/baseline/velocity_alignment_raw_full.npz \\
        --out_dir results/phase_transition/elf/baseline --label full
"""

import argparse
import json
import re
import sys
from pathlib import Path

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
    t_grid = d["t0_t"]  # not actually used, just confirms t0 exists
    N, L = d["gt_ids"].shape
    print(f"[bootstrap_PT3] {args.npz}: N={N} sequences, L={L} positions")

    # frac_valid_direction: per-sequence fraction of positions with a usable
    # u_yf, using t0's a_clean as a stand-in validity check (a_clean is 0
    # only at genuinely invalid positions -- reload valid_mask directly if present)
    valid_key = "valid_mask" if "valid_mask" in d.files else None

    results = {}
    # a_clean at t0 (earliest t): headline "vector field points toward truth
    # before native decode is meaningful" claim.
    a_clean_t0 = d["t0_a_clean"]  # (N,L)
    per_seq_a_clean = a_clean_t0.mean(axis=1)
    point, lo, hi, std = bootstrap_ci(per_seq_a_clean, args.n_boot, seed=args.seed)
    results["a_clean_t0_ci"] = {"point": point, "lo": lo, "hi": hi, "std": std}
    print(f"  a_clean(t_min) = {point:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # corr(C_i(t), -rank_raw(t)) real vs controls, at the LAST available t
    # (find the highest t{k}_ index present).
    idxs = sorted({int(m.group(1)) for k in d.files if (m := re.match(r"^t(\d+)_", k))})
    last_i = idxs[-1]
    rank_raw = d[f"t{last_i}_rank_raw"]  # (N,L)
    for variant in ["a_tok", "a_tok_random", "a_tok_orth", "a_tok_freqmatch"]:
        C = d[f"t{last_i}_C_{variant}"]  # (N,L)
        # per-sequence correlation is noisy with few positions; instead
        # bootstrap the OVERALL correlation by resampling SEQUENCES (each
        # resample re-pools all positions from the resampled sequences).
        rng = np.random.default_rng(args.seed)
        boot_corrs = np.empty(args.n_boot)
        for b in range(args.n_boot):
            idx = rng.integers(0, N, size=N)
            c_flat = C[idx].reshape(-1).astype(np.float64)
            r_flat = -rank_raw[idx].reshape(-1).astype(np.float64)
            if c_flat.std() < 1e-8 or r_flat.std() < 1e-8:
                boot_corrs[b] = np.nan
                continue
            boot_corrs[b] = np.corrcoef(c_flat, r_flat)[0, 1]
        boot_corrs = boot_corrs[np.isfinite(boot_corrs)]
        point_corr = np.corrcoef(C.reshape(-1).astype(np.float64),
                                  (-rank_raw).reshape(-1).astype(np.float64))[0, 1]
        lo_c, hi_c = np.percentile(boot_corrs, [2.5, 97.5])
        results[f"corr_{variant}_ci"] = {"point": float(point_corr), "lo": float(lo_c), "hi": float(hi_c)}
        print(f"  corr(C_{variant}, -rank) @t_last = {point_corr:.4f} [{lo_c:.4f}, {hi_c:.4f}]")

    # Does the real (a_tok) CI overlap the frequency-matched control's CI?
    # This is the headline "decision rule not satisfied" claim from the
    # main PT3 spec -- check it with actual CIs instead of eyeballing point estimates.
    real = results["corr_a_tok_ci"]
    freq = results["corr_a_tok_freqmatch_ci"]
    overlap = not (real["lo"] > freq["hi"] or freq["lo"] > real["hi"])
    results["real_vs_freqmatch_ci_overlap"] = overlap
    print(f"  CI(real) vs CI(freqmatch) overlap: {overlap} "
          f"({'controls NOT distinguishable from real signal -- decision rule genuinely not met' if overlap else 'real signal IS distinguishable from this control'})")

    json_path = out_dir / f"bootstrap_pt3_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bootstrap_PT3] Saved {json_path}")


if __name__ == "__main__":
    main()
