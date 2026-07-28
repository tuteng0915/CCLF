"""Rigor retrofit: sequence-level bootstrap CIs for EXP-PT1 and EXP-PT2's
already-saved per-position raw arrays.

This does NOT need new GPU compute -- both estimate_reference_prior.py (PT1)
and analyze_margin_trajectory.py (PT2) already persist per-position arrays
to disk (prior_subtraction_raw_<label>.npz / margin_trajectory_raw_<label>.npz).
This script reloads them, reduces to one summary value PER SEQUENCE (mean
over positions within that sequence), and bootstraps over sequences --
exactly what the suite doc's shared protocol asks for and what was missing
from every PT1-10 result reported so far.

Usage:
    conda run -n elf python experiments/phase_transition/bootstrap_pt1_pt2.py \\
        --pt1_npz results/phase_transition/elf/baseline/prior_subtraction_raw_full.npz \\
        --pt2_npz results/phase_transition/elf/baseline/margin_trajectory_raw_full.npz \\
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

from bootstrap_utils import bootstrap_ci, bootstrap_ratio_ci  # noqa: E402

REF_NAMES = ["gauss", "swap", "shuffle"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt1_npz", default=None)
    p.add_argument("--pt2_npz", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bootstrap_pt1(npz_path, n_boot, seed):
    d = np.load(npz_path)
    mask = d["mask"].astype(bool) if "mask" in d.files else None
    ell_gt, ell_f = d["t0_ell_gt"], d["t0_ell_f"]  # (N,L)
    N, L = ell_gt.shape

    def per_seq_mean(arr):
        if mask is not None:
            out = np.full(N, np.nan)
            for n in range(N):
                m = mask[n].astype(bool)
                out[n] = arr[n][m].mean() if m.any() else np.nan
            return out
        return arr.mean(axis=1)

    m_raw_per_seq = per_seq_mean(ell_gt) - per_seq_mean(ell_f)
    results = {"m_raw_ci": dict(zip(("point", "lo", "hi", "std"),
                                    bootstrap_ci(m_raw_per_seq, n_boot, seed=seed)))}
    rng = np.random.default_rng(seed)
    N_valid = np.isfinite(m_raw_per_seq).sum()
    for name in REF_NAMES:
        e_gt, e_f = d[f"t0_e_gt_{name}"], d[f"t0_e_f_{name}"]
        m_res_per_seq = per_seq_mean(e_gt) - per_seq_mean(e_f)
        point, lo, hi, std = bootstrap_ci(m_res_per_seq, n_boot, seed=seed)
        # "advantage retained" = |m_res|/|m_raw| as a magnitude ratio, matching
        # the convention already used in EXP-PT1-spec.md's prose (abs(), since
        # m_raw is reliably very negative and m_res can even flip sign -- a
        # signed ratio in that case is not interpretable as "% retained").
        ratio_point, ratio_lo, ratio_hi = bootstrap_ratio_ci(
            np.abs(m_res_per_seq), np.abs(m_raw_per_seq), n_boot, seed=seed)
        # Directly interpretable robustness check for the "does m_res flip
        # sign" finding: bootstrap P(mean(m_res) > 0) across resamples.
        valid = np.isfinite(m_res_per_seq)
        vals = m_res_per_seq[valid]
        boot_means = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                                for _ in range(n_boot)])
        prob_positive = float((boot_means > 0).mean())
        results[f"m_res_{name}_ci"] = {"point": point, "lo": lo, "hi": hi, "std": std,
                                        "bootstrap_prob_positive": prob_positive}
        results[f"advantage_retained_{name}_ci"] = {
            "point_pct": ratio_point * 100, "lo_pct": ratio_lo * 100, "hi_pct": ratio_hi * 100}
    return results, int(N_valid)


def bootstrap_pt2(npz_path, n_boot, seed):
    labels = np.load(npz_path.replace("margin_trajectory_raw", "transition_failure_labels"),
                      allow_pickle=True) if Path(
        npz_path.replace("margin_trajectory_raw", "transition_failure_labels")).exists() else None
    d = np.load(npz_path)
    N, L = d["t0_rank_raw"].shape

    results = {}
    if labels is not None:
        tau_e, tau_b, tau_s = labels["tau_e"], labels["tau_b"], labels["tau_s"]
        for name, arr in [("tau_e", tau_e), ("tau_b", tau_b), ("tau_s", tau_s)]:
            finite = np.isfinite(arr)
            per_seq = np.array([arr[n][finite[n]].mean() if finite[n].any() else np.nan
                                 for n in range(N)])
            point, lo, hi, std = bootstrap_ci(per_seq, n_boot, seed=seed)
            results[f"{name}_ci"] = {"point": point, "lo": lo, "hi": hi, "std": std}

        label_arr = labels["labels"]
        for cat in np.unique(label_arr):
            frac_per_seq = (label_arr == cat).mean(axis=1)
            point, lo, hi, std = bootstrap_ci(frac_per_seq, n_boot, seed=seed)
            results[f"frac_{cat}_ci"] = {"point": point, "lo": lo, "hi": hi, "std": std}
    else:
        results["note"] = "transition_failure_labels_<label>.npz not found next to this npz -- skipped tau/failure-fraction CIs"
    return results, N


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"label": args.label, "n_boot": args.n_boot}

    if args.pt1_npz:
        pt1_results, n1 = bootstrap_pt1(args.pt1_npz, args.n_boot, args.seed)
        summary["pt1"] = pt1_results
        summary["pt1_n_sequences"] = n1
        print(f"[bootstrap] PT1 (N={n1} sequences):")
        for name in REF_NAMES:
            a = pt1_results[f"advantage_retained_{name}_ci"]
            m = pt1_results[f"m_res_{name}_ci"]
            print(f"  advantage_retained[{name}] = {a['point_pct']:.2f}% "
                  f"[{a['lo_pct']:.2f}%, {a['hi_pct']:.2f}%]  "
                  f"m_res={m['point']:+.3f} [{m['lo']:+.3f},{m['hi']:+.3f}]  "
                  f"P(m_res>0 across resamples)={m['bootstrap_prob_positive']:.3f}")

    if args.pt2_npz:
        pt2_results, n2 = bootstrap_pt2(args.pt2_npz, args.n_boot, args.seed)
        summary["pt2"] = pt2_results
        summary["pt2_n_sequences"] = n2
        print(f"[bootstrap] PT2 (N={n2} sequences):")
        for k in ["tau_e_ci", "tau_b_ci", "tau_s_ci"]:
            if k in pt2_results:
                v = pt2_results[k]
                print(f"  {k[:-3]} = {v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
        for k, v in pt2_results.items():
            if k.startswith("frac_"):
                print(f"  {k[5:-3]} = {v['point']*100:.2f}% [{v['lo']*100:.2f}%, {v['hi']*100:.2f}%]")

    json_path = out_dir / f"bootstrap_ci_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[bootstrap] Saved {json_path}")


if __name__ == "__main__":
    main()
