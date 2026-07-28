"""EXP-PT2 (part 2): per-position transition-time statistics and the 6-way
failure taxonomy from docs/phase_transition_experiment_suite.md section 4.

Consumes the npz written by analyze_margin_trajectory.py and computes, per
position:
  tau_e  -- evidence-emergence time: first t_k s.t. m_res(t_j) > delta_e for
            K_e consecutive grid points j=k..k+K_e-1 (delta_e=0, K_e=args.k_e,
            matching the suite doc's recommended defaults, section 1).
  tau_b  -- boundary-crossing time: first t at which raw argmax == y.
  tau_s  -- stable-final time: first t from which raw argmax stays correct
            through the end of the grid.
  pre/post-crossing slope of m_res(t) around tau_b (simple linear fit in a
  small window on each side).
  number of zero crossings of m_res(t).
  distance between the first m_res(t) zero-crossing and tau_b.

Then classifies each position into exactly one of the doc's 6 failure/success
categories (priority-ordered -- see docstring on `classify_position` for the
resolution order when a position could match more than one).

Usage:
    conda run -n elf python experiments/phase_transition/classify_transition_failures.py \\
        --npz results/phase_transition/elf/baseline/margin_trajectory_raw_full.npz \\
        --out_dir results/phase_transition/elf/baseline --label full
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--delta_e", type=float, default=0.0)
    p.add_argument("--k_e", type=int, default=3)
    p.add_argument("--slope_window", type=int, default=2, help="grid points on each side of tau_b for slope fit")
    return p.parse_args()


def load_records(npz_path):
    data = np.load(npz_path)
    t_grid = data["t_grid"]
    n_t = len(t_grid)
    f1 = data["f1"]
    f2 = data["f2"]
    gt_ids = data["gt_ids"]
    records = []
    for i in range(n_t):
        rec = {}
        prefix = f"t{i}_"
        for key in data.files:
            if key.startswith(prefix):
                rec[key[len(prefix):]] = data[key]
        records.append(rec)
    return t_grid, records, f1, f2, gt_ids


def first_stable_run_time(bool_over_t, t_grid, k):
    """bool_over_t: (T,N,L). Returns (N,L) t of the first index k0 s.t.
    bool_over_t[k0:k0+k] are all True, or +inf if no such run exists."""
    T = bool_over_t.shape[0]
    if T < k:
        return np.full(bool_over_t.shape[1:], np.inf)
    result = np.full(bool_over_t.shape[1:], np.inf)
    for start in range(T - k + 1):
        window_all = bool_over_t[start:start + k].all(axis=0)
        newly = window_all & (result == np.inf)
        result = np.where(newly, t_grid[start], result)
    return result


def first_true_time(bool_over_t, t_grid):
    T = bool_over_t.shape[0]
    result = np.full(bool_over_t.shape[1:], np.inf)
    for k in range(T - 1, -1, -1):
        result = np.where(bool_over_t[k], t_grid[k], result)
    return result


def stable_from_time(bool_over_t, t_grid):
    """First t_k such that bool_over_t[j] is True for ALL j>=k (suffix-stable)."""
    T = bool_over_t.shape[0]
    suffix_all = np.ones(bool_over_t.shape[1:], dtype=bool)
    result = np.full(bool_over_t.shape[1:], np.inf)
    for k in range(T - 1, -1, -1):
        suffix_all = suffix_all & bool_over_t[k]
        result = np.where(suffix_all, t_grid[k], result)
        # once suffix_all becomes False at some k it stays a candidate=False
        # for smaller k too unless bool[k] is True AND all later already True
    return result


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or npz_path.stem

    t_grid, records, f1, f2, gt_ids = load_records(npz_path)
    T = len(t_grid)
    N, L = records[0]["rank_raw"].shape
    print(f"[classify_PT2] {npz_path}: T={T}, N={N}, L={L}")

    m_res = np.stack([r["e_gt"] - r["e_f1"] for r in records])      # (T,N,L)
    correct_raw = np.stack([r["rank_raw"] == 0 for r in records])   # (T,N,L)
    residual_top1 = np.stack([r["residual_top1"] for r in records])  # (T,N,L)

    # tau_e: first K_e-consecutive-hit time where m_res > delta_e
    emerged = m_res > args.delta_e
    tau_e = first_stable_run_time(emerged, t_grid, args.k_e)
    # tau_b: first time raw argmax correct
    tau_b = first_true_time(correct_raw, t_grid)
    # tau_s: stable-final time (suffix always correct)
    tau_s = stable_from_time(correct_raw, t_grid)

    # zero crossings of m_res(t), per position
    sign = np.sign(m_res)
    sign[sign == 0] = 1
    flips = (np.diff(sign, axis=0) != 0)
    n_zero_crossings = flips.sum(axis=0)  # (N,L)

    # first zero-crossing time (m_res goes from <=0 to >0)
    pos = m_res > 0
    first_pos_idx = np.full((N, L), T, dtype=np.int64)
    for k in range(T - 1, -1, -1):
        first_pos_idx = np.where(pos[k], k, first_pos_idx)
    margin_zero_t = np.where(first_pos_idx < T, t_grid[np.clip(first_pos_idx, 0, T - 1)], np.inf)
    with np.errstate(invalid="ignore"):
        margin_zero_vs_switch_dist = np.where(
            np.isfinite(margin_zero_t) & np.isfinite(tau_b), tau_b - margin_zero_t, np.nan)

    # pre/post-crossing slope of m_res(t) around tau_b, per position
    w = args.slope_window
    pre_slope = np.full((N, L), np.nan)
    post_slope = np.full((N, L), np.nan)
    tau_b_idx = np.full((N, L), -1, dtype=np.int64)
    for k, t in enumerate(t_grid):
        tau_b_idx = np.where(tau_b == t, k, tau_b_idx)
    for n in range(N):
        for l in range(L):
            k0 = tau_b_idx[n, l]
            if k0 < 0:
                continue
            a0, a1 = max(0, k0 - w), k0
            if a1 - a0 >= 2:
                x, y = t_grid[a0:a1 + 1], m_res[a0:a1 + 1, n, l]
                pre_slope[n, l] = np.polyfit(x, y, 1)[0]
            b0, b1 = k0, min(T - 1, k0 + w)
            if b1 - b0 >= 2:
                x, y = t_grid[b0:b1 + 1], m_res[b0:b1 + 1, n, l]
                post_slope[n, l] = np.polyfit(x, y, 1)[0]

    def classify_position(n, l):
        """Priority-ordered 6-way classification (doc section 4 failure
        taxonomy). Categories can overlap in principle (e.g. a position with
        multiple flips that also only ends correct at the last point); the
        priority order below is a deliberate, documented tie-break -- see
        EXP-PT2-spec.md."""
        m = m_res[:, n, l]
        cr = correct_raw[:, n, l]
        if not np.any(m > args.delta_e):
            return "no_emergence"
        if not cr.any():
            # emerged but native top-1 never becomes correct: check if a
            # specific *other* token dominates the residual race throughout
            final_res_top1 = residual_top1[-1, n, l]
            if final_res_top1 != gt_ids[n, l] and final_res_top1 != f1[n, l]:
                return "wrong_mode_accumulation"
            return "stalled_ambiguity"
        correct_flips = np.sum(np.diff(cr.astype(int)) != 0)
        if correct_flips > 2:
            return "multiple_revision"
        if cr[-1] and not cr[0] and np.argmax(cr) == T - 1:
            return "endpoint_only_correction"
        if cr.any() and not cr[-1]:
            return "premature_crossing"
        return "successful_monotonic"

    labels = np.empty((N, L), dtype=object)
    for n in range(N):
        for l in range(L):
            labels[n, l] = classify_position(n, l)

    unique, counts = np.unique(labels, return_counts=True)
    frac = {u: float(c) / (N * L) for u, c in zip(unique, counts)}

    with np.errstate(invalid="ignore"):
        delta_readout_mean = float(np.nanmean(np.where(
            np.isfinite(tau_e) & np.isfinite(tau_b), tau_b - tau_e, np.nan)))
        delta_stability_mean = float(np.nanmean(np.where(
            np.isfinite(tau_s) & np.isfinite(tau_b), tau_s - tau_b, np.nan)))

    result = {
        "npz": str(npz_path), "label": label, "n_samples": N, "seq_len": L,
        "t_grid": t_grid.tolist(),
        "failure_taxonomy_fractions": frac,
        "tau_e_mean_finite": float(np.nanmean(np.where(np.isfinite(tau_e), tau_e, np.nan))),
        "tau_b_mean_finite": float(np.nanmean(np.where(np.isfinite(tau_b), tau_b, np.nan))),
        "tau_s_mean_finite": float(np.nanmean(np.where(np.isfinite(tau_s), tau_s, np.nan))),
        "frac_tau_e_finite": float(np.mean(np.isfinite(tau_e))),
        "frac_tau_b_finite": float(np.mean(np.isfinite(tau_b))),
        "frac_tau_s_finite": float(np.mean(np.isfinite(tau_s))),
        "delta_readout_mean": delta_readout_mean,
        "delta_stability_mean": delta_stability_mean,
        "mean_zero_crossings": float(n_zero_crossings.mean()),
        "mean_margin_zero_vs_switch_dist": float(np.nanmean(margin_zero_vs_switch_dist)),
        "mean_pre_crossing_slope": float(np.nanmean(pre_slope)),
        "mean_post_crossing_slope": float(np.nanmean(post_slope)),
    }

    out_json = out_dir / f"transition_failure_analysis_{label}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    # Per-position arrays (labels, tau_e/b/s) -- needed by EXP-PT10's
    # failure-predictor regression, not otherwise persisted anywhere.
    out_npz = out_dir / f"transition_failure_labels_{label}.npz"
    np.savez_compressed(
        out_npz,
        labels=labels.astype(str), tau_e=tau_e, tau_b=tau_b, tau_s=tau_s,
        n_zero_crossings=n_zero_crossings, gt_ids=gt_ids, f1=f1,
    )
    print(f"[classify_PT2] Saved per-position labels to {out_npz}")

    print("\n[classify_PT2] Failure taxonomy fractions:")
    for u in sorted(frac, key=lambda k: -frac[k]):
        print(f"  {u:>26}: {frac[u]*100:6.2f}%")
    print(f"\n[classify_PT2] tau_e/tau_b/tau_s (mean over finite): "
          f"{result['tau_e_mean_finite']:.3f} / {result['tau_b_mean_finite']:.3f} / "
          f"{result['tau_s_mean_finite']:.3f}")
    print(f"[classify_PT2] Delta_readout (tau_b-tau_e) mean: {result['delta_readout_mean']:.3f}  "
          f"Delta_stability (tau_s-tau_b) mean: {result['delta_stability_mean']:.3f}")
    print(f"[classify_PT2] mean pre/post-crossing slope: "
          f"{result['mean_pre_crossing_slope']:.3f} / {result['mean_post_crossing_slope']:.3f}")
    print(f"[classify_PT2] Saved {out_json}")


if __name__ == "__main__":
    main()
