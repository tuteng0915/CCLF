"""EXP-PT1 (part 2): consume the compact per-t/per-position arrays produced by
estimate_reference_prior.py and compute the decision-rule metrics from
docs/phase_transition_experiment_suite.md section 3 / docs/specs/EXP-PT1-spec.md.

For each reference X in {gauss, swap, shuffle}:
  - residual vs raw true-token rank curves (mean over positions, per t)
  - raw margin m_raw(t) = ell(y) - ell(f)  and residual margin
    m_res_X(t) = e_X(y) - e_X(f)
  - fraction of positions where residual rank reaches 0 strictly before raw
    rank reaches 0 (residual-leads-raw)
  - fraction of positions where raw top-1 == null-mode token AND residual
    top-1 (for reference X) is sample-specific (!= null-mode token)
  - KL(p || q_X) curve

Then prints the "prior masking" decision-rule checklist from section 3.4 of
the suite doc (as numbers, not an automatic verdict -- see spec section 2's
methodology note on why we don't auto-classify).

Usage:
    conda run -n elf python experiments/phase_transition/analyze_prior_subtraction.py \\
        --npz results/phase_transition/elf/baseline/prior_subtraction_raw_pilot.npz \\
        --out_dir results/phase_transition/elf/baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default=None, help="defaults to npz stem")
    return p.parse_args()


REF_NAMES = ["gauss", "swap", "shuffle"]


def load_records(npz_path):
    data = np.load(npz_path)
    t_grid = data["t_grid"]
    n_t = len(t_grid)
    f_i = data["f_i"]
    null_mode_token = int(data["null_mode_token"])
    mask = data["mask"].astype(bool) if "mask" in data.files else None
    records = []
    for i in range(n_t):
        rec = {"t": float(t_grid[i])}
        for key in data.files:
            prefix = f"t{i}_"
            if key.startswith(prefix):
                val = data[key]
                rec[key[len(prefix):]] = val.item() if val.ndim == 0 else val
        records.append(rec)
    return t_grid, records, f_i, null_mode_token, mask


def first_hit_time(rank_over_t, t_grid):
    """rank_over_t: (T, N, L) array of ranks. Returns (N,L) array of the
    first t at which rank==0, or +inf if never."""
    T = rank_over_t.shape[0]
    hit = (rank_over_t == 0)
    first_idx = np.full(hit.shape[1:], T, dtype=np.int64)
    for ti in range(T - 1, -1, -1):
        first_idx = np.where(hit[ti], ti, first_idx)
    tau = np.where(first_idx < T, t_grid[np.clip(first_idx, 0, T - 1)], np.inf)
    return tau


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or npz_path.stem

    t_grid, records, f_i, null_mode_token, mask = load_records(npz_path)
    T = len(t_grid)
    N, L = records[0]["rank_raw"].shape
    if mask is None:
        print("[analyze_PT1] WARNING: no padding mask in this npz (older run, before the "
              "padding-exclusion fix) -- frac_null_mode_but_residual_specific may be "
              "contaminated by pad-token positions, see EXP-PT1-spec.md.")
        mask = np.ones((N, L), dtype=bool)

    print(f"[analyze_PT1] {npz_path}: T={T} t-points, N={N} sequences, L={L} positions, "
          f"null_mode_token={null_mode_token}, non_pad_frac={mask.mean():.4f}")

    rank_raw_stack = np.stack([r["rank_raw"] for r in records])   # (T,N,L)
    tau_raw = first_hit_time(rank_raw_stack, t_grid)               # (N,L)

    curves = {"t": t_grid.tolist(), "rank_raw_mean": rank_raw_stack.reshape(T, -1).mean(-1).tolist()}
    fractions = {}
    kl_curves = {}
    margin_curves = {}

    for name in REF_NAMES:
        rank_res_stack = np.stack([r[f"rank_res_{name}"] for r in records])  # (T,N,L)
        tau_res = first_hit_time(rank_res_stack, t_grid)

        curves[f"rank_res_{name}_mean"] = rank_res_stack.reshape(T, -1).mean(-1).tolist()
        kl_curves[name] = [r[f"KL_{name}"] for r in records]

        # m_raw(t) = ell_gt - ell_f ; m_res_X(t) = e_gt_X - e_f_X
        m_raw = [float((r["ell_gt"] - r["ell_f"]).mean()) for r in records]
        m_res = [float((r[f"e_gt_{name}"] - r[f"e_f_{name}"]).mean()) for r in records]
        margin_curves[name] = {"m_raw": m_raw, "m_res": m_res}

        # Fraction where residual rank reaches 0 strictly before raw rank does
        both_finite = np.isfinite(tau_res) & np.isfinite(tau_raw)
        residual_before_raw = both_finite & (tau_res < tau_raw)
        residual_only = np.isfinite(tau_res) & ~np.isfinite(tau_raw)
        frac_residual_before_raw = float((residual_before_raw | residual_only).mean())

        # Fraction where raw top-1 (at any t) == null-mode token but residual
        # top-1 (same t, same reference) is sample-specific. Padding
        # positions excluded (doc section 2's "Exclude padding"; see
        # EXP-PT1-spec.md for the pad-token-contamination bug this fixes).
        null_and_specific_hits = 0
        null_total = 0
        for r in records:
            is_null = (r["raw_top1"] == null_mode_token) & mask
            null_total += is_null.sum()
            is_specific = (r[f"residual_top1_{name}"] != null_mode_token)
            null_and_specific_hits += (is_null & is_specific).sum()
        frac_null_mode_but_residual_specific = (
            float(null_and_specific_hits / null_total) if null_total > 0 else float("nan")
        )

        fractions[name] = {
            "frac_residual_before_raw": frac_residual_before_raw,
            "frac_null_mode_but_residual_specific": frac_null_mode_but_residual_specific,
            "never_raw_correct_frac": float(np.mean(~np.isfinite(tau_raw))),
            "never_residual_correct_frac": float(np.mean(~np.isfinite(tau_res))),
        }

    result = {
        "npz": str(npz_path), "label": label, "t_grid": t_grid.tolist(),
        "curves": curves, "kl_curves": kl_curves, "margin_curves": margin_curves,
        "fractions": fractions,
    }

    out_json = out_dir / f"prior_subtraction_metrics_{label}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'ref':>8} | {'frac_res<raw':>13} | {'frac_null->specific':>19} | "
          f"{'never_raw':>10} | {'never_res':>10}")
    print("-" * 75)
    for name in REF_NAMES:
        fr = fractions[name]
        print(f"{name:>8} | {fr['frac_residual_before_raw']:13.4f} | "
              f"{fr['frac_null_mode_but_residual_specific']:19.4f} | "
              f"{fr['never_raw_correct_frac']:10.4f} | {fr['never_residual_correct_frac']:10.4f}")

    print("\n[analyze_PT1] Decision-rule checklist (suite doc section 3.4 -- numbers only, "
          "no auto-verdict; see EXP-PT1-spec.md section 2 for why):")
    print("  1. residual true-token rank improves significantly before raw top-1 crossing")
    print("     -> frac_residual_before_raw per reference (see table above)")
    print("  2. early default token loses most advantage after reference subtraction")
    print("     -> compare m_raw vs m_res_X at small t (see margin_curves in JSON)")
    print("  3. effect present for both ELF and LangFlow (needs a second run on the other backend)")

    for name in REF_NAMES:
        m_raw0 = margin_curves[name]["m_raw"][0]
        m_res0 = margin_curves[name]["m_res"][0]
        print(f"  [{name}] t={t_grid[0]:.3f}: m_raw={m_raw0:+.3f}  m_res={m_res0:+.3f}  "
              f"(advantage retained: {abs(m_res0/m_raw0)*100 if m_raw0 != 0 else float('nan'):.1f}%)")

    print(f"\n[analyze_PT1] Saved {out_json}")


if __name__ == "__main__":
    main()
