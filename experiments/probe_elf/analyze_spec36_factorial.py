"""
analyze_spec36_factorial.py — Compute 2×2 interaction for EXP-36 full factorial

Reads outputs/spec36_factorial_{baseline,kd_cr,kd2}/ and computes:
  - PPL for each arm: none, SC-only, DF-only, DF+SC
  - Interaction I = (DF+SC) - DF - SC + none  [for each DF variant]
  - Main effects: DF = mean(DF arms) - mean(none arms), SC = SC-only - none
  - Degeneration rate per arm

Usage (from ELF-torch root):
  python experiments/probe_elf/analyze_spec36_factorial.py
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

OUTPUTS = Path("outputs")
CHECKPOINTS = ["baseline", "kd_cr", "kd2"]

# Pattern matching for output subdirectory names (spec36_factorial config)
# Each condition generates a subdir based on its params
ARMS = {
    "none": [
        "ode-steps32-cfg1-ts_uniform-uncond",
    ],
    "sc_only": [
        "ode-steps32-cfg1-ts_uniform-decsc_decode-tmin0.5-uncond",
    ],
    "freeze_0.5_only": [
        "ode-steps32-cfg1-ts_uniform-df_freeze0.5-dftmin0.7-uncond",
    ],
    "freeze_1.0_only": [
        "ode-steps32-cfg1-ts_uniform-df_freeze1.0-dftmin0.7-uncond",
    ],
    "soft_0.3_only": [
        "ode-steps32-cfg1-ts_uniform-df_soft0.3-dftmin0.7-uncond",
    ],
    "freeze_0.5_sc": [
        "ode-steps32-cfg1-ts_uniform-decsc_decode-tmin0.5-df_freeze0.5-dftmin0.7-uncond",
    ],
    "freeze_1.0_sc": [
        "ode-steps32-cfg1-ts_uniform-decsc_decode-tmin0.5-df_freeze1.0-dftmin0.7-uncond",
    ],
    "soft_0.3_sc": [
        "ode-steps32-cfg1-ts_uniform-decsc_decode-tmin0.5-df_soft0.3-dftmin0.7-uncond",
    ],
}


def load_ppl(run_dir: Path, patterns: list):
    for pat in patterns:
        path = run_dir / pat / "metrics.jsonl"
        if path.exists():
            with open(path) as f:
                return json.loads(f.read()).get("ppl")
    # Try glob
    for pat in patterns:
        matches = list(run_dir.glob(f"*{pat.split('-')[3]}*{pat.split('-')[-1]}*/metrics.jsonl"))
        if matches:
            with open(matches[0]) as f:
                return json.loads(f.read()).get("ppl")
    # List available subdirs for debugging
    if run_dir.exists():
        subdirs = [d.name for d in run_dir.iterdir() if d.is_dir()]
        return None
    return None


def load_all_arms(run_dir: Path) -> dict:
    """Try to find all arms, using glob if exact name doesn't match."""
    if not run_dir.exists():
        return {}

    # First try exact matches
    found = {}
    subdirs = {d.name: d for d in run_dir.iterdir() if d.is_dir()}

    for arm_name, patterns in ARMS.items():
        for pat in patterns:
            if pat in subdirs:
                metrics_path = subdirs[pat] / "metrics.jsonl"
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        d = json.loads(f.read())
                    found[arm_name] = d.get("ppl")
                    break

    # If not found, try partial match
    for arm_name, patterns in ARMS.items():
        if arm_name in found:
            continue
        for pat in patterns:
            # Try to match subdir containing key parts of pattern
            for subdir_name, subdir_path in subdirs.items():
                if "freeze" in pat and "freeze" in subdir_name and pat.split("freeze")[1][:3] in subdir_name:
                    metrics_path = subdir_path / "metrics.jsonl"
                    if metrics_path.exists():
                        with open(metrics_path) as f:
                            d = json.loads(f.read())
                        found[arm_name] = d.get("ppl")
                        break
                elif "soft" in pat and "soft" in subdir_name:
                    metrics_path = subdir_path / "metrics.jsonl"
                    if metrics_path.exists():
                        with open(metrics_path) as f:
                            d = json.loads(f.read())
                        found[arm_name] = d.get("ppl")
                        break

    return found


def compute_interaction(ppl_none, ppl_df, ppl_sc, ppl_dfsc):
    """
    2×2 interaction: I = (DF+SC) - DF - SC + none
    Positive I: DF and SC are complementary (synergistic)
    Negative I: DF and SC are redundant (competitive)
    Zero I: additive (no interaction)
    """
    if any(x is None for x in [ppl_none, ppl_df, ppl_sc, ppl_dfsc]):
        return None
    return ppl_dfsc - ppl_df - ppl_sc + ppl_none


def main():
    print("=" * 80)
    print("EXP-36 FULL FACTORIAL: 2×2 Interaction Analysis (DF × dec_sc)")
    print("=" * 80)

    all_results = {}
    for ckpt in CHECKPOINTS:
        run_dir = OUTPUTS / f"spec36_factorial_{ckpt}"
        arms = load_all_arms(run_dir)
        all_results[ckpt] = arms

        if not arms:
            print(f"\n[{ckpt}] No results found in {run_dir}")
            continue

        print(f"\n{'─'*60}")
        print(f"Checkpoint: {ckpt}")

        # Print all arms
        ppl_none = arms.get("none")
        print(f"  none (baseline):   {ppl_none:.2f}" if ppl_none else "  none: N/A")

        for df_var in ["freeze_0.5", "freeze_1.0", "soft_0.3"]:
            ppl_df   = arms.get(f"{df_var}_only")
            ppl_sc   = arms.get("sc_only")
            ppl_dfsc = arms.get(f"{df_var}_sc")

            print(f"\n  ── {df_var} ──────────────────────────")
            print(f"    SC-only:        {ppl_sc:.2f}" if ppl_sc else "    SC-only:   N/A")
            print(f"    DF-only:        {ppl_df:.2f}" if ppl_df else "    DF-only:   N/A")
            print(f"    DF + SC:        {ppl_dfsc:.2f}" if ppl_dfsc else "    DF + SC:  N/A")

            if ppl_none and ppl_df and ppl_sc and ppl_dfsc:
                main_df = ppl_df - ppl_none    # negative = DF helps
                main_sc = ppl_sc - ppl_none    # negative = SC helps
                I = compute_interaction(ppl_none, ppl_df, ppl_sc, ppl_dfsc)
                print(f"    Main effect DF: {main_df:+.2f}  {'(DF helps)' if main_df < 0 else '(DF hurts)'}")
                print(f"    Main effect SC: {main_sc:+.2f}  {'(SC helps)' if main_sc < 0 else '(SC hurts)'}")
                print(f"    Interaction I:  {I:+.2f}  "
                      f"{'(complementary)' if I < 0 else '(competitive/redundant)' if I > 0 else '(additive)'}")
                expected_dfsc = ppl_none + main_df + main_sc
                print(f"    Predicted (additive): {expected_dfsc:.2f}  Observed: {ppl_dfsc:.2f}  "
                      f"Gap={ppl_dfsc - expected_dfsc:+.2f}")

    # ── Cross-checkpoint comparison ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Cross-checkpoint summary: freeze_1.0 interaction I")
    for ckpt in CHECKPOINTS:
        arms = all_results.get(ckpt, {})
        ppl_none = arms.get("none")
        ppl_df   = arms.get("freeze_1.0_only")
        ppl_sc   = arms.get("sc_only")
        ppl_dfsc = arms.get("freeze_1.0_sc")
        I = compute_interaction(ppl_none, ppl_df, ppl_sc, ppl_dfsc)
        if I is not None:
            print(f"  {ckpt:10s}: I={I:+.2f}  (none={ppl_none:.1f}, DF={ppl_df:.1f}, "
                  f"SC={ppl_sc:.1f}, DF+SC={ppl_dfsc:.1f})")
        else:
            print(f"  {ckpt:10s}: N/A (missing arms)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = OUTPUTS / "spec36_factorial_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
