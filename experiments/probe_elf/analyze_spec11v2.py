"""
analyze_spec11v2.py — Summarize spec-11v2 Diffusion Forcing results across 3 checkpoints.

Usage:
  cd /home/wjzhang/tt_workspace/model/CCLF/CCLF/models/ELF-torch
  python experiments/probe_elf/analyze_spec11v2.py
"""

import json, os, re
from pathlib import Path

OUTPUTS = Path("outputs")

CHECKPOINTS = ["baseline", "kd_cr", "kd2"]
CONDITIONS  = [
    ("none",       ["ode-steps32-cfg1-ts_uniform-uncond"]),
    ("freeze_0.3", ["ode-steps32-cfg1-ts_uniform-df_freeze0.3-uncond",
                    "ode-steps32-cfg1-ts_uniform-df_freeze0.3-dftmin0.7-uncond"]),
    ("freeze_0.5", ["ode-steps32-cfg1-ts_uniform-df_freeze0.5-uncond",
                    "ode-steps32-cfg1-ts_uniform-df_freeze0.5-dftmin0.7-uncond"]),
    ("freeze_1.0", ["ode-steps32-cfg1-ts_uniform-df_freeze1.0-uncond",
                    "ode-steps32-cfg1-ts_uniform-df_freeze1.0-dftmin0.7-uncond"]),
    ("soft_0.3",   ["ode-steps32-cfg1-ts_uniform-df_soft0.3-uncond",
                    "ode-steps32-cfg1-ts_uniform-df_soft0.3-dftmin0.7-uncond"]),
    ("soft_0.5",   ["ode-steps32-cfg1-ts_uniform-df_soft0.5-uncond",
                    "ode-steps32-cfg1-ts_uniform-df_soft0.5-dftmin0.7-uncond"]),
    ("soft_0.7",   ["ode-steps32-cfg1-ts_uniform-df_soft0.7-uncond",
                    "ode-steps32-cfg1-ts_uniform-df_soft0.7-dftmin0.7-uncond"]),
]


def load_ppl(ckpt, subdirs):
    for subdir in (subdirs if isinstance(subdirs, list) else [subdirs]):
        path = OUTPUTS / f"spec11v2_{ckpt}" / subdir / "metrics.jsonl"
        if path.exists():
            with open(path) as f:
                d = json.loads(f.read())
            return d.get("ppl")
    return None


def load_degen(ckpt, subdirs, n_samples=256):
    """Count degenerate samples (≥20% repeating unigrams)."""
    for subdir in (subdirs if isinstance(subdirs, list) else [subdirs]):
        path = OUTPUTS / f"spec11v2_{ckpt}" / subdir / "all_generated_0_95085.jsonl"
    if not path.exists():
        # Try any jsonl
        outdir = OUTPUTS / f"spec11v2_{ckpt}" / subdir
        if not outdir.exists():
            return None
        jsons = list(outdir.glob("all_generated*.jsonl"))
        if not jsons:
            return None
        path = jsons[0]

    degen = 0
    total = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("generated", d.get("text", ""))
            words = text.split()
            if len(words) >= 10:
                from collections import Counter
                cnt = Counter(words)
                most_common_frac = cnt.most_common(1)[0][1] / len(words)
                if most_common_frac >= 0.2:
                    degen += 1
            total += 1
    return degen / total if total > 0 else None


def main():
    print("=" * 80)
    print("spec-11v2: Diffusion Forcing (tmin=0.7 gate), PPL at 32 ODE steps")
    print("=" * 80)

    results = {}
    for ckpt in CHECKPOINTS:
        results[ckpt] = {}
        for cond_name, subdir in CONDITIONS:
            ppl  = load_ppl(ckpt, subdir)
            degen = load_degen(ckpt, subdir)
            results[ckpt][cond_name] = {"ppl": ppl, "degen": degen}

    # Print table
    header = f"{'condition':15s}" + "".join(f"  {c:>12s}" for c in CHECKPOINTS)
    print(header)
    print("-" * len(header))

    ctrl_ppls = {}
    for cond_name, _ in CONDITIONS:
        row = f"{cond_name:15s}"
        for ckpt in CHECKPOINTS:
            d = results[ckpt][cond_name]
            if d["ppl"] is None:
                row += f"  {'---':>12s}"
            else:
                if cond_name == "none":
                    ctrl_ppls[ckpt] = d["ppl"]
                row += f"  {d['ppl']:>12.2f}"
        print(row)

    print()
    print("Δ vs none (% change):")
    header2 = f"{'condition':15s}" + "".join(f"  {c:>12s}" for c in CHECKPOINTS)
    print(header2)
    print("-" * len(header2))
    for cond_name, _ in CONDITIONS:
        if cond_name == "none":
            continue
        row = f"{cond_name:15s}"
        for ckpt in CHECKPOINTS:
            d = results[ckpt][cond_name]
            ctrl = ctrl_ppls.get(ckpt)
            if d["ppl"] is None or ctrl is None:
                row += f"  {'---':>12s}"
            else:
                delta_pct = (d["ppl"] - ctrl) / ctrl * 100
                marker = " ✓" if delta_pct < -2 else ("  " if delta_pct < 2 else " ✗")
                row += f"  {delta_pct:>+10.1f}%{marker}"
        print(row)

    # Save JSON
    out_path = Path("results/spec11v2_all/analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
