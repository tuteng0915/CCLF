"""
analyze_spec31v2_multiseed.py — Aggregate EXP-31v2 multi-seed results

Reads outputs/spec31v2_{kd_cr,kd2}_seed{0..4}/ and reports:
  - Mean PPL ± std across 5 seeds for each checkpoint × condition
  - DF improvement delta: freeze_1.0 PPL − none PPL (per seed)
  - Sign consistency: does DF help (+) or hurt (-) across all seeds?
  - Degeneration rate per condition/seed
  - Does kd_cr ↔ kd2 sign reversal persist across seeds?

Usage (from ELF-torch root):
  python experiments/probe_elf/analyze_spec31v2_multiseed.py
"""

import json, os
from pathlib import Path
from collections import defaultdict
import numpy as np

OUTPUTS = Path("outputs")
CHECKPOINTS = ["kd_cr", "kd2"]
SEEDS = list(range(5))  # 0, 1, 2, 3, 4

CONDITIONS_SPEC = [
    ("none",       ["ode-steps32-cfg1-ts_uniform-uncond"]),
    ("freeze_0.5", ["ode-steps32-cfg1-ts_uniform-df_freeze0.5-dftmin0.7-uncond"]),
    ("freeze_1.0", ["ode-steps32-cfg1-ts_uniform-df_freeze1.0-dftmin0.7-uncond"]),
    ("soft_0.3",   ["ode-steps32-cfg1-ts_uniform-df_soft0.3-dftmin0.7-uncond"]),
]


def load_ppl(run_dir: Path, subdir_patterns: list):
    """Load PPL from metrics.jsonl in the first matching subdir."""
    for pat in subdir_patterns:
        path = run_dir / pat / "metrics.jsonl"
        if path.exists():
            with open(path) as f:
                d = json.loads(f.read())
            return d.get("ppl")
    # Try glob
    for pat in subdir_patterns:
        matches = list(run_dir.glob(f"{pat}*/metrics.jsonl"))
        if matches:
            with open(matches[0]) as f:
                d = json.loads(f.read())
            return d.get("ppl")
    return None


def load_degen_rate(run_dir: Path, subdir_patterns: list, threshold: float = 0.2):
    """Fraction of samples with repeating unigrams ≥ threshold."""
    for pat in subdir_patterns:
        paths = list(run_dir.glob(f"{pat}*/all_generated*.jsonl"))
        if not paths:
            paths = [run_dir / pat / "all_generated_0_95085.jsonl"]
        for path in paths:
            if not path.exists():
                continue
            n_degen = 0
            n_total = 0
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        text = d.get("generated", d.get("text", ""))
                        words = text.split()
                        if len(words) >= 10:
                            from collections import Counter
                            cnt = Counter(words)
                            if cnt.most_common(1)[0][1] / len(words) >= threshold:
                                n_degen += 1
                        n_total += 1
                    except Exception:
                        continue
            if n_total > 0:
                return n_degen / n_total
    return None


def main():
    print("=" * 80)
    print("EXP-31v2 Multi-seed: Diffusion Forcing (tmin=0.7), seeds 0-4")
    print("Key question: does DF sign reversal (kd_cr vs kd2) persist across seeds?")
    print("=" * 80)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # results[ckpt][seed][cond] = {"ppl": ..., "degen": ...}

    for ckpt in CHECKPOINTS:
        for seed in SEEDS:
            run_dir = OUTPUTS / f"spec31v2_{ckpt}_seed{seed}"
            if not run_dir.exists():
                print(f"  [missing] {run_dir}")
                continue
            for cond_name, patterns in CONDITIONS_SPEC:
                ppl  = load_ppl(run_dir, patterns)
                degen = load_degen_rate(run_dir, patterns)
                results[ckpt][seed][cond_name] = {"ppl": ppl, "degen": degen}

    # ── Per-checkpoint summary ───────────────────────────────────────────────
    agg = defaultdict(lambda: defaultdict(list))  # agg[ckpt][cond] = list of ppls

    for ckpt in CHECKPOINTS:
        print(f"\n{'─'*60}")
        print(f"Checkpoint: {ckpt}")
        print(f"{'seed':>5} | " + " ".join(f"{c:>12s}" for c, _ in CONDITIONS_SPEC) + "  | degeneration")
        print(f"{'─'*5}-|-" + "-".join("-" * 13 for _ in CONDITIONS_SPEC) + "--+------------")

        for seed in SEEDS:
            if seed not in results[ckpt]:
                continue
            row = f"{seed:>5} | "
            degen_none = results[ckpt][seed].get("none", {}).get("degen")
            for cond_name, _ in CONDITIONS_SPEC:
                d = results[ckpt][seed].get(cond_name, {})
                ppl = d.get("ppl")
                if ppl is not None:
                    row += f"{ppl:12.2f}  "
                    agg[ckpt][cond_name].append(ppl)
                else:
                    row += f"{'N/A':>12s}  "
            degen_pct = f"{100*degen_none:.1f}%" if degen_none is not None else "N/A"
            row += f"| degen(none)={degen_pct}"
            print(row)

        # Mean ± std
        print(f"{'mean':>5} | " + " ".join(
            f"{np.mean(agg[ckpt][c]):12.2f}  " if agg[ckpt][c] else f"{'N/A':>12s}  "
            for c, _ in CONDITIONS_SPEC
        ))
        print(f"{'std':>5} | " + " ".join(
            f"{np.std(agg[ckpt][c]):12.2f}  " if len(agg[ckpt][c]) > 1 else f"{'N/A':>12s}  "
            for c, _ in CONDITIONS_SPEC
        ))

    # ── Key analysis: DF delta (freeze_1.0 - none) per seed ─────────────────
    print(f"\n{'─'*60}")
    print("DF Delta: freeze_1.0 PPL − none PPL (negative = DF helps)")
    print(f"{'seed':>5} | " + " ".join(f"{c:>10s}" for c in CHECKPOINTS))

    deltas = defaultdict(list)  # deltas[ckpt] = list of delta per seed
    for seed in SEEDS:
        row = f"{seed:>5} | "
        for ckpt in CHECKPOINTS:
            ppl_none   = results[ckpt].get(seed, {}).get("none", {}).get("ppl")
            ppl_freeze = results[ckpt].get(seed, {}).get("freeze_1.0", {}).get("ppl")
            if ppl_none is not None and ppl_freeze is not None:
                delta = ppl_freeze - ppl_none
                deltas[ckpt].append(delta)
                row += f"{delta:+10.2f}  "
            else:
                row += f"{'N/A':>10s}  "
        print(row)

    print(f"\n{'mean':>5} | " + " ".join(
        f"{np.mean(deltas[c]):+10.2f}  " if deltas[c] else f"{'N/A':>10s}  "
        for c in CHECKPOINTS
    ))
    print(f"{'sign':>5} | " + " ".join(
        f"{'all_neg' if all(d < 0 for d in deltas[c]) else 'all_pos' if all(d > 0 for d in deltas[c]) else 'MIXED':>10s}  "
        if deltas[c] else f"{'N/A':>10s}  "
        for c in CHECKPOINTS
    ))

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("VERDICT: Does kd_cr ↔ kd2 sign reversal persist?")
    kd_cr_deltas = deltas.get("kd_cr", [])
    kd2_deltas   = deltas.get("kd2", [])
    if kd_cr_deltas and kd2_deltas:
        kd_cr_sign = np.sign(np.mean(kd_cr_deltas))
        kd2_sign   = np.sign(np.mean(kd2_deltas))
        if kd_cr_sign != kd2_sign:
            print(f"  YES — kd_cr mean_delta={np.mean(kd_cr_deltas):+.2f}, kd2 mean_delta={np.mean(kd2_deltas):+.2f}")
            print(f"  → Sign reversal is robust to seed choice")
            print(f"  → Supports hypothesis: kd_cr and kd2 have OPPOSITE DF sensitivity")
        else:
            print(f"  NO — kd_cr mean_delta={np.mean(kd_cr_deltas):+.2f}, kd2 mean_delta={np.mean(kd2_deltas):+.2f}")
            print(f"  → Same sign. Original finding was seed artifact (seed=123/456).")
            print(f"  → DF effect is similar across checkpoints when using neutral seeds.")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {"results": {k: {str(s): dict(v) for s, v in vv.items()} for k, vv in results.items()},
           "deltas": {k: v for k, v in deltas.items()}}
    out_path = OUTPUTS / "spec31v2_multiseed_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
