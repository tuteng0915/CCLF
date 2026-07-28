"""EXP-GS1 analysis: compute tau_topic / tau_syntax / tau_token from
probe_hierarchy_<label>.json and check whether tau_topic < tau_syntax < tau_token
(H1, global-before-local).

Threshold rule (suite doc Section 3): tau_k = min{t : G_k(t) >= 0.8 * G_k(clean)},
"clean-state performance" fraction, not a shared absolute threshold across tasks.

Usage:
    python experiments/global_state/analyze_probe_transition.py \\
        results/global_state/elf/baseline/probe_hierarchy_pilot.json
"""

import argparse
import json


METRICS = [
    ("G_topic", "tau_topic"),
    ("G_syntax_r2", "tau_syntax"),
    ("G_token", "tau_token"),
]


def find_tau(records, key, clean_value, frac=0.8):
    threshold = frac * clean_value
    non_clean = [r for r in records if not r.get("is_clean_ref")]
    non_clean.sort(key=lambda r: r["t"])
    for r in non_clean:
        if r[key] >= threshold:
            return r["t"], threshold
    return None, threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--frac", type=float, default=0.8)
    args = ap.parse_args()

    with open(args.json_path) as f:
        summary = json.load(f)

    records = summary["records"]
    clean_rec = next(r for r in records if r.get("is_clean_ref"))

    print(f"[GS1-analyze] model={summary['model']} checkpoint={summary['checkpoint']} "
          f"label={summary['label']} n_samples={summary['n_samples']}")
    print(f"[GS1-analyze] clean-ref (t={clean_rec['t']:.3f}) performance: "
          f"G_topic={clean_rec['G_topic']:.3f}  G_syntax_r2={clean_rec['G_syntax_r2']:.3f}  "
          f"G_token={clean_rec['G_token']:.3f}")
    print()

    print(f"{'t':>7} | {'G_topic':>8} | {'G_syntax_r2':>11} | {'G_syntax_cos':>12} | "
          f"{'G_sent':>7} | {'G_token':>8}")
    for r in sorted([r for r in records if not r.get("is_clean_ref")], key=lambda r: r["t"]):
        print(f"{r['t']:7.3f} | {r['G_topic']:8.3f} | {r['G_syntax_r2']:11.3f} | "
              f"{r['G_syntax_cos']:12.3f} | {r['G_sent']:7.3f} | {r['G_token']:8.3f}")
    print()

    taus = {}
    for key, name in METRICS:
        clean_value = clean_rec[key]
        tau, threshold = find_tau(records, key, clean_value, args.frac)
        taus[name] = tau
        tau_str = f"t={tau:.3f}" if tau is not None else "NEVER (within grid)"
        print(f"[GS1-analyze] {name}: threshold={threshold:.3f} "
              f"({args.frac:.0%} of clean={clean_value:.3f}) -> {tau_str}")

    print()
    vals = [taus["tau_topic"], taus["tau_syntax"], taus["tau_token"]]
    if all(v is not None for v in vals):
        ordered = taus["tau_topic"] <= taus["tau_syntax"] <= taus["tau_token"]
        strict = taus["tau_topic"] < taus["tau_syntax"] < taus["tau_token"]
        print(f"[GS1-analyze] tau_topic({taus['tau_topic']:.3f}) <= "
              f"tau_syntax({taus['tau_syntax']:.3f}) <= tau_token({taus['tau_token']:.3f}): "
              f"{ordered} (strict: {strict})")
        if strict:
            print("[GS1-analyze] H1 (global-before-local) ordering SUPPORTED at pilot scale.")
        elif ordered:
            print("[GS1-analyze] H1 ordering weakly supported (ties present).")
        else:
            print("[GS1-analyze] H1 ordering NOT supported at pilot scale -- "
                  "see EXP-GS1-spec.md caveat on G_token capacity mismatch before concluding.")
    else:
        print("[GS1-analyze] Cannot fully order tau's: at least one metric never reached "
              f"{args.frac:.0%} of clean performance within the pilot t-grid.")


if __name__ == "__main__":
    main()
