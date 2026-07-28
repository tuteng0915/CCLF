"""EXP-GS2 analysis: print the C_topic / C_struct / C_lex / C_sent consensus
curves vs t_start and find tau_k = min{t_start : C_k(t_start) >= threshold}
(all four metrics are already normalized to a fixed [0,1] "1=full consensus"
scale, so unlike EXP-GS1 no clean-state reference is needed -- see
EXP-GS2-spec.md Section 2).

Usage:
    python experiments/global_state/analyze_branch_hierarchy.py \\
        results/global_state/elf/baseline/branch_consensus_pilot.json
"""

import argparse
import json


METRICS = ["C_topic", "C_struct", "C_lex", "C_sent"]


def find_tau(records, key, threshold):
    for r in sorted(records, key=lambda r: r["t_start"]):
        if r[key] >= threshold:
            return r["t_start"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    with open(args.json_path) as f:
        summary = json.load(f)
    all_records = summary["records"]
    eta_grid = sorted(set(r["eta"] for r in all_records))

    print(f"[GS2-analyze] model={summary['model']} checkpoint={summary['checkpoint']} "
          f"label={summary['label']} n_docs={summary['n_docs']} K={summary['k_branches']} "
          f"eta_grid={eta_grid}")

    for eta in eta_grid:
        records = [r for r in all_records if r["eta"] == eta]
        print(f"\n=== eta={eta:g} ===")
        print(f"{'t_start':>8} | {'n_steps':>7} | {'C_topic':>8} | {'C_struct':>8} | "
              f"{'C_lex':>8} | {'C_sent':>8}")
        for r in sorted(records, key=lambda r: r["t_start"]):
            print(f"{r['t_start']:8.3f} | {r['n_steps']:7d} | {r['C_topic']:8.3f} | "
                  f"{r['C_struct']:8.3f} | {r['C_lex']:8.3f} | {r['C_sent']:8.3f}")

        taus = {}
        for key in METRICS:
            tau = find_tau(records, key, args.threshold)
            taus[key] = tau
            tau_str = f"t_start={tau:.3f}" if tau is not None else "NEVER (within grid)"
            print(f"[GS2-analyze] eta={eta:g} tau_{key}: threshold={args.threshold} -> {tau_str}")

        if taus["C_topic"] is not None and taus["C_lex"] is not None:
            rel = "<=" if taus["C_topic"] <= taus["C_lex"] else ">"
            verdict = "SUPPORTED" if taus["C_topic"] <= taus["C_lex"] else "NOT supported"
            print(f"[GS2-analyze] eta={eta:g}: tau_topic({taus['C_topic']:.3f}) {rel} "
                  f"tau_lex({taus['C_lex']:.3f}) -- global-before-local ordering {verdict}.")
        else:
            print(f"[GS2-analyze] eta={eta:g}: cannot compare tau_topic vs tau_lex "
                  f"(threshold={args.threshold} never reached by at least one metric).")

        if taus["C_topic"] is not None and taus["C_struct"] is not None:
            rel = "<=" if taus["C_topic"] <= taus["C_struct"] else ">"
            print(f"[GS2-analyze] eta={eta:g}: tau_topic({taus['C_topic']:.3f}) {rel} "
                  f"tau_struct({taus['C_struct']:.3f}) -- cross-check against EXP-GS1's "
                  f"tau_syntax < tau_topic finding (probe methodology).")


if __name__ == "__main__":
    main()
