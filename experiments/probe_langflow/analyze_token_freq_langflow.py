"""
EXP-27: LangFlow Token Frequency vs Commitment Timing (LangFlow analog of EXP-20)

Loads EXP-25 per-position commitment data and analyzes:
1. Relationship between T5 token rank (frequency proxy) and mean commit time t*
2. Never-committed token distribution (token types that are always uncertain)
3. Always-committed tokens (token types that commit earliest)

For T5 SentencePiece:
  - Token ID 0 = <pad>, 1 = </s>, 2 = <unk>
  - IDs 3~32099 are BPE/SentencePiece pieces in approximately
    decreasing frequency order (lower ID → more frequent in training)

Usage (from CCLF root):
  conda run -n elf python experiments/probe_langflow/analyze_token_freq_langflow.py \
    --exp25_dir results/exp25_langflow \
    --out_dir results/exp27_langflow
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SPECIAL_IDS = {0, 1, 2}  # pad, </s>, <unk>


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp25_dir", default="results/exp25_langflow")
    p.add_argument("--out_dir", default="results/exp27_langflow")
    p.add_argument("--min_occ", type=int, default=10, help="Min occurrences to include a token type")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.exp25_dir}")
    commit_tidx = np.load(f"{args.exp25_dir}/commit_tidx.npy")  # [n, L]
    gt_tokens   = np.load(f"{args.exp25_dir}/gt_tokens.npy")    # [n, L]
    t_grid      = np.load(f"{args.exp25_dir}/t_grid.npy")       # [n_t]

    n_samples, L = commit_tidx.shape
    n_t = len(t_grid)
    NEVER = n_t

    print(f"[config] n_samples={n_samples}, L={L}, n_t={n_t}")
    print(f"         t_grid: [{t_grid[0]:.3f}, {t_grid[-1]:.3f}]")

    # Load tokenizer for token string decoding
    try:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        has_tokenizer = True
        print("[tokenizer] T5Tokenizer loaded")
    except Exception as e:
        has_tokenizer = False
        print(f"[tokenizer] could not load: {e}")

    flat_tokens  = gt_tokens.reshape(-1).astype(int)   # [N]
    flat_tidx    = commit_tidx.reshape(-1).astype(int)  # [N]

    # ── Per-token-type aggregation ─────────────────────────────────────────
    tok_commit = defaultdict(list)  # token_id → list of commit t-indices (NEVER if never)
    for tok_id, tidx in zip(flat_tokens, flat_tidx):
        tok_commit[int(tok_id)].append(int(tidx))

    # ── Compute statistics per token type ─────────────────────────────────
    tok_stats = {}
    for tok_id, tidx_list in tok_commit.items():
        n = len(tidx_list)
        if n < args.min_occ:
            continue
        never_count  = sum(1 for x in tidx_list if x == NEVER)
        commit_tidxs = [x for x in tidx_list if x < NEVER]
        tok_stats[tok_id] = {
            "tok_id": tok_id,
            "n": n,
            "never_rate": never_count / n,
            "mean_t_star": float(np.mean([t_grid[x] for x in commit_tidxs])) if commit_tidxs else None,
            "median_t_star": float(np.median([t_grid[x] for x in commit_tidxs])) if commit_tidxs else None,
            "piece": tokenizer.convert_ids_to_tokens([tok_id])[0] if has_tokenizer else None,
        }

    print(f"[stats] {len(tok_stats)} token types with ≥{args.min_occ} occurrences")

    # ── Frequency band analysis ─────────────────────────────────────────────
    # T5 SentencePiece: ID 3..32099 sorted by decreasing training frequency
    # Bands: [3, 100), [100, 500), [500, 2000), [2000, 10000), [10000, 32100)
    bands = [
        ("very_common",  3,     100),
        ("common",       100,   500),
        ("medium",       500,   2000),
        ("uncommon",     2000,  10000),
        ("rare",         10000, 32100),
    ]

    band_results = []
    for band_name, lo, hi in bands:
        subset = [v for tok_id, v in tok_stats.items()
                  if lo <= tok_id < hi and tok_id not in SPECIAL_IDS]
        if not subset:
            continue
        all_t_stars = [v["mean_t_star"] for v in subset if v["mean_t_star"] is not None]
        all_never   = [v["never_rate"] for v in subset]
        band_results.append({
            "band": band_name,
            "id_range": f"[{lo}, {hi})",
            "n_types": len(subset),
            "mean_t_star": float(np.mean(all_t_stars)) if all_t_stars else None,
            "mean_never_rate": float(np.mean(all_never)),
        })

    # ── Special tokens ─────────────────────────────────────────────────────
    special_stats = {tok_id: tok_stats[tok_id] for tok_id in SPECIAL_IDS if tok_id in tok_stats}

    # ── Top/bottom never-commit token types ────────────────────────────────
    all_types = [(v["tok_id"], v["never_rate"], v["mean_t_star"], v["n"], v["piece"])
                 for v in tok_stats.values() if v["tok_id"] not in SPECIAL_IDS]

    top_never  = sorted(all_types, key=lambda x: -x[1])[:20]   # highest never_rate
    top_early  = sorted([x for x in all_types if x[2] is not None],
                        key=lambda x: x[2])[:20]                 # earliest mean t*
    top_late   = sorted([x for x in all_types if x[2] is not None],
                        key=lambda x: -x[2])[:20]                # latest mean t*

    # ── Correlation: token ID (log scale) vs mean t* ───────────────────────
    has_t = [(np.log10(max(v["tok_id"], 3)), v["mean_t_star"])
             for v in tok_stats.values()
             if v["mean_t_star"] is not None and v["tok_id"] not in SPECIAL_IDS]
    if len(has_t) > 10:
        log_ids = np.array([x[0] for x in has_t])
        t_stars = np.array([x[1] for x in has_t])
        corr = float(np.corrcoef(log_ids, t_stars)[0, 1])
    else:
        corr = None

    # ── Save results ───────────────────────────────────────────────────────
    out = {
        "config": vars(args),
        "n_samples": n_samples, "L": L, "n_t": n_t,
        "n_positions": n_samples * L,
        "frac_never_committed": float((flat_tidx == NEVER).mean()),
        "band_results": band_results,
        "special_token_stats": {str(k): v for k, v in special_stats.items()},
        "top_never_commit": [
            {"tok_id": x[0], "never_rate": x[1], "mean_t_star": x[2], "n": x[3], "piece": x[4]}
            for x in top_never
        ],
        "top_early_commit": [
            {"tok_id": x[0], "never_rate": x[1], "mean_t_star": x[2], "n": x[3], "piece": x[4]}
            for x in top_early
        ],
        "top_late_commit": [
            {"tok_id": x[0], "never_rate": x[1], "mean_t_star": x[2], "n": x[3], "piece": x[4]}
            for x in top_late
        ],
        "corr_logid_vs_tstar": corr,
    }
    out_path = out_dir / "token_freq_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}")

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n── EXP-27 LangFlow Token Frequency vs Commitment Timing ──────────────")
    print(f"  n_positions = {n_samples * L:,}   frac_never = {(flat_tidx==NEVER).mean():.3f}")
    print(f"  correlation(log10(tok_id), t*) = {corr:.4f}" if corr else "  correlation: N/A")
    print()
    print(f"  {'Band':12s}  {'ID range':12s}  {'n_types':>8}  {'mean_t*':>9}  {'never_rate':>11}")
    print(f"  {'-'*60}")
    for row in band_results:
        t_str = f"{row['mean_t_star']:.4f}" if row["mean_t_star"] is not None else "N/A"
        print(f"  {row['band']:12s}  {row['id_range']:12s}  {row['n_types']:>8}  "
              f"{t_str:>9}  {row['mean_never_rate']*100:>9.1f}%")

    if special_stats:
        print(f"\n  Special tokens (pad/EOS/unk):")
        for tok_id, v in special_stats.items():
            t_str = f"{v['mean_t_star']:.4f}" if v["mean_t_star"] is not None else "NEVER"
            print(f"    id={tok_id}  piece={v['piece']}  n={v['n']}  "
                  f"never_rate={v['never_rate']*100:.0f}%  mean_t*={t_str}")

    print(f"\n  Top-5 never-commit token types (excl. special):")
    for x in top_never[:5]:
        print(f"    id={x[0]:5d}  piece={str(x[4])!r:15s}  never_rate={x[1]*100:.0f}%  n={x[3]}")

    print(f"\n  Top-5 earliest-commit token types:")
    for x in top_early[:5]:
        print(f"    id={x[0]:5d}  piece={str(x[4])!r:15s}  mean_t*={x[2]:.4f}  n={x[3]}")

    print(f"\n  Top-5 latest-commit token types:")
    for x in top_late[:5]:
        print(f"    id={x[0]:5d}  piece={str(x[4])!r:15s}  mean_t*={x[2]:.4f}  n={x[3]}")


if __name__ == "__main__":
    main()
