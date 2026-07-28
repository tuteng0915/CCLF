"""
EXP-27v2: LangFlow Token Frequency vs Commitment Timing (GPT-2 Tokenizer Fix)

PROBLEM WITH EXP-27: Used T5 SentencePiece token IDs as frequency proxy.
LangFlow uses GPT-2 BPE tokenizer (vocab_size=50257). GPT-2 IDs are NOT
frequency-sorted (BPE merge order ≠ unigram frequency).

FIX: Compute actual GPT-2 unigram frequencies from a large OWT sample,
     then correlate with exp25 commitment timing.

OUTPUTS:
  results/exp27v2_langflow/
    gpt2_token_freq.json     — GPT-2 token ID → log10(freq_per_million) for all types
    freq_commitment_v2.json  — per-type stats + correlation results

Usage (from CCLF root, CPU-only):
  conda run -n elf python experiments/probe_langflow/analyze_freq_commitment_v2.py \\
    --exp25_dir results/exp25_langflow \\
    --out_dir   results/exp27v2_langflow \\
    --n_freq_docs 2000
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as stats


SPECIAL_IDS = {50256}  # GPT-2 EOS/PAD token (ID 50256 = <|endoftext|>)


FUNCTION_WORDS = frozenset([
    # Determiners / articles
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "every", "any", "no", "all", "both", "each", "either", "neither",
    "some", "another",
    # Core prepositions
    "of", "in", "to", "for", "on", "at", "with", "by", "from", "as",
    "into", "through", "about", "between", "after", "before", "around",
    "up", "out", "down", "under", "over", "above", "below", "across",
    "against", "along", "among", "behind", "beneath", "beside", "besides",
    "beyond", "during", "except", "inside", "near", "off", "since", "than",
    "throughout", "toward", "underneath", "until", "upon", "within", "without",
    # Conjunctions
    "and", "but", "or", "nor", "if", "when", "where", "while", "because",
    "since", "although", "though", "unless", "until", "so", "yet",
    # Auxiliaries / modals
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall", "can",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "who", "whom", "whose", "what", "which", "whoever", "whatever",
    # Other common function
    "not", "very", "just", "also", "even", "only", "well",
    "already", "still", "then", "there", "here",
    "however", "therefore", "thus", "also", "more",
    # Punctuation tokens
    ".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "-", "–", "—",
])


def is_function_token(token_str: str) -> bool:
    """True if the GPT-2 token surface form (after stripping leading space) is a function word."""
    # GPT-2 BPE: word-initial space encoded as 'Ġ' (U+0120)
    surf = token_str.lstrip("Ġ").lower()
    return surf in FUNCTION_WORDS or token_str.strip() in FUNCTION_WORDS


def load_owt_streaming(n_docs: int, tokenizer):
    """
    Load up to n_docs OWT documents and tokenize with GPT-2 tokenizer.
    Returns Counter of token_id → count.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets package not available")

    freq = Counter()
    n_loaded = 0
    total_tokens = 0

    for ds_name, ds_kwargs in [
        ("Skylion007/openwebtext", {}),
        ("stas/openwebtext-10k",   {}),
        ("wikitext", {"name": "wikitext-103-raw-v1"}),
    ]:
        try:
            ds = load_dataset(ds_name, split="train", streaming=True, **ds_kwargs)
            for ex in ds:
                text = ex["text"].strip()
                if len(text) < 50:
                    continue
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                for tid in ids:
                    if tid not in SPECIAL_IDS:
                        freq[tid] += 1
                        total_tokens += 1
                n_loaded += 1
                if n_loaded >= n_docs:
                    break
            if n_loaded > 0:
                print(f"[freq] loaded {n_loaded} docs from {ds_name}, {total_tokens:,} tokens")
                return freq, total_tokens
        except Exception as e:
            print(f"[freq] {ds_name} failed: {e}")

    raise RuntimeError("Could not load any OWT dataset.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp25_dir",   default="results/exp25_langflow")
    ap.add_argument("--out_dir",     default="results/exp27v2_langflow")
    ap.add_argument("--n_freq_docs", type=int, default=2000,
                    help="Number of OWT docs for frequency estimation (~512 tokens each = ~1M tokens)")
    ap.add_argument("--min_occ",     type=int, default=5,
                    help="Min occurrences in exp25 to include a token type in correlation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load GPT-2 tokenizer ────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[tok] GPT-2 tokenizer loaded, vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        raise RuntimeError(f"Cannot load GPT-2 tokenizer: {e}")

    # ── 2. Compute GPT-2 token frequencies ────────────────────────────────
    freq_path = out_dir / "gpt2_token_freq.json"
    if freq_path.exists():
        print(f"[freq] loading cached {freq_path}")
        with open(freq_path) as f:
            freq_data = json.load(f)
        freq_counter = Counter({int(k): v for k, v in freq_data["counts"].items()})
        total_tokens = freq_data["total_tokens"]
    else:
        print(f"[freq] computing from {args.n_freq_docs} OWT docs ...")
        freq_counter, total_tokens = load_owt_streaming(args.n_freq_docs, tokenizer)
        # Save raw counts
        freq_data = {
            "total_tokens": total_tokens,
            "n_docs": args.n_freq_docs,
            "counts": {str(k): v for k, v in freq_counter.items()},
        }
        with open(freq_path, "w") as f:
            json.dump(freq_data, f)
        print(f"[freq] saved → {freq_path}")

    # Compute log10(freq per million) for each token
    ppm = {tid: (cnt / total_tokens) * 1e6 for tid, cnt in freq_counter.items()}
    log_ppm = {tid: np.log10(v) for tid, v in ppm.items() if v > 0}

    print(f"[freq] {len(ppm):,} unique token types in OWT sample")
    print(f"       top-5 most frequent: {[tokenizer.decode([t]) for t, _ in freq_counter.most_common(5)]}")

    # ── 3. Load exp25 commitment data ─────────────────────────────────────
    print(f"\n[exp25] loading from {args.exp25_dir}")
    commit_tidx = np.load(f"{args.exp25_dir}/commit_tidx.npy")   # [n, L], int32
    gt_tokens   = np.load(f"{args.exp25_dir}/gt_tokens.npy")     # [n, L], int32
    t_grid      = np.load(f"{args.exp25_dir}/t_grid.npy")        # [n_t], float32
    n_t = len(t_grid)
    NEVER = n_t

    flat_tokens = gt_tokens.reshape(-1).astype(int)   # [N]
    flat_tidx   = commit_tidx.reshape(-1).astype(int)  # [N]
    N = len(flat_tokens)
    print(f"[exp25] {commit_tidx.shape[0]} seqs × {commit_tidx.shape[1]} pos = {N} occurrences")
    print(f"[exp25] never-committed: {(flat_tidx == NEVER).sum()} / {N} ({100*(flat_tidx==NEVER).mean():.1f}%)")

    # Build per-type stats
    tok_tidxs = defaultdict(list)
    for tid, ti in zip(flat_tokens, flat_tidx):
        tok_tidxs[int(tid)].append(int(ti))

    # ── 4. Compute per-type mean commit t* and join with frequency ─────────
    per_type = {}
    for tok_id, tidxs in tok_tidxs.items():
        if tok_id in SPECIAL_IDS:
            continue
        n_occ = len(tidxs)
        if n_occ < args.min_occ:
            continue
        n_never = sum(1 for x in tidxs if x == NEVER)
        commit_only = [t_grid[x] for x in tidxs if x < NEVER]
        mean_t = float(np.mean(commit_only)) if commit_only else None
        tok_str = tokenizer.decode([tok_id])

        per_type[tok_id] = {
            "token_str": tok_str,
            "n_occ":     n_occ,
            "n_never":   n_never,
            "never_rate": n_never / n_occ,
            "mean_commit_t": mean_t,
            "is_function": is_function_token(tok_str),
            "log_ppm": log_ppm.get(tok_id, None),
        }

    print(f"\n[types] {len(per_type)} token types with ≥{args.min_occ} occurrences")

    # ── 5. Correlation: log_ppm vs mean_commit_t (excluding never-only types)
    valid = [
        (d["log_ppm"], d["mean_commit_t"], tok_id)
        for tok_id, d in per_type.items()
        if d["log_ppm"] is not None and d["mean_commit_t"] is not None
    ]
    print(f"[corr] {len(valid)} types with both log_ppm and mean_commit_t")

    corr_results = {}
    if len(valid) >= 10:
        log_ppm_arr   = np.array([v[0] for v in valid])
        mean_commit_arr = np.array([v[1] for v in valid])

        r_pearson, p_pearson = stats.pearsonr(log_ppm_arr, mean_commit_arr)
        r_spearman, p_spearman = stats.spearmanr(log_ppm_arr, mean_commit_arr)

        corr_results = {
            "n_types": len(valid),
            "pearson_r":    float(r_pearson),
            "pearson_p":    float(p_pearson),
            "spearman_r":   float(r_spearman),
            "spearman_p":   float(p_spearman),
        }
        print(f"\n[CORRELATION RESULTS]")
        print(f"  Pearson  r = {r_pearson:+.4f}  (p={p_pearson:.4g})  [n={len(valid)} types]")
        print(f"  Spearman r = {r_spearman:+.4f}  (p={p_spearman:.4g})")
        print(f"  Direction: {'higher freq → earlier commit' if r_pearson < 0 else 'higher freq → later commit'}")

        # Compare function vs content word commitment
        func_types   = [(d["log_ppm"], d["mean_commit_t"]) for d in per_type.values()
                        if d["is_function"] and d["mean_commit_t"] is not None and d["log_ppm"] is not None]
        content_types = [(d["log_ppm"], d["mean_commit_t"]) for d in per_type.values()
                         if not d["is_function"] and d["mean_commit_t"] is not None and d["log_ppm"] is not None]

        if func_types and content_types:
            func_t_arr    = np.array([x[1] for x in func_types])
            content_t_arr = np.array([x[1] for x in content_types])
            u_stat, p_mwu = stats.mannwhitneyu(func_t_arr, content_t_arr, alternative="less")
            print(f"\n[FUNCTION vs CONTENT]")
            print(f"  function words: n={len(func_types)}, mean t*={func_t_arr.mean():.4f}")
            print(f"  content words:  n={len(content_types)}, mean t*={content_t_arr.mean():.4f}")
            print(f"  Δ(func - content) = {func_t_arr.mean() - content_t_arr.mean():+.4f}")
            print(f"  Mann-Whitney U (func < content): U={u_stat:.0f}, p={p_mwu:.4g}")
            corr_results["func_n_types"] = len(func_types)
            corr_results["content_n_types"] = len(content_types)
            corr_results["func_mean_t"] = float(func_t_arr.mean())
            corr_results["content_mean_t"] = float(content_t_arr.mean())
            corr_results["func_content_delta"] = float(func_t_arr.mean() - content_t_arr.mean())
            corr_results["mannwhitney_U"] = float(u_stat)
            corr_results["mannwhitney_p"] = float(p_mwu)

        # Partial correlation: r(log_ppm, t* | is_function) residual correlation
        # This checks: does frequency predict t* even within function-word and content-word groups?
        is_func_arr = np.array([
            1.0 if per_type[tok_id]["is_function"] else 0.0 for _, _, tok_id in valid
        ])
        # Regress out is_function from both log_ppm and t*, then correlate residuals
        from numpy.linalg import lstsq
        X_ctrl = np.column_stack([np.ones(len(is_func_arr)), is_func_arr])
        resid_ppm   = log_ppm_arr   - X_ctrl @ lstsq(X_ctrl, log_ppm_arr,   rcond=None)[0]
        resid_t     = mean_commit_arr - X_ctrl @ lstsq(X_ctrl, mean_commit_arr, rcond=None)[0]
        r_partial, p_partial = stats.pearsonr(resid_ppm, resid_t)
        print(f"\n[PARTIAL CORRELATION] r(log_ppm, t* | controlling for is_function)")
        print(f"  Pearson r_partial = {r_partial:+.4f}  (p={p_partial:.4g})")
        corr_results["partial_r_freq_given_func"] = float(r_partial)
        corr_results["partial_p_freq_given_func"] = float(p_partial)

        # Bin analysis: sort by frequency quintile, report mean t* per bin
        sorted_idx = np.argsort(log_ppm_arr)
        bins = np.array_split(sorted_idx, 5)
        print(f"\n[FREQUENCY BINS] (quintiles, low→high freq)")
        bin_summary = []
        for i, bidx in enumerate(bins):
            bin_log_ppm = log_ppm_arr[bidx].mean()
            bin_t       = mean_commit_arr[bidx].mean()
            print(f"  Q{i+1}: mean log_ppm={bin_log_ppm:.2f}, mean t*={bin_t:.4f} (n={len(bidx)})")
            bin_summary.append({"quintile": i+1, "mean_log_ppm": float(bin_log_ppm),
                                 "mean_t": float(bin_t), "n": len(bidx)})
        corr_results["frequency_quintiles"] = bin_summary

    # ── 6. Save results ────────────────────────────────────────────────────
    out = {
        "metadata": {
            "n_freq_docs": args.n_freq_docs,
            "total_tokens": total_tokens,
            "tokenizer": "gpt2",
            "min_occ": args.min_occ,
        },
        "correlation": corr_results,
        "per_type": {
            str(k): v for k, v in sorted(per_type.items(),
                key=lambda x: x[1]["n_occ"], reverse=True)[:500]
        },
    }
    out_path = out_dir / "freq_commitment_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
