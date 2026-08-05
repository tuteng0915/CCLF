"""
EXP-08: Coarse-to-Fine Hypothesis

Uses EXP-09's per-position commitment timing (commit_times_matrix.npy) to test:
1. Whether earlier-committing positions have lower GPT-2 perplexity (more predictable)
2. Whether positions that commit early are semantically "coarser" (function words, common words)

No GPU needed: reads EXP-09 output and applies GPT-2 on CPU.

Usage (from ELF-torch root):
  python experiments/probe_elf/probe_coarse_to_fine.py \
    --exp09_dir results/exp09_kd_cr \
    --output_dir results/exp08_kd_cr \
    [--gpt2_ppl]          # optional: compute GPT-2 PPL per position (slow on CPU)
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


FUNCTION_WORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "must",
    "that", "this", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "not", "no",
    "if", "so", "then", "than", "when", "where", "how", "what", "which",
    "who", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "too", "very", "just", "also",
    "up", "out", "about", "into", "through", "after", "before", "between",
    ".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "-", "—",
])


def load_exp09_outputs(exp09_dir):
    commit_times_path = os.path.join(exp09_dir, "commit_times_matrix.npy")
    y_tokens_path = os.path.join(exp09_dir, "y_tokens_ref.npy")
    json_path = os.path.join(exp09_dir, "contextual_bootstrap.json")

    if not os.path.exists(commit_times_path):
        raise FileNotFoundError(
            f"commit_times_matrix.npy not found in {exp09_dir}\n"
            "Run probe_contextual_bootstrap.py first (with updated script that saves this file)."
        )

    commit_times = np.load(commit_times_path)   # (B, L) int, index into t_values
    y_tokens = np.load(y_tokens_path) if os.path.exists(y_tokens_path) else None

    with open(json_path) as f:
        meta = json.load(f)
    t_values = meta["t_values"]
    never_val = len(t_values)  # sentinel for never-committed positions

    return commit_times, y_tokens, t_values, never_val


def token_to_surface(token_id, tokenizer):
    if tokenizer is None:
        return None
    return tokenizer.convert_ids_to_tokens([token_id])[0]


def analyze_commit_by_token_type(commit_times, y_tokens, t_values, never_val, tokenizer):
    """Break down commitment timing by token surface category."""
    results = {}

    all_times = commit_times.flatten()
    # Convert index to actual t value, using inf for never-committed
    t_star = np.where(all_times < never_val,
                      np.array(t_values + [float("inf")])[all_times],
                      float("inf"))

    committed = all_times < never_val
    results["overall_mean_t_star"] = float(t_star[committed].mean())
    results["frac_never_committed"] = float((~committed).mean())

    if y_tokens is not None and tokenizer is not None:
        B, L = y_tokens.shape
        is_func = np.zeros(B * L, dtype=bool)
        token_ids_flat = y_tokens.flatten()
        for i, tid in enumerate(token_ids_flat):
            surf = tokenizer.convert_ids_to_tokens([int(tid)])[0]
            norm = surf.lstrip('▁').lower()
            if surf in FUNCTION_WORDS or surf.lower() in FUNCTION_WORDS or norm in FUNCTION_WORDS:
                is_func[i] = True

        func_times = t_star[is_func & committed]
        content_times = t_star[(~is_func) & committed]

        results["function_word"] = {
            "frac_of_tokens": float(is_func.mean()),
            "mean_t_star": float(func_times.mean()) if len(func_times) > 0 else None,
            "frac_committed": float((is_func & committed).sum() / is_func.sum()) if is_func.sum() > 0 else None,
        }
        results["content_word"] = {
            "frac_of_tokens": float((~is_func).mean()),
            "mean_t_star": float(content_times.mean()) if len(content_times) > 0 else None,
            "frac_committed": float(((~is_func) & committed).sum() / (~is_func).sum()) if (~is_func).sum() > 0 else None,
        }

        # Commitment by t_values step — function vs content
        results["commitment_by_step"] = []
        for i, t in enumerate(t_values):
            fn_frac = ((commit_times.flatten() == i) & is_func).sum() / is_func.sum() if is_func.sum() > 0 else 0
            ct_frac = ((commit_times.flatten() == i) & (~is_func)).sum() / (~is_func).sum() if (~is_func).sum() > 0 else 0
            results["commitment_by_step"].append({
                "t": t,
                "frac_func_first_commit_here": float(fn_frac),
                "frac_content_first_commit_here": float(ct_frac),
            })

    return results


def analyze_spatial_vs_type(commit_times, y_tokens, t_values, never_val, tokenizer):
    """
    For each position, check if its nearest already-committed neighbor
    at the previous step was a function word. If function-word neighbors
    accelerate commitment of content words, that supports coarse-to-fine bootstrapping.
    """
    if y_tokens is None or tokenizer is None:
        return None

    B, L = y_tokens.shape
    token_ids_flat = y_tokens.flatten()
    is_func_2d = np.zeros((B, L), dtype=bool)
    for b in range(B):
        for l in range(L):
            surf = tokenizer.convert_ids_to_tokens([int(y_tokens[b, l])])[0]
            norm = surf.lstrip('▁').lower()
            if surf in FUNCTION_WORDS or surf.lower() in FUNCTION_WORDS or norm in FUNCTION_WORDS:
                is_func_2d[b, l] = True

    T = len(t_values)
    results = []
    for step_idx in range(T - 1):
        committed_now = (commit_times <= step_idx)          # (B, L)
        func_committed = committed_now & is_func_2d         # (B, L) — function words already committed
        uncommitted_content = (~committed_now) & (~is_func_2d)  # (B, L) — content words not yet committed
        committed_next = (commit_times <= step_idx + 1)

        # For each uncommitted content word, find nearest func-committed neighbor within d=5
        near_func = np.zeros((B, L), dtype=bool)
        for d in range(1, 6):
            near_func[:, d:] |= func_committed[:, :-d]
            near_func[:, :-d] |= func_committed[:, d:]

        mask_with_func_neighbor = uncommitted_content & near_func
        mask_without_func_neighbor = uncommitted_content & (~near_func)

        row = {"step": step_idx, "t": t_values[step_idx]}
        if mask_with_func_neighbor.sum() > 0 and mask_without_func_neighbor.sum() > 0:
            row["commit_rate_with_func_neighbor"] = float(committed_next[mask_with_func_neighbor].mean())
            row["commit_rate_without_func_neighbor"] = float(committed_next[mask_without_func_neighbor].mean())
            row["n_with"] = int(mask_with_func_neighbor.sum())
            row["n_without"] = int(mask_without_func_neighbor.sum())
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp09_dir", required=True, help="EXP-09 output dir with commit_times_matrix.npy")
    parser.add_argument("--output_dir", default="results/exp08")
    parser.add_argument("--gpt2_ppl", action="store_true", help="Compute GPT-2 PPL per position (slow)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[EXP-08] Loading EXP-09 outputs from {args.exp09_dir}")
    try:
        commit_times, y_tokens, t_values, never_val = load_exp09_outputs(args.exp09_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  commit_times shape: {commit_times.shape}")
    print(f"  t_values: {t_values}")
    if y_tokens is not None:
        print(f"  y_tokens shape: {y_tokens.shape}")

    # Load T5 tokenizer for surface form lookup
    tokenizer = None
    try:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        print(f"  T5 tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    except Exception as e:
        print(f"  Warning: could not load T5 tokenizer ({e}); skipping token-type analysis")

    # Analyze commitment by token type
    print("\n[EXP-08] Analyzing commitment by token type...")
    type_results = analyze_commit_by_token_type(commit_times, y_tokens, t_values, never_val, tokenizer)

    # Print key stats
    print(f"  Overall mean t* (committed positions): {type_results['overall_mean_t_star']:.4f}")
    print(f"  Frac never committed: {type_results['frac_never_committed']:.4f}")
    if "function_word" in type_results:
        fn = type_results["function_word"]
        ct = type_results["content_word"]
        print(f"  Function words ({fn['frac_of_tokens']:.1%}): mean t* = {fn['mean_t_star']:.4f}, frac committed = {fn['frac_committed']:.4f}")
        print(f"  Content words  ({ct['frac_of_tokens']:.1%}): mean t* = {ct['mean_t_star']:.4f}, frac committed = {ct['frac_committed']:.4f}")
        delta = fn["mean_t_star"] - ct["mean_t_star"] if fn["mean_t_star"] and ct["mean_t_star"] else None
        if delta is not None:
            print(f"  Δ(func - content) t*: {delta:+.4f} ({'function commits earlier' if delta < 0 else 'content commits earlier'})")

    # Spatial: does function-word commitment accelerate nearby content words?
    print("\n[EXP-08] Analyzing function-word bootstrapping of content words...")
    spatial_type = analyze_spatial_vs_type(commit_times, y_tokens, t_values, never_val, tokenizer)
    if spatial_type:
        print("\n  Content-word commitment rate: with func neighbor (d≤5) vs without")
        for row in spatial_type:
            if "commit_rate_with_func_neighbor" in row:
                w = row["commit_rate_with_func_neighbor"]
                wo = row["commit_rate_without_func_neighbor"]
                print(f"  step {row['step']} (t={row['t']:.1f}): {w:.3f} vs {wo:.3f} (Δ={w-wo:+.3f}), n={row['n_with']}/{row['n_without']}")

    # Save results
    output = {
        "t_values": t_values,
        "token_type_analysis": type_results,
        "spatial_type_bootstrapping": spatial_type,
    }
    out_path = os.path.join(args.output_dir, "coarse_to_fine.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[EXP-08] Results saved to {out_path}")


if __name__ == "__main__":
    main()
