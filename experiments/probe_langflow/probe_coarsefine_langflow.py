"""
EXP-25: LangFlow Coarse-to-Fine Hypothesis (LangFlow analog of EXP-08)

Tests whether LangFlow commits function words before content words,
analogous to EXP-08 for ELF (kd_cr: func mean t*=0.182, content=0.255, Δ=−0.073).

LangFlow prediction from EXP-22: overall commitment cliff at t≈0.83-0.93.
H_no_order: function/content words commit simultaneously (|Δ| < 0.03)
H_coarse_to_fine: function words commit measurably earlier (|Δ| > 0.05)

Protocol: fixed-epsilon oracle (same as EXP-22).
Uses T5 tokenizer with '▁' space marker (vs ELF's GPT-2 'Ġ' marker).

Usage (from CCLF root):
  CUDA_VISIBLE_DEVICES=X conda run -n elf python \
    experiments/probe_langflow/probe_coarsefine_langflow.py \
    --checkpoint Continuous-Rivals-Discrete/langflow-owt \
    --n_samples 64 --seq_len 128 --n_t_steps 51 \
    --entropy_thresh 1.0 --out_dir results/exp25_langflow
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_PROBE_DIR = Path(__file__).parent
_LF_SRC = _PROBE_DIR.parents[1] / "models" / "LangFlow"
for _p in [str(_LF_SRC)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_PROBE_DIR))

from probe_langflow import (
    load_langflow, encode_with_langflow, load_owt_texts,
    gamma_from_t, softmax_np,
)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Standard English function words — same base set as EXP-08 (probe_coarse_to_fine.py).
# T5 tokenizer uses '▁' (U+2581) as word-initial space marker, not 'Ġ'.
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


def is_function_word(token_id: int, tokenizer) -> bool:
    """True if the T5 token is a function word or punctuation."""
    piece = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    if piece is None:
        return False
    # Strip T5 word-initial marker '▁' (U+2581) to get surface form
    surf = piece.lstrip("▁").lower()
    return surf in FUNCTION_WORDS or piece in FUNCTION_WORDS


def collect_per_position_commitment(model, samples, t_grid, gamma_grid, entropy_thresh, sc):
    """
    For each sample × position, compute the first t where H < thresh AND top1 == gt_id.
    Returns:
        commit_tidx:  [n_samples, L] int, index into t_grid; len(t_grid) = "never committed"
        gt_tokens:    [n_samples, L] int
    """
    n_samples = len(samples)
    L = len(samples[0][0])  # gt_ids length
    n_t = len(t_grid)
    NEVER = n_t  # sentinel

    commit_tidx = np.full((n_samples, L), NEVER, dtype=np.int32)
    gt_tokens = np.zeros((n_samples, L), dtype=np.int32)

    for si, sample in enumerate(samples):
        if si % 10 == 0:
            print(f"  sample {si+1}/{n_samples}")

        gt_ids, clean_emb, attn_mask = sample
        gt_tokens[si] = gt_ids
        Lseq, d = clean_emb.shape

        rng = np.random.default_rng(si * 1000)
        eps_np = rng.standard_normal((Lseq, d)).astype(np.float32)
        eps_t = torch.from_numpy(eps_np).to(device)
        x_t = torch.from_numpy(clean_emb).to(device, dtype=torch.float32)

        already_committed = np.zeros(Lseq, dtype=bool)

        for ti, (t_val, gamma) in enumerate(zip(t_grid, gamma_grid)):
            alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma)).item())
            sigma = math.sqrt(torch.sigmoid(torch.tensor( gamma)).item())
            z_t = (alpha * x_t + sigma * eps_t)[None]  # [1, L, d]
            gamma_t = torch.full((1,), gamma, device=device, dtype=torch.float32)
            sc_in = torch.zeros_like(z_t) if sc else None

            with torch.no_grad():
                out = model(
                    noisy_embeds=z_t,
                    timesteps=gamma_t,
                    x_self_cond=sc_in,
                    return_dict=False,
                )
            logits_np = (out[0] if isinstance(out, (tuple, list)) else out)[0].cpu().float().numpy()
            p = softmax_np(logits_np, tau=1.0)       # [L, V]
            H = -(p * np.log(p + 1e-9)).sum(-1)      # [L]
            top1 = np.argmax(p, axis=-1)              # [L]

            newly_committed = (H < entropy_thresh) & (top1 == gt_ids) & ~already_committed
            if newly_committed.any():
                commit_tidx[si, newly_committed] = ti
                already_committed |= newly_committed

    return commit_tidx, gt_tokens


def analyze_coarse_fine(commit_tidx, gt_tokens, t_grid, tokenizer):
    """
    Classify each position as function/content word, compute mean commit time.
    Returns result dict.
    """
    n_t = len(t_grid)
    NEVER = n_t
    t_arr = np.array(list(t_grid) + [float("inf")])  # extended for never-committed

    n_samples, L = commit_tidx.shape
    flat_tidx = commit_tidx.flatten()                    # [N*L]
    flat_tokens = gt_tokens.flatten()                    # [N*L]

    committed_mask = flat_tidx < NEVER
    t_star = t_arr[flat_tidx]                            # [N*L], inf for never-committed

    is_func = np.array([is_function_word(tid, tokenizer) for tid in flat_tokens])

    func_committed = is_func & committed_mask
    cont_committed = (~is_func) & committed_mask

    func_t = t_star[func_committed]
    cont_t = t_star[cont_committed]

    result = {
        "n_positions": int(n_samples * L),
        "frac_never_committed": float((~committed_mask).mean()),
        "frac_function_words": float(is_func.mean()),
        "function_words": {
            "mean_t_star": float(func_t.mean()) if len(func_t) > 0 else None,
            "std_t_star": float(func_t.std()) if len(func_t) > 0 else None,
            "frac_committed": float(func_committed.mean()),
            "n": int(func_committed.sum()),
        },
        "content_words": {
            "mean_t_star": float(cont_t.mean()) if len(cont_t) > 0 else None,
            "std_t_star": float(cont_t.std()) if len(cont_t) > 0 else None,
            "frac_committed": float(cont_committed.mean()),
            "n": int(cont_committed.sum()),
        },
        "overall": {
            "mean_t_star": float(t_star[committed_mask].mean()),
            "std_t_star": float(t_star[committed_mask].std()),
        },
    }

    if result["function_words"]["mean_t_star"] is not None and \
       result["content_words"]["mean_t_star"] is not None:
        result["delta_func_minus_content"] = (
            result["function_words"]["mean_t_star"]
            - result["content_words"]["mean_t_star"]
        )

    # Commitment fraction at each t (for plotting)
    t_grid_list = list(t_grid)
    cum_func = []
    cum_cont = []
    for ti in range(n_t):
        cum_func.append(float((flat_tidx[is_func] <= ti).mean()))
        cum_cont.append(float((flat_tidx[~is_func] <= ti).mean()))
    result["commit_curve_func"] = cum_func
    result["commit_curve_cont"] = cum_cont
    result["t_grid"] = t_grid_list

    return result


def main():
    p = argparse.ArgumentParser(description="EXP-25: LangFlow coarse-to-fine hypothesis")
    p.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_steps", type=int, default=51,
                   help="Number of t values (dense around LangFlow cliff 0.65-1.00)")
    p.add_argument("--entropy_thresh", type=float, default=1.0)
    p.add_argument("--out_dir", default="results/exp25_langflow")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dense t-grid with extra density around LangFlow's commitment cliff (t=0.70-1.00)
    # 30 points uniform [0.03, 0.65], then 21 points dense [0.66, 1.00]
    t_coarse = np.linspace(0.03, 0.65, 30)
    t_dense = np.linspace(0.66, 1.00, 21)
    t_grid = np.unique(np.concatenate([t_coarse, t_dense]))
    print(f"[config] n_t_values={len(t_grid)}, range=[{t_grid[0]:.2f}, {t_grid[-1]:.2f}]")
    print(f"         cliff region t=0.66-1.00: {(t_grid >= 0.66).sum()} points")

    print(f"[load] Loading LangFlow from {args.checkpoint}")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    gamma_grid = gamma_from_t(t_grid, gamma_min, gamma_max)
    sc = model.config.self_conditioning

    print(f"[data] Loading {args.n_samples} OWT samples")
    texts = load_owt_texts(args.n_samples)
    samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)

    print(f"[EXP-25] Collecting per-position commitment (thresh={args.entropy_thresh} nats)...")
    commit_tidx, gt_tokens = collect_per_position_commitment(
        model, samples, t_grid, gamma_grid, args.entropy_thresh, sc
    )

    # Save raw data
    np.save(out_dir / "commit_tidx.npy", commit_tidx)
    np.save(out_dir / "gt_tokens.npy", gt_tokens)
    np.save(out_dir / "t_grid.npy", t_grid)
    print(f"[saved] commit_tidx.npy, gt_tokens.npy, t_grid.npy")

    print("[EXP-25] Analyzing coarse-to-fine ordering...")
    result = analyze_coarse_fine(commit_tidx, gt_tokens, t_grid, tokenizer)
    result["args"] = vars(args)

    out_path = out_dir / "coarse_fine_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")

    # Print summary
    fw = result["function_words"]
    cw = result["content_words"]
    delta = result.get("delta_func_minus_content", float("nan"))

    print("\n── EXP-25 LangFlow Coarse-to-Fine ──────────────────────────────────")
    print(f"  n_positions={result['n_positions']:,},  frac_never_committed={result['frac_never_committed']:.3f}")
    print(f"  frac_function_words={result['frac_function_words']:.3f}  (ELF EXP-08: ~0.122)")
    print()
    print(f"  Function words: mean t* = {fw['mean_t_star']:.4f}  (n={fw['n']:,})")
    print(f"  Content words:  mean t* = {cw['mean_t_star']:.4f}  (n={cw['n']:,})")
    print(f"  Δ (func - content) = {delta:+.4f}   [ELF kd_cr: −0.073, baseline: −0.154]")
    print()
    if abs(delta) < 0.03:
        print("  => H_no_order SUPPORTED: function/content commit nearly simultaneously")
    elif delta < -0.03:
        print("  => H_coarse_to_fine SUPPORTED: function words commit earlier")
    else:
        print("  => Content words commit EARLIER than function words (unexpected)")
    print()
    print(f"  Overall mean t* = {result['overall']['mean_t_star']:.4f}  [ELF kd_cr: 0.246]")
    print(f"  ELF ref: func=0.182, content=0.255, Δ=−0.073")


if __name__ == "__main__":
    main()
