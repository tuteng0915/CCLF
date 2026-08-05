"""
EXP-12v2: Paired rank analysis on fixed baseline-wrong set.

Fixes EXP-12 selection bias: old analysis compared ranks at DIFFERENT wrong sets
(baseline 38% wrong ≠ kd_cr 11% wrong). Here we use a fixed reference set defined
by baseline failures, then compare all checkpoints on that same set.

Reference set options:
  --ref_set baseline_wrong_t T  : positions where baseline predicts wrong at t=T
  --ref_set baseline_never      : positions where baseline T_stable = never (25.1%)

For the reference set, reports per checkpoint:
  - frac_correct: how many of these positions does this checkpoint get right
  - MRR: mean reciprocal rank of true token
  - median_rank: median rank of true token
  - rank_top1/5/10: fraction with true token rank ≤ 1/5/10
  - mean_logit_gap: mean(top1_logit - true_token_logit)

Usage (from ELF-torch root):
  python experiments/probe_elf/probe_rank_analysis.py \\
    --states_root results \\
    --exp09v3_baseline_dir results/exp09v3_baseline \\
    --output_dir results/exp12v2 \\
    --ref_set baseline_never
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

T_VALUES = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]

CHECKPOINTS = {
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
    "kd_cr":    "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":      "converted/elf_b-owt-kd2_torch.pt",
}


def load_decode_weights(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = ckpt.get("params", ckpt)
    return {
        "proj_kernel":    params["proj_kernel"].float().to(device),
        "proj_bias":      params["proj_bias"].float().to(device),
        "unembed_kernel": params["unembed_kernel"].float().to(device),
        "unembed_bias":   params["unembed_bias"].float().to(device),
    }


def decode_logits(x_768, weights, batch_size=256):
    """Returns logits (M, V)."""
    pk = weights["proj_kernel"]
    pb = weights["proj_bias"]
    uk = weights["unembed_kernel"]
    ub = weights["unembed_bias"]
    parts = []
    for i in range(0, x_768.shape[0], batch_size):
        xb = x_768[i:i + batch_size].to(pk.device)
        h = F.gelu(xb @ pk + pb, approximate="tanh")
        parts.append((h @ uk + ub).cpu())
    return torch.cat(parts, dim=0)  # (M, V)


def rank_of_true_token(logits, y_true):
    """
    logits: (M, V) float
    y_true: (M,) long
    Returns rank (1-indexed) of true token at each position.
    """
    M, V = logits.shape
    true_logit = logits[torch.arange(M), y_true]  # (M,)
    # rank = number of tokens with HIGHER logit + 1
    rank = (logits > true_logit.unsqueeze(1)).sum(dim=1) + 1  # (M,)
    return rank.long()


def rank_stats(ranks, top1_logit=None, true_logit=None, label=""):
    M = len(ranks)
    correct = (ranks == 1)
    stats = {
        "n": M,
        "frac_correct": correct.float().mean().item(),
        "mrr": (1.0 / ranks.float()).mean().item(),
        "median_rank": float(ranks.float().median().item()),
        "mean_rank": ranks.float().mean().item(),
        "rank_top1":  (ranks <= 1).float().mean().item(),
        "rank_top5":  (ranks <= 5).float().mean().item(),
        "rank_top10": (ranks <= 10).float().mean().item(),
        "rank_top50": (ranks <= 50).float().mean().item(),
    }
    if top1_logit is not None and true_logit is not None:
        gap = top1_logit - true_logit
        stats["mean_logit_gap"] = gap.mean().item()
        stats["median_logit_gap"] = float(gap.median().item())
    if label:
        print(f"    [{label}] n={M} | correct={stats['frac_correct']:.4f} | "
              f"MRR={stats['mrr']:.4f} | median_rank={stats['median_rank']:.0f} | "
              f"top5={stats['rank_top5']:.4f}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_root", default="results")
    ap.add_argument("--exp16v2_dir", default="results/exp16v2",
                    help="EXP-16v2 output (timing_arrays_baseline.pt for never-commit mask)")
    ap.add_argument("--output_dir", default="results/exp12v2")
    ap.add_argument("--ref_set", default="baseline_never",
                    choices=["baseline_never", "baseline_wrong_t"],
                    help="How to define the reference (wrong) set")
    ap.add_argument("--ref_t", type=float, default=0.30,
                    help="t value for baseline_wrong_t reference set")
    ap.add_argument("--t_values", default="0.10,0.20,0.30,0.50,0.70,1.00")
    args = ap.parse_args()

    t_values = [float(x) for x in args.t_values.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-12v2] ref_set={args.ref_set}, device={device}")

    # Load decode weights for all checkpoints
    weights = {}
    for name, ckpt_rel in CHECKPOINTS.items():
        print(f"  Loading {name} weights...")
        weights[name] = load_decode_weights(ckpt_rel, device)

    # Load reference data (y_tokens, attn_mask) from exp07b_v2_baseline state file
    first_state = os.path.join(args.states_root, "exp07b_v2_baseline",
                               f"layer_states_t{t_values[0]:.3f}.pt")
    d = torch.load(first_state, map_location="cpu", weights_only=False)
    y_tok = d["y_tokens"].reshape(-1).long()
    attn_mask = d["attn_mask"].reshape(-1).bool()
    y_valid = y_tok[attn_mask]  # (M,)
    M = y_valid.shape[0]
    print(f"  Total valid positions: {M:,}")

    # Build reference set mask
    if args.ref_set == "baseline_never":
        # Use EXP-16v2 timing arrays (same position space as exp07b_v2)
        timing_path = os.path.join(args.exp16v2_dir, "timing_arrays_baseline.pt")
        if not os.path.exists(timing_path):
            raise FileNotFoundError(f"Need {timing_path} (run EXP-16v2 first)")
        ta = torch.load(timing_path, map_location="cpu", weights_only=False)
        t_stable_idx = ta["t_stable_idx"]  # (M,) — NEVER = len(t_values)
        n_t = len(t_values)
        ref_mask = (t_stable_idx >= n_t)  # never stably committed
        ref_label = f"baseline_never_T_stable (K=3, n_t={n_t})"
    else:
        # baseline_wrong_t: compute baseline predictions at ref_t and identify wrong
        state_path = os.path.join(args.states_root, "exp07b_v2_baseline",
                                  f"layer_states_t{args.ref_t:.3f}.pt")
        d = torch.load(state_path, map_location="cpu", weights_only=False)
        x_last = d["layer_feats"][-1].reshape(-1, 768)[attn_mask].float()
        logits = decode_logits(x_last, weights["baseline"])
        preds = logits.argmax(dim=-1)
        ref_mask = (preds != y_valid)
        ref_label = f"baseline_wrong_at_t={args.ref_t:.2f}"

    n_ref = ref_mask.sum().item()
    print(f"  Reference set ({ref_label}): {n_ref:,} positions ({n_ref/M*100:.1f}%)")

    y_ref = y_valid[ref_mask]  # (N_ref,)

    # For each t and checkpoint, compute rank stats on reference set
    results = {}
    for t_val in t_values:
        print(f"\n=== t={t_val:.2f} ===")
        t_results = {}
        for name in ["baseline", "kd_cr", "kd2"]:
            states_dir = os.path.join(args.states_root, f"exp07b_v2_{name}")
            path = os.path.join(states_dir, f"layer_states_t{t_val:.3f}.pt")
            if not os.path.exists(path):
                print(f"  WARNING: {path} missing, skipping {name}")
                continue
            data = torch.load(path, map_location="cpu", weights_only=False)
            x_last = data["layer_feats"][-1].reshape(-1, 768)[attn_mask].float()
            x_ref = x_last[ref_mask]  # (N_ref, 768)

            logits = decode_logits(x_ref, weights[name])  # (N_ref, V)
            ranks = rank_of_true_token(logits, y_ref)
            top1_logit = logits.max(dim=-1).values
            true_logit = logits[torch.arange(len(y_ref)), y_ref]

            t_results[name] = rank_stats(
                ranks, top1_logit=top1_logit, true_logit=true_logit, label=name)
        results[t_val] = t_results

    # Print summary table
    print("\n=== Summary: MRR on reference set ===")
    print(f"{'t':>6}  {'baseline MRR':>14}  {'kd_cr MRR':>11}  {'kd2 MRR':>9}  {'baseline frac_correct':>22}  {'kd_cr frac_correct':>19}")
    for t_val in t_values:
        row = results.get(t_val, {})
        bl = row.get("baseline", {})
        kd = row.get("kd_cr", {})
        k2 = row.get("kd2", {})
        print(f"  {t_val:.2f}  "
              f"{bl.get('mrr', float('nan')):>14.4f}  "
              f"{kd.get('mrr', float('nan')):>11.4f}  "
              f"{k2.get('mrr', float('nan')):>9.4f}  "
              f"{bl.get('frac_correct', float('nan')):>22.4f}  "
              f"{kd.get('frac_correct', float('nan')):>19.4f}")

    print("\n=== Summary: median_rank on reference set ===")
    print(f"{'t':>6}  {'baseline':>10}  {'kd_cr':>8}  {'kd2':>6}")
    for t_val in t_values:
        row = results.get(t_val, {})
        print(f"  {t_val:.2f}  "
              f"{row.get('baseline', {}).get('median_rank', float('nan')):>10.0f}  "
              f"{row.get('kd_cr', {}).get('median_rank', float('nan')):>8.0f}  "
              f"{row.get('kd2', {}).get('median_rank', float('nan')):>6.0f}")

    out = {
        "ref_set": args.ref_set,
        "ref_label": ref_label,
        "n_ref": n_ref,
        "M_total": M,
        "frac_ref": n_ref / M,
        "t_values": t_values,
        "per_t": {str(t): r for t, r in results.items()},
    }
    out_path = os.path.join(args.output_dir, "rank_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[EXP-12v2] Saved to {out_path}")


if __name__ == "__main__":
    main()
