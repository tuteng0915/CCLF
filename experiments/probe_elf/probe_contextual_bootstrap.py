"""
EXP-09 (Basic Version): Contextual Bootstrapping Analysis

Uses existing exp07b layer_states files to test whether already-committed positions
help nearby uncommitted positions commit faster.

No GPU needed: reads saved layer_feats[-1] states and applies decode path.

Usage (from ELF-torch root):
  python experiments/probe_elf/probe_contextual_bootstrap.py \
    --checkpoint converted/elf_b-owt-kd_cr_torch.pt \
    --states_dir results/exp07b_kd_cr \
    --output_dir results/exp09_kd_cr
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_decode_path(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = ckpt.get("params", ckpt)

    def _to(x):
        if isinstance(x, torch.Tensor):
            return x.float().to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)

    proj_kernel = _to(params["proj_kernel"])    # (768, 512)
    proj_bias = _to(params["proj_bias"])        # (512,)
    unembed_kernel = _to(params["unembed_kernel"])  # (512, V)
    unembed_bias = _to(params["unembed_bias"])      # (V,)
    return proj_kernel, proj_bias, unembed_kernel, unembed_bias


def decode_path(hidden, proj_kernel, proj_bias, unembed_kernel, unembed_bias):
    h = F.gelu(hidden @ proj_kernel + proj_bias, approximate="tanh")
    return h @ unembed_kernel + unembed_bias


def load_states_at_t(states_dir, t_val):
    fname = os.path.join(states_dir, f"layer_states_t{t_val:.3f}.pt")
    if not os.path.exists(fname):
        return None
    d = torch.load(fname, map_location="cpu", weights_only=False)
    return d


def analyze_spatial_correlation(commit_matrix, max_dist=20):
    """
    commit_matrix: (T, B, L) bool — committed[step, seq, pos]
    Returns: for each distance d, the conditional commitment rate at next t
    given whether there's a committed neighbor within d positions.
    """
    T, B, L = commit_matrix.shape
    results = []
    for step_idx in range(T - 1):
        committed_now = commit_matrix[step_idx].numpy()       # (B, L)
        committed_next = commit_matrix[step_idx + 1].numpy()  # (B, L)

        row = {"step_from": step_idx, "step_to": step_idx + 1}
        for dist in [1, 2, 5, 10, 20]:
            # For each position, check if there's a committed neighbor within dist
            near_committed = np.zeros((B, L), dtype=bool)
            for d in range(1, dist + 1):
                near_committed[:, d:] |= committed_now[:, :-d]
                near_committed[:, :-d] |= committed_now[:, d:]

            # Uncommitted positions WITH near-committed neighbor
            mask_near = (~committed_now) & near_committed
            # Uncommitted positions WITHOUT near-committed neighbor
            mask_far = (~committed_now) & (~near_committed)

            if mask_near.sum() > 0 and mask_far.sum() > 0:
                rate_near = committed_next[mask_near].mean()
                rate_far = committed_next[mask_far].mean()
                row[f"d{dist}_near_rate"] = float(rate_near)
                row[f"d{dist}_far_rate"] = float(rate_far)
                row[f"d{dist}_near_n"] = int(mask_near.sum())
                row[f"d{dist}_far_n"] = int(mask_far.sum())
            else:
                row[f"d{dist}_near_rate"] = None
                row[f"d{dist}_far_rate"] = None

        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--states_dir", required=True)
    parser.add_argument("--output_dir", default="results/exp09")
    parser.add_argument("--max_seq", type=int, default=256,
                        help="Max sequences to use")
    parser.add_argument("--max_pos", type=int, default=256,
                        help="Max positions to use (first N positions)")
    parser.add_argument("--stable_k", type=int, default=3,
                        help="Min consecutive steps correct to count as committed (default 3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Available t values in exp07b states
    t_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    print(f"[EXP-09] Loading decode path from {args.checkpoint}")
    proj_k, proj_b, unembed_k, unembed_b = load_decode_path(args.checkpoint, device)
    print(f"[EXP-09] Vocab size: {unembed_k.shape[1]}, Hidden: {proj_k.shape[0]}")

    # Load states and compute per-position commitment at each t
    commit_list = []  # list of (B, L) bool arrays, one per t
    t_available = []
    y_tokens_ref = None

    for t in t_vals:
        d = load_states_at_t(args.states_dir, t)
        if d is None:
            print(f"  t={t:.1f}: not found, skipping")
            continue
        layer_feats = d["layer_feats"]
        y_tokens = d["y_tokens"][:args.max_seq, :args.max_pos]  # (B, L)
        hidden = layer_feats[-1][:args.max_seq, :args.max_pos]   # (B, L, 768)

        if y_tokens_ref is None:
            y_tokens_ref = y_tokens.numpy()

        with torch.no_grad():
            logits = decode_path(hidden.to(device), proj_k, proj_b, unembed_k, unembed_b)
            pred = logits.argmax(dim=-1)  # (B, L)
        correct = (pred == y_tokens.to(device)).cpu().numpy()  # (B, L) bool
        commit_list.append(correct)
        t_available.append(t)
        n_correct = correct.mean()
        print(f"  t={t:.1f}: G = {n_correct:.4f} (decode path)")

    if len(commit_list) < 2:
        print("Not enough t values with data. Exiting.")
        return

    commit_matrix = np.stack(commit_list, axis=0)  # (T, B, L)
    T, B, L = commit_matrix.shape
    print(f"\n[EXP-09] Commit matrix shape: T={T}, B={B}, L={L}")

    # Per-position commitment timing: first t where K consecutive steps are all correct.
    # K=1 is first-hit; K=3 (default) requires stable commitment.
    K = args.stable_k
    print(f"[EXP-09] Using stable_k={K} (requires {K} consecutive correct steps)")
    commit_times = np.full((B, L), len(t_available), dtype=np.int32)
    for step_idx in range(T - K + 1):
        if step_idx + K > T:
            break
        consecutive = commit_matrix[step_idx].copy()
        for k in range(1, K):
            consecutive &= commit_matrix[step_idx + k]
        stable_now = consecutive & (commit_times == len(t_available))
        commit_times[stable_now] = step_idx

    never_commit = (commit_times == len(t_available))
    print(f"  Never committed: {never_commit.mean():.4f}")
    for i, t in enumerate(t_available):
        frac = (commit_times == i).mean()
        cum = (commit_times <= i).mean()
        print(f"  First commit at t={t:.1f}: {frac:.4f} (cumulative: {cum:.4f})")

    # Spatial correlation analysis
    print(f"\n[EXP-09] Analyzing spatial correlation...")
    spatial_results = analyze_spatial_correlation(
        torch.tensor(commit_matrix), max_dist=20)

    # Print summary
    print("\nSpatial bootstrapping: near vs far commitment rates")
    print(f"{'step_from':>10} {'t_from':>7} {'t_to':>7} | d=1 near/far | d=5 near/far | d=20 near/far")
    for row in spatial_results:
        i = row["step_from"]
        tf = t_available[i] if i < len(t_available) else "?"
        tt = t_available[i + 1] if i + 1 < len(t_available) else "?"
        d1n = row.get("d1_near_rate")
        d1f = row.get("d1_far_rate")
        d5n = row.get("d5_near_rate")
        d5f = row.get("d5_far_rate")
        d20n = row.get("d20_near_rate")
        d20f = row.get("d20_far_rate")
        parts = []
        for near, far in [(d1n, d1f), (d5n, d5f), (d20n, d20f)]:
            if near is not None and far is not None:
                parts.append(f"{near:.3f}/{far:.3f}")
            else:
                parts.append("   --/--   ")
        print(f"  step {i} -> {i+1} ({tf:.1f}->{tt:.1f}): {' | '.join(parts)}")

    # Save per-position commit times matrix (for EXP-08 coarse-to-fine analysis)
    commit_times_path = os.path.join(args.output_dir, "commit_times_matrix.npy")
    np.save(commit_times_path, commit_times)
    print(f"[EXP-09] Per-position t* matrix saved to {commit_times_path}")

    # Save y_tokens reference for EXP-08 alignment
    if y_tokens_ref is not None:
        np.save(os.path.join(args.output_dir, "y_tokens_ref.npy"), y_tokens_ref)

    # Save results
    output = {
        "t_values": t_available,
        "commit_timing_hist": {
            str(i): {
                "t": float(t_available[i]),
                "frac_first_commit_here": float((commit_times == i).mean()),
                "frac_committed_cumulative": float((commit_times <= i).mean()),
            }
            for i in range(len(t_available))
        },
        "never_commit_frac": float(never_commit.mean()),
        "spatial_correlation": spatial_results,
    }
    out_path = os.path.join(args.output_dir, "contextual_bootstrap.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[EXP-09] Results saved to {out_path}")


if __name__ == "__main__":
    main()
