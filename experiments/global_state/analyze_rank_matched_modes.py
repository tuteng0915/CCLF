"""EXP-GS18 Part A: Rank- and Energy-Matched Residual Control.

GS12 compares a rank-8 low-rank component with its much larger complementary
residual and finds token identity concentrated in the residual. This does
NOT distinguish "lexical information is genuinely distributed across many
high-rank directions" from "the residual just has more dimensions/energy
than a rank-8 subspace, so of course it recovers more." This experiment
holds k FIXED and compares four equal-dimensional subspaces of the centered
per-sequence representation: the TOP-k singular directions (what GS3/GS12
call G_c), a MIDDLE-k band, the BOTTOM-k directions, and a random k-dim
subspace of the ambient space (not aligned to the data's own SVD basis at
all). See docs/specs/EXP-GS18-spec.md Part A.

Implementation notes / deviations from the spec:
  - Reports rank-matched conditions fully (same k for all four subspace
    kinds); "energy-matched" is reported only descriptively (retained_energy
    is logged for every condition so a reader can see the energy budgets
    side by side) -- a proper search for a k' that exactly matches top-k's
    retained energy for middle/bottom/random is NOT implemented.
  - Pilot n_sequences=64 (spec's formal minimum is 128), n_random_subspaces=5
    (spec asks 10), single t (spec doesn't mandate a sweep for this part).
  - "Causal removal" is a single passive forward pass with the component
    subtracted from the native oracle state (matching GS12's own single-step
    methodology), NOT a full multi-step rollout like GS4's causal tests.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_rank_matched_modes.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_samples 64 --label pilot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
_PT_DIR = _THIS_DIR.parent / "phase_transition"
for p in (_THIS_DIR, _PT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import (bootstrap_ci, decode_text, load_adapter, load_owt_docs,  # noqa: E402
                    masked_mean_pool, pos_histogram)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow", "plaid"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--t", type=float, default=0.28)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--n_random", type=int, default=5)
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pooled_summary(state, mask):
    """concat(mean, mean-of-squares) -- sensitive to zero-mean components'
    energy, not just their mean (see EXP-GS12-spec.md)."""
    mean_part = masked_mean_pool(state, mask).numpy()
    sq_part = masked_mean_pool(state ** 2, mask).numpy()
    return np.concatenate([mean_part, sq_part], axis=1)


def eval_structural_probe(pooled_train, pooled_test, pos_train, pos_test):
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0)
    reg.fit(pooled_train, pos_train)
    pred = reg.predict(pooled_test)
    ss_res = np.sum((pos_test - pred) ** 2)
    ss_tot = np.sum((pos_test - pos_test.mean(axis=0, keepdims=True)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def subspace_component(U_c, svd_cache, k, kind, rng):
    """U_c: (n_valid,d) centered. svd_cache: precomputed (U,S,Vt) for THIS
    sequence (computed ONCE outside the k/kind loop -- recomputing a full SVD
    per (k,kind) was the pilot's original bottleneck, ~8x redundant work per
    sequence since top/middle/bottom all need the same decomposition and
    'random' does not need an SVD at all).
    Returns (component (n_valid,d), retained_energy_frac)."""
    U, S, Vt = svd_cache
    r = len(S)
    total_energy = float((S ** 2).sum()) + 1e-12
    k_eff = min(k, r)
    if kind == "top":
        idx = np.arange(0, k_eff)
    elif kind == "bottom":
        idx = np.arange(r - k_eff, r)
    elif kind == "middle":
        start = max(0, (r - k_eff) // 2)
        idx = np.arange(start, start + k_eff)
    else:
        idx = None

    if idx is not None:
        comp = (U[:, idx] * S[idx]) @ Vt[idx, :]
        energy = float((S[idx] ** 2).sum() / total_energy)
    else:  # random: a k-dim ambient subspace unrelated to U_c's own SVD basis
        d = U_c.shape[1]
        Q, _ = np.linalg.qr(rng.normal(size=(d, k_eff)))
        comp = U_c @ Q @ Q.T
        energy = float(np.sum(comp ** 2) / (np.sum(U_c ** 2) + 1e-12))
    return comp.astype(np.float32), energy


def pad_to_full(mat_valid, valid_mask, L, d):
    out = np.zeros((L, d), dtype=mat_valid.dtype)
    out[valid_mask] = mat_valid
    return out


def broadcast_mean(mu, valid_mask, L, d):
    out = np.zeros((L, d), dtype=np.float32)
    out[valid_mask] = mu[None, :]
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed + 11)

    print(f"[GS18a] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    ids, mask, x_clean, tokenizer = load_owt_docs(adapter, args.model, args.n_samples)
    gt_ids = ids

    N, L, d = x_clean.shape
    mask_np = mask.numpy().astype(bool)
    print(f"[GS18a] {N} sequences, L={L}, d={d}, t={args.t}, ks={args.ks}")

    pos_hists = np.stack(
        [pos_histogram(decode_text(tokenizer, ids[i], mask[i])) for i in range(N)], axis=0)
    perm = np.random.RandomState(args.seed).permutation(N)
    n_test = int(round(args.test_frac * N))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    eps = adapter.sample_epsilon((N, L, d))
    t = float(args.t)
    z_t = adapter.make_oracle_state(x_clean.to(device), eps, t).cpu()
    out_model = adapter.forward_state(z_t, None, t, batch_size=args.batch_size)
    predicted_clean = out_model["predicted_clean"]

    # baseline (no removal) margin, per representation, for the causal-removal delta
    f_i_by_rep = {}
    margin_before_by_rep = {}
    for rep_name, Z_full in [("raw", z_t), ("model", predicted_clean)]:
        logp = torch.log_softmax(adapter.forward_state(
            Z_full, None, t, batch_size=args.batch_size)["logits"].float(), dim=-1)
        f_i = logp.argmax(-1)
        ell_y = logp.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)
        ell_f = logp.gather(-1, f_i.unsqueeze(-1)).squeeze(-1)
        f_i_by_rep[rep_name] = f_i
        margin_before_by_rep[rep_name] = (ell_y - ell_f).numpy()

    records = []
    for rep_name, Z_full in [("raw", z_t), ("model", predicted_clean)]:
        Z_np = Z_full.numpy()
        f_i = f_i_by_rep[rep_name]
        margin_before = margin_before_by_rep[rep_name]

        print(f"  [GS18a] rep={rep_name}: precomputing per-sequence SVD once "
              f"(reused across all k/top/middle/bottom conditions)...")
        mus, v_cs, svd_cache = [], [], []
        for i in range(N):
            v = Z_np[i][mask_np[i]]
            mu = v.mean(0)
            v_c = v - mu
            mus.append(mu)
            v_cs.append(v_c)
            svd_cache.append(np.linalg.svd(v_c, full_matrices=False))

        for k in args.ks:
            kinds = ["top", "middle", "bottom"] + ["random"] * args.n_random
            for kind_idx, kind in enumerate(kinds):
                recon = np.zeros((N, L, d), dtype=np.float32)
                removed = Z_np.copy()
                energies = np.zeros(N)
                for i in range(N):
                    comp, energy = subspace_component(v_cs[i], svd_cache[i], k, kind, rng)
                    energies[i] = energy
                    recon[i] = broadcast_mean(mus[i], mask_np[i], L, d) + \
                        pad_to_full(comp, mask_np[i], L, d)
                    removed[i][mask_np[i]] -= comp

                # reconstruction metrics
                recon_t = torch.from_numpy(recon)
                pooled = pooled_summary(recon_t, mask)
                struct_r2 = eval_structural_probe(pooled[train_idx], pooled[test_idx],
                                                   pos_hists[train_idx], pos_hists[test_idx])
                out_recon = adapter.forward_state(recon_t, None, t, batch_size=args.batch_size)
                token_acc = float(((out_recon["logits"].argmax(-1) == gt_ids) & mask.bool())
                                   .sum()) / float(mask.sum())
                logp_recon = torch.log_softmax(out_recon["logits"].float(), dim=-1)
                ell_y_r = logp_recon.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)
                ell_f_r = logp_recon.gather(-1, f_i.unsqueeze(-1)).squeeze(-1)
                margin_recon = (ell_y_r - ell_f_r).numpy()

                # causal removal metrics
                removed_t = torch.from_numpy(removed)
                out_removed = adapter.forward_state(removed_t, None, t, batch_size=args.batch_size)
                logp_rm = torch.log_softmax(out_removed["logits"].float(), dim=-1)
                ell_y_rm = logp_rm.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)
                ell_f_rm = logp_rm.gather(-1, f_i.unsqueeze(-1)).squeeze(-1)
                margin_after_removal = (ell_y_rm - ell_f_rm).numpy()
                delta_margin = margin_after_removal - margin_before  # (N,L)

                valid_np = mask_np
                delta_margin_seq_mean = [float(delta_margin[i][valid_np[i]].mean())
                                          for i in range(N)]
                margin_recon_seq_mean = [float(margin_recon[i][valid_np[i]].mean())
                                          for i in range(N)]
                point_dm, lo_dm, hi_dm = bootstrap_ci(delta_margin_seq_mean, n_boot=1000, seed=0)
                point_mr, lo_mr, hi_mr = bootstrap_ci(margin_recon_seq_mean, n_boot=1000, seed=0)

                rec = {
                    "rep": rep_name, "k": k,
                    "kind": kind if kind != "random" else f"random_{kind_idx - 2}",
                    "kind_group": kind, "retained_energy_mean": float(energies.mean()),
                    "struct_r2": struct_r2, "recon_token_acc": token_acc,
                    "recon_margin_mean": point_mr, "recon_margin_ci": [lo_mr, hi_mr],
                    "removal_delta_margin_mean": point_dm, "removal_delta_margin_ci": [lo_dm, hi_dm],
                }
                records.append(rec)
            print(f"  [GS18a] rep={rep_name} k={k:3d} done "
                  f"(top/middle/bottom/{args.n_random}xrandom)")

        del Z_np

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "t": t, "ks": args.ks, "n_random": args.n_random,
        "records": records,
        "notes": [
            "Rank-matched comparison only; energy-matched search (finding a k' for "
            "middle/bottom/random that exactly matches top-k's retained energy) is "
            "NOT implemented -- retained_energy_mean is reported per condition so "
            "energy budgets can be compared descriptively.",
            "'random' subspaces use a k-dim ambient random subspace (via QR of a "
            "Gaussian d x k matrix), NOT random linear combinations of the data's "
            "own top singular vectors -- unrelated to the data's SVD basis.",
            "Causal removal is a single passive forward pass with the component "
            "subtracted from the native oracle state (matches GS12's single-step "
            "methodology), not a full multi-step rollout like GS4.",
            f"Pilot scale (n_samples={N}, n_random={args.n_random}, single t={t}) "
            "-- see EXP-GS18-spec.md Part A decision rule before citing numbers.",
        ],
    }
    json_path = out_dir / f"rank_matched_modes_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS18a] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
