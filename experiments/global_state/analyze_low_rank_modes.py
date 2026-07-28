"""EXP-GS3: Low-Rank Global Mode Analysis.

Per-sequence SVD decomposition of Z_t into a rank-k "global mode" G_t^(k) and
residual R_t^(k) = Z_t - G_t^(k). Tests whether early global information is
concentrated in the shared low-rank component: does G_t^(k) support
structural-probe recovery and CKA-alignment-to-clean earlier than R_t^(k)?
Does token identity (native decode) live more in the residual?

Deliberately does NOT re-test topic/sentence-embedding recoverability here --
EXP-GS1 and EXP-GS2 independently found that mean-pooled-embedding cosine
similarity saturates near ceiling in this representation space and is not a
trustworthy signal; see EXP-GS3-spec.md Section 0. Reuses the two metrics
that DID show clean dynamic range in GS1/GS2: POS-histogram ridge probe
(structural) and native top-1 decode accuracy (token).

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_low_rank_modes.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline \\
        --n_samples 64 --label pilot
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

from common import (decode_text, load_adapter, load_owt_docs, masked_mean_pool,  # noqa: E402
                    pos_histogram)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--t_grid", type=float, nargs="+",
                    default=[0.05, 0.12, 0.20, 0.28, 0.38, 0.50, 0.65, 0.85])
    p.add_argument("--t_clean", type=float, default=0.99)
    p.add_argument("--k_values", type=int, nargs="+", default=[2, 8])
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def effective_rank(sigma):
    p = sigma / (sigma.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def svd_decompose(z_valid, k):
    """z_valid: (n_valid, d) numpy -> (G_k, R_k, sigma) all numpy."""
    U, S, Vt = np.linalg.svd(z_valid, full_matrices=False)
    k_eff = min(k, len(S))
    G_k = (U[:, :k_eff] * S[:k_eff]) @ Vt[:k_eff, :]
    R_k = z_valid - G_k
    return G_k, R_k, S


def linear_cka(X, Y):
    """X, Y: (n, d) numpy, same n. Standard linear CKA."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro") + 1e-12
    return float(num / denom)


def pad_to_full(mat_valid, valid_mask, L, d):
    out = np.zeros((L, d), dtype=mat_valid.dtype)
    out[valid_mask] = mat_valid
    return out


def eval_structural_probe(pooled_train, pooled_test, pos_train, pos_test):
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0)
    reg.fit(pooled_train, pos_train)
    pred = reg.predict(pooled_test)
    ss_res = np.sum((pos_test - pred) ** 2)
    ss_tot = np.sum((pos_test - pos_test.mean(axis=0, keepdims=True)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[GS3] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    ids, mask, x_clean, tokenizer = load_owt_docs(adapter, args.model, args.n_samples)
    gt_ids = ids

    N, L, d = x_clean.shape
    print(f"[GS3] {N} sequences, L={L}, d={d}, k_values={args.k_values}")

    print(f"[GS3] decoding + POS-tagging {N} sequences for structural target...")
    pos_hists = []
    for i in range(N):
        text = decode_text(tokenizer, ids[i], mask[i])
        pos_hists.append(pos_histogram(text))
    pos_hists = np.stack(pos_hists, axis=0)

    perm = np.random.RandomState(args.seed).permutation(N)
    n_test = int(round(args.test_frac * N))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    print(f"[GS3] train={len(train_idx)} test={len(test_idx)}")

    eps = adapter.sample_epsilon((N, L, d))

    # Per-sequence clean-state decomposition (reference for CKA), computed once.
    x_clean_np = x_clean.numpy()
    mask_np = mask.numpy().astype(bool)
    clean_decomp = {}  # k -> list of (G_clean_valid, R_clean_valid) per sequence
    for k in args.k_values:
        Gs, Rs = [], []
        for i in range(N):
            v = x_clean_np[i][mask_np[i]]
            G_k, R_k, _ = svd_decompose(v, k)
            Gs.append(G_k)
            Rs.append(R_k)
        clean_decomp[k] = (Gs, Rs)

    full_t_grid = [(t, False) for t in args.t_grid] + [(args.t_clean, True)]
    records = []
    for t, is_clean_ref in full_t_grid:
        t = float(t)
        z_t = adapter.make_oracle_state(x_clean.to(device), eps, t).cpu().numpy()

        r_eff_list = []
        per_k_metrics = {}
        for k in args.k_values:
            G_full = np.zeros((N, L, d), dtype=np.float32)
            R_full = np.zeros((N, L, d), dtype=np.float32)
            A_G_list, A_R_list = [], []
            G_clean_list, R_clean_list = clean_decomp[k]

            for i in range(N):
                v = z_t[i][mask_np[i]]
                G_k, R_k, sigma = svd_decompose(v, k)
                if k == args.k_values[0]:
                    r_eff_list.append(effective_rank(sigma))
                G_full[i] = pad_to_full(G_k.astype(np.float32), mask_np[i], L, d)
                R_full[i] = pad_to_full(R_k.astype(np.float32), mask_np[i], L, d)
                A_G_list.append(linear_cka(G_k, G_clean_list[i]))
                A_R_list.append(linear_cka(R_k, R_clean_list[i]))

            G_t_torch = torch.from_numpy(G_full)
            R_t_torch = torch.from_numpy(R_full)

            pooled_G = masked_mean_pool(G_t_torch, mask).numpy()
            pooled_R = masked_mean_pool(R_t_torch, mask).numpy()
            struct_G = eval_structural_probe(pooled_G[train_idx], pooled_G[test_idx],
                                              pos_hists[train_idx], pos_hists[test_idx])
            struct_R = eval_structural_probe(pooled_R[train_idx], pooled_R[test_idx],
                                              pos_hists[train_idx], pos_hists[test_idx])

            out_G = adapter.forward_state(G_t_torch, None, t, batch_size=args.batch_size)
            out_R = adapter.forward_state(R_t_torch, None, t, batch_size=args.batch_size)
            token_G = float(((out_G["logits"].argmax(-1) == gt_ids) & mask.bool()).sum()) \
                / float(mask.sum())
            token_R = float(((out_R["logits"].argmax(-1) == gt_ids) & mask.bool()).sum()) \
                / float(mask.sum())

            per_k_metrics[k] = {
                "A_G": float(np.mean(A_G_list)), "A_R": float(np.mean(A_R_list)),
                "G_syntax_G": struct_G, "G_syntax_R": struct_R,
                "G_token_G": token_G, "G_token_R": token_R,
            }
            del G_full, R_full, G_t_torch, R_t_torch, out_G, out_R

        rec = {"t": t, "is_clean_ref": is_clean_ref,
               "r_eff": float(np.mean(r_eff_list)), "per_k": per_k_metrics}
        records.append(rec)
        k0 = args.k_values[0]
        print(f"  [GS3] t={t:.3f}{' (clean-ref)' if is_clean_ref else ''}  "
              f"r_eff={rec['r_eff']:.1f}  [k={k0}] A_G={per_k_metrics[k0]['A_G']:.3f} "
              f"A_R={per_k_metrics[k0]['A_R']:.3f}  syntax_G={per_k_metrics[k0]['G_syntax_G']:.3f} "
              f"syntax_R={per_k_metrics[k0]['G_syntax_R']:.3f}  "
              f"token_G={per_k_metrics[k0]['G_token_G']:.3f} "
              f"token_R={per_k_metrics[k0]['G_token_R']:.3f}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "seq_len": L, "k_values": args.k_values,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "records": records,
        "notes": [
            "Topic/sentence-embedding NOT tested here -- GS1/GS2 found mean-pooled-embedding "
            "cosine similarity saturates near ceiling in this space (see EXP-GS3-spec.md "
            "Section 0). Only structural (POS ridge R^2) and token (native top-1) probes, "
            "both validated to have clean dynamic range in GS1/GS2, are reused here.",
            "G_token_G/G_token_R feed G_t^(k)/R_t^(k) directly through the trained backbone "
            "(passive diagnostic, sc=zeros) -- same convention as GS1's G_token, extended to "
            "decomposed inputs; not a causal claim (that's GLOBAL-4/EXP-GS4).",
            "SVD is per-sequence (no shared cross-sequence low-rank structure assumed) -- "
            "see EXP-GS3-spec.md Section 4 point 3.",
            "k swept over {2,8} only, not the suite doc's {1,2,4,8,16}.",
            "Pilot scale (n_samples=%d) -- see EXP-GS3-spec.md before citing numbers." % N,
        ],
    }
    json_path = out_dir / f"low_rank_modes_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS3] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
