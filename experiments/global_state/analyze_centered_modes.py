"""EXP-GS12 (P0-2): Centered Spectral Decomposition.

Redo of EXP-GS3's low-rank/residual split with two fixes demanded by review
and confirmed necessary by EXP-GS11:
  1. Explicitly separate the trivial cross-position MEAN before doing SVD
     (EXP-GS3 did uncentered SVD, whose leading singular directions can
     trivially capture the shared mean -- exactly what the structural probe
     then reads off).
  2. Repeat the analysis on the model's own predicted_clean output, not just
     the raw oracle state (EXP-GS11 showed these behave very differently).

Three reconstructed states per (t, representation): MEAN-only, MEAN+centered
low-rank G_c, MEAN+centered residual R_c (the latter two share the SAME mean,
so their comparison isolates the effect of the extra low-rank/residual
variance, not "who kept the mean"). See docs/specs/EXP-GS12-spec.md.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_centered_modes.py \\
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

from analyze_low_rank_modes import svd_decompose  # noqa: E402
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
    p.add_argument("--t_grid", type=float, nargs="+", default=[0.05, 0.28, 0.50, 0.99])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def broadcast_mean(mu, valid_mask, L, d):
    out = np.zeros((L, d), dtype=np.float32)
    out[valid_mask] = mu[None, :]
    return out


def pad_to_full(mat_valid, valid_mask, L, d):
    out = np.zeros((L, d), dtype=mat_valid.dtype)
    out[valid_mask] = mat_valid
    return out


def pooled_summary(state, mask):
    """(mean, mean-of-squares) concatenated, (2d,) per doc. Plain mean-pooling
    is blind to any mean-zero component (e.g. a centered low-rank G_c: its own
    pooled mean is ~0 by construction) -- the second-moment term makes the
    summary sensitive to the ENERGY/variance such a component contributes,
    not just its (necessarily ~zero) mean. See EXP-GS12-spec.md Section 1 and
    the smoke-test finding that motivated this fix."""
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[GS12] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    ids, mask, x_clean, tokenizer = load_owt_docs(adapter, args.model, args.n_samples)
    gt_ids = ids

    N, L, d = x_clean.shape
    mask_np = mask.numpy().astype(bool)
    print(f"[GS12] {N} sequences, L={L}, d={d}, k={args.k}")

    print(f"[GS12] decoding + POS-tagging {N} sequences for structural target...")
    pos_hists = np.stack(
        [pos_histogram(decode_text(tokenizer, ids[i], mask[i])) for i in range(N)], axis=0)

    perm = np.random.RandomState(args.seed).permutation(N)
    n_test = int(round(args.test_frac * N))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    print(f"[GS12] train={len(train_idx)} test={len(test_idx)}")

    eps = adapter.sample_epsilon((N, L, d))

    records = []
    for t in args.t_grid:
        t = float(t)
        z_t = adapter.make_oracle_state(x_clean.to(device), eps, t).cpu()
        out_model = adapter.forward_state(z_t, None, t, batch_size=args.batch_size)
        predicted_clean = out_model["predicted_clean"]

        for repr_name, Z_full in [("raw", z_t), ("model", predicted_clean)]:
            Z_np = Z_full.numpy()
            mean_only = np.zeros((N, L, d), dtype=np.float32)
            mean_plus_G = np.zeros((N, L, d), dtype=np.float32)
            mean_plus_R = np.zeros((N, L, d), dtype=np.float32)

            for i in range(N):
                v = Z_np[i][mask_np[i]]
                mu = v.mean(0)
                v_c = v - mu
                G_c, R_c, _ = svd_decompose(v_c, args.k)
                mean_only[i] = broadcast_mean(mu, mask_np[i], L, d)
                mean_plus_G[i] = mean_only[i] + pad_to_full(G_c.astype(np.float32), mask_np[i], L, d)
                mean_plus_R[i] = mean_only[i] + pad_to_full(R_c.astype(np.float32), mask_np[i], L, d)

            conditions = {"MEAN_only": mean_only, "MEAN_plus_Gc": mean_plus_G,
                          "MEAN_plus_Rc": mean_plus_R}
            cond_metrics = {}
            for cond_name, arr in conditions.items():
                arr_t = torch.from_numpy(arr)
                pooled = pooled_summary(arr_t, mask)
                struct_r2 = eval_structural_probe(pooled[train_idx], pooled[test_idx],
                                                   pos_hists[train_idx], pos_hists[test_idx])
                out_cond = adapter.forward_state(arr_t, None, t, batch_size=args.batch_size)
                token_acc = float(((out_cond["logits"].argmax(-1) == gt_ids) & mask.bool())
                                   .sum()) / float(mask.sum())
                cond_metrics[cond_name] = {"struct_r2": struct_r2, "token_acc": token_acc}
                del arr_t, out_cond

            rec = {"t": t, "repr": repr_name, **cond_metrics}
            records.append(rec)
            print(f"  [GS12] t={t:.3f} repr={repr_name:5s}  "
                  f"MEAN_only(struct={cond_metrics['MEAN_only']['struct_r2']:.3f},"
                  f"tok={cond_metrics['MEAN_only']['token_acc']:.3f})  "
                  f"MEAN+Gc(struct={cond_metrics['MEAN_plus_Gc']['struct_r2']:.3f},"
                  f"tok={cond_metrics['MEAN_plus_Gc']['token_acc']:.3f})  "
                  f"MEAN+Rc(struct={cond_metrics['MEAN_plus_Rc']['struct_r2']:.3f},"
                  f"tok={cond_metrics['MEAN_plus_Rc']['token_acc']:.3f})")

        del z_t, out_model, predicted_clean

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "seq_len": L, "k": args.k, "t_grid": args.t_grid,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "records": records,
        "notes": [
            "Structural probe input is pooled_summary() = concat(mean_pool(state), "
            "mean_pool(state**2)), not plain mean-pooling -- smoke test found plain "
            "mean-pooling is blind to G_c by construction (a centered low-rank component "
            "has ~zero pooled mean, so MEAN_only and MEAN_plus_Gc were numerically "
            "identical under plain mean-pooling). The squared term makes the summary "
            "sensitive to the energy G_c/R_c actually contribute.",
            "MEAN_plus_Gc and MEAN_plus_Rc share the SAME mean mu -- their comparison "
            "isolates the effect of centered low-rank vs residual variance, addressing "
            "the uncentered-SVD confound raised in review of EXP-GS3.",
            "Tested on both raw oracle state and the model's own predicted_clean output "
            "(EXP-GS11 found these behave very differently for self-retrieval).",
            "MEAN_only broadcasts a single vector to all positions -- an extreme OOD input "
            "for the token probe; interpret its absolute token_acc cautiously, focus on "
            "relative comparison to MEAN_plus_Gc/Rc.",
            "Pilot scale (n_samples=%d, sparse 4-point t_grid) -- see EXP-GS12-spec.md "
            "before citing numbers." % N,
        ],
    }
    json_path = out_dir / f"centered_modes_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS12] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
