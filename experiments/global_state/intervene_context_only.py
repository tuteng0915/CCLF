"""EXP-GS13 (P0-3): Context-Only Global-to-Local Intervention.

Fixes the confound in EXP-GS8: that experiment perturbed EVERY position,
including the one whose margin was measured, so any margin change could be
explained by the most direct path (the perturbation hit the target position's
own state) rather than any global-to-local mediation. Here the target
position's state is held EXACTLY fixed; only the OTHER (context) positions
are perturbed. If the target position's margin still moves, that requires the
perturbation to have propagated from context positions to the target position
(almost certainly via self-attention) -- a genuine causal chain claim.

Reuses EXP-GS8's topic-probe-direction construction and eta/alpha convention
exactly. See docs/specs/EXP-GS13-spec.md.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/intervene_context_only.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline \\
        --topic_centroids results/global_state/elf/baseline/topic_kmeans_centroids_pilot.npy \\
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

from common import load_adapter, load_owt_docs, masked_mean_pool  # noqa: E402
from intervene_global_to_local import build_directions, get_logp, nearest_topic  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--topic_centroids", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--n_docs_subset", type=int, default=12,
                    help="how many test docs to run the (expensive) context-only "
                         "intervention on")
    p.add_argument("--n_positions_per_doc", type=int, default=8)
    p.add_argument("--t", type=float, default=0.28)
    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 1.0])
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0,
                    help="LogisticRegression inverse regularization strength -- see the "
                         "same argument in intervene_global_to_local.py (EXP-GS8).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed + 7)

    print(f"[GS13] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    ids, mask, x_clean, _ = load_owt_docs(adapter, args.model, args.n_samples)
    gt_ids = ids

    N, L, d = x_clean.shape
    print(f"[GS13] {N} sequences, L={L}, d={d}, t={args.t}, eta={args.eta}")

    centroids = np.load(args.topic_centroids)
    clean_pooled = masked_mean_pool(x_clean, mask).numpy()
    c_true_all = nearest_topic(clean_pooled, centroids)

    perm = np.random.RandomState(args.seed).permutation(N)
    n_test = int(round(args.test_frac * N))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    print(f"[GS13] train={len(train_idx)} test={len(test_idx)}")

    eps = adapter.sample_epsilon((N, L, d))
    z_t = adapter.make_oracle_state(x_clean.to(device), eps, args.t).cpu()
    g_t = masked_mean_pool(z_t, mask).numpy()

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=args.C)
    clf.fit(g_t[train_idx], c_true_all[train_idx])
    test_acc = clf.score(g_t[test_idx], c_true_all[test_idx])
    print(f"[GS13] topic probe @ t={args.t}: test_acc={test_acc:.3f} "
          f"(chance={1.0 / centroids.shape[0]:.3f})")

    proba_test = clf.predict_proba(g_t[test_idx])
    class_to_col = {c: i for i, c in enumerate(clf.classes_)}

    z_test = z_t[test_idx]
    mask_test = mask[test_idx]
    gt_test = gt_ids[test_idx]
    log_p_before = get_logp(adapter, z_test, args.t, args.batch_size)
    f_i_all = log_p_before.argmax(-1)  # (n_test, L)
    ell_y_before = log_p_before.gather(-1, gt_test.unsqueeze(-1)).squeeze(-1)
    ell_f_before = log_p_before.gather(-1, f_i_all.unsqueeze(-1)).squeeze(-1)
    margin_before_all = ell_y_before - ell_f_before  # (n_test, L)
    doc_norms = z_test.reshape(z_test.shape[0], -1).norm(dim=1)

    # pick a subset of docs (with valid c_true/c_runnerup, matching GS8's filter)
    subset = []
    for j in range(len(test_idx)):
        c_true = c_true_all[test_idx[j]]
        if c_true not in class_to_col:
            continue
        p = proba_test[j].copy()
        p[class_to_col[c_true]] = -1.0
        c_runnerup = clf.classes_[p.argmax()]
        if c_runnerup not in class_to_col:
            continue
        subset.append((j, c_true, c_runnerup))
        if len(subset) >= args.n_docs_subset:
            break
    print(f"[GS13] using {len(subset)} test docs for context-only intervention "
          f"x {args.n_positions_per_doc} sampled positions each")

    # per-doc: fixed correct-topic direction (deterministic, from probe weights) +
    # sampled target positions. orthogonal/random directions are NOT fixed per doc --
    # they are freshly resampled for every (doc, alpha) draw below (see EXP-GS13-spec.md
    # "next steps": the pilot reused one fixed random draw across all alphas per doc,
    # which risked a single arbitrary draw dominating the "random" control's trend).
    doc_u_correct, doc_positions = {}, {}
    for j, c_true, c_runnerup in subset:
        u_correct = clf.coef_[class_to_col[c_true]] - clf.coef_[class_to_col[c_runnerup]]
        u_correct = u_correct / (np.linalg.norm(u_correct) + 1e-12)
        doc_u_correct[j] = u_correct.astype(np.float32)
        valid_positions = mask_test[j].bool().nonzero(as_tuple=True)[0].tolist()
        doc_positions[j] = rng.choice(valid_positions,
                                       size=min(args.n_positions_per_doc, len(valid_positions)),
                                       replace=False).tolist()

    records = []
    for dir_name in ["correct", "wrong", "orthogonal", "random"]:
        for alpha in args.alphas:
            batch_z, batch_meta = [], []
            for j, _, _ in subset:
                if dir_name in ("correct", "wrong"):
                    u = doc_u_correct[j] if dir_name == "correct" else -doc_u_correct[j]
                else:
                    # fresh orthogonal/random draw per (doc, alpha) -- rng continues
                    # advancing across calls, so no two draws repeat.
                    u = build_directions(doc_u_correct[j], rng)[dir_name]
                u_t = torch.from_numpy(u.astype(np.float32))
                valid = mask_test[j].bool()
                n_valid = int(valid.sum())
                for pos_i in doc_positions[j]:
                    perturb_mask = valid.clone()
                    perturb_mask[pos_i] = False  # target position untouched
                    n_pert = int(perturb_mask.sum())
                    delta_scale = (float(alpha) * args.eta * float(doc_norms[j])
                                   / (max(n_pert, 1) ** 0.5))
                    z_prime = z_test[j].clone()
                    z_prime[perturb_mask] += delta_scale * u_t
                    batch_z.append(z_prime)
                    batch_meta.append((j, pos_i))

            batch_z = torch.stack(batch_z, dim=0)
            log_p_after = get_logp(adapter, batch_z, args.t, args.batch_size)

            for b, (j, pos_i) in enumerate(batch_meta):
                ell_y_after = log_p_after[b, pos_i, gt_test[j, pos_i]]
                ell_f_after = log_p_after[b, pos_i, f_i_all[j, pos_i]]
                margin_after = float(ell_y_after - ell_f_after)
                delta_margin = margin_after - float(margin_before_all[j, pos_i])
                records.append({"doc": int(test_idx[j]), "position": int(pos_i),
                                 "direction": dir_name, "alpha": float(alpha),
                                 "delta_margin": delta_margin})
            print(f"  [GS13] direction={dir_name:10s} alpha={alpha:+.1f}  "
                  f"n={len(batch_meta)}  mean_delta_margin="
                  f"{np.mean([r['delta_margin'] for r in records if r['direction']==dir_name and r['alpha']==alpha]):+.4f}")

    def agg(direction, alpha):
        vals = [r["delta_margin"] for r in records
                if r["direction"] == direction and r["alpha"] == alpha]
        return float(np.mean(vals)) if vals else None

    print("\n[GS13] Summary (mean delta_margin, context-only intervention):")
    for direction in ["correct", "wrong", "orthogonal", "random"]:
        row = " ".join(f"a={a:+.1f}:{agg(direction, a):+.4f}" for a in args.alphas)
        print(f"  {direction:12s}: {row}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "t": args.t, "eta": args.eta, "alphas": args.alphas,
        "n_docs_subset": len(subset), "n_positions_per_doc": args.n_positions_per_doc,
        "topic_probe_test_acc": test_acc, "n_topic_clusters": centroids.shape[0],
        "records": records,
        "notes": [
            "Target position's own state is held EXACTLY fixed; perturbation only "
            "applied to the other valid positions -- addresses the direct-feature-"
            "intervention confound in EXP-GS8, see EXP-GS13-spec.md.",
            "8 target positions sampled per document (not all ~1000), n_docs_subset=%d "
            "test documents (not the full GS8 test set) -- controls compute." % len(subset),
            "Pilot scale -- see EXP-GS13-spec.md before citing numbers.",
        ],
    }
    json_path = out_dir / f"intervene_context_only_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS13] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
