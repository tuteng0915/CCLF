"""EXP-GS8: Global-to-Local Causal Chain.

At a t where the topic probe is already meaningfully above chance but native
token decode is still weak (t=0.28 in the GS1 pilot: G_topic=0.500,
G_token=0.352), perturbs Z_t along a "topic direction" derived from a linear
topic probe's own weights (correct-vs-runnerup class), and measures whether
the true-token margin at EVERY position moves accordingly -- without any
further rollout. Four direction conditions: correct, wrong (negated correct),
orthogonal control, random same-norm control.

All design decisions: docs/specs/EXP-GS8-spec.md.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/intervene_global_to_local.py \\
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

from common import load_adapter, load_owt_docs, masked_mean_pool, nearest_topic  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--topic_centroids", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--t", type=float, default=0.28)
    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--alphas", type=float, nargs="+", default=[-1.0, -0.5, 0.5, 1.0])
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0,
                    help="LogisticRegression inverse regularization strength. Default pilot "
                         "value (1.0, sklearn default) badly overfits an 8-way probe in d=768 "
                         "with only ~45 training docs (train_acc=0.933 vs test_acc=0.158, see "
                         "EXP-GS8-spec.md rigor audit); use a smaller value (e.g. 0.01-0.1) "
                         "together with a larger --n_samples when fitting for real.")
    return p.parse_args()



def build_directions(u_correct, rng):
    """u_correct: (d,) unit vector -> dict of 4 (d,) unit-vector directions."""
    d = u_correct.shape[0]
    r = rng.normal(size=d)
    r_orth = r - (r @ u_correct) * u_correct
    r_orth = r_orth / (np.linalg.norm(r_orth) + 1e-12)
    r_rand = rng.normal(size=d)
    r_rand = r_rand / (np.linalg.norm(r_rand) + 1e-12)
    return {"correct": u_correct, "wrong": -u_correct, "orthogonal": r_orth, "random": r_rand}


@torch.no_grad()
def get_logp(adapter, z, t, batch_size):
    out = adapter.forward_state(z, None, t, batch_size=batch_size)
    log_p = torch.log_softmax(out["logits"].float(), dim=-1)
    return log_p


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed + 7)

    print(f"[GS8] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    ids, mask, x_clean, _ = load_owt_docs(adapter, args.model, args.n_samples)
    gt_ids = ids

    N, L, d = x_clean.shape
    print(f"[GS8] {N} sequences, L={L}, d={d}, t={args.t}, eta={args.eta}")

    centroids = np.load(args.topic_centroids)
    clean_pooled = masked_mean_pool(x_clean, mask).numpy()
    c_true_all = nearest_topic(clean_pooled, centroids)

    perm = np.random.RandomState(args.seed).permutation(N)
    n_test = int(round(args.test_frac * N))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    print(f"[GS8] train={len(train_idx)} test={len(test_idx)}")

    eps = adapter.sample_epsilon((N, L, d))
    z_t = adapter.make_oracle_state(x_clean.to(device), eps, args.t).cpu()
    g_t = masked_mean_pool(z_t, mask).numpy()

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=args.C)
    clf.fit(g_t[train_idx], c_true_all[train_idx])
    test_acc = clf.score(g_t[test_idx], c_true_all[test_idx])
    print(f"[GS8] topic probe @ t={args.t}: train_acc={clf.score(g_t[train_idx], c_true_all[train_idx]):.3f} "
          f"test_acc={test_acc:.3f} (chance={1.0 / centroids.shape[0]:.3f})")

    proba_test = clf.predict_proba(g_t[test_idx])
    class_to_col = {c: i for i, c in enumerate(clf.classes_)}

    # pre-intervention logits / default competitor f_i for the whole test set
    z_test = z_t[test_idx]
    mask_test = mask[test_idx]
    gt_test = gt_ids[test_idx]
    log_p_before = get_logp(adapter, z_test, args.t, args.batch_size)
    f_i = log_p_before.argmax(-1)  # (n_test, L) default competitor = pre-intervention top-1
    ell_y_before = log_p_before.gather(-1, gt_test.unsqueeze(-1)).squeeze(-1)
    ell_f_before = log_p_before.gather(-1, f_i.unsqueeze(-1)).squeeze(-1)
    margin_before = ell_y_before - ell_f_before  # (n_test, L)

    doc_norms = z_test.reshape(z_test.shape[0], -1).norm(dim=1)  # (n_test,)

    records = []
    for j, doc_i in enumerate(test_idx):
        c_true = c_true_all[doc_i]
        if c_true not in class_to_col:
            continue  # probe never saw this class in training, skip (rare at pilot scale)
        p = proba_test[j].copy()
        p[class_to_col[c_true]] = -1.0
        c_runnerup = clf.classes_[p.argmax()]
        if c_runnerup not in class_to_col:
            continue
        u_correct = clf.coef_[class_to_col[c_true]] - clf.coef_[class_to_col[c_runnerup]]
        u_correct = u_correct / (np.linalg.norm(u_correct) + 1e-12)
        directions = build_directions(u_correct.astype(np.float32), rng)

        valid = mask_test[j].bool()
        for dir_name, u in directions.items():
            u_t = torch.from_numpy(u.astype(np.float32))
            for alpha in args.alphas:
                # u_t is a unit d-vector broadcast identically to all n_valid positions;
                # dividing by sqrt(n_valid) keeps the resulting (L,d) delta's Frobenius
                # norm equal to alpha*eta*||Z_t||_F, matching EXP-GS2's per-sequence
                # Frobenius-norm perturbation convention (see EXP-GS8-spec.md Section 0) --
                # without this, broadcasting the same vector to ~1000 positions inflates
                # the effective perturbation norm by sqrt(n_valid) (~32x at L=1024).
                n_valid_j = int(valid.sum())
                delta = (float(alpha) * args.eta * float(doc_norms[j])
                         / (n_valid_j ** 0.5)) * u_t  # (d,)
                z_prime = z_test[j:j + 1].clone()
                z_prime[0] += delta.unsqueeze(0)  # broadcast across all L positions

                log_p_after = get_logp(adapter, z_prime, args.t, args.batch_size)[0]
                ell_y_after = log_p_after.gather(-1, gt_test[j].unsqueeze(-1)).squeeze(-1)
                ell_f_after = log_p_after.gather(-1, f_i[j].unsqueeze(-1)).squeeze(-1)
                margin_after = ell_y_after - ell_f_after

                delta_margin = (margin_after - margin_before[j])[valid]
                records.append({
                    "doc": int(doc_i), "direction": dir_name, "alpha": float(alpha),
                    "delta_margin_mean": float(delta_margin.mean()),
                    "delta_margin_pos_frac": float((delta_margin > 0).float().mean()),
                })
        if (j + 1) % 5 == 0:
            print(f"  [GS8] processed {j + 1}/{len(test_idx)} test docs")

    def agg(direction, alpha, key):
        vals = [r[key] for r in records if r["direction"] == direction and r["alpha"] == alpha]
        return float(np.mean(vals)) if vals else None

    print("\n[GS8] Summary (mean delta_margin across test docs):")
    for direction in ["correct", "wrong", "orthogonal", "random"]:
        row = " ".join(f"a={a:+.2f}:{agg(direction, a, 'delta_margin_mean'):+.4f}"
                        for a in args.alphas)
        print(f"  {direction:12s}: {row}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "t": args.t, "eta": args.eta, "alphas": args.alphas,
        "topic_probe_test_acc": test_acc, "n_topic_clusters": centroids.shape[0],
        "records": records,
        "notes": [
            "Direction 'correct'/'wrong' derived from a freshly-fit topic logistic-regression "
            "probe's own weights (c_true vs probe's runner-up class), not an oracle/external "
            "topic direction -- see EXP-GS8-spec.md Section 2.",
            "f_i (default competitor) = pre-intervention native top-1 at each position, "
            "held fixed across all interventions for a given document.",
            "No rollout -- single-step margin comparison at the same t only.",
            "Pilot scale (n_samples=%d) -- see EXP-GS8-spec.md before citing numbers." % N,
        ],
    }
    json_path = out_dir / f"intervene_global_to_local_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS8] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
