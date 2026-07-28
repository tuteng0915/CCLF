"""EXP-GS1: Sequence-Level Probe Hierarchy.

Tests H1 (global-before-local): does a probe trained on the mean-pooled full
state g_t = mean_i(z_{i,t}) recover topic / sentence-embedding / POS-syntax
before a probe (here: the model's own native decode) recovers exact token
identity?

Target choices and all deviations from docs/global_state_formation_experiment_suite.md
Section 5 are documented in docs/specs/EXP-GS1-spec.md.

Noise protocol: ONE epsilon per sequence, reused across the whole t-grid
(a single oracle path per sequence), per suite-doc Section 4 ("同一 sequence
在所有 t 使用同一个 epsilon") -- NOT resampled per t like EXP-PT2.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/probe_sequence_hierarchy.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp36_baseline.yml \\
        --out_dir results/global_state/elf/baseline \\
        --n_samples 240 --label pilot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
_PT_DIR = _THIS_DIR.parent / "phase_transition"
for p in (_THIS_DIR, _PT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import (cosine_rows, load_adapter, load_owt_docs, masked_mean_pool,  # noqa: E402
                    pos_histogram)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=240)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--t_grid", type=float, nargs="+",
                    default=[0.05, 0.12, 0.20, 0.28, 0.38, 0.50, 0.65, 0.85])
    p.add_argument("--t_clean", type=float, default=0.99)
    p.add_argument("--n_topic_clusters", type=int, default=8)
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_labels(x_clean, mask, ids, tokenizer, n_clusters, seed):
    """Document-level targets computed once from clean data only."""
    from sklearn.cluster import KMeans

    clean_pooled = masked_mean_pool(x_clean, mask).numpy()

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    topic_labels = km.fit_predict(clean_pooled)

    print(f"[GS1] decoding + POS-tagging {ids.shape[0]} sequences for structural target...")
    pos_hists = []
    for i in range(ids.shape[0]):
        valid = ids[i][mask[i].bool()]
        text = tokenizer.decode(valid.tolist(), skip_special_tokens=True)
        pos_hists.append(pos_histogram(text))
        if (i + 1) % 50 == 0:
            print(f"  [GS1] POS-tagged {i + 1}/{ids.shape[0]}")
    pos_hists = np.stack(pos_hists, axis=0)

    return {
        "clean_pooled": clean_pooled,
        "topic_labels": topic_labels,
        "pos_hists": pos_hists,
        "topic_centroids": km.cluster_centers_,
    }


def eval_probes(g_train, g_test, labels, train_idx, test_idx, n_clusters, seed):
    from sklearn.linear_model import LogisticRegression, Ridge

    out = {}

    # --- G_topic: multinomial logistic regression -> cluster accuracy
    y_train = labels["topic_labels"][train_idx]
    y_test = labels["topic_labels"][test_idx]
    if len(set(y_train.tolist())) < 2:
        out["G_topic"] = float("nan")
    else:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(g_train, y_train)
        out["G_topic"] = float(clf.score(g_test, y_test))

    # --- G_sent: ridge regression -> clean pooled embedding, mean cosine sim
    y_train = labels["clean_pooled"][train_idx]
    y_test = labels["clean_pooled"][test_idx]
    reg = Ridge(alpha=1.0)
    reg.fit(g_train, y_train)
    pred = reg.predict(g_test)
    out["G_sent"] = float(np.mean(cosine_rows(pred, y_test)))

    # --- G_syntax: ridge regression -> POS histogram, mean cosine sim + R^2
    y_train = labels["pos_hists"][train_idx]
    y_test = labels["pos_hists"][test_idx]
    reg = Ridge(alpha=1.0)
    reg.fit(g_train, y_train)
    pred = reg.predict(g_test)
    ss_res = np.sum((y_test - pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2) + 1e-12
    out["G_syntax_r2"] = float(1.0 - ss_res / ss_tot)
    out["G_syntax_cos"] = float(np.mean(cosine_rows(pred, y_test)))

    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[GS1] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    if args.model == "elf" and args.seq_len != adapter.seq_len:
        print(f"[GS1] NOTE: --seq_len ignored for ELF; using config.max_length={adapter.seq_len}")
    ids, mask, x_clean, tokenizer = load_owt_docs(adapter, args.model, args.n_samples,
                                                   seq_len=args.seq_len)
    gt_ids = ids

    N, L, d = x_clean.shape
    print(f"[GS1] {N} sequences, L={L}, d={d}")

    labels = build_labels(x_clean, mask, ids, tokenizer, args.n_topic_clusters, args.seed)

    perm = np.random.RandomState(args.seed).permutation(N)
    n_test = int(round(args.test_frac * N))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    print(f"[GS1] train={len(train_idx)} test={len(test_idx)} "
          f"(document-level split, fixed across all t)")

    # single fixed noise per sequence, reused across the whole t-grid
    eps = adapter.sample_epsilon((N, L, d))

    records = []
    full_t_grid = list(args.t_grid) + [args.t_clean]
    for t in full_t_grid:
        t = float(t)
        is_clean_ref = abs(t - args.t_clean) < 1e-9
        z_t = adapter.make_oracle_state(x_clean.to(device), eps, t).cpu()
        g_t = masked_mean_pool(z_t, mask).numpy()

        out = adapter.forward_state(z_t, None, t, batch_size=args.batch_size)
        native_pred = out["logits"].argmax(-1)
        correct = (native_pred == gt_ids) & mask.bool()
        g_token = float(correct.sum()) / float(mask.sum())

        probe_out = eval_probes(g_t[train_idx], g_t[test_idx], labels,
                                 train_idx, test_idx, args.n_topic_clusters, args.seed)
        probe_out["t"] = t
        probe_out["is_clean_ref"] = is_clean_ref
        probe_out["G_token"] = g_token
        records.append(probe_out)
        print(f"  [GS1] t={t:.3f}{' (clean-ref)' if is_clean_ref else ''}  "
              f"G_topic={probe_out['G_topic']:.3f}  G_sent={probe_out['G_sent']:.3f}  "
              f"G_syntax_r2={probe_out['G_syntax_r2']:.3f}  G_token={probe_out['G_token']:.3f}")
        del z_t, out, native_pred, correct

    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "label": args.label,
        "n_samples": N,
        "seq_len": L,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_topic_clusters": args.n_topic_clusters,
        "t_grid": list(args.t_grid),
        "t_clean": args.t_clean,
        "records": records,
        "notes": [
            "topic labels = KMeans(k=%d) on clean pooled embeddings, not external annotation."
            % args.n_topic_clusters,
            "POS histogram is document-level (8 coarse buckets), no per-position T5-subword "
            "<-> word alignment.",
            "G_token is native top-1 decode accuracy (model's own nonlinear decode), NOT a "
            "linear probe on g_t -- not directly capacity-matched to G_topic/G_sent/G_syntax. "
            "See EXP-GS1-spec.md Section 4 point 3 for the interpretation caveat.",
            "Pilot scale -- see EXP-GS1-spec.md for known simplifications before citing numbers.",
        ],
    }
    json_path = out_dir / f"probe_hierarchy_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS1] Saved summary to {json_path}")

    centroids_path = out_dir / f"topic_kmeans_centroids_{args.label}.npy"
    np.save(centroids_path, labels["topic_centroids"])
    print(f"[GS1] Saved topic KMeans centroids ({labels['topic_centroids'].shape}) "
          f"to {centroids_path} (for reuse by GLOBAL-2/GLOBAL-6 topic-cluster assignment)")


if __name__ == "__main__":
    main()
