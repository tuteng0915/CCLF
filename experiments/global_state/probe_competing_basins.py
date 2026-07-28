"""EXP-GS6: Competing Global Basins.

Samples OWT documents, topic-labels them with EXP-GS1's KMeans centroids,
pairs up documents with DIFFERENT topics, interpolates their oracle states at
t=0.28 across a lambda grid, rolls each interpolated state out to t=0.99
(reusing EXP-GS2's rollout_branches), and classifies the final output as
basin-A / basin-B / other via nearest topic centroid. See
docs/specs/EXP-GS6-spec.md for design decisions.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/probe_competing_basins.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline \\
        --topic_centroids results/global_state/elf/baseline/topic_kmeans_centroids_pilot.npy \\
        --n_docs 16 --n_pairs 4 --label pilot
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

from branch_global_consensus import rollout_branches  # noqa: E402
from common import load_adapter, load_owt_docs, masked_mean_pool, nearest_topic  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--topic_centroids", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_docs", type=int, default=16)
    p.add_argument("--n_pairs", type=int, default=4)
    p.add_argument("--t", type=float, default=0.28)
    p.add_argument("--t_end", type=float, default=0.99)
    p.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--full_n_steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()




def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[GS6] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    ids, mask, x_clean, _ = load_owt_docs(adapter, args.model, args.n_docs)

    N, L, d = x_clean.shape
    centroids = np.load(args.topic_centroids)
    clean_pooled = masked_mean_pool(x_clean, mask).numpy()
    topic_ids = nearest_topic(clean_pooled, centroids)
    print(f"[GS6] {N} docs, topic label counts: "
          f"{dict(zip(*np.unique(topic_ids, return_counts=True)))}")

    # greedily pair up docs with DIFFERENT topic labels, no doc reused
    used = set()
    pairs = []
    for i in range(N):
        if i in used or len(pairs) >= args.n_pairs:
            continue
        for j in range(i + 1, N):
            if j in used:
                continue
            if topic_ids[j] != topic_ids[i]:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
    print(f"[GS6] formed {len(pairs)} pairs with distinct topics: "
          f"{[(int(a), int(topic_ids[a]), int(b), int(topic_ids[b])) for a, b in pairs]}")

    eps = adapter.sample_epsilon((N, L, d))

    records = []
    for pair_idx, (ia, ib) in enumerate(pairs):
        z_A = adapter.make_oracle_state(x_clean[ia:ia + 1].to(device),
                                         eps[ia:ia + 1], args.t).cpu()
        z_B = adapter.make_oracle_state(x_clean[ib:ib + 1].to(device),
                                         eps[ib:ib + 1], args.t).cpu()
        topic_A, topic_B = int(topic_ids[ia]), int(topic_ids[ib])

        for lam in args.lambdas:
            z_interp = lam * z_A + (1.0 - lam) * z_B
            z_final, sc_final, n_steps = rollout_branches(
                adapter, z_interp, args.t, args.t_end, args.full_n_steps, device)
            pooled_final = masked_mean_pool(z_final, mask[ia:ia + 1]).numpy()
            final_topic = int(nearest_topic(pooled_final, centroids)[0])
            basin = "A" if final_topic == topic_A else (
                "B" if final_topic == topic_B else "other")
            records.append({"pair": pair_idx, "doc_A": int(ia), "doc_B": int(ib),
                             "topic_A": topic_A, "topic_B": topic_B, "lambda": lam,
                             "final_topic": final_topic, "basin": basin})
            print(f"  [GS6] pair={pair_idx} lambda={lam:.1f} -> basin={basin} "
                  f"(final_topic={final_topic}, A={topic_A}, B={topic_B})")

    print("\n[GS6] P_A(lambda) / P_B(lambda) / P_other(lambda) across pairs:")
    summary_by_lambda = {}
    for lam in args.lambdas:
        recs = [r for r in records if r["lambda"] == lam]
        n = len(recs)
        p_a = sum(r["basin"] == "A" for r in recs) / n
        p_b = sum(r["basin"] == "B" for r in recs) / n
        p_other = sum(r["basin"] == "other" for r in recs) / n
        summary_by_lambda[lam] = {"P_A": p_a, "P_B": p_b, "P_other": p_other, "n": n}
        print(f"  lambda={lam:.1f}: P_A={p_a:.2f} P_B={p_b:.2f} P_other={p_other:.2f} (n={n})")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_docs": N, "n_pairs": len(pairs), "t": args.t, "t_end": args.t_end,
        "lambdas": args.lambdas, "records": records,
        "summary_by_lambda": {str(k): v for k, v in summary_by_lambda.items()},
        "notes": [
            "P_A(lambda) is aggregated ACROSS different document pairs at the same "
            "lambda, not repeated branches from the same starting point -- see "
            "EXP-GS6-spec.md Section 4 point 1.",
            "Only tested at a single t=0.28, not swept across t.",
            "Pilot scale (n_pairs=%d) -- see EXP-GS6-spec.md before citing numbers."
            % len(pairs),
        ],
    }
    json_path = out_dir / f"competing_basins_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS6] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
