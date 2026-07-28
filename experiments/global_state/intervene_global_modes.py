"""EXP-GS4: Global Mode Causal Intervention.

Upgrades EXP-GS3's passive single-shot decode of G_t^(k)/R_t^(k) into a real
causal test: use each decomposed state as the STARTING POINT of a full
multi-step reverse-ODE rollout (reusing EXP-GS2's rollout_branches) so the
backbone has a chance to pull an out-of-distribution starting state back onto
a normal trajectory before we measure anything. Four conditions:
  baseline : rollout from Z_t itself (sanity check the rollout mechanism)
  A        : rollout from R_t^(k) alone (remove global mode)
  B        : rollout from G_t^(k) + eta_matched (preserve only global mode)
  C        : rollout from G_t^A + R_t^B (swap, fixed derangement i -> i+1)

All design decisions: docs/specs/EXP-GS4-spec.md.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/intervene_global_modes.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline \\
        --topic_centroids results/global_state/elf/baseline/topic_kmeans_centroids_pilot.npy \\
        --n_docs 4 --label pilot
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

from analyze_low_rank_modes import pad_to_full, svd_decompose  # noqa: E402
from branch_global_consensus import rollout_branches  # noqa: E402
from common import (decode_text, load_adapter, load_owt_docs, masked_mean_pool,  # noqa: E402
                    nearest_topic, pos_histogram)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--topic_centroids", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_docs", type=int, default=4)
    p.add_argument("--t_starts", type=float, nargs="+", default=[0.05, 0.38])
    p.add_argument("--t_end", type=float, default=0.99)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--full_n_steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate_rollout(adapter, z_final, sc_final, t_end, mask, gt_ids, tokenizer,
                      target_pos_hist, target_topic_id, target_ids, batch_size=8):
    """Returns dict(topic_match, struct_cos, token_acc) comparing z_final's
    decode against a target document's clean POS histogram / topic id / gt ids."""
    out = adapter.forward_state(z_final, sc_final, t_end, batch_size=batch_size)
    pred_ids = out["logits"].argmax(-1)
    pooled = masked_mean_pool(z_final, mask).numpy()

    results = []
    for n in range(z_final.shape[0]):
        text = decode_text(tokenizer, pred_ids[n], mask[n])
        pred_hist = pos_histogram(text)
        struct_cos = float(pred_hist @ target_pos_hist[n] / (
            np.linalg.norm(pred_hist) * np.linalg.norm(target_pos_hist[n]) + 1e-12))
        pred_topic = nearest_topic(pooled[n:n + 1], target_topic_id[1])[0]
        topic_match = bool(pred_topic == target_topic_id[0][n])
        valid = mask[n].bool()
        token_acc = float((pred_ids[n][valid] == target_ids[n][valid]).float().mean())
        results.append({"struct_cos": struct_cos, "topic_match": topic_match,
                         "token_acc": token_acc})
    return results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS4] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    ids, mask, x_clean, tokenizer = load_owt_docs(adapter, args.model, args.n_docs)

    N, L, d = x_clean.shape
    K = args.k
    print(f"[GS4] {N} docs, L={L}, d={d}, k={K}")

    centroids = np.load(args.topic_centroids)
    mask_np = mask.numpy().astype(bool)

    # Target labels for each ORIGINAL document (used as ground truth reference).
    clean_pooled = masked_mean_pool(x_clean, mask).numpy()
    doc_topic_ids = nearest_topic(clean_pooled, centroids)
    doc_pos_hists = np.stack(
        [pos_histogram(decode_text(tokenizer, ids[i], mask[i])) for i in range(N)], axis=0)

    perm = np.arange(N)
    swap_partner = (perm + 1) % N  # fixed derangement i -> i+1

    eps = adapter.sample_epsilon((N, L, d))

    records = []
    for t_start in args.t_starts:
        t_start = float(t_start)
        z_t = adapter.make_oracle_state(x_clean.to(device), eps, t_start).cpu().numpy()

        G_full = np.zeros((N, L, d), dtype=np.float32)
        R_full = np.zeros((N, L, d), dtype=np.float32)
        for i in range(N):
            v = z_t[i][mask_np[i]]
            G_k, R_k, _ = svd_decompose(v, K)
            G_full[i] = pad_to_full(G_k.astype(np.float32), mask_np[i], L, d)
            R_full[i] = pad_to_full(R_k.astype(np.float32), mask_np[i], L, d)

        # eta_matched: per-channel mean/std of each doc's own residual (valid positions only)
        eps_matched = np.zeros_like(G_full)
        rng = np.random.RandomState(args.seed)
        for i in range(N):
            r_valid = R_full[i][mask_np[i]]
            mu, sigma = r_valid.mean(0), r_valid.std(0) + 1e-8
            eps_matched[i][mask_np[i]] = rng.normal(mu, sigma, size=r_valid.shape)

        conditions = {
            "baseline": z_t.copy(),
            "A_remove_global": R_full.copy(),
            "B_preserve_global": G_full + eps_matched,
            "C_swap": G_full + R_full[swap_partner],
        }

        for cond_name, z_start_np in conditions.items():
            z_start = torch.from_numpy(z_start_np.astype(np.float32))
            z_final, sc_final, n_steps = rollout_branches(
                adapter, z_start, t_start, args.t_end, args.full_n_steps, device)

            if cond_name != "C_swap":
                target_ids_ref = ids
                target_pos = doc_pos_hists
                target_topic = (doc_topic_ids, centroids)
                res = evaluate_rollout(adapter, z_final, sc_final, args.t_end, mask,
                                        ids, tokenizer, target_pos, target_topic, target_ids_ref)
                for n in range(N):
                    records.append({"t_start": t_start, "condition": cond_name, "doc": n,
                                     "compare_to": "self", **res[n]})
            else:
                # compare each swapped sequence against BOTH its A-source (global mode
                # donor, index n) and its B-source (residual donor, swap_partner[n])
                res_A = evaluate_rollout(adapter, z_final, sc_final, args.t_end, mask,
                                          ids, tokenizer, doc_pos_hists,
                                          (doc_topic_ids, centroids), ids)
                res_B = evaluate_rollout(adapter, z_final, sc_final, args.t_end, mask,
                                          ids, tokenizer, doc_pos_hists[swap_partner],
                                          (doc_topic_ids[swap_partner], centroids),
                                          ids[swap_partner])
                for n in range(N):
                    records.append({"t_start": t_start, "condition": cond_name, "doc": n,
                                     "compare_to": "A_donor", **res_A[n]})
                    records.append({"t_start": t_start, "condition": cond_name, "doc": n,
                                     "compare_to": "B_donor", **res_B[n]})

            print(f"  [GS4] t_start={t_start:.3f} cond={cond_name}: "
                  f"{n_steps} steps done")
            del z_final, sc_final

    def agg(cond, compare_to, t_start, key):
        vals = [r[key] for r in records
                if r["condition"] == cond and r["compare_to"] == compare_to
                and r["t_start"] == t_start]
        return float(np.mean(vals)) if vals else None

    print("\n[GS4] Summary (mean across docs):")
    for t_start in args.t_starts:
        for cond in ["baseline", "A_remove_global", "B_preserve_global"]:
            print(f"  t_start={t_start:.3f} {cond:20s}: "
                  f"topic={agg(cond, 'self', t_start, 'topic_match'):.2f}  "
                  f"struct={agg(cond, 'self', t_start, 'struct_cos'):.3f}  "
                  f"token={agg(cond, 'self', t_start, 'token_acc'):.3f}")
        for compare_to in ["A_donor", "B_donor"]:
            print(f"  t_start={t_start:.3f} C_swap (vs {compare_to:8s}): "
                  f"topic={agg('C_swap', compare_to, t_start, 'topic_match'):.2f}  "
                  f"struct={agg('C_swap', compare_to, t_start, 'struct_cos'):.3f}  "
                  f"token={agg('C_swap', compare_to, t_start, 'token_acc'):.3f}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_docs": N, "k": K, "t_end": args.t_end, "t_starts": args.t_starts,
        "records": records,
        "notes": [
            "eps_matched (condition B) is independent per-channel Gaussian noise matched "
            "to R_t^(k)'s own mean/std -- no covariance structure preserved.",
            "condition C uses a fixed derangement i -> (i+1) mod N, not multiple random "
            "pairings (n_docs=4 pilot too small to support that).",
            "Pilot scale (n_docs=%d) -- see EXP-GS4-spec.md before citing numbers." % N,
        ],
    }
    json_path = out_dir / f"intervene_global_modes_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS4] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
