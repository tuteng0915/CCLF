"""EXP-GS14 (P0-4): True-Trajectory Hierarchical Branching.

Redo of EXP-GS2's branch-consensus analysis, but starting from REAL
free-running trajectory checkpoints -- both Z_t and the actually-accumulated
SC_t (self-conditioning state) -- instead of GS2's oracle-constructed Z_t with
SC cold-started at zero. Addresses the review point that GS2 measures
"oracle-state recoverability", not "basin formation during real generation".

Stage 1: genuine free-running rollout from pure noise, saving (Z_t, SC_t) at
checkpoint t's (extends EXP-GS7's rollout_with_checkpoints to also save sc,
implemented standalone here to avoid touching that already-relied-upon
function).
Stage 2: from each real (Z_t, SC_t), perturb Z_t only (K branches, same eta
convention as GS2), continue rollout with SC initialized from the REAL SC_t
(not zeros), and compute the same C_topic/C_struct/C_lex/C_sent consensus
metrics as GS2.

See docs/specs/EXP-GS14-spec.md.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/branch_true_trajectory.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline \\
        --topic_centroids results/global_state/elf/baseline/topic_kmeans_centroids_pilot.npy \\
        --n_traj 4 --label pilot
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

from branch_global_consensus import entropy_from_counts  # noqa: E402
from common import (decode_text, load_adapter, masked_mean_pool,  # noqa: E402
                    nearest_topic, pos_histogram)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--topic_centroids", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_traj", type=int, default=4)
    p.add_argument("--k_branches", type=int, default=6)
    p.add_argument("--checkpoint_ts", type=float, nargs="+", default=[0.20, 0.38, 0.65])
    p.add_argument("--t_end", type=float, default=0.99)
    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--full_n_steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def rollout_with_checkpoints_and_sc(adapter, z_start, t_start, checkpoint_ts, n_steps, device):
    """Like EXP-GS7's rollout_with_checkpoints, but also saves sc_state at
    each checkpoint (not just z). Returns dict {t: (z_cpu, sc_cpu)}."""
    t_end = max(checkpoint_ts)
    base_grid = np.linspace(t_start, t_end, n_steps + 1)
    merged = sorted(set(base_grid.tolist()) | set(checkpoint_ts))
    checkpoint_set = set(round(t, 6) for t in checkpoint_ts)

    z = z_start.to(device)
    sc = torch.zeros_like(z)
    saved = {}
    if round(merged[0], 6) in checkpoint_set:
        saved[round(merged[0], 6)] = (z.cpu(), sc.cpu())

    for i in range(len(merged) - 1):
        t, t_next = merged[i], merged[i + 1]
        z, sc = adapter.solver_step(z, sc, t, t_next)
        if round(t_next, 6) in checkpoint_set:
            saved[round(t_next, 6)] = (z.cpu(), sc.cpu())
    return saved


@torch.no_grad()
def rollout_branches_from_state(adapter, z_start, sc_start, t_start, t_end, full_n_steps, device):
    """Like EXP-GS2's rollout_branches, but sc is initialized from sc_start
    (a REAL accumulated self-conditioning state) instead of zeros."""
    n_steps = max(4, round(full_n_steps * (t_end - t_start)))
    t_steps = torch.linspace(t_start, t_end, n_steps + 1).tolist()

    z = z_start.to(device)
    sc = sc_start.to(device)
    for i in range(len(t_steps) - 1):
        t, t_next = t_steps[i], t_steps[i + 1]
        z, sc = adapter.solver_step(z, sc, t, t_next)
    return z.cpu(), sc.cpu(), n_steps


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS14] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    tokenizer = adapter.tokenizer

    N, L, d = args.n_traj, adapter.seq_len, adapter.d_model
    K = args.k_branches
    t_start = adapter.t_eps
    print(f"[GS14] Stage 1: {N} genuine free-running trajectories, L={L}, d={d}")

    centroids = np.load(args.topic_centroids)
    n_clusters = centroids.shape[0]

    eps = adapter.sample_epsilon((N, L, d))
    all_checkpoint_ts = list(args.checkpoint_ts) + [args.t_end]
    saved = rollout_with_checkpoints_and_sc(adapter, eps, t_start, all_checkpoint_ts,
                                             args.full_n_steps, device)
    print(f"[GS14] Stage 1 done: saved (Z_t, SC_t) at t={all_checkpoint_ts}")

    mask_full = torch.ones(N, L, dtype=torch.long)  # free-running: no padding

    print(f"[GS14] Stage 2: K={K} branches from each real checkpoint, "
          f"resumed with the REAL accumulated SC (not cold-started)")
    records = []
    rng_seed = args.seed + 1
    for t_start_ck in args.checkpoint_ts:
        t_key = round(t_start_ck, 6)
        z_ck, sc_ck = saved[t_key]  # (N,L,d) each -- real trajectory state

        z_rep = z_ck.unsqueeze(1).expand(N, K, L, d).reshape(N * K, L, d).clone()
        sc_rep = sc_ck.unsqueeze(1).expand(N, K, L, d).reshape(N * K, L, d).clone()

        u = torch.randn(N * K, L, d, generator=torch.Generator().manual_seed(rng_seed))
        u = u / u.reshape(N * K, -1).norm(dim=1).view(N * K, 1, 1).clamp(min=1e-12)
        z_norm = z_ck.reshape(N, -1).norm(dim=1)
        delta_scale = (args.eta * z_norm).repeat_interleave(K).view(N * K, 1, 1)
        z_rep = z_rep + delta_scale * u

        z_final, sc_final, n_steps = rollout_branches_from_state(
            adapter, z_rep, sc_rep, t_start_ck, args.t_end, args.full_n_steps, device)
        print(f"  [GS14] t_start={t_start_ck:.3f}: rolled out {n_steps} steps "
              f"(batch={N * K}, resumed from real SC)")

        out = adapter.forward_state(z_final, sc_final, args.t_end, batch_size=16)
        branch_tokens = out["logits"].argmax(-1).reshape(N, K, L)
        mask_rep = torch.ones(N * K, L, dtype=torch.long)
        pooled = masked_mean_pool(z_final, mask_rep).numpy().reshape(N, K, d)

        doc_metrics = {"C_lex": [], "C_struct": [], "C_topic": [], "C_sent": []}
        for n in range(N):
            toks = branch_tokens[n].numpy()  # (K, L), no padding

            pos_entropies = []
            for pos in range(toks.shape[1]):
                _, counts = np.unique(toks[:, pos], return_counts=True)
                pos_entropies.append(entropy_from_counts(counts))
            h_lex = float(np.mean(pos_entropies))
            c_lex = 1.0 - h_lex / np.log(K) if K > 1 else 1.0

            texts = [decode_text(tokenizer, branch_tokens[n, k], mask_full[0]) for k in range(K)]
            pos_hists = np.stack([pos_histogram(t) for t in texts], axis=0)
            sims = []
            for i in range(K):
                for j in range(i + 1, K):
                    a, b = pos_hists[i], pos_hists[j]
                    sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            c_struct = float(np.mean(sims)) if sims else 1.0

            emb = pooled[n]
            topic_ids = nearest_topic(emb, centroids)
            _, counts = np.unique(topic_ids, return_counts=True)
            h_topic = entropy_from_counts(counts)
            max_h = np.log(min(K, n_clusters))
            c_topic = 1.0 - h_topic / max_h if max_h > 0 else 1.0

            sent_sims = []
            for i in range(K):
                for j in range(i + 1, K):
                    sent_sims.append(float(
                        emb[i] @ emb[j] / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]) + 1e-12)))
            c_sent = float(np.mean(sent_sims)) if sent_sims else 1.0

            doc_metrics["C_lex"].append(c_lex)
            doc_metrics["C_struct"].append(c_struct)
            doc_metrics["C_topic"].append(c_topic)
            doc_metrics["C_sent"].append(c_sent)

        rec = {"t_start": t_start_ck, "n_steps": n_steps,
               **{k: float(np.mean(v)) for k, v in doc_metrics.items()},
               **{f"{k}_per_doc": v for k, v in doc_metrics.items()}}
        records.append(rec)
        print(f"  [GS14] t_start={t_start_ck:.3f}  C_topic={rec['C_topic']:.3f}  "
              f"C_struct={rec['C_struct']:.3f}  C_lex={rec['C_lex']:.3f}  "
              f"C_sent={rec['C_sent']:.3f}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_traj": N, "k_branches": K, "eta": args.eta, "t_end": args.t_end,
        "checkpoint_ts": args.checkpoint_ts, "records": records,
        "notes": [
            "Branch starting points are REAL free-running trajectory checkpoints "
            "(Z_t, SC_t both saved from genuine generation), not oracle-constructed "
            "states with cold-started SC as in EXP-GS2.",
            "Branches resume with the REAL accumulated SC_t, not zeros.",
            "Compare directly against EXP-GS2's eta=0.03 results at the same "
            "checkpoint t's to see whether the cold-start-oracle simplification "
            "changed the qualitative consensus pattern.",
            "Pilot scale (n_traj=%d, K=%d) -- see EXP-GS14-spec.md before citing "
            "numbers." % (N, K),
        ],
    }
    json_path = out_dir / f"branch_true_trajectory_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS14] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
