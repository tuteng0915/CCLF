"""EXP-GS2: Hierarchical Branch Consensus.

From an intermediate oracle state Z_t, perturbs it K times and rolls each
perturbation forward (multi-step reverse ODE, ELFAdapter.solver_step) to
t≈0.99, then measures how much the K resulting continuations agree at the
topic / structural / lexical levels. If a global semantic basin is selected
before lexical realization is fixed, topic-consensus should rise (entropy
fall) earlier in t_start than lexical-consensus.

Cross-validates EXP-GS1's probe-based finding with a completely different
methodology (generation-branch entropy instead of linear-probe accuracy).
All design decisions and known simplifications: docs/specs/EXP-GS2-spec.md.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/branch_global_consensus.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline \\
        --topic_centroids results/global_state/elf/baseline/topic_kmeans_centroids_pilot.npy \\
        --n_docs 4 --k_branches 8 --label pilot
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

from common import (cosine_rows, decode_text, load_adapter, load_owt_docs,  # noqa: E402
                    masked_mean_pool, nearest_topic, pos_histogram)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--topic_centroids", required=True,
                    help="path to topic_kmeans_centroids_<label>.npy saved by EXP-GS1")
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_docs", type=int, default=4)
    p.add_argument("--k_branches", type=int, default=8)
    p.add_argument("--t_starts", type=float, nargs="+", default=[0.05, 0.20, 0.38, 0.65])
    p.add_argument("--t_end", type=float, default=0.99)
    p.add_argument("--eta", type=float, nargs="+", default=[0.01, 0.03, 0.1],
                    help="one or more perturbation scales; calibration found the suite doc's "
                         "suggested 1e-3 produces ~zero branch divergence at 1024-token scale "
                         "(see EXP-GS2-spec.md Section 1 calibration note)")
    p.add_argument("--full_n_steps", type=int, default=32,
                    help="ODE steps for a full 0->1 rollout; steps for a partial "
                         "t_start->t_end rollout scale proportionally to (t_end - t_start)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def entropy_from_counts(counts):
    """counts: 1D array of nonneg counts -> Shannon entropy (nats)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def rollout_branches(adapter, z_start, t_start, t_end, full_n_steps, device):
    """z_start: (B,L,d) on cpu -> (B,L,d) on cpu at t_end, via multi-step Euler ODE.
    Self-conditioning is cold-started at zeros (see EXP-GS2-spec.md Section 4.1)."""
    n_steps = max(4, round(full_n_steps * (t_end - t_start)))
    t_steps = torch.linspace(t_start, t_end, n_steps + 1).tolist()

    z = z_start.to(device)
    sc = torch.zeros_like(z)
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
    gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"[GS2] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    ids, mask, x_clean, tokenizer = load_owt_docs(adapter, args.model, args.n_docs)

    N, L, d = x_clean.shape
    K = args.k_branches
    print(f"[GS2] {N} docs x {K} branches, L={L}, d={d}")

    centroids = np.load(args.topic_centroids)
    n_clusters = centroids.shape[0]
    print(f"[GS2] loaded {n_clusters} topic centroids from {args.topic_centroids}")

    eps = adapter.sample_epsilon((N, L, d), generator=gen)  # one noise per doc

    # Same random perturbation directions reused across every (t_start, eta)
    # combination -- only the scale changes -- so eta's effect is isolated
    # from direction-sampling noise.
    u_base = torch.randn(N * K, L, d, generator=torch.Generator().manual_seed(args.seed + 1))
    u_base = u_base / u_base.reshape(N * K, -1).norm(dim=1).view(N * K, 1, 1).clamp(min=1e-12)

    records = []
    for eta in args.eta:
        eta = float(eta)
        for t_start in args.t_starts:
            t_start = float(t_start)
            z_t = adapter.make_oracle_state(x_clean.to(device), eps, t_start).cpu()

            # (N, K, L, d): K independent perturbed copies per doc
            z_rep = z_t.unsqueeze(1).expand(N, K, L, d).reshape(N * K, L, d).clone()
            z_norm = z_t.reshape(N, -1).norm(dim=1)  # (N,)
            delta_scale = (eta * z_norm).repeat_interleave(K).view(N * K, 1, 1)
            z_rep = z_rep + delta_scale * u_base

            mask_rep = mask.unsqueeze(1).expand(N, K, L).reshape(N * K, L)

            z_final, sc_final, n_steps = rollout_branches(
                adapter, z_rep, t_start, args.t_end, args.full_n_steps, device)
            print(f"  [GS2] eta={eta:g} t_start={t_start:.3f}: rolled out {n_steps} ODE steps "
                  f"(batch={N * K})")

            out = adapter.forward_state(z_final, sc_final, args.t_end, batch_size=16)
            branch_tokens = out["logits"].argmax(-1)  # (N*K, L)
            branch_tokens = branch_tokens.reshape(N, K, L)
            pooled = masked_mean_pool(z_final, mask_rep).numpy().reshape(N, K, d)

            doc_metrics = {"C_lex": [], "C_struct": [], "C_topic": [], "C_sent": []}
            for n in range(N):
                valid_pos = mask[n].bool()
                toks = branch_tokens[n][:, valid_pos].numpy()  # (K, n_valid)

                # lexical entropy: per-position categorical entropy over K branches
                pos_entropies = []
                for pos in range(toks.shape[1]):
                    _, counts = np.unique(toks[:, pos], return_counts=True)
                    pos_entropies.append(entropy_from_counts(counts))
                h_lex = float(np.mean(pos_entropies))
                c_lex = 1.0 - h_lex / np.log(K) if K > 1 else 1.0

                # structural: mean pairwise cosine sim of POS histograms
                texts = [decode_text(tokenizer, branch_tokens[n, k], mask[n]) for k in range(K)]
                pos_hists = np.stack([pos_histogram(t) for t in texts], axis=0)  # (K, 8)
                sims = []
                for i in range(K):
                    for j in range(i + 1, K):
                        a, b = pos_hists[i], pos_hists[j]
                        sims.append(float(
                            a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
                c_struct = float(np.mean(sims)) if sims else 1.0

                # topic: nearest-centroid assignment, entropy over K branches
                emb = pooled[n]  # (K, d)
                topic_ids = nearest_topic(emb, centroids)
                _, counts = np.unique(topic_ids, return_counts=True)
                h_topic = entropy_from_counts(counts)
                max_h = np.log(min(K, n_clusters))
                c_topic = 1.0 - h_topic / max_h if max_h > 0 else 1.0

                # sentence-embedding agreement: mean pairwise cosine sim of pooled embeddings
                sent_sims = []
                for i in range(K):
                    for j in range(i + 1, K):
                        sent_sims.append(float(
                            emb[i] @ emb[j]
                            / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]) + 1e-12)))
                c_sent = float(np.mean(sent_sims)) if sent_sims else 1.0

                doc_metrics["C_lex"].append(c_lex)
                doc_metrics["C_struct"].append(c_struct)
                doc_metrics["C_topic"].append(c_topic)
                doc_metrics["C_sent"].append(c_sent)

            rec = {"eta": eta, "t_start": t_start, "n_steps": n_steps,
                   **{k: float(np.mean(v)) for k, v in doc_metrics.items()},
                   **{f"{k}_per_doc": v for k, v in doc_metrics.items()}}
            records.append(rec)
            print(f"  [GS2] eta={eta:g} t_start={t_start:.3f}  C_topic={rec['C_topic']:.3f}  "
                  f"C_struct={rec['C_struct']:.3f}  C_lex={rec['C_lex']:.3f}  "
                  f"C_sent={rec['C_sent']:.3f}")

            del z_rep, z_final, sc_final, out, branch_tokens, pooled

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_docs": N, "k_branches": K, "eta_grid": args.eta, "t_end": args.t_end,
        "n_topic_clusters": n_clusters, "topic_centroids_source": args.topic_centroids,
        "records": records,
        "notes": [
            "eta swept over a small calibration grid (see --eta default); doc's suggested "
            "max 1e-2 was found (pre-pilot calibration) to produce ~zero branch divergence "
            "at 1024-token scale -- see EXP-GS2-spec.md Section 1.",
            "Cold-start self-conditioning (sc=zeros) at t_start, not a real accumulated "
            "trajectory SC state -- see EXP-GS2-spec.md Section 4.1.",
            "C_struct is mean pairwise POS-histogram cosine similarity, not a "
            "1-H/H_max entropy ratio like C_lex/C_topic -- only trends over t are "
            "comparable across metrics, not absolute values (Section 4.3).",
            "Pilot scale (n_docs=%d, K=%d) -- see EXP-GS2-spec.md before citing numbers."
            % (N, K),
        ],
    }
    json_path = out_dir / f"branch_consensus_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS2] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
