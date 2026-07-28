"""EXP-PT3: Velocity Alignment and Integrated Evidence.

Tests whether the vector field provides a weak *correct* drift toward the
true token before the native decoder shows a meaningful token.

Unlike PT1/PT2/PT5, this script uses a SINGLE noise draw per t (not an
ensemble average) -- the vector field v_theta(z_t,t) is inherently a
property of one specific state z_t, not something to Jensen-average over
multiple independent noise draws the way a posterior distribution is.

Velocity/drift direction: both adapters return `predicted_clean` from
forward_state; ELF's actual flow-matching velocity is
v = (predicted_clean - z) / (1-t) (see net_out_to_v_x in sampling_utils.py).
Since every alignment metric here is either cosine similarity (scale
invariant) or feeds into a raw dot product that we treat as a *relative*
"integrated evidence" score rather than a physically calibrated integral,
we use the UNSCALED drift direction (predicted_clean - z) for BOTH ELF and
LangFlow. This keeps the two backends on a comparable footing (LangFlow's
EDM step is not a linear-time v-field in the doc's (x-z)/(1-t) sense to
begin with) at the cost of C_i(t) not being unit-consistent with actual
state displacement for ELF specifically -- documented in EXP-PT3-spec.md.

Token-discriminative direction: only variant A (centroid direction) from the
suite doc is implemented. u_{y,f} = (c_y - c_f) / ||c_y - c_f||, with
centroids estimated on an INDEPENDENT split of sequences (never the ones
used for the main probing pass), using clean (unnoised) embeddings only.
Variant B (trained linear probe direction) is not implemented -- would reuse
the EXP-07v2 probe-training pipeline; left as a follow-up.

Controls implemented: random direction, orthogonalized-random direction,
frequency-matched wrong-token direction. NOT implemented: "same-token
direction from another sequence" and "oracle vs free-running states" (the
latter needs PT7's paired-rollout infrastructure) -- flagged as gaps.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/probe_velocity_alignment.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 128 --n_centroid_samples 256 --n_t_steps 21 --label full
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from estimate_reference_prior import rank_of_gt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=64, help="sequences for the main probing pass")
    p.add_argument("--n_centroid_samples", type=int, default=128,
                    help="sequences (independent split) for token-centroid estimation")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_steps", type=int, default=21)
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--min_centroid_count", type=int, default=3,
                    help="minimum occurrences in the centroid split for a token to get a centroid")
    p.add_argument("--first10pct_n", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_token_centroids(token_ids, clean_emb, min_count):
    """token_ids: (N,L) long, clean_emb: (N,L,d) float. Returns dict
    {token_id: centroid (d,)} for tokens seen >= min_count times, and a
    {token_id: count} frequency dict for ALL tokens (used by the
    frequency-matched control)."""
    flat_ids = token_ids.reshape(-1)
    flat_emb = clean_emb.reshape(-1, clean_emb.shape[-1])
    uniq, counts = torch.unique(flat_ids, return_counts=True)
    counts_dict = {int(u): int(c) for u, c in zip(uniq, counts)}

    centroids = {}
    d = flat_emb.shape[-1]
    sums = torch.zeros(int(flat_ids.max()) + 1, d)
    cnt = torch.zeros(int(flat_ids.max()) + 1)
    sums.index_add_(0, flat_ids, flat_emb)
    cnt.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))
    for tid, c in counts_dict.items():
        if c >= min_count:
            centroids[tid] = (sums[tid] / cnt[tid])
    return centroids, counts_dict


def pick_freq_matched_wrong_token(y, f, counts_dict, centroid_ids, rng):
    """Pick a token w with similar log-frequency to f (the default
    competitor), w != y and w != f, from the set of tokens that have a
    centroid. Falls back to a random centroid token if no good match found."""
    if f not in counts_dict:
        cand = centroid_ids
    else:
        target_log = np.log(counts_dict[f] + 1)
        scored = [(abs(np.log(counts_dict.get(t, 1) + 1) - target_log), t) for t in centroid_ids]
        scored.sort(key=lambda x: x[0])
        cand = [t for _, t in scored[:20]]
    cand = [t for t in cand if t != y and t != f]
    if not cand:
        cand = [t for t in centroid_ids if t != y]
    return cand[rng.integers(0, len(cand))] if cand else f


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"[PT3] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        seq_len = adapter.seq_len
        if args.seq_len != seq_len:
            print(f"[PT3] NOTE: --seq_len ignored for ELF; using config.max_length={seq_len}")
        # independent split for centroids vs main probing: pull 2 disjoint
        # batches by requesting n_centroid_samples + n_samples and slicing.
        ids_all, mask_all = adapter.load_owt_sequences(args.n_centroid_samples + args.n_samples, seq_len=seq_len)
        cen_ids, probe_ids = ids_all[:args.n_centroid_samples], ids_all[args.n_centroid_samples:]
        cen_mask, probe_mask = mask_all[:args.n_centroid_samples], mask_all[args.n_centroid_samples:]
        cen_emb = adapter.encode_clean(cen_ids, cen_mask).cpu()
        x_clean = adapter.encode_clean(probe_ids, probe_mask).cpu()
        gt_ids = probe_ids
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        seq_len = args.seq_len
        cen_ids, cen_mask, cen_emb = adapter.load_owt_sequences(args.n_centroid_samples, seq_len=seq_len)
        probe_ids, probe_mask, x_clean = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        gt_ids = probe_ids

    N, L, d = x_clean.shape
    t_grid = np.linspace(args.t_min, args.t_max, args.n_t_steps)
    print(f"[PT3] centroid split: {cen_ids.shape[0]} seqs; probe split: {N} seqs, L={L}, T={len(t_grid)}")

    print("[PT3] Building token centroids from independent split...")
    centroids, counts_dict = build_token_centroids(cen_ids, cen_emb, args.min_centroid_count)
    centroid_ids = list(centroids.keys())
    print(f"[PT3] {len(centroid_ids)} tokens have a usable centroid (>= {args.min_centroid_count} occurrences)")
    del cen_emb
    gc.collect()

    @torch.no_grad()
    def single_forward(z, t, batch_size):
        out = adapter.forward_state(z.cpu(), None, t, batch_size=batch_size)
        return out["logits"], out["predicted_clean"]

    # Pre-pass: earliest-time native top-1 -> default competitor f (matches
    # PT1/PT2's convention, computed on a single noise draw here).
    eps0 = adapter.sample_epsilon((N, L, d))
    z0 = adapter.make_oracle_state(x_clean.to(device), eps0, float(t_grid[0]))
    logits0, _ = single_forward(z0, float(t_grid[0]), args.batch_size)
    f_ids = logits0.argmax(-1)  # (N,L)
    del eps0, z0, logits0
    gc.collect()

    # Per-position token-discriminative direction u_{y,f} and the two token
    # controls (frequency-matched wrong token w, and a fixed random unit
    # vector -- the latter also used to build the orthogonalized control).
    print("[PT3] Building per-position token-discriminative directions...")
    u_yf = torch.zeros(N, L, d)
    u_yw = torch.zeros(N, L, d)
    valid_mask = torch.zeros(N, L, dtype=torch.bool)
    gt_np, f_np = gt_ids.numpy(), f_ids.numpy()
    for n in range(N):
        for l in range(L):
            y, f = int(gt_np[n, l]), int(f_np[n, l])
            if y not in centroids or f not in centroids or y == f:
                continue
            diff = centroids[y] - centroids[f]
            norm = diff.norm()
            if norm < 1e-8:
                continue
            u_yf[n, l] = diff / norm
            w = pick_freq_matched_wrong_token(y, f, counts_dict, centroid_ids, rng)
            if w in centroids:
                diff_w = centroids[y] - centroids[w]
                norm_w = diff_w.norm()
                if norm_w > 1e-8:
                    u_yw[n, l] = diff_w / norm_w
            valid_mask[n, l] = True
    frac_valid = float(valid_mask.float().mean())
    print(f"[PT3] {frac_valid*100:.1f}% of positions have a usable u_yf direction")

    random_dir = torch.randn(N, L, d)
    random_dir = random_dir / random_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # orthogonalize against u_yf: remove the component along u_yf, renormalize
    proj = (random_dir * u_yf).sum(-1, keepdim=True)
    orth_dir = random_dir - proj * u_yf
    orth_dir = orth_dir / orth_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    records = []
    for t in t_grid:
        t = float(t)
        eps = adapter.sample_epsilon((N, L, d))
        z_t = adapter.make_oracle_state(x_clean.to(device), eps, t)
        logits, pred_clean = single_forward(z_t, t, args.batch_size)
        z_t_cpu = z_t.cpu()
        drift = pred_clean - z_t_cpu  # (N,L,d) unscaled drift direction

        target_dir = x_clean - z_t_cpu
        target_norm = target_dir.norm(dim=-1)
        drift_norm = drift.norm(dim=-1)
        a_clean = (drift * target_dir).sum(-1) / (drift_norm * target_norm).clamp_min(1e-8)

        a_tok = (drift * u_yf).sum(-1)
        a_tok_random = (drift * random_dir).sum(-1)
        a_tok_orth = (drift * orth_dir).sum(-1)
        a_tok_freqmatch = (drift * u_yw).sum(-1)

        p_probs = F.softmax(logits.float(), dim=-1)
        rank_raw = rank_of_gt(p_probs, gt_ids)
        raw_top1 = p_probs.argmax(-1)
        log_p = torch.log(p_probs + 1e-12)
        ell_gt = log_p.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)
        ell_f = log_p.gather(-1, f_ids.unsqueeze(-1)).squeeze(-1)

        records.append({
            "t": t,
            "a_clean": a_clean.numpy().astype(np.float32),
            "a_tok": a_tok.numpy().astype(np.float32),
            "a_tok_random": a_tok_random.numpy().astype(np.float32),
            "a_tok_orth": a_tok_orth.numpy().astype(np.float32),
            "a_tok_freqmatch": a_tok_freqmatch.numpy().astype(np.float32),
            "rank_raw": rank_raw.numpy().astype(np.int32),
            "raw_top1": raw_top1.numpy().astype(np.int32),
            "m_raw": (ell_gt - ell_f).numpy().astype(np.float32),
        })
        print(f"  t={t:.3f}  mean_a_clean={float(a_clean[valid_mask].mean()):+.4f}  "
              f"mean_a_tok={float(a_tok[valid_mask].mean()):+.4f}  "
              f"mean_a_tok_random={float(a_tok_random[valid_mask].mean()):+.4f}  "
              f"G_raw={float((rank_raw==0).float().mean()):.4f}")

        del eps, z_t, z_t_cpu, logits, pred_clean, drift, target_dir, p_probs, log_p
        del rank_raw, raw_top1, ell_gt, ell_f, a_clean, a_tok, a_tok_random, a_tok_orth, a_tok_freqmatch
        gc.collect()

    # Integrated evidence C_i(t_k) = sum_{j<k} Delta t_j * a_tok(t_j), per variant.
    dt = np.diff(t_grid, prepend=t_grid[0])
    dt[0] = 0.0
    for key in ["a_tok", "a_tok_random", "a_tok_orth", "a_tok_freqmatch"]:
        stacked = np.stack([r[key] for r in records])  # (T,N,L)
        cum = np.cumsum(stacked * dt[:, None, None], axis=0)
        for i, r in enumerate(records):
            r[f"C_{key}"] = cum[i]

    npz_path = out_dir / f"velocity_alignment_raw_{args.label}.npz"
    save_dict = {"t_grid": t_grid, "gt_ids": gt_ids.numpy().astype(np.int32),
                 "f_ids": f_ids.numpy().astype(np.int32), "valid_mask": valid_mask.numpy()}
    for i, rec in enumerate(records):
        for k, v in rec.items():
            save_dict[f"t{i}_{k}"] = v if isinstance(v, np.ndarray) else np.asarray(v)
    np.savez_compressed(npz_path, **save_dict)

    # Correlate C_i(t) (real vs each control) against rank_raw(t) across
    # valid positions, per t -- Pearson r between C and -rank (higher C
    # should predict lower/better rank if the decision rule holds).
    vmask = valid_mask.numpy()
    corr_by_variant = {k: [] for k in ["a_tok", "a_tok_random", "a_tok_orth", "a_tok_freqmatch"]}
    for i, r in enumerate(records):
        neg_rank = -r["rank_raw"][vmask].astype(np.float64)
        for key in corr_by_variant:
            c = r[f"C_{key}"][vmask].astype(np.float64)
            if c.std() < 1e-8 or neg_rank.std() < 1e-8:
                corr_by_variant[key].append(float("nan"))
            else:
                corr_by_variant[key].append(float(np.corrcoef(c, neg_rank)[0, 1]))

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "n_centroid_samples": cen_ids.shape[0], "seq_len": L,
        "t_grid": t_grid.tolist(), "frac_positions_with_valid_direction": frac_valid,
        "n_tokens_with_centroid": len(centroid_ids),
        "mean_a_clean_by_t": [float(r["a_clean"][vmask].mean()) for r in records],
        "mean_a_tok_by_t": {
            "real": [float(r["a_tok"][vmask].mean()) for r in records],
            "random": [float(r["a_tok_random"][vmask].mean()) for r in records],
            "orth": [float(r["a_tok_orth"][vmask].mean()) for r in records],
            "freqmatch": [float(r["a_tok_freqmatch"][vmask].mean()) for r in records],
        },
        "corr_C_vs_neg_rank_by_t": corr_by_variant,
        "raw_npz": str(npz_path),
        "notes": [
            "Only centroid direction (variant A) implemented for u_{y,f}; probe direction "
            "(variant B) not implemented -- would reuse EXP-07v2's probe-training pipeline.",
            "Controls implemented: random, orthogonalized-random, frequency-matched-wrong-token. "
            "NOT implemented: same-token-direction-from-another-sequence, oracle-vs-free-running "
            "(needs PT7 paired-rollout infrastructure).",
            "Velocity is the UNSCALED drift (predicted_clean - z), not ELF's literal "
            "(predicted_clean - z)/(1-t) v-field -- see script docstring / EXP-PT3-spec.md for why.",
            "Single noise draw per t (not an ensemble average) -- v_theta(z_t,t) is a property "
            "of one specific state, unlike the posterior-estimation use case in PT1/PT2/PT5.",
        ],
    }
    json_path = out_dir / f"velocity_alignment_summary_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT3] frac positions with valid direction: {frac_valid:.3f}")
    print(f"[PT3] mean a_clean at t_min={summary['mean_a_clean_by_t'][0]:+.4f}, "
          f"t_max={summary['mean_a_clean_by_t'][-1]:+.4f}")
    print("[PT3] corr(C_i(t), -rank_raw(t)) real vs controls at a few t:")
    for idx in [0, len(t_grid)//2, len(t_grid)-1]:
        print(f"  t={t_grid[idx]:.3f}: " + "  ".join(
            f"{k}={corr_by_variant[k][idx]:.3f}" for k in corr_by_variant))
    print(f"[PT3] Saved {json_path}")


if __name__ == "__main__":
    main()
