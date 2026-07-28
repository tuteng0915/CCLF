"""EXP-PT3 supplement: token-discriminative direction variant B (trained
linear probe), compared against variant A (centroid direction, the one
EXP-PT3's main script implements).

u_{y,f} = (W_y - W_f) / ||W_y - W_f||, where W is a linear probe trained on
predicted_clean states from an INDEPENDENT split of sequences (never the
main probing split), at the SAME t_min used to define the default
competitor f_i elsewhere in this suite. Reuses the LinearProbe architecture
from EXP-PT9/this project's EXP-07c.

This is a LIGHTWEIGHT companion to probe_velocity_alignment.py, not a full
rerun of it: it only recomputes drift/alignment at a small subset of t
(not the full 21-point dense grid) to keep cost down, and reports variant-B
numbers alongside a quick recomputation of variant-A (centroid) at the SAME
t points for a fair, matched comparison (rather than trying to reuse
probe_velocity_alignment.py's saved npz, which doesn't persist the raw
drift/z tensors needed to reconstruct a_tok under a different direction).

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/probe_direction_supplement.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 128 --n_probe_samples 128 --n_t_points 6 --label full
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
from probe_cross_time_transfer import train_probe  # noqa: E402
from branch_around_transition import build_centroid_table  # noqa: E402 (vectorized, fast)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=128, help="main probing split")
    p.add_argument("--n_probe_samples", type=int, default=128,
                    help="independent split for both centroid table AND probe training")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_points", type=int, default=6, help="sparse t-grid (cheaper than full PT3 grid)")
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--min_centroid_count", type=int, default=3)
    p.add_argument("--n_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    print(f"[PT3-probeB] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        seq_len = adapter.seq_len
        vocab_size = adapter.vocab_size
        ids_all, mask_all = adapter.load_owt_sequences(args.n_probe_samples + args.n_samples, seq_len=seq_len)
        ind_ids, main_ids = ids_all[:args.n_probe_samples], ids_all[args.n_probe_samples:]
        ind_mask, main_mask = mask_all[:args.n_probe_samples], mask_all[args.n_probe_samples:]
        ind_emb = adapter.encode_clean(ind_ids, ind_mask).cpu()
        x_clean = adapter.encode_clean(main_ids, main_mask).cpu()
        gt_ids = main_ids
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        seq_len = args.seq_len
        vocab_size = adapter.vocab_size
        ind_ids, ind_mask, ind_emb = adapter.load_owt_sequences(args.n_probe_samples, seq_len=seq_len)
        main_ids, main_mask, x_clean = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        gt_ids = main_ids

    N, L, d = x_clean.shape
    t_grid = np.linspace(args.t_min, args.t_max, args.n_t_points)
    print(f"[PT3-probeB] independent split: {ind_ids.shape[0]}; main split: {N}, L={L}, T={len(t_grid)}")

    # Variant A ingredients (centroid table, vectorized -- fixes EXP-PT3's
    # documented slow-loop issue, see EXP-PT3-spec.md).
    centroid_table, has_centroid = build_centroid_table(ind_ids, ind_emb, args.min_centroid_count)
    max_id = centroid_table.shape[0] - 1

    # Variant B ingredient: linear probe trained on the SAME independent
    # split's predicted_clean states at t_min.
    @torch.no_grad()
    def get_predicted_clean(ids, x_clean_local, t):
        eps = adapter.sample_epsilon(x_clean_local.shape)
        z = adapter.make_oracle_state(x_clean_local.to(device), eps, float(t))
        out = adapter.forward_state(z.cpu(), None, float(t), batch_size=args.batch_size)
        return out["predicted_clean"]

    print(f"[PT3-probeB] Training variant-B probe at t_min={t_grid[0]:.3f} on independent split...")
    h_ind = get_predicted_clean(ind_ids, ind_emb, t_grid[0])
    probe = train_probe(h_ind.reshape(-1, d), ind_ids.reshape(-1), vocab_size, device, n_epochs=args.n_epochs)
    W = probe.linear.weight.detach().cpu()  # (vocab_size, d)
    del h_ind
    gc.collect()

    # Default competitor f_i at t_min (main split), matching PT1/PT2/PT3's convention.
    eps0 = adapter.sample_epsilon((N, L, d))
    z0 = adapter.make_oracle_state(x_clean.to(device), eps0, float(t_grid[0]))
    out0 = adapter.forward_state(z0.cpu(), None, float(t_grid[0]), batch_size=args.batch_size)
    f_ids = out0["logits"].argmax(-1)
    del eps0, z0, out0
    gc.collect()

    gt_clamped = gt_ids.clamp(max=max_id)
    f_clamped = f_ids.clamp(max=max_id)

    # Variant A: centroid direction (vectorized).
    u_yf_A = F.normalize(centroid_table[gt_clamped] - centroid_table[f_clamped], dim=-1)
    valid_A = has_centroid[gt_clamped] & has_centroid[f_clamped] & (gt_ids != f_ids)

    # Variant B: probe-weight direction.
    W_y = W[gt_ids.clamp(max=vocab_size - 1)]
    W_f = W[f_ids.clamp(max=vocab_size - 1)]
    diff_B = W_y - W_f
    valid_B = (diff_B.norm(dim=-1) > 1e-6) & (gt_ids != f_ids)
    u_yf_B = F.normalize(diff_B, dim=-1)

    frac_valid_A = float(valid_A.float().mean())
    frac_valid_B = float(valid_B.float().mean())
    print(f"[PT3-probeB] valid positions: variant A (centroid)={frac_valid_A*100:.1f}%  "
          f"variant B (probe)={frac_valid_B*100:.1f}%")

    both_valid = valid_A & valid_B
    cos_AB = (u_yf_A * u_yf_B).sum(-1)  # agreement between the two direction estimates
    mean_cos_AB = float(cos_AB[both_valid].mean()) if both_valid.any() else float("nan")

    records = []
    for t in t_grid:
        t = float(t)
        eps = adapter.sample_epsilon((N, L, d))
        z_t = adapter.make_oracle_state(x_clean.to(device), eps, t)
        out = adapter.forward_state(z_t.cpu(), None, t, batch_size=args.batch_size)
        pred_clean, logits = out["predicted_clean"], out["logits"]
        drift = pred_clean - z_t.cpu()

        a_tok_A = (drift * u_yf_A).sum(-1)
        a_tok_B = (drift * u_yf_B).sum(-1)

        probs = F.softmax(logits.float(), dim=-1)
        rank_raw = rank_of_gt(probs, gt_ids)

        records.append({
            "t": t,
            "mean_a_tok_A": float(a_tok_A[valid_A].mean()),
            "mean_a_tok_B": float(a_tok_B[valid_B].mean()),
            "corr_A_vs_neg_rank": float(np.corrcoef(
                a_tok_A[valid_A].numpy().astype(np.float64),
                -rank_raw[valid_A].numpy().astype(np.float64))[0, 1]) if valid_A.sum() > 1 else float("nan"),
            "corr_B_vs_neg_rank": float(np.corrcoef(
                a_tok_B[valid_B].numpy().astype(np.float64),
                -rank_raw[valid_B].numpy().astype(np.float64))[0, 1]) if valid_B.sum() > 1 else float("nan"),
        })
        print(f"  t={t:.3f}  a_tok_A={records[-1]['mean_a_tok_A']:+.4f}  "
              f"a_tok_B={records[-1]['mean_a_tok_B']:+.4f}  "
              f"corr_A={records[-1]['corr_A_vs_neg_rank']:.3f}  "
              f"corr_B={records[-1]['corr_B_vs_neg_rank']:.3f}")

        del eps, z_t, out, pred_clean, logits, drift, a_tok_A, a_tok_B, probs, rank_raw
        gc.collect()

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "n_independent_samples": ind_ids.shape[0], "seq_len": L,
        "frac_valid_variant_A_centroid": frac_valid_A,
        "frac_valid_variant_B_probe": frac_valid_B,
        "mean_cosine_A_vs_B": mean_cos_AB,
        "records": records,
        "notes": [
            "Lightweight companion to probe_velocity_alignment.py -- uses a sparser t-grid "
            "and does not persist per-position raw arrays, only aggregate curves.",
            "Uses the vectorized centroid-table construction (index_add/gather) instead of "
            "the slow per-position Python loop probe_velocity_alignment.py's main run used; "
            "see EXP-PT3-spec.md's performance note.",
        ],
    }
    json_path = out_dir / f"probe_direction_supplement_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT3-probeB] mean cosine(u_yf_A, u_yf_B) on jointly-valid positions: {mean_cos_AB:.4f}")
    print(f"[PT3-probeB] Saved {json_path}")


if __name__ == "__main__":
    main()
