"""EXP-PT9 supplement: cross-time transfer for the two remaining state
representations the suite doc lists (raw states z_t, native hidden states),
beyond the predicted_clean representation the main
probe_cross_time_transfer.py implements.

"Prior-subtracted logits" (the doc's 4th representation) is NOT implemented
here: a residual logit vector already lives in vocab-size space (32k-50k
dims), so training a genuinely new linear probe ON TOP of it would mean a
Linear(vocab_size, vocab_size) layer (~1B+ parameters) -- not tractable at
this sample size, and conceptually redundant (the residual logit already IS
a per-class score). This representation doesn't fit the "train a probe on a
d-dim hidden representation" paradigm the other three do; left as a
documented gap (see EXP-PT9-spec.md).

Native hidden state: last transformer block's output (matches this
project's EXP-07b finding that L10/L11 have the highest per-layer probe
accuracy for ELF; uses the analogous last-block output for LangFlow).

Same methodology as the main script: LinearProbe architecture, sequence-
level train/val split, transfer matrix M[a,b] = held-out accuracy of the
probe trained at t_a, evaluated at t_b.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/probe_cross_time_transfer_extra_reps.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 128 --n_t_steps 7 --label full
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from probe_cross_time_transfer import train_probe, eval_probe  # noqa: E402

REPRESENTATIONS = ["raw_z", "hidden"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_steps", type=int, default=7)
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--val_frac", type=float, default=0.3)
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

    print(f"[PT9-extra] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        seq_len = adapter.seq_len
        ids, mask = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        x_clean = adapter.encode_clean(ids, mask).cpu()
        gt_ids = ids
        vocab_size = adapter.vocab_size
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        seq_len = args.seq_len
        ids, mask, x_clean = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        gt_ids = ids
        vocab_size = adapter.vocab_size

    N, L, d = x_clean.shape
    t_grid = np.linspace(args.t_min, args.t_max, args.n_t_steps)
    print(f"[PT9-extra] {N} sequences, L={L}, T={len(t_grid)} t-points, vocab={vocab_size}")

    rng = np.random.default_rng(args.seed)
    seq_order = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_seqs, train_seqs = seq_order[:n_val], seq_order[n_val:]
    print(f"[PT9-extra] {len(train_seqs)} train sequences, {len(val_seqs)} val sequences")

    @torch.no_grad()
    def get_states(t, need_hidden):
        eps = adapter.sample_epsilon((N, L, d))
        z = adapter.make_oracle_state(x_clean.to(device), eps, float(t))
        out = adapter.forward_state(z.cpu(), None, float(t), batch_size=args.batch_size,
                                     capture_hidden=need_hidden) if need_hidden else \
            adapter.forward_state(z.cpu(), None, float(t), batch_size=args.batch_size)
        hidden_last = None
        if need_hidden:
            hidden_last = out["hidden_states"][-1]
            # ELF prepends mode/time/context prefix tokens before running the
            # transformer blocks and only slices them off right before the
            # final decode layer (models/ELF-torch/src/modules/model.py) --
            # the raw per-block hook captures we grab in forward_state still
            # include that prefix, so hidden_last.shape[1] > L. LangFlow has
            # no such prefix (hidden_last.shape[1] == L already). Without
            # this slice, X (hidden) and Y (gt_ids, length L) silently have
            # different row counts once flattened, which crashes CUDA with
            # an out-of-bounds index inside cross_entropy (found by running
            # this script and hitting exactly that crash).
            prefix_offset = hidden_last.shape[1] - L
            if prefix_offset > 0:
                hidden_last = hidden_last[:, prefix_offset:, :]
        return z.cpu(), hidden_last

    results = {}
    for rep in REPRESENTATIONS:
        need_hidden = (rep == "hidden")
        print(f"\n[PT9-extra] === representation: {rep} ===")
        train_X, train_Y, val_X, val_Y = {}, {}, {}, {}
        rep_dim = None
        for t in t_grid:
            z_t, hidden = get_states(t, need_hidden)
            h = hidden if rep == "hidden" else z_t
            rep_dim = h.shape[-1]
            h_train = h[train_seqs].reshape(-1, rep_dim)
            y_train = gt_ids[train_seqs].reshape(-1)
            h_val = h[val_seqs].reshape(-1, rep_dim)
            y_val = gt_ids[val_seqs].reshape(-1)
            train_X[float(t)], train_Y[float(t)] = h_train, y_train
            val_X[float(t)], val_Y[float(t)] = h_val, y_val
            print(f"  [features] t={t:.3f} done (dim={rep_dim})")

        T = len(t_grid)
        M = np.zeros((T, T))
        for ai, t_a in enumerate(t_grid):
            t_a = float(t_a)
            probe = train_probe(train_X[t_a], train_Y[t_a], vocab_size, device, n_epochs=args.n_epochs)
            for bi, t_b in enumerate(t_grid):
                t_b = float(t_b)
                acc = eval_probe(probe, val_X[t_b], val_Y[t_b], device)
                M[ai, bi] = acc
            print(f"  [probe t_a={t_a:.3f}] diag={M[ai,ai]:.4f}  row_mean={M[ai].mean():.4f}")
            del probe
            torch.cuda.empty_cache()

        upper = M[np.triu_indices(T, k=1)]
        lower = M[np.tril_indices(T, k=-1)]
        diag = np.diag(M)
        results[rep] = {
            "rep_dim": rep_dim, "transfer_matrix": M.tolist(),
            "diag_mean": float(diag.mean()), "upper_tri_mean": float(upper.mean()),
            "lower_tri_mean": float(lower.mean()),
        }
        print(f"[PT9-extra] {rep}: diag_mean={results[rep]['diag_mean']:.4f}  "
              f"upper_tri_mean={results[rep]['upper_tri_mean']:.4f}  "
              f"lower_tri_mean={results[rep]['lower_tri_mean']:.4f}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "n_train_seqs": len(train_seqs), "n_val_seqs": len(val_seqs),
        "seq_len": L, "t_grid": t_grid.tolist(), "representations": results,
        "notes": [
            "'hidden' = last transformer block's output (matches EXP-07b's L10/L11-highest finding for ELF).",
            "'prior-subtracted logits' (doc's 4th representation) not implemented -- would need a "
            "Linear(vocab_size, vocab_size) probe, not tractable/meaningful at this sample size; "
            "see script docstring.",
            "Sequence-level train/val split, same as the main probe_cross_time_transfer.py.",
        ],
    }
    json_path = out_dir / f"cross_time_transfer_extra_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[PT9-extra] Saved {json_path}")


if __name__ == "__main__":
    main()
