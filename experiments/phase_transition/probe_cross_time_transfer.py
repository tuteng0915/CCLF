"""EXP-PT9: Cross-Time Evidence-Direction Transfer.

Determines whether evidence accumulates along a stable token-discriminative
direction (probe trained at t_a keeps working at t_b far from t_a) or is
repeatedly re-encoded across time (only works near its own training t).

Method: train a linear probe P_{t_a} (full-vocab linear classifier, matching
the architecture used in EXP-07c's probe_cross_checkpoint_full.py -- see
docs/specs/EXP-PT9-spec.md) at each t_a on the predicted_clean state, then
evaluate it at every t_b to build a transfer matrix M[a,b] = accuracy.

Unlike EXP-07c (which trains and evaluates on the SAME flattened
position pool, i.e. reports train accuracy on the diagonal), this script
uses a SEQUENCE-level train/val split (same lesson as EXP-07v2 and this
suite's EXP-PT10): probes are trained on TRAIN sequences and the entire
transfer matrix -- including the diagonal M[a,a] -- is evaluated on held-out
VAL sequences. This makes the diagonal a genuine held-out accuracy, not an
inflated train accuracy, at the cost of not being directly comparable to
EXP-07c's numbers.

Only the "predicted_clean" state representation (doc's suggestion, and the
same one EXP-07d already used for cross-checkpoint transfer) is implemented.
NOT implemented: raw states, native hidden states, prior-subtracted logits
-- flagged as follow-ups in EXP-PT9-spec.md.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/probe_cross_time_transfer.py \\
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
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


class LinearProbe(torch.nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.linear = torch.nn.Linear(d_in, n_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


def train_probe(X, Y, n_classes, device, n_epochs=15, lr=1e-2, batch_size=2048):
    N, d = X.shape
    probe = LinearProbe(d, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    X_dev, Y_dev = X.to(device), Y.to(device)
    for _ in range(n_epochs):
        probe.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(probe(X_dev[idx]), Y_dev[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    probe.eval()
    return probe


@torch.no_grad()
def eval_probe(probe, X, Y, device, batch_size=4096):
    correct = []
    for i in range(0, X.shape[0], batch_size):
        xb, yb = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        correct.append((probe(xb).argmax(-1) == yb).float())
    return torch.cat(correct)  # (n_positions,) flat, caller reduces as needed


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

    print(f"[PT9] Loading {args.model} model...")
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
    print(f"[PT9] {N} sequences, L={L}, T={len(t_grid)} t-points, vocab={vocab_size}")

    rng = np.random.default_rng(args.seed)
    seq_order = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_seqs, train_seqs = seq_order[:n_val], seq_order[n_val:]
    print(f"[PT9] {len(train_seqs)} train sequences, {len(val_seqs)} val sequences (sequence-level split)")

    @torch.no_grad()
    def get_predicted_clean(t):
        eps = adapter.sample_epsilon((N, L, d))
        z = adapter.make_oracle_state(x_clean.to(device), eps, float(t))
        out = adapter.forward_state(z.cpu(), None, float(t), batch_size=args.batch_size)
        return out["predicted_clean"]  # (N,L,d)

    # Cache flattened, mask-free (all positions valid; no padding excluded
    # here -- see EXP-PT1-spec.md's padding caveat, same simplification
    # carried over) train/val features and labels per t.
    train_X, train_Y, val_X, val_Y = {}, {}, {}, {}
    for t in t_grid:
        h = get_predicted_clean(t)  # (N,L,d)
        h_train = h[train_seqs].reshape(-1, d)
        y_train = gt_ids[train_seqs].reshape(-1)
        h_val = h[val_seqs].reshape(-1, d)
        y_val = gt_ids[val_seqs].reshape(-1)
        train_X[float(t)], train_Y[float(t)] = h_train, y_train
        val_X[float(t)], val_Y[float(t)] = h_val, y_val
        print(f"  [features] t={t:.3f} done")

    T = len(t_grid)
    M = np.zeros((T, T))
    # Per-sequence accuracy for every (t_a, t_b) cell -- needed for sequence-
    # level bootstrap CI on diag_mean/upper_tri_mean/lower_tri_mean (rigor-audit
    # follow-up, same pattern as EXP-PT4's acc_per_seq / bootstrap_pt4.py).
    acc_per_seq_matrix = np.zeros((T, T, len(val_seqs)))
    for ai, t_a in enumerate(t_grid):
        t_a = float(t_a)
        probe = train_probe(train_X[t_a], train_Y[t_a], vocab_size, device, n_epochs=args.n_epochs)
        for bi, t_b in enumerate(t_grid):
            t_b = float(t_b)
            correct_flat = eval_probe(probe, val_X[t_b], val_Y[t_b], device)  # (n_val*L,)
            correct_per_seq = correct_flat.view(len(val_seqs), L).mean(dim=1)  # (n_val,)
            acc_per_seq_matrix[ai, bi] = correct_per_seq.cpu().numpy()
            M[ai, bi] = correct_per_seq.mean().item()
        print(f"  [probe t_a={t_a:.3f}] diag={M[ai,ai]:.4f}  row_mean={M[ai].mean():.4f}")
        del probe
        torch.cuda.empty_cache()

    upper = M[np.triu_indices(T, k=1)]
    lower = M[np.tril_indices(T, k=-1)]
    diag = np.diag(M)

    npz_path = out_dir / f"cross_time_transfer_raw_{args.label}.npz"
    np.savez_compressed(npz_path, acc_per_seq_matrix=acc_per_seq_matrix, t_grid=t_grid)
    print(f"[PT9] Saved per-sequence accuracy matrix to {npz_path}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "n_train_seqs": len(train_seqs), "n_val_seqs": len(val_seqs),
        "seq_len": L, "t_grid": t_grid.tolist(),
        "transfer_matrix": M.tolist(),
        "diag_mean": float(diag.mean()), "upper_tri_mean": float(upper.mean()),
        "lower_tri_mean": float(lower.mean()),
        "notes": [
            "predicted_clean state only -- raw states, native hidden states, and "
            "prior-subtracted logits (doc's other 3 representations) not implemented.",
            "Sequence-level train/val split (probes trained on TRAIN sequences, the WHOLE "
            "matrix incl. the diagonal is evaluated on held-out VAL sequences) -- unlike "
            "EXP-07c which reported train accuracy on its diagonal; numbers are not directly "
            "comparable to EXP-07c's.",
            "Padding positions not excluded (same simplification as EXP-PT1's known caveat).",
        ],
    }
    json_path = out_dir / f"cross_time_transfer_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT9] diag_mean={summary['diag_mean']:.4f}  "
          f"upper_tri_mean(early probe->later state)={summary['upper_tri_mean']:.4f}  "
          f"lower_tri_mean(late probe->earlier state)={summary['lower_tri_mean']:.4f}")
    print(f"[PT9] Saved {json_path}")


if __name__ == "__main__":
    main()
