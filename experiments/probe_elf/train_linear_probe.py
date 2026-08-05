"""
EXP-07 Phase 2: Train a linear probe on x̂_t states at each t and compare accuracy
against the native tied-weight head (Rec@1) and cosine readout (G).

Input: results/exp07/states/states_t{t:.3f}.pt (from collect_probe_states.py)
Output: results/exp07/probe_accuracies.json

Decision: if probe_acc(0.35) > Rec@1(0.35) + 10pp → Story A confirmed.

Usage (from ELF-torch root):
  python experiments/probe_elf/train_linear_probe.py \
    --states_dir results/exp07/states \
    --output_dir results/exp07 \
    --epochs 30 --lr 3e-3 --train_frac 0.8
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_state_file(path: str):
    """Load (x_hat, y_tokens, attn_mask) from a states file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    x_hat = data["x_hat"].float()       # (N, L, d)
    y = data["y_tokens"].long()          # (N, L)
    mask = data["attn_mask"].bool()      # (N, L) — True for valid positions
    return x_hat, y, mask


def flatten_for_probe(x_hat, y, mask):
    """Flatten to (valid_positions, d) and (valid_positions,)."""
    # Flatten batch × seq dims
    N, L, d = x_hat.shape
    x_flat = x_hat.reshape(N * L, d)         # (N*L, d)
    y_flat = y.reshape(N * L)                 # (N*L,)
    mask_flat = mask.reshape(N * L)           # (N*L,)
    x_valid = x_flat[mask_flat]               # (M, d)
    y_valid = y_flat[mask_flat]               # (M,)
    return x_valid, y_valid


def compute_native_accuracies(x_hat, y, mask, W, bias, chunk=4096):
    """Rec@1(t) and G(t) computed in chunks to avoid OOM."""
    x_flat, y_flat = flatten_for_probe(x_hat, y, mask)
    W_cuda = W.cuda()
    bias_cuda = bias.cuda() if bias is not None else None
    W_norm = W_cuda / (W_cuda.norm(dim=-1, keepdim=True) + 1e-8)

    rec1_correct = 0
    g_correct = 0
    n = x_flat.shape[0]
    for i in range(0, n, chunk):
        xc = x_flat[i: i + chunk].cuda()
        yc = y_flat[i: i + chunk].cuda()
        # Rec@1
        logits = xc @ W_cuda.T
        if bias_cuda is not None:
            logits = logits + bias_cuda
        rec1_correct += (logits.argmax(dim=-1) == yc).sum().item()
        # G
        xn = xc / (xc.norm(dim=-1, keepdim=True) + 1e-8)
        g_correct += ((xn @ W_norm.T).argmax(dim=-1) == yc).sum().item()

    rec1 = rec1_correct / n
    g_acc = g_correct / n
    return rec1, g_acc


def train_probe(x_train, y_train, x_val, y_val, vocab_size, d,
                epochs=30, lr=3e-3, batch_size=4096, device="cuda"):
    """Train a linear (logistic) probe and return (train_acc, val_acc)."""
    probe = nn.Linear(d, vocab_size).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    x_t = x_train.to(device)
    y_t = y_train.to(device)
    n = x_t.shape[0]

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            logits = probe(x_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(x_val.to(device))
                val_acc = (val_logits.argmax(dim=-1) == y_val.to(device)).float().mean().item()
            print(f"    epoch {epoch+1}/{epochs}  loss={total_loss/(n//batch_size+1):.4f}  val_acc={val_acc:.4f}")

    probe.eval()
    with torch.no_grad():
        # Chunked evaluation to avoid OOM
        def chunk_acc(x, y):
            x, y = x.to(device), y.to(device)
            correct = 0
            for i in range(0, len(x), batch_size):
                correct += (probe(x[i:i+batch_size]).argmax(dim=-1) == y[i:i+batch_size]).sum().item()
            return correct / len(x)
        train_acc = chunk_acc(x_train, y_train)
        val_acc   = chunk_acc(x_val, y_val)
    return train_acc, val_acc, probe


def seq_level_split(x_hat, y, mask, n_train_seqs):
    """Split at the sequence level: first n_train_seqs → train, rest → test.

    Returns (x_tr, y_tr, x_te, y_te, mask_tr, mask_te).
    This ensures no token from a test document leaks into probe training.
    """
    x_tr, y_tr, mask_tr = x_hat[:n_train_seqs], y[:n_train_seqs], mask[:n_train_seqs]
    x_te, y_te, mask_te = x_hat[n_train_seqs:], y[n_train_seqs:], mask[n_train_seqs:]
    return (
        *flatten_for_probe(x_tr, y_tr, mask_tr),
        *flatten_for_probe(x_te, y_te, mask_te),
    )


def shuffled_label_acc(probe_trained, x_val, y_val, vocab_size, d,
                       epochs, lr, batch_size, device):
    """Train a probe with shuffled labels; return val accuracy as a memorisation upper-bound."""
    y_shuf = y_val[torch.randperm(len(y_val))]   # CPU perm on CPU tensor
    n = min(len(x_val), 50000)
    x_s = x_val[:n]; y_s = y_shuf[:n]
    probe_shuf = nn.Linear(d, vocab_size).to(device)
    opt = torch.optim.AdamW(probe_shuf.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(n)   # keep perm on CPU to index CPU tensors
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            logits = probe_shuf(x_s[idx].to(device))
            F.cross_entropy(logits, y_s[idx].to(device)).backward()
            opt.step(); opt.zero_grad()
    probe_shuf.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, len(x_val), batch_size):
            xb = x_val[i:i+batch_size].to(device)
            yb = y_val[i:i+batch_size].to(device)
            correct += (probe_shuf(xb).argmax(-1) == yb).sum().item()
    return correct / len(x_val)


def main():
    parser = argparse.ArgumentParser(description="EXP-07v2: linear probe with document-level split")
    parser.add_argument("--states_dir", default="results/exp07/states")
    parser.add_argument("--output_dir", default="results/exp07")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--train_seq_frac", type=float, default=0.8,
                        help="Fraction of SEQUENCES (documents) for training. "
                             "All positions from a document go entirely to train or test.")
    parser.add_argument("--shuffled_control", action="store_true",
                        help="Also train a shuffled-label probe as memorisation control")
    parser.add_argument("--unembed_kernel_path", default=None,
                        help="ELF checkpoint path to load unembed_kernel for native head comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-07v2] Device: {device}  split=document-level (train_seq_frac={args.train_seq_frac})")

    # Load native embedding matrix (for Rec@1 and G comparisons)
    W, bias = None, None
    if args.unembed_kernel_path:
        ckpt = torch.load(args.unembed_kernel_path, map_location="cpu", weights_only=False)
        params = ckpt.get("params", ckpt)
        unembed_kernel = params["unembed_kernel"]  # (d, V)
        W = unembed_kernel.T.float()               # (V, d)
        bias = params.get("unembed_bias", None)
        if bias is not None:
            bias = bias.float()
        print(f"  Loaded W: {W.shape}")

    state_files = sorted(f for f in os.listdir(args.states_dir) if f.startswith("states_t") and f.endswith(".pt"))
    if not state_files:
        raise FileNotFoundError(f"No states_t*.pt files in {args.states_dir}")
    print(f"  Found {len(state_files)} state files")

    # Determine document-level split from the first file (N_seqs is fixed across all t)
    first = torch.load(os.path.join(args.states_dir, state_files[0]), map_location="cpu", weights_only=False)
    N_seqs = first["x_hat"].shape[0]
    n_train_seqs = int(N_seqs * args.train_seq_frac)
    n_test_seqs  = N_seqs - n_train_seqs
    print(f"  N_seqs={N_seqs}  train_seqs={n_train_seqs}  test_seqs={n_test_seqs}")

    results = []
    for fname in state_files:
        path = os.path.join(args.states_dir, fname)
        data = torch.load(path, map_location="cpu", weights_only=False)
        t_val = float(data["t"])
        print(f"\n[EXP-07v2] t={t_val:.3f} ...")

        x_hat, y, mask = load_state_file(path)
        d = x_hat.shape[-1]
        vocab_size = y.max().item() + 1
        if W is not None:
            vocab_size = W.shape[0]

        # Document-level split (critical fix vs position-level split)
        x_tr, y_tr, x_te, y_te = seq_level_split(x_hat, y, mask, n_train_seqs)
        print(f"  train_pos={len(x_tr)}  test_pos={len(x_te)}  d={d}  V={vocab_size}")

        train_acc, test_acc, probe = train_probe(
            x_tr, y_tr, x_te, y_te,
            vocab_size=vocab_size, d=d,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=str(device))
        print(f"  Probe  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
              f"overfit_gap={train_acc - test_acc:+.4f}")

        row = {"t": t_val, "probe_train_acc": train_acc, "probe_test_acc": test_acc,
               "overfit_gap": train_acc - test_acc}

        # Shuffled-label memorisation control
        if args.shuffled_control:
            shuf_acc = shuffled_label_acc(probe, x_te, y_te, vocab_size, d,
                                          epochs=args.epochs, lr=args.lr,
                                          batch_size=args.batch_size, device=str(device))
            row["shuffled_label_acc"] = shuf_acc
            print(f"  Shuffled-label test_acc={shuf_acc:.4f}  (should be ≈ random)")

        # Native head comparison on test sequences
        if W is not None:
            x_hat_te = x_hat[n_train_seqs:]
            y_te_full = y[n_train_seqs:]
            mask_te_full = mask[n_train_seqs:]
            rec1, g_acc = compute_native_accuracies(x_hat_te, y_te_full, mask_te_full, W, bias)
            row["rec1"] = rec1
            row["G"] = g_acc
            print(f"  Rec@1={rec1:.4f}  G={g_acc:.4f}  Gap(probe-Rec1)={test_acc-rec1:+.4f}")
        results.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "probe_accuracies_v2.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXP-07v2] Saved to {out_path}")

    # Decision summary
    print("\n=== EXP-07v2 Decision Summary (document-level test split) ===")
    print(f"{'t':>6} {'train':>8} {'test':>8} {'overfit':>8} {'Rec@1':>8} {'G':>8} {'gap':>8}")
    for r in results:
        rec_str = f"{r.get('rec1', float('nan')):.4f}"
        g_str   = f"{r.get('G',    float('nan')):.4f}"
        gap = r['probe_test_acc'] - r.get('rec1', r['probe_test_acc'])
        print(f"{r['t']:>6.3f} {r['probe_train_acc']:>8.4f} {r['probe_test_acc']:>8.4f} "
              f"{r['overfit_gap']:>+8.4f} {rec_str:>8} {g_str:>8} {gap:>+8.4f}")

    nearest = min(results, key=lambda r: abs(r['t'] - 0.35))
    gap = nearest['probe_test_acc'] - nearest.get('rec1', nearest['probe_test_acc'])
    overfit = nearest['overfit_gap']
    print(f"\nAt t≈0.35: probe_test={nearest['probe_test_acc']:.4f}  overfit_gap={overfit:+.4f}")
    if 'rec1' in nearest:
        if gap > 0.10:
            print(f"=> STORY A CONFIRMED (document-level): probe={nearest['probe_test_acc']:.4f} > Rec@1={nearest['rec1']:.4f} by {gap:+.4f}")
        elif gap > 0.05:
            print(f"=> MODERATE SIGNAL: gap={gap:+.4f}")
        else:
            print(f"=> STORY A NOT CONFIRMED: gap={gap:+.4f} (<5pp)")


if __name__ == "__main__":
    main()
