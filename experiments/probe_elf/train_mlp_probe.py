"""
EXP-07 MLP probe: tests whether the decoder > linear probe gap is due to nonlinearity.

Trains a 2-layer MLP (GELU activation) on x̂_t states and compares to:
  1. Linear probe (from train_linear_probe.py)
  2. Native decoder (factored gelu + unembed path)

If MLP probe ≈ native decoder >> linear probe → nonlinearity is key.
If MLP probe ≈ linear probe < decoder → decoder has specific structure learned from KD.

Usage:
  python experiments/probe_elf/train_mlp_probe.py \
    --states_dir results/exp07_kd_cr/states \
    --output_dir results/exp07_kd_cr \
    --hidden_size 512 \
    --epochs 30
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_state_file(path: str):
    data = torch.load(path, map_location="cpu", weights_only=False)
    x_hat = data["x_hat"].float()
    y = data["y_tokens"].long()
    mask = data["attn_mask"].bool()
    return x_hat, y, mask


def flatten_for_probe(x_hat, y, mask):
    N, L, d = x_hat.shape
    x_flat = x_hat.reshape(N * L, d)
    y_flat = y.reshape(N * L)
    mask_flat = mask.reshape(N * L)
    return x_flat[mask_flat], y_flat[mask_flat]


class MLPProbe(nn.Module):
    def __init__(self, d_in, hidden, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def train_probe(model, x_train, y_train, x_val, y_val,
                epochs=30, lr=3e-3, batch_size=4096, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    x_t = x_train.to(device)
    y_t = y_train.to(device)
    n = x_t.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            logits = model(x_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                for i in range(0, len(x_val), batch_size):
                    out = model(x_val[i: i + batch_size].to(device))
                    correct += (out.argmax(-1) == y_val[i: i + batch_size].to(device)).sum().item()
                val_acc = correct / len(x_val)
            print(f"    epoch {epoch+1}/{epochs}  loss={total_loss/(n//batch_size+1):.4f}  val_acc={val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, len(x_val), batch_size):
            out = model(x_val[i: i + batch_size].to(device))
            correct += (out.argmax(-1) == y_val[i: i + batch_size].to(device)).sum().item()
    return correct / len(x_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Hidden layer size (default=512 matches T5 latent dim)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--train_frac", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MLP-PROBE] Device: {device}, hidden={args.hidden_size}")

    state_files = sorted(f for f in os.listdir(args.states_dir)
                         if f.startswith("states_t") and f.endswith(".pt"))
    if not state_files:
        raise FileNotFoundError(f"No states_t*.pt files in {args.states_dir}")
    print(f"  Found {len(state_files)} state files")

    results = []
    for fname in state_files:
        path = os.path.join(args.states_dir, fname)
        data = torch.load(path, map_location="cpu", weights_only=False)
        t_val = float(data["t"])
        print(f"\n[MLP-PROBE] t={t_val:.3f} ...")

        x_hat, y, mask = load_state_file(path)
        x_flat, y_flat = flatten_for_probe(x_hat, y, mask)

        N = x_flat.shape[0]
        d = x_flat.shape[1]
        vocab_size = int(y_flat.max().item()) + 1

        n_train = int(N * args.train_frac)
        perm = torch.randperm(N)
        x_train = x_flat[perm[:n_train]]
        y_train = y_flat[perm[:n_train]]
        x_val = x_flat[perm[n_train:]]
        y_val = y_flat[perm[n_train:]]
        print(f"  N={N}: train={n_train}, val={N-n_train}, d={d}, V={vocab_size}")

        model = MLPProbe(d, args.hidden_size, vocab_size)
        probe_acc = train_probe(model, x_train, y_train, x_val, y_val,
                                epochs=args.epochs, lr=args.lr,
                                batch_size=args.batch_size, device=str(device))
        print(f"  MLP probe val_acc = {probe_acc:.4f}")
        results.append({"t": t_val, "mlp_probe_acc": probe_acc, "hidden": args.hidden_size})

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "mlp_probe_accuracies.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[MLP-PROBE] Saved to {out_path}")

    print("\n=== MLP Probe Summary ===")
    print(f"{'t':>6}  {'mlp_probe':>10}")
    for r in results:
        print(f"{r['t']:>6.3f}  {r['mlp_probe_acc']:>10.4f}")


if __name__ == "__main__":
    main()
