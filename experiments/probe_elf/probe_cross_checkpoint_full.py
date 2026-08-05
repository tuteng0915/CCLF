"""
EXP-07c-full: Full 3x3 cross-checkpoint probe transfer at every layer.

Trains linear probes on EACH checkpoint (baseline, kd-cr, kd2) at EACH layer,
and evaluates on ALL checkpoints. Gives full 3x3 transfer matrix per (layer, t).

Key question answered: Is L10's catastrophic geometric divergence from baseline's
perspective also catastrophic from kd-cr's and kd2's perspectives?

Usage (from ELF-torch root):
  CUDA_VISIBLE_DEVICES=5 python experiments/probe_elf/probe_cross_checkpoint_full.py \
    --states_dirs results/exp07b_baseline,results/exp07b_kd_cr,results/exp07b_kd2 \
    --checkpoint_names baseline,kd_cr,kd2 \
    --output_dir results/exp07c_full \
    --t_values 0.20,0.30,0.50,0.70
"""

import argparse, json, os, sys
import torch, torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


class LinearProbe(torch.nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.linear = torch.nn.Linear(d_in, n_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


def train_probe(X, Y, n_classes, device, n_epochs=15, lr=1e-2, batch_size=1024):
    N, d = X.shape
    probe = LinearProbe(d, n_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)
    X_dev = X.to(device)
    Y_dev = Y.to(device)

    # Full-batch SGD
    for ep in range(n_epochs):
        probe.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx  = perm[i:i+batch_size]
            logits = probe(X_dev[idx])
            loss   = F.cross_entropy(logits, Y_dev[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        all_logits = probe(X_dev)
        acc = (all_logits.argmax(-1) == Y_dev).float().mean().item()
    return probe, acc


def eval_probe(probe, X, Y, device, batch_size=2048):
    probe.eval()
    X_dev = X.to(device)
    Y_dev = Y.to(device)
    correct = []
    with torch.no_grad():
        for i in range(0, X_dev.shape[0], batch_size):
            xb = X_dev[i:i+batch_size]
            yb = Y_dev[i:i+batch_size]
            correct.append((probe(xb).argmax(-1) == yb).float())
    return torch.cat(correct).mean().item()


def load_layer_data(states_dir, t_val, device, layer_idx):
    path = os.path.join(states_dir, f"layer_states_t{t_val:.3f}.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    layer_feats = data["layer_feats"]  # list of (N, L, 768)
    x  = layer_feats[layer_idx]        # (N, L, 768)
    y  = data["y_tokens"]              # (N, L)
    m  = data["attn_mask"].bool()      # (N, L)

    x_flat = x.reshape(-1, x.shape[-1]).float()  # (N*L, d)
    y_flat = y.reshape(-1).long()                 # (N*L,)
    m_flat = m.reshape(-1)                        # (N*L,)

    return x_flat[m_flat], y_flat[m_flat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dirs",       required=True, help="comma-sep list of states dirs")
    ap.add_argument("--checkpoint_names",  required=True, help="comma-sep checkpoint names")
    ap.add_argument("--output_dir",        required=True)
    ap.add_argument("--t_values",          default="0.20,0.30,0.50,0.70")
    ap.add_argument("--n_layers",          type=int, default=12)
    ap.add_argument("--n_epochs",          type=int, default=15)
    args = ap.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states_dirs = args.states_dirs.split(",")
    ckpt_names  = args.checkpoint_names.split(",")
    t_vals      = [float(x) for x in args.t_values.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    assert len(states_dirs) == len(ckpt_names)
    n_ckpts = len(ckpt_names)

    # Get vocab size from first file
    first_data = torch.load(
        os.path.join(states_dirs[0], f"layer_states_t{t_vals[0]:.3f}.pt"),
        map_location="cpu", weights_only=False)
    n_classes = int(first_data["y_tokens"].max().item()) + 1
    print(f"vocab={n_classes} checkpoints={ckpt_names} t_vals={t_vals}")

    all_results = {}

    for t_val in t_vals:
        print(f"\n{'='*64}")
        print(f"t = {t_val:.2f}")
        print(f"{'='*64}")

        # Load data for all checkpoints
        ckpt_data = {}
        for c_name, s_dir in zip(ckpt_names, states_dirs):
            ckpt_data[c_name] = {}

        t_results = {}

        for layer_idx in range(args.n_layers):
            # Load features for this layer from all checkpoints
            feats = {}
            labels = {}
            for c_name, s_dir in zip(ckpt_names, states_dirs):
                X, Y = load_layer_data(s_dir, t_val, device, layer_idx)
                feats[c_name]  = X
                labels[c_name] = Y

            # Train a probe on each source checkpoint, eval on all
            layer_matrix = {}
            for src_name in ckpt_names:
                X_src = feats[src_name]
                Y_src = labels[src_name]
                probe, train_acc = train_probe(X_src, Y_src, n_classes, device, n_epochs=args.n_epochs)

                row = {}
                for tgt_name in ckpt_names:
                    X_tgt = feats[tgt_name]
                    Y_tgt = labels[tgt_name]
                    acc = eval_probe(probe, X_tgt, Y_tgt, device)
                    row[tgt_name] = acc
                layer_matrix[src_name] = row

            t_results[str(layer_idx)] = layer_matrix

            # Print summary row
            if layer_idx % 3 == 0 or layer_idx == args.n_layers - 1:
                b_to_k = layer_matrix["baseline"].get("kd_cr", 0)
                k_to_b = layer_matrix["kd_cr"].get("baseline", 0)
                k_to_k2 = layer_matrix["kd_cr"].get("kd2", 0)
                b_self = layer_matrix["baseline"].get("baseline", 0)
                print(f"  L{layer_idx:2d}: base->self={b_self:.4f}  base->kd={b_to_k:.4f}  kd->base={k_to_b:.4f}  kd->kd2={k_to_k2:.4f}")

        all_results[str(t_val)] = t_results

    out = os.path.join(args.output_dir, "cross_checkpoint_full.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[EXP-07c-full] Saved to {out}")


if __name__ == "__main__":
    main()
