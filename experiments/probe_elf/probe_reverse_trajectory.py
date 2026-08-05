"""
EXP-01: Forward-oracle probe (Protocol A) vs reverse-trajectory probe (Protocol B).

Protocol A: for each t, independently sample z_t = t*x_clean + (1-t)*eps, call backbone once.
            This is what existing probe_geo.py computes on JAX ELF.
Protocol B: run actual ODE/SDE sampler, save states at each step — this script.

Usage:
  # Step 1: generate trajectories (run from ELF-torch root):
  #   python src/eval.py --config src/configs/training_configs/eval_exp01.yml \
  #                      --checkpoint_path converted/elf_b-owt-kd-cr_torch.pt
  #
  # Step 2: probe saved trajectories:
  #   python experiments/probe_elf/probe_reverse_trajectory.py \
  #          --traj_dir results/exp01/trajectories \
  #          --checkpoint converted/elf_b-owt-kd-cr_torch.pt \
  #          --proto_a_json results/proto_a_metrics.json \
  #          --output_dir results/exp01
"""

import argparse
import json
import os
import time

import numpy as np
import torch


# ─── Metric functions ─────────────────────────────────────────────────────────

def compute_G(x_hat: torch.Tensor, E_norm: torch.Tensor, y: torch.Tensor) -> float:
    """Cosine-normalized token readout accuracy G(t)."""
    x_n = x_hat.float() / (x_hat.float().norm(dim=-1, keepdim=True) + 1e-8)  # (B, L, d) or (L, d)
    sims = x_n @ E_norm.T.float()  # (..., V)
    preds = sims.argmax(dim=-1)
    return (preds == y).float().mean().item()


def compute_rec1(x_hat: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None,
                 y: torch.Tensor) -> float:
    """Native linear readout Rec@1(t)."""
    logits = x_hat.float() @ W.float().T
    if bias is not None:
        logits = logits + bias.float()
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()


def compute_entropy(x_hat: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None) -> float:
    """Token-belief entropy H(t)."""
    logits = x_hat.float() @ W.float().T
    if bias is not None:
        logits = logits + bias.float()
    p = torch.softmax(logits, dim=-1)
    H = -(p * torch.log(p + 1e-10)).sum(dim=-1).mean().item()
    return H


def compute_rho(x_hat: torch.Tensor, E: torch.Tensor, W: torch.Tensor,
                bias: torch.Tensor | None) -> float:
    """Anchor mismatch ratio rho(t) = ||x_hat - E^T p_t|| / ||x_hat||."""
    logits = x_hat.float() @ W.float().T
    if bias is not None:
        logits = logits + bias.float()
    p = torch.softmax(logits, dim=-1)   # (..., V)
    a = p @ E.float()                   # (..., d)   barycenter
    r = x_hat.float() - a
    rho = (r.norm(dim=-1) / (x_hat.float().norm(dim=-1) + 1e-8)).mean().item()
    return rho


# ─── Load embeddings from checkpoint ──────────────────────────────────────────

def load_embeddings(checkpoint_path: str):
    """
    Returns (E, W, bias) where:
      E   : (V, d) input embeddings (tied with W in ELF)
      W   : (V, d) unembedding matrix
      bias: (V,) output bias, or None
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = ckpt.get("params", ckpt)

    # Try common key names
    # ELF uses factored decoder: unembed_kernel (text_encoder_dim, vocab_size).
    # E (vocab_size, text_encoder_dim) = unembed_kernel.T — rows are token centroids.
    unembed_kernel = None
    for key in ["unembed_kernel", "model.unembed_kernel"]:
        if key in params:
            unembed_kernel = params[key]
            break
    if unembed_kernel is None:
        for k, v in params.items():
            if hasattr(v, "shape"):
                print(f"  {k}: {v.shape}")
        raise KeyError(
            "Cannot locate unembed_kernel. Run with --list_keys to inspect checkpoint."
        )

    E = unembed_kernel.T  # (vocab_size, text_encoder_dim)
    # W = E for tied-weight computation of Rec@1; bias from unembed_bias
    W = E
    bias = params.get("unembed_bias", None)
    return E, W, bias


# ─── Protocol B: compute metrics on saved trajectories ────────────────────────

def probe_trajectories(traj_dir: str, E: torch.Tensor, W: torch.Tensor,
                       bias: torch.Tensor | None) -> dict:
    """
    Load *.pt trajectory files, compute per-step metrics.
    Returns {metric: list of (t, value)}.
    """
    E_norm = E / (E.norm(dim=-1, keepdim=True) + 1e-8)
    metrics = {"G": [], "Rec1": [], "entropy": [], "rho": []}

    files = sorted(f for f in os.listdir(traj_dir) if f.endswith(".pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {traj_dir}")

    print(f"  Found {len(files)} trajectory files in {traj_dir}")
    for fname in files:
        traj = torch.load(os.path.join(traj_dir, fname), map_location="cpu", weights_only=False)
        for step in traj:
            t_val = step["t"]
            x_hat = step["x_pred"]  # (B, L, d) on CPU

            # We don't have ground-truth tokens from the sampler (unconditional generation),
            # so we use the final-step x_pred argmax as a proxy for "committed" token.
            # For a cleaner comparison, compute G and Rec@1 against each other (correlation)
            # and report only entropy + rho where no GT is needed.
            # ──────────────────────────────────────────────────────────────────────
            # NOTE: for EXP-01 we compare SHAPE of G(t) curve vs Protocol A, not absolute value.
            # We use the terminal prediction (last step's argmax under W) as GT proxy.
            # (True GT would require running the probe on a validation set with known tokens.)
            # ──────────────────────────────────────────────────────────────────────
            pass  # GT-based metrics computed below after we have the last step
            H = compute_entropy(x_hat, W, bias)
            rho = compute_rho(x_hat, E, W, bias)
            metrics["entropy"].append((t_val, H))
            metrics["rho"].append((t_val, rho))

        # For G and Rec@1: use the last step's argmax-W prediction as GT proxy
        last_x = traj[-1]["x_pred"]
        y_proxy = (last_x.float() @ W.float().T).argmax(dim=-1)  # (B, L)
        if bias is not None:
            y_proxy = (last_x.float() @ W.float().T + bias.float()).argmax(dim=-1)
        for step in traj:
            t_val = step["t"]
            x_hat = step["x_pred"]
            metrics["G"].append((t_val, compute_G(x_hat, E_norm, y_proxy)))
            metrics["Rec1"].append((t_val, compute_rec1(x_hat, W, bias, y_proxy)))

    return metrics


# ─── Bin and aggregate ────────────────────────────────────────────────────────

def bin_metrics(raw: dict, n_bins: int = 20) -> dict:
    """Average metric values into n_bins equally-spaced t bins."""
    t_edges = np.linspace(0.0, 1.0, n_bins + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    binned = {}
    for metric, pairs in raw.items():
        ts = np.array([p[0] for p in pairs])
        vs = np.array([p[1] for p in pairs])
        means, stds, counts = [], [], []
        for lo, hi in zip(t_edges[:-1], t_edges[1:]):
            mask = (ts >= lo) & (ts < hi)
            if mask.any():
                means.append(float(vs[mask].mean()))
                stds.append(float(vs[mask].std()))
                counts.append(int(mask.sum()))
            else:
                means.append(float("nan"))
                stds.append(float("nan"))
                counts.append(0)
        binned[metric] = {
            "t": t_centers.tolist(),
            "mean": means,
            "std": stds,
            "count": counts,
        }
    return binned


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_comparison(binned_B: dict, proto_A_file: str | None, output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plot")
        return

    metrics = ["G", "Rec1", "entropy", "rho"]
    titles = ["G(t) cosine readout", "Rec@1(t) linear readout", "Entropy H(t)", "Rho(t)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    proto_A = None
    if proto_A_file and os.path.exists(proto_A_file):
        with open(proto_A_file) as f:
            proto_A = json.load(f)

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        b = binned_B.get(metric, {})
        t_B = b.get("t", [])
        v_B = b.get("mean", [])
        ax.plot(t_B, v_B, "o-", color="orange", label="Protocol B (reverse traj)", linewidth=2)

        if proto_A and metric in proto_A:
            pa = proto_A[metric]
            t_A = [x["t"] for x in pa]
            v_A = [x["val"] for x in pa]
            ax.plot(t_A, v_A, "s--", color="steelblue", label="Protocol A (oracle)", linewidth=2)

        ax.set_xlabel("t")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("EXP-01: Forward-Oracle vs Reverse-Trajectory Probe", fontsize=13)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "protocol_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot to {out_path}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EXP-01: probe reverse-generation trajectories")
    parser.add_argument("--traj_dir", default="results/exp01/trajectories",
                        help="Directory containing trajectory .pt files from generation")
    parser.add_argument("--checkpoint", required=True,
                        help="ELF-torch checkpoint path (for embedding matrix)")
    parser.add_argument("--proto_a_json", default=None,
                        help="Optional Protocol-A metric JSON for overlay (from probe_geo.py)")
    parser.add_argument("--output_dir", default="results/exp01",
                        help="Output directory for binned metrics JSON and plots")
    parser.add_argument("--list_keys", action="store_true",
                        help="List checkpoint keys and exit (useful for debugging)")
    args = parser.parse_args()

    if args.list_keys:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        params = ckpt.get("params", ckpt)
        for k, v in sorted(params.items()):
            shape = v.shape if hasattr(v, "shape") else type(v)
            print(f"  {k}: {shape}")
        return

    print(f"[EXP-01] Loading embeddings from {args.checkpoint} ...")
    E, W, bias = load_embeddings(args.checkpoint)
    print(f"  E shape: {E.shape}, W shape: {W.shape}, bias: {bias.shape if bias is not None else None}")

    print(f"[EXP-01] Probing trajectories in {args.traj_dir} ...")
    t0 = time.time()
    raw_B = probe_trajectories(args.traj_dir, E, W, bias)
    print(f"  Done in {time.time()-t0:.1f}s")

    binned_B = bin_metrics(raw_B)

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "proto_B_metrics.json")
    with open(out_json, "w") as f:
        json.dump(binned_B, f, indent=2)
    print(f"  Saved binned metrics to {out_json}")

    print(f"[EXP-01] Generating comparison plot ...")
    plot_comparison(binned_B, args.proto_a_json, args.output_dir)

    # Print summary table
    print("\n=== EXP-01 Protocol B metric summary (mean per t-bin) ===")
    for metric in ["G", "Rec1", "entropy", "rho"]:
        b = binned_B.get(metric, {})
        ts = b.get("t", [])
        vs = b.get("mean", [])
        print(f"\n  {metric}:")
        for t_val, v in zip(ts, vs):
            if not np.isnan(v):
                print(f"    t={t_val:.3f}  {v:.4f}")


if __name__ == "__main__":
    main()
