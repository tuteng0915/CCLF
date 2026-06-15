"""
Four-model cross-model comparison figure.

Left panel:  top-1 GT accuracy (all positions) vs t
Right panel: entropy_all (all positions) vs t

Models:
  ELF      — probe_v3 (tau=1.0 key), top1_gt_mean / entropy_mean
  LangFlow — probe_v2 (tau=1.0 key), top1_gt_mean / entropy_mean
  MDLM     — probe_v2,               top1_gt_all_mean / entropy_all_mean
  DUO      — probe_v2,               top1_gt_all_mean / entropy_all_mean

All models: t=0 noisy, t=1 clean.
"""

import argparse, json, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).parent.parent.parent  # CCLF/CCLF/

MODELS = {
    "ELF": {
        "path": ROOT / "results/elf/probe_v3/anchor_probe_v3.json",
        "tau_key": "1.0",
        "top1": "top1_gt_mean",
        "entropy": "entropy_mean",
        "color": "#2196F3",
        "ls": "-",
        "lw": 2.5,
    },
    "LangFlow": {
        "path": ROOT / "results/langflow/probe_v2/langflow_probe.json",
        "tau_key": None,
        "top1": "top1_gt_mean",
        "entropy": "entropy_mean",
        "color": "#FF9800",
        "ls": "-",
        "lw": 2.0,
    },
    "MDLM": {
        "path": ROOT / "results/mdlm/probe_v2/mdlm_probe.json",
        "tau_key": None,
        "top1": "top1_gt_all_mean",
        "entropy": "entropy_all_mean",
        "color": "#4CAF50",
        "ls": "--",
        "lw": 2.0,
    },
    "DUO": {
        "path": ROOT / "results/duo/probe_v2/duo_probe.json",
        "tau_key": None,
        "top1": "top1_gt_all_mean",
        "entropy": "entropy_all_mean",
        "color": "#E91E63",
        "ls": "--",
        "lw": 2.0,
    },
}


def load(cfg):
    with open(cfg["path"]) as f:
        d = json.load(f)
    if cfg["tau_key"] and cfg["tau_key"] in d:
        d = d[cfg["tau_key"]]
    elif "t" not in d:
        d = d[list(d.keys())[0]]
    return np.array(d["t"]), d


def main(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97, top=0.88, bottom=0.14)

    for name, cfg in MODELS.items():
        t, d = load(cfg)

        # left: top-1 accuracy
        y1 = np.array(d[cfg["top1"]])
        axes[0].plot(t, y1, color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"],
                     label=name)

        # right: entropy
        y2 = np.array(d[cfg["entropy"]])
        axes[1].plot(t, y2, color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"],
                     label=name)

    # ── left panel ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Diffusion time $t$  (noise → clean)", fontsize=9)
    ax.set_ylabel("Top-1 GT accuracy (all positions)", fontsize=9)
    ax.set_title("(a) Lexical recovery across DLMs", fontsize=10, pad=7)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # ── right panel ───────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_xlabel("Diffusion time $t$  (noise → clean)", fontsize=9)
    ax.set_ylabel("Token entropy $H$ — all positions (nats)", fontsize=9)
    ax.set_title("(b) Token uncertainty across DLMs", fontsize=10, pad=7)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # ── save ──────────────────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {str(out).replace('.pdf', '.png')}")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/figures/crossmodel_comparison.pdf")
    args = p.parse_args()
    main(args.out)
