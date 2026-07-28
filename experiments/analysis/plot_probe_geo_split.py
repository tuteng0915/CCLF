"""
plot_probe_geo_split.py — Split probe_geo metrics into two separate figures.

Fig A (denoiser metrics, 1×2):
  Left:  Cosine Margin (cos_1st - cos_2nd)
  Right: Residual ρ(t) = ||x̂_t - Σ_v p_v·E_v|| / ||x̂_t||

Fig B (decoder metric, standalone):
  Entropy of Decoder Head Output H(p_dec)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLIFF_T = 0.30
C_MARGIN = "#8c6d31"    # olive/gold for margin
C_RESID  = "#843c39"    # dark red for residual
C_ENTROPY = "#6b4c9a"   # purple for entropy


def load(path):
    with open(path) as f:
        return json.load(f)


def _shade(ax, t, mean, std, color, label, lw=2.0):
    ax.plot(t, mean, color=color, lw=lw, label=label)
    ax.fill_between(t, np.array(mean) - np.array(std),
                    np.array(mean) + np.array(std),
                    color=color, alpha=0.18)


def fig_denoiser(d, out_path):
    t       = np.array(d["t"])
    margin  = np.array(d["cos_margin_mean"])
    margin_s = np.array(d["cos_margin_std"])
    resid   = np.array(d["l2_residual_frac_mean"])
    resid_s = np.array(d["l2_residual_frac_std"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("ELF-B Denoiser-Side Probe Metrics", fontsize=12, y=1.01)

    # Left: Cosine Margin
    ax = axes[0]
    _shade(ax, t, margin, margin_s, C_MARGIN, "mean ± 1σ")
    ax.axvline(CLIFF_T, color="black", ls="--", lw=0.9, alpha=0.5)
    ax.text(CLIFF_T + 0.01, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.15,
            "cliff", fontsize=7.5, va="top", color="black", alpha=0.6)
    ax.set_xlabel("Diffusion time $t$  (0 = noisy, 1 = clean)", fontsize=9)
    ax.set_ylabel("Cosine margin (cos$_{1}$ − cos$_{2}$)", fontsize=9)
    ax.set_title("Geometric Commit Margin", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    # Right: Residual ρ(t)
    ax = axes[1]
    _shade(ax, t, resid, resid_s, C_RESID, "mean ± 1σ")
    ax.axvline(CLIFF_T, color="black", ls="--", lw=0.9, alpha=0.5)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8, alpha=0.7, label="ρ = 1")
    ax.set_xlabel("Diffusion time $t$  (0 = noisy, 1 = clean)", fontsize=9)
    ax.set_ylabel(
        r"$\rho(t) = \|\hat{x}_t - \sum_v p_v E_v\|\ /\ \|\hat{x}_t\|$",
        fontsize=9)
    ax.set_title("Anchor Residual Fraction", fontsize=10)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def fig_decoder_entropy(d, out_path):
    t     = np.array(d["t"])
    ent   = np.array(d["entropy_decoder_mean"])
    ent_s = np.array(d["entropy_decoder_std"])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    _shade(ax, t, ent, ent_s, C_ENTROPY, "mean ± 1σ")
    ax.axvline(CLIFF_T, color="black", ls="--", lw=0.9, alpha=0.5,
               label=f"cliff $t={CLIFF_T}$")
    ax.set_xlabel("Diffusion time $t$  (0 = noisy, 1 = clean)", fontsize=9)
    ax.set_ylabel("$H(p_{\\mathrm{dec}})$  [nats]", fontsize=9)
    ax.set_title("Decoder Head Entropy in ELF-B", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geo_json", default="results/elf/probe_geo_v1/probe_geo.json")
    p.add_argument("--out_dir",  default="models/ELF/src/outputs/figures")
    args = p.parse_args()

    d = load(args.geo_json)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig_denoiser(d, out / "fig3a_denoiser_metrics.png")
    fig_decoder_entropy(d, out / "fig3b_decoder_entropy.png")


if __name__ == "__main__":
    main()
