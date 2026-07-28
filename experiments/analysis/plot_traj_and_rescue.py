"""
plot_traj_and_rescue.py — Two figure variants for trajectory types + decoder rescue.

Option 1 (fig_combined): single 1×2 figure
  Left:  all 4 trajectory types overlaid (cos_gt vs t)
  Right: denoiser NN accuracy vs decoder head accuracy (rescue gap)

Option 2 (fig_split): two separate figures
  Fig A: compact 4-type overlay (replaces the 3-panel fig8)
  Fig B: decoder rescue with annotations
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT  = Path(__file__).parent.parent.parent   # CCLF/CCLF
TRAJ  = ROOT / "results/elf/probe_token_traj_v1/probe_token_traj.json"
GEO   = ROOT / "results/elf/probe_geo_v1/probe_geo.json"
DEC   = ROOT / "results/elf/probe_decode_v1/probe_decode_branch.json"
FIGS  = ROOT / "models/ELF/src/outputs/figures"

TYPE_COLOR = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#1f77b4", "D": "#d62728"}
TYPE_LABEL = {
    "A": "Correct Early",
    "B": "Wrong → Recommit",
    "C": "Late Commit",
    "D": "Stays Wrong",
}
CLIFF = 0.30


# ── data loading ──────────────────────────────────────────────────────────────

def load_all():
    with open(TRAJ) as f: traj = json.load(f)
    with open(GEO)  as f: geo  = json.load(f)
    with open(DEC)  as f: dec  = json.load(f)
    return traj, geo, dec


def traj_arrays(traj, k):
    """Return (t, mean, std, n, frac) for type k."""
    t    = np.array(traj["t"])
    mean = np.array(traj[f"{k}_mean"])
    std  = np.array(traj[f"{k}_std"])
    n    = traj[f"{k}_n"]
    frac = traj["fracs"][k]
    return t, mean, std, n, frac


def rescue_arrays(geo, dec):
    t       = np.array(geo["t"])
    cos_nn  = np.array(geo["cos_nn_correct_mean"])
    dec_top = np.array(dec["dec_top1_gt_mean"])
    # mask t=1.0 artifact (decoder degrades there due to two-pass convention)
    mask        = t < 0.99
    return t, cos_nn, dec_top, mask


# ── shared helpers ────────────────────────────────────────────────────────────

def _band(ax, t, mean, std, color, label, lw=2.0, ls="-", alpha_fill=0.15):
    valid = ~np.isnan(mean)
    ax.plot(t[valid], mean[valid], color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(t[valid],
                    (mean - std)[valid], (mean + std)[valid],
                    color=color, alpha=alpha_fill)


def _add_cliff(ax, ymax=1.05):
    ax.axvline(CLIFF, color="black", ls="--", lw=0.9, alpha=0.45)
    ax.text(CLIFF + 0.01, ymax * 0.97, "cliff",
            fontsize=7, va="top", color="grey")


def _traj_panel(ax, traj, title=""):
    """Draw all 4 types on one axes."""
    for k in ["A", "B", "C", "D"]:
        t, mean, std, n, frac = traj_arrays(traj, k)
        ls = "--" if k == "D" else "-"
        lw = 1.6  if k == "D" else 2.0
        label = f"{TYPE_LABEL[k]}  ({frac*100:.0f}%,  n={n})"
        _band(ax, t, mean, std, TYPE_COLOR[k], label, lw=lw, ls=ls)

    _add_cliff(ax)
    ax.axhline(0.6, color="grey", ls=":", lw=0.7, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("$t$  (noise → clean)", fontsize=9)
    ax.set_ylabel(r"$\cos(\hat{x}_t,\; E_{\mathrm{gt}})$", fontsize=9)
    ax.set_title(title or "Token Trajectory Types", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left",
              framealpha=0.9, handlelength=1.4)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=8)


def _rescue_panel(ax, geo, dec, title=""):
    """Draw cos_nn_correct vs dec_top1 with rescue shading."""
    t, cos_nn, dec_top, mask = rescue_arrays(geo, dec)

    ax.plot(t[mask], cos_nn[mask], color="#1f77b4", lw=2.0,
            label="Denoiser NN accuracy  $\\cos_{\\mathrm{nn}}=\\mathrm{gt}$")
    ax.plot(t[mask], dec_top[mask], color="#ff7f0e", lw=2.0,
            label="Decoder head  top-1 accuracy")

    # rescue fill (only where decoder > cos_nn)
    rescue_lo = np.maximum(cos_nn, dec_top)   # upper edge of geo
    rescue_hi = np.maximum(cos_nn, dec_top)   # will use where dec > cos_nn
    # fill between cos_nn and dec_top where dec_top > cos_nn
    fill_mask = mask & (dec_top > cos_nn)
    ax.fill_between(t, cos_nn, dec_top,
                    where=fill_mask, color="#2ca02c", alpha=0.20,
                    label="Decoder rescue zone")

    # residual "truly lost" fill (above dec_top to 1.0)
    # start from cliff (t>=0.30): pre-cliff region is prior-dominated and misleading
    ax.fill_between(t, dec_top, 1.0,
                    where=mask & (t >= CLIFF), color="#d62728", alpha=0.08,
                    label="Truly wrong after decoder (~2%)")

    # annotate rescue at t=0.70
    idx70 = np.argmin(np.abs(t - 0.70))
    geo_wrong = 1.0 - cos_nn[idx70]
    rescued   = (dec_top[idx70] - cos_nn[idx70]) / geo_wrong
    ax.annotate(
        f"t=0.70: rescues\n~{rescued*100:.0f}% of geo-wrong",
        xy=(t[idx70], (cos_nn[idx70] + dec_top[idx70]) / 2),
        xytext=(0.50, 0.76),
        fontsize=7.5, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.9),
        bbox=dict(fc="white", ec="#2ca02c", alpha=0.8, lw=0.8, pad=2),
    )

    # mark t=1.0 artifact
    ax.scatter([t[-1]], [dec_top[-1]], color="#ff7f0e", zorder=5,
               marker="x", s=60, lw=1.5)
    ax.text(t[-1] - 0.02, dec_top[-1] - 0.04,
            "t=1 artifact\n(two-pass)", fontsize=6.5,
            ha="right", color="#ff7f0e", alpha=0.8)

    _add_cliff(ax)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("$t$  (noise → clean)", fontsize=9)
    ax.set_ylabel("Fraction of positions correct", fontsize=9)
    ax.set_title(title or "Decoder Rescue Effect", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left",
              framealpha=0.9, handlelength=1.4)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))


# ── Option 1: combined 1×2 ───────────────────────────────────────────────────

def fig_combined(traj, geo, dec, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("ELF-B Token Commitment Trajectories and Decoder Rescue",
                 fontsize=11, y=1.02)

    _traj_panel(axes[0], traj, title="(a) Per-position commitment trajectories")
    _rescue_panel(axes[1], geo, dec, title="(b) Decoder head correction")

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── Option 2A: compact 4-type overlay ────────────────────────────────────────

def fig_traj_only(traj, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _traj_panel(ax, traj,
                title="Token Trajectory Types in ELF-B\n"
                      "(per-position, cosine to correct token centroid)")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── Option 2B: decoder rescue standalone ─────────────────────────────────────

def fig_rescue_only(geo, dec, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _rescue_panel(ax, geo, dec,
                  title="Decoder Head Rescues Geometrically Wrong Positions\n"
                        "(ELF-B, OWT, 64 samples)")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    traj, geo, dec = load_all()
    FIGS.mkdir(parents=True, exist_ok=True)

    # Option 1
    fig_combined(traj, geo, dec, FIGS / "fig_combined_traj_rescue.png")

    # Option 2
    fig_traj_only(traj,     FIGS / "fig_traj_overlay.png")
    fig_rescue_only(geo, dec, FIGS / "fig_decoder_rescue.png")


if __name__ == "__main__":
    main()
