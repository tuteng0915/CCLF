"""
Generate the two-panel intro figure for the CCLF paper.

Left panel:  Commitment State (stacked area) — committed-correct / committed-wrong / uncommitted
Right panel: Decode-branch advantage G_dec(t) — dec top-1 minus lin top-1, excluding t=1.0

Usage:
    python experiments/analysis/plot_intro_figure.py \
        --v3_json  results/elf/probe_v3/anchor_probe_v3.json \
        --dec_json results/elf/probe_decode_v1/probe_decode_branch.json \
        --out      results/elf/figures/intro_figure.pdf
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── colour palette (colour-blind friendly) ───────────────────────────────────
C_CORRECT   = "#2ca02c"   # green
C_WRONG     = "#d62728"   # red
C_UNCOMMIT  = "#aec7e8"   # light blue / grey
C_GDEC      = "#ff7f0e"   # orange
C_SHADING   = "#fff3cd"   # light yellow for plateau shading

def load_v3(path):
    with open(path) as f:
        d = json.load(f)
    # probe_v3 is tau-keyed; use τ=1.0
    sub = d.get("1.0", d[list(d.keys())[0]])
    t   = np.array(sub["t"])
    cc  = np.array(sub["committed_correct_mean"])
    cw  = np.array(sub["committed_wrong_mean"])
    uc  = np.array(sub["uncommitted_mean"])
    cc_std = np.array(sub["committed_correct_std"])
    cw_std = np.array(sub["committed_wrong_std"])
    return t, cc, cw, uc, cc_std, cw_std

def load_dec(path):
    with open(path) as f:
        d = json.load(f)
    t    = np.array(d["t"])
    gdec = np.array(d["gap_top1_mean"])
    gdec_std = np.array(d["gap_top1_std"])
    lin  = np.array(d["lin_top1_gt_mean"])
    dec_ = np.array(d["dec_top1_gt_mean"])
    return t, gdec, gdec_std, lin, dec_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3_json",  default="results/elf/probe_v3/anchor_probe_v3.json")
    parser.add_argument("--dec_json", default="results/elf/probe_decode_v1/probe_decode_branch.json")
    parser.add_argument("--out",      default="results/elf/figures/intro_figure.pdf")
    args = parser.parse_args()

    t_v3, cc, cw, uc, cc_std, cw_std = load_v3(args.v3_json)
    t_dec, gdec, gdec_std, lin, dec_ = load_dec(args.dec_json)

    # Exclude t=1.0 from G_dec panel (two-pass artifact)
    mask = t_dec < 0.999
    t_dec_p  = t_dec[mask]
    gdec_p   = gdec[mask]
    gdec_std_p = gdec_std[mask]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.97, top=0.88, bottom=0.15)

    PHASE_ALPHA = 0.10

    # ── LEFT: Commitment State stacked area ───────────────────────────────────
    ax = axes[0]

    # Stacked area: bottom=uncommitted, middle=committed-wrong, top=committed-correct
    ax.stackplot(
        t_v3,
        uc, cw, cc,
        colors=[C_UNCOMMIT, C_WRONG, C_CORRECT],
        alpha=0.85,
        labels=["Uncommitted", "Committed-wrong", "Committed-correct"],
    )

    # Phase annotation bands
    ax.axvspan(0.00, 0.10, alpha=PHASE_ALPHA, color="grey",    zorder=0)
    ax.axvspan(0.10, 0.35, alpha=PHASE_ALPHA, color="blue",    zorder=0)
    ax.axvspan(0.35, 0.95, alpha=0.07,        color=C_WRONG,   zorder=0)
    ax.axvspan(0.95, 1.00, alpha=PHASE_ALPHA, color="purple",  zorder=0)

    # Phase labels
    ax.text(0.05,  1.03, "Prior",      ha="center", va="bottom", fontsize=6.5, color="grey",    transform=ax.get_xaxis_transform())
    ax.text(0.225, 1.03, "Cliff",      ha="center", va="bottom", fontsize=6.5, color="#3333cc", transform=ax.get_xaxis_transform())
    ax.text(0.65,  1.03, "Plateau",    ha="center", va="bottom", fontsize=6.5, color="#aa1111", transform=ax.get_xaxis_transform())
    ax.text(0.975, 1.03, "Final",      ha="center", va="bottom", fontsize=6.5, color="purple",  transform=ax.get_xaxis_transform())

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Diffusion time $t$  (noise $\\to$ clean)", fontsize=9)
    ax.set_ylabel("Fraction of positions", fontsize=9)
    ax.set_title("(a) Lexical commitment state in ELF", fontsize=9.5, pad=8)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis="both", labelsize=8)

    handles = [
        mpatches.Patch(color=C_CORRECT,  alpha=0.85, label="Committed-correct"),
        mpatches.Patch(color=C_WRONG,    alpha=0.85, label="Committed-wrong"),
        mpatches.Patch(color=C_UNCOMMIT, alpha=0.85, label="Uncommitted"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="center right",
              framealpha=0.9, handlelength=1.2)

    # ── RIGHT: G_dec(t) = dec top-1 minus lin top-1 ───────────────────────────
    ax = axes[1]

    ax.fill_between(t_dec_p,
                    gdec_p - gdec_std_p,
                    gdec_p + gdec_std_p,
                    color=C_GDEC, alpha=0.18)
    ax.plot(t_dec_p, gdec_p, color=C_GDEC, linewidth=2.0, label="$G_{\\mathrm{dec}}(t)$")

    # Zero baseline
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    # Plateau shading
    ax.axvspan(0.35, 0.95, alpha=0.07, color=C_WRONG, zorder=0, label="Stable-but-imperfect plateau")

    # Annotate peak
    peak_idx = np.argmax(gdec_p)
    ax.annotate(
        f"+{gdec_p[peak_idx]:.2f}",
        xy=(t_dec_p[peak_idx], gdec_p[peak_idx]),
        xytext=(t_dec_p[peak_idx] + 0.08, gdec_p[peak_idx] + 0.01),
        fontsize=7.5,
        color=C_GDEC,
        arrowprops=dict(arrowstyle="->", color=C_GDEC, lw=1.0),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 0.35)
    ax.set_xlabel("Diffusion time $t$  (noise $\\to$ clean)", fontsize=9)
    ax.set_ylabel("$G_{\\mathrm{dec}}(t)$ = dec top-1 $-$ lin top-1", fontsize=9)
    ax.set_title("(b) Decode-branch advantage in ELF", fontsize=9.5, pad=8)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9, handlelength=1.5)

    # ── save ─────────────────────────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    # also save PNG for quick preview
    png_out = args.out.replace(".pdf", ".png")
    fig.savefig(png_out, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.out}")
    print(f"Saved: {png_out}")
    plt.close(fig)

if __name__ == "__main__":
    main()
