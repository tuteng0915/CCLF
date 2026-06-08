#!/usr/bin/env python3
"""Derive and validate the PDC mu(t) schedule from probe JSON outputs."""
import argparse
import json
import os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

# ELF: z_t = t·x + (1-t)·ε  →  SNR(t) = t² / (1-t)²
ELF_CLIFF_T = 0.25

# LangFlow: γ maps linearly  t=0 → γ_max, t=1 → γ_min
LF_GAMMA_MAX = 16.05   # t=0  (most noisy)
LF_GAMMA_MIN = 2.60    # t=1  (least noisy / most signal)
LF_CLIFF_T_LO = 0.70
LF_CLIFF_T_HI = 0.85

MU_T0 = 0.25   # before cliff → μ = 0
MU_T1 = 0.95   # last valid G_dec point
DEFAULT_DECODE_JSON = "results/elf/probe_decode_v1/probe_decode_branch.json"
DEFAULT_V4_JSON = "results/elf/probe_v4/anchor_probe_v4.json"
DEFAULT_OUT_DIR = "results/elf/pdc_schedule"


# ── SNR functions ──────────────────────────────────────────────────────────────

def snr_elf(t):
    """SNR for ELF: t²/(1-t)²"""
    t = np.asarray(t, dtype=float)
    return t**2 / (1.0 - t)**2


def snr_langflow(t):
    """SNR for LangFlow: exp(-γ(t)), γ linear from γ_max to γ_min"""
    t = np.asarray(t, dtype=float)
    gamma = LF_GAMMA_MAX + t * (LF_GAMMA_MIN - LF_GAMMA_MAX)
    return np.exp(-gamma)


# ── JSON loading + G_dec interpolation + μ(t) derivation ─────────────────────

def _as_float_array(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return arr


def load_gdec_from_decode_json(path):
    """Load G_dec(t)=dec_top1-lin_top1 from the decode-branch probe JSON."""
    with open(path) as f:
        dec = json.load(f)

    if isinstance(dec, dict) and "results" in dec:
        rows = [r for r in dec["results"] if float(r["t"]) < 1.0]
        if not rows:
            raise ValueError("decode JSON contains no t < 1.0 rows")
        t_vals = _as_float_array([r["t"] for r in rows], "t")
        if "dec_top1" in rows[0] and "lin_top1" in rows[0]:
            gdec = _as_float_array([r["dec_top1"] - r["lin_top1"] for r in rows], "G_dec")
        elif "dec_top1_gt_mean" in rows[0] and "lin_top1_gt_mean" in rows[0]:
            gdec = _as_float_array(
                [r["dec_top1_gt_mean"] - r["lin_top1_gt_mean"] for r in rows],
                "G_dec",
            )
        elif "gap_top1_mean" in rows[0]:
            gdec = _as_float_array([r["gap_top1_mean"] for r in rows], "G_dec")
        else:
            raise KeyError("decode JSON rows need dec/lin top1 fields or gap_top1_mean")
    elif isinstance(dec, dict) and "t" in dec:
        t_all = _as_float_array(dec["t"], "t")
        mask = t_all < 1.0
        t_vals = t_all[mask]
        if "dec_top1_gt_mean" in dec and "lin_top1_gt_mean" in dec:
            dec_top1 = _as_float_array(dec["dec_top1_gt_mean"], "dec_top1_gt_mean")
            lin_top1 = _as_float_array(dec["lin_top1_gt_mean"], "lin_top1_gt_mean")
            gdec = (dec_top1 - lin_top1)[mask]
        elif "gap_top1_mean" in dec:
            gdec = _as_float_array(dec["gap_top1_mean"], "gap_top1_mean")[mask]
        else:
            raise KeyError("decode JSON needs dec_top1_gt_mean/lin_top1_gt_mean or gap_top1_mean")
    else:
        raise ValueError("unsupported decode JSON schema")

    order = np.argsort(t_vals)
    return t_vals[order], gdec[order]


def load_fate_overlay(path, tau_key="1.0"):
    """Load probe-v4 decode-correction fate data for overlay validation."""
    with open(path) as f:
        v4 = json.load(f)

    if "fate" in v4:
        fate = v4["fate"]
    elif tau_key in v4 and "fate" in v4[tau_key]:
        fate = v4[tau_key]["fate"]
    else:
        tau_keys = [k for k, value in v4.items() if isinstance(value, dict) and "fate" in value]
        if not tau_keys:
            return None
        best_key = min(tau_keys, key=lambda k: abs(float(k) - float(tau_key)))
        fate = v4[best_key]["fate"]

    t_vals, corrected, corrected_std = [], [], []
    for t_key, sub in sorted(fate.items(), key=lambda item: float(item[0])):
        t_vals.append(float(t_key))
        corrected.append(float(sub.get("frac_decode_corrected_mean",
                                       sub.get("frac_decode_corrected", np.nan))))
        corrected_std.append(float(sub.get("frac_decode_corrected_std", 0.0)))

    return {
        "t": np.asarray(t_vals, dtype=float),
        "fate_corrected_by_decode": np.asarray(corrected, dtype=float),
        "fate_corrected_by_decode_std": np.asarray(corrected_std, dtype=float),
    }


def build_mu(t_grid, gdec_t, gdec_v):
    """Return μ(t) on t_grid: interpolate G_dec, clamp <0, normalize, zero before cliff."""
    interp = PchipInterpolator(gdec_t, gdec_v, extrapolate=False)
    gdec_raw = interp(t_grid)

    # NaN outside interpolation range → 0
    gdec_raw = np.where(np.isnan(gdec_raw), 0.0, gdec_raw)
    # Clamp negatives
    gdec_clamped = np.clip(gdec_raw, 0.0, None)
    # Zero outside the PDC active window.
    gdec_clamped[(t_grid < MU_T0) | (t_grid >= MU_T1)] = 0.0

    # Normalize to [0, 1]
    vmax = gdec_clamped.max()
    mu = gdec_clamped / vmax if vmax > 0 else gdec_clamped
    return mu, gdec_raw, gdec_clamped, vmax


# ── Analytical beta-shaped fit ────────────────────────────────────────────────

def beta_shape(t, A, alpha, beta):
    """A · (t-t0)^α · (t1-t)^β, defined on [t0, t1]."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = (t >= MU_T0) & (t <= MU_T1)
    dt0 = t[mask] - MU_T0
    dt1 = MU_T1 - t[mask]
    # guard against 0^negative
    dt0 = np.clip(dt0, 1e-9, None)
    dt1 = np.clip(dt1, 1e-9, None)
    out[mask] = A * dt0**alpha * dt1**beta
    return out


def fit_beta(t_grid, mu):
    """Fit beta_shape to μ(t) on the active region."""
    mask = (t_grid >= MU_T0) & (t_grid <= MU_T1)
    t_fit = t_grid[mask]
    mu_fit = mu[mask]
    p0 = [1.0, 0.5, 0.5]
    bounds = ([0, 0, 0], [100, 10, 10])
    try:
        popt, _ = curve_fit(beta_shape, t_fit, mu_fit, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        popt = p0
    return popt


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_snr(t_grid, out_dir):
    snr_e = snr_elf(t_grid)
    snr_l = snr_langflow(t_grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t_grid, snr_e, color="steelblue", lw=2, label="ELF  SNR(t) = t²/(1−t)²")
    ax.semilogy(t_grid, snr_l, color="tomato",    lw=2, label="LangFlow  SNR(t) = exp(−γ(t))")

    # ELF cliff
    ax.axvline(ELF_CLIFF_T, color="steelblue", ls="--", lw=1.2,
               label=f"ELF cliff  t={ELF_CLIFF_T:.2f}  SNR={snr_elf(ELF_CLIFF_T):.3f}")
    # LangFlow cliff band
    ax.axvspan(LF_CLIFF_T_LO, LF_CLIFF_T_HI, alpha=0.15, color="tomato",
               label=f"LF cliff  t=[{LF_CLIFF_T_LO},{LF_CLIFF_T_HI}]"
                     f"  SNR=[{snr_langflow(LF_CLIFF_T_HI):.4f},{snr_langflow(LF_CLIFF_T_LO):.4f}]")

    ax.set_xlabel("Diffusion time t  (0=noise, 1=clean)")
    ax.set_ylabel("SNR  (log scale)")
    ax.set_title("SNR Curves: ELF vs LangFlow")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-8, 1e4)

    fig.tight_layout()
    path = out_dir / "snr_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_gdec_mu(t_grid, gdec_t, gdec_v, mu, gdec_clamped, popt, fate, out_dir):
    mu_analytical = beta_shape(t_grid, *popt)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # ── top: G_dec raw + interpolated ────────────────────────────────────────
    ax = axes[0]
    ax.scatter(gdec_t, gdec_v, color="black", zorder=5, label="G_dec data (raw)")
    ax.plot(t_grid, gdec_clamped, color="darkorange", lw=2, label="G_dec interpolated (clamped)")
    ax.axvline(MU_T0, color="gray", ls=":", lw=1)
    ax.axvline(MU_T1, color="gray", ls=":", lw=1, label=f"active window [{MU_T0},{MU_T1}]")
    ax.set_ylabel("G_dec(t)")
    ax.set_title("G_dec(t) — decode-branch advantage")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── bottom: μ(t) normalized + analytical fit ─────────────────────────────
    ax = axes[1]
    ax.plot(t_grid, mu, color="mediumseagreen", lw=2.5, label="μ(t) normalized")
    ax.plot(t_grid, mu_analytical, color="purple", lw=1.5, ls="--",
            label=f"β-fit  A={popt[0]:.3f}  α={popt[1]:.3f}  β={popt[2]:.3f}")
    ax.axvline(MU_T0, color="gray", ls=":", lw=1)
    ax.axvline(MU_T1, color="gray", ls=":", lw=1)
    ax.set_xlabel("Diffusion time t")
    ax.set_ylabel("μ(t)")
    ax.set_title("μ(t) schedule for Progressive Decode Correction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)

    # ── bottom: fate-tracking validation overlay ────────────────────────────
    ax = axes[2]
    ax.plot(t_grid, mu, color="mediumseagreen", lw=2.0, label="μ(t) normalized")
    if fate is not None:
        ax.errorbar(
            fate["t"],
            fate["fate_corrected_by_decode"],
            yerr=fate["fate_corrected_by_decode_std"],
            color="crimson",
            marker="o",
            lw=1.8,
            capsize=3,
            label="fate corrected by decode",
        )
    ax.axvline(MU_T0, color="gray", ls=":", lw=1)
    ax.axvline(MU_T1, color="gray", ls=":", lw=1)
    ax.set_xlabel("Diffusion time t")
    ax.set_ylabel("normalized weight / fraction")
    ax.set_title("Fate-tracking validation overlay")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    path = out_dir / "pdc_schedule_plot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ── Key numbers ───────────────────────────────────────────────────────────────

def key_numbers(t_grid, mu):
    elf_cliff_snr = float(snr_elf(ELF_CLIFF_T))
    lf_cliff_snr_lo = float(snr_langflow(LF_CLIFF_T_HI))   # higher t → lower SNR
    lf_cliff_snr_hi = float(snr_langflow(LF_CLIFF_T_LO))
    snr_ratio_lo = elf_cliff_snr / lf_cliff_snr_hi
    snr_ratio_hi = elf_cliff_snr / lf_cliff_snr_lo

    def mu_at(t_val):
        idx = np.argmin(np.abs(t_grid - t_val))
        return float(mu[idx])

    nums = {
        "elf_cliff_t": ELF_CLIFF_T,
        "elf_cliff_snr": elf_cliff_snr,
        "lf_cliff_t_range": [LF_CLIFF_T_LO, LF_CLIFF_T_HI],
        "lf_cliff_snr_range": [lf_cliff_snr_lo, lf_cliff_snr_hi],
        "snr_ratio_elf_over_lf": [snr_ratio_lo, snr_ratio_hi],
        "mu_at_0.25": mu_at(0.25),
        "mu_at_0.30": mu_at(0.30),
        "mu_at_0.50": mu_at(0.50),
        "mu_at_0.75": mu_at(0.75),
        "mu_at_0.95": mu_at(0.95),
    }
    return nums


def print_snr_table(t_grid):
    checkpoints = [0.00, 0.10, 0.20, 0.25, 0.30, 0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    print("\n── SNR comparison table ──────────────────────────────────────────")
    print(f"{'t':>6}  {'SNR_ELF':>12}  {'SNR_LF':>12}  {'ratio ELF/LF':>14}")
    print("-" * 52)
    for t in checkpoints:
        se = snr_elf(t) if t < 1.0 else float("inf")
        sl = snr_langflow(t)
        ratio = se / sl if sl > 0 and t < 1.0 else float("inf")
        print(f"{t:>6.2f}  {se:>12.5f}  {sl:>12.6f}  {ratio:>14.2f}")
    print()


def print_key_numbers(nums, popt):
    print("── Key numbers ───────────────────────────────────────────────────")
    print(f"  ELF cliff:       t={nums['elf_cliff_t']:.2f}  SNR={nums['elf_cliff_snr']:.4f}")
    print(f"  LF  cliff:       t=[{nums['lf_cliff_t_range'][0]},{nums['lf_cliff_t_range'][1]}]"
          f"  SNR=[{nums['lf_cliff_snr_range'][0]:.5f}, {nums['lf_cliff_snr_range'][1]:.5f}]")
    lo, hi = nums["snr_ratio_elf_over_lf"]
    print(f"  SNR ratio (ELF/LF at cliff):  {lo:.1f}×–{hi:.1f}×")
    print()
    print("  μ(t) key values:")
    for k, v in nums.items():
        if k.startswith("mu_at_"):
            print(f"    {k}: {v:.4f}")
    print()
    print(f"  β-fit:  A={popt[0]:.4f}  α={popt[1]:.4f}  β={popt[2]:.4f}")
    span = MU_T1 - MU_T0
    print(f"  analytical form:  μ(t) = {popt[0]:.4f}·(t-{MU_T0})^{popt[1]:.4f}·({MU_T1}-t)^{popt[2]:.4f}")
    print(f"  (for t ∈ [{MU_T0}, {MU_T1}], else 0)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PDC μ(t) derivation from probe JSON")
    parser.add_argument("--decode_json", default=DEFAULT_DECODE_JSON,
                        help="path to probe_decode_branch.json")
    parser.add_argument("--v4_json", default=DEFAULT_V4_JSON,
                        help="path to anchor_probe_v4.json")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid = np.linspace(0.0, 1.0, 1001)

    print("\n=== PDC μ(t) Schedule ===")
    gdec_t, gdec_v = load_gdec_from_decode_json(args.decode_json)
    fate = load_fate_overlay(args.v4_json)
    print(f"  loaded {len(gdec_t)} G_dec points from {args.decode_json}")
    if fate is not None:
        print(f"  loaded {len(fate['t'])} fate-overlay points from {args.v4_json}")
    else:
        print(f"  no fate overlay found in {args.v4_json}")

    mu, gdec_raw, gdec_clamped, gdec_max = build_mu(t_grid, gdec_t, gdec_v)
    print(f"  G_dec peak (before normalization): {gdec_max:.4f}")

    popt = fit_beta(t_grid, mu)
    mu_formula = beta_shape(t_grid, *popt)
    plot_gdec_mu(t_grid, gdec_t, gdec_v, mu, gdec_clamped, popt, fate, out_dir)

    nums = key_numbers(t_grid, mu)
    nums["beta_fit"] = {"A": float(popt[0]), "alpha": float(popt[1]), "beta": float(popt[2]),
                        "t0": MU_T0, "t1": MU_T1}
    nums["gdec_peak_unnormalized"] = float(gdec_max)

    print_key_numbers(nums, popt)

    schedule = {
        "t": t_grid.tolist(),
        "gdec_raw": gdec_raw.tolist(),
        "gdec_clamped": gdec_clamped.tolist(),
        "mu_t": mu.tolist(),
        "mu_formula": mu_formula.tolist(),
        "fit": nums["beta_fit"],
        "gdec_source": {
            "t": gdec_t.tolist(),
            "gdec": gdec_v.tolist(),
        },
    }
    if fate is not None:
        schedule["fate_overlay"] = {
            "t": fate["t"].tolist(),
            "fate_corrected_by_decode": fate["fate_corrected_by_decode"].tolist(),
            "fate_corrected_by_decode_std": fate["fate_corrected_by_decode_std"].tolist(),
        }

    json_path = out_dir / "mu_t_schedule.json"
    with open(json_path, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"  saved {json_path}")
    print(f"\nAll outputs written to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
