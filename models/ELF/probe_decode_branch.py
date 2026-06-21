"""
ELF Decode Branch Probe

Decomposes the "final revision spike" by comparing two readout paths:

  Probe A — p_lin(t):
    z_t → backbone(t, decode=True) → x̂_t^den + logits_lin
    [decode head forced at denoising time t]

  Probe B — p_dec(t):
    z_t → backbone(t, decode=False) → x̂_t^den
         → backbone(t=1.0, decode=True) on x̂_t^den → h_t^dec + logits_dec
    [decode head at t=1 on denoising output; simulates final decode step]

Key gaps (answer: is the 19% final spike from trajectory or decode branch?):
  G_dec(t)   = top1(p_dec) − top1(p_lin)   decode correction gap at each t
  G_traj     = top1(p_lin @ t=1) − top1(p_lin @ t=0.95)
  G_final    = top1(p_dec @ t=1) − top1(p_lin @ t=1)

Decode residual:
  c_t = h_t^dec − x̂_t^den
  Measures the geometric "correction" the decode branch applies.
  We probe: ||c_t||, correlation with w→c transitions, interpolation.

Interpolation probe (residual sanity check):
  x̃_t(γ) = (1−γ)·x̂_t^den + γ·h_t^dec,   γ ∈ {0, 0.25, 0.5, 0.75, 1.0}
  Readout: logits via factored decode head applied directly to x̃_t(γ)
  If top1_gt increases monotonically with γ → residual is a valid correction direction.

Usage:
    cd ~/ELF
    CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \\
    python probe_decode_branch.py \\
        --config src/configs/training_configs/train_owt_ELF-B.yml \\
        --checkpoint embedded-language-flows/ELF-B-owt \\
        --n_samples 64 --seq_len 256 --out_dir ~/probe_decode_v1

    # smoke test
    python probe_decode_branch.py --stub --out_dir ~/probe_decode_stub
"""

import sys, os, argparse, copy, json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ELF_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if ELF_SRC not in sys.path:
    sys.path.insert(0, ELF_SRC)

import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass
import jax.numpy as jnp
import optax


# ─────────────────────────────────────────────────────────────────────────────
# Primitive metrics
# ─────────────────────────────────────────────────────────────────────────────

def softmax_np(logits: np.ndarray, tau: float) -> np.ndarray:
    l = logits / tau
    l -= l.max(axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)


def token_entropy(p: np.ndarray) -> np.ndarray:
    """Per-position entropy. p: [L, V] → [L]"""
    pc = np.clip(p, 1e-9, 1.0)
    return -(pc * np.log(pc)).sum(axis=-1)


def top1_acc(p: np.ndarray, ref_ids: np.ndarray) -> float:
    return float((np.argmax(p, axis=-1) == ref_ids).mean())


def topk_acc(p: np.ndarray, ref_ids: np.ndarray, k: int) -> float:
    topk = np.argsort(p, axis=-1)[:, -k:]
    return float((topk == ref_ids[:, None]).any(axis=-1).mean())


def ce_loss(p: np.ndarray, gt_ids: np.ndarray) -> float:
    """Mean cross-entropy: −log p[gt] averaged over positions."""
    pc = np.clip(p, 1e-9, 1.0)
    return float(-np.log(pc[np.arange(len(gt_ids)), gt_ids]).mean())


def committed_fracs(p: np.ndarray, gt_ids: np.ndarray,
                    commit_thresh: float) -> tuple:
    """Returns (committed_correct, committed_wrong, uncommitted) fractions."""
    H = token_entropy(p)
    top1 = np.argmax(p, axis=-1)
    committed = H < commit_thresh
    correct   = top1 == gt_ids
    return (
        float((committed & correct).mean()),
        float((committed & ~correct).mean()),
        float((~committed).mean()),
    )


def transition_fracs(prev_correct: Optional[np.ndarray],
                     curr_correct: np.ndarray) -> tuple:
    """Returns (c2c, c2w, w2c, w2w) or (nan×4) if no previous step."""
    if prev_correct is None:
        return (float("nan"),) * 4
    return (
        float((prev_correct  &  curr_correct).mean()),
        float((prev_correct  & ~curr_correct).mean()),
        float((~prev_correct &  curr_correct).mean()),
        float((~prev_correct & ~curr_correct).mean()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Two-pass logit collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_logits_two_pass(
    fwd_fn,           # fwd_fn(z [1,L,d_in], t float, mask [1,L]) → (xhat [L,d], logits [L,V])
    clean_emb,        # [L, d]  T5 embedding of clean text
    attn_mask,        # [L]
    t_grid,           # [T]
    n_noise: int,
    seed: int,
    has_sc: bool,
) -> tuple:
    """
    For each (t, noise_seed):
      Pass 1: z_t → fwd(z_t, t)  → x̂_t^den, logits_lin
      Pass 2: x̂_t^den → fwd(x̂_t^den, 1.0) → h_t^dec, logits_dec

    Returns:
      xhat_arr      [T, N, L, d]   denoising output x̂_t^den
      logits_lin    [T, N, L, V]   decode head at denoising time t
      hdec_arr      [T, N, L, d]   backbone output at t=1 on x̂_t^den
      logits_dec    [T, N, L, V]   decode head at t=1 on x̂_t^den
    """
    rng = np.random.default_rng(seed)
    L, d = clean_emb.shape
    mask_batch = attn_mask[None]   # [1, L]
    T, N = len(t_grid), n_noise

    # Lazy init after first forward to get V
    xhat_arr   = np.zeros((T, N, L, d),  dtype=np.float32)
    hdec_arr   = np.zeros((T, N, L, d),  dtype=np.float32)
    logits_lin = None
    logits_dec = None

    for ti, t in enumerate(t_grid):
        for si in range(N):
            eps = rng.standard_normal((1, L, d)).astype(np.float32)
            z_t = t * clean_emb[None] + (1.0 - t) * eps  # [1, L, d]

            # Pass 1: denoising forward at time t (decode=True to get logits_lin)
            if has_sc:
                z_t_in = np.concatenate([z_t, np.zeros_like(z_t)], axis=-1)
            else:
                z_t_in = z_t
            x_hat, l_lin = fwd_fn(z_t_in, t, mask_batch)   # [L,d], [L,V]

            # Pass 2: decode forward at t=1 using x̂_t^den as input
            x_hat_in = x_hat[None]  # [1, L, d]
            if has_sc:
                x_hat_in = np.concatenate([x_hat_in, np.zeros_like(x_hat_in)], axis=-1)
            h_dec, l_dec = fwd_fn(x_hat_in, 1.0, mask_batch)  # [L,d], [L,V]

            if logits_lin is None:
                V = l_lin.shape[-1]
                logits_lin = np.zeros((T, N, L, V), dtype=np.float32)
                logits_dec = np.zeros((T, N, L, V), dtype=np.float32)

            xhat_arr[ti, si]   = x_hat
            hdec_arr[ti, si]   = h_dec
            logits_lin[ti, si] = l_lin
            logits_dec[ti, si] = l_dec

    return xhat_arr, logits_lin, hdec_arr, logits_dec


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence metric computation
# ─────────────────────────────────────────────────────────────────────────────

INTERP_GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]

def compute_seq_metrics(
    xhat_arr,     # [T, N, L, d]
    logits_lin,   # [T, N, L, V]
    hdec_arr,     # [T, N, L, d]
    logits_dec,   # [T, N, L, V]
    t_grid,       # [T]
    gt_ids,       # [L]
    final_ids,    # [L]   argmax of logits_dec at t=1, seed=0
    tau: float,
    topk: int,
    commit_thresh: float,
) -> dict:
    """Returns per-t scalar metrics for one sequence."""
    T, N, L, V = logits_lin.shape

    keys_per_branch = [
        "top1_gt", "topk_gt", "top1_final",
        "entropy_mean", "ce",
        "comm_correct", "comm_wrong", "uncommitted",
        "c2c", "c2w", "w2c", "w2w",
    ]
    out = {f"lin_{k}": [] for k in keys_per_branch}
    out.update({f"dec_{k}": [] for k in keys_per_branch})
    out.update({
        "t": t_grid.tolist(),
        "gap_top1":    [],   # dec_top1_gt − lin_top1_gt
        "gap_topk":    [],   # dec_topk_gt − lin_topk_gt
        "gap_ce":      [],   # lin_ce − dec_ce  (positive = dec better)
        "gap_entropy": [],   # lin_entropy − dec_entropy
        "residual_norm": [],         # mean ||h_dec − x̂||
        "residual_w2c_corr": [],     # fraction w2c positions with above-median residual norm
    })
    # Interpolation: top1_gt for each gamma, at each t
    for g in INTERP_GAMMAS:
        out[f"interp_{g}_top1_gt"] = []
        out[f"interp_{g}_topk_gt"] = []

    prev_correct_lin = None
    prev_correct_dec = None

    for ti in range(T):
        # Aggregate over noise seeds
        acc_lin = {k: [] for k in keys_per_branch}
        acc_dec = {k: [] for k in keys_per_branch}
        acc_gap_top1, acc_gap_topk, acc_gap_ce, acc_gap_H = [], [], [], []
        acc_res_norm = []
        acc_res_w2c  = []
        interp_acc = {g: {"top1": [], "topk": []} for g in INTERP_GAMMAS}

        top1_lin_seeds = np.zeros((N, L), dtype=np.int32)
        top1_dec_seeds = np.zeros((N, L), dtype=np.int32)

        for si in range(N):
            p_lin = softmax_np(logits_lin[ti, si], tau)   # [L, V]
            p_dec = softmax_np(logits_dec[ti, si], tau)   # [L, V]
            x_hat = xhat_arr[ti, si]                       # [L, d]
            h_dec = hdec_arr[ti, si]                       # [L, d]

            top1_lin = np.argmax(p_lin, axis=-1)
            top1_dec = np.argmax(p_dec, axis=-1)
            top1_lin_seeds[si] = top1_lin
            top1_dec_seeds[si] = top1_dec

            H_lin = token_entropy(p_lin)
            H_dec = token_entropy(p_dec)

            # -- Lin branch metrics --
            cc_lin, cw_lin, uc_lin = committed_fracs(p_lin, gt_ids, commit_thresh)
            acc_lin["top1_gt"].append(top1_acc(p_lin, gt_ids))
            acc_lin["topk_gt"].append(topk_acc(p_lin, gt_ids, topk))
            acc_lin["top1_final"].append(top1_acc(p_lin, final_ids))
            acc_lin["entropy_mean"].append(float(H_lin.mean()))
            acc_lin["ce"].append(ce_loss(p_lin, gt_ids))
            acc_lin["comm_correct"].append(cc_lin)
            acc_lin["comm_wrong"].append(cw_lin)
            acc_lin["uncommitted"].append(uc_lin)

            # -- Dec branch metrics --
            cc_dec, cw_dec, uc_dec = committed_fracs(p_dec, gt_ids, commit_thresh)
            acc_dec["top1_gt"].append(top1_acc(p_dec, gt_ids))
            acc_dec["topk_gt"].append(topk_acc(p_dec, gt_ids, topk))
            acc_dec["top1_final"].append(top1_acc(p_dec, final_ids))
            acc_dec["entropy_mean"].append(float(H_dec.mean()))
            acc_dec["ce"].append(ce_loss(p_dec, gt_ids))
            acc_dec["comm_correct"].append(cc_dec)
            acc_dec["comm_wrong"].append(cw_dec)
            acc_dec["uncommitted"].append(uc_dec)

            # -- Gaps --
            acc_gap_top1.append(top1_acc(p_dec, gt_ids) - top1_acc(p_lin, gt_ids))
            acc_gap_topk.append(topk_acc(p_dec, gt_ids, topk) - topk_acc(p_lin, gt_ids, topk))
            acc_gap_ce.append(ce_loss(p_lin, gt_ids) - ce_loss(p_dec, gt_ids))
            acc_gap_H.append(float(H_lin.mean()) - float(H_dec.mean()))

            # -- Decode residual --
            c_t = h_dec - x_hat                          # [L, d]
            res_norm = np.linalg.norm(c_t, axis=-1)      # [L]
            acc_res_norm.append(float(res_norm.mean()))

            # Fraction of w→c positions that have above-median residual norm
            # (sanity: does larger residual predict correction?)
            curr_correct_lin = top1_lin == gt_ids         # [L]
            if prev_correct_lin is not None:
                w2c_mask = (~prev_correct_lin) & curr_correct_lin   # [L]
                if w2c_mask.sum() > 0:
                    median_norm = np.median(res_norm)
                    acc_res_w2c.append(float(res_norm[w2c_mask].mean() > median_norm))
                # also store transitions
                c2c, c2w, w2c, w2w = transition_fracs(prev_correct_lin, curr_correct_lin)
                acc_lin["c2c"].append(c2c); acc_lin["c2w"].append(c2w)
                acc_lin["w2c"].append(w2c); acc_lin["w2w"].append(w2w)
                curr_correct_dec = top1_dec == gt_ids
                c2c, c2w, w2c, w2w = transition_fracs(prev_correct_dec, curr_correct_dec)
                acc_dec["c2c"].append(c2c); acc_dec["c2w"].append(c2w)
                acc_dec["w2c"].append(w2c); acc_dec["w2w"].append(w2w)

            # -- Interpolation probe --
            for g in INTERP_GAMMAS:
                x_interp = (1 - g) * x_hat + g * h_dec   # [L, d]
                # Readout: apply factored decode head params indirectly via logit blend
                # (exact decode head needs param extraction; use logit interpolation as proxy)
                logits_interp = (1 - g) * logits_lin[ti, si] + g * logits_dec[ti, si]
                p_interp = softmax_np(logits_interp, tau)
                interp_acc[g]["top1"].append(top1_acc(p_interp, gt_ids))
                interp_acc[g]["topk"].append(topk_acc(p_interp, gt_ids, topk))

        # Update previous correct (majority vote over seeds)
        prev_correct_lin = (top1_lin_seeds == gt_ids[None]).mean(axis=0) > 0.5
        prev_correct_dec = (top1_dec_seeds == gt_ids[None]).mean(axis=0) > 0.5

        # Store aggregated metrics
        def mean_or_nan(lst):
            return float(np.mean(lst)) if lst else float("nan")

        for k in keys_per_branch:
            out[f"lin_{k}"].append(mean_or_nan(acc_lin[k]))
            out[f"dec_{k}"].append(mean_or_nan(acc_dec[k]))

        out["gap_top1"].append(mean_or_nan(acc_gap_top1))
        out["gap_topk"].append(mean_or_nan(acc_gap_topk))
        out["gap_ce"].append(mean_or_nan(acc_gap_ce))
        out["gap_entropy"].append(mean_or_nan(acc_gap_H))
        out["residual_norm"].append(mean_or_nan(acc_res_norm))
        out["residual_w2c_corr"].append(mean_or_nan(acc_res_w2c))

        for g in INTERP_GAMMAS:
            out[f"interp_{g}_top1_gt"].append(mean_or_nan(interp_acc[g]["top1"]))
            out[f"interp_{g}_topk_gt"].append(mean_or_nan(interp_acc[g]["topk"]))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across sequences
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(seq_results: list) -> dict:
    """seq_results: list of per-seq dicts. Returns mean ± std over sequences."""
    out = {"t": seq_results[0]["t"]}
    keys = [k for k in seq_results[0] if k != "t"]
    for k in keys:
        mat = np.array([s[k] for s in seq_results], dtype=np.float64)   # [S, T]
        out[f"{k}_mean"] = np.nanmean(mat, axis=0).tolist()
        out[f"{k}_std"]  = np.nanstd(mat,  axis=0).tolist()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Print summary + compute the 2×2 table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict, t_grid: np.ndarray):
    t_list = results["t"]

    def g(key, i):
        v = results[f"{key}_mean"][i]
        return "nan" if (v != v) else f"{v:.3f}"

    print(f"\n{'t':>5}  {'lin_top1':>8}  {'dec_top1':>8}  {'gap':>6}  "
          f"{'lin_ce':>7}  {'dec_ce':>7}  {'ce_gap':>7}  "
          f"{'res_norm':>8}  {'w2c_corr':>8}")
    print("-" * 75)
    for i, t in enumerate(t_list):
        print(f"{t:5.2f}  {g('lin_top1_gt',i):>8}  {g('dec_top1_gt',i):>8}  "
              f"{g('gap_top1',i):>6}  "
              f"{g('lin_ce',i):>7}  {g('dec_ce',i):>7}  {g('gap_ce',i):>7}  "
              f"{g('residual_norm',i):>8}  {g('residual_w2c_corr',i):>8}")

    # 2×2 table at t=0.95 and t=1.0
    def idx(t_val):
        return min(range(len(t_list)), key=lambda i: abs(t_list[i] - t_val))

    i95, i100 = idx(0.95), idx(1.0)
    print("\n── 2×2 Decomposition Table ──")
    print(f"{'':30s}  {'top1_gt':>8}  {'CE':>8}")
    print(f"{'x̂_0.95  + lin readout':30s}  {g('lin_top1_gt', i95):>8}  {g('lin_ce', i95):>8}")
    print(f"{'x̂_0.95  + dec branch':30s}  {g('dec_top1_gt', i95):>8}  {g('dec_ce', i95):>8}")
    print(f"{'x̂_1.00  + lin readout':30s}  {g('lin_top1_gt', i100):>8}  {g('lin_ce', i100):>8}")
    print(f"{'x̂_1.00  + dec branch  (ELF actual)':30s}  {g('dec_top1_gt', i100):>8}  {g('dec_ce', i100):>8}")

    g_traj  = results["lin_top1_gt_mean"][i100] - results["lin_top1_gt_mean"][i95]
    g_dec95 = results["gap_top1_mean"][i95]
    g_final = results["gap_top1_mean"][i100]
    print(f"\nG_traj  (lin: 0.95→1.0)    = {g_traj:+.3f}")
    print(f"G_dec   (dec−lin @ t=0.95) = {g_dec95:+.3f}")
    print(f"G_final (dec−lin @ t=1.00) = {g_final:+.3f}")

    print("\n── Interpolation Probe (top1_gt at t=0.95) ──")
    for gamma in INTERP_GAMMAS:
        v = results[f"interp_{gamma}_top1_gt_mean"][i95]
        print(f"  γ={gamma:.2f}  top1_gt={v:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_dir: str, label: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t = np.array(results["t"])

    def band(ax, key, color, ls="-", label=None):
        mean = np.array(results[f"{key}_mean"])
        std  = np.array(results[f"{key}_std"])
        mask = ~np.isnan(mean)
        ax.plot(t[mask], mean[mask], color=color, lw=2, ls=ls,
                label=label or key)
        ax.fill_between(t[mask], (mean-std)[mask], (mean+std)[mask],
                        alpha=0.15, color=color)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Panel 0: top1_gt — lin vs dec
    ax = axes[0]
    band(ax, "lin_top1_gt", "#3498db", label="lin (forced decode @ t)")
    band(ax, "dec_top1_gt", "#e74c3c", label="dec (backbone @ t=1 on x̂)")
    ax.set_title("Top-1 Accuracy vs GT\n(lin vs dec branch)")
    ax.set_ylim(0, 1.05); ax.set_xlabel("t"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 1: CE — lin vs dec
    ax = axes[1]
    band(ax, "lin_ce", "#3498db", label="lin CE")
    band(ax, "dec_ce", "#e74c3c", label="dec CE")
    ax.set_title("Cross-Entropy vs GT\n(↓ is better)")
    ax.set_xlabel("t"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: decode correction gap G_dec(t)
    ax = axes[2]
    band(ax, "gap_top1", "#27ae60", label="Δtop1 (dec − lin)")
    band(ax, "gap_ce",   "#8e44ad", ls="--", label="ΔCE (lin − dec, ↑=dec better)")
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.set_title("Decode Correction Gap G_dec(t)\n(> 0 means dec branch helps)")
    ax.set_xlabel("t"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 3: Entropy — lin vs dec
    ax = axes[3]
    band(ax, "lin_entropy_mean", "#3498db", label="lin entropy")
    band(ax, "dec_entropy_mean", "#e74c3c", label="dec entropy")
    ax.set_title("Token Entropy\n(lin vs dec)")
    ax.set_xlabel("t"); ax.set_ylabel("H [nats]"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 4: Decode residual norm + w2c correlation
    ax = axes[4]
    band(ax, "residual_norm", "#e67e22", label="||c_t|| = ||h_dec − x̂||")
    ax2 = ax.twinx()
    mean_w2c = np.array(results["residual_w2c_corr_mean"])
    mask = ~np.isnan(mean_w2c)
    ax2.plot(t[mask], mean_w2c[mask], color="#9b59b6", lw=1.5, ls="--",
             label="w→c frac with above-median ||c_t||")
    ax.set_title("Decode Residual c_t\n(norm + w→c correlation)")
    ax.set_xlabel("t"); ax.set_ylabel("||c_t||", color="#e67e22")
    ax2.set_ylabel("w→c above-median frac", color="#9b59b6")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 5: Interpolation at t=0.95
    ax = axes[5]
    t_list = results["t"]
    i95 = min(range(len(t_list)), key=lambda i: abs(t_list[i] - 0.95))
    top1_at_95 = [results[f"interp_{g}_top1_gt_mean"][i95] for g in INTERP_GAMMAS]
    topk_at_95 = [results[f"interp_{g}_topk_gt_mean"][i95] for g in INTERP_GAMMAS]
    ax.plot(INTERP_GAMMAS, top1_at_95, "o-", color="#e74c3c", lw=2, label="top-1 gt")
    ax.plot(INTERP_GAMMAS, topk_at_95, "s--", color="#3498db", lw=2, label="top-5 gt")
    ax.set_xlabel("γ  (0=x̂ only, 1=h_dec only)")
    ax.set_ylabel("top-k accuracy")
    ax.set_title("Interpolation along c_t @ t=0.95\n(does decode residual help?)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    fig.suptitle(f"Decode Branch Probe — {label}", fontsize=13)
    fig.tight_layout()
    path = str(Path(out_dir) / "probe_decode_branch.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ELF model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_elf(config_path: str, checkpoint_path: str, override_max_length: int = None):
    from modules.model import ELF_models
    from modules.t5_encoder import get_encoder
    from utils.checkpoint_utils import load_checkpoint, load_encoder_checkpoint
    from utils.train_utils import TrainState
    from configs.config import load_config_from_yaml
    from transformers import AutoTokenizer

    _orig_cwd = os.getcwd()
    os.chdir(ELF_SRC)
    abs_cfg = os.path.join(_orig_cwd, config_path) if not os.path.isabs(config_path) else config_path
    config = load_config_from_yaml(abs_cfg)
    os.chdir(_orig_cwd)

    if override_max_length and override_max_length != config.max_length:
        print(f"[elf] max_length {config.max_length} → {override_max_length}")
        config.max_length = override_max_length

    tokenizer   = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    enc_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)

    rng = jax.random.PRNGKey(42)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    d_enc = enc_config.d_model

    input_dim = 2 * d_enc if config.self_cond_prob > 0 else d_enc
    dummy_x   = jnp.ones((1, config.max_length, input_dim))
    dummy_t   = jnp.ones((1,))
    dummy_sc  = jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None

    model = ELF_models[config.model](
        text_encoder_dim=d_enc,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=tokenizer.vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    elf_params = model.init(init_rng, x=dummy_x, t=dummy_t,
                             deterministic=True, self_cond_cfg_scale=dummy_sc)
    optimizer = optax.adamw(learning_rate=1e-4)
    state = TrainState.create(
        apply_fn=model.apply, params=elf_params["params"], tx=optimizer,
        dropout_rng=dropout_rng, ema_params1=copy.deepcopy(elf_params["params"]),
    )
    state, step = load_checkpoint(checkpoint_path, state)
    print(f"[elf] checkpoint loaded (step {step})")
    return model, state.ema_params1, encoder_params, encoder_model, tokenizer, config, enc_config


# ─────────────────────────────────────────────────────────────────────────────
# Stub model
# ─────────────────────────────────────────────────────────────────────────────

class _StubModel:
    """Minimal stub: just random linear transforms so the pipeline runs."""
    def __init__(self, d=512, V=32128, seed=0):
        rng = np.random.default_rng(seed)
        self.W_den = rng.standard_normal((d, d)).astype(np.float32) * 0.05
        self.W_dec = rng.standard_normal((d, d)).astype(np.float32) * 0.05
        self.W_vocab = rng.standard_normal((d, V)).astype(np.float32) * 0.01

    def forward(self, z_in, t):
        z = z_in[..., :z_in.shape[-1]//2 if z_in.shape[-1] > 512 else z_in.shape[-1]]
        x_hat  = z[0] @ self.W_den
        logits = x_hat @ self.W_vocab
        return x_hat, logits


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_owt_texts(n: int) -> list:
    from datasets import load_dataset

    def _stream(name, **kw):
        ds = load_dataset(name, split="train", streaming=True, **kw)
        texts = []
        for ex in ds:
            t = ex["text"].strip()
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= n:
                break
        return texts[:n]

    for name, kw in [("Skylion007/openwebtext", {}),
                     ("stas/openwebtext-10k",   {}),
                     ("wikitext", {"name": "wikitext-103-raw-v1"})]:
        try:
            texts = _stream(name, **kw)
            if texts:
                print(f"[data] loaded from {name}")
                return texts
        except Exception as e:
            print(f"[data] {name} failed: {e}")
    raise RuntimeError("Could not load any text dataset.")


def encode_with_t5(texts, tokenizer, encoder_model, encoder_params,
                   seq_len, latent_mean, latent_std):
    results = []
    for text in texts:
        enc = tokenizer(text, return_tensors="np", truncation=True,
                        max_length=seq_len, padding="max_length")
        ids  = enc["input_ids"]
        mask = enc["attention_mask"]
        out  = encoder_model.apply(
            {"params": encoder_params},
            input_ids=ids, attention_mask=mask, deterministic=True,
        )
        emb = (np.array(out[0]) - latent_mean) / latent_std
        results.append((ids[0].astype(np.int32), emb, mask[0].astype(np.float32)))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        type=str, default=None)
    p.add_argument("--checkpoint",    type=str, default="embedded-language-flows/ELF-B-owt")
    p.add_argument("--stub",          action="store_true")
    p.add_argument("--n_samples",     type=int, default=64)
    p.add_argument("--seq_len",       type=int, default=256)
    p.add_argument("--n_t_steps",     type=int, default=21)
    p.add_argument("--tau",           type=float, default=1.0)
    p.add_argument("--topk",          type=int, default=5)
    p.add_argument("--n_noise",       type=int, default=4)
    p.add_argument("--commit_thresh", type=float, default=0.1)
    p.add_argument("--out_dir",       type=str, default="probe_decode_v1")
    return p.parse_args()


def main():
    args   = parse_args()
    t_grid = np.linspace(0.0, 1.0, args.n_t_steps)

    if args.stub:
        print("[mode] Stub")
        stub = _StubModel(d=512, V=32128)

        def fwd_fn(z_in, t, mask=None):
            return stub.forward(z_in, t)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(args.n_samples):
            ids  = rng.integers(0, 32128, size=args.seq_len).astype(np.int32)
            emb  = rng.standard_normal((args.seq_len, 512)).astype(np.float32)
            mask = np.ones(args.seq_len, dtype=np.float32)
            samples.append((ids, emb, mask))
        has_sc = False
        label  = "Stub"

    else:
        if args.config is None:
            raise ValueError("--config required")
        print(f"[mode] ELF: {args.checkpoint}")
        (model, ema_params, encoder_params, encoder_model,
         tokenizer, config, enc_config) = load_elf(
             args.config, args.checkpoint, override_max_length=args.seq_len)

        latent_mean = getattr(config, "latent_mean", 0.0)
        latent_std  = getattr(config, "latent_std",  1.0)
        has_sc      = config.self_cond_prob > 0
        has_sc_cfg  = config.num_self_cond_cfg_tokens > 0
        sc_scale    = jnp.zeros((1,)) if has_sc_cfg else None

        @jax.jit
        def _fwd(params, z_jax, t_jax, mask_jax, sc_jax):
            return model.apply(
                {"params": params}, z_jax, t_jax,
                attention_mask=mask_jax, deterministic=True,
                self_cond_cfg_scale=sc_jax,
                decoder_step_active=jnp.array(True),
            )

        def fwd_fn(z_in_np, t, attn_mask_np):
            x_hat_b, logits_b = _fwd(
                ema_params,
                jnp.array(z_in_np),
                jnp.array([t], dtype=jnp.float32),
                jnp.array(attn_mask_np, dtype=jnp.float32),
                sc_scale,
            )
            return np.array(x_hat_b[0]), np.array(logits_b[0])

        print(f"[data] loading {args.n_samples} OWT texts…")
        texts   = load_owt_texts(args.n_samples)
        print(f"[encode] T5 encoder…")
        samples = encode_with_t5(texts, tokenizer, encoder_model, encoder_params,
                                 args.seq_len, latent_mean, latent_std)
        label = "ELF-B OWT"

    # ── Probe ────────────────────────────────────────────────────────────────
    seq_results = []

    for i, (gt_ids, emb, mask) in enumerate(samples):
        print(f"\n── Seq {i+1}/{len(samples)} — two-pass logit collection…")
        xhat_arr, logits_lin, hdec_arr, logits_dec = collect_logits_two_pass(
            fwd_fn, emb, mask, t_grid, args.n_noise, seed=i, has_sc=has_sc)

        # final_ids: argmax of dec logits at t=1.0, seed=0 (ELF's actual output)
        final_ids = np.argmax(logits_dec[-1, 0], axis=-1).astype(np.int32)

        res = compute_seq_metrics(
            xhat_arr, logits_lin, hdec_arr, logits_dec,
            t_grid, gt_ids, final_ids,
            tau=args.tau, topk=args.topk, commit_thresh=args.commit_thresh,
        )

        # Print one-liner per t
        for ti, tv in enumerate(t_grid):
            print(f"  t={tv:.2f}  "
                  f"lin_top1={res['lin_top1_gt'][ti]:.3f}  "
                  f"dec_top1={res['dec_top1_gt'][ti]:.3f}  "
                  f"gap={res['gap_top1'][ti]:.3f}  "
                  f"res_norm={res['residual_norm'][ti]:.3f}")
        seq_results.append(res)

    # ── Aggregate + save + print + plot ──────────────────────────────────────
    agg = aggregate(seq_results)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(args.out_dir) / "probe_decode_branch.json"
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[save] → {json_path}")

    print_summary(agg, t_grid)
    plot_results(agg, args.out_dir, label=label)
    print("[done]")


if __name__ == "__main__":
    main()
