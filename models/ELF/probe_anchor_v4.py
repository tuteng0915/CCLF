"""
Anchor Emergence Probing for ELF — v4: Commitment Dynamics Edition.

Four new metric groups over v3:

  1. Residual norm distribution  ||r_t|| = ||x̂_t - p_t @ E||
       — p10 / p50 / p90 across positions (v3 only had mean)
       — Note: space mismatch in ELF (x̂_t in T5 contextual, E in T5 input embedding).
         LangFlow probe gives the clean version; here the TREND is still informative.

  2. Temporal stability JSD  JSD(p_t, p_{t-1}) per position
       — p10 / p50 / p90 split exposes "already stable" vs "still volatile" positions.
       — p10 curve = already-committed stable positions.
       — p90 curve = most volatile / uncertain positions.

  3. Wrong commitment fate tracking  (τ=1.0 only)
       At source t ∈ {0.25, 0.30, 0.40, 0.50}: for positions committed_wrong,
       track their fate through the remainder of the trajectory:
         frac_traj_corrected  : ever becomes correct at t' ∈ (src_t, 0.95] (trajectory self-fixes)
         frac_decode_corrected: still wrong at t=0.95 but correct at t=1.0 (decode branch fixes)
         frac_stays_wrong     : still wrong at t=1.0

  4. Entropy bimodality
       — 3-bin fractions: h_frac_low (H<0.1), h_frac_mid (0.1≤H<0.5), h_frac_high (H≥0.5)
       — bimodality_coeff BC = (skew²+1) / (excess_kurt + 3*(n-1)²/((n-2)*(n-3)))
         BC > 5/9 ≈ 0.555 suggests bimodality; Gaussian has BC = 1/3.

Usage:
    cd ~/ELF
    CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \\
    python probe_anchor_v4.py \\
        --config src/configs/training_configs/train_owt_ELF-B.yml \\
        --checkpoint embedded-language-flows/ELF-B-owt \\
        --n_samples 64 --seq_len 256 --out_dir ~/probe_results_v4

    # smoke test:
    python probe_anchor_v4.py --stub --out_dir ~/probe_stub_v4
"""

import sys, os, argparse, copy, json
from pathlib import Path
from typing import Optional, List

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
# 1. Primitive helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_p(logits: np.ndarray, tau: float) -> np.ndarray:
    """softmax(logits / tau).  [L, V] → [L, V]"""
    l = logits / tau
    l -= l.max(axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)


def topk_recovery(p: np.ndarray, ref_ids: np.ndarray, k: int) -> float:
    topk = np.argsort(p, axis=-1)[:, -k:]
    return float((topk == ref_ids[:, None]).any(axis=-1).mean())


def jsd_scalar(p_prev: Optional[np.ndarray], p_curr: np.ndarray) -> Optional[float]:
    """Mean JSD over positions."""
    if p_prev is None:
        return None
    pp = np.clip(p_prev, 1e-9, 1.0)
    pc = np.clip(p_curr, 1e-9, 1.0)
    m  = 0.5 * (pp + pc)
    return float((0.5 * ((pp * (np.log(pp) - np.log(m))).sum(-1)
                        + (pc * (np.log(pc) - np.log(m))).sum(-1))).mean())


def jsd_per_pos(p_prev: np.ndarray, p_curr: np.ndarray) -> np.ndarray:
    """JSD per position.  Returns [L] array."""
    pp = np.clip(p_prev, 1e-9, 1.0)
    pc = np.clip(p_curr, 1e-9, 1.0)
    m  = 0.5 * (pp + pc)
    return 0.5 * ((pp * (np.log(pp) - np.log(m))).sum(-1)
                + (pc * (np.log(pc) - np.log(m))).sum(-1))   # [L]


def geometric_anchoring(x_hat, p, E, E_sq):
    """(d_soft, d_nn, margin) averaged over positions."""
    soft_anchor = p @ E                                           # [L, d]
    d_soft = float(np.linalg.norm(x_hat - soft_anchor, axis=-1).mean())
    x_sq   = (x_hat ** 2).sum(-1)                                # [L]
    dists_sq = x_sq[:, None] + E_sq[None, :] - 2.0 * (x_hat @ E.T)
    dists_sq = np.maximum(dists_sq, 0.0)
    idx12  = np.argpartition(dists_sq, 1, axis=-1)[:, :2]
    d1_sq  = dists_sq[np.arange(len(x_hat)), idx12[:, 0]]
    d2_sq  = dists_sq[np.arange(len(x_hat)), idx12[:, 1]]
    d_nn   = float(np.sqrt(d1_sq).mean())
    margin = float((np.sqrt(d2_sq) - np.sqrt(d1_sq)).mean())
    return d_soft, d_nn, margin


def bimodality_coefficient(H: np.ndarray) -> float:
    """
    BC = (skew² + 1) / (excess_kurt + 3*(n-1)²/((n-2)*(n-3))).
    BC > 5/9 ≈ 0.555 indicates bimodality. Gaussian: BC = 1/3.
    """
    n = len(H)
    if n < 4:
        return float("nan")
    mu    = H.mean()
    sigma = H.std()
    if sigma < 1e-12:
        return float("nan")
    z     = (H - mu) / sigma
    skew  = float((z ** 3).mean())
    ekurt = float((z ** 4).mean() - 3.0)    # excess kurtosis
    denom = ekurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if abs(denom) < 1e-12:
        return float("nan")
    return float((skew ** 2 + 1.0) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logit collection (same as v3)
# ─────────────────────────────────────────────────────────────────────────────

def collect_logits(forward_fn, clean_emb, attn_mask, t_grid, n_noise, seed):
    """Returns logits_arr [T, N, L, V], xhat_arr [T, N, L, d]."""
    rng = np.random.default_rng(seed)
    L, d = clean_emb.shape
    mask_batch = attn_mask[None]
    T, N = len(t_grid), n_noise

    logits_arr = None
    xhat_arr   = np.zeros((T, N, L, d), dtype=np.float32)

    for ti, t in enumerate(t_grid):
        for si in range(N):
            eps = rng.standard_normal((1, L, d)).astype(np.float32)
            z_t = t * clean_emb[None] + (1.0 - t) * eps
            x_hat, logits = forward_fn(z_t, float(t), mask_batch)
            if logits_arr is None:
                logits_arr = np.zeros((T, N, L, logits.shape[-1]), dtype=np.float32)
            logits_arr[ti, si] = logits
            xhat_arr[ti, si]   = x_hat

    return logits_arr, xhat_arr


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metric names
# ─────────────────────────────────────────────────────────────────────────────

# All per-t metrics (v3 carry-overs + v4 new)
ALL_PER_T_METRICS = [
    # geometric (Conjecture 5)
    "d_soft_anchor", "d_nn", "margin",
    # residual norm distribution (NEW in v4)
    "residual_norm_p10", "residual_norm_p50", "residual_norm_p90",
    # entropy
    "entropy", "entropy_p10", "entropy_p50", "entropy_p90",
    # entropy bimodality (NEW in v4)
    "h_frac_low", "h_frac_mid", "h_frac_high", "bimodality_coeff",
    # recovery
    "topk_gt", "top1_gt", "top1_final",
    # commitment state
    "committed_correct", "committed_wrong", "uncommitted",
    # inter-seed consistency
    "noise_agree",
    # revision (scalar JSD — kept for backward compat)
    "rev_top1", "rev_jsd",
    # temporal stability distribution (NEW in v4)
    "stab_jsd_p10", "stab_jsd_p50", "stab_jsd_p90",
    # transition matrix
    "c2c", "c2w", "w2c", "w2w",
]

SCALAR_METRICS = ["commit_time_mean", "commit_time_std", "never_committed_frac"]

# Fate tracking source timesteps  (only at tau=1.0)
FATE_SOURCE_TS = [0.25, 0.30, 0.40, 0.50]
FATE_METRICS   = ["n_committed_wrong",
                  "frac_traj_corrected",
                  "frac_decode_corrected",
                  "frac_stays_wrong"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_for_tau(
    logits_arr:    np.ndarray,    # [T, N, L, V]
    xhat_arr:      np.ndarray,    # [T, N, L, d]
    E:             np.ndarray,    # [V, d]
    t_grid:        np.ndarray,    # [T]
    gt_ids:        np.ndarray,    # [L]
    final_ids:     np.ndarray,    # [L]
    tau:           float,
    topk:          int,
    commit_thresh: float,
    do_fate:       bool = False,  # only True when tau==1.0
    h_bin_lo:      float = 0.10,
    h_bin_hi:      float = 0.50,
) -> dict:
    """Compute all per-t and scalar metrics for one tau value."""
    T, N, L, V = logits_arr.shape

    out = {k: [] for k in ALL_PER_T_METRICS}
    out["t"] = t_grid.tolist()

    E_sq        = (E ** 2).sum(-1)           # [V]
    prev_p      = [None] * N                  # per-seed previous p
    prev_correct = None                        # bool [L], majority across seeds

    # Trajectory state arrays for fate tracking (majority vote)
    # correct_traj[ti] = fraction-of-seeds that are correct at that timestep
    correct_traj = np.zeros((T, L), dtype=np.float32) if do_fate else None
    cw_traj      = np.zeros((T, L), dtype=np.float32) if do_fate else None  # committed_wrong

    H_trajectory = np.zeros((T, N, L), dtype=np.float32)

    for ti in range(T):
        # ── per-seed accumulators ────────────────────────────────────────────
        a_dsoft, a_dnn, a_margin = [], [], []
        a_res_norms = []          # list of [L] arrays
        a_stab_jsd  = []          # list of [L] arrays (when prev_p available)
        a_top1gt, a_top1fin = [], []
        a_topkgt = []
        a_commcorr, a_commwrong, a_uncommit = [], [], []
        a_rev1, a_jsd = [], []

        top1_seeds   = np.zeros((N, L), dtype=np.int32)
        H_seeds      = np.zeros((N, L), dtype=np.float32)
        correct_seeds = np.zeros((N, L), dtype=bool)
        cw_seeds      = np.zeros((N, L), dtype=bool)

        for si in range(N):
            logits = logits_arr[ti, si]        # [L, V]
            x_hat  = xhat_arr[ti, si]          # [L, d]
            p      = compute_p(logits, tau)    # [L, V]
            top1   = np.argmax(p, axis=-1)     # [L]
            pc     = np.clip(p, 1e-9, 1.0)
            H_i    = -(pc * np.log(pc)).sum(-1)  # [L]

            top1_seeds[si]   = top1
            H_seeds[si]      = H_i
            correct_seeds[si] = (top1 == gt_ids)

            # Geometric anchoring + residual norm
            d_soft, d_nn, margin = geometric_anchoring(x_hat, p, E, E_sq)
            a_dsoft.append(d_soft)
            a_dnn.append(d_nn)
            a_margin.append(margin)

            # Residual per position (WARNING: space mismatch in ELF, proxy only)
            soft_anchor = p @ E                              # [L, d]
            r_t  = x_hat - soft_anchor                      # [L, d]
            a_res_norms.append(np.linalg.norm(r_t, axis=-1))  # [L]

            # Temporal stability JSD per position
            if prev_p[si] is not None:
                a_stab_jsd.append(jsd_per_pos(prev_p[si], p))  # [L]
            # Scalar JSD + top-1 revision
            a_jsd.append(jsd_scalar(prev_p[si], p))
            if prev_p[si] is not None:
                a_rev1.append(float((np.argmax(prev_p[si], -1) != top1).mean()))
            prev_p[si] = p

            # Recovery
            a_topkgt.append(topk_recovery(p, gt_ids, topk))
            a_top1gt.append(float((top1 == gt_ids).mean()))
            a_top1fin.append(float((top1 == final_ids).mean()))

            # Commitment state
            committed = H_i < commit_thresh
            correct   = top1 == gt_ids
            a_commcorr.append(float((committed & correct).mean()))
            a_commwrong.append(float((committed & ~correct).mean()))
            a_uncommit.append(float((~committed).mean()))
            cw_seeds[si] = committed & ~correct

        H_trajectory[ti] = H_seeds

        # Fate state tracking (majority vote across seeds)
        if do_fate:
            correct_traj[ti] = correct_seeds.mean(axis=0)    # [L]
            cw_traj[ti]      = cw_seeds.mean(axis=0)         # [L]

        # Noise agreement
        noise_agree = float((top1_seeds == top1_seeds[0:1]).all(axis=0).mean())

        # Transition matrix
        curr_correct = (top1_seeds == gt_ids[None]).mean(axis=0) > 0.5  # [L]

        # ── aggregate per-seed → per-t ───────────────────────────────────────
        out["d_soft_anchor"].append(float(np.mean(a_dsoft)))
        out["d_nn"].append(float(np.mean(a_dnn)))
        out["margin"].append(float(np.mean(a_margin)))

        # Residual norm distribution (stack seeds → [N, L] then flatten)
        res_flat = np.concatenate(a_res_norms)          # [N*L]
        out["residual_norm_p10"].append(float(np.percentile(res_flat, 10)))
        out["residual_norm_p50"].append(float(np.percentile(res_flat, 50)))
        out["residual_norm_p90"].append(float(np.percentile(res_flat, 90)))

        # Entropy
        H_flat = H_seeds.flatten()                       # [N*L]
        out["entropy"].append(float(H_flat.mean()))
        out["entropy_p10"].append(float(np.percentile(H_flat, 10)))
        out["entropy_p50"].append(float(np.percentile(H_flat, 50)))
        out["entropy_p90"].append(float(np.percentile(H_flat, 90)))

        # Entropy bimodality
        lo, hi = h_bin_lo, h_bin_hi
        out["h_frac_low"].append(float((H_flat < lo).mean()))
        out["h_frac_mid"].append(float(((H_flat >= lo) & (H_flat < hi)).mean()))
        out["h_frac_high"].append(float((H_flat >= hi).mean()))
        out["bimodality_coeff"].append(bimodality_coefficient(H_flat))

        # Recovery
        out["topk_gt"].append(float(np.mean(a_topkgt)))
        out["top1_gt"].append(float(np.mean(a_top1gt)))
        out["top1_final"].append(float(np.mean(a_top1fin)))

        # Commitment state
        out["committed_correct"].append(float(np.mean(a_commcorr)))
        out["committed_wrong"].append(float(np.mean(a_commwrong)))
        out["uncommitted"].append(float(np.mean(a_uncommit)))

        # Noise agreement
        out["noise_agree"].append(noise_agree)

        # Revision (scalar)
        out["rev_top1"].append(float(np.mean(a_rev1)) if a_rev1 else float("nan"))
        valid_jsd = [x for x in a_jsd if x is not None]
        out["rev_jsd"].append(float(np.mean(valid_jsd)) if valid_jsd else float("nan"))

        # Temporal stability JSD distribution
        if a_stab_jsd:
            stab_flat = np.concatenate(a_stab_jsd)      # [N*L]
            out["stab_jsd_p10"].append(float(np.percentile(stab_flat, 10)))
            out["stab_jsd_p50"].append(float(np.percentile(stab_flat, 50)))
            out["stab_jsd_p90"].append(float(np.percentile(stab_flat, 90)))
        else:
            out["stab_jsd_p10"].append(float("nan"))
            out["stab_jsd_p50"].append(float("nan"))
            out["stab_jsd_p90"].append(float("nan"))

        # Transition matrix
        if prev_correct is not None:
            out["c2c"].append(float((prev_correct  &  curr_correct).mean()))
            out["c2w"].append(float((prev_correct  & ~curr_correct).mean()))
            out["w2c"].append(float((~prev_correct &  curr_correct).mean()))
            out["w2w"].append(float((~prev_correct & ~curr_correct).mean()))
        else:
            for k in ["c2c", "c2w", "w2c", "w2w"]:
                out[k].append(float("nan"))
        prev_correct = curr_correct

    # ── Commitment time ──────────────────────────────────────────────────────
    commit_mask  = H_trajectory < commit_thresh          # [T, N, L]
    commit_times = np.full((N, L), np.nan)
    for si in range(N):
        for ti in range(T):
            unset = np.isnan(commit_times[si])
            commit_times[si, unset & commit_mask[ti, si]] = t_grid[ti]
    out["commit_time_mean"]     = float(np.nanmean(commit_times))
    out["commit_time_std"]      = float(np.nanstd(commit_times))
    out["never_committed_frac"] = float(np.isnan(commit_times).mean())

    # ── Wrong commitment fate tracking ───────────────────────────────────────
    if do_fate:
        t_list = t_grid.tolist()
        # index of t=0.95 (use the last t < 1.0)
        pre_final_idx = T - 2 if T >= 2 else T - 1
        final_idx     = T - 1

        fate = {}
        for src_t in FATE_SOURCE_TS:
            # find closest index in t_grid
            src_idx = int(np.argmin(np.abs(t_grid - src_t)))

            # committed_wrong at src_t  (majority: > 0.5 of seeds)
            cw_mask = cw_traj[src_idx] > 0.5      # [L]
            n_cw    = int(cw_mask.sum())

            if n_cw == 0:
                fate[f"{src_t:.2f}"] = {k: float("nan") for k in FATE_METRICS}
                fate[f"{src_t:.2f}"]["n_committed_wrong"] = 0
                continue

            # "correct" at time t = majority of seeds correct
            # Trajectory self-correction: correct at ANY t in (src_idx, pre_final_idx]
            ever_correct_in_traj = np.zeros(L, dtype=bool)
            for ti in range(src_idx + 1, pre_final_idx + 1):
                ever_correct_in_traj |= (correct_traj[ti] > 0.5)

            # Decode correction: still wrong after trajectory, but correct at t=1.0
            correct_at_final = correct_traj[final_idx] > 0.5         # [L]
            not_traj_fixed   = cw_mask & ~ever_correct_in_traj        # [L]
            decode_fixed     = not_traj_fixed & correct_at_final       # [L]
            stays_wrong      = not_traj_fixed & ~correct_at_final      # [L]
            traj_fixed       = cw_mask & ever_correct_in_traj          # [L]

            fate[f"{src_t:.2f}"] = {
                "n_committed_wrong":     n_cw,
                "frac_traj_corrected":   float(traj_fixed[cw_mask].mean()),
                "frac_decode_corrected": float(decode_fixed[cw_mask].mean()),
                "frac_stays_wrong":      float(stays_wrong[cw_mask].mean()),
            }
        out["fate"] = fate

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(per_seq_per_tau: dict) -> dict:
    out = {}
    for tau, seq_list in per_seq_per_tau.items():
        agg = {"t": seq_list[0]["t"]}

        # Per-t metrics
        for m in ALL_PER_T_METRICS:
            mat = np.array([s[m] for s in seq_list], dtype=np.float64)  # [S, T]
            agg[f"{m}_mean"] = np.nanmean(mat, axis=0).tolist()
            agg[f"{m}_std"]  = np.nanstd(mat,  axis=0).tolist()

        # Scalar metrics
        for m in SCALAR_METRICS:
            vals = np.array([s[m] for s in seq_list])
            agg[f"{m}_agg_mean"] = float(np.nanmean(vals))
            agg[f"{m}_agg_std"]  = float(np.nanstd(vals))

        # Fate tracking (only present when do_fate=True, i.e. tau==1.0)
        if "fate" in seq_list[0]:
            fate_agg = {}
            for src_t_key in seq_list[0]["fate"]:
                sub = {}
                for fm in FATE_METRICS:
                    vals = np.array([s["fate"][src_t_key][fm]
                                     for s in seq_list if "fate" in s],
                                    dtype=np.float64)
                    if fm == "n_committed_wrong":
                        sub["n_committed_wrong_mean"] = float(np.nanmean(vals))
                        sub["n_committed_wrong_std"]  = float(np.nanstd(vals))
                    else:
                        sub[f"{fm}_mean"] = float(np.nanmean(vals))
                        sub[f"{fm}_std"]  = float(np.nanstd(vals))
                fate_agg[src_t_key] = sub
            agg["fate"] = fate_agg

        out[str(tau)] = agg
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict, tau_ref: float = 1.0):
    ref = results[str(tau_ref)]
    t   = ref["t"]

    print(f"\n{'t':>5}  {'H':>6}  {'top1':>6}  {'cc':>6}  {'cw':>6}  "
          f"{'uncmt':>5}  {'w2c':>6}  {'c2w':>6}  "
          f"{'stab_p50':>8}  {'res_p50':>8}  {'BC':>6}  {'lo%':>5}  {'hi%':>5}")
    print("-" * 108)
    for i, tv in enumerate(t):
        def g(k, fmt=".3f"):
            v = ref[f"{k}_mean"][i]
            return "nan" if (v != v) else format(v, fmt)
        print(f"{tv:5.2f}  {g('entropy'):>6}  {g('top1_gt'):>6}  "
              f"{g('committed_correct'):>6}  {g('committed_wrong'):>6}  "
              f"{g('uncommitted'):>5}  {g('w2c'):>6}  {g('c2w'):>6}  "
              f"{g('stab_jsd_p50'):>8}  {g('residual_norm_p50'):>8}  "
              f"{g('bimodality_coeff'):>6}  {g('h_frac_low'):>5}  {g('h_frac_high'):>5}")

    print(f"\nCommitment time (H<thresh, τ={tau_ref}):")
    print(f"  mean  = {ref['commit_time_mean_agg_mean']:.3f} ± {ref['commit_time_mean_agg_std']:.3f}")
    print(f"  never = {ref['never_committed_frac_agg_mean']:.1%} of positions")

    if "fate" in ref:
        print(f"\nWrong-commitment fate tracking (τ={tau_ref}):")
        print(f"  {'src_t':>6}  {'n_cw':>6}  {'traj_corr':>10}  {'dec_corr':>9}  {'stays_wrong':>11}")
        print("  " + "-" * 50)
        for src_t_key, sub in sorted(ref["fate"].items()):
            def fg(k):
                v = sub.get(f"{k}_mean", float("nan"))
                return "nan" if (v != v) else f"{v:.3f}"
            n  = sub.get("n_committed_wrong_mean", float("nan"))
            print(f"  {float(src_t_key):>6.2f}  {n:>6.1f}  "
                  f"{fg('frac_traj_corrected'):>10}  "
                  f"{fg('frac_decode_corrected'):>9}  "
                  f"{fg('frac_stays_wrong'):>11}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plotting
# ─────────────────────────────────────────────────────────────────────────────

TAU_COLORS = {0.1: "#e74c3c", 0.5: "#e67e22", 1.0: "#2ecc71",
              2.0: "#3498db", 5.0: "#9b59b6"}


def _band(ax, t, res, key, color, label=None, lw=2, ls="-"):
    mean = np.array(res[f"{key}_mean"])
    std  = np.array(res[f"{key}_std"])
    mask = ~np.isnan(mean)
    ax.plot(t[mask], mean[mask], color=color, lw=lw, ls=ls,
            label=label or key)
    ax.fill_between(t[mask], (mean-std)[mask], (mean+std)[mask],
                    alpha=0.15, color=color)


def plot_results(results: dict, out_dir: str, label: str, tau_ref: float = 1.0):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ref = results[str(tau_ref)]
    t   = np.array(ref["t"])

    # ── Figure 1: v4 new metrics (2×2 grid) ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Panel 0: Residual norm distribution
    ax = axes[0]
    _band(ax, t, ref, "residual_norm_p50", "#3498db", label="p50 (median)")
    ax.plot(t, ref["residual_norm_p10_mean"], color="#3498db", lw=1, ls=":", alpha=0.7, label="p10")
    ax.plot(t, ref["residual_norm_p90_mean"], color="#3498db", lw=1, ls="--", alpha=0.7, label="p90")
    ax.set_title("Residual Norm  ||r_t|| = ||x̂_t − p_t @ E||\n"
                 "(↓ with t → geometric convergence  |  PROXY in ELF, exact in LangFlow)")
    ax.set_xlabel("t"); ax.set_ylabel("||r_t|| (embedding units)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 1: Temporal stability JSD distribution
    ax = axes[1]
    ax.plot(t, ref["stab_jsd_p10_mean"], color="#27ae60", lw=1, ls=":", alpha=0.7, label="p10 (stable)")
    _band(ax, t, ref, "stab_jsd_p50", "#27ae60", label="p50 (median)")
    ax.plot(t, ref["stab_jsd_p90_mean"], color="#27ae60", lw=1, ls="--", alpha=0.7, label="p90 (volatile)")
    ax.set_title("Temporal Stability  JSD(p_t, p_{t-Δ}) per position\n"
                 "(p10 = already stable committed | p90 = still volatile)")
    ax.set_xlabel("t"); ax.set_ylabel("JSD (nats)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Entropy bimodality
    ax = axes[2]
    lo_m  = np.array(ref["h_frac_low_mean"])
    mid_m = np.array(ref["h_frac_mid_mean"])
    hi_m  = np.array(ref["h_frac_high_mean"])
    ax.stackplot(t, hi_m, mid_m, lo_m,
                 labels=["H≥0.5 (high / uncommitted)",
                         "0.1≤H<0.5 (medium)",
                         "H<0.1 (low / committed)"],
                 colors=["#e74c3c", "#f39c12", "#2ecc71"], alpha=0.8)
    ax2 = ax.twinx()
    bc_m = np.array(ref["bimodality_coeff_mean"])
    ax2.plot(t, bc_m, color="black", lw=1.5, ls="--", label="BC")
    ax2.axhline(5/9, color="gray", ls=":", lw=1, label="BC=5/9 threshold")
    ax2.set_ylabel("Bimodality coeff BC", color="black")
    ax2.legend(fontsize=7, loc="upper right")
    ax.set_title("Entropy Bimodality\n(BC > 5/9 → bimodal; two populations: committed vs. uncommitted)")
    ax.set_xlabel("t"); ax.set_ylabel("Fraction of positions")
    ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)

    # Panel 3: Wrong commitment fate tracking
    ax = axes[3]
    if "fate" in ref:
        src_ts  = sorted(ref["fate"].keys(), key=float)
        src_xs  = [float(k) for k in src_ts]
        traj_y  = [ref["fate"][k].get("frac_traj_corrected_mean",   float("nan")) for k in src_ts]
        dec_y   = [ref["fate"][k].get("frac_decode_corrected_mean", float("nan")) for k in src_ts]
        wrong_y = [ref["fate"][k].get("frac_stays_wrong_mean",      float("nan")) for k in src_ts]
        traj_e  = [ref["fate"][k].get("frac_traj_corrected_std",    0) for k in src_ts]
        dec_e   = [ref["fate"][k].get("frac_decode_corrected_std",  0) for k in src_ts]
        wrong_e = [ref["fate"][k].get("frac_stays_wrong_std",       0) for k in src_ts]
        x = np.arange(len(src_xs))
        w = 0.25
        ax.bar(x - w, traj_y,  w, yerr=traj_e,  label="traj self-corrected", color="#27ae60", alpha=0.8)
        ax.bar(x,     dec_y,   w, yerr=dec_e,   label="decode corrected",    color="#3498db", alpha=0.8)
        ax.bar(x + w, wrong_y, w, yerr=wrong_e, label="stays wrong",         color="#e74c3c", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"src_t={v}" for v in src_xs], fontsize=8)
        ax.set_title("Wrong Commitment Fate\n(at each source t: what happens to committed-wrong positions?)")
        ax.set_ylabel("Fraction of committed-wrong positions")
        ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "fate tracking disabled\n(only at tau=1.0)", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle(f"v4 New Metrics (τ={tau_ref}) — {label}", fontsize=13)
    fig.tight_layout()
    p1 = str(Path(out_dir) / "anchor_probe_v4_new.png")
    fig.savefig(p1, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {p1}")

    # ── Figure 2: v3 carry-over panels ──────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    # [0] entropy + percentiles
    ax = axes[0]
    _band(ax, t, ref, "entropy", "#3498db", label="mean")
    ax.plot(t, ref["entropy_p10_mean"], color="#3498db", lw=1, ls=":", alpha=0.7, label="p10/p90")
    ax.plot(t, ref["entropy_p90_mean"], color="#3498db", lw=1, ls=":", alpha=0.7)
    ax.plot(t, ref["entropy_p50_mean"], color="#1a6fa0", lw=1, ls="--", alpha=0.7, label="p50")
    ax.set_title("Token Entropy"); ax.set_xlabel("t"); ax.set_ylabel("H [nats]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # [1] top-1 / top-5
    ax = axes[1]
    _band(ax, t, ref, "top1_gt",    "#e74c3c", label="top-1 gt")
    _band(ax, t, ref, "topk_gt",    "#2ecc71", label="top-5 gt", ls="--")
    _band(ax, t, ref, "top1_final", "#c0392b", label="top-1 final", ls=":")
    ax.set_title("Top-k Recovery"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("t"); ax.set_ylabel("fraction"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # [2] commitment stackplot
    ax = axes[2]
    cc = np.array(ref["committed_correct_mean"])
    cw = np.array(ref["committed_wrong_mean"])
    uc = np.array(ref["uncommitted_mean"])
    ax.stackplot(t, uc, cw, cc,
                 labels=["uncommitted", "committed-wrong", "committed-correct"],
                 colors=["#bdc3c7", "#e74c3c", "#2ecc71"], alpha=0.8)
    ax.set_title("Commitment State")
    ax.set_xlabel("t"); ax.set_ylabel("fraction"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    # [3] revision decomposition
    ax = axes[3]
    _band(ax, t, ref, "w2c",     "#27ae60", label="wrong→correct")
    _band(ax, t, ref, "c2w",     "#c0392b", label="correct→wrong")
    _band(ax, t, ref, "rev_jsd", "#8e44ad", label="scalar JSD", ls="--")
    ax.set_title("Revision Decomposition"); ax.set_xlabel("t"); ax.set_ylabel("fraction")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    # [4] noise agreement
    ax = axes[4]
    _band(ax, t, ref, "noise_agree", "#8e44ad", label="all seeds agree")
    ax.set_title("Noise Agreement"); ax.set_xlabel("t"); ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # [5] c2c + w2w
    ax = axes[5]
    _band(ax, t, ref, "c2c", "#27ae60", label="correct→correct")
    _band(ax, t, ref, "w2w", "#e74c3c", label="wrong→wrong")
    ax.set_title("Stable Fractions"); ax.set_xlabel("t"); ax.set_ylabel("fraction")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    # [6] geometric (d_soft, d_nn)
    ax = axes[6]
    _band(ax, t, ref, "d_soft_anchor", "#3498db", label="D_soft = residual mean")
    _band(ax, t, ref, "d_nn",          "#e74c3c", label="D_NN", ls="--")
    ax.set_title("Geometric Anchoring (Conjecture 5)"); ax.set_xlabel("t")
    ax.set_ylabel("distance"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # [7] stability JSD overlaid with entropy
    ax = axes[7]
    _band(ax, t, ref, "stab_jsd_p50", "#27ae60", label="stab_JSD p50")
    ax.plot(t, ref["stab_jsd_p10_mean"], color="#27ae60", lw=1, ls=":", alpha=0.7, label="stab_JSD p10")
    ax.plot(t, ref["stab_jsd_p90_mean"], color="#27ae60", lw=1, ls="--", alpha=0.7, label="stab_JSD p90")
    ax.set_title("Temporal Stability JSD (per position)"); ax.set_xlabel("t")
    ax.set_ylabel("JSD"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # [8] residual norm overlaid with top-1
    ax = axes[8]
    _band(ax, t, ref, "residual_norm_p50", "#3498db", label="||r_t|| p50")
    ax.plot(t, ref["residual_norm_p10_mean"], color="#3498db", lw=1, ls=":", alpha=0.7, label="p10")
    ax.plot(t, ref["residual_norm_p90_mean"], color="#3498db", lw=1, ls="--", alpha=0.7, label="p90")
    ax2 = ax.twinx()
    ax2.plot(t, ref["top1_gt_mean"], color="#e74c3c", lw=2, label="top-1 gt")
    ax2.set_ylabel("top-1 accuracy", color="#e74c3c")
    ax.set_title("Residual norm vs top-1\n(↓ norm + ↑ top1 = geometric commitment)"); ax.set_xlabel("t")
    ax.set_ylabel("||r_t||", color="#3498db"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Full Commitment Analysis v4 (τ={tau_ref}) — {label}", fontsize=13)
    fig.tight_layout()
    p2 = str(Path(out_dir) / "anchor_probe_v4_full.png")
    fig.savefig(p2, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {p2}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ELF model loading (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

def load_elf(config_path, checkpoint_path, override_max_length=None):
    from modules.model import ELF_models
    from modules.t5_encoder import get_encoder
    from utils.checkpoint_utils import load_checkpoint, load_encoder_checkpoint
    from utils.train_utils import TrainState
    from configs.config import load_config_from_yaml
    from transformers import AutoTokenizer

    _orig_cwd = os.getcwd()
    os.chdir(ELF_SRC)
    abs_cfg = os.path.join(_orig_cwd, config_path) if not os.path.isabs(config_path) else config_path
    config  = load_config_from_yaml(abs_cfg)
    os.chdir(_orig_cwd)

    if override_max_length is not None and override_max_length != config.max_length:
        config.max_length = override_max_length

    tokenizer     = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    enc_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)

    rng = jax.random.PRNGKey(42)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    d_enc     = enc_config.d_model
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
    optimizer  = optax.adamw(learning_rate=1e-4)
    state = TrainState.create(
        apply_fn=model.apply, params=elf_params["params"], tx=optimizer,
        dropout_rng=dropout_rng, ema_params1=copy.deepcopy(elf_params["params"]),
    )
    state, step = load_checkpoint(checkpoint_path, state)
    print(f"[elf] checkpoint loaded (step {step})")
    return model, state.ema_params1, encoder_params, encoder_model, tokenizer, config, enc_config


# ─────────────────────────────────────────────────────────────────────────────
# 9. Stub model
# ─────────────────────────────────────────────────────────────────────────────

class _RandomStubModel:
    def __init__(self, d=512, V=32128, seed=0):
        rng  = np.random.default_rng(seed)
        self.E = rng.standard_normal((V, d)).astype(np.float32)
        self.E /= np.linalg.norm(self.E, axis=-1, keepdims=True) + 1e-8
        self._A = rng.standard_normal((d, d)).astype(np.float32) * 0.05

    def forward(self, z_t, _t):
        x_hat  = z_t @ self._A.T
        logits = x_hat @ self.E.T
        return x_hat, logits


# ─────────────────────────────────────────────────────────────────────────────
# 10. Data loading (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

def load_owt_texts(n):
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
        enc  = tokenizer(text, return_tensors="np", truncation=True,
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
# 11. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        type=str, default=None)
    p.add_argument("--checkpoint",    type=str, default="embedded-language-flows/ELF-B-owt")
    p.add_argument("--stub",          action="store_true")
    p.add_argument("--n_samples",     type=int, default=64)
    p.add_argument("--seq_len",       type=int, default=256)
    p.add_argument("--n_t_steps",     type=int, default=21)
    p.add_argument("--tau_list",      type=str, default="0.5,1.0,2.0")
    p.add_argument("--topk",          type=int, default=5)
    p.add_argument("--n_noise",       type=int, default=4)
    p.add_argument("--commit_thresh", type=float, default=0.1)
    p.add_argument("--out_dir",       type=str, default="probe_results_v4")
    return p.parse_args()


def main():
    args     = parse_args()
    t_grid   = np.linspace(0.0, 1.0, args.n_t_steps)
    tau_list = [float(x) for x in args.tau_list.split(",")]

    if args.stub:
        print("[mode] RandomStub")
        stub = _RandomStubModel(d=512, V=32128)
        E    = stub.E

        def forward_fn(z_t, t, mask=None):
            return stub.forward(z_t[0], t)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        rng       = np.random.default_rng(0)
        samples   = []
        for _ in range(args.n_samples):
            ids  = rng.integers(0, 32128, size=args.seq_len).astype(np.int32)
            emb  = rng.standard_normal((args.seq_len, 512)).astype(np.float32)
            mask = np.ones(args.seq_len, dtype=np.float32)
            samples.append((ids, emb, mask))
        label = "RandomStub"

    else:
        if args.config is None:
            raise ValueError("--config required")
        print(f"[mode] ELF: {args.checkpoint}")
        (model, ema_params, encoder_params, encoder_model,
         tokenizer, config, enc_config) = load_elf(
             args.config, args.checkpoint, override_max_length=args.seq_len)

        latent_mean = getattr(config, "latent_mean", 0.0)
        latent_std  = getattr(config, "latent_std",  1.0)

        E_raw      = np.array(encoder_params["shared"]["embedding"])
        vocab_size = tokenizer.vocab_size
        E = E_raw[:vocab_size] / latent_std
        print(f"[embed] E: {E.shape}  latent_std={latent_std}  "
              f"(WARNING: T5 input embeddings, not contextual centroids)")

        has_sc     = config.self_cond_prob > 0
        has_sc_cfg = config.num_self_cond_cfg_tokens > 0
        sc_scale   = jnp.zeros((1,)) if has_sc_cfg else None

        @jax.jit
        def _fwd(params, z_jax, t_jax, mask_jax, sc_jax):
            return model.apply(
                {"params": params}, z_jax, t_jax,
                attention_mask=mask_jax, deterministic=True,
                self_cond_cfg_scale=sc_jax,
                decoder_step_active=jnp.array(True),
            )

        def forward_fn(z_t, t, attn_mask):
            z_in = np.concatenate([z_t, np.zeros_like(z_t)], axis=-1) if has_sc else z_t
            x_hat_b, logits_b = _fwd(
                ema_params,
                jnp.array(z_in),
                jnp.array([t], dtype=jnp.float32),
                jnp.array(attn_mask, dtype=jnp.float32),
                sc_scale,
            )
            return np.array(x_hat_b[0]), np.array(logits_b[0])

        print(f"[data] loading {args.n_samples} OWT texts…")
        texts   = load_owt_texts(args.n_samples)
        print(f"[encode] T5 encoder…")
        samples = encode_with_t5(texts, tokenizer, encoder_model, encoder_params,
                                 args.seq_len, latent_mean, latent_std)
        label   = "ELF-B OWT"

    # ── Probe ────────────────────────────────────────────────────────────────
    per_seq_per_tau = {tau: [] for tau in tau_list}

    for i, (gt_ids, emb, mask) in enumerate(samples):
        print(f"\n── Seq {i+1}/{len(samples)} — collecting logits…")
        logits_arr, xhat_arr = collect_logits(
            forward_fn, emb, mask, t_grid, args.n_noise, seed=i)

        final_logits = logits_arr[-1, 0]
        final_ids    = np.argmax(final_logits, axis=-1).astype(np.int32)

        for tau in tau_list:
            do_fate = (abs(tau - 1.0) < 1e-6)
            res = compute_metrics_for_tau(
                logits_arr, xhat_arr, E, t_grid,
                gt_ids, final_ids,
                tau=tau, topk=args.topk,
                commit_thresh=args.commit_thresh,
                do_fate=do_fate,
            )
            if do_fate:
                print(f"  τ=1.0  t=0.30  "
                      f"H={res['entropy'][6]:.3f}  "
                      f"cc={res['committed_correct'][6]:.3f}  "
                      f"cw={res['committed_wrong'][6]:.3f}  "
                      f"stab_p50={res['stab_jsd_p50'][6]:.4f}  "
                      f"res_p50={res['residual_norm_p50'][6]:.2f}  "
                      f"BC={res['bimodality_coeff'][6]:.3f}  "
                      f"lo={res['h_frac_low'][6]:.3f}  "
                      f"hi={res['h_frac_high'][6]:.3f}")
            per_seq_per_tau[tau].append(res)

    # ── Aggregate + save + plot ───────────────────────────────────────────────
    final = aggregate(per_seq_per_tau)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(args.out_dir) / "anchor_probe_v4.json"
    with open(json_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[save] → {json_path}")

    print_summary(final, tau_ref=1.0)
    plot_results(final, args.out_dir, label=label, tau_ref=1.0)
    print("[done]")


if __name__ == "__main__":
    main()
