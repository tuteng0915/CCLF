"""
Anchor Emergence Probing for ELF — v3 Extended Edition.

New metrics over v2:
  top1_gt / top1_final   — top-1 accuracy (vs top-5 in v2)
  committed_correct      — H < thresh AND top-1 == gt
  committed_wrong        — H < thresh AND top-1 != gt
  uncommitted            — H >= thresh
  c2c / c2w / w2c / w2w — per-step transition matrix (decomposes revision delta)
  noise_agree            — fraction of positions all N seeds agree on top-1
  entropy_p10/p50/p90   — entropy distribution percentiles (detects bimodality)
  commit_time_mean/std   — first t at which each position commits (per sequence)

Key question this answers:
  The 19% revision delta at t=1 — is it c2w (disturbance) or w2c (correction)?

Usage:
    cd ~/ELF
    CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \\
    python probe_anchor_v3.py \\
        --config src/configs/training_configs/train_owt_ELF-B.yml \\
        --checkpoint embedded-language-flows/ELF-B-owt \\
        --n_samples 64 --seq_len 256 --out_dir ~/probe_results_v3

    # pipeline test (no checkpoint):
    python probe_anchor_v3.py --stub --out_dir ~/probe_stub_v3
"""

import sys, os, argparse, copy, json
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELF_SRC = _SCRIPT_DIR / "src"
if not _ELF_SRC.exists():
    _ELF_SRC = _SCRIPT_DIR.parents[1] / "models" / "ELF" / "src"
ELF_SRC = str(_ELF_SRC)
DEFAULT_CONFIG = str(_ELF_SRC / "configs" / "training_configs" / "train_owt_ELF-B.yml")
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
# 1. Primitive functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_p(logits: np.ndarray, tau: float) -> np.ndarray:
    """softmax(logits / tau).  [L, V] → [L, V]"""
    l = logits / tau
    l -= l.max(axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)


def anchor_distance(x_hat: np.ndarray, p: np.ndarray, E: np.ndarray) -> float:
    """mean_L ||x̂_i − (p_i @ E)||.  E: [V, d]"""
    return float(np.linalg.norm(x_hat - p @ E, axis=-1).mean())


def topk_recovery(p: np.ndarray, ref_ids: np.ndarray, k: int) -> float:
    topk = np.argsort(p, axis=-1)[:, -k:]
    return float((topk == ref_ids[:, None]).any(axis=-1).mean())


def jsd(p_prev: Optional[np.ndarray], p_curr: np.ndarray) -> Optional[float]:
    if p_prev is None:
        return None
    pp = np.clip(p_prev, 1e-9, 1.0)
    pc = np.clip(p_curr, 1e-9, 1.0)
    m  = 0.5 * (pp + pc)
    return float((0.5 * ((pp * (np.log(pp) - np.log(m))).sum(-1)
                       + (pc * (np.log(pc) - np.log(m))).sum(-1))).mean())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logit collection (unchanged from v2)
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
# 3. Extended metric computation
# ─────────────────────────────────────────────────────────────────────────────

ALL_PER_T_METRICS = [
    "anchor_dist",
    # geometric anchoring (Conjecture 5)
    "d_soft_anchor",      # ||x̂ - p@E||  (same as anchor_dist, explicit name)
    "d_nn",               # min_v ||x̂ - E_v||   (nearest-neighbor distance)
    "margin",             # ||x̂ - E_2nd|| - ||x̂ - E_1st||  (↑ = real anchoring)
    # entropy
    "entropy", "entropy_p10", "entropy_p50", "entropy_p90",
    # recovery
    "topk_gt", "topk_final",
    "top1_gt", "top1_final",
    # commitment state
    "committed_correct", "committed_wrong", "uncommitted",
    # inter-seed consistency
    "noise_agree",
    # revision
    "rev_top1", "rev_jsd",
    # transition matrix
    "c2c", "c2w", "w2c", "w2w",
]

def geometric_anchoring(x_hat: np.ndarray, p: np.ndarray, E: np.ndarray,
                         E_sq: np.ndarray) -> tuple:
    """
    Compute three geometric metrics for Conjecture 5 verification.

    Returns (d_soft_anchor, d_nn, margin) averaged over positions.
      d_soft_anchor: ||x̂ - p@E||                (soft expected anchor distance)
      d_nn:          min_v ||x̂ - E_v||           (nearest-neighbor distance)
      margin:        ||x̂ - E_2nd|| - ||x̂ - E_1st||  (separation margin)

    E_sq: [V]  precomputed ||E_v||^2 for speed.
    """
    # x_hat: [L, d], E: [V, d], E_sq: [V]
    soft_anchor = p @ E                               # [L, d]
    d_soft = float(np.linalg.norm(x_hat - soft_anchor, axis=-1).mean())

    x_sq = (x_hat ** 2).sum(-1)                      # [L]
    # ||x̂ - E_v||^2 = ||x̂||^2 + ||E_v||^2 - 2 x̂·E_v
    dists_sq = x_sq[:, None] + E_sq[None, :] - 2.0 * (x_hat @ E.T)  # [L, V]
    dists_sq = np.maximum(dists_sq, 0.0)              # numerical safety

    # Nearest and second-nearest
    idx1 = np.argpartition(dists_sq, 1, axis=-1)[:, :2]   # [L, 2]
    d1_sq = dists_sq[np.arange(len(x_hat)), idx1[:, 0]]
    d2_sq = dists_sq[np.arange(len(x_hat)), idx1[:, 1]]
    d_nn   = float(np.sqrt(d1_sq).mean())
    margin = float((np.sqrt(d2_sq) - np.sqrt(d1_sq)).mean())

    return d_soft, d_nn, margin


def compute_metrics_for_tau(
    logits_arr: np.ndarray,   # [T, N, L, V]
    xhat_arr:   np.ndarray,   # [T, N, L, d]
    E:          np.ndarray,   # [V, d]
    E_geom:     Optional[np.ndarray],  # [V_seen, d] for NN/margin; defaults to E
    t_grid:     np.ndarray,   # [T]
    gt_ids:     np.ndarray,   # [L]
    final_ids:  np.ndarray,   # [L]
    tau:        float,
    topk:       int,
    commit_thresh: float,
) -> dict:
    """Compute all per-t metrics + scalar commitment-time stats for one tau."""
    T, N, L, V = logits_arr.shape

    out = {k: [] for k in ALL_PER_T_METRICS}
    out["t"] = t_grid.tolist()

    prev_p       = [None] * N
    prev_correct = None          # bool [L], majority vote across seeds

    if E_geom is None:
        E_geom = E
    # Precompute ||E_v||^2 once for geometric_anchoring
    E_sq = (E_geom ** 2).sum(-1)

    # Will accumulate H_mat per t for commitment-time post-processing
    H_trajectory = np.zeros((T, N, L), dtype=np.float32)

    for ti in range(T):
        # Per-seed accumulators
        a_anchor, a_topkgt, a_topkfin, a_top1gt, a_top1fin = [], [], [], [], []
        a_commcorr, a_commwrong, a_uncommit = [], [], []
        a_rev1, a_jsd = [], []
        a_d_soft, a_d_nn, a_margin = [], [], []
        top1_seeds = np.zeros((N, L), dtype=np.int32)
        H_seeds    = np.zeros((N, L), dtype=np.float32)

        for si in range(N):
            logits = logits_arr[ti, si]        # [L, V]
            x_hat  = xhat_arr[ti, si]          # [L, d]
            p = compute_p(logits, tau)         # [L, V]
            top1 = np.argmax(p, axis=-1)       # [L]
            pc = np.clip(p, 1e-9, 1.0)
            H_i = -(pc * np.log(pc)).sum(-1)   # [L]

            top1_seeds[si] = top1
            H_seeds[si]    = H_i

            d_soft = anchor_distance(x_hat, p, E)
            _, d_nn, margin = geometric_anchoring(x_hat, p, E_geom, E_sq)
            a_anchor.append(d_soft)
            a_d_soft.append(d_soft)
            a_d_nn.append(d_nn)
            a_margin.append(margin)
            a_topkgt.append(topk_recovery(p, gt_ids, topk))
            a_topkfin.append(topk_recovery(p, final_ids, topk))
            a_top1gt.append(float((top1 == gt_ids).mean()))
            a_top1fin.append(float((top1 == final_ids).mean()))

            committed = H_i < commit_thresh
            correct   = top1 == gt_ids
            a_commcorr.append(float((committed & correct).mean()))
            a_commwrong.append(float((committed & ~correct).mean()))
            a_uncommit.append(float((~committed).mean()))

            r1  = None if prev_p[si] is None else float(
                (np.argmax(prev_p[si], -1) != top1).mean())
            jsd_ = jsd(prev_p[si], p)
            if r1   is not None: a_rev1.append(r1)
            if jsd_ is not None: a_jsd.append(jsd_)
            prev_p[si] = p

        H_trajectory[ti] = H_seeds

        # Noise agreement: all N seeds predict same top-1 for each position
        noise_agree = float((top1_seeds == top1_seeds[0:1]).all(axis=0).mean())

        # Transition matrix: "correct" = majority of seeds has top1 == gt
        curr_correct = (top1_seeds == gt_ids[None]).mean(axis=0) > 0.5  # [L]

        # Store per-t
        out["anchor_dist"].append(float(np.mean(a_anchor)))
        out["d_soft_anchor"].append(float(np.mean(a_d_soft)))
        out["d_nn"].append(float(np.mean(a_d_nn)))
        out["margin"].append(float(np.mean(a_margin)))
        H_flat = H_seeds.flatten()
        out["entropy"].append(float(H_flat.mean()))
        out["entropy_p10"].append(float(np.percentile(H_flat, 10)))
        out["entropy_p50"].append(float(np.percentile(H_flat, 50)))
        out["entropy_p90"].append(float(np.percentile(H_flat, 90)))
        out["topk_gt"].append(float(np.mean(a_topkgt)))
        out["topk_final"].append(float(np.mean(a_topkfin)))
        out["top1_gt"].append(float(np.mean(a_top1gt)))
        out["top1_final"].append(float(np.mean(a_top1fin)))
        out["committed_correct"].append(float(np.mean(a_commcorr)))
        out["committed_wrong"].append(float(np.mean(a_commwrong)))
        out["uncommitted"].append(float(np.mean(a_uncommit)))
        out["noise_agree"].append(noise_agree)
        out["rev_top1"].append(float(np.mean(a_rev1)) if a_rev1 else float("nan"))
        out["rev_jsd"].append(float(np.mean(a_jsd))  if a_jsd  else float("nan"))

        if prev_correct is not None:
            out["c2c"].append(float((prev_correct  &  curr_correct).mean()))
            out["c2w"].append(float((prev_correct  & ~curr_correct).mean()))
            out["w2c"].append(float((~prev_correct &  curr_correct).mean()))
            out["w2w"].append(float((~prev_correct & ~curr_correct).mean()))
        else:
            for k in ["c2c", "c2w", "w2c", "w2w"]:
                out[k].append(float("nan"))

        prev_correct = curr_correct

    # Commitment time: first t at which each position drops below H_thresh
    # H_trajectory: [T, N, L]
    commit_mask = H_trajectory < commit_thresh   # [T, N, L]
    commit_times = np.full((N, L), np.nan)
    for si in range(N):
        for ti in range(T):
            unset = np.isnan(commit_times[si])
            commit_times[si, unset & commit_mask[ti, si]] = t_grid[ti]

    out["commit_time_mean"]      = float(np.nanmean(commit_times))
    out["commit_time_std"]       = float(np.nanstd(commit_times))
    out["never_committed_frac"]  = float(np.isnan(commit_times).mean())

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Aggregation across sequences
# ─────────────────────────────────────────────────────────────────────────────

SCALAR_METRICS = ["commit_time_mean", "commit_time_std", "never_committed_frac"]

def aggregate(per_seq_per_tau: dict) -> dict:
    out = {}
    for tau, seq_list in per_seq_per_tau.items():
        agg = {"t": seq_list[0]["t"]}
        for m in ALL_PER_T_METRICS:
            mat = np.array([s[m] for s in seq_list], dtype=np.float64)   # [S, T]
            agg[f"{m}_mean"] = np.nanmean(mat, axis=0).tolist()
            agg[f"{m}_std"]  = np.nanstd(mat,  axis=0).tolist()
        for m in SCALAR_METRICS:
            vals = np.array([s[m] for s in seq_list])
            agg[f"{m}_agg_mean"] = float(np.nanmean(vals))
            agg[f"{m}_agg_std"]  = float(np.nanstd(vals))
        out[str(tau)] = agg
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plotting
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

    # ── Figure 1: commitment state decomposition (τ=1.0) ────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    # Panel 0: entropy + percentiles
    ax = axes[0]
    _band(ax, t, ref, "entropy", "#3498db", label="mean")
    ax.plot(t, ref["entropy_p10_mean"], color="#3498db", lw=1, ls=":", alpha=0.7, label="p10/p90")
    ax.plot(t, ref["entropy_p90_mean"], color="#3498db", lw=1, ls=":", alpha=0.7)
    ax.plot(t, ref["entropy_p50_mean"], color="#1a6fa0", lw=1, ls="--", alpha=0.7, label="p50")
    ax.set_title("Entropy (mean ± std, percentiles)")
    ax.set_xlabel("t"); ax.set_ylabel("H(p_t) [nats]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 1: top1 vs top5
    ax = axes[1]
    _band(ax, t, ref, "top1_gt",   "#e74c3c", label="top-1 gt")
    _band(ax, t, ref, "topk_gt",   "#2ecc71", label="top-5 gt", ls="--")
    _band(ax, t, ref, "top1_final","#c0392b", label="top-1 final", ls=":")
    ax.set_title("Top-k Recovery"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("t"); ax.set_ylabel("fraction"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: commitment state breakdown
    ax = axes[2]
    cc = np.array(ref["committed_correct_mean"])
    cw = np.array(ref["committed_wrong_mean"])
    uc = np.array(ref["uncommitted_mean"])
    ax.stackplot(t, uc, cw, cc,
                 labels=["uncommitted", "committed-wrong", "committed-correct"],
                 colors=["#bdc3c7", "#e74c3c", "#2ecc71"], alpha=0.8)
    ax.set_title(f"Commitment State (H_thresh={0.1})")
    ax.set_xlabel("t"); ax.set_ylabel("fraction of positions"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    # Panel 3: transition matrix (c2w vs w2c — the key "纠错 vs 扰动" panel)
    ax = axes[3]
    _band(ax, t, ref, "w2c", "#27ae60", label="wrong→correct (correction)")
    _band(ax, t, ref, "c2w", "#c0392b", label="correct→wrong (disturbance)")
    _band(ax, t, ref, "rev_jsd", "#8e44ad", label="JSD (total revision)", ls="--")
    ax.set_title("Revision Decomposition\n(纠错 vs 扰动)")
    ax.set_xlabel("t"); ax.set_ylabel("fraction of positions")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    # Panel 4: noise agreement
    ax = axes[4]
    _band(ax, t, ref, "noise_agree", "#8e44ad", label="all seeds agree")
    ax.set_title("Noise Agreement\n(all N seeds have same top-1)")
    ax.set_xlabel("t"); ax.set_ylabel("fraction"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 5: c2c + w2w (staying put)
    ax = axes[5]
    _band(ax, t, ref, "c2c", "#27ae60", label="correct→correct")
    _band(ax, t, ref, "w2w", "#e74c3c", label="wrong→wrong")
    ax.set_title("Stable Fractions (no revision)")
    ax.set_xlabel("t"); ax.set_ylabel("fraction of positions")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    # Panel 6: Conjecture 5 — geometric anchoring (D_soft, D_NN)
    ax = axes[6]
    _band(ax, t, ref, "d_soft_anchor", "#3498db", label="D_soft (||x̂ - p@E||)")
    _band(ax, t, ref, "d_nn",          "#e74c3c", label="D_NN (nearest token)", ls="--")
    ax.set_title("Conjecture 5: Geometric Anchoring\n(both should ↓ with t)")
    ax.set_xlabel("t"); ax.set_ylabel("distance (embedding units)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 7: Conjecture 5 — margin (real anchoring vs collapse)
    ax = axes[7]
    _band(ax, t, ref, "margin", "#27ae60", label="Margin (D_2nd - D_1st)")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("Conjecture 5: Separation Margin\n(↑ = genuine anchoring, not collapse)")
    ax.set_xlabel("t"); ax.set_ylabel("margin (embedding units)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 8: entropy vs D_NN scatter proxy (overlay on t axis)
    ax = axes[8]
    H_m   = np.array(ref["entropy_mean"])
    dnn_m = np.array(ref["d_nn_mean"])
    sc = ax.scatter(t, H_m,   color="#3498db", s=30, label="Entropy (left)", zorder=3)
    ax2 = ax.twinx()
    ax2.plot(t, dnn_m, color="#e74c3c", lw=2, label="D_NN (right)")
    ax.set_xlabel("t"); ax.set_ylabel("H(p_t) [nats]", color="#3498db")
    ax2.set_ylabel("D_NN", color="#e74c3c")
    ax.set_title("Entropy & D_NN co-movement\n(should be correlated if Conj.5 holds)")
    ax.grid(alpha=0.3)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=8)

    fig.suptitle(f"Commitment Analysis (τ={tau_ref}) — {label}", fontsize=13)
    fig.tight_layout()
    p1 = str(Path(out_dir) / "anchor_probe_commitment.png")
    fig.savefig(p1, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── Figure 2: tau sweep (entropy + JSD) ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for tau_str, res in results.items():
        tau = float(tau_str)
        col = TAU_COLORS.get(tau, "gray")
        tv  = np.array(res["t"])
        for ax, key in [(axes[0], "entropy"), (axes[1], "rev_jsd")]:
            mean = np.array(res[f"{key}_mean"])
            mask = ~np.isnan(mean)
            ax.plot(tv[mask], mean[mask], color=col, lw=2, label=f"τ={tau}")
    axes[0].set_title("Token Entropy (τ sweep)"); axes[0].legend(fontsize=8)
    axes[1].set_title("JSD Revision (τ sweep)");  axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("t"); ax.grid(alpha=0.3)
    fig.suptitle(f"Temperature Sweep — {label}", fontsize=12)
    fig.tight_layout()
    p2 = str(Path(out_dir) / "anchor_probe_tau_sweep.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    print(f"[plot] saved → {p1}")
    print(f"[plot] saved → {p2}")


def print_summary(results: dict, tau_ref: float = 1.0):
    """Print key per-t table + commit-time summary to stdout."""
    ref = results[str(tau_ref)]
    t   = ref["t"]
    print(f"\n{'t':>5}  {'H':>6}  {'top1_gt':>7}  {'top5_gt':>7}  "
          f"{'comm_c':>6}  {'comm_w':>6}  {'uncmt':>6}  "
          f"{'w2c':>6}  {'c2w':>6}  {'agree':>6}  {'JSD':>7}")
    print("-" * 85)
    for i, tv in enumerate(t):
        def g(k):
            v = ref[f"{k}_mean"][i]
            return "nan" if (v != v) else f"{v:.3f}"
        print(f"{tv:5.2f}  {g('entropy'):>6}  {g('top1_gt'):>7}  {g('topk_gt'):>7}  "
              f"{g('committed_correct'):>6}  {g('committed_wrong'):>6}  {g('uncommitted'):>6}  "
              f"{g('w2c'):>6}  {g('c2w'):>6}  {g('noise_agree'):>6}  {g('rev_jsd'):>7}")

    print(f"\nCommitment time (H<0.1, τ={tau_ref}):")
    print(f"  mean  = {ref['commit_time_mean_agg_mean']:.3f} ± {ref['commit_time_mean_agg_std']:.3f}")
    print(f"  never = {ref['never_committed_frac_agg_mean']:.1%} of positions")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ELF model loading (unchanged from v2)
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

    if override_max_length is not None and override_max_length != config.max_length:
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
# 7. Stub model
# ─────────────────────────────────────────────────────────────────────────────

class _RandomStubModel:
    def __init__(self, d=512, V=32128, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((V, d)).astype(np.float32)
        self.E /= np.linalg.norm(self.E, axis=-1, keepdims=True) + 1e-8
        self._A = rng.standard_normal((d, d)).astype(np.float32) * 0.05

    def forward(self, z_t, _t):
        x_hat  = z_t @ self._A.T
        logits = x_hat @ self.E.T
        return x_hat, logits


# ─────────────────────────────────────────────────────────────────────────────
# 8. Data loading
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
# 9. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        type=str, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint",    type=str, default="embedded-language-flows/ELF-B-owt")
    p.add_argument("--stub",          action="store_true")
    p.add_argument("--n_samples",     type=int, default=64)
    p.add_argument("--seq_len",       type=int, default=256)
    p.add_argument("--n_t_steps",     type=int, default=21)
    p.add_argument("--tau_list",      type=str, default="0.1,0.5,1.0,2.0,5.0")
    p.add_argument("--topk",          type=int, default=5)
    p.add_argument("--n_noise",       type=int, default=4)
    p.add_argument("--commit_thresh", type=float, default=0.1,
                   help="Entropy threshold for 'committed' state (nats).")
    p.add_argument("--centroid_path", type=str, default=None,
                   help="Path to token_centroids.npz (contextual centroids). "
                        "If omitted, falls back to raw T5 input embeddings (space mismatch).")
    p.add_argument("--out_dir",       type=str, default="probe_results_v3")
    return p.parse_args()


def main():
    args     = parse_args()
    t_grid   = np.linspace(0.0, 1.0, args.n_t_steps)
    tau_list = [float(x) for x in args.tau_list.split(",")]

    # ── Model / data setup ───────────────────────────────────────────────────
    if args.stub:
        print("[mode] RandomStub")
        stub = _RandomStubModel(d=512, V=32128)
        E    = stub.E
        E_geom = E

        def forward_fn(z_t, t, mask=None):
            return stub.forward(z_t[0], t)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(args.n_samples):
            ids  = rng.integers(0, 32128, size=args.seq_len).astype(np.int32)
            emb  = rng.standard_normal((args.seq_len, 512)).astype(np.float32)
            mask = np.ones(args.seq_len, dtype=np.float32)
            samples.append((ids, emb, mask))
        label = "RandomStub"

    else:
        print(f"[mode] ELF: {args.checkpoint}")
        (model, ema_params, encoder_params, encoder_model,
         tokenizer, config, enc_config) = load_elf(
             args.config, args.checkpoint, override_max_length=args.seq_len)

        latent_mean = getattr(config, "latent_mean", 0.0)
        latent_std  = getattr(config, "latent_std",  1.0)

        if args.centroid_path:
            data = np.load(args.centroid_path)
            E_raw = data["centroids"].astype(np.float32)
            print(f"[embed] E: {E_raw.shape}  source=contextual_centroid ({args.centroid_path})")
            centroid_latent_std = float(data["latent_std"]) if "latent_std" in data else None
            if "counts" in data:
                centroid_counts = data["counts"][:tokenizer.vocab_size]
            else:
                centroid_counts = None
        else:
            E_raw = np.array(encoder_params["shared"]["embedding"])
            print(f"[embed] E: {E_raw.shape}  source=raw_input_embedding  "
                  f"(WARNING: space mismatch - use --centroid_path for correct geometry)")
            centroid_latent_std = None
            centroid_counts = None
        vocab_size = tokenizer.vocab_size
        already_normalized = (
            centroid_latent_std is not None
            and np.isclose(centroid_latent_std, float(latent_std))
        )
        E = E_raw[:vocab_size] if already_normalized else E_raw[:vocab_size] / latent_std
        if centroid_counts is not None:
            seen_mask = centroid_counts > 0
            E_geom = E[seen_mask]
            print(f"[embed] geometric E uses {int(seen_mask.sum())}/{len(seen_mask)} seen centroids")
        else:
            E_geom = E
        norm_note = "already normalized" if already_normalized else "divided by latent_std"
        print(f"[embed] normalized E: {E.shape}  latent_std={latent_std}  ({norm_note})")

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
        label = "ELF-B OWT"

    # ── Probe ────────────────────────────────────────────────────────────────
    per_seq_per_tau = {tau: [] for tau in tau_list}

    for i, (gt_ids, emb, mask) in enumerate(samples):
        print(f"\n── Seq {i+1}/{len(samples)} — collecting logits…")
        logits_arr, xhat_arr = collect_logits(
            forward_fn, emb, mask, t_grid, args.n_noise, seed=i)

        final_logits = logits_arr[-1, 0]
        final_ids    = np.argmax(final_logits, axis=-1).astype(np.int32)

        for tau in tau_list:
            res = compute_metrics_for_tau(
                logits_arr, xhat_arr, E, E_geom, t_grid,
                gt_ids, final_ids, tau=tau, topk=args.topk,
                commit_thresh=args.commit_thresh,
            )
            if tau == 1.0:
                for ti, t_val in enumerate(t_grid):
                    print(f"  τ=1.0  t={t_val:.2f}  "
                          f"H={res['entropy'][ti]:.3f}  "
                          f"top1_gt={res['top1_gt'][ti]:.3f}  "
                          f"top5_gt={res['topk_gt'][ti]:.3f}  "
                          f"cc={res['committed_correct'][ti]:.3f}  "
                          f"cw={res['committed_wrong'][ti]:.3f}  "
                          f"w2c={res['w2c'][ti]:.4f}  "
                          f"c2w={res['c2w'][ti]:.4f}  "
                          f"agree={res['noise_agree'][ti]:.3f}")
            per_seq_per_tau[tau].append(res)

    # ── Aggregate + save + plot ───────────────────────────────────────────────
    final = aggregate(per_seq_per_tau)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(args.out_dir) / "anchor_probe_v3.json"
    with open(json_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[save] → {json_path}")

    print_summary(final, tau_ref=1.0)
    plot_results(final, args.out_dir, label=label, tau_ref=1.0)
    print("[done]")


if __name__ == "__main__":
    main()
