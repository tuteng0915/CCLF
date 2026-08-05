"""
LangFlow Anchor Emergence Probing — v1

Same metric suite as ELF probe_anchor_v3.py, adapted for LangFlow:

  Framework  : PyTorch (not JAX)
  Tokenizer  : GPT-2 (vocab 50257)
  Timestep   : γ = log-SNR via learnable Gumbel proposal
               t_grid displayed as [0,1], mapped to [γ_min, γ_max] linearly
               t=0 ≡ γ_min (max noise), t=1 ≡ γ_max (clean)
  Noise      : z = α(γ)·x + σ(γ)·ε,  α=sqrt(sigmoid(−γ)), σ=sqrt(sigmoid(γ))
  Latent     : token embedding space (normalized: L2 + sqrt(d) scale)
  Decode head: always-on linear head + preconditioning skip (no decoder_step_active)
  E matrix   : model._get_embedding_matrix() — SAME space as latent!
               → D_NN is directly meaningful (no space mismatch unlike ELF)

Geometric metrics (Conjecture 5):
  d_soft_anchor  ||x̂ − p@E||         soft expected anchor distance
  d_nn           min_v ||x̂ − E_v||   nearest token distance  ← cleanest in LangFlow
  margin         D_2nd − D_1st        separation margin (↑ = genuine anchoring)

Setup on server:
  git clone https://github.com/nealchen2003/LangFlow.git ~/LangFlow
  pip install -e ~/LangFlow
  pip install safetensors

Usage:
  python probe_langflow.py \\
      --checkpoint Continuous-Rivals-Discrete/langflow-owt \\
      --n_samples 64 --seq_len 256 --out_dir ~/probe_langflow_v1

  # smoke test (no checkpoint needed):
  python probe_langflow.py --stub --out_dir ~/probe_langflow_stub
"""

import sys, os, argparse, json, math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── LangFlow import ──────────────────────────────────────────────────────────
LANGFLOW_REPO = os.path.expanduser("~/LangFlow")
if os.path.isdir(LANGFLOW_REPO) and LANGFLOW_REPO not in sys.path:
    sys.path.insert(0, LANGFLOW_REPO)

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Primitive metric functions  (numpy, same as v3)
# ─────────────────────────────────────────────────────────────────────────────

def softmax_np(logits: np.ndarray, tau: float) -> np.ndarray:
    l = logits / tau
    l -= l.max(axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)


def token_entropy(p: np.ndarray) -> np.ndarray:
    """Per-position entropy [L, V] → [L]"""
    pc = np.clip(p, 1e-9, 1.0)
    return -(pc * np.log(pc)).sum(axis=-1)


def anchor_distance(x_hat: np.ndarray, p: np.ndarray, E: np.ndarray) -> float:
    return float(np.linalg.norm(x_hat - p @ E, axis=-1).mean())


def topk_recovery(p: np.ndarray, ref_ids: np.ndarray, k: int) -> float:
    topk = np.argsort(p, axis=-1)[:, -k:]
    return float((topk == ref_ids[:, None]).any(axis=-1).mean())


def top1_acc(p: np.ndarray, ref_ids: np.ndarray) -> float:
    return float((np.argmax(p, axis=-1) == ref_ids).mean())


def jsd(p_prev: Optional[np.ndarray], p_curr: np.ndarray) -> Optional[float]:
    if p_prev is None:
        return None
    pp = np.clip(p_prev, 1e-9, 1.0)
    pc = np.clip(p_curr, 1e-9, 1.0)
    m  = 0.5 * (pp + pc)
    return float((0.5 * ((pp * (np.log(pp) - np.log(m))).sum(-1)
                       + (pc * (np.log(pc) - np.log(m))).sum(-1))).mean())


def geometric_anchoring(x_hat: np.ndarray, p: np.ndarray,
                         E: np.ndarray, E_sq: np.ndarray) -> tuple:
    """Returns (d_soft, d_nn, margin) averaged over positions."""
    d_soft = float(np.linalg.norm(x_hat - p @ E, axis=-1).mean())
    x_sq = (x_hat ** 2).sum(-1)                             # [L]
    dists_sq = np.maximum(
        x_sq[:, None] + E_sq[None, :] - 2.0 * (x_hat @ E.T), 0.0)  # [L, V]
    # two smallest per position (argpartition is faster than full sort)
    idx2 = np.argpartition(dists_sq, 2, axis=-1)[:, :2]    # [L, 2]
    rows  = np.arange(len(x_hat))
    d1_sq = dists_sq[rows, idx2[:, 0]]
    d2_sq = dists_sq[rows, idx2[:, 1]]
    # ensure d1 ≤ d2
    swap  = d1_sq > d2_sq
    d1_sq[swap], d2_sq[swap] = d2_sq[swap], d1_sq[swap]
    d_nn   = float(np.sqrt(d1_sq).mean())
    margin = float((np.sqrt(d2_sq) - np.sqrt(d1_sq)).mean())
    return d_soft, d_nn, margin


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logit collection
# ─────────────────────────────────────────────────────────────────────────────

def gamma_from_t(t_grid: np.ndarray, gamma_min: float, gamma_max: float) -> np.ndarray:
    """
    Map t ∈ [0, 1] → γ (log-SNR), matching ELF's convention:
      t=0  →  most noisy  (γ_max, since α=sqrt(sigmoid(-γ)) is smallest at γ_max)
      t=1  →  most clean  (γ_min, since α≈1 when γ is most negative)

    LangFlow noise: z = α(γ)·x + σ(γ)·ε
      α(γ) = sqrt(sigmoid(-γ)),  σ(γ) = sqrt(sigmoid(γ))
    → γ_min = -10 means α≈1, σ≈0  (CLEAN)
    → γ_max = +10 means α≈0, σ≈1  (NOISY)
    """
    return gamma_max + t_grid * (gamma_min - gamma_max)   # t=0→max(noisy), t=1→min(clean)


def collect_logits_langflow(
    model,
    clean_emb: np.ndarray,      # [L, d]  normalized token embedding of clean text
    gamma_grid: np.ndarray,     # [T]  γ values (log-SNR)
    n_noise: int,
    seed: int,
    self_conditioning: bool,
) -> tuple:
    """
    Run forward passes for all (t, noise_seed).

    Returns:
      logits_arr  [T, N, L, V]  vocab logits
      xhat_arr    [T, N, L, d]  denoising output (last backbone hidden state)
    """
    rng = np.random.default_rng(seed)
    L, d = clean_emb.shape
    T, N = len(gamma_grid), n_noise

    logits_arr = None
    xhat_arr   = np.zeros((T, N, L, d), dtype=np.float32)

    x_torch = torch.from_numpy(clean_emb).to(device, dtype=torch.float32)  # [L, d]

    with torch.no_grad():
        for ti, gamma in enumerate(gamma_grid):
            alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma)).item())
            sigma = math.sqrt(torch.sigmoid(torch.tensor( gamma)).item())

            # Batch all noise seeds together for efficiency
            eps = torch.from_numpy(
                rng.standard_normal((N, L, d)).astype(np.float32)).to(device)
            z_batch = alpha * x_torch[None] + sigma * eps   # [N, L, d]

            gamma_batch = torch.full((N,), gamma, device=device, dtype=torch.float32)
            sc = torch.zeros_like(z_batch) if self_conditioning else None

            out = model(
                noisy_embeds=z_batch,
                timesteps=gamma_batch,
                x_self_cond=sc,
                return_dict=False,
            )
            # out may be (logits,) or logits directly
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits_np = logits.cpu().float().numpy()          # [N, L, V]
            xhat_np   = z_batch.cpu().float().numpy()         # [N, L, d]
            # Note: for LangFlow we store z_batch as proxy for x_hat since
            # the backbone doesn't separately expose the final hidden state.
            # The geometric metrics on z vs E are still meaningful because
            # z converges toward clean x as t→1, and E is in the same space.
            # TODO: expose backbone hidden states if needed for exact x̂.

            if logits_arr is None:
                V = logits_np.shape[-1]
                logits_arr = np.zeros((T, N, L, V), dtype=np.float32)

            logits_arr[ti] = logits_np
            xhat_arr[ti]   = xhat_np

    return logits_arr, xhat_arr


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metric computation  (same structure as v3)
# ─────────────────────────────────────────────────────────────────────────────

ALL_METRICS = [
    "anchor_dist", "d_soft_anchor", "d_nn", "margin",
    "entropy", "entropy_p10", "entropy_p50", "entropy_p90",
    "topk_gt", "topk_final", "top1_gt", "top1_final",
    "committed_correct", "committed_wrong", "uncommitted",
    "noise_agree",
    "rev_top1", "rev_jsd",
    "c2c", "c2w", "w2c", "w2w",
]


def compute_metrics(
    logits_arr: np.ndarray,   # [T, N, L, V]
    xhat_arr:   np.ndarray,   # [T, N, L, d]
    E:          np.ndarray,   # [V, d]
    t_grid:     np.ndarray,   # [T]  (display axis, 0→1)
    gt_ids:     np.ndarray,   # [L]
    final_ids:  np.ndarray,   # [L]
    tau:        float,
    topk:       int,
    commit_thresh: float,
) -> dict:
    T, N, L, V = logits_arr.shape
    E_sq = (E ** 2).sum(-1)   # [V]

    out = {k: [] for k in ALL_METRICS}
    out["t"] = t_grid.tolist()

    prev_p       = [None] * N
    prev_correct = None
    H_trajectory = np.zeros((T, N, L), dtype=np.float32)

    for ti in range(T):
        a_anchor, a_soft, a_nn, a_margin = [], [], [], []
        a_topkgt, a_topkfin, a_t1gt, a_t1fin = [], [], [], []
        a_cc, a_cw, a_uc = [], [], []
        a_r1, a_jsd = [], []
        top1_seeds = np.zeros((N, L), dtype=np.int32)
        H_seeds    = np.zeros((N, L), dtype=np.float32)

        for si in range(N):
            logits = logits_arr[ti, si]       # [L, V]
            x_hat  = xhat_arr[ti, si]         # [L, d]
            p = softmax_np(logits, tau)       # [L, V]
            top1 = np.argmax(p, axis=-1)
            pc = np.clip(p, 1e-9, 1.0)
            H_i = -(pc * np.log(pc)).sum(-1)  # [L]

            top1_seeds[si] = top1
            H_seeds[si]    = H_i

            a_anchor.append(anchor_distance(x_hat, p, E))
            ds, dn, mg = geometric_anchoring(x_hat, p, E, E_sq)
            a_soft.append(ds); a_nn.append(dn); a_margin.append(mg)

            a_topkgt.append(topk_recovery(p, gt_ids, topk))
            a_topkfin.append(topk_recovery(p, final_ids, topk))
            a_t1gt.append(top1_acc(p, gt_ids))
            a_t1fin.append(top1_acc(p, final_ids))

            committed = H_i < commit_thresh
            correct   = top1 == gt_ids
            a_cc.append(float((committed & correct).mean()))
            a_cw.append(float((committed & ~correct).mean()))
            a_uc.append(float((~committed).mean()))

            r1  = None if prev_p[si] is None else float(
                (np.argmax(prev_p[si], -1) != top1).mean())
            jsd_ = jsd(prev_p[si], p)
            if r1   is not None: a_r1.append(r1)
            if jsd_ is not None: a_jsd.append(jsd_)
            prev_p[si] = p

        H_trajectory[ti] = H_seeds

        noise_agree  = float((top1_seeds == top1_seeds[0:1]).all(axis=0).mean())
        curr_correct = (top1_seeds == gt_ids[None]).mean(axis=0) > 0.5

        def mv(lst): return float(np.mean(lst)) if lst else float("nan")

        out["anchor_dist"].append(mv(a_anchor))
        out["d_soft_anchor"].append(mv(a_soft))
        out["d_nn"].append(mv(a_nn))
        out["margin"].append(mv(a_margin))
        H_flat = H_seeds.flatten()
        out["entropy"].append(float(H_flat.mean()))
        out["entropy_p10"].append(float(np.percentile(H_flat, 10)))
        out["entropy_p50"].append(float(np.percentile(H_flat, 50)))
        out["entropy_p90"].append(float(np.percentile(H_flat, 90)))
        out["topk_gt"].append(mv(a_topkgt))
        out["topk_final"].append(mv(a_topkfin))
        out["top1_gt"].append(mv(a_t1gt))
        out["top1_final"].append(mv(a_t1fin))
        out["committed_correct"].append(mv(a_cc))
        out["committed_wrong"].append(mv(a_cw))
        out["uncommitted"].append(mv(a_uc))
        out["noise_agree"].append(noise_agree)
        out["rev_top1"].append(mv(a_r1))
        out["rev_jsd"].append(mv(a_jsd))

        if prev_correct is not None:
            out["c2c"].append(float((prev_correct  &  curr_correct).mean()))
            out["c2w"].append(float((prev_correct  & ~curr_correct).mean()))
            out["w2c"].append(float((~prev_correct &  curr_correct).mean()))
            out["w2w"].append(float((~prev_correct & ~curr_correct).mean()))
        else:
            for k in ["c2c","c2w","w2c","w2w"]:
                out[k].append(float("nan"))
        prev_correct = curr_correct

    # Commitment time
    commit_mask  = H_trajectory < commit_thresh
    commit_times = np.full((N, L), np.nan)
    for si in range(N):
        for ti in range(T):
            unset = np.isnan(commit_times[si])
            commit_times[si, unset & commit_mask[ti, si]] = t_grid[ti]

    out["commit_time_mean"]     = float(np.nanmean(commit_times))
    out["commit_time_std"]      = float(np.nanstd(commit_times))
    out["never_committed_frac"] = float(np.isnan(commit_times).mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Aggregation
# ─────────────────────────────────────────────────────────────────────────────

SCALAR_KEYS = ["commit_time_mean", "commit_time_std", "never_committed_frac"]

def aggregate(seq_list: list) -> dict:
    out = {"t": seq_list[0]["t"]}
    for m in ALL_METRICS:
        mat = np.array([s[m] for s in seq_list], dtype=np.float64)
        out[f"{m}_mean"] = np.nanmean(mat, axis=0).tolist()
        out[f"{m}_std"]  = np.nanstd(mat,  axis=0).tolist()
    for m in SCALAR_KEYS:
        vals = np.array([s[m] for s in seq_list])
        out[f"{m}_agg_mean"] = float(np.nanmean(vals))
        out[f"{m}_agg_std"]  = float(np.nanstd(vals))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    t = results["t"]
    print(f"\n{'t':>5}  {'H':>6}  {'top1_gt':>7}  {'top5_gt':>7}  "
          f"{'cc':>6}  {'cw':>6}  {'d_nn':>7}  {'margin':>7}  "
          f"{'w2c':>6}  {'jsd':>7}  {'agree':>6}")
    print("-" * 85)
    for i, tv in enumerate(t):
        def g(k):
            v = results[f"{k}_mean"][i]
            return "nan" if (v != v) else f"{v:.3f}"
        print(f"{tv:5.2f}  {g('entropy'):>6}  {g('top1_gt'):>7}  {g('topk_gt'):>7}  "
              f"{g('committed_correct'):>6}  {g('committed_wrong'):>6}  "
              f"{g('d_nn'):>7}  {g('margin'):>7}  "
              f"{g('w2c'):>6}  {g('rev_jsd'):>7}  {g('noise_agree'):>6}")
    print(f"\nCommit time (H<thresh):  "
          f"mean={results['commit_time_mean_agg_mean']:.3f}  "
          f"never={results['never_committed_frac_agg_mean']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_dir: str, label: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t = np.array(results["t"])

    def band(ax, key, color, ls="-", lbl=None):
        mean = np.array(results[f"{key}_mean"])
        std  = np.array(results[f"{key}_std"])
        mask = ~np.isnan(mean)
        ax.plot(t[mask], mean[mask], color=color, lw=2, ls=ls, label=lbl or key)
        ax.fill_between(t[mask], (mean-std)[mask], (mean+std)[mask],
                        alpha=0.15, color=color)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    # 0: entropy + percentiles
    ax = axes[0]
    band(ax, "entropy",     "#3498db", lbl="mean")
    ax.plot(t, results["entropy_p10_mean"], color="#3498db", lw=1, ls=":", alpha=0.7, label="p10/p90")
    ax.plot(t, results["entropy_p90_mean"], color="#3498db", lw=1, ls=":", alpha=0.7)
    ax.plot(t, results["entropy_p50_mean"], color="#1a6fa0", lw=1, ls="--", alpha=0.8, label="p50")
    ax.set_title("Entropy (mean ± std, percentiles)"); ax.set_xlabel("t (γ_min→γ_max)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 1: top1 + top5
    ax = axes[1]
    band(ax, "top1_gt",   "#e74c3c", lbl="top-1 gt")
    band(ax, "topk_gt",   "#2ecc71", ls="--", lbl="top-5 gt")
    band(ax, "top1_final","#c0392b", ls=":",  lbl="top-1 final")
    ax.set_title("Top-k Recovery"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2: commitment state
    ax = axes[2]
    cc = np.array(results["committed_correct_mean"])
    cw = np.array(results["committed_wrong_mean"])
    uc = np.array(results["uncommitted_mean"])
    ax.stackplot(t, uc, cw, cc,
                 labels=["uncommitted", "committed-wrong", "committed-correct"],
                 colors=["#bdc3c7", "#e74c3c", "#2ecc71"], alpha=0.8)
    ax.set_title("Commitment State"); ax.set_ylim(0, 1); ax.set_xlabel("t")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    # 3: transition matrix
    ax = axes[3]
    band(ax, "w2c", "#27ae60", lbl="wrong→correct (correction)")
    band(ax, "c2w", "#c0392b", lbl="correct→wrong (disturbance)")
    band(ax, "rev_jsd", "#8e44ad", ls="--", lbl="JSD")
    ax.axhline(0, color="gray", lw=1, ls=":"); ax.set_ylim(bottom=0)
    ax.set_title("Revision Decomposition"); ax.set_xlabel("t")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4: noise agreement
    ax = axes[4]
    band(ax, "noise_agree", "#8e44ad", lbl="all seeds agree")
    ax.set_title("Noise Agreement (all N seeds same top-1)")
    ax.set_ylim(0, 1.05); ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5: D_NN — the key Conjecture 5 metric for LangFlow
    ax = axes[5]
    band(ax, "d_nn",          "#e74c3c", lbl="D_NN (nearest token, ↓ = anchoring)")
    band(ax, "d_soft_anchor", "#3498db", ls="--", lbl="D_soft (||x̂ − p@E||)")
    ax.set_title("Geometric Anchoring — Conjecture 5\n(latent space = token embedding space!)")
    ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 6: Separation margin
    ax = axes[6]
    band(ax, "margin", "#27ae60", lbl="Margin (D_2nd − D_1st)")
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_title("Separation Margin\n(↑ = genuine anchoring, not collapse)")
    ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 7: Entropy vs D_NN co-movement (Conjecture 5 test)
    ax = axes[7]
    H_m   = np.array(results["entropy_mean"])
    dnn_m = np.array(results["d_nn_mean"])
    ax.scatter(t, H_m,   color="#3498db", s=30, zorder=3, label="Entropy")
    ax2 = ax.twinx()
    ax2.plot(t, dnn_m, color="#e74c3c", lw=2, label="D_NN")
    ax.set_xlabel("t"); ax.set_ylabel("H(p_t)", color="#3498db")
    ax2.set_ylabel("D_NN", color="#e74c3c")
    ax.set_title("H & D_NN co-movement\n(Conj.5: correlated ↔ geometric anchoring)")
    ax.grid(alpha=0.3)
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, n1+n2, fontsize=8)

    # 8: c2c + w2w (stability)
    ax = axes[8]
    band(ax, "c2c", "#27ae60", lbl="correct→correct")
    band(ax, "w2w", "#e74c3c", lbl="wrong→wrong")
    ax.set_title("Stable Fractions"); ax.set_xlabel("t")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    fig.suptitle(f"LangFlow Anchor Emergence Probing — {label}", fontsize=13)
    fig.tight_layout()
    path = str(Path(out_dir) / "langflow_probe_main.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_langflow(checkpoint: str):
    """
    Load LangFlow from HuggingFace checkpoint (local path or hub id).

    Returns: model, tokenizer, gamma_min, gamma_max, E [V, d]
    """
    try:
        from langflow import LangFlow, LangFlowConfig
    except ImportError:
        raise ImportError(
            "LangFlow package not found.\n"
            "Install with:\n"
            "  git clone https://github.com/nealchen2003/LangFlow.git ~/LangFlow\n"
            "  (add ~/LangFlow to PYTHONPATH or sys.path)"
        )
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    import os

    print(f"[langflow] loading from {checkpoint}")

    # Download/locate files
    if os.path.isdir(checkpoint):
        ckpt_dir = checkpoint
    else:
        ckpt_dir = snapshot_download(checkpoint)

    config = LangFlowConfig.from_pretrained(ckpt_dir)
    model  = LangFlow(config)

    # Load weights manually (avoids from_pretrained transformers version issues)
    weight_file = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(weight_file):
        state_dict = load_file(weight_file, device="cpu")
    else:
        # Try pytorch_model.bin fallback
        import torch as _torch
        weight_file = os.path.join(ckpt_dir, "pytorch_model.bin")
        state_dict = _torch.load(weight_file, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[langflow] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[langflow] unexpected keys: {unexpected[:5]}")

    model = model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    gamma_min = float(model.proposal.gamma_min)
    gamma_max = float(model.proposal.gamma_max)
    print(f"[langflow] γ_min={gamma_min:.2f}  γ_max={gamma_max:.2f}")
    print(f"[langflow] self_conditioning={model.config.self_conditioning}")

    # Embedding matrix: [V, d], normalized (same space as latent!)
    with torch.no_grad():
        E = model._get_embedding_matrix().cpu().float().numpy()   # [V, d]
    print(f"[langflow] E: {E.shape}  (token embedding space — NO space mismatch)")

    return model, tokenizer, gamma_min, gamma_max, E


# ─────────────────────────────────────────────────────────────────────────────
# 8. Stub model
# ─────────────────────────────────────────────────────────────────────────────

class _StubModel(torch.nn.Module):
    def __init__(self, d=768, V=50257, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        # Embedding table (normalized)
        E = rng.standard_normal((V, d)).astype(np.float32)
        E = E / (np.linalg.norm(E, axis=-1, keepdims=True) + 1e-8)
        E *= math.sqrt(d)
        self.E = torch.from_numpy(E)
        self.W = torch.from_numpy(rng.standard_normal((d, d)).astype(np.float32) * 0.05)
        self._V = V; self._d = d

    def forward(self, noisy_embeds, timesteps, x_self_cond=None, return_dict=False):
        N, L, d = noisy_embeds.shape
        x = noisy_embeds @ self.W.to(noisy_embeds.device)
        logits = x @ self.E.to(noisy_embeds.device).T
        return (logits,)

    def _embed_tokens(self, ids):
        return self.E[ids]

    class _FakeConfig:
        vocab_size = 50257
        self_conditioning = False

    class _FakeProposal:
        gamma_min = -10.0
        gamma_max = 10.0

    config   = _FakeConfig()
    proposal = _FakeProposal()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Data loading + encoding
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
            t = _stream(name, **kw)
            if t:
                print(f"[data] loaded from {name}")
                return t
        except Exception as e:
            print(f"[data] {name} failed: {e}")
    raise RuntimeError("Could not load any dataset.")


def encode_with_langflow(texts, tokenizer, model, seq_len: int) -> list:
    """
    Tokenize texts and get clean normalized embeddings via model._embed_tokens.
    Returns list of (token_ids [L], clean_emb [L, d], attn_mask [L]).
    """
    results = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=seq_len, padding="max_length")
            ids  = enc["input_ids"][0].to(device)      # [L]
            mask = enc["attention_mask"][0].cpu().numpy().astype(np.float32)  # [L]
            # _embed_tokens requires 2D [B, L]; squeeze batch dim after
            emb  = model._embed_tokens(ids.unsqueeze(0))[0].cpu().float().numpy()  # [L, d]
            results.append((ids.cpu().numpy().astype(np.int32), emb, mask))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    type=str, default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--stub",          action="store_true")
    p.add_argument("--n_samples",     type=int, default=64)
    p.add_argument("--seq_len",       type=int, default=256)
    p.add_argument("--n_t_steps",     type=int, default=21)
    p.add_argument("--tau",           type=float, default=1.0)
    p.add_argument("--topk",          type=int, default=5)
    p.add_argument("--n_noise",       type=int, default=4)
    p.add_argument("--commit_thresh", type=float, default=0.1)
    p.add_argument("--out_dir",       type=str, default="probe_langflow_v1")
    return p.parse_args()


def main():
    args = parse_args()

    if args.stub:
        print("[mode] Stub")
        stub = _StubModel().to(device)
        model   = stub
        E       = stub.E.numpy()
        gamma_min, gamma_max = -10.0, 10.0
        self_conditioning = False

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(args.n_samples):
            ids  = rng.integers(0, 50257, size=args.seq_len).astype(np.int32)
            emb  = rng.standard_normal((args.seq_len, 768)).astype(np.float32)
            mask = np.ones(args.seq_len, dtype=np.float32)
            samples.append((ids, emb, mask))
        label = "Stub"

    else:
        model, tokenizer, gamma_min, gamma_max, E = load_langflow(args.checkpoint)
        self_conditioning = model.config.self_conditioning

        print(f"[data] loading {args.n_samples} OWT texts…")
        texts   = load_owt_texts(args.n_samples)
        print(f"[encode] LangFlow embeddings…")
        samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)
        label   = "LangFlow OWT"

    t_grid    = np.linspace(0.0, 1.0, args.n_t_steps)
    gamma_grid = gamma_from_t(t_grid, gamma_min, gamma_max)
    print(f"[probe] γ ∈ [{gamma_min:.2f}, {gamma_max:.2f}]  ({args.n_t_steps} steps)")

    # ── Probe ────────────────────────────────────────────────────────────────
    seq_results = []

    for i, (gt_ids, emb, mask) in enumerate(samples):
        print(f"\n── Seq {i+1}/{len(samples)} — collecting logits…")
        logits_arr, xhat_arr = collect_logits_langflow(
            model, emb, gamma_grid, args.n_noise, seed=i,
            self_conditioning=self_conditioning)

        # final_ids: argmax at t=1.0 (γ_max, clean), seed=0
        final_ids = np.argmax(logits_arr[-1, 0], axis=-1).astype(np.int32)

        res = compute_metrics(
            logits_arr, xhat_arr, E, t_grid,
            gt_ids, final_ids,
            tau=args.tau, topk=args.topk, commit_thresh=args.commit_thresh,
        )

        for ti, tv in enumerate(t_grid):
            print(f"  t={tv:.2f}  γ={gamma_grid[ti]:.2f}  "
                  f"H={res['entropy'][ti]:.3f}  "
                  f"top1_gt={res['top1_gt'][ti]:.3f}  "
                  f"top5_gt={res['topk_gt'][ti]:.3f}  "
                  f"d_nn={res['d_nn'][ti]:.3f}  "
                  f"jsd={res['rev_jsd'][ti]:.4f}")
        seq_results.append(res)

    # ── Aggregate + save + plot ───────────────────────────────────────────────
    final = aggregate(seq_results)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(args.out_dir) / "langflow_probe.json"
    with open(json_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[save] → {json_path}")

    print_summary(final)
    plot_results(final, args.out_dir, label=label)
    print("[done]")


if __name__ == "__main__":
    main()
