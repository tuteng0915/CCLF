"""
probe_geo_langflow.py — LangFlow Geometric Commitment Probe

LangFlow operates in token embedding space (GPT-2 embeddings).
The noise process is variance-preserving:
  z_t = α(γ) · E[gt] + σ(γ) · ε
where γ = log-SNR, t=0 noisy (large γ), t=1 clean (small γ).

Key difference from ELF: LangFlow backbone has NO separate x̂_t in D-space.
It maps z_t → logits directly. So we use z_t as the geometric representation.

"x_clean" for LangFlow = E[gt_id] (the token embedding of the correct token).

This directly parallels ELF's probe_geo.py where:
  ELF: x̂_t = backbone(z_t)       in T5 contextual space
  LangFlow: z_t (the input)       in token embedding space

Metrics computed (same names as probe_geo.py for direct comparison):
  cos_to_clean     cos(z_t[i], E[gt_i])  — SNR-driven convergence
  cos_nn_max       max_v cos(z_t[i], E_v)
  cos_nn_correct   argmax_v cos(z_t[i], E_v) == gt_id
  cos_margin       cos_1st - cos_2nd
  l2_residual_frac ||z_t - E^T p|| / ||z_t||  — anchor residual (L_anc analog)
  d_nn_l2          min_v ||z_t - E_v||  — decreases toward t=1 (cf. ELF: increases)
  top1_gt_decoder  decoder top-1 (from logits)  — for cross-reference

Usage:
  conda run -n elf python experiments/probe_langflow/probe_geo_langflow.py \\
      --n_samples 64 --seq_len 128 --n_t_steps 21 --n_noise 4 \\
      --out_dir results/langflow/probe_geo_v1
"""

import sys, os, argparse, json, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── LangFlow path (mirrors probe_langflow.py) ────────────────────────────────
LANGFLOW_REPO = os.path.expanduser("~/LangFlow")
_PROBE_DIR    = Path(__file__).parent
_LF_SRC       = _PROBE_DIR.parents[1] / "models" / "LangFlow"
for _p in [LANGFLOW_REPO, str(_LF_SRC)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reuse loaders from probe_langflow.py
sys.path.insert(0, str(_PROBE_DIR))
from probe_langflow import (
    load_langflow, encode_with_langflow, load_owt_texts,
    gamma_from_t, softmax_np, token_entropy,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometry metrics (parallel to probe_geo.py)
# ─────────────────────────────────────────────────────────────────────────────

def geo_metrics_langflow(
    z_t:      np.ndarray,   # [L, d]  noisy input (serves as x̂_t for LangFlow)
    x_clean:  np.ndarray,   # [L, d]  E[gt_id] — clean token embedding per position
    E_norm:   np.ndarray,   # [V, d]  L2-normalized token embedding matrix
    E_all:    np.ndarray,   # [V, d]  raw token embedding matrix
    p:        np.ndarray,   # [L, V]  decoder distribution
    gt_ids:   np.ndarray,   # [L]     ground truth token ids
) -> dict:
    L = len(z_t)

    # ── cos(z_t, x_clean) ─────────────────────────────────────────────────────
    zt_n = z_t   / (np.linalg.norm(z_t,   axis=-1, keepdims=True) + 1e-9)
    xc_n = x_clean / (np.linalg.norm(x_clean, axis=-1, keepdims=True) + 1e-9)
    cos_to_clean = (zt_n * xc_n).sum(-1)   # [L]

    # ── cosine NN to token embeddings ─────────────────────────────────────────
    cos_to_E = zt_n @ E_norm.T              # [L, V]  (E_norm already L2 normalized)

    top2_idx = np.argpartition(cos_to_E, -2, axis=-1)[:, -2:]
    v0 = cos_to_E[np.arange(L), top2_idx[:, 0]]
    v1 = cos_to_E[np.arange(L), top2_idx[:, 1]]
    swap = v0 < v1
    top2_idx[swap] = top2_idx[swap][:, ::-1]
    cos1 = cos_to_E[np.arange(L), top2_idx[:, 0]]
    cos2 = cos_to_E[np.arange(L), top2_idx[:, 1]]

    cos_nn_id      = top2_idx[:, 0]
    cos_nn_correct = (cos_nn_id == gt_ids).astype(float)
    cos_margin     = cos1 - cos2

    # ── anchor residual ||z_t - E^T p|| / ||z_t|| ────────────────────────────
    anchor   = p @ E_all                               # [L, d]
    residual = z_t - anchor
    res_norm  = np.linalg.norm(residual, axis=-1)
    zt_norm   = np.linalg.norm(z_t,     axis=-1)
    l2_residual_frac = res_norm / (zt_norm + 1e-9)

    # ── L2 d_nn: should DECREASE for LangFlow (valid commitment metric) ───────
    E_sq   = (E_all ** 2).sum(-1)
    z_sq   = (z_t ** 2).sum(-1)
    dists_sq = np.maximum(z_sq[:, None] + E_sq[None, :] - 2.0 * (z_t @ E_all.T), 0.0)
    d_nn_l2 = float(np.sqrt(dists_sq.min(-1)).mean())

    return {
        "cos_to_clean":       float(cos_to_clean.mean()),
        "cos_to_clean_p10":   float(np.percentile(cos_to_clean, 10)),
        "cos_to_clean_p50":   float(np.percentile(cos_to_clean, 50)),
        "cos_to_clean_p90":   float(np.percentile(cos_to_clean, 90)),
        "cos_nn_max":         float(cos1.mean()),
        "cos_nn_correct":     float(cos_nn_correct.mean()),
        "cos_margin":         float(cos_margin.mean()),
        "l2_residual_frac":   float(l2_residual_frac.mean()),
        "d_nn_l2":            d_nn_l2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Main probe loop
# ─────────────────────────────────────────────────────────────────────────────

ALL_METRICS = [
    "cos_to_clean", "cos_to_clean_p10", "cos_to_clean_p50", "cos_to_clean_p90",
    "cos_nn_max", "cos_nn_correct", "cos_margin",
    "l2_residual_frac", "d_nn_l2",
    "top1_gt_decoder",
    "entropy_decoder",
]


def probe_sample(
    model, sample, E_all, E_norm,
    t_grid, gamma_grid, n_noise, seed, tau, self_conditioning
) -> dict:
    gt_ids, clean_emb, attn_mask = sample   # clean_emb = E[gt_ids] here
    L, d = clean_emb.shape
    rng  = np.random.default_rng(seed)

    out = {k: [] for k in ALL_METRICS}
    out["t"] = t_grid.tolist()

    x_torch = torch.from_numpy(clean_emb).to(device, dtype=torch.float32)

    for ti, (t, gamma) in enumerate(zip(t_grid, gamma_grid)):
        alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma)).item())
        sigma = math.sqrt(torch.sigmoid(torch.tensor( gamma)).item())

        seed_metrics = {k: [] for k in ALL_METRICS}

        for _ in range(n_noise):
            eps   = rng.standard_normal((1, L, d)).astype(np.float32)
            eps_t = torch.from_numpy(eps).to(device)
            z_t_torch = alpha * x_torch[None] + sigma * eps_t   # [1, L, d]

            gamma_t = torch.full((1,), gamma, device=device, dtype=torch.float32)
            sc = torch.zeros_like(z_t_torch) if self_conditioning else None

            with torch.no_grad():
                out_model = model(
                    noisy_embeds=z_t_torch,
                    timesteps=gamma_t,
                    x_self_cond=sc,
                    return_dict=False,
                )
            logits = out_model[0] if isinstance(out_model, (tuple, list)) else out_model
            logits_np = logits[0].cpu().float().numpy()    # [L, V]
            z_np      = z_t_torch[0].cpu().float().numpy() # [L, d]

            p = softmax_np(logits_np, tau)                 # [L, V]

            g = geo_metrics_langflow(z_np, clean_emb, E_norm, E_all, p, gt_ids)
            for k, v in g.items():
                if k in seed_metrics:
                    seed_metrics[k].append(v)

            seed_metrics["top1_gt_decoder"].append(
                float((np.argmax(p, -1) == gt_ids).mean()))
            seed_metrics["entropy_decoder"].append(
                float(token_entropy(p).mean()))

        for k in ALL_METRICS:
            out[k].append(float(np.nanmean(seed_metrics[k])))

    return out


def aggregate(seq_results):
    out = {"t": seq_results[0]["t"]}
    for k in ALL_METRICS:
        mat = np.array([s[k] for s in seq_results])
        out[f"{k}_mean"] = np.nanmean(mat, axis=0).tolist()
        out[f"{k}_std"]  = np.nanstd(mat,  axis=0).tolist()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(agg, out_dir):
    t = np.array(agg["t"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("LangFlow Geometric Commitment Probe (cosine-based, n=64)", fontsize=11)

    # Panel 1: cos_to_clean vs cos_nn_correct vs top1_dec
    ax = axes[0]
    ax.plot(t, agg["cos_to_clean_mean"],   lw=2, color="#FF9800", label="cos(z_t, E[gt])")
    ax.plot(t, agg["cos_nn_correct_mean"], lw=2, color="#4CAF50", label="cosine NN accuracy")
    ax.plot(t, agg["top1_gt_decoder_mean"],lw=2, color="#2196F3", ls="--", label="decoder top-1")
    ax.set_title("(a) Geometric vs decoder commitment", fontsize=9)
    ax.set_xlabel("t  (0=noisy, 1=clean)", fontsize=8)
    ax.set_ylabel("accuracy / cosine sim", fontsize=8)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

    # Panel 2: cos_margin
    ax = axes[1]
    m  = np.array(agg["cos_margin_mean"])
    ms = np.array(agg["cos_margin_std"])
    ax.plot(t, m, lw=2.5, color="#FF5722")
    ax.fill_between(t, m - ms, m + ms, alpha=0.2, color="#FF5722")
    ax.set_title("(b) Cosine margin (cos_1st − cos_2nd)", fontsize=9)
    ax.set_xlabel("t  (0=noisy, 1=clean)", fontsize=8)
    ax.set_ylabel("cosine margin", fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

    # Panel 3: l2_residual_frac and d_nn_l2 (normalized)
    ax = axes[2]
    res = np.array(agg["l2_residual_frac_mean"])
    dnn = np.array(agg["d_nn_l2_mean"])
    ax.plot(t, res,           lw=2, color="#F44336", label="||z_t−E^Tp||/||z_t|| (anchor residual)")
    ax.plot(t, dnn / dnn.max(), lw=2, color="gray",  ls="--", label="L2 d_nn (normalized)")
    ax.set_title("(c) Anchor residual fraction", fontsize=9)
    ax.set_xlabel("t  (0=noisy, 1=clean)", fontsize=8)
    ax.set_ylabel("fraction / normalized", fontsize=8)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

    fig.tight_layout()
    out_path = out_dir / "probe_geo_langflow.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--n_samples",  type=int, default=64)
    p.add_argument("--seq_len",    type=int, default=128)
    p.add_argument("--n_t_steps",  type=int, default=21)
    p.add_argument("--n_noise",    type=int, default=4)
    p.add_argument("--tau",        type=float, default=1.0)
    p.add_argument("--out_dir",    default="results/langflow/probe_geo_v1")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_grid  = np.linspace(0.0, 1.0, args.n_t_steps)

    print("[probe_geo_langflow] loading model…")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    gamma_grid = gamma_from_t(t_grid, gamma_min, gamma_max)

    # L2-normalize E for cosine
    E_norm = E_all / (np.linalg.norm(E_all, axis=-1, keepdims=True) + 1e-9)
    print(f"[embed] E: {E_all.shape}")

    print(f"[data] loading {args.n_samples} OWT texts…")
    texts   = load_owt_texts(args.n_samples)
    samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)

    sc = model.config.self_conditioning

    seq_results = []
    for i, sample in enumerate(samples):
        print(f"\n── sample {i+1}/{len(samples)}")
        res = probe_sample(
            model, sample, E_all, E_norm,
            t_grid, gamma_grid, args.n_noise,
            seed=i * 1000, tau=args.tau, self_conditioning=sc,
        )
        seq_results.append(res)
        for ti, tv in enumerate(t_grid):
            print(f"  t={tv:.2f}  "
                  f"cos_clean={res['cos_to_clean'][ti]:.3f}  "
                  f"cos_nn_cor={res['cos_nn_correct'][ti]:.3f}  "
                  f"cos_margin={res['cos_margin'][ti]:.4f}  "
                  f"top1_dec={res['top1_gt_decoder'][ti]:.3f}  "
                  f"res_frac={res['l2_residual_frac'][ti]:.3f}")

    agg = aggregate(seq_results)
    agg["args"] = vars(args)

    out_json = out_dir / "probe_geo_langflow.json"
    with open(out_json, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[saved] {out_json}")

    plot_results(agg, out_dir)

    print(f"\n── Summary ──────────────────────────────────────────────────────────")
    print(f"{'t':>5}  {'cos_clean':>9}  {'cos_nn_cor':>10}  "
          f"{'cos_margin':>10}  {'top1_dec':>8}  {'res_frac':>8}  {'d_nn_l2':>7}")
    for ti, tv in enumerate(t_grid):
        def g(k): return agg[f"{k}_mean"][ti]
        print(f"{tv:>5.2f}  {g('cos_to_clean'):>9.3f}  {g('cos_nn_correct'):>10.3f}  "
              f"{g('cos_margin'):>10.4f}  {g('top1_gt_decoder'):>8.3f}  "
              f"{g('l2_residual_frac'):>8.3f}  {g('d_nn_l2'):>7.3f}")


if __name__ == "__main__":
    main()
