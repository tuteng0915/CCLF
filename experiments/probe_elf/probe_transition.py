"""
probe_transition.py — ELF Per-Step Commitment Transition Probe

For each consecutive pair (t_i → t_{i+1}) along a FIXED noise trajectory,
measures at every token position:
  G(t)    : cos_nn_correct — does x̂_t point to the right centroid?
  w2c(t)  : fraction of positions wrong at t_i that flip correct at t_{i+1}
  c2w(t)  : fraction of positions correct at t_i that flip wrong at t_{i+1}
  net(t)  : w2c - c2w (net per-step gain)

Unlike probe_geo.py which draws independent ε per t, here ONE ε is fixed for
the whole trajectory so that per-position predictions are comparable across t.

Usage:
  cd /home/wjzhang/tt_workspace/model/CCLF/CCLF
  conda run -n elf python experiments/probe_elf/probe_transition.py \\
      --checkpoint embedded-language-flows/ELF-B-owt \\
      --config models/ELF/src/configs/training_configs/train_owt_ELF-B.yml \\
      --centroid_path results/data/token_centroids.npz \\
      --n_samples 64 --seq_len 128 --n_t_steps 21 --n_traj 4 \\
      --out_dir results/elf/probe_transition_v1 \\
      --gpu 5
"""

import sys, os, argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ELF source path ───────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_ELF_SRC    = _SCRIPT_DIR / "src"
if not _ELF_SRC.exists():
    _ELF_SRC = _SCRIPT_DIR.parents[1] / "models" / "ELF" / "src"
ELF_SRC       = str(_ELF_SRC)
DEFAULT_CONFIG = str(_ELF_SRC / "configs" / "training_configs" / "train_owt_ELF-B.yml")
if ELF_SRC not in sys.path:
    sys.path.insert(0, ELF_SRC)
# probe_anchor_v3 lives alongside this script
sys.path.insert(0, str(_SCRIPT_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# ELF + data loading  (mirrors probe_geo.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_elf_and_data(args):
    import jax, jax.numpy as jnp
    from probe_anchor_v3 import load_elf, encode_with_t5, load_owt_texts

    (model, ema_params, encoder_params, encoder_model,
     tokenizer, config, enc_config) = load_elf(
         args.config, args.checkpoint,
         override_max_length=args.seq_len)

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

    # Returns only x_hat (logits not needed for geometric transition probe).
    def forward_fn(z_t, t, attn_mask):
        z_in = np.concatenate([z_t, np.zeros_like(z_t)], axis=-1) if has_sc else z_t
        x_hat_b, _logits = _fwd(
            ema_params,
            jnp.array(z_in),
            jnp.array([t], dtype=jnp.float32),
            jnp.array(attn_mask, dtype=jnp.float32),
            sc_scale,
        )
        return np.array(x_hat_b[0])   # [L, d]

    print("[data] loading OWT texts…")
    texts   = load_owt_texts(args.n_samples)
    print("[encode] T5 encoder…")
    samples = encode_with_t5(texts, tokenizer, encoder_model, encoder_params,
                             args.seq_len, latent_mean, latent_std)

    data = np.load(args.centroid_path)
    E_raw = data["centroids"].astype(np.float32)
    latent_std_file = float(data["latent_std"]) if "latent_std" in data else None
    already_norm = (latent_std_file is not None
                    and np.isclose(latent_std_file, float(latent_std)))
    vocab_size = tokenizer.vocab_size
    E_all  = E_raw[:vocab_size] if already_norm else E_raw[:vocab_size] / latent_std
    if "counts" in data:
        seen_mask = data["counts"][:vocab_size] > 0
    else:
        seen_mask = np.ones(vocab_size, dtype=bool)
    E_norm = E_all / (np.linalg.norm(E_all, axis=-1, keepdims=True) + 1e-9)

    print(f"[embed] {E_all.shape}  seen={int(seen_mask.sum())}/{vocab_size}")
    return forward_fn, samples, E_norm, seen_mask


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def cos_nn_ids(x_hat: np.ndarray, E_norm: np.ndarray,
               seen_mask: np.ndarray) -> np.ndarray:
    """argmax_v cos(x̂, E_v) restricted to seen tokens. Shape: [L]."""
    x_n    = x_hat / (np.linalg.norm(x_hat, axis=-1, keepdims=True) + 1e-9)
    cos    = x_n @ E_norm.T          # [L, V]
    cos[:, ~seen_mask] = -np.inf
    return cos.argmax(axis=-1)        # [L]


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample trajectory probe
# ─────────────────────────────────────────────────────────────────────────────

def probe_sample(forward_fn, sample, E_norm, seen_mask,
                 t_grid, n_traj, seed) -> dict:
    """
    For one sample, runs n_traj fixed-noise trajectories and returns:
      G    : [n_t_steps]     cos_nn_correct at each t
      w2c  : [n_t_steps-1]   wrong→correct rate at each consecutive step
      c2w  : [n_t_steps-1]   correct→wrong rate at each consecutive step
      net  : [n_t_steps-1]   w2c - c2w
    (each is the mean across trajectories)
    """
    gt_ids, clean_emb, attn_mask = sample
    valid = attn_mask.astype(bool)   # [L]
    L, d  = clean_emb.shape
    rng   = np.random.default_rng(seed)

    G_mat      = np.zeros((n_traj, len(t_grid)))
    cos_gt_mat = np.zeros((n_traj, len(t_grid)))  # soft: cos(x̂_t, E_gt) per position
    w2c_mat    = np.zeros((n_traj, len(t_grid) - 1))
    c2w_mat    = np.zeros((n_traj, len(t_grid) - 1))

    mask_batch = attn_mask[None]   # [1, L]
    gt_valid   = gt_ids[valid]     # [n_valid]

    for traj_idx in range(n_traj):
        # Fix ONE noise realization for the entire t-trajectory
        eps = rng.standard_normal((1, L, d)).astype(np.float32)

        pred_list = []   # cos_nn_ids at each t, shape [L]

        for ti, t in enumerate(t_grid):
            z_t  = t * clean_emb[None] + (1.0 - t) * eps   # [1, L, d]
            x_hat = forward_fn(z_t, float(t), mask_batch)   # [L, d]

            ids = cos_nn_ids(x_hat, E_norm, seen_mask)      # [L]
            pred_list.append(ids)
            G_mat[traj_idx, ti] = (ids[valid] == gt_valid).mean()

            # Soft metric: cos(x̂_t, E_{gt}) — directional alignment to correct centroid
            x_n_valid = x_hat[valid] / (
                np.linalg.norm(x_hat[valid], axis=-1, keepdims=True) + 1e-9)  # [n_valid, d]
            cos_gt_mat[traj_idx, ti] = (x_n_valid * E_norm[gt_valid]).sum(-1).mean()

        for ti in range(len(t_grid) - 1):
            cur  = pred_list[ti][valid]
            nxt  = pred_list[ti + 1][valid]
            right_cur  = cur  == gt_valid
            right_next = nxt == gt_valid
            w2c_mat[traj_idx, ti] = (~right_cur &  right_next).mean()
            c2w_mat[traj_idx, ti] = ( right_cur & ~right_next).mean()

    return {
        "G":       G_mat.mean(0).tolist(),
        "G_sem":   (G_mat.std(0) / np.sqrt(n_traj)).tolist(),
        "cos_gt":  cos_gt_mat.mean(0).tolist(),
        "w2c":     w2c_mat.mean(0).tolist(),
        "c2w":     c2w_mat.mean(0).tolist(),
        "net":     (w2c_mat - c2w_mat).mean(0).tolist(),
    }


def aggregate(seq_results: list) -> dict:
    """Average per-sample dicts across samples."""
    keys = ["G", "G_sem", "cos_gt", "w2c", "c2w", "net"]
    out  = {}
    for k in keys:
        mat = np.array([s[k] for s in seq_results])   # [N_samples, T]
        out[f"{k}_mean"] = mat.mean(0).tolist()
        out[f"{k}_sem"]  = (mat.std(0) / np.sqrt(len(seq_results))).tolist()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(agg: dict, t_grid: np.ndarray, out_dir: Path):
    tm     = (t_grid[:-1] + t_grid[1:]) / 2   # midpoint of each step
    G      = np.array(agg["G_mean"])
    G_se   = np.array(agg["G_sem_mean"])       # SEM of within-traj SEM, across samples
    cos_gt = np.array(agg["cos_gt_mean"])      # soft metric: cos(x̂_t, E_gt)
    w2c    = np.array(agg["w2c_mean"])
    c2w    = np.array(agg["c2w_mean"])
    net    = np.array(agg["net_mean"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ELF-B — Per-Step Commitment Transitions (Fixed Trajectory)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: G(t) hard + cos_gt soft ─────────────────────────────────────
    ax = axes[0]
    ax.plot(t_grid, G * 100, lw=2, color="#e05c5c", label="G(t)  hard top-1")
    ax.fill_between(t_grid,
                    (G - G_se) * 100, (G + G_se) * 100,
                    alpha=0.2, color="#e05c5c")
    # cos_gt ranges roughly [-1, 1]; rescale to [0, 100] for overlay
    # map: -1 → 0%, 1 → 100%  i.e. (cos_gt + 1) / 2 * 100
    cos_gt_pct = (cos_gt + 1) / 2 * 100
    ax.plot(t_grid, cos_gt_pct, lw=2, color="#9467bd", linestyle="--",
            label="cos(x̂, E_gt) soft [rescaled]")
    ax.set_xlabel("t  (0=noisy → 1=clean)")
    ax.set_ylabel("Commitment (%)")
    ax.set_title("Hard G(t) vs Soft cos(x̂, E_gt)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # ── Panel 2: w2c and c2w per step ─────────────────────────────────────────
    ax = axes[1]
    ax.plot(tm, w2c * 100, lw=2, color="#2ca02c", label="w→c (gain)")
    ax.plot(tm, c2w * 100, lw=2, color="#d62728", linestyle="--", label="c→w (loss)")
    ax.fill_between(tm, 0, w2c * 100, alpha=0.15, color="#2ca02c")
    ax.fill_between(tm, 0, c2w * 100, alpha=0.15, color="#d62728")

    peak_idx = int(np.argmax(w2c))
    ax.axvline(tm[peak_idx], color="#2ca02c", linestyle=":", linewidth=1.5)
    ax.annotate(
        f"peak w→c\nt={tm[peak_idx]:.2f}\n{w2c[peak_idx]*100:.1f}%",
        xy=(tm[peak_idx], w2c[peak_idx] * 100),
        xytext=(min(tm[peak_idx] + 0.07, 0.88), w2c[peak_idx] * 100 + 0.5),
        fontsize=8, color="#2ca02c",
    )
    ax.set_xlabel("t  (step midpoint)")
    ax.set_ylabel("Fraction of positions (%)")
    ax.set_title("Per-Step Transition Rates")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.grid(True, alpha=0.3)

    # ── Panel 3: net commitment gain per step (bar chart) ─────────────────────
    ax = axes[2]
    bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in net]
    step = t_grid[1] - t_grid[0]
    ax.bar(tm, net * 100, width=step * 0.8, color=bar_colors, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("t  (step midpoint)")
    ax.set_ylabel("Net gain (w→c minus c→w) (%)")
    ax.set_title("Net Commitment Gain per Step")
    ax.set_xlim(0, 1); ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = out_dir / "probe_transition.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="embedded-language-flows/ELF-B-owt")
    p.add_argument("--config",        default=DEFAULT_CONFIG)
    p.add_argument("--centroid_path", default="results/data/token_centroids.npz")
    p.add_argument("--n_samples",     type=int, default=64)
    p.add_argument("--seq_len",       type=int, default=128)
    p.add_argument("--n_t_steps",     type=int, default=21,
                   help="Number of t values (linspace 0..1); gives n_t_steps-1 transitions")
    p.add_argument("--n_traj",        type=int, default=4,
                   help="Fixed-noise trajectories per sample; results averaged")
    p.add_argument("--out_dir",       default="results/elf/probe_transition_v1")
    p.add_argument("--gpu",           type=int, default=None,
                   help="GPU index to use (sets CUDA_VISIBLE_DEVICES before JAX loads)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import jax   # imported after GPU env is set
    print(f"JAX devices: {jax.devices()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_grid = np.linspace(0.0, 1.0, args.n_t_steps)

    print("[probe_transition] loading ELF-B-owt…")
    forward_fn, samples, E_norm, seen_mask = load_elf_and_data(args)

    seq_results = []
    for i, sample in enumerate(samples):
        print(f"\n── sample {i+1}/{len(samples)}")
        res = probe_sample(
            forward_fn, sample, E_norm, seen_mask,
            t_grid=t_grid, n_traj=args.n_traj, seed=i * 997,
        )
        seq_results.append(res)

        G   = np.array(res["G"])
        w2c = np.array(res["w2c"])
        c2w = np.array(res["c2w"])
        tm  = (t_grid[:-1] + t_grid[1:]) / 2
        idx30 = np.argmin(np.abs(t_grid - 0.30))
        print(f"  G(0.30)={G[idx30]*100:.1f}%  "
              f"w2c_peak@t={tm[w2c.argmax()]:.2f}({w2c.max()*100:.2f}%)  "
              f"max_c2w={c2w.max()*100:.2f}%")

    agg = aggregate(seq_results)
    agg["t"]     = t_grid.tolist()
    agg["t_mid"] = ((t_grid[:-1] + t_grid[1:]) / 2).tolist()
    agg["args"]  = vars(args)

    out_json = out_dir / "probe_transition.json"
    with open(out_json, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[saved] {out_json}")

    plot_results(agg, t_grid, out_dir)

    # ── console summary table ──────────────────────────────────────────────────
    tm_arr  = np.array(agg["t_mid"])
    G_arr   = np.array(agg["G_mean"])
    w2c_arr = np.array(agg["w2c_mean"])
    c2w_arr = np.array(agg["c2w_mean"])
    net_arr = np.array(agg["net_mean"])

    print(f"\n── Transition summary {'─'*42}")
    print(f"{'step':>11}  {'G_end%':>6}  {'w→c%':>6}  {'c→w%':>6}  {'net%':>6}")
    for i, tm_val in enumerate(tm_arr):
        print(f"{t_grid[i]:.2f}→{t_grid[i+1]:.2f}   "
              f"{G_arr[i+1]*100:6.1f}  "
              f"{w2c_arr[i]*100:6.2f}  "
              f"{c2w_arr[i]*100:6.2f}  "
              f"{net_arr[i]*100:6.2f}")


if __name__ == "__main__":
    main()
