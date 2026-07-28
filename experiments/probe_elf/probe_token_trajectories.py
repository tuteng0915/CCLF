"""
probe_token_trajectories.py — Real per-position token trajectory analysis for ELF-B.

For each token position in each sample, we track cos(x̂_t, E_gt) across a fixed
noise trajectory (same ε for all t). Positions are classified into 4 types based
on their commitment pattern:

  A. Correct Early Commit  — commits to correct token before t=0.10
  B. Wrong → Recommit      — committed wrong at t=0.10, correct at t=0.70
  C. Late Commit            — uncommitted at t=0.10, correct at t=0.70
  D. Stays Wrong            — not correct at t=0.70

Outputs:
  probe_token_traj.json  — per-type mean±std cos_gt(t) curves + fraction stats
  fig8_token_trajectories.png — figure with real measured data
"""

import sys, os, argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
_SCRIPT = Path(__file__).parent
_ELF_SRC = _SCRIPT.parents[1] / "models" / "ELF" / "src"
assert _ELF_SRC.exists(), f"ELF src not found: {_ELF_SRC}"
if str(_ELF_SRC) not in sys.path:
    sys.path.insert(0, str(_ELF_SRC))


# ── type labels and colors ─────────────────────────────────────────────────────
TYPE_LABELS = {
    "A": "Correct Early Commit",
    "B": "Wrong → Recommit",
    "C": "Late Commit",
    "D": "Stays Wrong / Uncommitted",
}
TYPE_COLORS = {
    "A": "#2ca02c",   # green
    "B": "#ff7f0e",   # orange
    "C": "#1f77b4",   # blue
    "D": "#d62728",   # red
}


# ── geometry helpers ───────────────────────────────────────────────────────────

def cos_to_centroids(x_hat, E_norm, seen_mask):
    """
    x_hat:    [L, d]  denoised representation (raw, unnormalized)
    E_norm:   [V, d]  L2-normalized centroids
    seen_mask:[V] bool

    Returns:
      cos_mat: [L, V]  cosine similarity (unseen = -inf)
    """
    xn = x_hat / (np.linalg.norm(x_hat, axis=-1, keepdims=True) + 1e-9)
    cos = xn @ E_norm.T            # [L, V]
    cos[:, ~seen_mask] = -np.inf
    return cos


def per_position_metrics(x_hat, E_norm, seen_mask, gt_ids):
    """
    Returns dict of per-position arrays (shape [L]):
      cos_gt, cos_nn_id, cos1, cos2, cos_margin
    """
    L = x_hat.shape[0]
    cos = cos_to_centroids(x_hat, E_norm, seen_mask)   # [L, V]

    # nearest two centroids
    top2 = np.argpartition(cos, -2, axis=-1)[:, -2:]
    v0 = cos[np.arange(L), top2[:, 0]]
    v1 = cos[np.arange(L), top2[:, 1]]
    swap = v0 < v1
    top2[swap] = top2[swap][:, ::-1]
    cos1 = cos[np.arange(L), top2[:, 0]]   # best cosine
    cos2 = cos[np.arange(L), top2[:, 1]]   # 2nd best
    cos_nn_id = top2[:, 0]                  # nearest centroid id

    # cosine to ground truth centroid
    xn = x_hat / (np.linalg.norm(x_hat, axis=-1, keepdims=True) + 1e-9)
    cos_gt = (xn * E_norm[gt_ids]).sum(-1)  # [L]

    return {
        "cos_gt":     cos_gt,
        "cos_nn_id":  cos_nn_id,
        "cos1":       cos1,
        "cos2":       cos2,
        "cos_margin": cos1 - cos2,
    }


# ── main probe logic ───────────────────────────────────────────────────────────

def run_probe(args):
    import jax
    import jax.numpy as jnp

    # load ELF model
    sys.path.insert(0, str(_SCRIPT))
    from probe_anchor_v3 import load_elf, encode_with_t5, load_owt_texts

    print("[probe_token_traj] loading ELF-B...")
    (model, ema_params, encoder_params, encoder_model,
     tokenizer, config, _enc_config) = load_elf(
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

    def forward_fn(z_t, t, attn_mask):
        z_in = np.concatenate([z_t, np.zeros_like(z_t)], axis=-1) if has_sc else z_t
        out = _fwd(
            ema_params,
            jnp.array(z_in),
            jnp.array([t], dtype=jnp.float32),
            jnp.array(attn_mask, dtype=jnp.float32),
            sc_scale,
        )
        x_hat_b = np.array(out[0] if isinstance(out, tuple) else out)  # [1, L, d]
        return x_hat_b[0]  # [L, d]

    # load & encode data
    print("[data] loading OWT texts...")
    texts = load_owt_texts(n=args.n_samples)
    print("[data] encoding with T5...")
    samples = encode_with_t5(
        texts, tokenizer, encoder_model, encoder_params,
        seq_len=args.seq_len, latent_mean=latent_mean, latent_std=latent_std,
    )

    # load token centroids
    print("[embed] loading centroids...")
    data = np.load(args.centroid_path)
    E_all  = data["centroids"].astype(np.float32)    # [V, d]
    V = E_all.shape[0]
    counts = data.get("counts", np.ones(V, dtype=np.int32))
    seen_mask = counts > 0
    E_norm = E_all / (np.linalg.norm(E_all, axis=-1, keepdims=True) + 1e-9)

    t_grid = np.linspace(0.0, 1.0, args.n_t_steps)
    T = len(t_grid)

    # anchor time indices for classification
    t_early_idx = int(round(args.t_early * (T - 1)))
    t_late_idx  = int(round(args.t_late  * (T - 1)))
    margin_thresh = args.margin_thresh

    print(f"[config] t_grid: {T} steps, t_early={t_grid[t_early_idx]:.2f} "
          f"(idx={t_early_idx}), t_late={t_grid[t_late_idx]:.2f} (idx={t_late_idx})")
    print(f"[config] margin threshold for 'committed wrong': {margin_thresh}")

    rng = np.random.default_rng(seed=0)

    # storage: per position-trajectory, store cos_gt across all t
    # keys: type label → list of np arrays shape [T]
    type_trajs = {k: [] for k in "ABCD"}

    for si, sample in enumerate(samples):
        gt_ids, clean_emb, attn_mask = sample
        valid = attn_mask.astype(bool)
        gt_valid  = gt_ids[valid]
        L_valid   = int(valid.sum())
        L_full    = clean_emb.shape[0]

        mask_batch = attn_mask[None, :]  # [1, L]

        for traj_idx in range(args.n_traj):
            eps = rng.standard_normal((1, L_full, clean_emb.shape[-1])).astype(np.float32)

            # per-position arrays across t
            cos_gt_traj  = np.zeros((L_valid, T))   # cos to gt per position
            cos_nn_traj  = np.zeros((L_valid, T), dtype=np.int32)  # argmax nn id
            cos1_traj    = np.zeros((L_valid, T))   # cosine of nearest centroid

            for ti, t in enumerate(t_grid):
                z_t = t * clean_emb[None] + (1.0 - t) * eps  # [1, L, d]
                x_hat = forward_fn(z_t, float(t), mask_batch)  # [L, d]
                x_hat_valid = x_hat[valid]                       # [L_valid, d]

                m = per_position_metrics(x_hat_valid, E_norm, seen_mask, gt_valid)
                cos_gt_traj[:, ti]  = m["cos_gt"]
                cos_nn_traj[:, ti]  = m["cos_nn_id"]
                cos1_traj[:, ti]    = m["cos1"]

            # classify each valid position
            # "committed wrong" = wrong nearest centroid is ahead of GT by > threshold
            # (previously used cos1-cos2 which compared NN to 2nd-nearest, not GT)
            correct_early = (cos_nn_traj[:, t_early_idx] == gt_valid)
            correct_late  = (cos_nn_traj[:, t_late_idx]  == gt_valid)
            margin_over_gt_early = cos1_traj[:, t_early_idx] - cos_gt_traj[:, t_early_idx]
            committed_wrong_early = (
                (~correct_early) &
                (margin_over_gt_early > margin_thresh)
            )

            for pos in range(L_valid):
                traj = cos_gt_traj[pos]  # [T]
                if correct_early[pos]:
                    type_trajs["A"].append(traj)
                elif correct_late[pos]:
                    if committed_wrong_early[pos]:
                        type_trajs["B"].append(traj)
                    else:
                        type_trajs["C"].append(traj)
                else:
                    type_trajs["D"].append(traj)

        total = sum(len(v) for v in type_trajs.values())
        print(f"  sample {si+1}/{len(samples)}  "
              f"A={len(type_trajs['A'])}  B={len(type_trajs['B'])}  "
              f"C={len(type_trajs['C'])}  D={len(type_trajs['D'])}  "
              f"({total} pos-trajs)")

    # aggregate
    results = {"t": t_grid.tolist()}
    fracs = {}
    total = sum(len(v) for v in type_trajs.values())
    for k in "ABCD":
        arr = np.array(type_trajs[k])   # [N_k, T]
        n_k = len(arr)
        fracs[k] = n_k / total if total > 0 else 0.0
        if n_k > 0:
            results[f"{k}_mean"] = arr.mean(axis=0).tolist()
            results[f"{k}_std"]  = arr.std(axis=0).tolist()
            results[f"{k}_n"]    = n_k
        else:
            results[f"{k}_mean"] = [float("nan")] * T
            results[f"{k}_std"]  = [float("nan")] * T
            results[f"{k}_n"]    = 0
        print(f"  Type {k} ({TYPE_LABELS[k]}): {n_k} pos-trajs ({fracs[k]*100:.1f}%)")

    results["fracs"] = fracs
    results["args"] = vars(args)
    return results, t_grid


def plot(results, t_grid, out_path, cliff_t=0.30):
    t = np.array(t_grid)
    total = sum(results["fracs"].values())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    fig.suptitle(
        "Token Trajectory Types in ELF-B  (measured per-position, cosine to correct centroid)",
        fontsize=11, y=1.02)

    SHOW = ["A", "B", "C"]  # 3 main story types; D is residual
    type_order_labels = {
        "A": "(a) Correct Early Commit",
        "B": "(b) Wrong → Recommit",
        "C": "(c) Late Commit",
    }

    for ax, k in zip(axes, SHOW):
        mean = np.array(results[f"{k}_mean"])
        std  = np.array(results[f"{k}_std"])
        n    = results[f"{k}_n"]
        frac = results["fracs"][k]
        color = TYPE_COLORS[k]

        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.20)
        ax.plot(t, mean, color=color, lw=2.2, label=f"mean ± 1σ  (n={n})")

        ax.axvline(cliff_t, color="black", ls="--", lw=0.9, alpha=0.5)
        ax.axhline(0.6, color="grey", ls=":", lw=0.7, alpha=0.6)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("$t$  (noise → clean)", fontsize=9)
        ax.set_title(f"{type_order_labels[k]}\n~{frac*100:.0f}% of positions",
                     fontsize=9.5, color=color)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("cos$(\\hat{x}_t,\\ E_{\\mathrm{gt}})$", fontsize=9)

    # Add Type D info as text annotation on last panel
    d_frac = results["fracs"]["D"]
    d_n    = results["D_n"]
    axes[-1].text(
        0.97, 0.05,
        f"Type D (Stays Wrong)\n~{d_frac*100:.0f}% of positions (n={d_n})\nnot shown",
        ha="right", va="bottom", fontsize=7, color=TYPE_COLORS["D"],
        transform=axes[-1].transAxes,
        bbox=dict(fc="white", ec=TYPE_COLORS["D"], alpha=0.7, lw=0.8, pad=3),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="embedded-language-flows/ELF-B-owt")
    p.add_argument("--config",         default=str(_ELF_SRC / "configs/training_configs/train_owt_ELF-B.yml"))
    p.add_argument("--centroid_path",  default="results/data/token_centroids.npz")
    p.add_argument("--n_samples",      type=int,   default=48)
    p.add_argument("--n_traj",         type=int,   default=3,
                   help="noise trajectories per sample")
    p.add_argument("--seq_len",        type=int,   default=128)
    p.add_argument("--n_t_steps",      type=int,   default=21)
    p.add_argument("--t_early",        type=float, default=0.10)
    p.add_argument("--t_late",         type=float, default=0.70)
    p.add_argument("--margin_thresh",  type=float, default=0.03,
                   help="cos margin threshold to consider 'committed' at t_early")
    p.add_argument("--out_dir",        default="results/elf/probe_token_traj_v1")
    p.add_argument("--gpu",            type=int,   default=7)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_FLAGS",
        "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false")

    import jax
    print(f"JAX devices: {jax.devices()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, t_grid = run_probe(args)

    json_out = out_dir / "probe_token_traj.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {json_out}")

    # save to figures dir too
    fig_out_main = out_dir / "fig8_token_trajectories.png"
    fig_out_copy = _ELF_SRC / "outputs/figures/fig8_token_trajectories.png"

    plot(results, t_grid, fig_out_main)
    plot(results, t_grid, fig_out_copy)


if __name__ == "__main__":
    main()
