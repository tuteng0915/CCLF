"""
probe_geo.py — ELF Geometric Commitment Probe (cosine-based)

Addresses the limitation of L2 d_nn in contextual embedding space.
In T5 contextual space, d_nn (L2 to centroid) INCREASES toward t=1 because
clean representations are more context-specific than their average centroids.
Cosine similarity is scale-invariant and better captures directional alignment.

Five new metrics per position, per t:
  cos_to_clean     cos(x̂_t[i], x_clean[i])
                   How well has the backbone recovered the clean T5 encoding?
                   Directly measures denoising quality in contextual space.

  cos_nn_max       max_v  cos(x̂_t[i], E_v)    (seen tokens only)
                   Cosine analog of -d_nn. Higher = more clearly pointing to
                   some specific token centroid.

  cos_nn_correct   fraction of positions where argmax_v cos(x̂_t[i], E_v) == gt_id
                   Geometric top-1 accuracy via cosine NN (no decoder head).
                   Does x̂_t geometrically commit to the right token?

  cos_margin       cos_1st - cos_2nd
                   Cosine version of margin: separation between 1st and 2nd
                   nearest centroid in cosine space.

  l2_residual_frac ||x̂_t - E^T p_t|| / ||x̂_t||  (normalized anchor residual)
                   Relevant to L_anc: what fraction of x̂_t lies outside the
                   token-anchored component?

Usage:
  conda run -n elf python experiments/probe_elf/probe_geo.py \\
      --checkpoint embedded-language-flows/ELF-B-owt \\
      --config /path/to/config.yaml \\
      --centroid_path results/data/token_centroids.npz \\
      --n_samples 64 --seq_len 128 \\
      --out_dir results/elf/probe_geo_v1
"""

import sys, os, argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# ── ELF source path (mirrors probe_anchor_v3 setup) ──────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_ELF_SRC    = _SCRIPT_DIR / "src"
if not _ELF_SRC.exists():
    _ELF_SRC = _SCRIPT_DIR.parents[1] / "models" / "ELF" / "src"
ELF_SRC       = str(_ELF_SRC)
DEFAULT_CONFIG = str(_ELF_SRC / "configs" / "training_configs" / "train_owt_ELF-B.yml")
if ELF_SRC not in sys.path:
    sys.path.insert(0, ELF_SRC)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def softmax_np(logits: np.ndarray, tau: float = 1.0) -> np.ndarray:
    l = logits / tau
    l -= l.max(axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)

def token_entropy(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-9, 1.0)
    return -(pc * np.log(pc)).sum(axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Geometry metrics
# ─────────────────────────────────────────────────────────────────────────────

def geo_metrics(
    x_hat:     np.ndarray,   # [L, d]  denoised representation
    x_clean:   np.ndarray,   # [L, d]  clean T5 encoding
    E_all:     np.ndarray,   # [V, d]  centroids in latent space (un-normalized)
    E_norm:    np.ndarray,   # [V, d]  L2-normalized centroids
    seen_mask: np.ndarray,   # [V] bool  which tokens have valid centroids
    p:         np.ndarray,   # [L, V]  decoder distribution
    gt_ids:    np.ndarray,   # [L]     ground truth token ids
) -> dict:
    L = len(x_hat)

    # ── cos(x̂_t, x_clean): backbone recovery in contextual space ─────────────
    xh_n = x_hat  / (np.linalg.norm(x_hat,  axis=-1, keepdims=True) + 1e-9)
    xc_n = x_clean / (np.linalg.norm(x_clean, axis=-1, keepdims=True) + 1e-9)
    cos_to_clean = (xh_n * xc_n).sum(-1)   # [L]

    # ── cosine NN to centroid (seen tokens only) ───────────────────────────────
    # [L, V]: first compute full cosine matrix, then mask unseen to -inf
    cos_to_E = xh_n @ E_norm.T              # [L, V]
    cos_to_E[:, ~seen_mask] = -np.inf       # mask out tokens with no centroid

    top2_idx = np.argpartition(cos_to_E, -2, axis=-1)[:, -2:]   # [L, 2]
    # guarantee top2_idx[:, 0] = 1st best, [:, 1] = 2nd best
    v0 = cos_to_E[np.arange(L), top2_idx[:, 0]]
    v1 = cos_to_E[np.arange(L), top2_idx[:, 1]]
    swap = v0 < v1
    top2_idx[swap] = top2_idx[swap][:, ::-1]
    cos1 = cos_to_E[np.arange(L), top2_idx[:, 0]]
    cos2 = cos_to_E[np.arange(L), top2_idx[:, 1]]

    cos_nn_id      = top2_idx[:, 0]                    # [L]
    cos_nn_correct = (cos_nn_id == gt_ids).astype(float)  # [L]
    cos_margin     = cos1 - cos2                       # [L]

    # ── normalized anchor residual ||x̂ - E^T p|| / ||x̂|| ─────────────────────
    anchor   = p @ E_all                               # [L, d]
    residual = x_hat - anchor                          # [L, d]
    res_norm  = np.linalg.norm(residual, axis=-1)      # [L]
    xhat_norm = np.linalg.norm(x_hat,   axis=-1)       # [L]
    l2_residual_frac = res_norm / (xhat_norm + 1e-9)   # [L]

    # ── L2 d_nn (reference, matches probe_v3) ─────────────────────────────────
    E_seen = E_all[seen_mask]
    E_sq   = (E_seen ** 2).sum(-1)
    x_sq   = (x_hat ** 2).sum(-1)
    dists_sq = np.maximum(
        x_sq[:, None] + E_sq[None, :] - 2.0 * (x_hat @ E_seen.T), 0.0)
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
# 3. Load ELF + data
# ─────────────────────────────────────────────────────────────────────────────

PROBE_DIR = Path(__file__).parent
sys.path.insert(0, str(PROBE_DIR))


def load_elf_and_data(args):
    from probe_anchor_v3 import load_elf, encode_with_t5, load_owt_texts
    import optax, copy

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

    print("[data] loading OWT texts…")
    texts   = load_owt_texts(args.n_samples)
    print("[encode] T5 encoder…")
    samples = encode_with_t5(texts, tokenizer, encoder_model, encoder_params,
                             args.seq_len, latent_mean, latent_std)

    # Load centroids (same logic as probe_anchor_v3)
    data    = np.load(args.centroid_path)
    E_raw   = data["centroids"].astype(np.float32)
    latent_std_file = float(data["latent_std"]) if "latent_std" in data else None
    already_normalized = (
        latent_std_file is not None
        and np.isclose(latent_std_file, float(latent_std)))
    vocab_size = tokenizer.vocab_size
    E_all = E_raw[:vocab_size] if already_normalized else E_raw[:vocab_size] / latent_std

    seen_mask = None
    if "counts" in data:
        counts    = data["counts"][:vocab_size]
        seen_mask = counts > 0
        n_seen = int(seen_mask.sum())
        print(f"[embed] centroid: {E_all.shape}  seen={n_seen}/{vocab_size}")
    else:
        seen_mask = np.ones(vocab_size, dtype=bool)
        print(f"[embed] centroid: {E_all.shape}  (no counts, treating all as seen)")

    # L2-normalize for cosine computations
    E_norm = E_all / (np.linalg.norm(E_all, axis=-1, keepdims=True) + 1e-9)

    return forward_fn, tokenizer, samples, E_all, E_norm, seen_mask


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main probe loop
# ─────────────────────────────────────────────────────────────────────────────

ALL_METRICS = [
    "cos_to_clean", "cos_to_clean_p10", "cos_to_clean_p50", "cos_to_clean_p90",
    "cos_nn_max", "cos_nn_correct", "cos_margin",
    "l2_residual_frac", "d_nn_l2",
    "top1_gt_decoder",
    "entropy_decoder",
]


def probe_sample(
    forward_fn, sample, E_all, E_norm, seen_mask,
    t_grid, n_noise, seed, tau
) -> dict:
    gt_ids, clean_emb, attn_mask = sample   # tuples from encode_with_t5
    L, d = clean_emb.shape
    rng = np.random.default_rng(seed)

    out = {k: [] for k in ALL_METRICS}
    out["t"] = t_grid.tolist()

    mask_batch = attn_mask[None]   # [1, L]

    for ti, t in enumerate(t_grid):
        seed_metrics = {k: [] for k in ALL_METRICS}

        for _ in range(n_noise):
            eps  = rng.standard_normal((1, L, d)).astype(np.float32)
            z_t  = t * clean_emb[None] + (1.0 - t) * eps
            x_hat, logits = forward_fn(z_t, t, mask_batch)  # [L,d], [L,V]

            p = softmax_np(logits, tau)

            g = geo_metrics(x_hat, clean_emb, E_all, E_norm, seen_mask, p, gt_ids)
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
# 5. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(agg, out_dir):
    t = np.array(agg["t"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("ELF Geometric Commitment Probe (cosine-based)", fontsize=12)

    panels = [
        ("cos_to_clean_mean",     "cos(x̂_t, x_clean) — backbone recovery",   "cosine sim"),
        ("cos_nn_correct_mean",   "Cosine NN accuracy (geometric top-1)",       "fraction"),
        ("cos_nn_max_mean",       "max_v cos(x̂_t, E_v) — nearest centroid",   "cosine sim"),
        ("cos_margin_mean",       "Cosine margin (cos_1st − cos_2nd)",          "margin"),
        ("l2_residual_frac_mean", "||r_t|| / ||x̂_t|| — anchor residual frac",  "fraction"),
        ("d_nn_l2_mean",          "L2 d_nn (reference, matches probe_v3)",      "L2 dist"),
    ]

    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        y = np.array(agg.get(key, [float("nan")] * len(t)))
        ax.plot(t, y, lw=2, color="steelblue")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("t  (0=noisy, 1=clean)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    out_path = out_dir / "probe_geo.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="embedded-language-flows/ELF-B-owt")
    p.add_argument("--config",        default=DEFAULT_CONFIG)
    p.add_argument("--centroid_path", default="results/data/token_centroids.npz")
    p.add_argument("--n_samples",     type=int, default=64)
    p.add_argument("--seq_len",       type=int, default=128)
    p.add_argument("--n_t_steps",     type=int, default=21)
    p.add_argument("--n_noise",       type=int, default=4)
    p.add_argument("--tau",           type=float, default=1.0)
    p.add_argument("--out_dir",       default="results/elf/probe_geo_v1")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_grid  = np.linspace(0.0, 1.0, args.n_t_steps)

    print("[probe_geo] loading ELF model + centroids…")
    forward_fn, tokenizer, samples, E_all, E_norm, seen_mask = \
        load_elf_and_data(args)

    seq_results = []
    for i, sample in enumerate(samples):
        print(f"\n── sample {i+1}/{len(samples)}")
        res = probe_sample(
            forward_fn, sample, E_all, E_norm, seen_mask,
            t_grid, args.n_noise, seed=i * 1000, tau=args.tau,
        )
        seq_results.append(res)

        for ti, tv in enumerate(t_grid):
            print(f"  t={tv:.2f}  "
                  f"cos_clean={res['cos_to_clean'][ti]:.3f}  "
                  f"cos_nn_cor={res['cos_nn_correct'][ti]:.3f}  "
                  f"cos_margin={res['cos_margin'][ti]:.3f}  "
                  f"top1_dec={res['top1_gt_decoder'][ti]:.3f}  "
                  f"res_frac={res['l2_residual_frac'][ti]:.3f}")

    agg = aggregate(seq_results)
    agg["args"] = vars(args)

    out_json = out_dir / "probe_geo.json"
    with open(out_json, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[saved] {out_json}")

    plot_results(agg, out_dir)

    # summary table
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
