"""
Anchor Emergence Probing for ELF — Verification Edition.

NOTE: p_t is derived from the *decode head* (decoder_step_active=True),
applied at every t. During training this head is only used for the 20%
decoder branch; here we force it at intermediate t as a probe. This is
stated explicitly so findings are not over-interpreted as the model's
true denoising trajectory.

Verification checks (per user request):
  Q1. Decode head confirmed (see above).
  Q2. Temperature sweep: recomputes metrics for τ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
      from the same cached logits; phase pattern should be τ-invariant.
  Q3. Two top-k variants:
        topk_gt    — ground-truth token in top-k  (correctness)
        topk_final — final-output token in top-k  (internal consistency)
  Q4. Two revision measures:
        rev_top1   — fraction of positions where argmax changes
        rev_jsd    — Jensen-Shannon divergence between consecutive p_t

Usage:
    cd ~/ELF
    CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \\
    python probe_anchor.py \\
        --config src/configs/training_configs/train_owt_ELF-B.yml \\
        --checkpoint embedded-language-flows/ELF-B-owt \\
        --n_samples 64 --seq_len 256 --out_dir ~/probe_results_v2

    # pipeline test (no checkpoint):
    python probe_anchor.py --stub --out_dir ~/probe_stub_v2
"""

import sys, os, argparse, copy, json
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── ELF src on path ──────────────────────────────────────────────────────────
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
# 1. Primitive metric functions (operate on numpy arrays)
# ─────────────────────────────────────────────────────────────────────────────

def compute_p(logits: np.ndarray, tau: float) -> np.ndarray:
    """softmax(logits / tau).  [L, V] → [L, V]"""
    l = logits / tau
    l -= l.max(axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)


def anchor_distance(x_hat: np.ndarray, p: np.ndarray, E: np.ndarray) -> float:
    """mean_L ||x̂_i − (p_i @ E)||.  E: [V, d]"""
    a = p @ E
    return float(np.linalg.norm(x_hat - a, axis=-1).mean())


def token_entropy(p: np.ndarray) -> float:
    """H(p) averaged over positions.  p: [L, V]"""
    pc = np.clip(p, 1e-9, 1.0)
    return float(-(pc * np.log(pc)).sum(axis=-1).mean())


def topk_recovery(p: np.ndarray, ref_ids: np.ndarray, k: int) -> float:
    """Fraction of positions where ref_ids[i] ∈ top-k of p[i]."""
    topk = np.argsort(p, axis=-1)[:, -k:]
    return float((topk == ref_ids[:, None]).any(axis=-1).mean())


def top1_flip_rate(p_prev: Optional[np.ndarray], p_curr: np.ndarray) -> Optional[float]:
    if p_prev is None:
        return None
    return float((np.argmax(p_prev, axis=-1) != np.argmax(p_curr, axis=-1)).mean())


def jsd(p_prev: Optional[np.ndarray], p_curr: np.ndarray) -> Optional[float]:
    """Jensen-Shannon divergence between consecutive distributions, averaged over positions."""
    if p_prev is None:
        return None
    pp = np.clip(p_prev, 1e-9, 1.0)
    pc = np.clip(p_curr, 1e-9, 1.0)
    m  = 0.5 * (pp + pc)
    kl_p = (pp * (np.log(pp) - np.log(m))).sum(axis=-1)
    kl_c = (pc * (np.log(pc) - np.log(m))).sum(axis=-1)
    return float((0.5 * (kl_p + kl_c)).mean())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logit collection (model-forward only, independent of tau)
# ─────────────────────────────────────────────────────────────────────────────

def collect_logits(
    forward_fn,
    clean_emb: np.ndarray,   # [L, d]
    attn_mask: np.ndarray,   # [L]
    t_grid: np.ndarray,      # [T]
    n_noise: int,
    seed: int,
) -> tuple:
    """
    Run forward passes for all (t, noise_seed) combinations.

    Returns:
        logits_arr  : np.float32  [T, N, L, V]   — raw decoder logits
        xhat_arr    : np.float32  [T, N, L, d]   — x_hat in embedding space
    """
    rng = np.random.default_rng(seed)
    L, d = clean_emb.shape
    mask_batch = attn_mask[None]   # [1, L]
    T, N = len(t_grid), n_noise

    logits_arr = np.zeros((T, N, L, 0), dtype=np.float32)  # shape fixed below
    xhat_arr   = np.zeros((T, N, L, d), dtype=np.float32)
    first = True

    for ti, t in enumerate(t_grid):
        for si in range(N):
            eps = rng.standard_normal((1, L, d)).astype(np.float32)
            z_t = t * clean_emb[None] + (1.0 - t) * eps
            x_hat, logits = forward_fn(z_t, float(t), mask_batch)

            if first:
                V = logits.shape[-1]
                logits_arr = np.zeros((T, N, L, V), dtype=np.float32)
                first = False

            logits_arr[ti, si] = logits
            xhat_arr[ti, si]   = x_hat

    return logits_arr, xhat_arr


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metric computation from cached logits (tau-sweep friendly)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_for_tau(
    logits_arr: np.ndarray,   # [T, N, L, V]
    xhat_arr:   np.ndarray,   # [T, N, L, d]
    E:          np.ndarray,   # [V, d]
    t_grid:     np.ndarray,   # [T]
    gt_ids:     np.ndarray,   # [L]   ground-truth token IDs
    final_ids:  np.ndarray,   # [L]   model's argmax at t=1, τ=1
    tau:        float,
    topk:       int,
) -> dict:
    """
    Compute all metrics for one tau from pre-collected logits.
    Returns per-t stats (mean over noise seeds).
    """
    T, N = logits_arr.shape[:2]
    keys = ["anchor_dist", "entropy",
            "topk_gt", "topk_final",
            "rev_top1", "rev_jsd"]
    out = {k: [] for k in ["t"] + keys}
    out["t"] = t_grid.tolist()

    prev_p = [None] * N   # per noise-seed previous p

    for ti in range(T):
        vals = {k: [] for k in keys}

        for si in range(N):
            logits = logits_arr[ti, si]   # [L, V]
            x_hat  = xhat_arr[ti, si]     # [L, d]
            p = compute_p(logits, tau)    # [L, V]

            vals["anchor_dist"].append(anchor_distance(x_hat, p, E))
            vals["entropy"].append(token_entropy(p))
            vals["topk_gt"].append(topk_recovery(p, gt_ids, k=topk))
            vals["topk_final"].append(topk_recovery(p, final_ids, k=topk))

            r1  = top1_flip_rate(prev_p[si], p)
            jsd_ = jsd(prev_p[si], p)
            if r1  is not None: vals["rev_top1"].append(r1)
            if jsd_ is not None: vals["rev_jsd"].append(jsd_)

            prev_p[si] = p

        for k in keys:
            out[k].append(float(np.mean(vals[k])) if vals[k] else float("nan"))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. ELF model loading
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
# 5. Stub model
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
# 6. Data loading + encoding
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
# 7. Aggregation across sequences
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(per_seq_per_tau: dict) -> dict:
    """
    per_seq_per_tau: {tau: [seq_result, ...]}
    Returns: {tau: {metric_mean: [...], metric_std: [...]}}
    """
    out = {}
    metrics = ["anchor_dist", "entropy", "topk_gt", "topk_final", "rev_top1", "rev_jsd"]
    for tau, seq_list in per_seq_per_tau.items():
        agg = {"t": seq_list[0]["t"]}
        for m in metrics:
            mat = np.array([s[m] for s in seq_list])   # [N_seq, T]
            agg[f"{m}_mean"] = mat.mean(axis=0).tolist()
            agg[f"{m}_std"]  = mat.std(axis=0).tolist()
        out[str(tau)] = agg
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plotting
# ─────────────────────────────────────────────────────────────────────────────

TAU_COLORS = {0.1: "#e74c3c", 0.5: "#e67e22", 1.0: "#2ecc71",
              2.0: "#3498db", 5.0: "#9b59b6"}

def plot_results(results: dict, out_dir: str, label: str, tau_ref: float = 1.0):
    """
    results: output of aggregate()
    Produces two figures:
      anchor_probe_tau_sweep.png  — entropy + top1-flip across all tau
      anchor_probe_main.png       — full 6-panel at tau_ref
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ref = results[str(tau_ref)]
    t   = np.array(ref["t"])

    # ── Figure 1: tau sweep (entropy + JSD) ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for tau_str, res in results.items():
        tau = float(tau_str)
        col = TAU_COLORS.get(tau, "gray")
        tv  = np.array(res["t"])
        for ax, key, ylabel in [
            (axes[0], "entropy",  "H(p_t) [nats]"),
            (axes[1], "rev_top1", "top-1 flip rate"),
        ]:
            mean = np.array(res[f"{key}_mean"])
            ax.plot(tv, mean, color=col, lw=2, label=f"τ={tau}")

    for ax, title in zip(axes, ["Token Entropy (τ sweep)", "Revision top-1 (τ sweep)"]):
        ax.set_xlabel("t  (noise → clean)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Temperature Sweep — {label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "anchor_probe_tau_sweep.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: main 6-panel at tau_ref ────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.3)

    def _panel(ax, key, ylabel, title, color, ylim=None):
        mean = np.array(ref[f"{key}_mean"])
        std  = np.array(ref[f"{key}_std"])
        ax.plot(t, mean, color=color, lw=2, label=f"τ={tau_ref}")
        ax.fill_between(t, mean-std, mean+std, alpha=0.2, color=color)
        ax.set_xlabel("t"); ax.set_ylabel(ylabel); ax.set_title(title)
        if ylim: ax.set_ylim(*ylim)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    _panel(fig.add_subplot(gs[0, 0]), "anchor_dist",
           "‖x̂ − p@E‖", "Anchor Distance (E=input embed, placeholder)", "#95a5a6")
    _panel(fig.add_subplot(gs[0, 1]), "entropy",
           "H(p_t) [nats]", "Token Entropy ↓", "#3498db")
    _panel(fig.add_subplot(gs[0, 2]), "topk_gt",
           "P(gt ∈ top-k)", "Top-k vs Ground Truth ↑", "#2ecc71", ylim=(0, 1))

    _panel(fig.add_subplot(gs[1, 0]), "topk_final",
           "P(final ∈ top-k)", "Top-k vs Final Output ↑\n(internal consistency)", "#27ae60", ylim=(0, 1))

    # rev_top1 and rev_jsd on same axis
    ax_rev = fig.add_subplot(gs[1, 1])
    r1  = np.array(ref["rev_top1_mean"])
    j   = np.array(ref["rev_jsd_mean"])
    mask1 = ~np.isnan(r1);  maskj = ~np.isnan(j)
    ax_rev.plot(t[mask1], r1[mask1], color="#e74c3c", lw=2, label="top-1 flip")
    ax_rev.plot(t[maskj], j[maskj],  color="#c0392b", lw=2, ls="--", label="JSD")
    ax_rev.set_xlabel("t"); ax_rev.set_ylabel("revision measure")
    ax_rev.set_title("Revision Delta (top-1 flip & JSD)")
    ax_rev.set_ylim(0); ax_rev.legend(fontsize=8); ax_rev.grid(True, alpha=0.3)

    # consistency ratio: topk_final / topk_gt
    ax_c = fig.add_subplot(gs[1, 2])
    gt_m = np.array(ref["topk_gt_mean"])
    fn_m = np.array(ref["topk_final_mean"])
    safe = np.where(gt_m > 0.01, gt_m, np.nan)
    ratio = fn_m / safe
    ax_c.plot(t, ratio, color="#8e44ad", lw=2)
    ax_c.axhline(1.0, color="gray", ls="--", lw=1)
    ax_c.set_xlabel("t"); ax_c.set_ylabel("topk_final / topk_gt")
    ax_c.set_title("Consistency Ratio\n(>1: model consistent beyond correct)")
    ax_c.grid(True, alpha=0.3)

    fig.suptitle(f"Anchor Emergence Probing (τ={tau_ref}) — {label}", fontsize=13)
    fig.savefig(str(Path(out_dir) / "anchor_probe_main.png"), bbox_inches="tight", dpi=150)
    print(f"[plot] saved → {out_dir}/anchor_probe_{{main,tau_sweep}}.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default=None)
    p.add_argument("--checkpoint", type=str, default="embedded-language-flows/ELF-B-owt")
    p.add_argument("--stub",       action="store_true")
    p.add_argument("--n_samples",  type=int, default=64)
    p.add_argument("--seq_len",    type=int, default=256)
    p.add_argument("--n_t_steps",  type=int, default=21)
    p.add_argument("--tau_list",   type=str, default="0.1,0.5,1.0,2.0,5.0",
                   help="Comma-separated temperatures to sweep.")
    p.add_argument("--topk",       type=int, default=5)
    p.add_argument("--n_noise",    type=int, default=4)
    p.add_argument("--out_dir",    type=str, default="probe_results_v2")
    return p.parse_args()


def main():
    args  = parse_args()
    t_grid = np.linspace(0.0, 1.0, args.n_t_steps)
    tau_list = [float(x) for x in args.tau_list.split(",")]

    # ── Model / data setup ───────────────────────────────────────────────────
    if args.stub:
        print("[mode] RandomStub")
        stub = _RandomStubModel(d=512, V=32128)
        E    = stub.E

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
        if args.config is None:
            raise ValueError("--config required")
        print(f"[mode] ELF: {args.checkpoint}")
        (model, ema_params, encoder_params, encoder_model,
         tokenizer, config, enc_config) = load_elf(
             args.config, args.checkpoint, override_max_length=args.seq_len)

        latent_mean = getattr(config, "latent_mean", 0.0)
        latent_std  = getattr(config, "latent_std",  1.0)

        # Anchor matrix (T5 input embeddings — placeholder until contextual centroids computed)
        E_raw      = np.array(encoder_params["shared"]["embedding"])
        vocab_size = tokenizer.vocab_size
        E = E_raw[:vocab_size] / latent_std
        print(f"[embed] E: {E.shape}  latent_std={latent_std}  "
              f"(WARNING: using T5 input embeddings, not contextual centroids)")

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
    # For each sequence: collect logits once, then sweep tau.
    per_seq_per_tau = {tau: [] for tau in tau_list}

    for i, (gt_ids, emb, mask) in enumerate(samples):
        print(f"\n── Seq {i+1}/{len(samples)} — collecting logits…")
        logits_arr, xhat_arr = collect_logits(
            forward_fn, emb, mask, t_grid, args.n_noise, seed=i)

        # Final output: argmax at t=1, τ=1.0, noise_seed=0
        final_logits = logits_arr[-1, 0]          # [L, V], t=1.0, seed=0
        final_ids    = np.argmax(final_logits, axis=-1).astype(np.int32)

        for tau in tau_list:
            res = compute_metrics_for_tau(
                logits_arr, xhat_arr, E, t_grid,
                gt_ids, final_ids, tau=tau, topk=args.topk,
            )
            # Print summary at tau=1.0
            if tau == 1.0:
                for ti, t_val in enumerate(t_grid):
                    print(f"  τ=1.0  t={t_val:.2f}  "
                          f"H={res['entropy'][ti]:.3f}  "
                          f"top{args.topk}_gt={res['topk_gt'][ti]:.3f}  "
                          f"top{args.topk}_final={res['topk_final'][ti]:.3f}  "
                          f"jsd={res['rev_jsd'][ti]:.4f}")
            per_seq_per_tau[tau].append(res)

    # ── Aggregate + save + plot ───────────────────────────────────────────────
    final = aggregate(per_seq_per_tau)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(args.out_dir) / "anchor_probe_v2.json"
    with open(json_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[save] → {json_path}")

    plot_results(final, args.out_dir, label=label, tau_ref=1.0)
    print("[done]")


if __name__ == "__main__":
    main()
