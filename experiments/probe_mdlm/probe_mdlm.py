"""
probe_mdlm.py — MDLM Discrete Masked Diffusion Anchor Emergence Probe

Probes token crystallization dynamics in MDLM (Shi et al. 2024,
"Simple and Effective Masked Diffusion Language Models").

Key differences from continuous diffusion (ELF / LangFlow):
  - No continuous latent: positions are either [MASK] or clean (binary state)
  - No geometric metrics: d_nn / margin / d_soft are meaningless without continuous latent
  - "Soft commitment": argmax of p_θ(x_0_i | x_t) for masked positions
  - "Hard commitment": the actual unmasking event during sampling (irreversible!)
  - Metrics are computed for MASKED positions only (unmasked positions are trivially known)

The key scientific question: does MDLM show a "commitment cliff" analogous to ELF's
t≈0.25 crystallization? And if so, is it triggered by masking rate or by context density?

Noise schedule (loglinear, MDLM default):
  α_t = exp(−λ·(1−t))   where λ = −log(α_min), t=0→noisy, t=1→clean
  σ_t = −log(α_t) = λ·(1−t)   (conditioning input to model)

Convention: t=0 → fully masked (noisy), t=1 → fully unmasked (clean)

Setup:
  git clone https://github.com/kuleshov-group/mdlm ~/mdlm
  cd ~/mdlm && pip install -e .

Usage:
  python probe_mdlm.py \\
      --checkpoint kuleshov-group/mdlm-owt \\
      --n_samples 64 --seq_len 128 --out_dir ~/probe_mdlm_v1

  # smoke test (no model needed):
  python probe_mdlm.py --stub --out_dir ~/probe_mdlm_stub
"""

import sys, os, argparse, json, math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MDLM repo path ────────────────────────────────────────────────────────────
MDLM_REPO = os.path.expanduser("~/mdlm")
if os.path.isdir(MDLM_REPO) and MDLM_REPO not in sys.path:
    sys.path.insert(0, MDLM_REPO)

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def loglinear_alpha(t: float, alpha_min: float) -> float:
    """α_t = exp(−λ·(1−t)), λ = −log(α_min). t=0→α_min (noisy), t=1→1 (clean)."""
    lam = -math.log(alpha_min)
    return math.exp(-lam * (1.0 - t))


def loglinear_sigma(t: float, alpha_min: float) -> float:
    """σ_t = −log(α_t) = λ·(1−t). Used as model conditioning."""
    lam = -math.log(alpha_min)
    return lam * (1.0 - t)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Primitive metric functions (numpy)
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


def jsd_scalar(p_prev: Optional[np.ndarray], p_curr: np.ndarray) -> Optional[float]:
    if p_prev is None:
        return None
    pp = np.clip(p_prev, 1e-9, 1.0)
    pc = np.clip(p_curr, 1e-9, 1.0)
    m  = 0.5 * (pp + pc)
    return float((0.5 * ((pp * (np.log(pp) - np.log(m))).sum(-1)
                       + (pc * (np.log(pc) - np.log(m))).sum(-1))).mean())


def bimodality_coefficient(H: np.ndarray) -> float:
    """BC = (skew²+1)/(excess_kurt + 3(n-1)²/((n-2)(n-3))). >5/9 → bimodal."""
    n = len(H)
    if n < 4:
        return float("nan")
    mu, sigma = H.mean(), H.std()
    if sigma < 1e-9:
        return float("nan")
    z = (H - mu) / sigma
    skew  = float((z**3).mean())
    ekurt = float((z**4).mean() - 3.0)
    denom = ekurt + 3.0 * (n-1)**2 / ((n-2) * (n-3))
    return float((skew**2 + 1.0) / denom) if abs(denom) > 1e-9 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Masking utilities
# ─────────────────────────────────────────────────────────────────────────────

def apply_masking(
    ids: np.ndarray,       # [L] clean token IDs
    alpha: float,          # signal fraction (1-α = masking prob)
    mask_id: int,
    rng: np.random.Generator,
    attention_mask: Optional[np.ndarray] = None,  # [L] 1=real token, 0=padding
) -> tuple:
    """
    Independently mask each real token with prob (1-alpha).
    Returns (x_t, is_masked) both [L].
    Padding positions are never masked.
    """
    prob = 1.0 - alpha
    draw = rng.random(len(ids))
    is_masked = draw < prob
    if attention_mask is not None:
        is_masked &= (attention_mask > 0)
    x_t = ids.copy()
    x_t[is_masked] = mask_id
    return x_t, is_masked


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mdlm(checkpoint: str):
    """
    Load MDLM model + tokenizer.

    Supports two backends:
      A) MDLM repo (~/mdlm): loads via OmegaConf config + custom model class
      B) HuggingFace hub: tries AutoModel if checkpoint looks like a HF repo ID

    Returns: (model, tokenizer, mask_token_id, config_dict)
    """
    from transformers import AutoTokenizer, GPT2Tokenizer

    # ── Try MDLM repo approach ────────────────────────────────────────────────
    if os.path.isdir(MDLM_REPO):
        try:
            return _load_mdlm_from_repo(checkpoint)
        except Exception as e:
            print(f"[mdlm] repo load failed ({e}), trying HuggingFace...")

    # ── HuggingFace fallback ───────────────────────────────────────────────────
    return _load_mdlm_from_hf(checkpoint)


def _load_mdlm_from_repo(checkpoint: str):
    """Load via kuleshov-group/mdlm repo's OmegaConf + checkpoint."""
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    # Locate config
    local_path = Path(checkpoint)
    if local_path.is_dir():
        cfg_path  = local_path / "config.yaml"
        ckpt_path = next(local_path.glob("*.ckpt"), None) or next(local_path.glob("*.pt"), None)
    else:
        # HF hub path — download checkpoint
        from huggingface_hub import snapshot_download
        dl_path   = snapshot_download(checkpoint)
        dl_path   = Path(dl_path)
        cfg_path  = dl_path / "config.yaml"
        ckpt_path = next(dl_path.glob("*.ckpt"), None) or next(dl_path.glob("*.pt"), None)

    cfg = OmegaConf.load(cfg_path)

    # Import model class from MDLM repo
    # Try multiple common module paths used by different versions of the repo
    model_cls = None
    for mod_path, cls_name in [
        ("diffusion",  "Diffusion"),   # kuleshov-group/mdlm repo: main class is Diffusion
        ("algo",       "MDLM"),        # s-sahoo/duo repo also has an MDLM class
        ("diffusion",  "MDLM"),        # fallback
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            model_cls = getattr(mod, cls_name)
            break
        except (ImportError, AttributeError):
            continue

    if model_cls is None:
        raise ImportError("Cannot find MDLM model class in repo. "
                          "Check MDLM_REPO path and repo structure.")

    model = model_cls(cfg)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state)
        # Strip "model." prefix if present
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"[mdlm] loaded checkpoint: {ckpt_path}")
    else:
        print("[mdlm] WARNING: no checkpoint file found, using random weights")

    tokenizer_name = getattr(cfg, "tokenizer", getattr(cfg.data, "tokenizer_name", "gpt2"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    mask_id = tokenizer.mask_token_id

    model = model.to(device).eval()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return model, tokenizer, mask_id, cfg_dict


def _load_mdlm_from_hf(checkpoint: str):
    """HuggingFace fallback: load with trust_remote_code for custom MDLM architecture.

    kuleshov-group/mdlm-owt has a custom tokenizer class that is a fast tokenizer
    but lacks the required tokenizer.json backend file, causing instantiation to fail
    even with use_fast=False.  We bypass this by loading GPT-2 tokenizer directly —
    MDLM-OWT is trained on OpenWebText with the GPT-2 BPE tokenizer anyway.
    """
    from transformers import GPT2Tokenizer, AutoModel
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    # Load the MDLM model with custom code (exposes sigma-conditioned forward)
    try:
        model = AutoModel.from_pretrained(
            checkpoint, trust_remote_code=True).to(device).eval()
    except Exception as e:
        print(f"[mdlm] AutoModel failed ({e}), trying AutoModelForMaskedLM...")
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(
            checkpoint, trust_remote_code=True).to(device).eval()
    mask_id = tokenizer.mask_token_id
    print(f"[mdlm] loaded via HuggingFace (trust_remote_code): {checkpoint}")
    return model, tokenizer, mask_id, {}


def mdlm_forward(model, x_t_ids: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Unified forward wrapper.
    x_t_ids : [B, L] masked token IDs
    sigma    : [B]   noise level σ = −log(α)
    Returns  : [B, L, V] logits

    MDLM repo typically exposes one of these signatures:
      model(x, sigma)                   ← standard MDLM
      model(input_ids, sigma)
      model(input_ids=x, sigma=sigma)
      model(input_ids=x)                ← HF AutoModelForMaskedLM (ignores sigma)
    """
    with torch.no_grad():
        # Try MDLM-style forward
        for call in [
            lambda: model(x_t_ids, sigma),
            lambda: model(input_ids=x_t_ids, sigma=sigma),
            lambda: model(x_t_ids, t=sigma),
            lambda: model(input_ids=x_t_ids).logits,   # HF fallback
        ]:
            try:
                out = call()
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if hasattr(out, "logits"):
                    out = out.logits
                return out.float()   # [B, L, V]
            except TypeError:
                continue
    raise RuntimeError("Could not call MDLM model. Check mdlm_forward() call signatures.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stub model for testing
# ─────────────────────────────────────────────────────────────────────────────

class _StubMDLM(torch.nn.Module):
    """Random model for unit-testing the probe pipeline."""
    def __init__(self, vocab_size: int = 50258, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        # Bias toward uniform distribution; context from x_t subtly shifts it
        self.embed = torch.nn.Embedding(vocab_size, 64)
        self.head  = torch.nn.Linear(64, vocab_size, bias=False)

    def forward(self, x, sigma=None):
        h = self.embed(x).mean(dim=1, keepdim=True).expand_as(self.embed(x))
        return self.head(h)


def make_stub(vocab_size: int = 50258):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    mask_id = tokenizer.mask_token_id
    model   = _StubMDLM(vocab_size=len(tokenizer)).to(device).eval()
    return model, tokenizer, mask_id, {"stub": True}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Logit collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_logits_mdlm(
    model,
    ids_clean:    np.ndarray,       # [L] clean token IDs
    t_grid:       np.ndarray,       # [T] probe timesteps (0=noisy, 1=clean)
    n_noise:      int,              # independent masking seeds per (seq, t)
    seed:         int,
    mask_id:      int,
    alpha_min:    float,
    attn_mask:    Optional[np.ndarray] = None,  # [L] 1=real, 0=padding
) -> tuple:
    """
    For each t in t_grid, apply n_noise independent maskings and run model.

    Returns:
      logits_arr  [T, N, L, V]  — vocab logits (0 for unmasked positions too)
      masked_arr  [T, N, L]     — bool, True = masked at this (t, seed)
      alpha_arr   [T]           — α_t values
    """
    rng = np.random.default_rng(seed)
    L   = len(ids_clean)
    T, N = len(t_grid), n_noise

    logits_arr = None
    masked_arr = np.zeros((T, N, L), dtype=bool)
    alpha_arr  = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for ti, t in enumerate(t_grid):
            alpha = loglinear_alpha(t, alpha_min)
            sigma = loglinear_sigma(t, alpha_min)
            alpha_arr[ti] = alpha

            # Build batch of N masked sequences
            x_batch     = np.zeros((N, L), dtype=np.int64)
            is_masked_b = np.zeros((N, L), dtype=bool)
            for si in range(N):
                x_t, is_m = apply_masking(ids_clean, alpha, mask_id, rng, attn_mask)
                x_batch[si]     = x_t
                is_masked_b[si] = is_m

            masked_arr[ti] = is_masked_b

            x_torch = torch.from_numpy(x_batch).long().to(device)       # [N, L]
            sigma_t = torch.full((N,), sigma, device=device, dtype=torch.float32)  # [N]

            logits_np = mdlm_forward(model, x_torch, sigma_t).cpu().numpy()  # [N, L, V]

            if logits_arr is None:
                V = logits_np.shape[-1]
                logits_arr = np.zeros((T, N, L, V), dtype=np.float32)
            logits_arr[ti] = logits_np

    return logits_arr, masked_arr, alpha_arr


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metric computation
# ─────────────────────────────────────────────────────────────────────────────

ALL_METRICS = [
    # masking state
    "mask_rate",
    # per-masked-position metrics (averaged over masked positions)
    "entropy_masked", "entropy_masked_p10", "entropy_masked_p50", "entropy_masked_p90",
    "top1_gt_masked", "top5_gt_masked",
    # soft commitment of masked positions
    "soft_commit_frac",      # fraction of masked pos where H < thresh
    "soft_commit_correct",   # of those, fraction correct
    "soft_commit_wrong",     # of those, fraction wrong
    # inter-step transitions (masked positions only)
    "jsd_masked", "rev_top1_masked",
    "w2c_masked", "c2w_masked",
    # bimodality of entropy distribution across all masked positions
    "bimodality_coeff_masked",
    # all-position metrics
    "entropy_all",
    "top1_gt_all", "top5_gt_all",
    "jsd_all", "rev_top1_all", "w2c_all", "c2w_all",
]

SCALAR_KEYS = ["commit_time_mean", "commit_time_std", "never_committed_frac"]


def compute_metrics_mdlm(
    logits_arr:    np.ndarray,   # [T, N, L, V]
    masked_arr:    np.ndarray,   # [T, N, L] bool
    alpha_arr:     np.ndarray,   # [T]
    gt_ids:        np.ndarray,   # [L]
    t_grid:        np.ndarray,   # [T]
    tau:           float,
    commit_thresh: float,
    topk:          int,
) -> dict:
    T, N, L, V = logits_arr.shape
    out = {k: [] for k in ALL_METRICS}
    out["t"] = t_grid.tolist()

    prev_p_masked   = [None] * N   # previous-step p for each seed (masked positions, ragged)
    prev_p_all      = [None] * N   # previous-step p for all positions
    prev_correct_ag = None         # [L] bool, aggregate over seeds

    # For soft commitment time tracking: per (noise-seed, position) → first t where H<thresh
    commit_times = np.full((N, L), np.nan)

    for ti in range(T):
        alpha = float(alpha_arr[ti])

        # Aggregate across noise seeds
        a_ent, a_ent_p = [], []
        a_ent_all = []
        a_t1_mask, a_t5_mask, a_t1_all, a_t5_all = [], [], [], []
        a_sc_frac, a_sc_cor, a_sc_wrg  = [], [], []
        a_jsd, a_rev, a_w2c, a_c2w     = [], [], [], []
        a_jsd_all, a_rev_all, a_w2c_all, a_c2w_all = [], [], [], []
        a_bim                           = []
        mask_rates                      = []

        curr_correct_seeds = np.zeros((N, L), dtype=bool)

        for si in range(N):
            logits    = logits_arr[ti, si]    # [L, V]
            is_masked = masked_arr[ti, si]    # [L] bool
            p_all     = softmax_np(logits, tau)  # [L, V]

            n_masked = is_masked.sum()
            mask_rates.append(float(n_masked / L))

            # ── Masked-position metrics ──────────────────────────────────────
            if n_masked == 0:
                for lst in [a_ent, a_t1_mask, a_t5_mask,
                            a_sc_frac, a_sc_cor, a_sc_wrg, a_bim]:
                    lst.append(float("nan"))
                a_ent_p.append((float("nan"),) * 3)
            else:
                p_m   = p_all[is_masked]        # [n_masked, V]
                gt_m  = gt_ids[is_masked]        # [n_masked]
                H_m   = token_entropy(p_m)       # [n_masked]

                a_ent.append(float(H_m.mean()))
                a_ent_p.append((float(np.percentile(H_m, 10)),
                                float(np.percentile(H_m, 50)),
                                float(np.percentile(H_m, 90))))

                top1_m = np.argmax(p_m, axis=-1)
                correct_m = top1_m == gt_m
                a_t1_mask.append(float(correct_m.mean()))

                top5_m = np.argsort(p_m, axis=-1)[:, -5:]
                a_t5_mask.append(float((top5_m == gt_m[:, None]).any(-1).mean()))

                # Soft commitment
                sc = H_m < commit_thresh
                a_sc_frac.append(float(sc.mean()))
                if sc.sum() > 0:
                    a_sc_cor.append(float(correct_m[sc].mean()))
                    a_sc_wrg.append(float((~correct_m[sc]).mean()))
                else:
                    a_sc_cor.append(float("nan"))
                    a_sc_wrg.append(float("nan"))

                a_bim.append(bimodality_coefficient(H_m))

                # Commitment time tracking for this seed
                newly_committed = sc & np.isnan(commit_times[si][is_masked])
                # map back to full L indices
                masked_indices = np.where(is_masked)[0]
                for ii, idx in enumerate(masked_indices):
                    if sc[ii] and np.isnan(commit_times[si, idx]):
                        commit_times[si, idx] = t_grid[ti]

            # ── All-position metrics ─────────────────────────────────────────
            H_all   = token_entropy(p_all)
            top1_all = np.argmax(p_all, -1)
            correct_all = (top1_all == gt_ids)
            a_ent_all.append(float(H_all.mean()))
            a_t1_all.append(float(correct_all.mean()))
            top5_all = np.argsort(p_all, axis=-1)[:, -5:]
            a_t5_all.append(float((top5_all == gt_ids[:, None]).any(-1).mean()))

            # inter-step all-position transitions
            if prev_p_all[si] is not None:
                a_jsd_all.append(jsd_scalar(prev_p_all[si], p_all))
                prev_top1_all = np.argmax(prev_p_all[si], -1)
                prev_c_all = (prev_top1_all == gt_ids)
                a_rev_all.append(float((prev_top1_all != top1_all).mean()))
                a_w2c_all.append(float((~prev_c_all &  correct_all).mean()))
                a_c2w_all.append(float(( prev_c_all & ~correct_all).mean()))
            prev_p_all[si] = p_all

            # ── Inter-step JSD + transition (masked positions only) ──────────
            if n_masked > 0 and prev_p_masked[si] is not None:
                prev_pm = prev_p_masked[si]
                # Intersect masked positions with previous step
                min_len = min(len(prev_pm), n_masked)
                if min_len > 0:
                    # JSD over current masked positions' predictions vs previous
                    # Note: masking patterns differ between steps (different random masks)
                    # so we compute JSD on the full distribution at masked positions
                    a_jsd.append(jsd_scalar(prev_pm[:min_len], p_m[:min_len]))
                    prev_top1 = np.argmax(prev_pm[:min_len], -1)
                    curr_top1 = np.argmax(p_m[:min_len], -1)
                    gt_m_prev = gt_m[:min_len]
                    prev_c = prev_top1 == gt_m_prev
                    curr_c = curr_top1 == gt_m_prev
                    a_w2c.append(float((~prev_c &  curr_c).mean()))
                    a_c2w.append(float(( prev_c & ~curr_c).mean()))
                    a_rev.append(float((prev_top1 != curr_top1).mean()))

            prev_p_masked[si] = p_m if n_masked > 0 else None
            curr_correct_seeds[si] = (np.argmax(p_all, -1) == gt_ids)

        def mv(lst):
            return float(np.nanmean(lst)) if len(lst) > 0 else float("nan")

        out["mask_rate"].append(mv(mask_rates))
        out["entropy_masked"].append(mv(a_ent))
        ent_ps = [x for x in a_ent_p if not any(math.isnan(v) for v in x)]
        out["entropy_masked_p10"].append(mv([x[0] for x in ent_ps]) if ent_ps else float("nan"))
        out["entropy_masked_p50"].append(mv([x[1] for x in ent_ps]) if ent_ps else float("nan"))
        out["entropy_masked_p90"].append(mv([x[2] for x in ent_ps]) if ent_ps else float("nan"))
        out["top1_gt_masked"].append(mv(a_t1_mask))
        out["top5_gt_masked"].append(mv(a_t5_mask))
        out["soft_commit_frac"].append(mv(a_sc_frac))
        out["soft_commit_correct"].append(mv(a_sc_cor))
        out["soft_commit_wrong"].append(mv(a_sc_wrg))
        out["jsd_masked"].append(mv(a_jsd))
        out["rev_top1_masked"].append(mv(a_rev))
        out["w2c_masked"].append(mv(a_w2c))
        out["c2w_masked"].append(mv(a_c2w))
        out["bimodality_coeff_masked"].append(mv(a_bim))
        out["entropy_all"].append(mv(a_ent_all))
        out["top1_gt_all"].append(mv(a_t1_all))
        out["top5_gt_all"].append(mv(a_t5_all))
        out["jsd_all"].append(mv(a_jsd_all))
        out["rev_top1_all"].append(mv(a_rev_all))
        out["w2c_all"].append(mv(a_w2c_all))
        out["c2w_all"].append(mv(a_c2w_all))

    # Commitment time stats
    out["commit_time_mean"]     = float(np.nanmean(commit_times))
    out["commit_time_std"]      = float(np.nanstd(commit_times))
    out["never_committed_frac"] = float(np.isnan(commit_times).mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. Aggregation + printing
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(seq_results: list) -> dict:
    out = {"t": seq_results[0]["t"]}
    for m in ALL_METRICS:
        mat = np.array([s[m] for s in seq_results], dtype=np.float64)
        out[f"{m}_mean"] = np.nanmean(mat, axis=0).tolist()
        out[f"{m}_std"]  = np.nanstd(mat,  axis=0).tolist()
    for m in SCALAR_KEYS:
        vals = np.array([s[m] for s in seq_results])
        out[f"{m}_agg"] = float(np.nanmean(vals))
    return out


def print_row(t, alpha, n_masked_frac, H, top1_gt, jsd_val):
    jsd_s = f"{jsd_val:.4f}" if jsd_val is not None and not math.isnan(jsd_val) else "   nan"
    print(f"  t={t:.2f}  α={alpha:.4f}  mask={n_masked_frac:.3f}"
          f"  H_masked={H:.3f}  top1_gt={top1_gt:.3f}  jsd={jsd_s}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(agg: dict, out_dir: Path):
    t = np.array(agg["t"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("MDLM Anchor Emergence Probe")

    panels = [
        ("entropy_masked_mean", "Entropy H (masked positions)", "H(p_θ(x₀|xₜ))"),
        ("top1_gt_masked_mean", "Top-1 GT (masked positions)", "Fraction correct"),
        ("mask_rate_mean",      "Masking rate α(t)",           "Fraction masked"),
        ("soft_commit_frac_mean","Soft commitment fraction",   "Fraction (H < thresh)"),
        ("jsd_masked_mean",     "JSD (consecutive steps)",     "JSD"),
        ("w2c_masked_mean",     "Wrong→Correct rate (masked)", "Fraction"),
    ]

    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        y = np.array(agg.get(key, [float("nan")] * len(t)))
        ax.plot(t, y, lw=2, color="steelblue")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t (0=noisy, 1=clean)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    # Overlay soft_commit_correct/wrong on commit panel
    ax = axes[1, 0]
    cor = np.array(agg.get("soft_commit_correct_mean", [float("nan")] * len(t)))
    wrg = np.array(agg.get("soft_commit_wrong_mean",   [float("nan")] * len(t)))
    ax.plot(t, cor, lw=1.5, ls="--", color="green",  label="correct")
    ax.plot(t, wrg, lw=1.5, ls="--", color="red",    label="wrong")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "mdlm_probe.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Data loading
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


def encode_text(text: str, tokenizer, seq_len: int) -> tuple:
    """Tokenize and return (ids [L], attn_mask [L])."""
    enc  = tokenizer(text, return_tensors="np", truncation=True,
                     max_length=seq_len, padding="max_length")
    ids  = enc["input_ids"][0].astype(np.int64)
    mask = enc["attention_mask"][0].astype(np.float32)
    return ids, mask


# ─────────────────────────────────────────────────────────────────────────────
# 11. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MDLM anchor emergence probe")
    p.add_argument("--checkpoint",    type=str,   default="kuleshov-group/mdlm-owt")
    p.add_argument("--stub",          action="store_true", help="Use random model (no checkpoint needed)")
    p.add_argument("--n_samples",     type=int,   default=64)
    p.add_argument("--seq_len",       type=int,   default=128)
    p.add_argument("--n_t_steps",     type=int,   default=21)
    p.add_argument("--n_noise",       type=int,   default=4,   help="Independent masking seeds per (seq, t)")
    p.add_argument("--tau",           type=float, default=1.0, help="Softmax temperature")
    p.add_argument("--topk",          type=int,   default=5)
    p.add_argument("--commit_thresh", type=float, default=0.1, help="Entropy threshold for soft commitment")
    p.add_argument("--alpha_min",     type=float, default=1e-4, help="Loglinear schedule α_min (masking intensity)")
    p.add_argument("--out_dir",       type=str,   default="./probe_mdlm_results")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # t_grid: t=0 (noisy/all-masked) → t=1 (clean/unmasked)
    t_grid = np.linspace(0.0, 1.0, args.n_t_steps)

    # ── Load model ─────────────────────────────────────────────────────────────
    if args.stub:
        print("[mode] Stub MDLM (random weights)")
        model, tokenizer, mask_id, cfg = make_stub()
    else:
        print(f"[mode] MDLM: {args.checkpoint}")
        model, tokenizer, mask_id, cfg = load_mdlm(args.checkpoint)

    print(f"[mdlm] mask_token_id={mask_id}  device={device}")
    print(f"[schedule] loglinear  alpha_min={args.alpha_min}  "
          f"lambda={-math.log(args.alpha_min):.3f}")
    print(f"[probe] t_grid: {t_grid[0]:.2f}→{t_grid[-1]:.2f} ({args.n_t_steps} steps)"
          f"  n_noise={args.n_noise}  seq_len={args.seq_len}")

    # Preview masking rates
    print("\n── Masking rate preview ──────────────────────────────────────────")
    for t in t_grid[::max(1, args.n_t_steps // 6)]:
        a = loglinear_alpha(t, args.alpha_min)
        s = loglinear_sigma(t, args.alpha_min)
        print(f"  t={t:.2f}  α={a:.5f}  mask_prob={1-a:.5f}  σ={s:.3f}")
    print()

    # ── Load data ──────────────────────────────────────────────────────────────
    texts = load_owt_texts(args.n_samples)
    print(f"[data] {len(texts)} texts loaded")

    # ── Main probe loop ────────────────────────────────────────────────────────
    seq_results = []

    for seq_idx, text in enumerate(texts):
        ids, attn_mask = encode_text(text, tokenizer, args.seq_len)

        print(f"\n── Seq {seq_idx+1}/{args.n_samples} — collecting logits…")

        logits_arr, masked_arr, alpha_arr = collect_logits_mdlm(
            model       = model,
            ids_clean   = ids,
            t_grid      = t_grid,
            n_noise     = args.n_noise,
            seed        = seq_idx * 1000,
            mask_id     = mask_id,
            alpha_min   = args.alpha_min,
            attn_mask   = attn_mask,
        )

        metrics = compute_metrics_mdlm(
            logits_arr    = logits_arr,
            masked_arr    = masked_arr,
            alpha_arr     = alpha_arr,
            gt_ids        = ids,
            t_grid        = t_grid,
            tau           = args.tau,
            commit_thresh = args.commit_thresh,
            topk          = args.topk,
        )
        seq_results.append(metrics)

        # Live printout
        for ti, t in enumerate(t_grid):
            print_row(
                t       = t,
                alpha   = float(alpha_arr[ti]),
                n_masked_frac = metrics["mask_rate"][ti],
                H       = metrics["entropy_masked"][ti],
                top1_gt = metrics["top1_gt_masked"][ti],
                jsd_val = metrics["jsd_masked"][ti],
            )

    # ── Aggregate + save ────────────────────────────────────────────────────────
    agg = aggregate(seq_results)
    agg["args"] = vars(args)
    agg["config"] = cfg

    json_path = out_dir / "mdlm_probe.json"
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[saved] {json_path}")

    plot_results(agg, out_dir)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n── Summary (aggregated over sequences) ───────────────────────────")
    print(f"{'t':>5}  {'mask':>6}  {'H_mask':>8}  {'top1_gt':>8}"
          f"  {'soft_c%':>8}  {'sc_cor':>7}  {'jsd':>8}")
    print("-" * 65)
    for ti, t in enumerate(t_grid):
        def v(k): return agg.get(f"{k}_mean", [float("nan")]*len(t_grid))[ti]
        print(f"{t:>5.2f}  {v('mask_rate'):>6.3f}  {v('entropy_masked'):>8.3f}"
              f"  {v('top1_gt_masked'):>8.3f}  {v('soft_commit_frac'):>8.3f}"
              f"  {v('soft_commit_correct'):>7.3f}  {v('jsd_masked'):>8.4f}")

    print(f"\n  commit_time_mean (agg): {agg.get('commit_time_mean_agg', float('nan')):.4f}")
    print(f"  never_committed  (agg): {agg.get('never_committed_frac_agg', float('nan')):.4f}")


if __name__ == "__main__":
    main()
