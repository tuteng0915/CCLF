"""
probe_duo.py — DUO Discrete Diffusion Anchor Emergence Probe

Probes token crystallization in DUO (Sahoo et al. 2024,
"Simple and Effective Masked Diffusion Language Models" / "Duo" framework).
https://github.com/s-sahoo/duo

KEY DIFFERENCE FROM MDLM:
  MDLM (AbsorbingState): corrupted positions replaced with [MASK] → model KNOWS which to predict
  DUO  (UniformState):   corrupted positions replaced with RANDOM token → model must INFER them

This makes DUO scientifically more interesting for commitment analysis:
  - The model has no explicit masking signal — it must build internal representations
    to identify which positions are corrupted before it can correct them.
  - We externally track which positions WERE corrupted (gt != x_t) and measure
    recovery quality separately for corrupted vs. accidentally-clean positions.

Noise process (UniformState, loglinear schedule):
  q(x_t | x_0) = α_t · δ(x_t = x_0) + (1−α_t) · Uniform(V)
  σ_t = −log(α_t) = λ · (1−t)   [probe convention: t=0=noisy, t=1=clean]

Model forward:
  log_probs = model(xt, sigma)   # sigma: [B, 1],  output: [B, L, V] LOG probabilities

Metric split:
  _corr   computed for positions where x_t ≠ x_0   (uniformly corrupted)
  _clean  computed for positions where x_t == x_0   (happened to keep original)
  _all    computed for all positions (no split)

Setup:
  git clone https://github.com/s-sahoo/duo ~/duo
  cd ~/duo && pip install -e .

Usage:
  python probe_duo.py \\
      --checkpoint /path/to/model.ckpt \\
      --config /path/to/config.yaml \\
      --n_samples 64 --seq_len 128 --out_dir ~/probe_duo_v1

  # smoke test:
  python probe_duo.py --stub --out_dir ~/probe_duo_stub
"""

import sys, os, argparse, json, math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUO_REPO = os.path.expanduser("~/duo")
if os.path.isdir(DUO_REPO) and DUO_REPO not in sys.path:
    sys.path.insert(0, DUO_REPO)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Noise schedule (loglinear, same as MDLM but without masking)
# ─────────────────────────────────────────────────────────────────────────────

def loglinear_alpha(t: float, alpha_min: float) -> float:
    """α_t = exp(−λ·(1−t)), λ = −log(α_min). t=0→α_min (noisy), t=1→1 (clean)."""
    lam = -math.log(alpha_min)
    return math.exp(-lam * (1.0 - t))


def loglinear_sigma(t: float, alpha_min: float) -> float:
    """σ_t = −log(α_t) = λ·(1−t). Shape expected by DUO model: scalar."""
    lam = -math.log(alpha_min)
    return lam * (1.0 - t)


# ─────────────────────────────────────────────────────────────────────────────
# 2. UniformState corruption
# ─────────────────────────────────────────────────────────────────────────────

def apply_uniform_noise(
    ids: np.ndarray,          # [L] clean token IDs
    alpha: float,             # preservation prob
    vocab_size: int,
    rng: np.random.Generator,
    attn_mask: Optional[np.ndarray] = None,  # [L] 1=real token, 0=padding
    bos_eos_ids: Optional[set] = None,       # IDs to never corrupt
) -> tuple:
    """
    UniformState forward process:
      x_t[i] = x_0[i]          with prob α
              = Uniform(V)      with prob (1−α)

    Returns:
      x_t         [L] corrupted token IDs
      is_corrupted [L] bool — True where x_t != x_0 (i.e., position was re-sampled)
    """
    L = len(ids)
    draw = rng.random(L)
    replace = draw >= alpha                     # positions to corrupt
    if attn_mask is not None:
        replace &= (attn_mask > 0)             # never corrupt padding
    if bos_eos_ids:
        for bid in bos_eos_ids:
            replace &= (ids != bid)            # never corrupt BOS/EOS

    random_tokens = rng.integers(0, vocab_size, size=L).astype(np.int64)
    x_t = ids.copy()
    x_t[replace] = random_tokens[replace]

    # A position is "truly corrupted" iff it was replaced AND the random token ≠ original
    # (If Uniform(V) happens to draw the same token, it's clean in value but corrupted in process)
    # We track PROCESS corruption: was replace[i] True?
    is_corrupted = replace.copy()
    return x_t, is_corrupted


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def logprob_to_prob(log_probs: np.ndarray, tau: float) -> np.ndarray:
    """Convert log-prob [L, V] → prob [L, V], with optional temperature scaling."""
    scaled = log_probs / tau
    scaled -= scaled.max(axis=-1, keepdims=True)
    p = np.exp(scaled)
    return p / p.sum(axis=-1, keepdims=True)


def token_entropy(p: np.ndarray) -> np.ndarray:
    """[L, V] → [L]"""
    pc = np.clip(p, 1e-9, 1.0)
    return -(pc * np.log(pc)).sum(axis=-1)


def top1_correct(p: np.ndarray, ref: np.ndarray) -> float:
    return float((np.argmax(p, -1) == ref).mean())


def topk_correct(p: np.ndarray, ref: np.ndarray, k: int) -> float:
    topk = np.argsort(p, -1)[:, -k:]
    return float((topk == ref[:, None]).any(-1).mean())


def jsd_scalar(p_prev: Optional[np.ndarray], p_curr: np.ndarray) -> Optional[float]:
    if p_prev is None or len(p_prev) == 0 or len(p_curr) == 0:
        return None
    pp = np.clip(p_prev, 1e-9, 1.0)
    pc = np.clip(p_curr, 1e-9, 1.0)
    m  = 0.5 * (pp + pc)
    return float((0.5 * ((pp * (np.log(pp) - np.log(m))).sum(-1)
                        + (pc * (np.log(pc) - np.log(m))).sum(-1))).mean())


def bimodality_coefficient(H: np.ndarray) -> float:
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
# 4. Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_duo(checkpoint_path: str, config_path: Optional[str] = None):
    """
    Load DUO model.

    DUO model forward: log_probs = model(input_ids, timesteps=sigma)
      input_ids : [B, L] integer token IDs (uniformly corrupted)
      timesteps : [B]   noise level σ = −log(α_t)
      output    : [B, L, V] log probabilities (return_dict=False)

    Returns: (model, tokenizer, vocab_size, config_dict)
    """
    # Try HuggingFace native loading first (s-sahoo/duo uses AutoModelForMaskedLM + custom code)
    try:
        return _load_duo_from_hf(checkpoint_path)
    except Exception as e:
        print(f"[duo] HF load failed ({e})")

    if os.path.isdir(DUO_REPO):
        try:
            return _load_duo_from_repo(checkpoint_path, config_path)
        except Exception as e:
            print(f"[duo] repo load failed ({e}), falling back to generic...")

    return _load_duo_direct(checkpoint_path)


def _load_duo_from_hf(checkpoint_path: str):
    """Load DUO via HuggingFace with trust_remote_code (patches flash_attn away)."""
    from transformers import GPT2Tokenizer, AutoModelForMaskedLM
    # DUO uses GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    model = AutoModelForMaskedLM.from_pretrained(
        checkpoint_path, trust_remote_code=True).to(device).eval()
    vocab_size = len(tokenizer)
    print(f"[duo] loaded via HuggingFace (trust_remote_code): {checkpoint_path}")
    return model, tokenizer, vocab_size, {}


def _load_duo_from_repo(checkpoint_path: str, config_path: Optional[str]):
    """Load via DUO repo's Lightning + OmegaConf system."""
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    local = Path(checkpoint_path)
    if local.is_dir():
        # Local directory with config + checkpoint inside
        cfg_path  = config_path or str(local / "config.yaml")
        ckpt_path = next(local.glob("*.ckpt"), None) or next(local.glob("*.pt"), None)
    elif local.exists() and local.suffix in (".ckpt", ".pt"):
        # Local .ckpt file — config must be provided separately
        cfg_path  = config_path
        ckpt_path = local
    else:
        # Treat as HuggingFace model ID — download snapshot
        from huggingface_hub import snapshot_download
        dl = Path(snapshot_download(checkpoint_path))
        cfg_path  = config_path or str(dl / "config.yaml")
        ckpt_path = next(dl.glob("*.ckpt"), None) or next(dl.glob("*.pt"), None)

    if config_path is None:
        config_path = cfg_path
    if not Path(config_path).exists():
        raise FileNotFoundError(f"No config.yaml at {config_path}")

    cfg = OmegaConf.load(config_path)

    # Load tokenizer
    tokenizer_name = (getattr(cfg.data, "tokenizer", None) or
                      getattr(cfg.data, "tokenizer_name", "gpt2"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for tok in ["bos_token", "eos_token", "pad_token"]:
        if getattr(tokenizer, tok) is None:
            tokenizer.add_special_tokens({tok: f"[{tok.upper()}]"})

    vocab_size = len(tokenizer)

    # Try to import the DUO diffusion model class
    model = None
    for mod_name, cls_name in [
        ("algo",         "DUO"),           # s-sahoo/duo: DUO extends DUO_BASE(UniformState)
        ("algo",         "DUO_BASE"),
        ("trainer_base", "UniformState"),
    ]:
        try:
            import importlib
            mod   = importlib.import_module(mod_name)
            cls   = getattr(mod, cls_name)
            model = cls.load_from_checkpoint(
                checkpoint_path, tokenizer=tokenizer, config=cfg, strict=False)
            print(f"[duo] loaded via {mod_name}.{cls_name}")
            break
        except (ImportError, AttributeError, Exception) as e:
            continue

    if model is None:
        raise RuntimeError("Cannot load DUO model class from repo.")

    model = model.to(device).eval()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return model, tokenizer, vocab_size, cfg_dict


def _load_duo_direct(checkpoint_path: str):
    """Fallback: load raw state_dict and wrap in a generic module."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("[duo] WARNING: loading as generic AutoModel, sigma conditioning ignored")
    tokenizer  = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model      = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    vocab_size = len(tokenizer)
    return model, tokenizer, vocab_size, {}


def duo_forward(model, x_t: torch.Tensor, sigma: torch.Tensor) -> np.ndarray:
    """
    DUO forward pass → probabilities [B, L, V].
    DUO outputs LOG probabilities; we convert to prob here.
    sigma: [B] noise level = -log(alpha_t)

    HF DUO model signature: forward(input_ids, timesteps=sigma)
    """
    with torch.no_grad():
        for call in [
            lambda: model(input_ids=x_t, timesteps=sigma),  # HF DUO (s-sahoo/duo)
            lambda: model(x_t, sigma),                       # positional: timesteps
            lambda: model(x_t, sigma.unsqueeze(1)),          # [B,1] sigma
            lambda: model(input_ids=x_t, sigma=sigma),       # old kwarg
            lambda: model(input_ids=x_t).logits,             # causal LM fallback
        ]:
            try:
                out = call()
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if hasattr(out, "logits"):
                    out = out.logits
                out = out.float().cpu().numpy()   # [B, L, V]
                # DUO returns log-probs; convert to prob
                out = np.exp(out - out.max(-1, keepdims=True))
                out = out / out.sum(-1, keepdims=True)
                return out
            except (TypeError, AttributeError):
                # TypeError: unexpected keyword; AttributeError: e.g. GPT-2 uses sigma as past_kv
                continue
    raise RuntimeError("Could not call DUO model. Check duo_forward() signatures.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stub model for testing
# ─────────────────────────────────────────────────────────────────────────────

class _StubDUO(torch.nn.Module):
    def __init__(self, vocab_size: int = 50257, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, 64)
        self.head  = torch.nn.Linear(64, vocab_size, bias=False)

    def forward(self, x, sigma=None):
        h = self.embed(x)
        return torch.log_softmax(self.head(h), dim=-1)   # [B, L, V] log-probs


def make_stub(vocab_size: int = 50257):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model     = _StubDUO(vocab_size=vocab_size).to(device).eval()
    return model, tokenizer, vocab_size, {"stub": True}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Logit collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_probs_duo(
    model,
    ids_clean:   np.ndarray,       # [L]
    t_grid:      np.ndarray,       # [T]
    n_noise:     int,
    seed:        int,
    vocab_size:  int,
    alpha_min:   float,
    attn_mask:   Optional[np.ndarray] = None,
    bos_eos_ids: Optional[set] = None,
) -> tuple:
    """
    Returns:
      probs_arr    [T, N, L, V]  — probabilities (after softmax on log-probs)
      corrupt_arr  [T, N, L]    — bool, True = position was uniformly corrupted
      alpha_arr    [T]          — α_t values
    """
    rng = np.random.default_rng(seed)
    L   = len(ids_clean)
    T, N = len(t_grid), n_noise

    probs_arr  = np.zeros((T, N, L, vocab_size), dtype=np.float32)
    corrupt_arr = np.zeros((T, N, L), dtype=bool)
    alpha_arr  = np.zeros(T, dtype=np.float32)

    for ti, t in enumerate(t_grid):
        alpha = loglinear_alpha(t, alpha_min)
        sigma = loglinear_sigma(t, alpha_min)
        alpha_arr[ti] = alpha

        x_batch = np.zeros((N, L), dtype=np.int64)
        corr_b  = np.zeros((N, L), dtype=bool)
        for si in range(N):
            x_t, is_c = apply_uniform_noise(
                ids_clean, alpha, vocab_size, rng, attn_mask, bos_eos_ids)
            x_batch[si] = x_t
            corr_b[si]  = is_c

        corrupt_arr[ti] = corr_b

        x_torch = torch.from_numpy(x_batch).long().to(device)
        sigma_t = torch.full((N,), sigma, device=device, dtype=torch.float32)

        probs_np = duo_forward(model, x_torch, sigma_t)   # [N, L, V]
        probs_arr[ti] = probs_np

    return probs_arr, corrupt_arr, alpha_arr


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metric computation
# ─────────────────────────────────────────────────────────────────────────────

ALL_METRICS = [
    "corr_rate",           # fraction of positions corrupted at this t
    # Corrupted positions: model must RECOVER from random token
    "entropy_corr",   "entropy_corr_p10", "entropy_corr_p50", "entropy_corr_p90",
    "top1_gt_corr",   "top5_gt_corr",
    "soft_commit_corr",          # H < thresh for corrupted positions
    "soft_commit_corr_correct",  # of those, fraction correct
    "soft_commit_corr_wrong",
    "bimodality_corr",
    # Clean positions: model should confirm original token
    "entropy_clean",  "top1_gt_clean",
    # All positions
    "entropy_all",    "top1_gt_all",   "top5_gt_all",
    # Inter-step transitions (all positions)
    "jsd_all",   "w2c_all",   "c2w_all",   "rev_top1_all",
    # Inter-step for corrupted positions only
    "jsd_corr",  "w2c_corr",  "c2w_corr",
]
SCALAR_KEYS = ["commit_time_mean", "commit_time_std", "never_committed_frac"]


def compute_metrics_duo(
    probs_arr:    np.ndarray,   # [T, N, L, V]
    corrupt_arr:  np.ndarray,   # [T, N, L] bool
    alpha_arr:    np.ndarray,   # [T]
    gt_ids:       np.ndarray,   # [L]
    t_grid:       np.ndarray,   # [T]
    tau:          float,
    commit_thresh: float,
    topk:         int,
) -> dict:
    T, N, L, V = probs_arr.shape
    out = {k: [] for k in ALL_METRICS}
    out["t"] = t_grid.tolist()

    prev_p_all  = [None] * N    # [L, V] previous step p for each seed
    prev_p_corr = [None] * N    # previous p restricted to corrupted positions

    commit_times_corr = np.full((N, L), np.nan)   # soft commit time for each position

    for ti in range(T):
        a = {k: [] for k in ALL_METRICS}

        for si in range(N):
            p_all    = probs_arr[ti, si]          # [L, V]
            is_corr  = corrupt_arr[ti, si]         # [L] bool

            # Rescale probabilities with tau (probs are already from softmax)
            if tau != 1.0:
                log_p = np.log(np.clip(p_all, 1e-9, 1.0)) / tau
                log_p -= log_p.max(-1, keepdims=True)
                p_all = np.exp(log_p) / np.exp(log_p).sum(-1, keepdims=True)

            H_all  = token_entropy(p_all)           # [L]
            top1_a = np.argmax(p_all, -1)           # [L]

            # ── All positions ────────────────────────────────────────────────
            a["corr_rate"].append(float(is_corr.mean()))
            a["entropy_all"].append(float(H_all.mean()))
            a["top1_gt_all"].append(float((top1_a == gt_ids).mean()))
            tk_all = np.argsort(p_all, -1)[:, -topk:]
            a["top5_gt_all"].append(float((tk_all == gt_ids[:, None]).any(-1).mean()))

            # ── Corrupted positions ──────────────────────────────────────────
            n_corr = is_corr.sum()
            if n_corr > 0:
                p_c   = p_all[is_corr]              # [n_corr, V]
                gt_c  = gt_ids[is_corr]             # [n_corr]
                H_c   = H_all[is_corr]
                top1_c = top1_a[is_corr]
                correct_c = (top1_c == gt_c)

                a["entropy_corr"].append(float(H_c.mean()))
                a["entropy_corr_p10"].append(float(np.percentile(H_c, 10)))
                a["entropy_corr_p50"].append(float(np.percentile(H_c, 50)))
                a["entropy_corr_p90"].append(float(np.percentile(H_c, 90)))
                a["top1_gt_corr"].append(float(correct_c.mean()))
                tk_c = np.argsort(p_c, -1)[:, -topk:]
                a["top5_gt_corr"].append(float((tk_c == gt_c[:, None]).any(-1).mean()))

                sc = H_c < commit_thresh
                a["soft_commit_corr"].append(float(sc.mean()))
                a["soft_commit_corr_correct"].append(float(correct_c[sc].mean()) if sc.sum() > 0 else float("nan"))
                a["soft_commit_corr_wrong"].append(float((~correct_c)[sc].mean()) if sc.sum() > 0 else float("nan"))
                a["bimodality_corr"].append(bimodality_coefficient(H_c))

                # Commitment time for corrupted positions
                for ii, idx in enumerate(np.where(is_corr)[0]):
                    if sc[ii] and np.isnan(commit_times_corr[si, idx]):
                        commit_times_corr[si, idx] = t_grid[ti]

                # Inter-step transitions for corrupted positions
                prev_pc = prev_p_corr[si]
                if prev_pc is not None:
                    min_n = min(len(prev_pc), n_corr)
                    if min_n > 0:
                        jd = jsd_scalar(prev_pc[:min_n], p_c[:min_n])
                        a["jsd_corr"].append(jd if jd is not None else float("nan"))
                        pt1_prev = np.argmax(prev_pc[:min_n], -1)
                        pt1_curr = np.argmax(p_c[:min_n], -1)
                        gt_prev  = gt_c[:min_n]
                        pc_prev = pt1_prev == gt_prev
                        pc_curr = pt1_curr == gt_prev
                        a["w2c_corr"].append(float((~pc_prev & pc_curr).mean()))
                        a["c2w_corr"].append(float((pc_prev & ~pc_curr).mean()))
                prev_p_corr[si] = p_c

            else:
                for k in ["entropy_corr","entropy_corr_p10","entropy_corr_p50","entropy_corr_p90",
                          "top1_gt_corr","top5_gt_corr","soft_commit_corr",
                          "soft_commit_corr_correct","soft_commit_corr_wrong",
                          "bimodality_corr","jsd_corr","w2c_corr","c2w_corr"]:
                    a[k].append(float("nan"))
                prev_p_corr[si] = None

            # ── Clean positions ──────────────────────────────────────────────
            n_clean = (~is_corr).sum()
            if n_clean > 0:
                p_cl  = p_all[~is_corr]
                gt_cl = gt_ids[~is_corr]
                a["entropy_clean"].append(float(token_entropy(p_cl).mean()))
                a["top1_gt_clean"].append(float((np.argmax(p_cl, -1) == gt_cl).mean()))
            else:
                a["entropy_clean"].append(float("nan"))
                a["top1_gt_clean"].append(float("nan"))

            # ── All-position inter-step ──────────────────────────────────────
            if prev_p_all[si] is not None:
                a["jsd_all"].append(jsd_scalar(prev_p_all[si], p_all))
                prev1 = np.argmax(prev_p_all[si], -1)
                curr1 = np.argmax(p_all, -1)
                pc_all_prev = (prev1 == gt_ids)
                pc_all_curr = (curr1 == gt_ids)
                a["w2c_all"].append(float((~pc_all_prev &  pc_all_curr).mean()))
                a["c2w_all"].append(float(( pc_all_prev & ~pc_all_curr).mean()))
                a["rev_top1_all"].append(float((prev1 != curr1).mean()))
            prev_p_all[si] = p_all

        def mv(lst):
            arr = [x for x in lst if x is not None and not (isinstance(x, float) and math.isnan(x))]
            return float(np.mean(arr)) if arr else float("nan")

        for k in ALL_METRICS:
            out[k].append(mv(a[k]))

    # Commitment time stats (corrupted positions)
    out["commit_time_mean"]     = float(np.nanmean(commit_times_corr))
    out["commit_time_std"]      = float(np.nanstd(commit_times_corr))
    out["never_committed_frac"] = float(np.isnan(commit_times_corr).mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. Aggregation
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


# ─────────────────────────────────────────────────────────────────────────────
# 9. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(agg: dict, out_dir: Path):
    t = np.array(agg["t"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("DUO (UniformState) Anchor Emergence Probe")

    panels = [
        ("corr_rate_mean",           "Corruption rate (1−α_t)",         "Fraction corrupted"),
        ("entropy_corr_mean",        "Entropy H — corrupted positions",  "H(p_θ)"),
        ("top1_gt_corr_mean",        "Top-1 GT — corrupted positions",   "Recovery rate"),
        ("soft_commit_corr_mean",    "Soft commit frac (corrupted pos)", "Fraction H<thresh"),
        ("jsd_all_mean",             "JSD all positions (consecutive)",  "JSD"),
        ("w2c_corr_mean",            "Wrong→Correct (corrupted pos)",    "Fraction"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        y = np.array(agg.get(key, [float("nan")] * len(t)))
        ax.plot(t, y, lw=2, color="tomato")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t (0=noisy, 1=clean)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    # Overlay clean vs corrupted top1 on panel 3
    ax = axes[0, 2]
    clean = np.array(agg.get("top1_gt_clean_mean", [float("nan")] * len(t)))
    ax.plot(t, clean, lw=1.5, ls="--", color="steelblue", label="clean pos")
    ax.legend(fontsize=8)

    # Overlay correct/wrong soft-commit on panel 4
    ax = axes[1, 0]
    cor = np.array(agg.get("soft_commit_corr_correct_mean", [float("nan")] * len(t)))
    wrg = np.array(agg.get("soft_commit_corr_wrong_mean",   [float("nan")] * len(t)))
    ax.plot(t, cor, lw=1.5, ls="--", color="green", label="correct")
    ax.plot(t, wrg, lw=1.5, ls="--", color="red",   label="wrong")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "duo_probe.png"
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


# ─────────────────────────────────────────────────────────────────────────────
# 11. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DUO UniformState anchor emergence probe")
    p.add_argument("--checkpoint",    type=str, default=None,
                   help="Path to .ckpt file or HuggingFace model ID")
    p.add_argument("--config",        type=str, default=None,
                   help="Path to config.yaml (required unless --stub)")
    p.add_argument("--stub",          action="store_true")
    p.add_argument("--n_samples",     type=int, default=64)
    p.add_argument("--seq_len",       type=int, default=128)
    p.add_argument("--n_t_steps",     type=int, default=21)
    p.add_argument("--n_noise",       type=int, default=4)
    p.add_argument("--tau",           type=float, default=1.0)
    p.add_argument("--topk",          type=int, default=5)
    p.add_argument("--commit_thresh", type=float, default=0.1)
    p.add_argument("--alpha_min",     type=float, default=1e-4)
    p.add_argument("--out_dir",       type=str, default="./probe_duo_results")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid = np.linspace(0.0, 1.0, args.n_t_steps)

    # ── Load model ─────────────────────────────────────────────────────────────
    if args.stub:
        print("[mode] Stub DUO (random weights)")
        model, tokenizer, vocab_size, cfg = make_stub()
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required unless --stub")
        print(f"[mode] DUO: {args.checkpoint}")
        model, tokenizer, vocab_size, cfg = load_duo(args.checkpoint, args.config)

    bos_eos = set()
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        v = getattr(tokenizer, attr, None)
        if v is not None:
            bos_eos.add(v)

    print(f"[duo] vocab_size={vocab_size}  device={device}")
    print(f"[schedule] loglinear  alpha_min={args.alpha_min}"
          f"  lambda={-math.log(args.alpha_min):.3f}")
    print(f"[probe] {args.n_t_steps} steps  n_noise={args.n_noise}  seq_len={args.seq_len}")

    # Noise schedule preview
    print("\n── Noise schedule preview ───────────────────────────────────────")
    for t in t_grid[::max(1, args.n_t_steps // 6)]:
        a = loglinear_alpha(t, args.alpha_min)
        print(f"  t={t:.2f}  α={a:.5f}  corr_prob={1-a:.5f}  σ={loglinear_sigma(t,args.alpha_min):.3f}")
    print()

    # ── Load data ──────────────────────────────────────────────────────────────
    texts = load_owt_texts(args.n_samples)
    print(f"[data] {len(texts)} texts loaded")

    # ── Main probe loop ────────────────────────────────────────────────────────
    seq_results = []

    for seq_idx, text in enumerate(texts):
        enc  = tokenizer(text, return_tensors="np", truncation=True,
                         max_length=args.seq_len, padding="max_length")
        ids  = enc["input_ids"][0].astype(np.int64)
        attn = enc["attention_mask"][0].astype(np.float32)

        print(f"\n── Seq {seq_idx+1}/{args.n_samples} — collecting probs…")

        probs_arr, corrupt_arr, alpha_arr = collect_probs_duo(
            model       = model,
            ids_clean   = ids,
            t_grid      = t_grid,
            n_noise     = args.n_noise,
            seed        = seq_idx * 1000,
            vocab_size  = vocab_size,
            alpha_min   = args.alpha_min,
            attn_mask   = attn,
            bos_eos_ids = bos_eos,
        )

        metrics = compute_metrics_duo(
            probs_arr     = probs_arr,
            corrupt_arr   = corrupt_arr,
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
            jd = metrics["jsd_corr"][ti]
            jd_s = f"{jd:.4f}" if not math.isnan(jd) else "   nan"
            print(f"  t={t:.2f}  α={alpha_arr[ti]:.4f}  corr={metrics['corr_rate'][ti]:.3f}"
                  f"  H_corr={metrics['entropy_corr'][ti]:.3f}"
                  f"  top1_rec={metrics['top1_gt_corr'][ti]:.3f}"
                  f"  jsd={jd_s}")

    # ── Aggregate + save ────────────────────────────────────────────────────────
    agg = aggregate(seq_results)
    agg["args"]   = vars(args)
    agg["config"] = cfg

    json_path = out_dir / "duo_probe.json"
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[saved] {json_path}")

    plot_results(agg, out_dir)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Summary (DUO UniformState) ────────────────────────────────────")
    print(f"{'t':>5}  {'corr':>6}  {'H_corr':>8}  {'top1_rec':>9}"
          f"  {'H_clean':>8}  {'top1_cln':>9}  {'jsd_corr':>9}")
    print("-" * 70)
    for ti, t in enumerate(t_grid):
        def v(k): return agg.get(f"{k}_mean", [float("nan")]*len(t_grid))[ti]
        print(f"{t:>5.2f}  {v('corr_rate'):>6.3f}  {v('entropy_corr'):>8.3f}"
              f"  {v('top1_gt_corr'):>9.3f}  {v('entropy_clean'):>8.3f}"
              f"  {v('top1_gt_clean'):>9.3f}  {v('jsd_corr'):>9.4f}")

    print(f"\n  commit_time_mean (corrupted pos): "
          f"{agg.get('commit_time_mean_agg', float('nan')):.4f}")
    print(f"  never_committed  (corrupted pos): "
          f"{agg.get('never_committed_frac_agg', float('nan')):.4f}")


if __name__ == "__main__":
    main()
