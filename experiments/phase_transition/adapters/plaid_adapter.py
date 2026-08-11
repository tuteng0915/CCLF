"""FlowModelAdapter implementation for Plaid (Gulrajani & Hashimoto, NeurIPS 2023,
"Likelihood-Based Diffusion Language Models", igul222/plaid).

Wraps Plaid's VDM-style continuous embedding diffusion model so
phase_transition/*.py and global_state/*.py analysis scripts can drive it
through the same interface as ELFAdapter/LangFlowAdapter, for EXP-GS20
(cross-family replication of the GS16/GS17 mechanism).

Chosen over the spec's originally-proposed CDCD because DeepMind never
released CDCD code or checkpoints (verified 2026-08-01: only two unofficial,
partial, no-checkpoint community reproductions exist). Plaid is a real
alternative continuous diffusion LM with a released 1B-parameter checkpoint
trained on OpenWebText2 (same corpus family as the OWT-derived data ELF and
LangFlow both use), verified end-to-end in this environment (fluent,
coherent generation; word-level guidance and lexical constraints work
correctly) after building FlashAttention 1.0.4 and NVIDIA Apex from source
into an isolated `plaid` conda env (both required a matching CUDA 11.8
toolkit installed inside the env, plus a small patch removing 3 references
to the since-removed `torch._six` in the pinned Apex commit -- see this
session's setup notes).

Convention notes (audited directly against plaid/sample.py and
plaid/lib/models.py, NOT guessed):
  - Plaid's OWN native diffusion time has t=1 = noisiest (pure noise start),
    t=0 = cleanest, the OPPOSITE of this repo's t=0-noisy/t=1-clean
    convention used throughout ELF/LangFlow -- every method here converts
    via t_native = 1 - t.
  - VDM parameterization identical in form to LangFlowAdapter's:
    alpha_sq = sigmoid(-gamma), sigma_sq = sigmoid(gamma), gamma = gamma_0 +
    (gamma_1-gamma_0) * NoiseSchedule(t_native) with gamma_0/gamma_1 learned
    parameters loaded from the checkpoint (their nn.Parameter init values
    passed to GammaBounds() are overwritten by load_state_dict, so the
    literal init values used here don't matter).
  - The embedding space is only embed_dim=16 (vs. ELF's 512 / LangFlow's
    768) -- much lower-dimensional. Rank-based analyses (e.g. GS18 Part A)
    will behave very differently here (k=8 is HALF the ambient dimension,
    not a tiny fraction) -- flag this when interpreting any such results on
    Plaid, don't silently reuse ELF-scale k values.
  - solver_step implements Plaid's NATIVE ancestral (stochastic) VDM sampler
    (Appendix A.4 eq 33 of the VDM paper, transcribed directly from
    generate_samples() in sample.py), not a deterministic ODE step like ELF
    or LangFlow's Euler steps -- this is a real architecture difference, not
    a simplification; a DDIM-style deterministic variant exists in Plaid's
    own code (ddim_sampler=True) but is not exposed here since GS16-19's
    branching/calibration logic already assumes exploration works via
    injected randomness (matches ELF/LangFlow's own random-direction
    perturbations, just injected differently).
  - State tensors (z) are kept in float32 at the adapter boundary for
    interface consistency with ELF/LangFlow; internally, solver_step and
    make_oracle_state upcast to float64 for the ancestral-update arithmetic
    (matching Plaid's own sample.py, which does the whole outer sampling
    loop in fp64 "because of lots of annoying big/small numbers", while the
    transformer forward pass itself runs in fp32/bf16 either way) -- this
    repo's torch.set_default_dtype/device are NOT modified globally, unlike
    sample.py, to avoid side-effecting anything else running in the same
    process.
  - Self-conditioning input is x_selfcond (embedding-space predicted-clean
    from the previous step), cold-started at zero -- same role as ELF's/
    LangFlow's sc_state.

Usage: see docs/specs/EXP-GS20-spec.md for the adapter-gate verification
checklist (clean decode, oracle corruption, one solver step vs. reference,
fixed-seed reproducibility, generation quality, monotone log-SNR) that
should be run before trusting any GS16/17 numbers produced through this
adapter.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

PLAID_REPO = os.environ.get("PLAID_REPO_PATH",
                             "/home/wjzhang/tt_workspace/cdlm_candidates/plaid")
if os.path.isdir(PLAID_REPO) and PLAID_REPO not in sys.path:
    sys.path.insert(0, PLAID_REPO)

DEFAULT_WEIGHTS_PATH = os.environ.get(
    "PLAID_WEIGHTS_PATH",
    "/home/wjzhang/tt_workspace/cdlm_candidates/plaid_ckpt/plaid1b_weights")


class PlaidAdapter:
    """FlowModelAdapter for Plaid 1B."""

    name = "plaid"

    def __init__(self, modules, tokenizer, device, dim=2048, n_heads=32,
                 seq_len=1024, vocab_size=32768, embed_dim=16):
        self.modules = modules
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        self.d_model = embed_dim
        self.vocab_size = vocab_size
        self.t_eps = 0.05
        with torch.no_grad():
            self._embedding_matrix = modules["embedding_matrix"]().detach()  # (V,d), normalized

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, weights_path=None, dim=2048, n_blocks=24, n_heads=32,
             seq_len=1024, vocab_size=32768, embed_dim=16, device=None):
        import lib.models
        import mup

        weights_path = weights_path or DEFAULT_WEIGHTS_PATH
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def create_modules(d, h):
            return {
                "noise_schedule": lib.models.NoiseSchedule().float(),
                "gamma_bounds": lib.models.GammaBounds(-3., 6.).float(),
                "embedding_matrix": lib.models.EmbeddingMatrix(vocab_size, embed_dim).float(),
                "model": lib.models.DiffusionModel(d, embed_dim, n_blocks, h, vocab_size).float(),
            }

        modules = create_modules(dim, n_heads)
        base_modules = create_modules(256, 4)
        delta_modules = create_modules(128, 2)
        for key in modules:
            mup.set_base_shapes(modules[key], base_modules[key], delta=delta_modules[key])
            modules[key].to(device)

        for name, module in modules.items():
            state = torch.load(os.path.join(weights_path, f"{name}.pt"), map_location=device)
            module.load_state_dict(state)
            module.eval()

        print(f"[PlaidAdapter] loaded weights from {weights_path} "
              f"(dim={dim}, n_blocks={n_blocks}, n_heads={n_heads}, embed_dim={embed_dim})")

        # lib.datasets.openwebtext2_tokenizer() uses a path relative to the
        # plaid repo's own CWD ('misc/owt2_tokenizer.json') -- load directly
        # via an absolute path instead so this works regardless of caller CWD.
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(PLAID_REPO, "misc", "owt2_tokenizer.json"))
        return cls(modules, tokenizer, device, dim=dim, n_heads=n_heads,
                    seq_len=seq_len, vocab_size=vocab_size, embed_dim=embed_dim)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _gamma(self, t):
        """t: python float, adapter convention (0=noisiest,1=clean) -> scalar
        double tensor gamma (Plaid's negative-log-SNR, native convention).

        NoiseSchedule.forward's own body creates bare literal tensors (e.g.
        `torch.tensor([0.], device='cuda')`) with no explicit dtype, relying
        on sample.py's global `torch.set_default_dtype(torch.float64)` to
        make those float64 too. This adapter deliberately does NOT set that
        global default (to avoid side-effecting anything else in the same
        process), so it must be scoped narrowly here instead, around only
        this call into Plaid's own library code."""
        t_native = 1.0 - float(t)
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            gamma_0, gamma_1 = self.modules["gamma_bounds"]()
            t_batch = torch.tensor([t_native], device=self.device, dtype=torch.float64)
            g_tilde = self.modules["noise_schedule"](t_batch)  # (1,) in [0,1], double
            gamma = (gamma_0 + (gamma_1 - gamma_0) * g_tilde)[0]  # scalar double tensor
        finally:
            torch.set_default_dtype(prev_dtype)
        return gamma

    def native_logsnr(self, t):
        """Higher = cleaner, matching ELF/LangFlow's native_logsnr convention.
        Plaid's gamma is a negative log-SNR (SNR = exp(-gamma)), so
        log-SNR = -gamma."""
        return float(-self._gamma(t))

    def _alpha_sigma(self, t):
        gamma = self._gamma(t)
        alpha = torch.sigmoid(-gamma).sqrt()
        sigma = torch.sigmoid(gamma).sqrt()
        return alpha, sigma, gamma

    # ------------------------------------------------------------------
    def encode_clean(self, token_ids, attention_mask=None):
        """token_ids: (N,L) long -> (N,L,embed_dim) float32, normalized
        embedding lookup (Plaid diffuses directly in this space)."""
        emb = self._embedding_matrix[token_ids.to(self.device)]
        return emb.float()

    def sample_epsilon(self, shape, generator=None):
        return torch.randn(shape, device=self.device, generator=generator)

    @torch.no_grad()
    def make_oracle_state(self, clean_state, epsilon, t):
        """z = alpha(t)*x_clean + sigma(t)*eps, converted to Plaid's native
        (reversed) time convention internally. float64 internally for the
        VDM arithmetic, matching sample.py; returns float32.

        Explicitly no_grad (not just relying on _gamma's no_grad): calling
        the learned gamma_bounds/noise_schedule nn.Modules without no_grad
        leaves the output graph-tracked (requires_grad=True), which crashes
        any caller doing a plain `.numpy()` on the result. ELF/LangFlow never
        hit this because their make_oracle_state does pure arithmetic on a
        python-float t with no nn.Module call involved; Plaid's t must be
        run through learned modules to get alpha/sigma. First surfaced by
        EXP-GS18 Part A (analyze_rank_matched_modes.py), the first script to
        call make_oracle_state on real oracle-corrupted text rather than
        starting from pure noise like GS16/17/19 do."""
        alpha, sigma, _ = self._alpha_sigma(t)
        z = alpha.float() * clean_state.to(self.device) + sigma.float() * epsilon.to(self.device)
        return z.float()

    @torch.no_grad()
    def make_null_state(self, epsilon, t):
        _, sigma, _ = self._alpha_sigma(t)
        return (sigma.float() * epsilon.to(self.device)).float()

    # ------------------------------------------------------------------
    def forward_state(self, state, sc_state, t, capture_hidden=False, batch_size=8):
        """state, sc_state: (N,L,embed_dim) float32 (sc_state may be None).
        Returns dict(logits, predicted_clean, velocity=None, hidden_states=None)."""
        N = state.shape[0]
        _, _, gamma = self._alpha_sigma(t)
        all_logits, all_xrec = [], []

        with torch.no_grad():
            for i in range(0, N, batch_size):
                z_b = state[i:i + batch_size].to(self.device).float()
                B_b = z_b.shape[0]
                sc_b = (sc_state[i:i + batch_size].to(self.device).float()
                        if sc_state is not None else torch.zeros_like(z_b))
                gamma_b = gamma.float().expand(B_b)
                logits, x_reconst = self.modules["model"](
                    z=z_b, gamma=gamma_b, embedding_matrix=self._embedding_matrix,
                    bias_scale=1.0, x_selfcond=sc_b,
                )
                all_logits.append(logits.float().cpu())
                all_xrec.append(x_reconst.float().cpu())

        return {
            "logits": torch.cat(all_logits, dim=0),
            "predicted_clean": torch.cat(all_xrec, dim=0),
            "velocity": None,
            "hidden_states": None,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def solver_step(self, state, sc_state, t, t_next, generator=None, noise=None):
        """Native ancestral VDM step (stochastic; VDM paper Appendix A.4 eq
        33, transcribed from Plaid's generate_samples()). t, t_next: this
        repo's convention (0=noisy,1=clean); internally t_next > t must map
        to native t_next_native < t_native (moving toward native 0=clean).
        Returns (z_next, x_reconst) -- x_reconst becomes the next sc_state,
        matching ELF/LangFlow's solver_step convention."""
        B, L, d = state.shape
        alpha_sq_t = torch.sigmoid(-self._gamma(t)).double()
        gamma_s = self._gamma(t_next)
        alpha_sq_s = torch.sigmoid(-gamma_s).double()
        sigma_sq_s = torch.sigmoid(gamma_s).double()
        gamma_t = self._gamma(t)

        out = self.forward_state(state, sc_state, t, batch_size=B)
        x_reconst = out["predicted_clean"].to(self.device).double()
        z = state.to(self.device).double()

        c = -torch.expm1(gamma_s.double() - gamma_t.double())
        z_next = z * (1 - c) * alpha_sq_s.sqrt() / alpha_sq_t.sqrt()
        z_next = z_next + c * alpha_sq_s.sqrt() * x_reconst
        noise_std = (c * (1 - alpha_sq_s)).sqrt()
        if float(t_next) < 1.0 - 1e-9:  # skip final-step noise, matching sample.py's `if t > 0`
            if noise is None:
                noise = torch.randn(
                    z.shape, device=self.device, generator=generator, dtype=torch.float64
                )
            else:
                noise = noise.to(device=self.device, dtype=torch.float64)
                if noise.shape != z.shape:
                    raise ValueError(
                        f"explicit step noise shape {noise.shape} != state shape {z.shape}"
                    )
            z_next = z_next + noise_std * noise

        return z_next.float(), x_reconst.float()

    # ------------------------------------------------------------------
    def full_state_clone(self, state, sc_state):
        sc_clone = sc_state.clone() if sc_state is not None else None
        return state.clone(), sc_clone

    def decode_ids(self, ids_1d):
        return self.tokenizer.decode(ids_1d.tolist(), skip_special_tokens=False)

    # ------------------------------------------------------------------
    def load_owt_sequences(self, n_samples, seq_len=None):
        """Returns (token_ids (N,L) long, attention_mask (N,L) float,
        clean_emb (N,L,embed_dim)) -- same 3-return-value pattern as
        LangFlowAdapter (embedding lookup is cheap/deterministic here too)."""
        from datasets import load_dataset
        seq_len = seq_len or self.seq_len
        pad_id = 0  # '<|endoftext_R9VQqF0Ag7|>', verified via tokenizer.get_vocab()

        def _stream(name, **kw):
            ds = load_dataset(name, split="train", streaming=True, **kw)
            texts = []
            for ex in ds:
                t = ex["text"].strip()
                if len(t) > 200:
                    texts.append(t)
                if len(texts) >= n_samples:
                    break
            return texts

        texts = None
        for name, kw in [("Skylion007/openwebtext", {}),
                          ("stas/openwebtext-10k", {}),
                          ("wikitext", {"name": "wikitext-103-raw-v1"})]:
            try:
                texts = _stream(name, **kw)
                if texts:
                    break
            except Exception as e:
                print(f"[PlaidAdapter] {name} failed: {e}")
        if not texts:
            raise RuntimeError("Could not load any OWT-like dataset for Plaid.")

        ids_list, mask_list = [], []
        for text in texts:
            ids = self.tokenizer.encode(text).ids[:seq_len]
            n_valid = len(ids)
            mask = [1] * n_valid + [0] * (seq_len - n_valid)
            ids = ids + [pad_id] * (seq_len - n_valid)
            ids_list.append(ids)
            mask_list.append(mask)

        ids_t = torch.tensor(ids_list, dtype=torch.long)
        mask_t = torch.tensor(mask_list, dtype=torch.float32)
        emb = self.encode_clean(ids_t).cpu()
        return ids_t, mask_t, emb
