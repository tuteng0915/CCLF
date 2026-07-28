"""FlowModelAdapter implementation for ELF (PyTorch backend, models/ELF-torch).

Wraps the load_model / backbone-forward / _ode_step machinery that was
previously duplicated verbatim across experiments/probe_elf/probe_*.py
(probe_reverse_trajectory.py, probe_prior_null.py, probe_head_null.py), so
that phase_transition/*.py analysis scripts can stay architecture-agnostic.

Conventions kept identical to the existing P0 oracle-probe scripts (EXP-01v3,
EXP-05v3), on purpose, so PT-series numbers stay comparable to prior results:
  - oracle noising is z_t = t*x_clean + (1-t)*eps  (NOT scaled by
    config.denoiser_noise_scale -- that scale is a *training*-time schedule
    knob; every existing oracle-probe experiment omits it, and PT1 does the
    same for consistency).
  - self-conditioning input defaults to zeros (a no-op SC state) unless a
    caller explicitly threads x_pred_prev through solver_step.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
_ELF_TORCH_DIR = _THIS_DIR.parents[2] / "models" / "ELF-torch"
_ELF_SRC = _ELF_TORCH_DIR / "src"
if str(_ELF_SRC) not in sys.path:
    sys.path.insert(0, str(_ELF_SRC))

from configs.config import load_config_from_yaml  # noqa: E402
from modules.model import ELF_models  # noqa: E402
from modules.t5_encoder import get_encoder  # noqa: E402
from utils.sampling_utils import _ode_step  # noqa: E402

CHECKPOINTS = {
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
    "kd_cr": "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2": "converted/elf_b-owt-kd2_torch.pt",
}


class ELFAdapter:
    """FlowModelAdapter for ELF."""

    name = "elf"

    def __init__(self, model, tokenizer, t5_model, config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.t5_model = t5_model
        self.config = config
        self.device = device
        self.latent_mean = getattr(config, "latent_mean", 0.0)
        self.latent_std = getattr(config, "latent_std", 0.2)
        self.seq_len = config.max_length
        self.d_model = 512  # T5-small hidden dim
        self.vocab_size = tokenizer.vocab_size
        self.t_eps = getattr(config, "t_eps", 0.05)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, checkpoint, config_path, device):
        """checkpoint: absolute path OR a key in CHECKPOINTS (resolved
        relative to models/ELF-torch)."""
        orig_dir = os.getcwd()
        os.chdir(str(_ELF_TORCH_DIR))
        try:
            abs_cfg = config_path if os.path.isabs(config_path) else os.path.join(orig_dir, config_path)
            config = load_config_from_yaml(abs_cfg)

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)
            enc_config, t5_model = get_encoder(config.encoder_model_name, dtype=torch.float32)
            t5_model = t5_model.to(device).eval()

            model_cls = ELF_models[config.model]
            model = model_cls(
                text_encoder_dim=enc_config.d_model,
                max_length=config.max_length,
                attn_drop=config.attn_dropout,
                proj_drop=config.proj_dropout,
                num_time_tokens=config.num_time_tokens,
                num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
                vocab_size=tokenizer.vocab_size,
                num_model_mode_tokens=config.num_model_mode_tokens,
                bottleneck_dim=config.bottleneck_dim,
            ).to(device).eval()

            ckpt_path = CHECKPOINTS.get(checkpoint, checkpoint)
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(str(_ELF_TORCH_DIR), ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "params" in ckpt:
                params = ckpt.get("ema_params1", ckpt["params"])
                missing, _ = model.load_state_dict(params, strict=False)
                if missing:
                    print(f"[ELFAdapter] missing {len(missing)} keys")
                print(f"[ELFAdapter] loaded {ckpt_path} (EMA: {'ema_params1' in ckpt})")
        finally:
            os.chdir(orig_dir)

        return cls(model, tokenizer, t5_model, config, device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_clean(self, token_ids, attention_mask):
        """token_ids, attention_mask: (N, L) long tensors -> x_clean (N, L, 512)."""
        out = self.t5_model.model(input_ids=token_ids.to(self.device),
                                   attention_mask=attention_mask.to(self.device))
        emb = (out.last_hidden_state - self.latent_mean) / self.latent_std
        return emb

    def sample_epsilon(self, shape, generator=None):
        return torch.randn(shape, device=self.device, generator=generator)

    def make_oracle_state(self, clean_state, epsilon, t):
        """z_t = t*x_clean + (1-t)*eps. t: python float."""
        return float(t) * clean_state + (1.0 - float(t)) * epsilon

    def make_null_state(self, epsilon, t):
        """Zero-signal reference state (no x_clean term), matching EXP-05v3."""
        return (1.0 - float(t)) * epsilon

    # ------------------------------------------------------------------
    def forward_state(self, state, sc_state, t, capture_hidden=False, batch_size=8):
        """state, sc_state: (N, L, 512). t: python float scalar (same t for whole batch).
        Returns dict(logits, predicted_clean, hidden_states or None).
        """
        N = state.shape[0]
        all_logits, all_xpred = [], []
        hidden_layers = None
        hooks = []
        captured = {}

        if capture_hidden:
            hidden_layers = [[] for _ in self.model.blocks]

            def make_hook(i):
                def hook(module, inp, out):
                    captured[i] = out.detach().float().cpu()
                return hook

            for i, block in enumerate(self.model.blocks):
                hooks.append(block.register_forward_hook(make_hook(i)))

        try:
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    z_b = state[i:i + batch_size].to(self.device)
                    B_b = z_b.shape[0]
                    if sc_state is not None and self.config.self_cond_prob > 0:
                        sc_b = sc_state[i:i + batch_size].to(self.device)
                        z_in = torch.cat([z_b, sc_b], dim=-1)
                    elif self.config.self_cond_prob > 0:
                        z_in = torch.cat([z_b, torch.zeros_like(z_b)], dim=-1)
                    else:
                        z_in = z_b
                    t_b = torch.full((B_b,), float(t), device=self.device, dtype=torch.float32)
                    sc_scale = (torch.zeros(B_b, device=self.device)
                                if self.config.num_self_cond_cfg_tokens > 0 else None)
                    x_pred, logits, _ = self.model(z_in, t_b, self_cond_cfg_scale=sc_scale,
                                                    decoder_step_active=True)
                    all_logits.append(logits.float().cpu())
                    all_xpred.append(x_pred.float().cpu())
                    if capture_hidden:
                        for li in range(len(self.model.blocks)):
                            hidden_layers[li].append(captured[li])
        finally:
            for h in hooks:
                h.remove()

        result = {
            "logits": torch.cat(all_logits, dim=0),
            "predicted_clean": torch.cat(all_xpred, dim=0),
            "velocity": None,
            "hidden_states": None,
        }
        if capture_hidden:
            result["hidden_states"] = [torch.cat(layer, dim=0) for layer in hidden_layers]
        return result

    # ------------------------------------------------------------------
    def solver_step(self, state, sc_state, t, t_next, cfg_scale=1.0, self_cond_cfg_scale=1.0):
        """Single reverse ODE (Euler) step. state: (B,L,512) on device."""
        B, L, _ = state.shape
        cond_seq = torch.zeros_like(state)
        cond_seq_mask = torch.zeros(B, L, device=state.device)
        x_pred_prev = sc_state if sc_state is not None else torch.zeros_like(state)
        z_next, x_pred = _ode_step(
            model=self.model, z=state, t=t, t_next=t_next, x_pred_prev=x_pred_prev,
            config=self.config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        )
        return z_next, x_pred

    # ------------------------------------------------------------------
    def native_logsnr(self, t):
        """ELF has no native log-SNR parametrization (linear flow-matching
        schedule alpha=t, sigma=1-t). This derived value is ONLY for
        cross-architecture log-SNR alignment with LangFlow -- it is not used
        anywhere in ELF's own forward pass."""
        import math
        t = min(max(float(t), 1e-4), 1 - 1e-4)
        return math.log(t ** 2) - math.log((1 - t) ** 2)

    def full_state_clone(self, state, sc_state):
        sc_clone = sc_state.clone() if sc_state is not None else None
        return state.clone(), sc_clone

    # ------------------------------------------------------------------
    def load_owt_sequences(self, n_samples, seq_len=None, split="train"):
        """Returns (token_ids (N,L) long, attention_mask (N,L) long) from the
        pre-tokenized T5 OpenWebText corpus used by every existing ELF probe."""
        from datasets import load_dataset
        seq_len = seq_len or self.seq_len
        ds = load_dataset("embedded-language-flows/openwebtext-t5", split=split, streaming=True)
        pad_id = self.tokenizer.eos_token_id or 1
        ids_list, mask_list = [], []
        for ex in ds:
            ids = torch.tensor(ex["input_ids"][:seq_len], dtype=torch.long)
            if len(ids) < seq_len:
                ids = F.pad(ids, (0, seq_len - len(ids)), value=pad_id)
            mask = (ids != pad_id).long()
            ids_list.append(ids)
            mask_list.append(mask)
            if len(ids_list) >= n_samples:
                break
        return torch.stack(ids_list), torch.stack(mask_list)
