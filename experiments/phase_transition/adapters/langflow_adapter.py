"""FlowModelAdapter implementation for LangFlow.

Wraps models/LangFlow/probe_langflow.py's load_langflow/encode_with_langflow
plus the native LangFlow class methods (_forward_diffusion, _euler_edm_step),
so phase_transition/*.py analysis scripts can drive LangFlow through the same
interface as ELFAdapter.

Convention choices (documented in docs/specs/EXP-PT1-spec.md):
  - native_logsnr uses model.proposal (the learned GumbelProposal), NOT the
    linear gamma_from_t() used only for plotting in probe_langflow.py.
  - predicted_clean / self-conditioning state = _embed_tokens(softmax(logits)),
    matching LangFlow's own generate_samples() self-conditioning update.
"""

import os
import sys

import torch
import torch.nn.functional as F

_LANGFLOW_REPO = os.path.expanduser("~/LangFlow")
if os.path.isdir(_LANGFLOW_REPO) and _LANGFLOW_REPO not in sys.path:
    sys.path.insert(0, _LANGFLOW_REPO)

DEFAULT_CHECKPOINT = "Continuous-Rivals-Discrete/langflow-owt"


class LangFlowAdapter:
    """FlowModelAdapter for LangFlow."""

    name = "langflow"

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gamma_min = float(model.proposal.gamma_min)
        self.gamma_max = float(model.proposal.gamma_max)
        self.vocab_size = model.config.vocab_size
        self.seq_len = getattr(model.config, "model_length", 128)
        with torch.no_grad():
            self.E = model._get_embedding_matrix()  # (V, d), same space as latent
        self.d_model = self.E.shape[-1]
        self.self_conditioning = bool(model.config.self_conditioning)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, checkpoint=DEFAULT_CHECKPOINT, device=None):
        from langflow import LangFlow, LangFlowConfig
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = checkpoint if os.path.isdir(checkpoint) else snapshot_download(checkpoint)
        config = LangFlowConfig.from_pretrained(ckpt_dir)
        model = LangFlow(config)

        weight_file = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(weight_file):
            state_dict = load_file(weight_file, device="cpu")
        else:
            state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
        missing, _ = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[LangFlowAdapter] missing {len(missing)} keys")
        model = model.eval().to(device)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        print(f"[LangFlowAdapter] loaded {checkpoint}: "
              f"gamma_min={float(model.proposal.gamma_min):.2f} "
              f"gamma_max={float(model.proposal.gamma_max):.2f} "
              f"self_cond={model.config.self_conditioning}")

        return cls(model, tokenizer, device)

    # ------------------------------------------------------------------
    def encode_clean(self, token_ids, attention_mask=None):
        """token_ids: (N, L) long -> normalized token embeddings (N, L, d)."""
        with torch.no_grad():
            return self.model._embed_tokens(token_ids.to(self.device))

    def sample_epsilon(self, shape, generator=None):
        return torch.randn(shape, device=self.device, generator=generator)

    def t_to_gamma(self, t):
        """Linear display mapping t in [0,1] -> gamma (t=0 noisiest, t=1 clean).
        Only used to pick a comparable grid point; native_logsnr(t) below is
        the canonical schedule for actual noising/forward calls."""
        return self.gamma_max + float(t) * (self.gamma_min - self.gamma_max)

    def make_oracle_state(self, clean_state, epsilon, t):
        """z = alpha(gamma)*x + sigma(gamma)*eps, gamma = native_logsnr(t)."""
        gamma = self.native_logsnr(t)
        alpha = torch.sigmoid(torch.tensor(-gamma, device=self.device)).sqrt()
        sigma = torch.sigmoid(torch.tensor(gamma, device=self.device)).sqrt()
        return clean_state * alpha + epsilon * sigma

    def make_null_state(self, epsilon, t):
        gamma = self.native_logsnr(t)
        sigma = torch.sigmoid(torch.tensor(gamma, device=self.device)).sqrt()
        return epsilon * sigma

    # ------------------------------------------------------------------
    def forward_state(self, state, sc_state, t, capture_hidden=False, batch_size=8):
        """state, sc_state: (N, L, d). t: python float. Returns dict(logits,
        predicted_clean, hidden_states or None)."""
        N = state.shape[0]
        gamma = self.native_logsnr(t)
        all_logits, all_xpred = [], []
        hidden_all = None

        with torch.no_grad():
            for i in range(0, N, batch_size):
                z_b = state[i:i + batch_size].to(self.device)
                B_b = z_b.shape[0]
                sc_b = None
                if sc_state is not None:
                    sc_b = sc_state[i:i + batch_size].to(self.device)
                gamma_b = torch.full((B_b,), gamma, device=self.device, dtype=torch.float32)
                model_out = self.model(
                    noisy_embeds=z_b, timesteps=gamma_b, x_self_cond=sc_b,
                    output_hidden_states=capture_hidden, return_dict=False,
                )
                # forward() returns a bare tensor when output_hidden_states=False,
                # and (logits, hidden_states) only when it's True.
                logits, hidden = model_out if capture_hidden else (model_out, None)
                probs = F.softmax(logits.float(), dim=-1)
                x_pred = self.model._embed_tokens(probs)
                all_logits.append(logits.float().cpu())
                all_xpred.append(x_pred.float().cpu())
                if capture_hidden:
                    if hidden_all is None:
                        hidden_all = [[] for _ in hidden]
                    for li, h in enumerate(hidden):
                        hidden_all[li].append(h.detach().float().cpu())

        result = {
            "logits": torch.cat(all_logits, dim=0),
            "predicted_clean": torch.cat(all_xpred, dim=0),
            "velocity": None,
            "hidden_states": ([torch.cat(layer, dim=0) for layer in hidden_all]
                               if capture_hidden else None),
        }
        return result

    # ------------------------------------------------------------------
    def solver_step(self, state, sc_state, t, t_next):
        """Single Euler-EDM step, gamma-space (t, t_next are the *t* grid
        values; internally converted to gamma via native_logsnr)."""
        gamma_t = self.native_logsnr(t)
        gamma_s = self.native_logsnr(t_next)
        out = self.forward_state(state, sc_state, t)
        x_pred = out["predicted_clean"].to(self.device)
        gt = torch.tensor(gamma_t, device=self.device)
        gs = torch.tensor(gamma_s, device=self.device)
        z_next = self.model._euler_edm_step(state, x_pred, gt, gs)
        sc_next = x_pred if self.self_conditioning else None
        return z_next, sc_next

    # ------------------------------------------------------------------
    def native_logsnr(self, t):
        """Canonical LangFlow log-SNR schedule: the learned GumbelProposal,
        evaluated via inverse-CDF at quantile t (t=0 -> gamma_max/noisiest,
        t=1 -> gamma_min/clean), matching generate_samples()'s
        `gamma = self.proposal(t)` usage with t = linspace(1-eps, eps, ...)."""
        u = 1.0 - float(t)
        u = min(max(u, 1e-5), 1 - 1e-5)
        with torch.no_grad():
            gamma = self.model.proposal(torch.tensor(u, device=self.device))
        return float(gamma)

    def full_state_clone(self, state, sc_state):
        sc_clone = sc_state.clone() if sc_state is not None else None
        return state.clone(), sc_clone

    # ------------------------------------------------------------------
    def load_owt_sequences(self, n_samples, seq_len=None):
        """Returns (token_ids (N,L) long, attention_mask (N,L) float, clean_emb (N,L,d))."""
        from datasets import load_dataset
        seq_len = seq_len or self.seq_len

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
                print(f"[LangFlowAdapter] {name} failed: {e}")
        if not texts:
            raise RuntimeError("Could not load any OWT-like dataset for LangFlow.")

        ids_list, mask_list, emb_list = [], [], []
        with torch.no_grad():
            for text in texts:
                enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=seq_len, padding="max_length")
                ids = enc["input_ids"][0]
                mask = enc["attention_mask"][0]
                emb = self.model._embed_tokens(ids.to(self.device).unsqueeze(0))[0].cpu()
                ids_list.append(ids)
                mask_list.append(mask)
                emb_list.append(emb)
        return torch.stack(ids_list), torch.stack(mask_list), torch.stack(emb_list)
