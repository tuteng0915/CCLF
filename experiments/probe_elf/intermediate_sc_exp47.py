#!/usr/bin/env python3
"""
EXP-47: Intermediate-Layer SC

EXP-43 showed that baseline B11's residual update creates a geometric tradeoff:
it improves L_rec but worsens L_dec (h11 lands in a "reconstruction-friendly but
decode-hostile" direction). KD eliminates this tradeoff.

This experiment tests whether the SC quality (interaction I) depends on which
layer's x̂_t we use as SC conditioning input:

    x̂_α = final_layer(h_10 + α*(h_11 - h_10))

Arms:
  α=0.0  → SC uses x̂_t from h_10 (bypass B11 entirely)
  α=0.25 → partial B11
  α=0.50 → midpoint
  α=0.75 → mostly B11
  α=1.0  → standard SC (x̂_t from h_11, same as normal EXP-36v2 SC-only)
  "none" → no SC conditioning (baseline for I calculation)

Expected: baseline I(α) should improve (become less positive / negative) as α
decreases from 1.0 toward the "sweet spot" at α≈0.25 where EXP-43 showed
both L_dec and L_rec improving.

Metric: SC interaction I(α) = PPL(α, SC) - PPL(none).
        Negative → SC helps; Positive → SC hurts.

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/intermediate_sc_exp47.py --device cuda:0
    conda run -n elf python3 experiments/probe_elf/intermediate_sc_exp47.py --device cuda:0 --ckpt baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ── path setup ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]  # models/ELF-torch/
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import restore_cond, _ode_step, get_sampling_steps

# ── config ───────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
    "kd_cr":    "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":      "converted/elf_b-owt-kd2_torch.pt",
}

ALPHAS       = [0.0, 0.25, 0.50, 0.75, 1.0]
N_SEQ        = 64      # number of generated sequences per arm
N_STEPS      = 32      # ODE steps
SEED         = 42
MAX_LENGTH   = 128     # tokens
BATCH_SIZE   = 16
SC_T_MIN     = 0.5    # only apply intermediate SC when t >= SC_T_MIN
                       # (matches EXP-36v2 gate: dec_sc_apply_t_min=0.5)
OUT_DIR      = Path("results/exp47_intermediate_sc")

# ELF-B model hyperparameters (from eval configs).
# Note: ELF_B() already sets depth=12, hidden_size=768, num_heads=12 — do not repeat.
MODEL_CFG = dict(
    text_encoder_dim=512,
    max_length=MAX_LENGTH,
    num_time_tokens=4,
    num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4,
    vocab_size=32100,
    bottleneck_dim=128,
)
# For reference only (used in noise init):
HIDDEN_DIM = 768

# PPL evaluation model
PPL_MODEL_NAME = "gpt2-large"


# ── model loading ─────────────────────────────────────────────────────────────

def load_elf_model(ckpt_name: str, device: torch.device) -> torch.nn.Module:
    ckpt_path = REPO_ROOT / CHECKPOINTS[ckpt_name]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = ckpt["params"]

    model = ELF_B(**MODEL_CFG)
    missing, unexpected = model.load_state_dict(params, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}")
    model.eval().to(device)
    return model


# ── intermediate-layer hook ───────────────────────────────────────────────────

class IntermediateSCHook:
    """
    Registers two hooks on an ELF model:
      1. Forward hook on blocks[10]  → saves h_10 WITH prefix tokens
      2. Forward pre-hook on final_layer → saves h_11 WITHOUT prefix tokens

    After each forward call, call .get_xhat_alpha(alpha) to obtain
    x̂_α = final_layer(h_10_seq + α*(h_11_seq - h_10_seq)).
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._h10_with_prefix = None   # [B, prefix+L, 768]
        self._h11_no_prefix   = None   # [B, L, 768]

        self._h10_handle = model.blocks[10].register_forward_hook(self._cap_h10)
        self._h11_handle = model.final_layer.register_forward_pre_hook(self._cap_h11)

    def _cap_h10(self, module, inp, out):
        self._h10_with_prefix = out.detach().float()  # clone out of autocast

    def _cap_h11(self, module, inp):
        # inp is a tuple; inp[0] is h_11 (already prefix-stripped by model.forward line 229)
        self._h11_no_prefix = inp[0].detach().float()

    def get_xhat_alpha(self, alpha: float) -> torch.Tensor:
        """Compute x̂_α = final_layer(h_10_seq + α*(h_11_seq - h_10_seq)).

        Returns [B, L, 512] tensor on same device as model.
        """
        h10 = self._h10_with_prefix
        h11 = self._h11_no_prefix
        if h10 is None or h11 is None:
            raise RuntimeError("Hooks did not fire — was a forward pass run?")

        seq_len = h11.shape[1]                   # L (without prefix)
        h10_seq = h10[:, -seq_len:, :]           # strip prefix: last L positions

        # Interpolate in hidden space
        h_alpha = h10_seq + alpha * (h11 - h10_seq)

        # Apply final_layer (RMSNorm + Linear)
        with torch.no_grad():
            xhat = self.model.final_layer(h_alpha.to(next(self.model.parameters()).device))
        return xhat

    def remove(self):
        self._h10_handle.remove()
        self._h11_handle.remove()


# ── generation loop with intermediate SC ─────────────────────────────────────

class _Config:
    """Minimal config duck-type for _ode_step."""
    def __init__(self):
        self.t_eps = 0.05
        self.self_cond_prob = 1.0
        self.num_self_cond_cfg_tokens = 4
        self.denoiser_noise_scale = 1.0
        self.use_bf16 = True


@torch.no_grad()
def generate_intermediate_sc(
    model: torch.nn.Module,
    hook: IntermediateSCHook,
    z0: torch.Tensor,          # [N, L, 512] initial noise
    t_steps: torch.Tensor,
    alpha: float,              # interpolation coefficient
    sc_t_min: float,           # minimum t to apply intermediate SC
    use_sc: bool,              # if False: no SC conditioning (none arm)
    device: torch.device,
    cond_seq=None,
    cond_seq_mask=None,
) -> torch.Tensor:
    """
    Custom generation loop with intermediate-layer SC.

    Returns z at t=1 (the final generated embedding).
    """
    cfg = _Config()
    n = t_steps.shape[0]
    batch_size, max_length, d_model = z0.shape

    if cond_seq is None:
        cond_seq = torch.zeros_like(z0)
        cond_seq_mask = torch.zeros(batch_size, max_length, dtype=z0.dtype, device=device)

    step_kwargs = dict(
        model=model,
        config=cfg,
        cfg_scale=1.0,
        self_cond_cfg_scale=1.0,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
    )

    z = restore_cond(z0.clone(), cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 2):
            t      = t_steps[i].item()
            t_next = t_steps[i + 1].item()

            z, x_pred = _ode_step(
                z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs
            )

            # After the step, x_pred = x̂_{h11} (standard).
            # If use_sc and we are above the gate: replace with x̂_{h_α}.
            if use_sc and t_next >= sc_t_min and abs(alpha - 1.0) > 1e-6:
                x_pred_alpha = hook.get_xhat_alpha(alpha)
                x_pred = restore_cond(x_pred_alpha.to(device), cond_seq, cond_seq_mask)
            elif not use_sc:
                # Force zero SC: replace x_pred with zeros so the next step
                # receives no useful conditioning signal.
                x_pred = restore_cond(torch.zeros_like(x_pred), cond_seq, cond_seq_mask)

        # Last step (always ODE)
        t      = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z, x_pred = _ode_step(
            z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs
        )

    return z


# ── decode z → token ids ──────────────────────────────────────────────────────

@torch.no_grad()
def decode_z_to_ids(z: torch.Tensor, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Run decode branch at t=1 to get token IDs. Returns [B, L]."""
    B = z.shape[0]
    sc_scale = torch.ones(B, dtype=z.dtype, device=device)
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)  # [B, L, 1024]
    t_final = torch.ones(B, dtype=z.dtype, device=device)

    use_bf16 = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_logits, _ = model(
            z_in, t_final,
            deterministic=True,
            self_cond_cfg_scale=sc_scale,
            decoder_step_active=True,
        )
    return decoder_logits.argmax(dim=-1)  # [B, L]


# ── text decoding & PPL ───────────────────────────────────────────────────────

def ids_to_text(token_ids: torch.Tensor, tokenizer) -> list[str]:
    """Convert [B, L] token IDs to a list of strings."""
    eos_id = tokenizer.eos_token_id
    texts = []
    for row in token_ids.tolist():
        # Truncate at first EOS
        try:
            row = row[:row.index(eos_id)]
        except ValueError:
            pass
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def compute_ppl(texts: list[str], ppl_model, ppl_tokenizer, device: torch.device,
                max_length: int = 256) -> float:
    """Compute GPT-2 PPL on a list of text strings."""
    if not texts:
        return float("nan")

    enc = ppl_tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length, return_attention_mask=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask  = enc["attention_mask"].to(device)

    total_nll = 0.0
    total_tok = 0

    chunk = 8
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], chunk):
            ids_b  = input_ids[i:i + chunk]
            mask_b = attn_mask[i:i + chunk]

            out    = ppl_model(input_ids=ids_b, attention_mask=mask_b)
            logits = out.logits[:, :-1, :].float()   # [B, L-1, V]
            tgts   = ids_b[:, 1:]                     # [B, L-1]
            mask_s = mask_b[:, 1:].bool()             # [B, L-1]

            log_p  = F.log_softmax(logits, dim=-1)
            tgt_lp = log_p.gather(-1, tgts.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
            nll    = -(tgt_lp * mask_s.float()).sum().item()
            total_nll += nll
            total_tok += mask_s.sum().item()

    if total_tok == 0:
        return float("nan")
    return (total_nll / total_tok)  # mean NLL; PPL = exp(mean_NLL)


# ── main ──────────────────────────────────────────────────────────────────────

def run_ckpt(ckpt_name: str, device: torch.device, ppl_model, ppl_tok, elf_tok):
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_name}")
    print(f"{'='*60}")

    model = load_elf_model(ckpt_name, device)
    hook  = IntermediateSCHook(model)

    # Fixed initial noise for reproducibility
    gen  = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, MODEL_CFG["text_encoder_dim"],
                         generator=gen, device=device)

    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    results = {}

    # ── arm: none (no SC) ─────────────────────────────────────────────────
    print("  arm: none")
    texts_none = []
    for b_start in range(0, N_SEQ, BATCH_SIZE):
        z0_b = all_z0[b_start:b_start + BATCH_SIZE]
        z_final = generate_intermediate_sc(
            model=model, hook=hook, z0=z0_b, t_steps=t_steps,
            alpha=1.0, sc_t_min=SC_T_MIN, use_sc=False, device=device,
        )
        ids = decode_z_to_ids(z_final, model, device)
        texts_none.extend(ids_to_text(ids.cpu(), elf_tok))
    ppl_none = compute_ppl(texts_none, ppl_model, ppl_tok, device)
    results["none"] = {"ppl": ppl_none, "mean_nll": ppl_none}
    print(f"    PPL(none) = {ppl_none:.2f}")

    # ── arms: SC_α for each alpha ──────────────────────────────────────────
    for alpha in ALPHAS:
        arm_key = f"sc_alpha_{alpha:.2f}"
        print(f"  arm: SC α={alpha:.2f}")
        texts_sc = []
        for b_start in range(0, N_SEQ, BATCH_SIZE):
            z0_b = all_z0[b_start:b_start + BATCH_SIZE]
            z_final = generate_intermediate_sc(
                model=model, hook=hook, z0=z0_b, t_steps=t_steps,
                alpha=alpha, sc_t_min=SC_T_MIN, use_sc=True, device=device,
            )
            ids = decode_z_to_ids(z_final, model, device)
            texts_sc.extend(ids_to_text(ids.cpu(), elf_tok))
        ppl_sc = compute_ppl(texts_sc, ppl_model, ppl_tok, device)
        interaction = ppl_sc - ppl_none
        results[arm_key] = {
            "alpha": alpha,
            "ppl": ppl_sc,
            "interaction_I": interaction,
        }
        print(f"    PPL = {ppl_sc:.2f},  I(α={alpha:.2f}) = {interaction:+.2f}")
        # Sample texts (first 2)
        for t in texts_sc[:2]:
            print(f"      » {t[:100]!r}")

    hook.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt", default="all",
                        help="all | baseline | kd_cr | kd2")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_list = list(CHECKPOINTS.keys()) if args.ckpt == "all" else [args.ckpt]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load PPL model (once, reused across checkpoints)
    print(f"Loading PPL model ({PPL_MODEL_NAME})...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    ppl_tok  = AutoTokenizer.from_pretrained(PPL_MODEL_NAME)
    if ppl_tok.pad_token is None:
        ppl_tok.pad_token    = ppl_tok.eos_token
        ppl_tok.pad_token_id = ppl_tok.eos_token_id
    ppl_model = (AutoModelForCausalLM
                 .from_pretrained(PPL_MODEL_NAME, torch_dtype=torch.bfloat16)
                 .to(device).eval())

    # Load T5 tokenizer for decoding ELF token IDs
    print("Loading T5 tokenizer...")
    elf_tok = T5Tokenizer.from_pretrained("t5-small")

    all_results = {}
    for ckpt_name in ckpt_list:
        all_results[ckpt_name] = run_ckpt(ckpt_name, device, ppl_model, ppl_tok, elf_tok)

    # Save
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    print("\n=== SUMMARY: SC Interaction I(α) = PPL(SC_α) - PPL(none) ===")
    header = f"{'ckpt':<12}" + "".join(f"  α={a:.2f}" for a in ALPHAS)
    print(header)
    print("-" * len(header))
    for ckpt_name, res in all_results.items():
        row = f"{ckpt_name:<12}"
        for alpha in ALPHAS:
            key = f"sc_alpha_{alpha:.2f}"
            I = res.get(key, {}).get("interaction_I", float("nan"))
            row += f"  {I:+7.1f}"
        print(row)
    print(f"\nPositive I → SC hurts; Negative I → SC helps")
    print(f"EXP-36v2 reference (α=1.0): baseline I≈+1594, kd_cr I≈-65, kd2 I≈+158")


if __name__ == "__main__":
    main()
