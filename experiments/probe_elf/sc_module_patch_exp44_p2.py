#!/usr/bin/env python3
"""
EXP-44 Phase 2: SC Module Patching — SC Interaction Experiment

Phase 1 showed oracle acc and L_rec are nearly identical between kd_cr and kd2
(L_rec 76.2 vs 76.4), ruling out x̂_t quality as the source of the SC interaction
gap (kd_cr I≈-65 helpful vs kd2 I≈+158 harmful, from EXP-36v2).

Phase 2 directly tests the SC conditioning module by swapping self_cond_proj
(and optionally the full SC module) between checkpoints during actual generation,
then measuring whether SC interaction I flips.

Arms per base checkpoint:
  none          → SC=False (no SC conditioning; establishes PPL baseline)
  native_sc     → SC=True, native SC module (should reproduce EXP-36v2 I direction)
  proj_swap     → SC=True, cross-checkpoint self_cond_proj only
  full_sc_swap  → SC=True, full SC module swapped (proj + cfg_embedder + cfg_tokens)

If proj_swap or full_sc_swap flips the sign of I, the SC module is causal.
If neither flips → the x̂_t direction in embedding space is more likely causal (→ EXP-45).

SC_PROJ_KEYS: ["self_cond_proj.weight", "self_cond_proj.bias"]
SC_FULL_KEYS: above + cfg_embedder MLP weights + cfg_tokens

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/sc_module_patch_exp44_p2.py --device cuda:1
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import restore_cond, _ode_step, get_sampling_steps

# ── config ─────────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "kd_cr": "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":   "converted/elf_b-owt-kd2_torch.pt",
}
N_SEQ       = 64
N_STEPS     = 32
SEED        = 42
MAX_LENGTH  = 128
BATCH_SIZE  = 16
SC_T_MIN    = 0.5
OUT_DIR     = Path("results/exp44_module_patch")

MODEL_CFG = dict(
    text_encoder_dim=512,
    max_length=MAX_LENGTH,
    num_time_tokens=4,
    num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4,
    vocab_size=32100,
    bottleneck_dim=128,
)

SC_PROJ_KEYS = ["self_cond_proj.weight", "self_cond_proj.bias"]
SC_FULL_KEYS = SC_PROJ_KEYS + [
    "self_cond_cfg_embedder.mlp_0.weight", "self_cond_cfg_embedder.mlp_0.bias",
    "self_cond_cfg_embedder.mlp_2.weight", "self_cond_cfg_embedder.mlp_2.bias",
    "self_cond_cfg_tokens",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_model(ckpt_name: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(REPO_ROOT / CHECKPOINTS[ckpt_name], map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)
    return model


def load_params(ckpt_name: str) -> dict:
    ckpt = torch.load(REPO_ROOT / CHECKPOINTS[ckpt_name], map_location="cpu", weights_only=False)
    return ckpt["params"]


def patch_model(model: torch.nn.Module, params_donor: dict, keys: list, device: torch.device):
    """Copy specified params from params_donor into model in-place."""
    sd = model.state_dict()
    for k in keys:
        sd[k] = params_donor[k].clone().to(device=device, dtype=sd[k].dtype)
    model.load_state_dict(sd)


def restore_model(model: torch.nn.Module, params_native: dict, keys: list, device: torch.device):
    """Restore model parameters to native values."""
    patch_model(model, params_native, keys, device)


class _Config:
    def __init__(self):
        self.t_eps = 0.05
        self.self_cond_prob = 1.0
        self.num_self_cond_cfg_tokens = 4
        self.denoiser_noise_scale = 1.0
        self.use_bf16 = True


@torch.no_grad()
def generate(model, z0, t_steps, use_sc, device):
    """Run ODE generation with or without SC conditioning."""
    cfg = _Config()
    n = t_steps.shape[0]
    B, L, D = z0.shape
    cond_seq = torch.zeros_like(z0)
    cond_mask = torch.zeros(B, L, dtype=z0.dtype, device=device)

    step_kw = dict(
        model=model, config=cfg, cfg_scale=1.0, self_cond_cfg_scale=1.0,
        cond_seq=cond_seq, cond_seq_mask=cond_mask,
    )
    z = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)

    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 2):
            t      = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kw)
            # Zero SC signal if disabled or below gate
            if not use_sc or t_next < SC_T_MIN:
                x_pred = restore_cond(torch.zeros_like(x_pred), cond_seq, cond_mask)
        t      = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z, _ = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kw)
    return z


@torch.no_grad()
def decode_z(z, model, device):
    B = z.shape[0]
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
    t1 = torch.ones(B, dtype=z.dtype, device=device)
    sc = torch.ones(B, dtype=z.dtype, device=device)
    use_bf16 = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, logits, _ = model(z_in, t1, deterministic=True,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
    return logits.argmax(dim=-1)


def ids_to_text(ids, tokenizer):
    eos = tokenizer.eos_token_id
    texts = []
    for row in ids.tolist():
        try:
            row = row[:row.index(eos)]
        except ValueError:
            pass
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def compute_ppl(texts, ppl_model, ppl_tok, device, max_length=256):
    if not texts:
        return float("nan")
    enc = ppl_tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, return_attention_mask=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], 8):
            ids_b = input_ids[i:i+8]
            msk_b = attn_mask[i:i+8]
            out = ppl_model(input_ids=ids_b, attention_mask=msk_b)
            logits = out.logits[:, :-1, :].float()
            tgts = ids_b[:, 1:]
            mask_s = msk_b[:, 1:].bool()
            log_p = F.log_softmax(logits, dim=-1)
            nll = -(log_p.gather(-1, tgts.unsqueeze(-1)).squeeze(-1) * mask_s.float()).sum().item()
            total_nll += nll
            total_tok += mask_s.sum().item()
    if total_tok == 0:
        return float("nan")
    import math
    return math.exp(total_nll / total_tok)


def run_arm(model, all_z0, t_steps, use_sc, device, ppl_model, ppl_tok, elf_tok, arm_name):
    texts = []
    for b in range(0, N_SEQ, BATCH_SIZE):
        z0_b = all_z0[b:b+BATCH_SIZE]
        z_f = generate(model, z0_b, t_steps, use_sc, device)
        ids = decode_z(z_f, model, device)
        texts.extend(ids_to_text(ids.cpu(), elf_tok))
    ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
    print(f"  {arm_name:<30} PPL = {ppl:.2f}")
    return ppl


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = torch.device(args.device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PPL model (gpt2-large)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    ppl_tok = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tok.pad_token is None:
        ppl_tok.pad_token = ppl_tok.eos_token
        ppl_tok.pad_token_id = ppl_tok.eos_token_id
    ppl_model = (AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.bfloat16)
                 .to(device).eval())
    elf_tok = T5Tokenizer.from_pretrained("t5-small")

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    params_cr  = load_params("kd_cr")
    params_kd2 = load_params("kd2")

    all_results = {}

    for base_ck, donor_ck, base_params, donor_params in [
        ("kd_cr", "kd2",   params_cr,  params_kd2),
        ("kd2",   "kd_cr", params_kd2, params_cr),
    ]:
        print(f"\n{'='*60}")
        print(f"Base: {base_ck}  |  Donor: {donor_ck}")
        print(f"{'='*60}")

        model = load_model(base_ck, device)

        # arm: none (SC=False)
        ppl_none = run_arm(model, all_z0, t_steps, use_sc=False, device=device,
                           ppl_model=ppl_model, ppl_tok=ppl_tok, elf_tok=elf_tok,
                           arm_name="none")

        # arm: native_sc (SC=True, no swap)
        ppl_native = run_arm(model, all_z0, t_steps, use_sc=True, device=device,
                             ppl_model=ppl_model, ppl_tok=ppl_tok, elf_tok=elf_tok,
                             arm_name="native_sc")

        # arm: proj_swap (swap self_cond_proj only)
        patch_model(model, donor_params, SC_PROJ_KEYS, device)
        ppl_proj_swap = run_arm(model, all_z0, t_steps, use_sc=True, device=device,
                                ppl_model=ppl_model, ppl_tok=ppl_tok, elf_tok=elf_tok,
                                arm_name="proj_swap")
        restore_model(model, base_params, SC_PROJ_KEYS, device)

        # arm: full_sc_swap (swap full SC module)
        patch_model(model, donor_params, SC_FULL_KEYS, device)
        ppl_full_swap = run_arm(model, all_z0, t_steps, use_sc=True, device=device,
                                ppl_model=ppl_model, ppl_tok=ppl_tok, elf_tok=elf_tok,
                                arm_name="full_sc_swap")
        restore_model(model, base_params, SC_FULL_KEYS, device)

        I_native    = ppl_native    - ppl_none
        I_proj_swap = ppl_proj_swap - ppl_none
        I_full_swap = ppl_full_swap - ppl_none

        print(f"\n  PPL_none    = {ppl_none:.2f}")
        print(f"  I_native    = {I_native:+.2f}  (reference)")
        print(f"  I_proj_swap = {I_proj_swap:+.2f}  (self_cond_proj from {donor_ck})")
        print(f"  I_full_swap = {I_full_swap:+.2f}  (full SC module from {donor_ck})")

        all_results[base_ck] = {
            "ppl_none":          ppl_none,
            "ppl_native_sc":     ppl_native,
            "ppl_proj_swap":     ppl_proj_swap,
            "ppl_full_sc_swap":  ppl_full_swap,
            "I_native":          I_native,
            "I_proj_swap":       I_proj_swap,
            "I_full_sc_swap":    I_full_swap,
            "donor_ckpt":        donor_ck,
        }

        del model
        torch.cuda.empty_cache()

    out_path = OUT_DIR / "phase2_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== SUMMARY ===")
    print(f"{'arm':<35}  {'kd_cr base':>12}  {'kd2 base':>12}")
    print("-" * 62)
    for key, label in [
        ("I_native",       "I_native (SC=True)"),
        ("I_proj_swap",    "I_proj_swap"),
        ("I_full_sc_swap", "I_full_sc_swap"),
    ]:
        cr_val  = all_results["kd_cr"][key]
        kd2_val = all_results["kd2"][key]
        print(f"{label:<35}  {cr_val:>+12.2f}  {kd2_val:>+12.2f}")
    print(f"\nNegative I → SC helps; Positive I → SC hurts")
    print(f"EXP-36v2 reference: kd_cr I≈-65, kd2 I≈+158")


if __name__ == "__main__":
    main()
