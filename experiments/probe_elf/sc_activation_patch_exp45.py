#!/usr/bin/env python3
"""
EXP-45: SC Activation Patch

Causal test: during kd2 generation, replace the SC conditioning input (x̂_t_prev)
with kd_cr's prediction from a parallel rollout on the same z trajectory.

Arms:
  kd2_native   λ=0   kd2 backbone, kd2 own x̂_t as SC  (reference)
  kd_cr_native  —    kd_cr backbone, kd_cr own SC       (upper bound)
  patch_lam05  λ=0.5 kd2 backbone, 50/50 mix SC
  patch_lam10  λ=1.0 kd2 backbone, kd_cr x̂_t as SC     (full patch)
  zeros_sc      —    kd2 backbone, zeros SC throughout   (no-SC ablation)

Decision rule (from spec):
  patch_lam10 flips kd2 I from positive → negative  ↔  x̂_t format determines SC utility
  patch_lam10 unchanged (still positive)             ↔  SC diff comes from backbone itself

N=64 seqs, 32 steps, 128 tokens, seed=42
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import net_out_to_v_x, get_sampling_steps

CHECKPOINTS = {
    "kd_cr":    "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":      "converted/elf_b-owt-kd2_torch.pt",
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
}
N_SEQ      = 64
N_STEPS    = 32
SEED       = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
T_EPS      = 0.05
OUT_DIR    = Path("results/exp45_sc_activation_patch")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)


def load_model(ckpt_name, device):
    ckpt = torch.load(REPO_ROOT / CHECKPOINTS[ckpt_name], map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    return model.eval().to(device)


@torch.no_grad()
def _forward_one(model, z, t_batch, sc_input, device):
    """Single ELF forward pass with explicit SC input.

    sc_input: (B, L, 512) tensor used as x̂_t_prev (the SC conditioning signal).
    Returns (v_pred, x_pred).
    """
    sc_scale = torch.ones(z.shape[0], dtype=z.dtype, device=device)
    z_input  = torch.cat([z, sc_input], dim=-1)       # (B, L, 1024)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        net_out  = model(z_input, t_batch, deterministic=True, self_cond_cfg_scale=sc_scale)
    v, x = net_out_to_v_x(net_out, z, t_batch, T_EPS)
    return v, x


@torch.no_grad()
def generate_patched(model_kd2, model_kd_cr, z0, t_steps, lam, device):
    """ODE-32 with kd2 backbone; SC input patched by kd_cr's x̂_t (weight λ).

    λ=0 → kd2 native SC
    λ=1 → kd_cr x̂_t as SC input for kd2
    model_kd_cr may be None when λ=0 or zeros_sc mode.
    """
    B, L, D = z0.shape
    z        = z0.clone()
    xp_kd2   = torch.zeros(B, L, D, dtype=z.dtype, device=device)
    xp_kd_cr = torch.zeros(B, L, D, dtype=z.dtype, device=device)

    all_steps = list(zip(t_steps[:-2].tolist(), t_steps[1:-1].tolist())) + \
                [(t_steps[-2].item(), t_steps[-1].item())]

    for t, t_next in all_steps:
        t_batch = torch.full((B,), t, dtype=z.dtype, device=device)

        # Build patched SC input for kd2
        xp_sc = (1.0 - lam) * xp_kd2 + lam * xp_kd_cr

        # kd2 forward with patched SC
        v_kd2, xp_kd2_new = _forward_one(model_kd2, z, t_batch, xp_sc, device)

        # kd_cr forward on same z (only needed when λ > 0)
        if lam > 0 and model_kd_cr is not None:
            _, xp_kd_cr_new = _forward_one(model_kd_cr, z, t_batch, xp_kd_cr, device)
        else:
            xp_kd_cr_new = xp_kd_cr

        z        = z + (t_next - t) * v_kd2
        xp_kd2   = xp_kd2_new
        xp_kd_cr = xp_kd_cr_new

    return z


@torch.no_grad()
def generate_zeros_sc(model_kd2, z0, t_steps, device):
    """kd2 with zeros as SC input throughout (no SC signal)."""
    B, L, D = z0.shape
    z = z0.clone()
    zeros_sc = torch.zeros(B, L, D, dtype=z.dtype, device=device)

    all_steps = list(zip(t_steps[:-2].tolist(), t_steps[1:-1].tolist())) + \
                [(t_steps[-2].item(), t_steps[-1].item())]

    for t, t_next in all_steps:
        t_batch = torch.full((B,), t, dtype=z.dtype, device=device)
        v, _ = _forward_one(model_kd2, z, t_batch, zeros_sc, device)
        z = z + (t_next - t) * v

    return z


@torch.no_grad()
def generate_native(model, z0, t_steps, device):
    """Native SC rollout for a single model (kd_cr or any checkpoint)."""
    B, L, D = z0.shape
    z    = z0.clone()
    xp   = torch.zeros(B, L, D, dtype=z.dtype, device=device)

    all_steps = list(zip(t_steps[:-2].tolist(), t_steps[1:-1].tolist())) + \
                [(t_steps[-2].item(), t_steps[-1].item())]

    for t, t_next in all_steps:
        t_batch = torch.full((B,), t, dtype=z.dtype, device=device)
        v, xp = _forward_one(model, z, t_batch, xp, device)
        z = z + (t_next - t) * v

    return z


@torch.no_grad()
def decode_z(z, model, device):
    B = z.shape[0]
    zeros = torch.zeros_like(z)
    z_in  = torch.cat([z, zeros], dim=-1)
    t1    = torch.ones(B, dtype=z.dtype, device=device)
    sc    = torch.ones(B, dtype=z.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
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


def text_quality(texts):
    """Fraction non-ASCII and fraction repetitive (>35% single word)."""
    non_ascii, rep = 0, 0
    for t in texts:
        chars = list(t)
        if chars and sum(1 for c in chars if ord(c) > 127) / len(chars) > 0.02:
            non_ascii += 1
        words = t.split()
        if words:
            mc = max(set(words), key=words.count)
            if words.count(mc) / len(words) > 0.35:
                rep += 1
    n = len(texts)
    return non_ascii / n if n else 0.0, rep / n if n else 0.0


def compute_ppl(texts, ppl_model, ppl_tok, device, max_length=256):
    if not texts:
        return float("nan")
    enc = ppl_tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, return_attention_mask=True)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    total_nll = total_tok = 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], 8):
            ib, mb = ids[i:i+8], mask[i:i+8]
            out = ppl_model(input_ids=ib, attention_mask=mb)
            logits = out.logits[:, :-1, :].float()
            tgts   = ib[:, 1:]
            ms     = mb[:, 1:].bool()
            nll = -(F.log_softmax(logits, dim=-1)
                    .gather(-1, tgts.unsqueeze(-1)).squeeze(-1)
                    * ms.float()).sum().item()
            total_nll += nll
            total_tok += ms.sum().item()
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


def run_experiment(device, ppl_model, ppl_tok, elf_tok):
    print(f"\n{'='*65}")
    print(f"EXP-45: SC Activation Patch  (N={N_SEQ}, steps={N_STEPS}, seed={SEED})")
    print(f"{'='*65}")

    print("Loading models...")
    model_kd2   = load_model("kd2",   device)
    model_kd_cr = load_model("kd_cr", device)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0  = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    arm_defs = [
        ("kd2_native",    "kd2 with own SC (λ=0)",               0.0,  False),
        ("patch_lam05",   "kd2 backbone, 50% kd_cr x̂_t SC",      0.5,  False),
        ("patch_lam10",   "kd2 backbone, 100% kd_cr x̂_t SC",     1.0,  False),
        ("zeros_sc",      "kd2 backbone, zeros SC (no SC)",       None, True),
        ("kd_cr_native",  "kd_cr with own SC",                   None, False),
    ]

    results = {}
    ppl_ref  = None

    for arm_name, arm_desc, lam, is_zeros in arm_defs:
        texts = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            z0_b = all_z0[b:b+BATCH_SIZE]

            if arm_name == "kd_cr_native":
                z_out = generate_native(model_kd_cr, z0_b, t_steps, device)
                ids   = decode_z(z_out, model_kd_cr, device)
            elif is_zeros:
                z_out = generate_zeros_sc(model_kd2, z0_b, t_steps, device)
                ids   = decode_z(z_out, model_kd2, device)
            else:
                z_out = generate_patched(model_kd2, model_kd_cr, z0_b, t_steps, lam, device)
                ids   = decode_z(z_out, model_kd2, device)

            texts.extend(ids_to_text(ids.cpu(), elf_tok))

        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        na, rep = text_quality(texts)
        if arm_name == "kd2_native":
            ppl_ref = ppl
        I = ppl - ppl_ref if ppl_ref is not None else 0.0
        flag = " ← KEY" if arm_name == "patch_lam10" else ""
        print(f"  {arm_name:<20} PPL={ppl:7.2f}  I={I:+8.2f}"
              f"  non-ASCII={na:.1%}  rep={rep:.1%}{flag}")
        print(f"    [{arm_desc}]")
        results[arm_name] = {
            "ppl": ppl, "I": I,
            "non_ascii": na, "rep": rep,
            "desc": arm_desc, "lam": lam,
        }

    print(f"\n{'='*65}")
    print("VERDICT:")
    i_kd2  = results["kd2_native"]["I"]      # always 0
    i_kd_cr = results.get("kd_cr_native", {}).get("I", float("nan"))
    i_patch = results.get("patch_lam10", {}).get("I", float("nan"))
    i_zeros = results.get("zeros_sc", {}).get("I", float("nan"))
    print(f"  kd2 native I   = {i_kd2:+.2f}  (reference)")
    print(f"  kd_cr native I = {i_kd_cr:+.2f}  (upper bound)")
    print(f"  zeros SC I     = {i_zeros:+.2f}  (no SC signal)")
    print(f"  patch λ=1 I    = {i_patch:+.2f}  ← does kd_cr x̂_t flip the sign?")
    if i_patch < 0:
        print("  >> FLIPPED: x̂_t format IS the key SC compatibility factor")
    elif i_patch < i_kd2 + (i_kd_cr - i_kd2) * 0.5:
        print("  >> PARTIAL: x̂_t format contributes but is not the sole factor")
    else:
        print("  >> NO FLIP: SC compatibility comes from backbone, not x̂_t format")
    print(f"{'='*65}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args   = parser.parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 Large PPL model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    ppl_tok = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tok.pad_token is None:
        ppl_tok.pad_token    = ppl_tok.eos_token
        ppl_tok.pad_token_id = ppl_tok.eos_token_id
    ppl_model = (AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.bfloat16)
                 .to(device).eval())
    elf_tok = T5Tokenizer.from_pretrained("t5-small")

    results = run_experiment(device, ppl_model, ppl_tok, elf_tok)

    out_path = OUT_DIR / "patch_results.json"
    with open(out_path, "w") as f:
        json.dump({"config": {"N_SEQ": N_SEQ, "N_STEPS": N_STEPS,
                               "SEED": SEED, "MAX_LENGTH": MAX_LENGTH},
                   "results": results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
