#!/usr/bin/env python3
"""
EXP-48: Intermediate-Layer SC — Proper Pipeline

EXP-47 showed that lower α (more h_10, less h_11) gives better SC quality for
KD checkpoints, but had two bugs:
  1. compute_ppl returned mean NLL, not actual PPL (missing math.exp)
  2. "none" arm zeroed x_pred every step → incompatible with EXP-36v2 baseline

EXP-48 fixes both:
  1. compute_ppl returns math.exp(mean_NLL) = true PPL
  2. "none" arm: x_pred starts at zero but evolves naturally from _ode_step
     (standard ELF SC behavior — same as EXP-36v2 "none/SC=off" arm)
     For α arms: after each step, x_pred is REPLACED with final_layer(h_α)

Arms:
  natural (α=1.0)   → standard SC: x_pred from ODE step used unchanged (baseline)
  h10 (α=0.0)       → x_pred replaced by final_layer(h_10) every step
  mid (α=0.5)       → x_pred replaced by final_layer(h_10 + 0.5*(h_11-h_10))
  none              → x_pred zeroed every step (total SC off; for reference)

Metric: I(α) = PPL(α) - PPL(natural)
  negative → intermediate-layer SC is BETTER than standard SC
  positive → worse

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/intermediate_sc_proper_exp48.py --device cuda:1
    conda run -n elf python3 experiments/probe_elf/intermediate_sc_proper_exp48.py --device cuda:1 --ckpt kd2
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
from utils.sampling_utils import restore_cond, _ode_step, get_sampling_steps

CHECKPOINTS = {
    "kd_cr": "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":   "converted/elf_b-owt-kd2_torch.pt",
}
N_SEQ      = 64
N_STEPS    = 32
SEED       = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
SC_T_MIN   = 0.5       # only apply intermediate SC when t_next >= SC_T_MIN
OUT_DIR    = Path("results/exp48_intermediate_sc_proper")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)

ALPHAS = [0.0, 0.5]   # test α=0 (h10) and α=0.5; natural (α=1) and none run separately


# ── hook ──────────────────────────────────────────────────────────────────────

class IntermediateSCHook:
    """Captures h_10 (with prefix) and h_11 (without prefix) from forward passes."""

    def __init__(self, model):
        self.model = model
        self._h10_with_prefix = None
        self._h11_no_prefix   = None
        self._h10_handle = model.blocks[10].register_forward_hook(self._cap_h10)
        self._h11_handle = model.final_layer.register_forward_pre_hook(self._cap_h11)

    def _cap_h10(self, module, inp, out):
        self._h10_with_prefix = out.detach().float()

    def _cap_h11(self, module, inp):
        self._h11_no_prefix = inp[0].detach().float()

    def get_xhat_alpha(self, alpha):
        h10, h11 = self._h10_with_prefix, self._h11_no_prefix
        seq_len = h11.shape[1]
        h10_seq = h10[:, -seq_len:, :]
        h_alpha = h10_seq + alpha * (h11 - h10_seq)
        with torch.no_grad():
            return self.model.final_layer(h_alpha.to(next(self.model.parameters()).device))

    def remove(self):
        self._h10_handle.remove()
        self._h11_handle.remove()


# ── generation ────────────────────────────────────────────────────────────────

class _Cfg:
    t_eps = 0.05; self_cond_prob = 1.0; num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 1.0; use_bf16 = True


@torch.no_grad()
def generate(model, hook, z0, t_steps, arm, device):
    """
    arm="natural" : x_pred from ODE step used unchanged (standard SC)
    arm="none"    : x_pred zeroed every step (no SC)
    arm=float α   : x_pred replaced with final_layer(h_α) when t_next >= SC_T_MIN
    """
    cfg = _Cfg()
    B, L, D = z0.shape
    cond_seq  = torch.zeros_like(z0)
    cond_mask = torch.zeros(B, L, dtype=z0.dtype, device=device)
    step_kw = dict(
        model=model, config=cfg, cfg_scale=1.0, self_cond_cfg_scale=1.0,
        cond_seq=cond_seq, cond_seq_mask=cond_mask,
    )
    z      = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)

    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(t_steps.shape[0] - 2):
            t      = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kw)

            if arm == "none":
                x_pred = restore_cond(torch.zeros_like(x_pred), cond_seq, cond_mask)
            elif isinstance(arm, float) and t_next >= SC_T_MIN:
                x_pred_alpha = hook.get_xhat_alpha(arm)
                x_pred = restore_cond(x_pred_alpha.to(device), cond_seq, cond_mask)
            # arm == "natural": leave x_pred from ODE step unchanged

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
        _, logits, _ = model(z_in, t1, deterministic=True, self_cond_cfg_scale=sc,
                             decoder_step_active=True)
    return logits.argmax(dim=-1)


def ids_to_text(ids, tokenizer):
    eos = tokenizer.eos_token_id
    texts = []
    for row in ids.tolist():
        try: row = row[:row.index(eos)]
        except ValueError: pass
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def compute_ppl(texts, ppl_model, ppl_tok, device, max_length=256):
    if not texts: return float("nan")
    enc = ppl_tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, return_attention_mask=True)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    total_nll = total_tok = 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], 8):
            ids_b, msk_b = ids[i:i+8], mask[i:i+8]
            out = ppl_model(input_ids=ids_b, attention_mask=msk_b)
            logits = out.logits[:, :-1, :].float()
            tgts   = ids_b[:, 1:]
            msk_s  = msk_b[:, 1:].bool()
            nll = -(F.log_softmax(logits, dim=-1)
                    .gather(-1, tgts.unsqueeze(-1)).squeeze(-1)
                    * msk_s.float()).sum().item()
            total_nll += nll
            total_tok += msk_s.sum().item()
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


# ── main ──────────────────────────────────────────────────────────────────────

def run_ckpt(ckpt_name, device, ppl_model, ppl_tok, elf_tok):
    print(f"\n{'='*60}\nCheckpoint: {ckpt_name}\n{'='*60}")

    ckpt = torch.load(REPO_ROOT / CHECKPOINTS[ckpt_name], map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)
    hook = IntermediateSCHook(model)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    results = {}

    def run_arm(arm, arm_key):
        texts = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            z_f = generate(model, hook, all_z0[b:b+BATCH_SIZE], t_steps, arm, device)
            ids = decode_z(z_f, model, device)
            texts.extend(ids_to_text(ids.cpu(), elf_tok))
        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        print(f"  {arm_key:<25} PPL = {ppl:.2f}")
        return ppl

    # Reference arms
    ppl_natural = run_arm("natural", "natural (α=1.0, std SC)")
    ppl_none    = run_arm("none",    "none (zero SC)")

    results["natural"] = {"ppl": ppl_natural, "I": 0.0}  # reference
    results["none"]    = {"ppl": ppl_none,    "I": ppl_none - ppl_natural}

    # Intermediate α arms
    for alpha in ALPHAS:
        key = f"alpha_{alpha:.2f}"
        ppl_a = run_arm(alpha, f"α={alpha:.2f}")
        I = ppl_a - ppl_natural
        results[key] = {"ppl": ppl_a, "alpha": alpha, "I": I}
        print(f"    → I(α={alpha:.2f}) = {I:+.2f}  ({'better' if I<0 else 'worse'} than standard SC)")

    hook.remove()
    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--ckpt", default="all", help="all | kd_cr | kd2")
    args = parser.parse_args()
    device = torch.device(args.device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 Large PPL model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    ppl_tok   = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tok.pad_token is None:
        ppl_tok.pad_token = ppl_tok.eos_token
        ppl_tok.pad_token_id = ppl_tok.eos_token_id
    ppl_model = (AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.bfloat16)
                 .to(device).eval())
    elf_tok = T5Tokenizer.from_pretrained("t5-small")

    ckpt_list = list(CHECKPOINTS) if args.ckpt == "all" else [args.ckpt]
    all_results = {}
    for ck in ckpt_list:
        all_results[ck] = run_ckpt(ck, device, ppl_model, ppl_tok, elf_tok)

    out = OUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")

    print("\n=== SUMMARY: I(α) = PPL(α) - PPL(natural), negative = better ===")
    header = f"{'arm':<28}" + "".join(f"  {ck:<12}" for ck in ckpt_list)
    print(header); print("-" * len(header))
    all_keys = ["none"] + [f"alpha_{a:.2f}" for a in ALPHAS]
    for key in all_keys:
        row = f"{key:<28}"
        for ck in ckpt_list:
            I = all_results.get(ck, {}).get(key, {}).get("I", float("nan"))
            row += f"  {I:>+12.2f}"
        print(row)


if __name__ == "__main__":
    main()
