#!/usr/bin/env python3
"""
EXP-54: Validate h₁₀ SC with standard self_cond_cfg_scale=3.0

EXP-48 showed kd2 PPL 247→91 (I=-157) with h_10 SC, but used sccfg=1.
The standard EXP-36v2 pipeline uses sccfg=3 → kd2 ODE-32 PPL=142.6.

This script repeats EXP-48 h₁₀ arm with sccfg=3 to check whether the
improvement survives the standard inference setting.

Arms (kd2 only):
  natural  : standard SC (x_pred from ODE step, sccfg=3) — reference
  h10      : x_pred replaced by final_layer(h_10) each step, sccfg=3
  sccfg1   : natural SC with sccfg=1 (reproduces EXP-48 natural for sanity check)

Reference baselines (from elf_b-owt-kd2-eval-pt-full):
  ODE-32 sccfg3 = 142.58 PPL  (952 samples, full eval)

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/h10_sc_sccfg_exp54.py --device cuda:1
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
    "kd2":   "converted/elf_b-owt-kd2_torch.pt",
}
N_SEQ      = 256    # More sequences for more stable PPL than EXP-48's 64
N_STEPS    = 32
SEED       = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
SC_T_MIN   = 0.5    # Apply h_10 SC only when t_next >= 0.5 (same as EXP-48)
OUT_DIR    = Path("results/exp54_h10_sc_sccfg")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)


# ── Config mock ────────────────────────────────────────────────────────────────

class _Cfg:
    t_eps = 0.05
    self_cond_prob = 1.0
    num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 1.0
    use_bf16 = True


# ── Hook ──────────────────────────────────────────────────────────────────────

class IntermediateSCHook:
    """Captures h_10 and h_11 during model forward passes."""

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

    def get_xhat_h10(self):
        """Return final_layer(h_10) — bypass block 11."""
        h10, h11 = self._h10_with_prefix, self._h11_no_prefix
        seq_len = h11.shape[1]
        h10_seq = h10[:, -seq_len:, :]    # strip prefix tokens
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            return self.model.final_layer(h10_seq.to(dev))

    def remove(self):
        self._h10_handle.remove()
        self._h11_handle.remove()


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, hook, z0, t_steps, arm, sccfg, device):
    """
    arm="natural" : standard SC, x_pred unchanged after each step
    arm="h10"     : x_pred replaced by final_layer(h_10) when t_next >= SC_T_MIN
    sccfg         : self_cond_cfg_scale (1.0 for EXP-48 match, 3.0 for standard)
    """
    cfg = _Cfg()
    B, L, D = z0.shape
    cond_seq  = torch.zeros_like(z0)
    cond_mask = torch.zeros(B, L, dtype=z0.dtype, device=device)
    step_kw = dict(
        model=model, config=cfg, cfg_scale=1.0, self_cond_cfg_scale=sccfg,
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

            if arm == "h10" and t_next >= SC_T_MIN:
                x_pred_h10 = hook.get_xhat_h10()
                x_pred = restore_cond(x_pred_h10.to(device), cond_seq, cond_mask)
            # arm == "natural": leave x_pred unchanged

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
    if not texts:
        return float("nan")
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


# ── Main ──────────────────────────────────────────────────────────────────────

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

    def run_arm(arm, sccfg, label):
        texts = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            z_f = generate(model, hook, all_z0[b:b+BATCH_SIZE], t_steps, arm, sccfg, device)
            ids = decode_z(z_f, model, device)
            texts.extend(ids_to_text(ids.cpu(), elf_tok))
        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        print(f"  {label:<35} PPL = {ppl:.2f}")
        return ppl

    # Sanity check: reproduce EXP-48 natural (sccfg=1)
    ppl_sccfg1 = run_arm("natural", 1.0, "natural sccfg=1 (EXP-48 sanity)")

    # Standard pipeline baseline (sccfg=3)
    ppl_natural3 = run_arm("natural", 3.0, "natural sccfg=3 (standard)")

    # h_10 SC with standard sccfg=3
    ppl_h10_3 = run_arm("h10", 3.0, "h_10 SC sccfg=3 (EXP-54 target)")

    # h_10 SC with sccfg=1 (reproduces EXP-48 alpha_0.00)
    ppl_h10_1 = run_arm("h10", 1.0, "h_10 SC sccfg=1 (EXP-48 α=0 sanity)")

    results[ckpt_name] = {
        "natural_sccfg1":  {"ppl": ppl_sccfg1,  "I": 0.0},
        "natural_sccfg3":  {"ppl": ppl_natural3, "I": 0.0},
        "h10_sccfg3":      {"ppl": ppl_h10_3,   "I": ppl_h10_3 - ppl_natural3},
        "h10_sccfg1":      {"ppl": ppl_h10_1,   "I": ppl_h10_1 - ppl_sccfg1},
        "ref_standard_ode32_sccfg3": 142.58,  # from elf_b-owt-kd2-eval-pt-full, 952 samples
    }

    print(f"\n  I(h_10, sccfg=3) = {ppl_h10_3 - ppl_natural3:+.2f}  "
          f"({'improvement' if ppl_h10_3 < ppl_natural3 else 'degradation'})")
    print(f"  I(h_10, sccfg=1) = {ppl_h10_1 - ppl_sccfg1:+.2f}  "
          f"(EXP-48 replicate, expected ≈ -157)")

    hook.remove()
    del model
    torch.cuda.empty_cache()
    return results[ckpt_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = torch.device(args.device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 Large PPL model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    ppl_tok = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tok.pad_token is None:
        ppl_tok.pad_token = ppl_tok.eos_token
        ppl_tok.pad_token_id = ppl_tok.eos_token_id
    ppl_model = (AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.bfloat16)
                 .to(device).eval())
    elf_tok = T5Tokenizer.from_pretrained("t5-small")

    results = {}
    for ck in CHECKPOINTS:
        results[ck] = run_ckpt(ck, device, ppl_model, ppl_tok, elf_tok)

    out = OUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")

    print("\n=== SUMMARY ===")
    for ck, r in results.items():
        print(f"\n{ck}:")
        print(f"  natural  sccfg=1 : {r['natural_sccfg1']['ppl']:.1f}  (EXP-48 sanity)")
        print(f"  natural  sccfg=3 : {r['natural_sccfg3']['ppl']:.1f}  (standard, ref=142.6)")
        print(f"  h10 SC   sccfg=3 : {r['h10_sccfg3']['ppl']:.1f}  I={r['h10_sccfg3']['I']:+.1f}")
        print(f"  h10 SC   sccfg=1 : {r['h10_sccfg1']['ppl']:.1f}  I={r['h10_sccfg1']['I']:+.1f}  (EXP-48 α=0 sanity)")


if __name__ == "__main__":
    main()
