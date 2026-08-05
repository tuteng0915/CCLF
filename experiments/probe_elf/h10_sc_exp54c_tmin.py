#!/usr/bin/env python3
"""
EXP-54c: SC_T_MIN sweep for h₁₀ SC

EXP-48 and EXP-54 both use SC_T_MIN=0.5, meaning h₁₀ SC is applied only
when t_next >= 0.5 (first half of the [0,1] trajectory). This gate was
inherited without validation.

This script sweeps SC_T_MIN ∈ {0.0, 0.1, 0.25, 0.5} to test:
  - Does applying h₁₀ SC from t=0 (SC_T_MIN=0.0) improve further?
  - Is there an optimal threshold, or does full-range SC always win?

Arms:
  natural  (sccfg=1): reference, SC_T_MIN has no effect
  h₁₀ SC  (sccfg=1): t_min ∈ {0.0, 0.1, 0.25, 0.5}

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/h10_sc_exp54c_tmin.py --device cuda:1
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

CHECKPOINT   = "converted/elf_b-owt-kd2_torch.pt"
TMIN_VALUES  = [0.0, 0.1, 0.25, 0.5]
N_SEQ        = 256
N_STEPS      = 32
SEED         = 42
MAX_LENGTH   = 128
BATCH_SIZE   = 16
OUT_DIR      = Path("results/exp54c_tmin_sweep")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)


class _Cfg:
    t_eps = 0.05
    self_cond_prob = 1.0
    num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 1.0
    use_bf16 = True


class IntermediateSCHook:
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
        h10, h11 = self._h10_with_prefix, self._h11_no_prefix
        seq_len = h11.shape[1]
        h10_seq = h10[:, -seq_len:, :]
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            return self.model.final_layer(h10_seq.to(dev))

    def remove(self):
        self._h10_handle.remove()
        self._h11_handle.remove()


@torch.no_grad()
def generate(model, hook, z0, t_steps, arm, sccfg, sc_t_min, device):
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
            if arm == "h10" and t_next >= sc_t_min:
                x_pred_h10 = hook.get_xhat_h10()
                x_pred = restore_cond(x_pred_h10.to(device), cond_seq, cond_mask)

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


def run_arm(arm_label, texts_list, ppl_model, ppl_tok, device):
    return compute_ppl(texts_list, ppl_model, ppl_tok, device)


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
        ppl_tok.pad_token    = ppl_tok.eos_token
        ppl_tok.pad_token_id = ppl_tok.eos_token_id
    ppl_model = (AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.bfloat16)
                 .to(device).eval())
    elf_tok = T5Tokenizer.from_pretrained("t5-small")

    print("Loading kd2 model...")
    ckpt  = torch.load(REPO_ROOT / CHECKPOINT, map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)
    hook = IntermediateSCHook(model)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    results = {"seed": SEED, "n_seq": N_SEQ, "arms": {}}

    # Reference: natural SC (SC_T_MIN irrelevant)
    print("\n--- natural sccfg=1 (reference) ---")
    texts_nat = []
    for b in range(0, N_SEQ, BATCH_SIZE):
        z_f = generate(model, hook, all_z0[b:b+BATCH_SIZE], t_steps, "natural", 1.0, 0.0, device)
        ids = decode_z(z_f, model, device)
        texts_nat.extend(ids_to_text(ids.cpu(), elf_tok))
    ppl_nat = compute_ppl(texts_nat, ppl_model, ppl_tok, device)
    results["arms"]["natural_sccfg1"] = {"ppl": ppl_nat, "sc_t_min": None}
    print(f"  natural sccfg=1  PPL = {ppl_nat:.2f}  (reference)")

    # h₁₀ SC sweep over SC_T_MIN values
    for tmin in TMIN_VALUES:
        label = f"h10_tmin{tmin:.2f}"
        print(f"\n--- h₁₀ SC sccfg=1, SC_T_MIN={tmin} ---")
        texts = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            z_f = generate(model, hook, all_z0[b:b+BATCH_SIZE], t_steps, "h10", 1.0, tmin, device)
            ids = decode_z(z_f, model, device)
            texts.extend(ids_to_text(ids.cpu(), elf_tok))
        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        I   = ppl - ppl_nat
        results["arms"][label] = {"ppl": ppl, "I": I, "sc_t_min": tmin}
        # Count how many ODE steps use h₁₀ SC (t_next >= tmin, t_next runs from ~1 to ~0)
        n_sc_steps = sum(1 for i in range(t_steps.shape[0] - 2)
                         if t_steps[i+1].item() >= tmin)
        print(f"  h₁₀ SC tmin={tmin}  PPL = {ppl:.2f}  I = {I:+.2f}  "
              f"(SC applied at {n_sc_steps}/{N_STEPS-1} steps)")

    hook.remove()

    print("\n\n=== SUMMARY ===")
    print(f"  natural_sccfg1           PPL = {results['arms']['natural_sccfg1']['ppl']:.1f}  (ref)")
    for tmin in TMIN_VALUES:
        label = f"h10_tmin{tmin:.2f}"
        r = results["arms"][label]
        print(f"  h10 SC_T_MIN={tmin:<5}      PPL = {r['ppl']:.1f}  I = {r['I']:+.1f}")

    out = OUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
