#!/usr/bin/env python3
"""
EXP-51: Evaluate SC interaction for D1 and D3 fine-tuned checkpoints.

Uses the same intermediate-layer SC pipeline as EXP-48:
  - natural (standard h_11 SC) as baseline
  - h_10 (α=0.0) as the intermediate-layer arm
  - none (zero SC) for reference

Compares: kd_cr (baseline), D1 checkpoint, D3 checkpoint
We want D1 or D3 to show improved I(α=0.0) relative to kd_cr.
kd_cr reference from EXP-48: PPL_natural=303.4, PPL_h10=192.1, I(h10)=-111.3

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/eval_finetuned_sc_exp51.py --device cuda:1
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
    "kd_cr":  "converted/elf_b-owt-kd-cr_torch.pt",
    "d1":     "results/finetune_quick/d1.pt",
    "d3":     "results/finetune_quick/d3.pt",
}
N_SEQ      = 64
N_STEPS    = 32
SEED       = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
SC_T_MIN   = 0.5
OUT_DIR    = Path("results/exp51_eval_finetuned_sc")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)


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


class _Cfg:
    t_eps = 0.05; self_cond_prob = 1.0; num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 1.0; use_bf16 = True


@torch.no_grad()
def generate(model, hook, z0, t_steps, arm, device):
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


def run_ckpt(ckpt_name, ckpt_path, device, ppl_model, ppl_tok, elf_tok, all_z0, t_steps):
    print(f"\n{'='*60}\nCheckpoint: {ckpt_name}\n{'='*60}")

    ckpt = torch.load(REPO_ROOT / ckpt_path, map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)
    hook = IntermediateSCHook(model)

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

    ppl_natural = run_arm("natural", "natural (h11, std SC)")
    ppl_none    = run_arm("none",    "none (zero SC)")
    ppl_h10     = run_arm(0.0,       "h10 (α=0.0)")

    results["natural"] = {"ppl": ppl_natural, "I": 0.0}
    results["none"]    = {"ppl": ppl_none,    "I": ppl_none - ppl_natural}
    results["alpha_0.00"] = {"ppl": ppl_h10, "I": ppl_h10 - ppl_natural}

    I_h10 = ppl_h10 - ppl_natural
    print(f"  I(h10) = {I_h10:+.2f}  ({'better' if I_h10 < 0 else 'worse'} than std SC)")

    hook.remove()
    del model
    torch.cuda.empty_cache()
    return results


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

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    all_results = {}
    for name, path in CHECKPOINTS.items():
        all_results[name] = run_ckpt(name, path, device, ppl_model, ppl_tok, elf_tok,
                                     all_z0, t_steps)

    out = OUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")

    print("\n=== SUMMARY: PPL by arm and checkpoint ===")
    ckpts = list(CHECKPOINTS)
    print(f"{'arm':<20}" + "".join(f"  {c:<12}" for c in ckpts))
    print("-" * (20 + 14 * len(ckpts)))
    for arm in ["natural", "none", "alpha_0.00"]:
        row = f"{arm:<20}"
        for c in ckpts:
            ppl = all_results.get(c, {}).get(arm, {}).get("ppl", float("nan"))
            row += f"  {ppl:>12.2f}"
        print(row)
    print()
    print(f"{'I(h10)':<20}" + "".join(f"  {c:<12}" for c in ckpts))
    print("-" * (20 + 14 * len(ckpts)))
    row = " " * 20
    for c in ckpts:
        I = all_results.get(c, {}).get("alpha_0.00", {}).get("I", float("nan"))
        row += f"  {I:>+12.2f}"
    print(row)


if __name__ == "__main__":
    main()
