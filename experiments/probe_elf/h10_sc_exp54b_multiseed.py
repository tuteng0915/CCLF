#!/usr/bin/env python3
"""
EXP-54b: Multi-seed variance validation for h₁₀ SC

Re-runs the three key arms from EXP-54 (natural sccfg=1, natural sccfg=3, h₁₀ SC sccfg=1)
across seeds {42, 123, 456} to obtain mean ± CI estimates.

Questions answered:
  1. Is "natural sccfg=3 slightly worse than sccfg=1" a stable finding or seed noise?
  2. What is the 95% CI on I(h₁₀, sccfg=1)?
  3. Is seed=42 EXP-54 h₁₀ PPL=155.4 representative of the true PPL?

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/h10_sc_exp54b_multiseed.py --device cuda:1
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import restore_cond, _ode_step, get_sampling_steps

CHECKPOINT = "converted/elf_b-owt-kd2_torch.pt"
SEEDS      = [42, 123, 456]
N_SEQ      = 256
N_STEPS    = 32
MAX_LENGTH = 128
BATCH_SIZE = 16
SC_T_MIN   = 0.5
OUT_DIR    = Path("results/exp54b_multiseed")

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
def generate(model, hook, z0, t_steps, arm, sccfg, device):
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


def ci95(values):
    """95% CI using t-distribution (n-1 df)."""
    n = len(values)
    if n < 2:
        return float("nan")
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std  = arr.std(ddof=1)
    # t critical value for 95% CI with n-1 df (n=3 → t=4.303)
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.0)
    margin = t_crit * std / math.sqrt(n)
    return mean, std, margin


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

    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    per_seed = {s: {} for s in SEEDS}   # seed → {arm: ppl}

    arms = [
        ("natural", 1.0, "natural_sccfg1"),
        ("natural", 3.0, "natural_sccfg3"),
        ("h10",     1.0, "h10_sccfg1"),
    ]

    for seed in SEEDS:
        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)

        for arm, sccfg, label in arms:
            texts = []
            for b in range(0, N_SEQ, BATCH_SIZE):
                z_f = generate(model, hook, all_z0[b:b+BATCH_SIZE], t_steps, arm, sccfg, device)
                ids = decode_z(z_f, model, device)
                texts.extend(ids_to_text(ids.cpu(), elf_tok))
            ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
            per_seed[seed][label] = ppl
            print(f"  {label:<30} PPL = {ppl:.2f}")

    hook.remove()

    # Aggregate across seeds
    print("\n\n=== AGGREGATED RESULTS ===")
    agg = {}
    for _, _, label in arms:
        vals = [per_seed[s][label] for s in SEEDS]
        mean, std, margin = ci95(vals)
        agg[label] = {"mean": mean, "std": std, "ci95_margin": margin, "per_seed": vals}
        print(f"  {label:<30} {mean:.1f} ± {margin:.1f}  (std={std:.1f}, seeds={vals})")

    # Derive I statistics
    ref_nat1  = agg["natural_sccfg1"]["mean"]
    ref_nat3  = agg["natural_sccfg3"]["mean"]
    h10_ppls  = [per_seed[s]["h10_sccfg1"] - per_seed[s]["natural_sccfg1"] for s in SEEDS]
    I_mean, I_std, I_margin = ci95(h10_ppls)
    print(f"\n  I(h10_sccfg1 - natural_sccfg1) = {I_mean:.1f} ± {I_margin:.1f}  (std={I_std:.1f})")
    print(f"  Delta(nat_sccfg3 - nat_sccfg1) = {ref_nat3 - ref_nat1:.1f}")

    results = {
        "seeds": SEEDS,
        "n_seq_per_seed": N_SEQ,
        "sc_t_min": SC_T_MIN,
        "per_seed": per_seed,
        "aggregated": agg,
        "I_h10_sccfg1": {"mean": I_mean, "std": I_std, "ci95_margin": I_margin},
        "delta_nat_sccfg3_vs_1": ref_nat3 - ref_nat1,
    }

    out = OUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
