#!/usr/bin/env python3
"""
EXP-59: Multi-seed validation of kd_cr pipeline_avg (EXP-58 follow-up)

EXP-58 found kd_cr pipeline_avg: I=−142, D1=0.310↑, D2=0.809↑ (single seed).
This script replicates with 3 seeds to get variance and 95% CI.

Arms: standard (ODE-32), pipeline_avg
Model: kd_cr only
Seeds: configurable via --seed (run once per seed, pool externally)
N=256, MAX_LENGTH=128, T_PIPE=16, SCCFG=1.0

Usage:
    cd models/ELF-torch
    for seed in 42 123 456; do
      conda run -n elf python3 experiments/probe_elf/pipeline_avg_multiseed_exp59.py \
          --device cuda:2 --seed $seed &
    done
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
from utils.sampling_utils import _ode_step, get_sampling_steps

CHECKPOINT = "converted/elf_b-owt-kd-cr_torch.pt"
N_SEQ      = 256
N_STEPS    = 32
T_PIPE     = 16
MAX_LENGTH = 128
BATCH_SIZE = 16
SCCFG      = 1.0
OUT_DIR    = Path("results/exp59_pipeline_avg_multiseed")

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


def compute_distinct(texts, n):
    all_ng, uniq_ng = [], set()
    for text in texts:
        toks = text.split()
        ngs = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
        all_ng.extend(ngs)
        uniq_ng.update(ngs)
    return len(uniq_ng) / len(all_ng) if all_ng else 0.0


def compute_ppl(texts, ppl_model, ppl_tok, device, max_length=200):
    if not texts:
        return float("nan")
    enc = ppl_tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, return_attention_mask=True)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    total_nll = total_tok = 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], 8):
            ids_b, msk_b = ids[i:i+8], mask[i:i+8]
            logits = ppl_model(input_ids=ids_b, attention_mask=msk_b).logits[:, :-1, :].float()
            tgts  = ids_b[:, 1:]
            msk_s = msk_b[:, 1:].bool()
            nll = -(F.log_softmax(logits, dim=-1)
                    .gather(-1, tgts.unsqueeze(-1)).squeeze(-1)
                    * msk_s.float()).sum().item()
            total_nll += nll
            total_tok  += msk_s.sum().item()
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


@torch.no_grad()
def decode_z(z, model, device):
    B = z.shape[0]
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
    t1 = torch.ones(B, dtype=z.dtype, device=device)
    sc = torch.ones(B, dtype=z.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(z_in, t1, deterministic=True,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
    return logits.argmax(dim=-1)


@torch.no_grad()
def run_standard(z0, model, t_steps, device, sccfg=SCCFG):
    z = z0.clone()
    xp = torch.zeros_like(z)
    cfg = _Cfg()
    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(t_steps.shape[0] - 1):
            z, xp = _ode_step(z=z, t=t_steps[i].item(), t_next=t_steps[i+1].item(),
                              x_pred_prev=xp, model=model, config=cfg,
                              cfg_scale=1.0, self_cond_cfg_scale=sccfg,
                              cond_seq=None, cond_seq_mask=None)
    return z


@torch.no_grad()
def run_pipeline_avg(z0, model, T, device, sccfg=SCCFG):
    cfg = _Cfg()
    B, N, D = z0.shape
    CHUNK = max(1, N // T)
    TOTAL = 2 * T - 1

    z   = z0.clone()
    xp  = torch.zeros_like(z)
    pos_idx  = torch.arange(N, device=device)
    group_of = torch.clamp(pos_idx // CHUNK, 0, T - 1)

    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for k in range(TOTAL):
            j_min = max(0, k - T + 1)
            j_max = min(T - 1, k)
            t      = ((k - j_max) + (k - j_min)) / (2.0 * T)
            t_next = min(t + 1.0 / T, 1.0)

            z_full, x_pred = _ode_step(
                z=z, t=t, t_next=t_next, x_pred_prev=xp,
                model=model, config=cfg,
                cfg_scale=1.0, self_cond_cfg_scale=sccfg,
                cond_seq=None, cond_seq_mask=None,
            )

            active    = (group_of >= j_min) & (group_of <= j_max)
            active_3d = active.unsqueeze(0).unsqueeze(-1).expand(B, N, D)
            z  = torch.where(active_3d, z_full,  z)
            xp = torch.where(active_3d, x_pred,  xp)
    return z


def ids_to_texts(ids, tokenizer):
    eos = tokenizer.eos_token_id
    texts = []
    for row in ids.tolist():
        try: row = row[:row.index(eos)]
        except ValueError: pass
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    print(f"Seed={args.seed}  Device={device}")
    print("Loading GPT-2 Large...")
    ppl_tok = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tok.pad_token is None:
        ppl_tok.pad_token    = ppl_tok.eos_token
        ppl_tok.pad_token_id = ppl_tok.eos_token_id
    ppl_model = (AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.bfloat16)
                 .to(device).eval())
    elf_tok = T5Tokenizer.from_pretrained("t5-small")

    print("Loading kd_cr checkpoint...")
    ckpt  = torch.load(REPO_ROOT / CHECKPOINT, map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    results = {}
    std_ppl = None

    for arm_name in ("standard", "pipeline_avg"):
        texts = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            z0_b = all_z0[b:b+BATCH_SIZE]
            if arm_name == "standard":
                z   = run_standard(z0_b, model, t_steps, device)
                ids = decode_z(z, model, device)
            else:
                z   = run_pipeline_avg(z0_b, model, T_PIPE, device)
                ids = decode_z(z, model, device)
            texts.extend(ids_to_texts(ids.cpu(), elf_tok))

        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        d1  = compute_distinct(texts, 1)
        d2  = compute_distinct(texts, 2)
        if arm_name == "standard":
            std_ppl = ppl
        I_val = ppl - std_ppl if std_ppl is not None else 0.0
        results[arm_name] = {"ppl": ppl, "I": I_val, "d1": d1, "d2": d2}
        print(f"  {arm_name:<16} PPL={ppl:.1f}  D1={d1:.3f}  D2={d2:.3f}  I={I_val:+.1f}")

    out = OUT_DIR / f"results_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump({"seed": args.seed, "n_seq": N_SEQ, **results}, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
