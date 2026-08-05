#!/usr/bin/env python3
"""Evaluate a native Wavefront Flow Forcing pilot checkpoint.

Runs matched standard ODE and native per-token-clock samplers from identical
initial noise.  Invoke once for the synchronous fine-tuning control and once
for the WFF-trained checkpoint, then compare the sampler x training
interaction rather than attributing generic extra fine-tuning to WFF.
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
from utils.sampling_utils import (
    _ode_step,
    _wff_ode_step,
    get_sampling_steps,
    make_wff_time_vector,
)

N_STEPS = 32
MAX_LENGTH = 128
BATCH_SIZE = 16
SCCFG = 1.0
OUT_DIR = Path("results/exp60_wff_pilot")

MODEL_CFG = dict(
    text_encoder_dim=512,
    max_length=MAX_LENGTH,
    num_time_tokens=4,
    num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4,
    vocab_size=32100,
    bottleneck_dim=128,
    per_token_time_conditioning=True,
)


class _Cfg:
    t_eps = 0.05
    self_cond_prob = 1.0
    num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 2.0
    use_bf16 = True


def compute_distinct(texts, n):
    all_ngrams, unique_ngrams = [], set()
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams) / len(all_ngrams) if all_ngrams else 0.0


def repetition_rate(texts, n=4):
    rates = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        rates.append(1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0)
    return sum(rates) / len(rates) if rates else 0.0


def compute_ppl(texts, model, tokenizer, device, max_length=200):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    total_nll = total_tokens = 0
    with torch.no_grad():
        for start in range(0, ids.shape[0], 8):
            ids_b, mask_b = ids[start:start + 8], mask[start:start + 8]
            logits = model(input_ids=ids_b, attention_mask=mask_b).logits[:, :-1].float()
            targets = ids_b[:, 1:]
            valid = mask_b[:, 1:].bool()
            nll = -(
                F.log_softmax(logits, dim=-1)
                .gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                * valid.float()
            ).sum().item()
            total_nll += nll
            total_tokens += valid.sum().item()
    return math.exp(total_nll / total_tokens) if total_tokens else float("nan")


@torch.no_grad()
def decode_z(z, model, device):
    batch_size = z.shape[0]
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
    t_final = torch.ones(batch_size, dtype=z.dtype, device=device)
    sc = torch.ones(batch_size, dtype=z.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(
            z_input,
            t_final,
            deterministic=True,
            self_cond_cfg_scale=sc,
            decoder_step_active=True,
        )
    return logits.argmax(dim=-1)


@torch.no_grad()
def run_standard(z0, model, t_steps, device):
    z, x_pred = z0.clone(), torch.zeros_like(z0)
    cfg = _Cfg()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for index in range(t_steps.shape[0] - 1):
            z, x_pred = _ode_step(
                z=z,
                t=t_steps[index].item(),
                t_next=t_steps[index + 1].item(),
                x_pred_prev=x_pred,
                model=model,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=SCCFG,
                cond_seq=None,
                cond_seq_mask=None,
            )
    return z


@torch.no_grad()
def run_wff(z0, model, t_steps, device, delta, order):
    z, x_pred = z0.clone(), torch.zeros_like(z0)
    cfg = _Cfg()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for index in range(t_steps.shape[0] - 1):
            t = t_steps[index].item()
            t_next = t_steps[index + 1].item()
            tau = make_wff_time_vector(
                t, z.shape[1], delta, order, device=device, dtype=z.dtype
            )
            tau_next = make_wff_time_vector(
                t_next, z.shape[1], delta, order, device=device, dtype=z.dtype
            )
            z, x_pred = _wff_ode_step(
                z=z,
                t=tau,
                t_next=tau_next,
                x_pred_prev=x_pred,
                model=model,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=SCCFG,
                cond_seq=None,
                cond_seq_mask=None,
            )
    return z


def ids_to_texts(ids, tokenizer):
    eos = tokenizer.eos_token_id
    texts = []
    for row in ids.tolist():
        if eos in row:
            row = row[:row.index(eos)]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seq", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    ppl_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()
    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    weights = checkpoint.get("ema_params1", checkpoint["params"])
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    gate = float(torch.tanh(model.local_time_gate.detach()).cpu())
    print(f"label={args.label} seed={args.seed} local_time_gate={gate:+.6f}")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    all_z0 = torch.randn(
        args.n_seq, MAX_LENGTH, 512, generator=generator, device=device
    ) * _Cfg.denoiser_noise_scale
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    arms = {
        "standard": None,
        "wff_ltr_d10": (0.10, "ltr"),
        "wff_ltr_d20": (0.20, "ltr"),
        "wff_rtl_d20": (0.20, "rtl"),
    }
    results = {}
    standard_ppl = None
    for arm_name, wave in arms.items():
        texts = []
        for start in range(0, args.n_seq, BATCH_SIZE):
            z0 = all_z0[start:start + BATCH_SIZE]
            z = (
                run_standard(z0, model, t_steps, device)
                if wave is None
                else run_wff(z0, model, t_steps, device, wave[0], wave[1])
            )
            token_ids = decode_z(z, model, device)
            texts.extend(ids_to_texts(token_ids.cpu(), elf_tokenizer))

        ppl = compute_ppl(texts, ppl_model, ppl_tokenizer, device)
        if standard_ppl is None:
            standard_ppl = ppl
        result = {
            "ppl": ppl,
            "ppl_delta": ppl - standard_ppl,
            "d1": compute_distinct(texts, 1),
            "d2": compute_distinct(texts, 2),
            "rep4": repetition_rate(texts, 4),
            "samples": texts[:4],
        }
        results[arm_name] = result
        print(
            f"{arm_name:<16} PPL={result['ppl']:.1f} "
            f"dPPL={result['ppl_delta']:+.1f} D1={result['d1']:.3f} "
            f"D2={result['d2']:.3f} rep4={result['rep4']:.3f}"
        )

    out_path = OUT_DIR / f"{args.label}_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "label": args.label,
                "seed": args.seed,
                "n_seq": args.n_seq,
                "checkpoint": args.checkpoint,
                "local_time_gate": gate,
                "results": results,
            },
            handle,
            indent=2,
        )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
