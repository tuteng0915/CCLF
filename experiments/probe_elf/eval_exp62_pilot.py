#!/usr/bin/env python3
"""Native generation gate for one EXP-62 checkpoint.

This intentionally evaluates only the ordinary ELF ODE sampler.  The pilot's
causal contrast is full-KD versus matched continued training; asynchronous or
Pipeline samplers would add a second intervention before that contrast is
established.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import get_sampling_steps

from eval_wff_pilot import (
    BATCH_SIZE,
    MAX_LENGTH,
    N_STEPS,
    _Cfg,
    compute_distinct,
    compute_ppl,
    decode_z,
    ids_to_texts,
    repetition_rate,
    run_standard,
)

OUT_DIR = Path("results/exp62_checkpoint_panel")

MODEL_CFG = dict(
    text_encoder_dim=512,
    max_length=MAX_LENGTH,
    num_time_tokens=4,
    num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4,
    vocab_size=32100,
    bottleneck_dim=128,
    per_token_time_conditioning=False,
)


def degeneration_rate(texts):
    """Fraction of empty, non-ASCII-heavy, or highly repetitive samples."""
    bad = 0
    for text in texts:
        tokens = text.split()
        empty = not tokens
        non_ascii = bool(text) and sum(ord(char) > 127 for char in text) / len(text) > 0.3
        repetitive = len(tokens) >= 8 and len(set(tokens)) / len(tokens) < 0.4
        bad += int(empty or non_ascii or repetitive)
    return bad / len(texts) if texts else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seq", type=int, default=256)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
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
    weights = checkpoint["ema_params1"] if "ema_params1" in checkpoint else checkpoint["params"]
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    all_z0 = torch.randn(
        args.n_seq, MAX_LENGTH, 512, generator=generator, device=device
    ) * _Cfg.denoiser_noise_scale
    t_steps = get_sampling_steps(args.n_steps, time_schedule="uniform", device=device)

    texts = []
    for start in range(0, args.n_seq, BATCH_SIZE):
        z0 = all_z0[start:start + BATCH_SIZE]
        z = run_standard(z0, model, t_steps, device, args.sccfg)
        token_ids = decode_z(z, model, device)
        texts.extend(ids_to_texts(token_ids.cpu(), elf_tokenizer))

    result = {
        "ppl": compute_ppl(texts, ppl_model, ppl_tokenizer, device),
        "d1": compute_distinct(texts, 1),
        "d2": compute_distinct(texts, 2),
        "rep4": repetition_rate(texts, 4),
        "degeneration_rate": degeneration_rate(texts),
        "samples": texts[:8],
    }
    print(
        f"{args.label}: PPL={result['ppl']:.1f} D1={result['d1']:.3f} "
        f"D2={result['d2']:.3f} rep4={result['rep4']:.3f} "
        f"deg={result['degeneration_rate']:.3f}"
    )

    step_suffix = "" if args.n_steps == N_STEPS else f"_ode{args.n_steps}"
    out_path = OUT_DIR / f"{args.label}_seed{args.seed}{step_suffix}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "label": args.label,
                "checkpoint": args.checkpoint,
                "weights": "ema_params1" if "ema_params1" in checkpoint else "params",
                "seed": args.seed,
                "n_seq": args.n_seq,
                "length": MAX_LENGTH,
                "ode_steps": args.n_steps,
                "noise_scale": _Cfg.denoiser_noise_scale,
                "sccfg": args.sccfg,
                "result": result,
            },
            handle,
            indent=2,
        )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
