#!/usr/bin/env python3
"""EXP-61: revalidate Pipeline ODE under ELF's native evaluation path.

The historical EXP-58/59 scripts sampled z0 ~ N(0, I). Native ELF generation
samples z0 = denoiser_noise_scale * eps (scale 2.0 in the relevant configs).

The converted checkpoints use the generic outer key ``params`` even when the
converter selected JAX ``ema_params1`` (its default). ``auto`` therefore
matches the native loader: prefer an explicit PyTorch EMA shadow when present,
otherwise use the converted ``params`` payload. Explicit ``params``/``ema``
choices remain available for provenance audits of newly trained checkpoints.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pipeline_avg_multiseed_exp59 as legacy
from modules.model import ELF_B
from utils.sampling_utils import get_sampling_steps


CHECKPOINTS = {
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
    "kd_cr": "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2": "converted/elf_b-owt-kd2_torch.pt",
}
OUT_DIR = Path("results/exp61_pipeline_native_revalidation")


def build_model_cfg(max_length):
    return dict(
        text_encoder_dim=512,
        max_length=max_length,
        num_time_tokens=4,
        num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4,
        vocab_size=32100,
        bottleneck_dim=128,
    )


def compute_ppl(texts, model, tokenizer, device, max_length=200):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, ids.shape[0], 8):
            ids_b = ids[start:start + 8]
            mask_b = mask[start:start + 8]
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


def repetition_rate(texts, n=4):
    per_sample = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        per_sample.append(
            1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
        )
    return sum(per_sample) / len(per_sample) if per_sample else 0.0


def degeneration_rate(texts):
    flags = []
    for text in texts:
        tokens = text.split()
        max_word_fraction = (
            max((tokens.count(token) for token in set(tokens)), default=0) / len(tokens)
            if tokens else 1.0
        )
        non_ascii_fraction = (
            sum(ord(char) > 127 for char in text) / len(text) if text else 1.0
        )
        flags.append(
            (not text.strip())
            or max_word_fraction > 0.35
            or non_ascii_fraction > 0.02
        )
    return sum(flags) / len(flags) if flags else 0.0


def load_weights(checkpoint, source):
    if source == "auto":
        return checkpoint.get("ema_params1", checkpoint["params"])
    if source == "params":
        return checkpoint["params"]
    if "ema_params1" not in checkpoint:
        raise KeyError("EMA weights requested but checkpoint has no ema_params1")
    return checkpoint["ema_params1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument(
        "--weights", choices=("auto", "params", "ema"), default="auto"
    )
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="reproducibility seed; keep fixed across controlled comparisons",
    )
    parser.add_argument("--n_seq", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--pipeline_groups", type=int, default=16)
    parser.add_argument(
        "--ppl_max_length",
        type=int,
        default=200,
        help="GPT-2 evaluator truncation length; use 1024 for length-1024 runs",
    )
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()

    if args.noise_scale <= 0:
        raise ValueError("noise_scale must be positive")
    if min(args.max_length, args.n_steps, args.pipeline_groups, args.ppl_max_length) <= 0:
        raise ValueError("length, step, group, and evaluator limits must be positive")
    if args.n_steps != 2 * args.pipeline_groups:
        print(
            "WARNING: standard and Pipeline compute are not approximately matched: "
            f"ODE={args.n_steps} calls, Pipeline={2 * args.pipeline_groups - 1} calls"
        )
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_cfg = build_model_cfg(args.max_length)

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    print(
        f"checkpoint={args.checkpoint} weights={args.weights} "
        f"noise_scale={args.noise_scale:g} seed={args.seed} n={args.n_seq} "
        f"length={args.max_length} ode_steps={args.n_steps} "
        f"pipeline_calls={2 * args.pipeline_groups - 1}"
    )
    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    ppl_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()
    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**model_cfg)
    missing, unexpected = model.load_state_dict(
        load_weights(checkpoint, args.weights), strict=False
    )
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    model.eval().to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    all_z0 = args.noise_scale * torch.randn(
        args.n_seq,
        args.max_length,
        model_cfg["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    t_steps = get_sampling_steps(args.n_steps, time_schedule="uniform", device=device)

    results = {}
    standard_ppl = None
    for arm in ("standard", "pipeline_avg"):
        texts = []
        for start in range(0, args.n_seq, args.batch_size):
            z0 = all_z0[start:start + args.batch_size]
            if arm == "standard":
                z = legacy.run_standard(z0, model, t_steps, device)
            else:
                z = legacy.run_pipeline_avg(z0, model, args.pipeline_groups, device)
            token_ids = legacy.decode_z(z, model, device)
            texts.extend(legacy.ids_to_texts(token_ids.cpu(), elf_tokenizer))

        ppl = compute_ppl(
            texts,
            ppl_model,
            ppl_tokenizer,
            device,
            max_length=args.ppl_max_length,
        )
        if standard_ppl is None:
            standard_ppl = ppl
        result = {
            "ppl": ppl,
            "ppl_delta": ppl - standard_ppl,
            "d1": legacy.compute_distinct(texts, 1),
            "d2": legacy.compute_distinct(texts, 2),
            "rep4": repetition_rate(texts),
            "degeneration_rate": degeneration_rate(texts),
            "samples": texts[:4],
        }
        results[arm] = result
        print(
            f"{arm:<14} PPL={ppl:.2f} dPPL={result['ppl_delta']:+.2f} "
            f"D1={result['d1']:.3f} D2={result['d2']:.3f} "
            f"rep4={result['rep4']:.3f} deg={result['degeneration_rate']:.3f}"
        )

    safe_scale = str(args.noise_scale).replace(".", "p")
    out_path = OUT_DIR / (
        f"{args.label}_{args.checkpoint}_{args.weights}_ns{safe_scale}_"
        f"l{args.max_length}_ode{args.n_steps}_seed{args.seed}_n{args.n_seq}.json"
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "weights": args.weights,
                "noise_scale": args.noise_scale,
                "seed": args.seed,
                "n_seq": args.n_seq,
                "max_length": args.max_length,
                "ppl_max_length": args.ppl_max_length,
                "n_steps": args.n_steps,
                "pipeline_groups": args.pipeline_groups,
                "standard_model_calls": args.n_steps,
                "pipeline_model_calls": 2 * args.pipeline_groups - 1,
                "results": results,
            },
            handle,
            indent=2,
        )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
