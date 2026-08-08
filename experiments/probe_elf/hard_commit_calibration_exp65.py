#!/usr/bin/env python3
"""Held-out native-recipe calibration for within-trajectory hard commitment.

The historical commit times were selected under a non-native generation
recipe. This script calibrates commit time and confidence threshold on a
dedicated noise bank while keeping the native ELF ODE recipe fixed. It runs
the standard arm once, evaluates every hard-commit grid point on paired noise,
and checkpoints the JSON after every arm.
"""

import argparse
import json
from pathlib import Path

import torch

from unified_method_eval_exp64 import (
    CHECKPOINTS,
    SamplingConfig,
    decode,
    decode_texts,
    hard_commit_ode,
    load_weights,
    model_config,
    standard_ode,
    text_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument(
        "--commit_times", type=float, nargs="+", default=[0.30, 0.40, 0.50, 0.60]
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.60, 0.70, 0.80]
    )
    parser.add_argument("--label", default="native_calibration")
    return parser.parse_args()


@torch.no_grad()
def generate_arm(z0, model, t_steps, args, commit_time=None, threshold=None):
    texts = []
    fractions = []
    for start in range(0, z0.shape[0], args.batch_size):
        batch = z0[start : start + args.batch_size]
        if commit_time is None:
            z, _ = standard_ode(batch, model, t_steps, args.sccfg)
        else:
            z, _, fraction = hard_commit_ode(
                batch,
                model,
                t_steps,
                args.sccfg,
                commit_time,
                threshold,
            )
            fractions.append(fraction)
        ids = decode(z, model, z.device)
        texts.extend(decode_texts(ids.cpu(), args.elf_tokenizer))
    return texts, (sum(fractions) / len(fractions) if fractions else None)


def save_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def main():
    args = parse_args()
    if args.n_steps != 32:
        raise ValueError("EXP-65 calibration fixes the primary budget at ODE-32")
    if any(not 0.0 < value < 1.0 for value in args.commit_times):
        raise ValueError("commit times must lie strictly between zero and one")
    if any(not 0.0 < value < 1.0 for value in args.thresholds):
        raise ValueError("confidence thresholds must lie strictly between zero and one")

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    from modules.model import ELF_B
    from utils.sampling_utils import get_sampling_steps
    from unified_method_eval_exp64 import REPO_ROOT

    device = torch.device(args.device)
    SamplingConfig.denoiser_noise_scale = args.noise_scale

    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    ppl_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()
    args.elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**model_config(args.max_length))
    missing, unexpected = model.load_state_dict(load_weights(checkpoint), strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device).eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    z0 = args.noise_scale * torch.randn(
        args.n_samples,
        args.max_length,
        512,
        generator=generator,
        device=device,
    )
    t_steps = get_sampling_steps(args.n_steps, time_schedule="uniform", device=device)

    output_path = (
        Path("results/exp65_hard_commit_calibration")
        / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    )
    payload = {
        "status": "running",
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "seed_role": "held-out calibration bank",
        "seed": args.seed,
        "n_samples": args.n_samples,
        "max_length": args.max_length,
        "n_steps": args.n_steps,
        "noise_scale": args.noise_scale,
        "sccfg": args.sccfg,
        "commit_times": args.commit_times,
        "thresholds": args.thresholds,
        "results": {},
    }

    print(f"[{args.checkpoint}] standard")
    texts, _ = generate_arm(z0, model, t_steps, args)
    metrics = text_metrics(texts, ppl_model, ppl_tokenizer, device)
    metrics["texts"] = texts
    payload["results"]["standard"] = metrics
    save_payload(output_path, payload)
    print(
        f"  PPL={metrics['ppl']:.1f} D1={metrics['d1']:.3f} "
        f"D2={metrics['d2']:.3f} deg={metrics['degeneration_rate']:.3f}"
    )

    for commit_time in args.commit_times:
        for threshold in args.thresholds:
            key = f"hard_t{commit_time:.2f}_c{threshold:.2f}"
            print(f"[{args.checkpoint}] {key}")
            texts, fraction = generate_arm(
                z0, model, t_steps, args, commit_time, threshold
            )
            metrics = text_metrics(texts, ppl_model, ppl_tokenizer, device)
            metrics["commit_time"] = commit_time
            metrics["confidence_threshold"] = threshold
            metrics["commit_fraction"] = fraction
            metrics["texts"] = texts
            payload["results"][key] = metrics
            save_payload(output_path, payload)
            print(
                f"  PPL={metrics['ppl']:.1f} D1={metrics['d1']:.3f} "
                f"D2={metrics['d2']:.3f} deg={metrics['degeneration_rate']:.3f} "
                f"commit={fraction:.3f}"
            )

    payload["status"] = "complete"
    save_payload(output_path, payload)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
