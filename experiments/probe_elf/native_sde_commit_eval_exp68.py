#!/usr/bin/env python3
"""Paired native SDE-32 fidelity check for calibrated hard commitment."""

import argparse
import json
from pathlib import Path

import torch

from unified_method_eval_exp64 import (
    CHECKPOINTS,
    SamplingConfig,
    build_condition_data,
    confidence_mask,
    decode,
    decode_texts,
    empty_condition,
    get_gutenberg_pairs,
    load_weights,
    model_config,
    restore_cond,
    rouge_l_f1,
    text_metrics,
)


FIDELITY_CHECKPOINTS = (
    "baseline",
    "ct_control",
    "kd_early",
    "ct_control_s7",
    "kd_early_s7",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=FIDELITY_CHECKPOINTS, required=True)
    commit = parser.add_mutually_exclusive_group(required=True)
    commit.add_argument("--commit_time", type=float)
    commit.add_argument("--commit_step", type=int)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_uncond", type=int, default=256)
    parser.add_argument("--n_cond", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--prefix_length", type=int, default=512)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--sde_gamma", type=float, default=1.5)
    parser.add_argument("--p_mean", type=float, default=-0.8)
    parser.add_argument("--p_std", type=float, default=0.8)
    parser.add_argument("--label", default="native_sde32")
    return parser.parse_args()


@torch.no_grad()
def sde_rollout(
    z0,
    model,
    t_steps,
    args,
    step_noise_seed,
    hard_commit=False,
    cond_seq=None,
    cond_mask=None,
):
    from utils.sampling_utils import _ode_step, _sde_step

    if cond_seq is None:
        cond_seq, cond_mask = empty_condition(z0)
    active_seq = cond_seq.clone()
    active_mask = cond_mask.clone()
    z = restore_cond(z0.clone(), active_seq, active_mask)
    x_pred = restore_cond(torch.zeros_like(z), active_seq, active_mask)
    committed = False
    commit_fraction = 0.0
    device_index = z.device.index if z.device.index is not None else 0

    step_kwargs = dict(
        model=model,
        config=SamplingConfig(),
        cfg_scale=1.0,
        self_cond_cfg_scale=args.sccfg,
    )
    with torch.random.fork_rng(devices=[device_index]):
        torch.cuda.manual_seed(step_noise_seed)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for index in range(t_steps.shape[0] - 2):
                t = t_steps[index].item()
                t_next = t_steps[index + 1].item()
                z, x_pred = _sde_step(
                    z=z,
                    t=t,
                    t_next=t_next,
                    x_pred_prev=x_pred,
                    cond_seq=active_seq,
                    cond_seq_mask=active_mask,
                    gamma=args.sde_gamma,
                    generator=None,
                    **step_kwargs,
                )
                reached_commit = (
                    args.commit_step is not None and index + 1 == args.commit_step
                ) or (
                    args.commit_time is not None and t_next >= args.commit_time
                )
                if hard_commit and not committed and reached_commit:
                    selected = confidence_mask(x_pred, model, args.confidence)
                    selected &= active_mask < 0.5
                    eligible = (cond_mask < 0.5).sum().item()
                    commit_fraction = selected.sum().item() / max(eligible, 1)
                    active_seq = torch.where(
                        selected.unsqueeze(-1), x_pred.detach(), active_seq
                    )
                    active_mask = torch.maximum(
                        active_mask, selected.to(active_mask.dtype)
                    )
                    z = restore_cond(z, active_seq, active_mask)
                    x_pred = restore_cond(x_pred, active_seq, active_mask)
                    committed = True

            t = t_steps[-2].item()
            t_next = t_steps[-1].item()
            z, x_pred = _ode_step(
                z=z,
                t=t,
                t_next=t_next,
                x_pred_prev=x_pred,
                cond_seq=active_seq,
                cond_seq_mask=active_mask,
                **step_kwargs,
            )
    return z, commit_fraction


def batch_time_grid(args, device, batch_index):
    from utils.sampling_utils import get_sampling_steps

    device_index = device.index if device.index is not None else 0
    with torch.random.fork_rng(devices=[device_index]):
        torch.cuda.manual_seed(args.seed * 100003 + 1709 + batch_index)
        return get_sampling_steps(
            args.n_steps,
            time_schedule="logit_normal",
            P_mean=args.p_mean,
            P_std=args.p_std,
            device=device,
        )


@torch.no_grad()
def evaluate_pair(
    z0,
    model,
    args,
    elf_tokenizer,
    ppl_model,
    ppl_tokenizer,
    cond_seq=None,
    cond_mask=None,
    suffix_start=0,
    references=None,
):
    texts = {"standard": [], "hard_commit": []}
    fractions = []
    for batch_index, start in enumerate(range(0, z0.shape[0], args.batch_size)):
        end = start + args.batch_size
        batch_z0 = z0[start:end]
        batch_seq = cond_seq[start:end] if cond_seq is not None else None
        batch_mask = cond_mask[start:end] if cond_mask is not None else None
        t_steps = batch_time_grid(args, z0.device, batch_index)
        step_noise_seed = args.seed * 100003 + 7919 + batch_index
        for arm, use_commit in (("standard", False), ("hard_commit", True)):
            z, fraction = sde_rollout(
                batch_z0,
                model,
                t_steps,
                args,
                step_noise_seed,
                hard_commit=use_commit,
                cond_seq=batch_seq,
                cond_mask=batch_mask,
            )
            ids = decode(z, model, z.device)
            texts[arm].extend(decode_texts(ids.cpu(), elf_tokenizer, suffix_start))
            if use_commit:
                fractions.append(fraction)

    results = {}
    for arm in ("standard", "hard_commit"):
        metrics = text_metrics(
            texts[arm],
            ppl_model,
            ppl_tokenizer,
            z0.device,
            max_length=max(args.max_length - suffix_start, 2),
        )
        if references is not None:
            metrics["rouge_l"] = sum(
                rouge_l_f1(hypothesis, reference)
                for hypothesis, reference in zip(texts[arm], references)
            ) / max(len(references), 1)
        metrics["commit_fraction"] = (
            sum(fractions) / len(fractions) if arm == "hard_commit" else None
        )
        metrics["texts"] = texts[arm]
        results[arm] = metrics
    return results


def main():
    args = parse_args()
    if args.n_steps != 32:
        raise ValueError("EXP-68 fixes the native fidelity budget at SDE-32")
    if args.commit_step is not None and not 1 <= args.commit_step < args.n_steps:
        raise ValueError("commit_step must lie in [1, n_steps - 1]")
    if args.max_length <= args.prefix_length:
        raise ValueError("prefix_length must be smaller than max_length")

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    from modules.model import ELF_B
    from modules.t5_encoder import get_encoder
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
    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**model_config(args.max_length))
    missing, unexpected = model.load_state_dict(load_weights(checkpoint), strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device).eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    total = max(args.n_uncond, args.n_cond)
    all_z0 = args.noise_scale * torch.randn(
        total,
        args.max_length,
        512,
        generator=generator,
        device=device,
    )

    print(f"[{args.checkpoint}] native SDE unconditional")
    unconditional = evaluate_pair(
        all_z0[: args.n_uncond],
        model,
        args,
        elf_tokenizer,
        ppl_model,
        ppl_tokenizer,
    )

    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    pairs = get_gutenberg_pairs(
        elf_tokenizer,
        args.n_cond,
        args.prefix_length,
        args.max_length - args.prefix_length,
    )
    cond_seq, cond_mask, references = build_condition_data(
        pairs,
        elf_tokenizer,
        encoder,
        device,
        args.max_length,
        args.prefix_length,
    )
    cond_z0 = all_z0[: args.n_cond].clone()
    cond_z0[:, : args.prefix_length] = cond_seq[:, : args.prefix_length].to(
        cond_z0.dtype
    )
    print(f"[{args.checkpoint}] native SDE conditioned")
    conditioned = evaluate_pair(
        cond_z0,
        model,
        args,
        elf_tokenizer,
        ppl_model,
        ppl_tokenizer,
        cond_seq,
        cond_mask,
        args.prefix_length,
        references,
    )

    for arm in ("standard", "hard_commit"):
        u = unconditional[arm]
        c = conditioned[arm]
        print(
            f"{arm}: PPL={u['ppl']:.1f} D1={u['d1']:.3f} "
            f"D2={u['d2']:.3f} deg={u['degeneration_rate']:.3f} "
            f"cond_PPL={c['ppl']:.1f} cond_RL={c['rouge_l']:.3f}"
        )

    output_dir = Path("results/exp68_native_sde_commit")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "sampling_method": "native_sde",
                "paired_sde_noise": True,
                "seed": args.seed,
                "n_uncond": args.n_uncond,
                "n_cond": args.n_cond,
                "max_length": args.max_length,
                "prefix_length": args.prefix_length,
                "n_steps": args.n_steps,
                "noise_scale": args.noise_scale,
                "sccfg": args.sccfg,
                "sde_gamma": args.sde_gamma,
                "time_schedule": "logit_normal",
                "p_mean": args.p_mean,
                "p_std": args.p_std,
                "commit_time": args.commit_time,
                "commit_step": args.commit_step,
                "confidence": args.confidence,
                "results": {
                    "standard": {
                        "unconditional": unconditional["standard"],
                        "conditioned": conditioned["standard"],
                    },
                    "hard_commit": {
                        "unconditional": unconditional["hard_commit"],
                        "conditioned": conditioned["hard_commit"],
                    },
                },
            },
            handle,
            indent=2,
        )
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
