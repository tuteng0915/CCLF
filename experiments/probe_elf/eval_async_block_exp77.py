#!/usr/bin/env python3
"""Paired generation panel for EXP-77 staggered block-transition models."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import eval_multitime_exp72 as exp72
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from utils.sampling_utils import _wff_ode_step, get_sampling_steps


ARMS = ("standard32", "standard64", "block_ltr", "block_rtl", "block_random")
OUT_DIR = Path("results/exp77_async_block")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seq", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--refine_steps", type=int, default=8)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    return parser.parse_args()


def config(max_length):
    cfg = common.model_config(max_length)
    cfg.update(per_token_time_conditioning=True, per_layer_time_conditioning=True)
    return cfg


def group_map(length, groups, order, device, seed):
    natural = torch.div(
        torch.arange(length, device=device) * groups, length, rounding_mode="floor"
    )
    if order == "ltr":
        return natural
    if order == "rtl":
        return groups - 1 - natural
    generator = torch.Generator(device=device).manual_seed(seed)
    permutation = torch.randperm(groups, generator=generator, device=device)
    return permutation[natural]


@torch.no_grad()
def block_wave(z0, model, args, order):
    cfg = common.SamplingConfig()
    group = group_map(z0.shape[1], args.groups, order, z0.device, args.seed)
    async_updates = args.groups - args.refine_steps
    total_stages = args.groups + async_updates - 1
    z, x_pred = z0.clone(), torch.zeros_like(z0)
    calls = 0
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for stage in range(total_stages):
            local_step = (stage - group).clamp(min=0, max=async_updates)
            next_step = (stage + 1 - group).clamp(min=0, max=async_updates)
            tau = local_step.float() / args.groups
            tau_next = next_step.float() / args.groups
            active = tau_next > tau
            z, predicted = _wff_ode_step(
                model=model,
                z=z,
                t=tau,
                t_next=tau_next,
                x_pred_prev=x_pred,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=args.sccfg,
                cond_seq=None,
                cond_seq_mask=None,
            )
            x_pred = torch.where(active.view(1, -1, 1), predicted, x_pred)
            calls += 1
        if args.refine_steps:
            start_t = async_updates / args.groups
            grid = torch.linspace(
                start_t, 1.0, args.refine_steps + 1, device=z.device
            )
            for index in range(args.refine_steps):
                tau = torch.full(
                    (z.shape[1],), grid[index].item(), device=z.device
                )
                tau_next = torch.full_like(tau, grid[index + 1].item())
                z, x_pred = _wff_ode_step(
                    model=model,
                    z=z,
                    t=tau,
                    t_next=tau_next,
                    x_pred_prev=x_pred,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=args.sccfg,
                    cond_seq=None,
                    cond_seq_mask=None,
                )
                calls += 1
    return z, calls


def run_arm(arm, z0, model, args):
    if arm.startswith("standard"):
        steps = 32 if arm == "standard32" else 64
        grid = get_sampling_steps(steps, "uniform", device=z0.device)
        z, _ = common.standard_ode(z0, model, grid, args.sccfg)
        return z, steps
    return block_wave(z0, model, args, arm.removeprefix("block_"))


def main():
    args = parse_args()
    if not 0 <= args.refine_steps < args.groups:
        raise ValueError("refine_steps must be in [0, groups)")
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ELF_B(**config(args.max_length))
    weights = checkpoint.get("ema_params1", checkpoint["params"])
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    ppl_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()
    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z0 = args.noise_scale * torch.randn(
        args.n_seq, args.max_length, config(args.max_length)["text_encoder_dim"],
        generator=generator, device=device,
    )
    results = {}
    for arm in args.arms:
        texts, calls = [], None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for start in range(0, args.n_seq, args.batch_size):
            z, batch_calls = run_arm(
                arm, z0[start:start + args.batch_size], model, args
            )
            calls = batch_calls if calls is None else calls
            ids = common.decode(z, model, device)
            texts.extend(common.decode_texts(ids.cpu(), elf_tokenizer))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        metrics = common.text_metrics(
            texts, ppl_model, ppl_tokenizer, device, max_length=args.max_length
        )
        metrics.update({"model_calls": calls, "wall_seconds": elapsed, "samples": texts[:4], "texts": texts})
        results[arm] = metrics
        print(
            f"{arm:<16} PPL={metrics['ppl']:.1f} D1={metrics['d1']:.3f} "
            f"D2={metrics['d2']:.3f} deg={metrics['degeneration_rate']:.3f} calls={calls}"
        )
    sensitivity_args = argparse.Namespace(
        sccfg=args.sccfg, refine_start=0.875
    )
    sensitivity = exp72.clock_sensitivity(z0[: min(4, args.n_seq)], model, sensitivity_args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_seed{args.seed}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({**vars(args), "clock_sensitivity": sensitivity, "results": results}, handle, indent=2)
    print("clock_sensitivity=" + json.dumps(sensitivity))
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
