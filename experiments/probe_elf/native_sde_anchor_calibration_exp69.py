#!/usr/bin/env python3
"""Calibrate native-SDE anchor density before causal commitment tests.

Unlike the ODE experiments, ELF's native logit-normal grid is stochastic and
its nominal time crossings need not leave a comparable unresolved set.  This
script measures lexical-confidence distributions at fixed solver-step indices
without intervening on the trajectory.
"""

import argparse
import json
from pathlib import Path

import torch

from native_sde_commit_eval_exp68 import batch_time_grid
from unified_method_eval_exp64 import (
    CHECKPOINTS,
    SamplingConfig,
    empty_condition,
    load_weights,
    model_config,
)


CALIBRATION_CHECKPOINTS = ("baseline", "ct_control", "kd_early")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CALIBRATION_CHECKPOINTS, default="baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--sde_gamma", type=float, default=1.5)
    parser.add_argument("--p_mean", type=float, default=-0.8)
    parser.add_argument("--p_std", type=float, default=0.8)
    parser.add_argument("--probe_steps", default="4,8,12,16,20,24,28")
    parser.add_argument("--thresholds", default="0.60,0.70,0.80,0.90,0.95,0.99")
    parser.add_argument("--label", default="calibration")
    return parser.parse_args()


def comma_values(text, cast):
    return tuple(cast(value.strip()) for value in text.split(",") if value.strip())


@torch.no_grad()
def lexical_confidence(x_pred, model):
    batch = x_pred.shape[0]
    z_in = torch.cat([x_pred, torch.zeros_like(x_pred)], dim=-1)
    ones = torch.ones(batch, dtype=x_pred.dtype, device=x_pred.device)
    _, logits, _ = model(
        z_in,
        ones,
        deterministic=True,
        self_cond_cfg_scale=ones,
        decoder_step_active=True,
    )
    logits = logits.float()
    top_value = logits.max(dim=-1).values
    return (top_value - torch.logsumexp(logits, dim=-1)).exp()


@torch.no_grad()
def collect_batch(z0, model, t_steps, args, probe_steps, step_noise_seed):
    from utils.sampling_utils import _sde_step

    cond_seq, cond_mask = empty_condition(z0)
    z = z0.clone()
    x_pred = torch.zeros_like(z)
    device_index = z.device.index if z.device.index is not None else 0
    records = {}
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
                z, x_pred = _sde_step(
                    z=z,
                    t=t_steps[index].item(),
                    t_next=t_steps[index + 1].item(),
                    x_pred_prev=x_pred,
                    cond_seq=cond_seq,
                    cond_seq_mask=cond_mask,
                    gamma=args.sde_gamma,
                    generator=None,
                    **step_kwargs,
                )
                completed_step = index + 1
                if completed_step in probe_steps:
                    confidence = lexical_confidence(x_pred, model)
                    records[completed_step] = {
                        "time": float(t_steps[index + 1].item()),
                        "confidence": confidence.float().cpu(),
                    }
    return records


def summarize(per_batch, probe_steps, thresholds):
    quantile_levels = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    output = {}
    for step in probe_steps:
        available = [record[step] for record in per_batch if step in record]
        if not available:
            continue
        confidence = torch.cat([record["confidence"].flatten() for record in available])
        times = torch.tensor([record["time"] for record in available])
        quantiles = torch.quantile(confidence, quantile_levels)
        output[str(step)] = {
            "mean_time": float(times.mean().item()),
            "std_time": float(times.std(unbiased=False).item()),
            "confidence_quantiles": {
                f"q{int(level.item() * 100):02d}": float(value.item())
                for level, value in zip(quantile_levels, quantiles)
            },
            "anchor_fraction": {
                f"{threshold:.2f}": float((confidence >= threshold).float().mean().item())
                for threshold in thresholds
            },
        }
    return output


def main():
    args = parse_args()
    if args.n_steps != 32:
        raise ValueError("EXP-69 calibrates the native SDE-32 protocol")
    probe_steps = comma_values(args.probe_steps, int)
    thresholds = comma_values(args.thresholds, float)
    if not probe_steps or min(probe_steps) < 1 or max(probe_steps) >= args.n_steps:
        raise ValueError("probe_steps must lie in [1, n_steps - 1]")

    from modules.model import ELF_B
    from unified_method_eval_exp64 import REPO_ROOT

    device = torch.device(args.device)
    SamplingConfig.denoiser_noise_scale = args.noise_scale
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
    per_batch = []
    for batch_index, start in enumerate(range(0, args.n_samples, args.batch_size)):
        batch = z0[start : start + args.batch_size]
        t_steps = batch_time_grid(args, device, batch_index)
        per_batch.append(
            collect_batch(
                batch,
                model,
                t_steps,
                args,
                set(probe_steps),
                args.seed * 100003 + 7919 + batch_index,
            )
        )
        print(f"batch {batch_index + 1}/{(args.n_samples + args.batch_size - 1) // args.batch_size}")

    summary = summarize(per_batch, probe_steps, thresholds)
    for step, record in summary.items():
        print(
            f"step={int(step):02d} t={record['mean_time']:.3f} "
            f"q50={record['confidence_quantiles']['q50']:.3f} "
            f"fractions={record['anchor_fraction']}"
        )

    output_dir = Path("results/exp69_native_sde_anchor_calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "seed": args.seed,
                "n_samples": args.n_samples,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "n_steps": args.n_steps,
                "sampling_method": "native_sde",
                "time_schedule": "logit_normal",
                "p_mean": args.p_mean,
                "p_std": args.p_std,
                "sde_gamma": args.sde_gamma,
                "noise_scale": args.noise_scale,
                "sccfg": args.sccfg,
                "probe_steps": probe_steps,
                "thresholds": thresholds,
                "summary": summary,
            },
            handle,
            indent=2,
        )
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
