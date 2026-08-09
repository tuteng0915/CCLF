#!/usr/bin/env python3
"""Evaluate EXP-72 layerwise multi-time checkpoints and clock sensitivity."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import unified_method_eval_exp64 as common
from modules.model import ELF_B
from utils.sampling_utils import (
    _ode_step,
    _wff_ode_step,
    get_sampling_steps,
    make_wff_time_vector,
    net_out_to_v_x,
)


OUT_DIR = Path("results/exp72_multitime_v2")
ARMS = {
    "standard": None,
    "ltr_d05": (0.05, "ltr"),
    "ltr_d10": (0.10, "ltr"),
    "ltr_d15": (0.15, "ltr"),
    "rtl_d10": (0.10, "rtl"),
    "random_d10": (0.10, "random"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seq", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--refine_start", type=float, default=0.875)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    return parser.parse_args()


def model_config(max_length):
    cfg = common.model_config(max_length)
    cfg.update(
        per_token_time_conditioning=True,
        per_layer_time_conditioning=True,
    )
    return cfg


def random_time_vector(global_t, length, delta, refine_start, device, dtype, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    permutation = torch.randperm(length, generator=generator, device=device)
    rank = torch.empty_like(permutation)
    rank[permutation] = torch.arange(length, device=device)
    if length > 1:
        offset = 1.0 - 2.0 * rank.to(dtype) / (length - 1)
    else:
        offset = torch.zeros(length, dtype=dtype, device=device)
    normalized = min(max(float(global_t) / refine_start, 0.0), 1.0)
    envelope = math.sin(math.pi * normalized)
    return (float(global_t) + delta * envelope * offset).clamp(0.0, 1.0)


def time_vector(global_t, length, delta, order, args, device, dtype):
    if order == "random":
        return random_time_vector(
            global_t, length, delta, args.refine_start, device, dtype, args.seed
        )
    return make_wff_time_vector(
        global_t,
        length,
        delta,
        order,
        device=device,
        dtype=dtype,
        refine_start=args.refine_start,
    )


@torch.no_grad()
def run_standard(z0, model, grid, args):
    return common.standard_ode(z0, model, grid, args.sccfg)[0]


@torch.no_grad()
def run_wave(z0, model, grid, delta, order, args):
    cfg = common.SamplingConfig()
    z = z0.clone()
    x_pred = torch.zeros_like(z)
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for index in range(grid.shape[0] - 1):
            tau = time_vector(
                grid[index].item(), z.shape[1], delta, order, args, z.device, z.dtype
            )
            tau_next = time_vector(
                grid[index + 1].item(),
                z.shape[1],
                delta,
                order,
                args,
                z.device,
                z.dtype,
            )
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
    return z


@torch.no_grad()
def clock_sensitivity(z0, model, args):
    cfg = common.SamplingConfig()
    grid = get_sampling_steps(16, time_schedule="uniform", device=z0.device)
    z = z0.clone()
    x_pred = torch.zeros_like(z)
    for index in range(8):
        z, x_pred = _ode_step(
            model=model,
            z=z,
            t=grid[index].item(),
            t_next=grid[index + 1].item(),
            x_pred_prev=x_pred,
            config=cfg,
            cfg_scale=1.0,
            self_cond_cfg_scale=args.sccfg,
            cond_seq=None,
            cond_seq_mask=None,
        )
    base_t = grid[8].item()
    scalar = torch.full(
        z.shape[:2], base_t, dtype=z.dtype, device=z.device
    )
    ltr = time_vector(
        base_t, z.shape[1], 0.10, "ltr", args, z.device, z.dtype
    )[None].expand(z.shape[0], -1)
    rtl = torch.flip(ltr, dims=(1,))
    model_input = torch.cat([z, x_pred], dim=-1)
    sc = torch.full(
        (z.shape[0],), args.sccfg, dtype=z.dtype, device=z.device
    )

    def velocity(tau):
        net_out = model(
            model_input,
            tau,
            deterministic=True,
            self_cond_cfg_scale=sc,
            decoder_step_active=None,
        )[0]
        return net_out_to_v_x(net_out, z, tau, cfg.t_eps)[0].float()

    v_scalar = velocity(scalar)
    v_ltr = velocity(ltr)
    v_rtl = velocity(rtl)

    def normalized_change(v_other, tau_other):
        numerator = (v_other - v_scalar).norm(dim=-1).mean().item()
        denominator = (tau_other - scalar).abs().mean().item()
        return numerator / max(denominator, 1e-8)

    return {
        "global_t": base_t,
        "S_tau_ltr": normalized_change(v_ltr, ltr),
        "S_tau_rtl": normalized_change(v_rtl, rtl),
        "ltr_rtl_velocity_cosine": float(
            torch.nn.functional.cosine_similarity(
                v_ltr.reshape(-1, v_ltr.shape[-1]),
                v_rtl.reshape(-1, v_rtl.shape[-1]),
                dim=-1,
            ).mean().item()
        ),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**model_config(args.max_length))
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
        args.n_seq,
        args.max_length,
        model_config(args.max_length)["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    grid = get_sampling_steps(args.n_steps, time_schedule="uniform", device=device)
    results = {}
    for arm in args.arms:
        texts = []
        wave = ARMS[arm]
        for start in range(0, args.n_seq, args.batch_size):
            batch = z0[start:start + args.batch_size]
            z = (
                run_standard(batch, model, grid, args)
                if wave is None
                else run_wave(batch, model, grid, wave[0], wave[1], args)
            )
            ids = common.decode(z, model, device)
            texts.extend(common.decode_texts(ids.cpu(), elf_tokenizer))
        metrics = common.text_metrics(
            texts, ppl_model, ppl_tokenizer, device, max_length=args.max_length
        )
        metrics["samples"] = texts[:4]
        metrics["texts"] = texts
        results[arm] = metrics
        print(
            f"{arm:<12} PPL={metrics['ppl']:.1f} D1={metrics['d1']:.3f} "
            f"D2={metrics['d2']:.3f} deg={metrics['degeneration_rate']:.3f}"
        )

    sensitivity = clock_sensitivity(z0[: min(4, args.n_seq)], model, args)
    scales = model.local_time_scales.detach().float().cpu().tolist()
    projection_norms = [
        float(layer.weight.detach().float().norm().cpu())
        for layer in model.local_time_projections
    ]
    print("clock_sensitivity=" + json.dumps(sensitivity, indent=2))
    print(f"local_time_scales={scales}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_seed{args.seed}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "label": args.label,
                "checkpoint": str(checkpoint_path),
                "seed": args.seed,
                "n_seq": args.n_seq,
                "noise_scale": args.noise_scale,
                "sccfg": args.sccfg,
                "refine_start": args.refine_start,
                "local_time_scales": scales,
                "local_time_projection_norms": projection_norms,
                "clock_sensitivity": sensitivity,
                "results": results,
            },
            handle,
            indent=2,
        )
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
