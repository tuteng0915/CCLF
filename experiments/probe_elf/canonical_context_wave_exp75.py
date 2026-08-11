#!/usr/bin/env python3
"""EXP-75: canonical predicted-clean context for a heterogeneous block wave."""

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

import pipeline_factorization_exp70 as exp70
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


CHECKPOINTS = {
    name: common.CHECKPOINTS[name]
    for name in ("baseline", "ct_control", "kd_early")
}
ARMS = (
    "standard",
    "pipeline_local",
    "canonical_ltr",
    "canonical_ltr_refine8",
    "canonical_rtl",
    "canonical_shuffled",
)
OUT_DIR = Path("results/exp75_canonical_context")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seq", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--diagnostic_samples", type=int, default=4)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--label", default="screen")
    return parser.parse_args()


def canonical_input(z, x_pred, selected, shuffled=False):
    context = x_pred
    if shuffled:
        context = (
            torch.roll(x_pred, shifts=1, dims=0)
            if x_pred.shape[0] > 1
            else torch.roll(x_pred, shifts=max(x_pred.shape[1] // 4, 1), dims=1)
        )
    return torch.where(selected.view(1, -1, 1), z, context)


@torch.no_grad()
def canonical_pipeline(
    z0,
    model,
    groups,
    sccfg,
    order="ltr",
    refine_steps=0,
    shuffled=False,
    seed=42,
    cond_seq=None,
    cond_mask=None,
    eligible_mask=None,
):
    cfg = common.SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = exp70.empty_condition(z0)
    group_of = exp70.balanced_group_map(
        z0.shape[1], groups, order, z0.device, seed, eligible_mask
    )
    async_updates = groups - refine_steps
    total_stages = groups + async_updates - 1
    dt = 1.0 / groups
    z = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    calls = 0
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for stage in range(total_stages):
            first = max(0, stage - async_updates + 1)
            last = min(groups - 1, stage)
            z_stage, pred_stage = z, x_pred
            z_next, pred_next = z.clone(), x_pred.clone()
            for group in range(first, last + 1):
                local_step = stage - group
                local_t = local_step / groups
                selected = group_of == group
                z_input = canonical_input(z_stage, pred_stage, selected, shuffled)
                z_full, pred_full = _ode_step(
                    model=model,
                    z=z_input,
                    t=local_t,
                    t_next=local_t + dt,
                    x_pred_prev=pred_stage,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=sccfg,
                    cond_seq=cond_seq,
                    cond_seq_mask=cond_mask,
                )
                z_next = exp70.masked_replace(z_next, z_full, selected)
                pred_next = exp70.masked_replace(pred_next, pred_full, selected)
                calls += 1
            z = restore_cond(z_next, cond_seq, cond_mask)
            x_pred = restore_cond(pred_next, cond_seq, cond_mask)

        if refine_steps:
            start_t = async_updates / groups
            grid = torch.linspace(start_t, 1.0, refine_steps + 1, device=z.device)
            for index in range(refine_steps):
                z, x_pred = _ode_step(
                    model=model,
                    z=z,
                    t=grid[index].item(),
                    t_next=grid[index + 1].item(),
                    x_pred_prev=x_pred,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=sccfg,
                    cond_seq=cond_seq,
                    cond_seq_mask=cond_mask,
                )
                calls += 1
    return z, x_pred, calls


@torch.no_grad()
def diagnose(z0, model, groups, sccfg):
    cfg = common.SamplingConfig()
    cond_seq, cond_mask = exp70.empty_condition(z0)
    group_of = exp70.balanced_group_map(z0.shape[1], groups, "ltr", z0.device, 42)
    dt = 1.0 / groups
    sync_states, sync_preds = [], []
    z_sync, x_sync = z0.clone(), torch.zeros_like(z0)
    for step in range(groups):
        sync_states.append(z_sync.clone())
        sync_preds.append(x_sync.clone())
        z_sync, x_sync = _ode_step(
            model, z_sync, step / groups, (step + 1) / groups, x_sync,
            cfg, 1.0, sccfg, cond_seq, cond_mask,
        )

    totals = {name: {"n": 0, "cos": 0.0, "mse": 0.0} for name in ("raw", "canonical")}
    selected_stages = {0, groups - 1, groups, 2 * groups - 2}
    z, x_pred = z0.clone(), torch.zeros_like(z0)
    for stage in range(2 * groups - 1):
        first, last = max(0, stage - groups + 1), min(groups - 1, stage)
        z_stage, pred_stage = z, x_pred
        z_next, pred_next = z.clone(), x_pred.clone()
        for group in range(first, last + 1):
            local_step = stage - group
            local_t = local_step / groups
            selected = group_of == group
            z_input = canonical_input(z_stage, pred_stage, selected)
            z_canon, pred_canon = _ode_step(
                model, z_input, local_t, local_t + dt, pred_stage,
                cfg, 1.0, sccfg, cond_seq, cond_mask,
            )
            z_next = exp70.masked_replace(z_next, z_canon, selected)
            pred_next = exp70.masked_replace(pred_next, pred_canon, selected)
            if stage in selected_stages:
                z_raw, _ = _ode_step(
                    model, z_stage, local_t, local_t + dt, pred_stage,
                    cfg, 1.0, sccfg, cond_seq, cond_mask,
                )
                z_ref, _ = _ode_step(
                    model, sync_states[local_step], local_t, local_t + dt,
                    sync_preds[local_step], cfg, 1.0, sccfg, cond_seq, cond_mask,
                )
                velocities = {
                    "raw": (z_raw - z_stage) / dt,
                    "canonical": (z_canon - z_input) / dt,
                }
                v_ref = (z_ref - sync_states[local_step]) / dt
                for name, velocity in velocities.items():
                    stats = exp70.vector_stats(velocity, v_ref, selected)
                    totals[name]["n"] += stats["n"]
                    totals[name]["cos"] += stats["one_minus_cosine_sum"]
                    totals[name]["mse"] += stats["mse_sum"]
        z, x_pred = z_next, pred_next
    return {
        f"E_{name}": values["cos"] / max(values["n"], 1)
        for name, values in totals.items()
    } | {
        f"MSE_{name}": values["mse"] / max(values["n"], 1)
        for name, values in totals.items()
    }


def run_arm(name, z0, model, grid, args):
    if name == "standard":
        z, x = common.standard_ode(z0, model, grid, args.sccfg)
        return z, x, args.n_steps
    if name == "pipeline_local":
        return exp70.pipeline_local(z0, model, args.groups, args.sccfg, seed=args.seed)
    if name == "canonical_ltr":
        return canonical_pipeline(z0, model, args.groups, args.sccfg, seed=args.seed)
    if name == "canonical_ltr_refine8":
        return canonical_pipeline(
            z0, model, args.groups, args.sccfg, refine_steps=8, seed=args.seed
        )
    if name == "canonical_rtl":
        return canonical_pipeline(
            z0, model, args.groups, args.sccfg, order="rtl", seed=args.seed
        )
    if name == "canonical_shuffled":
        return canonical_pipeline(
            z0, model, args.groups, args.sccfg, shuffled=True, seed=args.seed
        )
    raise ValueError(name)


def main():
    args = parse_args()
    if args.n_steps != 2 * args.groups:
        raise ValueError("n_steps must equal 2 * groups")
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(args.max_length))
    missing, unexpected = model.load_state_dict(common.load_weights(checkpoint), strict=False)
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
        args.n_seq, args.max_length, common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator, device=device,
    )
    grid = get_sampling_steps(args.n_steps, "uniform", device=device)
    results = {}
    for arm in args.arms:
        texts, calls = [], None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for start in range(0, args.n_seq, args.batch_size):
            z, _, batch_calls = run_arm(
                arm, z0[start:start + args.batch_size], model, grid, args
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
            f"{arm:<24} PPL={metrics['ppl']:.1f} D1={metrics['d1']:.3f} "
            f"D2={metrics['d2']:.3f} deg={metrics['degeneration_rate']:.3f} calls={calls}"
        )
    diagnostics = diagnose(
        z0[: min(args.diagnostic_samples, args.n_seq)], model, args.groups, args.sccfg
    )
    print("diagnostics=" + json.dumps(diagnostics, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({**vars(args), "results": results, "diagnostics": diagnostics}, handle, indent=2)
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
