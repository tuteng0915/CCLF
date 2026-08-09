#!/usr/bin/env python3
"""EXP-70: factor Pipeline failure into clock, mixed-state, and refinement errors.

The per-block local-time arm is intentionally expensive.  It evaluates the
same heterogeneous full-sequence state once per active local-time bucket and
retains only the corresponding block output.  It is a diagnostic oracle, not
an efficient sampler.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import unified_method_eval_exp64 as common
from modules.model import ELF_B
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


CHECKPOINTS = {
    name: common.CHECKPOINTS[name]
    for name in ("baseline", "ct_control", "kd_early")
}
ARMS = (
    "standard",
    "pipeline_shared",
    "pipeline_local",
    "pipeline_local_refine4",
    "pipeline_local_refine8",
    "pipeline_local_rtl",
    "pipeline_local_random",
)
OUT_DIR = Path("results/exp70_pipeline_factorization")


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
    parser.add_argument("--diagnostic_samples", type=int, default=1)
    parser.add_argument("--skip_diagnostics", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--label", default="screen")
    return parser.parse_args()


def balanced_group_map(length, groups, order, device, seed):
    positions = torch.arange(length, device=device)
    base = torch.div(positions * groups, length, rounding_mode="floor")
    if order == "ltr":
        return base
    if order == "rtl":
        return groups - 1 - base
    if order == "random":
        generator = torch.Generator(device=device).manual_seed(seed)
        permutation = torch.randperm(length, generator=generator, device=device)
        rank = torch.empty_like(permutation)
        rank[permutation] = torch.arange(length, device=device)
        return torch.div(rank * groups, length, rounding_mode="floor")
    raise ValueError(order)


def empty_condition(z0):
    batch, length, _ = z0.shape
    return (
        torch.zeros_like(z0),
        torch.zeros(batch, length, dtype=z0.dtype, device=z0.device),
    )


def masked_replace(old, new, position_mask):
    mask = position_mask.view(1, -1, 1)
    return torch.where(mask, new, old)


@torch.no_grad()
def pipeline_shared(z0, model, groups, sccfg, cond_seq=None, cond_mask=None):
    """Correctly reproduce the current average-clock Pipeline with balanced blocks."""
    cfg = common.SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = empty_condition(z0)
    group_of = balanced_group_map(z0.shape[1], groups, "ltr", z0.device, 0)
    z = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    total = 2 * groups - 1
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z0.device.type == "cuda"
    ):
        for stage in range(total):
            first = max(0, stage - groups + 1)
            last = min(groups - 1, stage)
            shared_t = stage / (2.0 * groups)
            next_t = min(shared_t + 1.0 / groups, 1.0)
            z_full, pred_full = _ode_step(
                model=model,
                z=z,
                t=shared_t,
                t_next=next_t,
                x_pred_prev=x_pred,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            active = (group_of >= first) & (group_of <= last)
            z = masked_replace(z, z_full, active)
            x_pred = masked_replace(x_pred, pred_full, active)
            z = restore_cond(z, cond_seq, cond_mask)
            x_pred = restore_cond(x_pred, cond_seq, cond_mask)
    return z, x_pred, total


@torch.no_grad()
def pipeline_local(
    z0,
    model,
    groups,
    sccfg,
    order="ltr",
    refine_steps=0,
    cond_seq=None,
    cond_mask=None,
    seed=42,
):
    """Block-Jacobi Pipeline using the target block's intended local time."""
    if not 0 <= refine_steps < groups:
        raise ValueError("refine_steps must be in [0, groups)")
    cfg = common.SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = empty_condition(z0)
    group_of = balanced_group_map(z0.shape[1], groups, order, z0.device, seed)
    async_updates = groups - refine_steps
    total_stages = groups + async_updates - 1
    dt = 1.0 / groups
    z = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    model_calls = 0
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z0.device.type == "cuda"
    ):
        for stage in range(total_stages):
            first = max(0, stage - async_updates + 1)
            last = min(groups - 1, stage)
            z_stage, pred_stage = z, x_pred
            z_next, pred_next = z.clone(), x_pred.clone()
            for group in range(first, last + 1):
                local_step = stage - group
                local_t = local_step / groups
                z_full, pred_full = _ode_step(
                    model=model,
                    z=z_stage,
                    t=local_t,
                    t_next=local_t + dt,
                    x_pred_prev=pred_stage,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=sccfg,
                    cond_seq=cond_seq,
                    cond_seq_mask=cond_mask,
                )
                selected = group_of == group
                z_next = masked_replace(z_next, z_full, selected)
                pred_next = masked_replace(pred_next, pred_full, selected)
                model_calls += 1
            z = restore_cond(z_next, cond_seq, cond_mask)
            x_pred = restore_cond(pred_next, cond_seq, cond_mask)

        if refine_steps:
            start_t = async_updates / groups
            refine_grid = torch.linspace(
                start_t,
                1.0,
                refine_steps + 1,
                dtype=torch.float32,
                device=z.device,
            )
            for index in range(refine_steps):
                z, x_pred = _ode_step(
                    model=model,
                    z=z,
                    t=refine_grid[index].item(),
                    t_next=refine_grid[index + 1].item(),
                    x_pred_prev=x_pred,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=sccfg,
                    cond_seq=cond_seq,
                    cond_seq_mask=cond_mask,
                )
                model_calls += 1
    expected = groups * async_updates + refine_steps
    if model_calls != expected:
        raise AssertionError(f"expected {expected} calls, observed {model_calls}")
    return z, x_pred, model_calls


def vector_stats(left, right, position_mask):
    selected_left = left[:, position_mask].float().reshape(-1, left.shape[-1])
    selected_right = right[:, position_mask].float().reshape(-1, right.shape[-1])
    cosine = F.cosine_similarity(selected_left, selected_right, dim=-1)
    return {
        "n": int(cosine.numel()),
        "one_minus_cosine_sum": float((1.0 - cosine).sum().item()),
        "mse_sum": float(((selected_left - selected_right) ** 2).mean(-1).sum().item()),
        "left_norm_sum": float(selected_left.norm(dim=-1).sum().item()),
        "right_norm_sum": float(selected_right.norm(dim=-1).sum().item()),
    }


def add_stats(total, update, prefix):
    total[f"{prefix}_n"] += update["n"]
    for name in ("one_minus_cosine_sum", "mse_sum", "left_norm_sum", "right_norm_sum"):
        total[f"{prefix}_{name}"] += update[name]


@torch.no_grad()
def decoder_logits(x_pred, model):
    batch = x_pred.shape[0]
    ones = torch.ones(batch, dtype=x_pred.dtype, device=x_pred.device)
    model_input = torch.cat([x_pred, torch.zeros_like(x_pred)], dim=-1)
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=x_pred.device.type == "cuda"
    ):
        _, logits, _ = model(
            model_input,
            ones,
            deterministic=True,
            self_cond_cfg_scale=ones,
            decoder_step_active=True,
        )
    return logits


@torch.no_grad()
def diagnose(z0, model, groups, sccfg):
    """Measure target-clock and mixed-state discrepancies at four stages."""
    cfg = common.SamplingConfig()
    cond_seq, cond_mask = empty_condition(z0)
    group_of = balanced_group_map(z0.shape[1], groups, "ltr", z0.device, 0)
    dt = 1.0 / groups

    sync_states = []
    sync_preds = []
    z_sync = z0.clone()
    pred_sync = torch.zeros_like(z0)
    for local_step in range(groups):
        sync_states.append(z_sync.clone())
        sync_preds.append(pred_sync.clone())
        z_sync, pred_sync = _ode_step(
            model=model,
            z=z_sync,
            t=local_step / groups,
            t_next=(local_step + 1) / groups,
            x_pred_prev=pred_sync,
            config=cfg,
            cfg_scale=1.0,
            self_cond_cfg_scale=sccfg,
            cond_seq=cond_seq,
            cond_seq_mask=cond_mask,
        )

    totals = {
        **{f"{prefix}_n": 0 for prefix in ("clock", "state", "x_clock", "x_state")},
        **{
            f"{prefix}_{name}": 0.0
            for prefix in ("clock", "state", "x_clock", "x_state")
            for name in ("one_minus_cosine_sum", "mse_sum", "left_norm_sum", "right_norm_sum")
        },
        "kl_clock_sum": 0.0,
        "kl_clock_n": 0,
    }
    selected_stages = {0, groups - 1, groups, 2 * groups - 2}
    z = z0.clone()
    x_pred = torch.zeros_like(z0)
    for stage in range(2 * groups - 1):
        first = max(0, stage - groups + 1)
        last = min(groups - 1, stage)
        z_stage, pred_stage = z, x_pred
        z_next, pred_next = z.clone(), x_pred.clone()
        shared_z = shared_pred = None
        if stage in selected_stages:
            shared_t = stage / (2.0 * groups)
            shared_z, shared_pred = _ode_step(
                model=model,
                z=z_stage,
                t=shared_t,
                t_next=min(shared_t + dt, 1.0),
                x_pred_prev=pred_stage,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            v_shared = (shared_z - z_stage) / dt

        for group in range(first, last + 1):
            local_step = stage - group
            local_t = local_step / groups
            local_z, local_pred = _ode_step(
                model=model,
                z=z_stage,
                t=local_t,
                t_next=local_t + dt,
                x_pred_prev=pred_stage,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            selected = group_of == group
            z_next = masked_replace(z_next, local_z, selected)
            pred_next = masked_replace(pred_next, local_pred, selected)
            if stage not in selected_stages:
                continue

            sync_z_next, sync_pred = _ode_step(
                model=model,
                z=sync_states[local_step],
                t=local_t,
                t_next=local_t + dt,
                x_pred_prev=sync_preds[local_step],
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            v_local = (local_z - z_stage) / dt
            v_sync = (sync_z_next - sync_states[local_step]) / dt
            add_stats(totals, vector_stats(v_shared, v_local, selected), "clock")
            add_stats(totals, vector_stats(v_local, v_sync, selected), "state")
            add_stats(totals, vector_stats(shared_pred, local_pred, selected), "x_clock")
            add_stats(totals, vector_stats(local_pred, sync_pred, selected), "x_state")

            logits_local = decoder_logits(local_pred, model)[:, selected].float()
            logits_shared = decoder_logits(shared_pred, model)[:, selected].float()
            log_p_local = F.log_softmax(logits_local, dim=-1)
            log_p_shared = F.log_softmax(logits_shared, dim=-1)
            kl = (log_p_local.exp() * (log_p_local - log_p_shared)).sum(-1)
            totals["kl_clock_sum"] += float(kl.sum().item())
            totals["kl_clock_n"] += int(kl.numel())
        z, x_pred = z_next, pred_next

    result = {"n_position_events": totals["clock_n"]}
    for prefix in ("clock", "state", "x_clock", "x_state"):
        count = max(totals[f"{prefix}_n"], 1)
        result[f"E_{prefix}"] = totals[f"{prefix}_one_minus_cosine_sum"] / count
        result[f"MSE_{prefix}"] = totals[f"{prefix}_mse_sum"] / count
        result[f"norm_left_{prefix}"] = totals[f"{prefix}_left_norm_sum"] / count
        result[f"norm_right_{prefix}"] = totals[f"{prefix}_right_norm_sum"] / count
    result["KL_clock"] = totals["kl_clock_sum"] / max(totals["kl_clock_n"], 1)
    return result


def run_arm(name, z0, model, t_steps, args):
    if name == "standard":
        z, x_pred = common.standard_ode(z0, model, t_steps, args.sccfg)
        return z, x_pred, args.n_steps
    if name == "pipeline_shared":
        return pipeline_shared(z0, model, args.groups, args.sccfg)
    if name == "pipeline_local":
        return pipeline_local(z0, model, args.groups, args.sccfg, seed=args.seed)
    if name == "pipeline_local_refine4":
        return pipeline_local(
            z0, model, args.groups, args.sccfg, refine_steps=4, seed=args.seed
        )
    if name == "pipeline_local_refine8":
        return pipeline_local(
            z0, model, args.groups, args.sccfg, refine_steps=8, seed=args.seed
        )
    if name == "pipeline_local_rtl":
        return pipeline_local(
            z0, model, args.groups, args.sccfg, order="rtl", seed=args.seed
        )
    if name == "pipeline_local_random":
        return pipeline_local(
            z0, model, args.groups, args.sccfg, order="random", seed=args.seed
        )
    raise ValueError(name)


def main():
    args = parse_args()
    if args.n_steps != 2 * args.groups:
        raise ValueError("EXP-70 fixes n_steps = 2 * groups")
    if args.n_seq <= 0 or args.batch_size <= 0:
        raise ValueError("n_seq and batch_size must be positive")
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
    all_z0 = args.noise_scale * torch.randn(
        args.n_seq,
        args.max_length,
        common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    t_steps = get_sampling_steps(args.n_steps, time_schedule="uniform", device=device)

    results = {}
    for arm in args.arms:
        texts = []
        call_count = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        for start in range(0, args.n_seq, args.batch_size):
            z, _, calls = run_arm(
                arm, all_z0[start:start + args.batch_size], model, t_steps, args
            )
            if call_count is None:
                call_count = calls
            elif call_count != calls:
                raise AssertionError("model call count changed across batches")
            ids = common.decode(z, model, device)
            texts.extend(common.decode_texts(ids.cpu(), elf_tokenizer))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        metrics = common.text_metrics(
            texts,
            ppl_model,
            ppl_tokenizer,
            device,
            max_length=args.max_length,
        )
        metrics.update(
            {
                "model_calls": call_count,
                "wall_seconds": elapsed,
                "seconds_per_sequence": elapsed / args.n_seq,
                "samples": texts[:4],
                "texts": texts,
            }
        )
        results[arm] = metrics
        print(
            f"{arm:<24} PPL={metrics['ppl']:.1f} D1={metrics['d1']:.3f} "
            f"D2={metrics['d2']:.3f} deg={metrics['degeneration_rate']:.3f} "
            f"calls={call_count} sec={elapsed:.1f}"
        )

    diagnostics = None
    if not args.skip_diagnostics and args.diagnostic_samples > 0:
        diagnostics = diagnose(
            all_z0[: min(args.diagnostic_samples, args.n_seq)],
            model,
            args.groups,
            args.sccfg,
        )
        print("diagnostics=" + json.dumps(diagnostics, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "seed": args.seed,
                "n_seq": args.n_seq,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "groups": args.groups,
                "n_steps": args.n_steps,
                "noise_scale": args.noise_scale,
                "sccfg": args.sccfg,
                "arms": args.arms,
                "results": results,
                "diagnostics": diagnostics,
            },
            handle,
            indent=2,
        )
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
