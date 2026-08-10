#!/usr/bin/env python3
"""Timing audit for EXP-78 Standard versus four-step revisable anchoring."""

import argparse
import json
from pathlib import Path

import torch

import robust_revisable_commit_exp78 as exp78
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


OUT_DIR = Path("results/exp78_robust_revisable_commit")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=exp78.CHECKPOINTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--commit_time", type=float, default=0.40)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--lock_steps", type=int, default=4)
    parser.add_argument("--label", default="timing")
    return parser.parse_args()


@torch.no_grad()
def traced_rollout(z0, model, grid, args, unlock):
    cond_seq, cond_mask = common.empty_condition(z0)
    base_seq, base_mask = cond_seq.clone(), cond_mask.clone()
    active_seq, active_mask = cond_seq.clone(), cond_mask.clone()
    z = z0.clone()
    x_pred = torch.zeros_like(z)
    trace = []
    selected = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)
    commit_ids = torch.full(z.shape[:2], -1, dtype=torch.long, device=z.device)
    committed = False
    release_index = None
    release_trace_index = None

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for index in range(grid.shape[0] - 1):
            if release_index is not None and index >= release_index:
                active_seq = base_seq.clone()
                active_mask = base_mask.clone()
                release_trace_index = len(trace)
                release_index = None

            z, x_pred = _ode_step(
                model=model,
                z=z,
                t=grid[index].item(),
                t_next=grid[index + 1].item(),
                x_pred_prev=x_pred,
                config=common.SamplingConfig(),
                cfg_scale=1.0,
                self_cond_cfg_scale=args.sccfg,
                cond_seq=active_seq,
                cond_seq_mask=active_mask,
            )
            token_ids, confidence = exp78.lexical_readout(x_pred, model)

            if unlock and not committed and grid[index + 1].item() >= args.commit_time:
                selected = (confidence >= args.confidence) & (base_mask < 0.5)
                commit_ids = token_ids.clone()
                active_seq = torch.where(
                    selected.unsqueeze(-1), x_pred.detach(), active_seq
                )
                active_mask = torch.maximum(
                    active_mask, selected.to(active_mask.dtype)
                )
                z = restore_cond(z, active_seq, active_mask)
                x_pred = restore_cond(x_pred, active_seq, active_mask)
                committed = True
                release_index = index + 1 + args.lock_steps

            trace.append(token_ids.cpu())

    return {
        "trace": torch.stack(trace, dim=1),
        "final_ids": common.decode(z, model, z.device).cpu(),
        "selected": selected.cpu(),
        "commit_ids": commit_ids.cpu(),
        "release_trace_index": release_trace_index,
    }


def timing_metrics(trace, final_ids, times):
    matches = trace == final_ids[:, None, :]
    any_match = matches.any(dim=1)
    first_index = matches.float().argmax(dim=1)
    first = torch.where(any_match, times[first_index], torch.ones_like(first_index).float())

    stable_windows = []
    for index in range(trace.shape[1]):
        if index + 3 <= trace.shape[1]:
            stable_windows.append(matches[:, index : index + 3].all(dim=1))
        else:
            stable_windows.append(torch.zeros_like(matches[:, 0]))
    stable_windows = torch.stack(stable_windows, dim=1)
    any_stable = stable_windows.any(dim=1)
    stable_index = stable_windows.float().argmax(dim=1)
    stable = torch.where(
        any_stable, times[stable_index], torch.ones_like(stable_index).float()
    )
    revisions = (trace[:, 1:] != trace[:, :-1]).sum(dim=1).float()
    return {"first": first, "stable": stable, "revisions": revisions}


def masked_mean(value, mask):
    chosen = value[mask]
    return float(chosen.float().mean().item()) if chosen.numel() else float("nan")


def summarize(metrics, selected):
    groups = {
        "all": torch.ones_like(selected, dtype=torch.bool),
        "selected": selected,
        "unselected": ~selected,
    }
    return {
        group: {name: masked_mean(value, mask) for name, value in metrics.items()}
        for group, mask in groups.items()
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    checkpoint_path = common.REPO_ROOT / exp78.CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(args.max_length))
    model.load_state_dict(common.load_weights(checkpoint), strict=False)
    model.to(device).eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    z0 = args.noise_scale * torch.randn(
        args.n_samples,
        args.max_length,
        512,
        generator=generator,
        device=device,
    )
    grid = get_sampling_steps(args.n_steps, "uniform", device=device)

    records = {"standard": [], "unlock4": []}
    for start in range(0, args.n_samples, args.batch_size):
        end = min(start + args.batch_size, args.n_samples)
        records["standard"].append(
            traced_rollout(z0[start:end], model, grid, args, unlock=False)
        )
        records["unlock4"].append(
            traced_rollout(z0[start:end], model, grid, args, unlock=True)
        )

    branches = {}
    for name, parts in records.items():
        branches[name] = {
            key: torch.cat([part[key] for part in parts])
            for key in ("trace", "final_ids", "selected", "commit_ids")
        }
        branches[name]["release_trace_index"] = parts[0]["release_trace_index"]

    selected = branches["unlock4"]["selected"]
    times = grid[1:].cpu()
    raw = {
        name: timing_metrics(branch["trace"], branch["final_ids"], times)
        for name, branch in branches.items()
    }
    timing = {name: summarize(metrics, selected) for name, metrics in raw.items()}
    deltas = {
        group: {
            metric: timing["unlock4"][group][metric]
            - timing["standard"][group][metric]
            for metric in ("first", "stable", "revisions")
        }
        for group in ("all", "selected", "unselected")
    }

    release_index = branches["unlock4"]["release_trace_index"]
    unlock_trace = branches["unlock4"]["trace"]
    post_release_revisions = (
        (unlock_trace[:, release_index + 1 :] != unlock_trace[:, release_index:-1])
        .sum(dim=1)
        .float()
        if release_index is not None and release_index + 1 < unlock_trace.shape[1]
        else torch.zeros_like(selected, dtype=torch.float)
    )
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "max_length": args.max_length,
        "n_steps": args.n_steps,
        "noise_scale": args.noise_scale,
        "sccfg": args.sccfg,
        "commit_time": args.commit_time,
        "confidence": args.confidence,
        "lock_steps": args.lock_steps,
        "anchor_fraction": float(selected.float().mean().item()),
        "release_time": float(times[release_index].item()) if release_index is not None else None,
        "timing": timing,
        "unlock_minus_standard": deltas,
        "endpoint_agreement": {
            group: masked_mean(
                branches["unlock4"]["final_ids"]
                == branches["standard"]["final_ids"],
                selected if group == "selected" else ~selected,
            )
            for group in ("selected", "unselected")
        },
        "selected_anchor_changed_by_endpoint": masked_mean(
            branches["unlock4"]["final_ids"]
            != branches["unlock4"]["commit_ids"],
            selected,
        ),
        "selected_post_release_revisions": masked_mean(
            post_release_revisions, selected
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
