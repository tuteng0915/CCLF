#!/usr/bin/env python3
"""Mechanism audit for confidence-gated within-trajectory commitment.

The script forks a natural ODE trajectory at the frozen commit checkpoint into
true-anchor and frequency/confidence-matched shuffled-anchor continuations.
It measures token timing, revisions, immediate endpoint-margin effects, and
final text quality on paired trajectories.
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
    empty_condition,
    load_weights,
    model_config,
    restore_cond,
    text_metrics,
)


AUDIT_CHECKPOINTS = ("baseline", "ct_control", "kd_early")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=AUDIT_CHECKPOINTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--commit_time", type=float, default=0.40)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--label", default="pilot")
    return parser.parse_args()


@torch.no_grad()
def lexical_top_conf(x_pred, model):
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
    top_value, top_id = logits.max(dim=-1)
    confidence = (top_value - torch.logsumexp(logits, dim=-1)).exp()
    return top_id, confidence


@torch.no_grad()
def natural_rollout(z0, model, t_steps, args):
    from utils.sampling_utils import _ode_step

    cond_seq, cond_mask = empty_condition(z0)
    z = z0.clone()
    x_pred = torch.zeros_like(z)
    token_trace = []
    commit = None
    first_post_x = None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for index in range(t_steps.shape[0] - 1):
            z, x_pred = _ode_step(
                model=model,
                z=z,
                t=t_steps[index].item(),
                t_next=t_steps[index + 1].item(),
                x_pred_prev=x_pred,
                config=SamplingConfig(),
                cfg_scale=1.0,
                self_cond_cfg_scale=args.sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            top_id, confidence = lexical_top_conf(x_pred, model)
            token_trace.append(top_id.cpu())
            if commit is None and t_steps[index + 1].item() >= args.commit_time:
                selected = confidence >= args.confidence
                commit = {
                    "step": index,
                    "z": z.detach().cpu(),
                    "x_pred": x_pred.detach().cpu(),
                    "selected": selected.cpu(),
                    "confidence": confidence.cpu(),
                    "token": top_id.cpu(),
                }
            elif commit is not None and index == commit["step"] + 1:
                first_post_x = x_pred.detach().cpu()

    final_ids = decode(z, model, z.device).cpu()
    return {
        "trace": torch.stack(token_trace, dim=1),
        "commit": commit,
        "first_post_x": first_post_x,
        "final_ids": final_ids,
        "final_z": z.detach().cpu(),
    }


@torch.no_grad()
def anchored_continuation(
    z_commit,
    x_commit,
    anchor_seq,
    selected,
    prefix_trace,
    commit_step,
    model,
    t_steps,
    args,
):
    from utils.sampling_utils import _ode_step

    active_mask = selected.to(dtype=z_commit.dtype)
    z = restore_cond(z_commit.clone(), anchor_seq, active_mask)
    x_pred = restore_cond(x_commit.clone(), anchor_seq, active_mask)
    token_trace = [value.clone() for value in prefix_trace.unbind(dim=1)]
    first_post_x = None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        top_id, _ = lexical_top_conf(x_pred, model)
        token_trace.append(top_id.cpu())
        for index in range(commit_step + 1, t_steps.shape[0] - 1):
            z, x_pred = _ode_step(
                model=model,
                z=z,
                t=t_steps[index].item(),
                t_next=t_steps[index + 1].item(),
                x_pred_prev=x_pred,
                config=SamplingConfig(),
                cfg_scale=1.0,
                self_cond_cfg_scale=args.sccfg,
                cond_seq=anchor_seq,
                cond_seq_mask=active_mask,
            )
            if first_post_x is None:
                first_post_x = x_pred.detach().cpu()
            top_id, _ = lexical_top_conf(x_pred, model)
            token_trace.append(top_id.cpu())

    return {
        "trace": torch.stack(token_trace, dim=1),
        "first_post_x": first_post_x,
        "final_ids": decode(z, model, z.device).cpu(),
        "final_z": z.detach().cpu(),
    }


def make_matched_shuffle(x_pred, selected, confidence, token_ids, frequency, seed):
    """Shuffle anchor vectors within confidence/frequency quartiles."""
    output = torch.zeros_like(x_pred)
    flat_selected = selected.flatten()
    slots = flat_selected.nonzero(as_tuple=False).squeeze(1)
    flat_x = x_pred.reshape(-1, x_pred.shape[-1])
    flat_conf = confidence.flatten()[slots]
    flat_token = token_ids.flatten()[slots]
    flat_freq = frequency[flat_token].float().clamp_min(1).log()
    generator = torch.Generator().manual_seed(seed)

    def bins(values):
        if values.numel() < 4:
            return torch.zeros_like(values, dtype=torch.long)
        boundaries = torch.quantile(values.float(), torch.tensor([0.25, 0.50, 0.75]))
        return torch.bucketize(values.float(), boundaries)

    conf_bin = bins(flat_conf)
    freq_bin = bins(flat_freq)
    donor = torch.arange(slots.numel())
    for cbin in range(4):
        for fbin in range(4):
            group = ((conf_bin == cbin) & (freq_bin == fbin)).nonzero(
                as_tuple=False
            ).squeeze(1)
            if group.numel() > 1:
                order = group[torch.randperm(group.numel(), generator=generator)]
                donor[order] = torch.roll(order, shifts=1)

    singleton = donor == torch.arange(slots.numel())
    singleton_ids = singleton.nonzero(as_tuple=False).squeeze(1)
    for index in singleton_ids.tolist():
        distance = (flat_conf - flat_conf[index]).abs()
        distance += (flat_freq - flat_freq[index]).abs() / (
            flat_freq.std().clamp_min(1e-6)
        )
        distance[index] = float("inf")
        different_token = flat_token != flat_token[index]
        if different_token.any():
            distance = torch.where(
                different_token, distance, torch.full_like(distance, float("inf"))
            )
        donor[index] = distance.argmin()

    flat_output = output.reshape(-1, output.shape[-1])
    flat_output[slots] = flat_x[slots[donor]]
    same_token = (flat_token == flat_token[donor]).float().mean().item()
    return output, {
        "n_anchors": int(slots.numel()),
        "same_token_fraction": same_token,
        "mean_confidence_mismatch": float(
            (flat_conf - flat_conf[donor]).abs().mean().item()
        ),
        "mean_log_frequency_mismatch": float(
            (flat_freq - flat_freq[donor]).abs().mean().item()
        ),
    }


@torch.no_grad()
def endpoint_margin(x_pred, endpoint_ids, model, batch_size, device):
    margins = []
    for start in range(0, x_pred.shape[0], batch_size):
        batch_x = x_pred[start : start + batch_size].to(device)
        batch_y = endpoint_ids[start : start + batch_size].to(device)
        size = batch_x.shape[0]
        z_in = torch.cat([batch_x, torch.zeros_like(batch_x)], dim=-1)
        ones = torch.ones(size, dtype=batch_x.dtype, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, logits, _ = model(
                z_in,
                ones,
                deterministic=True,
                self_cond_cfg_scale=ones,
                decoder_step_active=True,
            )
        logits = logits.float()
        target = logits.gather(-1, batch_y.unsqueeze(-1)).squeeze(-1)
        top2_value, top2_id = logits.topk(2, dim=-1)
        competitor = torch.where(
            top2_id[..., 0] == batch_y, top2_value[..., 1], top2_value[..., 0]
        )
        margins.append((target - competitor).cpu())
    return torch.cat(margins, dim=0)


def timing_metrics(trace, final_ids, times):
    matches = trace == final_ids[:, None, :]
    any_match = matches.any(dim=1)
    first_index = matches.float().argmax(dim=1)
    first = times[first_index]
    first = torch.where(any_match, first, torch.ones_like(first))

    stable_windows = []
    for index in range(trace.shape[1]):
        if index + 3 <= trace.shape[1]:
            window = matches[:, index : index + 3].all(dim=1)
        else:
            window = torch.zeros_like(matches[:, 0])
        stable_windows.append(window)
    stable_windows = torch.stack(stable_windows, dim=1)
    any_stable = stable_windows.any(dim=1)
    stable_index = stable_windows.float().argmax(dim=1)
    stable = times[stable_index]
    stable = torch.where(any_stable, stable, torch.ones_like(stable))
    revisions = (trace[:, 1:] != trace[:, :-1]).sum(dim=1).float()
    return {"first": first, "stable": stable, "revisions": revisions}


def masked_mean(values, mask):
    selected = values[mask].float()
    return float(selected.mean().item()) if selected.numel() else float("nan")


def summarize_timing(metrics, selected):
    groups = {
        "all": torch.ones_like(selected, dtype=torch.bool),
        "selected": selected,
        "unresolved": ~selected,
    }
    return {
        group: {
            key: masked_mean(value, mask)
            for key, value in metrics.items()
        }
        for group, mask in groups.items()
    }


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    from modules.model import ELF_B
    from unified_method_eval_exp64 import REPO_ROOT
    from utils.sampling_utils import get_sampling_steps

    device = torch.device(args.device)
    SamplingConfig.denoiser_noise_scale = args.noise_scale
    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**model_config(args.max_length))
    model.load_state_dict(load_weights(checkpoint), strict=False)
    model.to(device).eval()
    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    z0 = args.noise_scale * torch.randn(
        args.n_samples,
        args.max_length,
        512,
        generator=generator,
        device=device,
    )
    t_steps = get_sampling_steps(
        args.n_steps, time_schedule="uniform", device=device
    )

    natural_parts = []
    for start in range(0, args.n_samples, args.batch_size):
        natural_parts.append(
            natural_rollout(z0[start : start + args.batch_size], model, t_steps, args)
        )

    natural_trace = torch.cat([part["trace"] for part in natural_parts])
    natural_final = torch.cat([part["final_ids"] for part in natural_parts])
    natural_z = torch.cat([part["final_z"] for part in natural_parts])
    natural_first_post = torch.cat([part["first_post_x"] for part in natural_parts])
    commit_step = natural_parts[0]["commit"]["step"]
    z_commit = torch.cat([part["commit"]["z"] for part in natural_parts])
    x_commit = torch.cat([part["commit"]["x_pred"] for part in natural_parts])
    selected = torch.cat([part["commit"]["selected"] for part in natural_parts])
    confidence = torch.cat([part["commit"]["confidence"] for part in natural_parts])
    commit_token = torch.cat([part["commit"]["token"] for part in natural_parts])
    frequency = torch.bincount(
        natural_final.flatten(), minlength=32100
    ).clamp_min(1)
    shuffled_anchor, shuffle_audit = make_matched_shuffle(
        x_commit,
        selected,
        confidence,
        commit_token,
        frequency,
        args.seed + 1009,
    )
    true_anchor = torch.where(selected.unsqueeze(-1), x_commit, torch.zeros_like(x_commit))

    branch_parts = {"true_anchor": [], "shuffled_anchor": []}
    for start in range(0, args.n_samples, args.batch_size):
        end = start + args.batch_size
        prefix = natural_trace[start:end, :commit_step]
        for name, anchors in (
            ("true_anchor", true_anchor),
            ("shuffled_anchor", shuffled_anchor),
        ):
            branch_parts[name].append(
                anchored_continuation(
                    z_commit[start:end].to(device),
                    x_commit[start:end].to(device),
                    anchors[start:end].to(device),
                    selected[start:end].to(device),
                    prefix,
                    commit_step,
                    model,
                    t_steps,
                    args,
                )
            )

    branches = {
        "natural": {
            "trace": natural_trace,
            "first_post_x": natural_first_post,
            "final_ids": natural_final,
            "final_z": natural_z,
        }
    }
    for name, parts in branch_parts.items():
        branches[name] = {
            key: torch.cat([part[key] for part in parts])
            for key in ("trace", "first_post_x", "final_ids", "final_z")
        }

    times = t_steps[1:].cpu()
    timing = {}
    for name, branch in branches.items():
        raw = timing_metrics(branch["trace"], branch["final_ids"], times)
        timing[name] = {
            "summary": summarize_timing(raw, selected),
            "raw": raw,
        }

    natural_endpoint_margins = {
        name: endpoint_margin(
            branch["first_post_x"], natural_final, model, args.batch_size, device
        )
        for name, branch in branches.items()
    }
    own_endpoint_margins = {
        name: endpoint_margin(
            branch["first_post_x"],
            branch["final_ids"],
            model,
            args.batch_size,
            device,
        )
        for name, branch in branches.items()
    }
    unresolved = ~selected

    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    ppl_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()

    quality = {}
    samples = {}
    for name, branch in branches.items():
        texts = decode_texts(branch["final_ids"], elf_tokenizer)
        quality[name] = text_metrics(
            texts, ppl_model, ppl_tokenizer, device, max_length=args.max_length
        )
        samples[name] = texts[:4]

    def timing_delta(key, group, left, right):
        mask = selected if group == "selected" else unresolved
        return masked_mean(timing[left]["raw"][key] - timing[right]["raw"][key], mask)

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "max_length": args.max_length,
        "n_steps": args.n_steps,
        "noise_scale": args.noise_scale,
        "sccfg": args.sccfg,
        "commit_time_requested": args.commit_time,
        "commit_time_actual": float(t_steps[commit_step + 1].item()),
        "confidence": args.confidence,
        "commit_fraction": float(selected.float().mean().item()),
        "shuffle_audit": shuffle_audit,
        "timing": {
            name: value["summary"] for name, value in timing.items()
        },
        "paired_timing_deltas": {
            "true_minus_natural_unresolved": {
                key: timing_delta(key, "unresolved", "true_anchor", "natural")
                for key in ("first", "stable", "revisions")
            },
            "shuffled_minus_natural_unresolved": {
                key: timing_delta(key, "unresolved", "shuffled_anchor", "natural")
                for key in ("first", "stable", "revisions")
            },
        },
        "first_postcommit_natural_endpoint_margin": {
            name: masked_mean(value, unresolved)
            for name, value in natural_endpoint_margins.items()
        },
        "first_postcommit_natural_endpoint_margin_delta": {
            "true_minus_natural": masked_mean(
                natural_endpoint_margins["true_anchor"]
                - natural_endpoint_margins["natural"],
                unresolved,
            ),
            "shuffled_minus_natural": masked_mean(
                natural_endpoint_margins["shuffled_anchor"]
                - natural_endpoint_margins["natural"],
                unresolved,
            ),
            "true_minus_shuffled": masked_mean(
                natural_endpoint_margins["true_anchor"]
                - natural_endpoint_margins["shuffled_anchor"],
                unresolved,
            ),
        },
        "first_postcommit_own_endpoint_margin": {
            name: masked_mean(value, unresolved)
            for name, value in own_endpoint_margins.items()
        },
        "first_postcommit_own_endpoint_margin_delta": {
            "true_minus_natural": masked_mean(
                own_endpoint_margins["true_anchor"]
                - own_endpoint_margins["natural"],
                unresolved,
            ),
            "shuffled_minus_natural": masked_mean(
                own_endpoint_margins["shuffled_anchor"]
                - own_endpoint_margins["natural"],
                unresolved,
            ),
            "true_minus_shuffled": masked_mean(
                own_endpoint_margins["true_anchor"]
                - own_endpoint_margins["shuffled_anchor"],
                unresolved,
            ),
        },
        "unresolved_endpoint_agreement_with_natural": {
            name: masked_mean(branch["final_ids"] == natural_final, unresolved)
            for name, branch in branches.items()
        },
        "quality": quality,
        "samples": samples,
    }

    output_dir = Path("results/exp67_hard_commit_mechanism")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        "commit_fraction": result["commit_fraction"],
        "timing_delta": result["paired_timing_deltas"],
        "natural_endpoint_margin_delta": result[
            "first_postcommit_natural_endpoint_margin_delta"
        ],
        "own_endpoint_margin_delta": result[
            "first_postcommit_own_endpoint_margin_delta"
        ],
        "endpoint_agreement": result["unresolved_endpoint_agreement_with_natural"],
        "quality": quality,
    }, indent=2))
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
