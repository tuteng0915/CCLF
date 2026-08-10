#!/usr/bin/env python3
"""EXP-78: paired ODE/SDE evaluation of persistent and revisable commitment."""

import argparse
import json
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

import unified_method_eval_exp64 as common
from modules.model import ELF_B
from modules.t5_encoder import get_encoder
from utils.sampling_utils import _ode_step, _sde_step, get_sampling_steps, restore_cond


CHECKPOINTS = {
    name: common.CHECKPOINTS[name]
    for name in ("baseline", "ct_control", "kd_early")
}
ARMS = ("standard", "hard_highconf", "hard_stable", "unlock4", "unlock8")
OUT_DIR = Path("results/exp78_robust_revisable_commit")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--sampler", choices=("ode", "sde"), default="ode")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_uncond", type=int, default=128)
    parser.add_argument("--n_cond", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--read_time", type=float, default=0.30)
    parser.add_argument("--commit_time", type=float, default=0.40)
    parser.add_argument("--high_confidence", type=float, default=0.90)
    parser.add_argument("--stable_confidence", type=float, default=0.60)
    parser.add_argument("--sde_gamma", type=float, default=1.5)
    parser.add_argument("--p_mean", type=float, default=-0.8)
    parser.add_argument("--p_std", type=float, default=0.8)
    parser.add_argument("--label", default="formal")
    return parser.parse_args()


@torch.no_grad()
def lexical_readout(x_pred, model):
    batch = x_pred.shape[0]
    model_input = torch.cat([x_pred.detach(), torch.zeros_like(x_pred)], dim=-1)
    ones = torch.ones(batch, dtype=x_pred.dtype, device=x_pred.device)
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
    probabilities = F.softmax(logits.float(), dim=-1)
    confidence, token_ids = probabilities.max(-1)
    return token_ids, confidence


def batch_grid(args, device, batch_index):
    if args.sampler == "ode":
        return get_sampling_steps(args.n_steps, "uniform", device=device)
    device_index = device.index if device.index is not None else 0
    with torch.random.fork_rng(devices=[device_index]):
        torch.cuda.manual_seed(args.seed * 100003 + 1709 + batch_index)
        return get_sampling_steps(
            args.n_steps,
            "logit_normal",
            P_mean=args.p_mean,
            P_std=args.p_std,
            device=device,
        )


@torch.no_grad()
def rollout(
    z0,
    model,
    grid,
    args,
    arm,
    step_noise_seed,
    cond_seq=None,
    cond_mask=None,
):
    cfg = common.SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = common.empty_condition(z0)
    base_seq, base_mask = cond_seq.clone(), cond_mask.clone()
    active_seq, active_mask = cond_seq.clone(), cond_mask.clone()
    z = restore_cond(z0.clone(), active_seq, active_mask)
    x_pred = restore_cond(torch.zeros_like(z), active_seq, active_mask)
    previous_ids = None
    read_recorded = False
    committed = False
    selected_total = 0
    eligible_total = int((base_mask < 0.5).sum().item())
    release_index = None
    readout_calls = 0
    device_index = z.device.index if z.device.index is not None else 0

    context = (
        torch.random.fork_rng(devices=[device_index])
        if args.sampler == "sde"
        else torch.random.fork_rng(devices=[])
    )
    with context:
        if args.sampler == "sde":
            torch.cuda.manual_seed(step_noise_seed)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
        ):
            for index in range(grid.shape[0] - 1):
                if release_index is not None and index >= release_index:
                    active_seq = base_seq.clone()
                    active_mask = base_mask.clone()
                    release_index = None

                t = grid[index].item()
                t_next = grid[index + 1].item()
                step_kwargs = dict(
                    model=model,
                    z=z,
                    t=t,
                    t_next=t_next,
                    x_pred_prev=x_pred,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=args.sccfg,
                    cond_seq=active_seq,
                    cond_seq_mask=active_mask,
                )
                if args.sampler == "sde" and index < grid.shape[0] - 2:
                    z, x_pred = _sde_step(
                        **step_kwargs, gamma=args.sde_gamma, generator=None
                    )
                else:
                    z, x_pred = _ode_step(**step_kwargs)

                needs_stability = arm == "hard_stable"
                if (
                    needs_stability
                    and not read_recorded
                    and t_next >= args.read_time
                    and t_next < args.commit_time
                ):
                    previous_ids, _ = lexical_readout(x_pred, model)
                    readout_calls += 1
                    read_recorded = True

                if arm != "standard" and not committed and t_next >= args.commit_time:
                    token_ids, confidence = lexical_readout(x_pred, model)
                    readout_calls += 1
                    if needs_stability:
                        selected = (
                            (confidence >= args.stable_confidence)
                            & (token_ids == previous_ids)
                            if previous_ids is not None
                            else torch.zeros_like(confidence, dtype=torch.bool)
                        )
                    else:
                        selected = confidence >= args.high_confidence
                    selected &= base_mask < 0.5
                    selected_total = int(selected.sum().item())
                    active_seq = torch.where(
                        selected.unsqueeze(-1), x_pred.detach(), active_seq
                    )
                    active_mask = torch.maximum(
                        active_mask, selected.to(active_mask.dtype)
                    )
                    z = restore_cond(z, active_seq, active_mask)
                    x_pred = restore_cond(x_pred, active_seq, active_mask)
                    committed = True
                    if arm == "unlock4":
                        release_index = index + 1 + 4
                    elif arm == "unlock8":
                        release_index = index + 1 + 8

    return z, {
        "anchor_fraction": selected_total / max(eligible_total, 1),
        "readout_calls": readout_calls,
    }


@torch.no_grad()
def evaluate(
    z0,
    model,
    args,
    elf_tokenizer,
    ppl_model,
    ppl_tokenizer,
    cond_seq=None,
    cond_mask=None,
    prefix_ids=None,
    suffix_start=0,
    references=None,
):
    outputs = {arm: {"texts": [], "fractions": [], "prefix": [], "calls": []} for arm in args.arms}
    for batch_index, start in enumerate(range(0, z0.shape[0], args.batch_size)):
        end = min(start + args.batch_size, z0.shape[0])
        batch_z0 = z0[start:end]
        batch_seq = cond_seq[start:end] if cond_seq is not None else None
        batch_mask = cond_mask[start:end] if cond_mask is not None else None
        grid = batch_grid(args, z0.device, batch_index)
        noise_seed = args.seed * 100003 + 7919 + batch_index
        for arm in args.arms:
            z, info = rollout(
                batch_z0,
                model,
                grid,
                args,
                arm,
                noise_seed,
                batch_seq,
                batch_mask,
            )
            ids = common.decode(z, model, z.device)
            outputs[arm]["texts"].extend(
                common.decode_texts(ids.cpu(), elf_tokenizer, suffix_start)
            )
            outputs[arm]["fractions"].append(info["anchor_fraction"])
            outputs[arm]["calls"].append(info["readout_calls"])
            if prefix_ids is not None:
                expected = prefix_ids[start:end].to(ids.device)
                outputs[arm]["prefix"].extend(
                    (ids[:, : expected.shape[1]] == expected).all(dim=1).float().cpu().tolist()
                )

    results = {}
    for arm, records in outputs.items():
        metrics = common.text_metrics(
            records["texts"],
            ppl_model,
            ppl_tokenizer,
            z0.device,
            max_length=max(args.max_length - suffix_start, 2),
        )
        if references is not None:
            metrics["rouge_l"] = sum(
                common.rouge_l_f1(hypothesis, reference)
                for hypothesis, reference in zip(records["texts"], references)
            ) / max(len(references), 1)
        if records["prefix"]:
            metrics["exact_prefix"] = sum(records["prefix"]) / len(records["prefix"])
        metrics["anchor_fraction"] = sum(records["fractions"]) / len(records["fractions"])
        metrics["denoiser_calls"] = args.n_steps
        metrics["mean_readout_calls"] = sum(records["calls"]) / len(records["calls"])
        metrics["texts"] = records["texts"]
        results[arm] = metrics
    return results


def prefix_targets(pairs, prefix_length, device):
    targets = torch.zeros(len(pairs), prefix_length, dtype=torch.long, device=device)
    for row, (prefix, _) in enumerate(pairs):
        width = min(prefix_length, len(prefix))
        targets[row, :width] = torch.tensor(prefix[:width], device=device)
    return targets


def main():
    args = parse_args()
    if args.max_length <= args.prefix_length:
        raise ValueError("prefix_length must be smaller than max_length")
    if args.sampler == "sde" and args.n_steps != 32:
        raise ValueError("native SDE fidelity fixes n_steps=32")
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
    total = max(args.n_uncond, args.n_cond)
    z0 = args.noise_scale * torch.randn(
        total, args.max_length, common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator, device=device,
    )

    unconditional = evaluate(
        z0[: args.n_uncond], model, args, elf_tokenizer, ppl_model, ppl_tokenizer
    )
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    pairs = common.get_gutenberg_pairs(
        elf_tokenizer,
        args.n_cond,
        args.prefix_length,
        args.max_length - args.prefix_length,
    )
    cond_seq, cond_mask, references = common.build_condition_data(
        pairs,
        elf_tokenizer,
        encoder,
        device,
        args.max_length,
        args.prefix_length,
    )
    cond_z0 = z0[: args.n_cond].clone()
    cond_z0[:, : args.prefix_length] = cond_seq[:, : args.prefix_length].to(cond_z0.dtype)
    targets = prefix_targets(pairs, args.prefix_length, device)
    conditioned = evaluate(
        cond_z0,
        model,
        args,
        elf_tokenizer,
        ppl_model,
        ppl_tokenizer,
        cond_seq,
        cond_mask,
        targets,
        args.prefix_length,
        references,
    )

    for arm in args.arms:
        u, c = unconditional[arm], conditioned[arm]
        print(
            f"{arm:<14} PPL={u['ppl']:.1f} D1={u['d1']:.3f} D2={u['d2']:.3f} "
            f"deg={u['degeneration_rate']:.3f} anchor={u['anchor_fraction']:.3f} "
            f"condPPL={c['ppl']:.1f} RL={c['rouge_l']:.3f} prefix={c['exact_prefix']:.3f}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_{args.sampler}_{args.checkpoint}_seed{args.seed}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "checkpoint_path": str(checkpoint_path),
                "paired_step_noise": args.sampler == "sde",
                "results": {
                    arm: {
                        "unconditional": unconditional[arm],
                        "conditioned": conditioned[arm],
                    }
                    for arm in args.arms
                },
            },
            handle,
            indent=2,
        )
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
