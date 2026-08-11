#!/usr/bin/env python3
"""EXP-82: paired U/C screen of transition, anchor density, and lock duration."""

import argparse
import json
import math
import re
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

import late_coupled_blocks_exp79 as exp79
import paired_conditional_revalidation_exp80 as exp80
import robust_revisable_commit_exp78 as exp78
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from modules.t5_encoder import get_encoder
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


CHECKPOINTS = {
    name: common.CHECKPOINTS[name]
    for name in ("baseline", "ct_control", "kd_early")
}
DEFAULT_ARMS = (
    "standard32",
    "readout_t40",
    "topq_t30_q25_h4", "topq_t30_q50_h4", "topq_t30_q75_h4", "topq_t30_q875_h4",
    "topq_t40_q25_h4", "topq_t40_q50_h4", "topq_t40_q75_h4", "topq_t40_q875_h4",
    "topq_t50_q25_h4", "topq_t50_q50_h4", "topq_t50_q75_h4", "topq_t50_q875_h4",
    "topq_t40_q50_h1", "topq_t40_q50_h8",
    "topq_t40_q75_h1", "topq_t40_q75_h8",
    "random_t40_q50_h4", "shuffled_t40_q50_h4",
)
CELL_RE = re.compile(
    r"^(?P<mode>topq|random|shuffled)_t(?P<time>\d+)_q(?P<density>\d+)_h(?P<horizon>\d+)$"
)
READOUT_RE = re.compile(r"^readout_t(?P<time>\d+)$")
OUT_DIR = Path("results/exp82_transition_unlock_pareto")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_uncond", type=int, default=32)
    parser.add_argument("--n_cond", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument(
        "--conditional_dataset", choices=("owt", "gutenberg"), default="owt"
    )
    parser.add_argument("--owt_offset", type=int, default=13000)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--label", default="screen")
    return parser.parse_args()


def parse_arm(arm):
    if arm == "standard32":
        return {"mode": "standard", "time": None, "density": 0.0, "horizon": 0}
    match = READOUT_RE.match(arm)
    if match:
        return {
            "mode": "readout",
            "time": int(match.group("time")) / 100,
            "density": 0.0,
            "horizon": 0,
        }
    match = CELL_RE.match(arm)
    if not match:
        raise ValueError(f"invalid arm name: {arm}")
    density_digits = match.group("density")
    density = int(density_digits) / (1000 if len(density_digits) == 3 else 100)
    cell = {
        "mode": match.group("mode"),
        "time": int(match.group("time")) / 100,
        "density": density,
        "horizon": int(match.group("horizon")),
    }
    if not 0 < cell["density"] < 1 or not 0 < cell["time"] < 1:
        raise ValueError(f"invalid arm values: {arm}")
    return cell


def exact_budget_mask(scores, eligible, fraction):
    selected = torch.zeros_like(eligible, dtype=torch.bool)
    for row in range(scores.shape[0]):
        candidates = torch.nonzero(eligible[row], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        count = max(1, min(candidates.numel(), int(round(fraction * candidates.numel()))))
        order = torch.topk(scores[row, candidates], count, largest=True).indices
        selected[row, candidates[order]] = True
    return selected


@torch.no_grad()
def rollout(z0, model, grid, args, arm, rng_seed, cond_seq=None, cond_mask=None):
    cell = parse_arm(arm)
    if cell["mode"] == "standard":
        z, _ = common.standard_ode(
            z0, model, grid, args.sccfg, cond_seq, cond_mask
        )
        return z, {
            "denoiser_calls": args.n_steps,
            "readout_calls": 0,
            "anchor_fraction": 0.0,
            "anchor_confidence": float("nan"),
        }

    cfg = common.SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = common.empty_condition(z0)
    base_seq, base_mask = cond_seq.clone(), cond_mask.clone()
    active_seq, active_mask = cond_seq.clone(), cond_mask.clone()
    z = restore_cond(z0.clone(), active_seq, active_mask)
    x_pred = restore_cond(torch.zeros_like(z), active_seq, active_mask)
    triggered = False
    release_index = None
    selected_total = 0
    eligible_total = int((base_mask < 0.5).sum().item())
    selected_confidence_sum = 0.0
    readout_calls = 0
    random_generator = torch.Generator(device=z.device).manual_seed(rng_seed)

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for index in range(grid.shape[0] - 1):
            if release_index is not None and index >= release_index:
                active_seq = base_seq.clone()
                active_mask = base_mask.clone()
                release_index = None

            z, x_pred = _ode_step(
                model=model,
                z=z,
                t=grid[index].item(),
                t_next=grid[index + 1].item(),
                x_pred_prev=x_pred,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=args.sccfg,
                cond_seq=active_seq,
                cond_seq_mask=active_mask,
            )
            t_next = grid[index + 1].item()
            if not triggered and t_next >= cell["time"]:
                _, confidence = exp78.lexical_readout(x_pred, model)
                readout_calls += 1
                triggered = True
                if cell["mode"] == "readout":
                    continue
                eligible = base_mask < 0.5
                scores = confidence
                if cell["mode"] == "random":
                    scores = torch.rand(
                        confidence.shape,
                        generator=random_generator,
                        device=confidence.device,
                    )
                selected = exact_budget_mask(scores, eligible, cell["density"])
                selected_total = int(selected.sum().item())
                selected_confidence_sum = float(confidence[selected].sum().item())
                content = x_pred.detach()
                if cell["mode"] == "shuffled":
                    if z.shape[0] < 2:
                        raise ValueError("shuffled-content control requires batch_size >= 2")
                    content = content.roll(1, dims=0)
                active_seq = torch.where(
                    selected.unsqueeze(-1), content, active_seq
                )
                active_mask = torch.maximum(
                    active_mask, selected.to(active_mask.dtype)
                )
                z = restore_cond(z, active_seq, active_mask)
                x_pred = restore_cond(x_pred, active_seq, active_mask)
                release_index = index + 1 + cell["horizon"]

    return z, {
        "denoiser_calls": args.n_steps,
        "readout_calls": readout_calls,
        "anchor_fraction": selected_total / max(eligible_total, 1),
        "anchor_confidence": (
            selected_confidence_sum / selected_total
            if selected_total
            else float("nan")
        ),
        "trigger_time": cell["time"],
        "lock_horizon": cell["horizon"],
        "selection_mode": cell["mode"],
    }


@torch.no_grad()
def generate_scope(
    arm,
    z0,
    model,
    tokenizer,
    args,
    cond_seq=None,
    cond_mask=None,
    prefix_targets=None,
):
    texts, decoded_prefix, clamp_errors, infos = [], [], [], []
    started = time.perf_counter()
    grid = get_sampling_steps(args.n_steps, "uniform", device=z0.device)
    for batch_index, start in enumerate(range(0, z0.shape[0], args.batch_size)):
        end = min(start + args.batch_size, z0.shape[0])
        batch_seq = cond_seq[start:end] if cond_seq is not None else None
        batch_mask = cond_mask[start:end] if cond_mask is not None else None
        z, info = rollout(
            z0[start:end],
            model,
            grid,
            args,
            arm,
            args.seed * 100003 + batch_index * 7919,
            batch_seq,
            batch_mask,
        )
        infos.append((end - start, info))
        ids = common.decode(z, model, z.device)
        suffix_start = args.prefix_length if cond_seq is not None else 0
        texts.extend(common.decode_texts(ids.cpu(), tokenizer, suffix_start))
        if cond_seq is not None:
            expected = prefix_targets[start:end].to(ids.device)
            decoded_prefix.extend(
                (ids[:, : args.prefix_length] == expected)
                .all(dim=1).float().cpu().tolist()
            )
            selected_prompt = batch_mask.bool()
            clamp_errors.append(
                float((z[selected_prompt] - batch_seq[selected_prompt]).abs().max().item())
            )
    elapsed = time.perf_counter() - started
    total = sum(count for count, _ in infos)
    first = infos[0][1]
    for _, info in infos[1:]:
        if info["denoiser_calls"] != first["denoiser_calls"] or info["readout_calls"] != first["readout_calls"]:
            raise AssertionError(f"call count changed across batches for {arm}")
    result = dict(first)
    averaged_keys = ["anchor_fraction", "anchor_confidence"]
    if "release_fraction" in first:
        averaged_keys.append("release_fraction")
    for key in averaged_keys:
        values = [
            (count, info[key]) for count, info in infos
            if info[key] == info[key]
        ]
        result[key] = (
            sum(count * value for count, value in values) / sum(count for count, _ in values)
            if values else float("nan")
        )
    result.update({
        "wall_seconds": elapsed,
        "seconds_per_sequence": elapsed / max(total, 1),
        "processed_token_calls": result["denoiser_calls"] * args.max_length,
        "decoded_prefix_agreement": (
            sum(decoded_prefix) / len(decoded_prefix) if decoded_prefix else float("nan")
        ),
        "max_prompt_clamp_error": max(clamp_errors, default=0.0),
    })
    if result["max_prompt_clamp_error"] > 1e-6:
        raise RuntimeError(f"prompt clamp failed for {arm}")
    return texts, result


def main():
    args = parse_args()
    for arm in args.arms:
        parse_arm(arm)
    if args.n_steps != 32:
        raise ValueError("EXP-82 fixes ODE-32")
    if not 0 < args.prefix_length < args.max_length:
        raise ValueError("prefix_length must lie inside max_length")
    if "shuffled_t40_q50_h4" in args.arms and args.batch_size < 2:
        raise ValueError("shuffled control requires batch_size >= 2")
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    from transformers import T5Tokenizer

    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(args.max_length))
    missing, unexpected = model.load_state_dict(common.load_weights(checkpoint), strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder.to(device).eval()

    pairs = exp80.load_pairs(args, tokenizer)
    cond_seq, cond_mask, references = common.build_condition_data(
        pairs, tokenizer, encoder, device, args.max_length, args.prefix_length
    )
    targets = exp80.prefix_targets(pairs, args.prefix_length, device)
    prompts = common.decode_texts(targets.cpu(), tokenizer)
    shuffled_prompts = prompts[1:] + prompts[:1]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    total = max(args.n_uncond, args.n_cond)
    noise = args.noise_scale * torch.randn(
        total,
        args.max_length,
        common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    cond_noise = noise[: args.n_cond].clone()
    cond_noise[:, : args.prefix_length] = cond_seq[:, : args.prefix_length]

    generated = {}
    for arm in args.arms:
        print(f"[{arm}] unconditional", flush=True)
        u_texts, u_info = generate_scope(
            arm, noise[: args.n_uncond], model, tokenizer, args
        )
        print(f"[{arm}] conditional", flush=True)
        c_texts, c_info = generate_scope(
            arm, cond_noise, model, tokenizer, args,
            cond_seq, cond_mask, targets,
        )
        if (u_info["denoiser_calls"], u_info["readout_calls"]) != (c_info["denoiser_calls"], c_info["readout_calls"]):
            raise AssertionError(f"paired call mismatch for {arm}")
        generated[arm] = (u_texts, u_info, c_texts, c_info)

    model.cpu(); encoder.cpu(); del model, encoder, checkpoint
    torch.cuda.empty_cache()
    evaluator = ppl_tokenizer = None
    if not args.skip_ppl:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        if ppl_tokenizer.pad_token is None:
            ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
            ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
        evaluator = AutoModelForCausalLM.from_pretrained(
            "gpt2-large", torch_dtype=torch.bfloat16
        ).to(device).eval()

    results, compact = {}, {}
    for arm, (u_texts, u_info, c_texts, c_info) in generated.items():
        u = exp80.add_quality_metrics(
            u_texts, u_info, evaluator, ppl_tokenizer, device, args.max_length
        )
        c = exp80.add_quality_metrics(
            c_texts, c_info, evaluator, ppl_tokenizer, device,
            args.max_length - args.prefix_length,
        )
        c["rouge_l"] = sum(
            common.rouge_l_f1(hypothesis, reference)
            for hypothesis, reference in zip(c_texts, references)
        ) / len(references)
        if evaluator is None:
            true_ppl = shuffled_ppl = float("nan")
        else:
            true_ppl = exp79.conditional_boundary_ppl(
                prompts, c_texts, evaluator, ppl_tokenizer, device,
                suffix_tokens=None, max_length=1024,
            )
            shuffled_ppl = exp79.conditional_boundary_ppl(
                shuffled_prompts, c_texts, evaluator, ppl_tokenizer, device,
                suffix_tokens=None, max_length=1024,
            )
        c["prompt_conditioned_ppl"] = true_ppl
        c["shuffled_prompt_ppl"] = shuffled_ppl
        c["prompt_gain_nats"] = (
            math.log(shuffled_ppl) - math.log(true_ppl)
            if true_ppl == true_ppl and shuffled_ppl == shuffled_ppl
            else float("nan")
        )
        results[arm] = {"unconditional": u, "conditional": c}
        compact[arm] = {
            "u_ppl": u["ppl"], "u_d1": u["d1"], "u_rep4": u["rep4"],
            "u_deg": u["degeneration_rate"], "c_ppl": c["prompt_conditioned_ppl"],
            "c_gain": c["prompt_gain_nats"], "c_rl": c["rouge_l"],
            "c_d1": c["d1"], "c_rep4": c["rep4"],
            "c_deg": c["degeneration_rate"],
            "anchor_fraction": c["anchor_fraction"],
            "anchor_confidence": c["anchor_confidence"],
            "calls": c["denoiser_calls"], "readouts": c["readout_calls"],
            "clamp": c["max_prompt_clamp_error"],
        }
    print(json.dumps(compact, indent=2))
    output = {
        **vars(args),
        "checkpoint_path": str(checkpoint_path),
        "paired_suffix_noise": True,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
