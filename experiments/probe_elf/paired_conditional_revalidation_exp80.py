#!/usr/bin/env python3
"""EXP-80: paired unconditional/conditional revalidation of ELF methods.

Every arm is evaluated on paired unconditional noise and on a fixed-prefix
continuation panel. Conditional schedules are defined only over free suffix
positions; observed prompt positions stay in ELF's native ``cond_seq`` path.
"""

import argparse
import json
import math
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

import canonical_context_wave_exp75 as exp75
import late_coupled_blocks_exp79 as exp79
import pipeline_factorization_exp70 as exp70
import robust_revisable_commit_exp78 as exp78
import soft_anchor_pipeline_exp71 as exp71
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from modules.t5_encoder import get_encoder
from utils.sampling_utils import get_sampling_steps


CHECKPOINTS = {
    name: common.CHECKPOINTS[name]
    for name in ("baseline", "ct_control", "kd_early")
}
ARMS = (
    "standard32",
    "standard64",
    "standard136",
    "pipeline_local_refine8",
    "soft_ltr",
    "soft_random",
    "canonical_ltr_refine8",
    "unlock4",
)
OUT_DIR = Path("results/exp80_paired_conditional")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_uncond", type=int, default=64)
    parser.add_argument("--n_cond", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--wave_start", type=float, default=0.15)
    parser.add_argument("--wave_end", type=float, default=0.75)
    parser.add_argument("--read_time", type=float, default=0.30)
    parser.add_argument("--commit_time", type=float, default=0.40)
    parser.add_argument("--high_confidence", type=float, default=0.90)
    parser.add_argument("--stable_confidence", type=float, default=0.60)
    parser.add_argument(
        "--conditional_dataset", choices=("owt", "gutenberg"), default="owt"
    )
    parser.add_argument("--owt_offset", type=int, default=10000)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--label", default="screen")
    return parser.parse_args()


def load_owt_pairs(n_pairs, prefix_length, suffix_length, offset):
    """Load a deterministic in-domain OWT panel.

    The released ELF dataset exposes only a train split, so this panel is
    deliberately labeled in-domain rather than train-disjoint/held-out.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "embedded-language-flows/openwebtext-t5",
        split="train",
        streaming=True,
    )
    total = prefix_length + suffix_length
    pairs = []
    for index, example in enumerate(dataset):
        if index < offset:
            continue
        token_ids = list(example["input_ids"])
        attention = example.get("attention_mask")
        if attention is not None:
            valid = int(sum(attention))
            token_ids = token_ids[:valid]
        if len(token_ids) < total:
            continue
        segment = token_ids[:total]
        pairs.append((segment[:prefix_length], segment[prefix_length:]))
        if len(pairs) == n_pairs:
            return pairs
    raise RuntimeError(f"requested {n_pairs} OWT pairs, found {len(pairs)}")


def load_pairs(args, tokenizer):
    suffix_length = args.max_length - args.prefix_length
    if args.conditional_dataset == "owt":
        return load_owt_pairs(
            args.n_cond,
            args.prefix_length,
            suffix_length,
            args.owt_offset,
        )
    return common.get_gutenberg_pairs(
        tokenizer,
        args.n_cond,
        args.prefix_length,
        suffix_length,
    )


def prefix_targets(pairs, prefix_length, device):
    targets = torch.zeros(
        len(pairs), prefix_length, dtype=torch.long, device=device
    )
    for row, (prefix, _) in enumerate(pairs):
        width = min(prefix_length, len(prefix))
        targets[row, :width] = torch.tensor(
            prefix[:width], dtype=torch.long, device=device
        )
    return targets


def arm_grid(steps, device):
    return get_sampling_steps(steps, "uniform", device=device)


@torch.no_grad()
def run_arm(arm, z0, model, args, cond_seq=None, cond_mask=None):
    eligible_mask = (
        cond_mask[0] < 0.5
        if cond_mask is not None
        else torch.ones(z0.shape[1], dtype=torch.bool, device=z0.device)
    )
    if arm.startswith("standard"):
        steps = int(arm[len("standard") :])
        z, _ = common.standard_ode(
            z0,
            model,
            arm_grid(steps, z0.device),
            args.sccfg,
            cond_seq,
            cond_mask,
        )
        return z, {"denoiser_calls": steps, "readout_calls": 0}
    if arm == "pipeline_local_refine8":
        z, _, calls = exp70.pipeline_local(
            z0,
            model,
            args.groups,
            args.sccfg,
            refine_steps=8,
            cond_seq=cond_seq,
            cond_mask=cond_mask,
            seed=args.seed,
            eligible_mask=eligible_mask,
        )
        return z, {"denoiser_calls": calls, "readout_calls": 0}
    if arm in ("soft_ltr", "soft_random"):
        z, _, info = exp71.two_forward_ode(
            z0,
            model,
            arm_grid(args.n_steps, z0.device),
            args.sccfg,
            arm,
            args,
            cond_seq,
            cond_mask,
        )
        return z, {
            "denoiser_calls": info["denoiser_calls"],
            "readout_calls": info["lexical_readout_calls"],
            "leader_fraction": info["mean_leader_fraction"],
        }
    if arm == "canonical_ltr_refine8":
        z, _, calls = exp75.canonical_pipeline(
            z0,
            model,
            args.groups,
            args.sccfg,
            refine_steps=8,
            seed=args.seed,
            cond_seq=cond_seq,
            cond_mask=cond_mask,
            eligible_mask=eligible_mask,
        )
        return z, {"denoiser_calls": calls, "readout_calls": 0}
    if arm == "unlock4":
        args.sampler = "ode"
        z, info = exp78.rollout(
            z0,
            model,
            arm_grid(args.n_steps, z0.device),
            args,
            "unlock4",
            step_noise_seed=0,
            cond_seq=cond_seq,
            cond_mask=cond_mask,
        )
        return z, {
            "denoiser_calls": args.n_steps,
            "readout_calls": info["readout_calls"],
            "anchor_fraction": info["anchor_fraction"],
        }
    raise ValueError(arm)


@torch.no_grad()
def generate_scope(
    arm,
    z0,
    model,
    tokenizer,
    args,
    cond_seq=None,
    cond_mask=None,
    prefix_targets_tensor=None,
):
    texts = []
    decoded_prefix = []
    clamp_errors = []
    info = None
    started = time.perf_counter()
    for start in range(0, z0.shape[0], args.batch_size):
        end = min(start + args.batch_size, z0.shape[0])
        batch_cond = cond_seq[start:end] if cond_seq is not None else None
        batch_mask = cond_mask[start:end] if cond_mask is not None else None
        z, batch_info = run_arm(
            arm, z0[start:end], model, args, batch_cond, batch_mask
        )
        if info is None:
            info = batch_info
        elif info["denoiser_calls"] != batch_info["denoiser_calls"]:
            raise AssertionError("denoiser call count changed across batches")
        ids = common.decode(z, model, z.device)
        suffix_start = args.prefix_length if cond_seq is not None else 0
        texts.extend(common.decode_texts(ids.cpu(), tokenizer, suffix_start))
        if cond_seq is not None:
            expected = prefix_targets_tensor[start:end].to(ids.device)
            decoded_prefix.extend(
                (ids[:, : args.prefix_length] == expected)
                .all(dim=1)
                .float()
                .cpu()
                .tolist()
            )
            selected = batch_mask.bool()
            clamp_errors.append(
                float((z[selected] - batch_cond[selected]).abs().max().item())
            )
    elapsed = time.perf_counter() - started
    info = dict(info or {})
    info.update(
        {
            "wall_seconds": elapsed,
            "seconds_per_sequence": elapsed / max(z0.shape[0], 1),
            "processed_token_calls": info.get("denoiser_calls", 0)
            * args.max_length,
            "decoded_prefix_agreement": (
                sum(decoded_prefix) / len(decoded_prefix)
                if decoded_prefix
                else float("nan")
            ),
            "max_prompt_clamp_error": max(clamp_errors, default=0.0),
        }
    )
    if info["max_prompt_clamp_error"] > 1e-6:
        raise RuntimeError(
            f"prompt clamp gate failed for {arm}: "
            f"{info['max_prompt_clamp_error']:.6g}"
        )
    return texts, info


def add_quality_metrics(
    texts,
    info,
    evaluator,
    ppl_tokenizer,
    device,
    max_length,
):
    if evaluator is None:
        metrics = exp79.text_metrics_without_ppl(texts)
    else:
        metrics = common.text_metrics(
            texts, evaluator, ppl_tokenizer, device, max_length=max_length
        )
    metrics.update(info)
    metrics["samples"] = texts[:4]
    metrics["texts"] = texts
    return metrics


def main():
    args = parse_args()
    if args.n_steps != 2 * args.groups:
        raise ValueError("n_steps must equal 2 * groups")
    if not 0 < args.prefix_length < args.max_length:
        raise ValueError("prefix_length must lie inside the sequence")
    if min(args.n_uncond, args.n_cond, args.batch_size) <= 0:
        raise ValueError("sample counts and batch_size must be positive")
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale

    from transformers import T5Tokenizer

    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = ELF_B(**common.model_config(args.max_length))
    missing, unexpected = model.load_state_dict(
        common.load_weights(checkpoint), strict=False
    )
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder.to(device).eval()

    pairs = load_pairs(args, tokenizer)
    if len(pairs) != args.n_cond:
        raise RuntimeError(f"requested {args.n_cond} pairs, got {len(pairs)}")
    cond_seq, cond_mask, references = common.build_condition_data(
        pairs,
        tokenizer,
        encoder,
        device,
        args.max_length,
        args.prefix_length,
    )
    targets = prefix_targets(pairs, args.prefix_length, device)
    prompts = common.decode_texts(targets.cpu(), tokenizer)

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
        uncond_texts, uncond_info = generate_scope(
            arm,
            noise[: args.n_uncond],
            model,
            tokenizer,
            args,
        )
        print(f"[{arm}] conditional", flush=True)
        cond_texts, cond_info = generate_scope(
            arm,
            cond_noise,
            model,
            tokenizer,
            args,
            cond_seq,
            cond_mask,
            targets,
        )
        if uncond_info["denoiser_calls"] != cond_info["denoiser_calls"]:
            raise AssertionError(f"paired call mismatch for {arm}")
        generated[arm] = {
            "unconditional": (uncond_texts, uncond_info),
            "conditional": (cond_texts, cond_info),
        }

    model.cpu()
    encoder.cpu()
    del model, encoder, checkpoint
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

    shuffled_prompts = prompts[1:] + prompts[:1]
    results = {}
    for arm, scopes in generated.items():
        uncond_texts, uncond_info = scopes["unconditional"]
        cond_texts, cond_info = scopes["conditional"]
        uncond = add_quality_metrics(
            uncond_texts,
            uncond_info,
            evaluator,
            ppl_tokenizer,
            device,
            args.max_length,
        )
        cond = add_quality_metrics(
            cond_texts,
            cond_info,
            evaluator,
            ppl_tokenizer,
            device,
            args.max_length - args.prefix_length,
        )
        cond["rouge_l"] = sum(
            common.rouge_l_f1(hypothesis, reference)
            for hypothesis, reference in zip(cond_texts, references)
        ) / len(references)
        if evaluator is None:
            true_ppl = shuffled_ppl = float("nan")
        else:
            true_ppl = exp79.conditional_boundary_ppl(
                prompts,
                cond_texts,
                evaluator,
                ppl_tokenizer,
                device,
                suffix_tokens=None,
                max_length=1024,
            )
            shuffled_ppl = exp79.conditional_boundary_ppl(
                shuffled_prompts,
                cond_texts,
                evaluator,
                ppl_tokenizer,
                device,
                suffix_tokens=None,
                max_length=1024,
            )
        cond["prompt_conditioned_ppl"] = true_ppl
        cond["shuffled_prompt_ppl"] = shuffled_ppl
        cond["prompt_gain_nats"] = (
            math.log(shuffled_ppl) - math.log(true_ppl)
            if true_ppl == true_ppl and shuffled_ppl == shuffled_ppl
            else float("nan")
        )
        results[arm] = {"unconditional": uncond, "conditional": cond}

    compact = {}
    for arm, scopes in results.items():
        u, c = scopes["unconditional"], scopes["conditional"]
        compact[arm] = {
            "u_ppl": u["ppl"],
            "u_d1": u["d1"],
            "u_d2": u["d2"],
            "u_rep4": u["rep4"],
            "u_deg": u["degeneration_rate"],
            "c_suffix_ppl": c["ppl"],
            "c_prompt_ppl": c["prompt_conditioned_ppl"],
            "c_shuffled_prompt_ppl": c["shuffled_prompt_ppl"],
            "c_prompt_gain_nats": c["prompt_gain_nats"],
            "c_rouge_l": c["rouge_l"],
            "c_d1": c["d1"],
            "c_d2": c["d2"],
            "c_rep4": c["rep4"],
            "c_deg": c["degeneration_rate"],
            "c_decoded_prefix": c["decoded_prefix_agreement"],
            "clamp_error": c["max_prompt_clamp_error"],
            "calls": c["denoiser_calls"],
            "readouts": c["readout_calls"],
            "token_calls": c["processed_token_calls"],
        }
    print(json.dumps(compact, indent=2))

    output = {
        **vars(args),
        "checkpoint_path": str(checkpoint_path),
        "conditional_panel_note": (
            "OWT train-split in-domain panel; not guaranteed train-disjoint"
            if args.conditional_dataset == "owt"
            else "Gutenberg out-of-domain panel"
        ),
        "paired_suffix_noise": True,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
