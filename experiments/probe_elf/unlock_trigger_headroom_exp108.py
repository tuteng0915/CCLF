#!/usr/bin/env python3
"""EXP-108: paired per-trajectory Unlock-4 trigger-time headroom on ELF ODE."""

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

import paired_conditional_revalidation_exp80 as exp80  # noqa: E402
import robust_revisable_commit_exp78 as exp78  # noqa: E402
import subset_selector_headroom_exp93 as exp93  # noqa: E402
import unified_method_eval_exp64 as common  # noqa: E402
from modules.model import ELF_B  # noqa: E402
from modules.t5_encoder import get_encoder  # noqa: E402
from utils.sampling_utils import get_sampling_steps  # noqa: E402


OUT_DIR = Path("results/exp108_unlock_trigger_headroom")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=exp78.CHECKPOINTS, default="baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--owt_offset", type=int, default=40000)
    parser.add_argument("--n_cond", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--trigger_times", type=float, nargs="+", default=(.25,.30,.35,.40,.45,.50,.55,.60))
    parser.add_argument("--fixed_trigger", type=float, default=.40)
    parser.add_argument("--high_confidence", type=float, default=.90)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--label", default="discovery")
    return parser.parse_args()


def bootstrap(delta, samples, seed):
    generator = torch.Generator().manual_seed(seed + 1080041)
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(0, len(delta), (len(delta),), generator=generator)
        estimates[index] = delta[rows].mean()
    return [
        float(torch.quantile(estimates, .025)),
        float(torch.quantile(estimates, .975)),
    ]


def quality_delta(candidate, reference):
    return {
        "d1": candidate["d1"] - reference["d1"],
        "d2": candidate["d2"] - reference["d2"],
        "rep4": candidate["rep4"] - reference["rep4"],
        "degeneration_rate": candidate["degeneration_rate"]
        - reference["degeneration_rate"],
        "prompt_gain_nats": candidate["prompt_gain_nats"]
        - reference["prompt_gain_nats"],
    }


def quality_gate(delta):
    return (
        delta["d1"] >= -.005
        and delta["rep4"] <= .005
        and delta["degeneration_rate"] <= .015
        and delta["prompt_gain_nats"] >= -.01
    )


def summarize(texts, references, prompts, shuffled_prompts, evaluator, tokenizer, device):
    nll, counts = exp93.conditional_sequence_nlls(
        prompts, texts, evaluator, tokenizer, device
    )
    shuffled_nll, shuffled_counts = exp93.conditional_sequence_nlls(
        shuffled_prompts, texts, evaluator, tokenizer, device
    )
    metrics = exp93.summarize_texts(texts, references, nll, counts)
    shuffled_ppl = exp93.aggregate_ppl(shuffled_nll, shuffled_counts)
    metrics["shuffled_prompt_ppl"] = shuffled_ppl
    metrics["prompt_gain_nats"] = math.log(shuffled_ppl) - math.log(
        metrics["prompt_conditioned_ppl"]
    )
    return metrics, nll, counts, shuffled_nll


@torch.no_grad()
def generate(model, tokenizer, cond_z0, cond_seq, cond_mask, prefix_ids, args, grid):
    names = ["standard"] + [f"trigger_{value:.2f}" for value in args.trigger_times]
    records = {name: {"texts": [], "fractions": [], "calls": []} for name in names}
    reference_agreement = None

    for batch_index, start in enumerate(range(0, args.n_cond, args.batch_size)):
        end = min(start + args.batch_size, args.n_cond)
        z0 = cond_z0[start:end]
        seq = cond_seq[start:end]
        mask = cond_mask[start:end]
        standard_z, standard_info = exp78.rollout(
            z0, model, grid, args, "standard", 0, seq, mask
        )
        standard_ids = common.decode(standard_z, model, z0.device)
        expected_prefix = prefix_ids[start:end].to(standard_ids.device)
        if not (standard_ids[:, : args.prefix_length] == expected_prefix).all():
            raise RuntimeError("standard arm changed the observed prefix")
        records["standard"]["texts"].extend(
            common.decode_texts(standard_ids.cpu(), tokenizer, args.prefix_length)
        )
        records["standard"]["fractions"].append(standard_info["anchor_fraction"])
        records["standard"]["calls"].append(standard_info["readout_calls"])

        if batch_index == 0:
            duplicate_z, _ = exp78.rollout(
                z0, model, grid, args, "standard", 0, seq, mask
            )
            duplicate_ids = common.decode(duplicate_z, model, z0.device)
            reference_agreement = float((duplicate_ids == standard_ids).float().mean())
            if reference_agreement != 1.0:
                raise RuntimeError(f"standard reference agreement={reference_agreement}")

        for trigger in args.trigger_times:
            args.commit_time = trigger
            z, info = exp78.rollout(z0, model, grid, args, "unlock4", 0, seq, mask)
            ids = common.decode(z, model, z0.device)
            if not (ids[:, : args.prefix_length] == expected_prefix).all():
                raise RuntimeError(f"trigger {trigger:.2f} changed the observed prefix")
            name = f"trigger_{trigger:.2f}"
            records[name]["texts"].extend(
                common.decode_texts(ids.cpu(), tokenizer, args.prefix_length)
            )
            records[name]["fractions"].append(info["anchor_fraction"])
            records[name]["calls"].append(info["readout_calls"])
        print(f"generation completed {end}/{args.n_cond}", flush=True)

    actual_times = {
        f"{value:.2f}": next(float(t) for t in grid[1:] if float(t) >= value)
        for value in args.trigger_times
    }
    return records, reference_agreement, actual_times


def main():
    args = parse_args()
    if args.n_steps != 32 or args.fixed_trigger not in args.trigger_times:
        raise ValueError("EXP-108 fixes ODE-32 and requires fixed trigger in candidates")
    if len(set(args.trigger_times)) != len(args.trigger_times):
        raise ValueError("trigger times must be unique")
    args.sampler = "ode"
    args.read_time = .30
    args.stable_confidence = .60
    args.sde_gamma = 1.5
    args.p_mean = -.8
    args.p_std = .8

    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    checkpoint_path = REPO_ROOT / exp78.CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(args.max_length))
    model.load_state_dict(common.load_weights(checkpoint), strict=False)
    model.to(device).eval()

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    args.conditional_dataset = "owt"
    pairs = exp80.load_pairs(args, tokenizer)
    cond_seq, cond_mask, references = common.build_condition_data(
        pairs, tokenizer, encoder, device, args.max_length, args.prefix_length
    )
    prefix_ids = exp80.prefix_targets(pairs, args.prefix_length, device)
    prompts = common.decode_texts(prefix_ids.cpu(), tokenizer)
    shuffled_prompts = prompts[1:] + prompts[:1]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cond_z0 = args.noise_scale * torch.randn(
        args.n_cond,
        args.max_length,
        common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    cond_z0[:, : args.prefix_length] = cond_seq[:, : args.prefix_length].to(
        cond_z0.dtype
    )
    grid = get_sampling_steps(args.n_steps, "uniform", device=device)
    records, agreement, actual_times = generate(
        model, tokenizer, cond_z0, cond_seq, cond_mask, prefix_ids, args, grid
    )

    model.cpu(); encoder.cpu(); del model, encoder, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    evaluator = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()

    summaries, nlls, counts, shuffled = {}, {}, {}, {}
    for name, record in records.items():
        print(f"evaluating {name}", flush=True)
        summaries[name], nlls[name], counts[name], shuffled[name] = summarize(
            record["texts"], references, prompts, shuffled_prompts,
            evaluator, ppl_tokenizer, device,
        )
        summaries[name]["anchor_fraction"] = sum(record["fractions"]) / len(record["fractions"])
        summaries[name]["mean_readout_calls"] = sum(record["calls"]) / len(record["calls"])

    candidate_names = [f"trigger_{value:.2f}" for value in args.trigger_times]
    trigger_nll = torch.stack([nlls[name] for name in candidate_names])
    trigger_counts = torch.stack([counts[name] for name in candidate_names])
    best_index = trigger_nll.argmin(dim=0)
    rows = torch.arange(args.n_cond)
    best_nll = trigger_nll[best_index, rows]
    best_counts = trigger_counts[best_index, rows]
    best_texts = [
        records[candidate_names[int(best_index[row])]]["texts"][row]
        for row in range(args.n_cond)
    ]
    best_shuffled, best_shuffled_counts = exp93.conditional_sequence_nlls(
        shuffled_prompts, best_texts, evaluator, ppl_tokenizer, device
    )
    oracle = exp93.summarize_texts(best_texts, references, best_nll, best_counts)
    oracle["shuffled_prompt_ppl"] = exp93.aggregate_ppl(
        best_shuffled, best_shuffled_counts
    )
    oracle["prompt_gain_nats"] = math.log(oracle["shuffled_prompt_ppl"]) - math.log(
        oracle["prompt_conditioned_ppl"]
    )
    fixed_name = f"trigger_{args.fixed_trigger:.2f}"
    delta = best_nll - nlls[fixed_name]
    interval = bootstrap(delta, args.bootstrap_samples, args.seed)
    q_delta = quality_delta(oracle, summaries[fixed_name])
    improvement = 100 * (
        summaries[fixed_name]["prompt_conditioned_ppl"] - oracle["prompt_conditioned_ppl"]
    ) / summaries[fixed_name]["prompt_conditioned_ppl"]
    headroom = {
        "fixed_ppl": summaries[fixed_name]["prompt_conditioned_ppl"],
        "oracle_ppl": oracle["prompt_conditioned_ppl"],
        "improvement_pct": improvement,
        "mean_delta_nats": float(delta.double().mean()),
        "mean_delta_ci95": interval,
        "quality_delta": q_delta,
        "gate_passed": improvement >= 5.0 and interval[1] < 0.0 and quality_gate(q_delta),
    }
    aggregate_fixed = min(
        candidate_names,
        key=lambda name: summaries[name]["prompt_conditioned_ppl"],
    )
    payload = {
        **vars(args),
        "checkpoint_path": str(checkpoint_path),
        "oracle_is_deployable": False,
        "native_reference_agreement": agreement,
        "actual_trigger_times": actual_times,
        "summaries": summaries,
        "oracle": oracle,
        "oracle_vs_fixed": headroom,
        "best_aggregate_fixed": aggregate_fixed,
        "winner_histogram": {
            f"{value:.2f}": int((best_index == index).sum())
            for index, value in enumerate(args.trigger_times)
        },
        "per_sequence": {
            "trigger_nll": trigger_nll.tolist(),
            "trigger_counts": trigger_counts.tolist(),
            "best_index": best_index.tolist(),
            "best_nll": best_nll.tolist(),
        },
        "texts": {name: record["texts"] for name, record in records.items()},
        "oracle_texts": best_texts,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "oracle_vs_fixed": headroom,
        "best_aggregate_fixed": aggregate_fixed,
        "winner_histogram": payload["winner_histogram"],
    }, indent=2), flush=True)
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()
