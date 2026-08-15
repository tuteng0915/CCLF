#!/usr/bin/env python3
"""EXP-112 final: frozen unconditional Two-Wave-New confirmation."""

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import robust_revisable_commit_exp78 as exp78  # noqa: E402
import subset_selector_headroom_exp93 as exp93  # noqa: E402
import two_wave_unlock_exp111 as exp111  # noqa: E402
import unified_method_eval_exp64 as common  # noqa: E402
import unlock_trigger_headroom_exp108 as exp108  # noqa: E402
from modules.model import ELF_B  # noqa: E402
from utils.sampling_utils import get_sampling_steps  # noqa: E402


ARMS = ("standard", "fixed40", "fixed40_sham45", "two_wave_new")
OUT_DIR = Path("results/exp112_two_wave_confirmation")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n_uncond", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--high_confidence", type=float, default=.90)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--label", default="unconditional_final")
    return parser.parse_args()


def quality_delta(candidate, reference):
    return {
        "d1": candidate["d1"] - reference["d1"],
        "d2": candidate["d2"] - reference["d2"],
        "rep4": candidate["rep4"] - reference["rep4"],
        "degeneration_rate": candidate["degeneration_rate"] - reference["degeneration_rate"],
    }


def main():
    args = parse_args()
    if args.n_steps != 32 or args.max_length != 128:
        raise ValueError("EXP-112 freezes length-128 uniform ODE-32")
    args.n_cond = args.n_uncond
    args.prefix_length = 0
    args.degeneration_tolerance = .015625
    args.owt_offset = -1
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    checkpoint_path = REPO_ROOT / exp78.CHECKPOINTS["baseline"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(args.max_length))
    model.load_state_dict(common.load_weights(checkpoint), strict=False)
    model.to(device).eval()

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z0 = args.noise_scale * torch.randn(
        args.n_uncond,
        args.max_length,
        common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    cond_seq, cond_mask = common.empty_condition(z0)
    grid = get_sampling_steps(args.n_steps, "uniform", device=device)
    exp111.ARMS = ARMS
    records, native_agreement, actual_events = exp111.generate(
        model, tokenizer, z0, cond_seq, cond_mask, args, grid
    )

    model.cpu(); del model, checkpoint
    torch.cuda.empty_cache()
    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    evaluator = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()

    summaries, nlls, counts = {}, {}, {}
    empty_prompts = [""] * args.n_uncond
    for arm, record in records.items():
        print(f"evaluating {arm}", flush=True)
        nlls[arm], counts[arm] = exp93.conditional_sequence_nlls(
            empty_prompts, record["texts"], evaluator, ppl_tokenizer, device
        )
        summaries[arm] = common.text_metrics(
            record["texts"], evaluator, ppl_tokenizer, device,
            max_length=args.max_length,
        )
        summaries[arm]["ppl"] = exp93.aggregate_ppl(nlls[arm], counts[arm])
        summaries[arm].update({
            "wave1_fraction": sum(record["wave1"]) / args.n_uncond,
            "wave2_fraction": sum(record["wave2"]) / args.n_uncond,
            "overlap_fraction": sum(record["overlap"]) / args.n_uncond,
            "revision_rate": (
                record["revision_numerator"] / max(record["revision_denominator"], 1)
            ),
            "mean_readout_calls": sum(record["calls"]) / len(record["calls"]),
            "denoiser_calls": args.n_steps,
        })

    sham_agreement = sum(
        a == b for a, b in zip(records["fixed40"]["texts"], records["fixed40_sham45"]["texts"])
    ) / args.n_uncond
    fixed, wave = summaries["fixed40"], summaries["two_wave_new"]
    delta = nlls["two_wave_new"] - nlls["fixed40"]
    interval = exp108.bootstrap(delta, args.bootstrap_samples, args.seed + 112)
    q_delta = quality_delta(wave, fixed)
    improvement = 100 * (fixed["ppl"] - wave["ppl"]) / fixed["ppl"]
    gate = (
        improvement >= 2.0
        and interval[1] < 0.0
        and q_delta["d1"] >= -.005
        and q_delta["rep4"] <= .005
        and q_delta["degeneration_rate"] <= .015625
        and wave["wave2_fraction"] >= .05
        and sham_agreement == 1.0
        and native_agreement == 1.0
    )
    result = {
        **vars(args),
        "checkpoint": "baseline",
        "checkpoint_path": str(checkpoint_path),
        "actual_event_times": actual_events,
        "native_fixed40_reference_agreement": native_agreement,
        "fixed40_sham_text_agreement": sham_agreement,
        "summaries": summaries,
        "two_wave_new_vs_fixed40": {
            "improvement_pct": improvement,
            "mean_delta_nats": float(delta.mean()),
            "mean_delta_ci95": interval,
            "quality_delta": q_delta,
            "gate_passed": gate,
        },
        "per_sequence": {
            "nll": {arm: nlls[arm].tolist() for arm in ARMS},
            "counts": {arm: counts[arm].tolist() for arm in ARMS},
        },
        "texts": {arm: records[arm]["texts"] for arm in ARMS},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_seed{args.seed}.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "summaries": summaries,
        "two_wave_new_vs_fixed40": result["two_wave_new_vs_fixed40"],
        "sham_agreement": sham_agreement,
    }, indent=2), flush=True)
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
