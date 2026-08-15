#!/usr/bin/env python3
"""EXP-110 Stage B: calibrate one frozen late-trigger fallback threshold."""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import paired_conditional_revalidation_exp80 as exp80  # noqa: E402
import subset_selector_headroom_exp93 as exp93  # noqa: E402
import unified_method_eval_exp64 as common  # noqa: E402
import unlock_trigger_headroom_exp108 as exp108  # noqa: E402


DELAY_COUNTS = (8, 12, 16, 20, 24, 28, 32)
SIGNAL = "shadow_confidence_response"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headroom_json", type=Path, required=True)
    parser.add_argument("--features_json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def reconstruct_panel(payload, tokenizer):
    data_args = SimpleNamespace(
        conditional_dataset="owt",
        n_cond=int(payload["n_cond"]),
        max_length=int(payload["max_length"]),
        prefix_length=int(payload["prefix_length"]),
        owt_offset=int(payload["owt_offset"]),
    )
    pairs = exp80.load_pairs(data_args, tokenizer)
    prefix_ids = exp80.prefix_targets(pairs, data_args.prefix_length, "cpu")
    prompts = common.decode_texts(prefix_ids, tokenizer)
    references = [
        tokenizer.decode(suffix, skip_special_tokens=True)
        for _, suffix in pairs
    ]
    return prompts, prompts[1:] + prompts[:1], references


def threshold_for_top_k(scores, k):
    ordered = torch.sort(scores, descending=True).values
    if k >= len(ordered):
        return float(ordered[-1] - 1e-12)
    return float((ordered[k - 1] + ordered[k]) / 2)


def main():
    args = parse_args()
    headroom = json.loads(args.headroom_json.read_text())
    feature_payload = json.loads(args.features_json.read_text())
    if len(feature_payload["banks"]) != 1:
        raise ValueError("Stage-B feature file must contain exactly one calibration bank")
    bank = feature_payload["banks"][0]
    if int(bank["seed"]) != int(headroom["seed"]):
        raise ValueError("feature and headroom seeds do not match")
    if SIGNAL not in bank["features"]:
        raise ValueError("calibration feature file lacks the frozen Stage-A signal")
    for agreement in bank["output_agreement"].values():
        if agreement != 1.0:
            raise ValueError("features were not extracted from exactly reproduced arms")

    names = [f"trigger_{float(value):.2f}" for value in headroom["trigger_times"]]
    early_name, late_name = "trigger_0.40", "trigger_0.45"
    early_row, late_row = names.index(early_name), names.index(late_name)
    nll = torch.tensor(headroom["per_sequence"]["trigger_nll"], dtype=torch.float64)
    counts = torch.tensor(headroom["per_sequence"]["trigger_counts"], dtype=torch.long)
    scores = torch.tensor(bank["features"][SIGNAL], dtype=torch.float64)
    if scores.numel() != int(headroom["n_cond"]):
        raise ValueError("score count does not match the calibration bank")

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, shuffled_prompts, references = reconstruct_panel(headroom, elf_tokenizer)
    device = torch.device(args.device)
    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    evaluator = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()
    shuffled = {}
    shuffled_counts = {}
    for name in (early_name, late_name):
        shuffled[name], shuffled_counts[name] = exp93.conditional_sequence_nlls(
            shuffled_prompts, headroom["texts"][name], evaluator, ppl_tokenizer, device
        )

    fixed = headroom["summaries"][early_name]
    candidates = []
    for k in DELAY_COUNTS:
        selected_rows = torch.argsort(scores, descending=True)[:k]
        choose_late = torch.zeros(scores.numel(), dtype=torch.bool)
        choose_late[selected_rows] = True
        selected_nll = torch.where(choose_late, nll[late_row], nll[early_row])
        selected_counts = torch.where(choose_late, counts[late_row], counts[early_row])
        texts = [
            headroom["texts"][late_name if choose_late[row] else early_name][row]
            for row in range(scores.numel())
        ]
        selected_shuffled = torch.where(
            choose_late, shuffled[late_name], shuffled[early_name]
        )
        selected_shuffled_counts = torch.where(
            choose_late, shuffled_counts[late_name], shuffled_counts[early_name]
        )
        metrics = exp93.summarize_texts(texts, references, selected_nll, selected_counts)
        metrics["shuffled_prompt_ppl"] = exp93.aggregate_ppl(
            selected_shuffled, selected_shuffled_counts
        )
        metrics["prompt_gain_nats"] = math.log(metrics["shuffled_prompt_ppl"]) - math.log(
            metrics["prompt_conditioned_ppl"]
        )
        delta = selected_nll - nll[early_row]
        interval = exp108.bootstrap(delta, args.bootstrap_samples, int(headroom["seed"]) + k)
        improvement = 100 * (
            fixed["prompt_conditioned_ppl"] - metrics["prompt_conditioned_ppl"]
        ) / fixed["prompt_conditioned_ppl"]
        quality_delta = exp108.quality_delta(metrics, fixed)
        gate = (
            improvement >= 2.0
            and interval[1] < 0.0
            and exp108.quality_gate(quality_delta)
            and k >= 8
        )
        candidates.append({
            "delay_count": k,
            "threshold": threshold_for_top_k(scores, k),
            "metrics": metrics,
            "improvement_pct": improvement,
            "mean_delta_nats": float(delta.mean()),
            "mean_delta_ci95": interval,
            "quality_delta": quality_delta,
            "gate_passed": gate,
            "selected_rows": selected_rows.tolist(),
        })

    passing = [candidate for candidate in candidates if candidate["gate_passed"]]
    selected = min(
        passing,
        key=lambda candidate: candidate["metrics"]["prompt_conditioned_ppl"],
        default=None,
    )
    result = {
        "headroom_json": str(args.headroom_json),
        "features_json": str(args.features_json),
        "signal": SIGNAL,
        "orientation": "delay_if_score_ge_threshold",
        "fallback": early_name,
        "candidate_delay_counts": DELAY_COUNTS,
        "fixed_metrics": fixed,
        "candidates": candidates,
        "selected_policy": selected,
        "stage_b_passed": selected is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "stage_b_passed": result["stage_b_passed"],
        "selected_policy": selected,
    }, indent=2))
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
