#!/usr/bin/env python3
"""EXP-102: freeze a native local-utility signal and evaluate trigger choice."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_late_coupled_blocks as quality_base  # noqa: E402
import eval_plaid_conditional_late_coupling as conditional_base  # noqa: E402


DIRECTIONS = {
    "confidence_gain": ("max",),
    "entropy_reduction": ("max",),
    "margin_gain": ("max",),
    "xhat_control_cosine_distance": ("max", "min"),
    "lexical_disagreement": ("max", "min"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery_local", required=True)
    parser.add_argument("--validation_local", required=True)
    parser.add_argument("--discovery_bank", required=True)
    parser.add_argument("--validation_bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load(local_path, bank_path):
    local = json.loads(Path(local_path).read_text())
    bank = json.loads(Path(bank_path).read_text())
    for key in ("seed", "panel_offset", "triggers", "fixed_trigger"):
        if local[key] != bank[key]:
            raise ValueError(f"local/source mismatch for {key}")
    signals = torch.tensor(local["local_signals"], dtype=torch.float64)
    nll = torch.tensor(bank["per_sequence"]["trigger_nll"], dtype=torch.float64).T
    counts = torch.tensor(
        bank["per_sequence"]["trigger_token_counts"], dtype=torch.long
    ).T
    shuffled_nll = torch.tensor(
        bank["per_sequence"]["trigger_shuffled_nll"], dtype=torch.float64
    ).T
    if tuple(signals.shape[:2]) != tuple(nll.shape):
        raise ValueError("local signal/final NLL trajectory shape mismatch")
    return {
        "local": local,
        "bank": bank,
        "signals": signals,
        "nll": nll,
        "counts": counts,
        "shuffled_nll": shuffled_nll,
    }


def aggregate_ppl(nll, counts):
    return math.exp(float((nll * counts.double()).sum() / counts.sum()))


def pairwise_accuracy(score, nll):
    score_diff = score[:, :, None] - score[:, None, :]
    utility_diff = -nll[:, :, None] + nll[:, None, :]
    candidates = score.shape[1]
    upper = torch.triu(
        torch.ones((candidates, candidates), dtype=torch.bool), diagonal=1
    )[None]
    valid = upper & (score_diff.abs() > 1e-12) & (utility_diff.abs() > 1e-12)
    return float(((score_diff * utility_diff) > 0)[valid].double().mean())


def mean_spearman(score, nll):
    score_rank = score.argsort(dim=1).argsort(dim=1).double()
    utility_rank = (-nll).argsort(dim=1).argsort(dim=1).double()
    score_rank -= score_rank.mean(dim=1, keepdim=True)
    utility_rank -= utility_rank.mean(dim=1, keepdim=True)
    numerator = (score_rank * utility_rank).sum(dim=1)
    denominator = score_rank.square().sum(dim=1).sqrt() * utility_rank.square().sum(
        dim=1
    ).sqrt()
    return float((numerator / denominator.clamp_min(1e-12)).mean())


def bootstrap(delta, samples, seed):
    generator = torch.Generator().manual_seed(seed + 1020031)
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(0, len(delta), (len(delta),), generator=generator)
        estimates[index] = delta[rows].mean()
    return [
        float(torch.quantile(estimates, 0.025)),
        float(torch.quantile(estimates, 0.975)),
    ]


def evaluate(data, rule, bootstrap_samples, seed):
    local, bank = data["local"], data["bank"]
    lookahead_index = local["lookaheads"].index(rule["lookahead"])
    signal_index = local["signal_names"].index(rule["signal"])
    raw_score = data["signals"][:, :, lookahead_index, signal_index]
    score = raw_score if rule["direction"] == "max" else -raw_score
    selected_index = score.argmax(dim=1)
    rows = torch.arange(len(selected_index))
    fixed_index = bank["triggers"].index(bank["fixed_trigger"])
    selected_nll = data["nll"][rows, selected_index]
    selected_counts = data["counts"][rows, selected_index]
    selected_shuffled = data["shuffled_nll"][rows, selected_index]
    fixed_nll = data["nll"][:, fixed_index]
    fixed_counts = data["counts"][:, fixed_index]
    delta = selected_nll - fixed_nll
    selected_texts = [
        bank["texts"]["by_trigger"][str(bank["triggers"][int(index)])][row]
        for row, index in enumerate(selected_index)
    ]
    quality = quality_base.text_quality(selected_texts)
    selected_ppl = aggregate_ppl(selected_nll, selected_counts)
    shuffled_ppl = aggregate_ppl(selected_shuffled, selected_counts)
    quality.update(
        prompt_conditioned_ppl=selected_ppl,
        shuffled_prompt_ppl=shuffled_ppl,
        prompt_gain_nats=math.log(shuffled_ppl) - math.log(selected_ppl),
        rouge_l=float(
            np.mean(
                [
                    conditional_base.rouge_l_f1(prediction, reference)
                    for prediction, reference in zip(
                        selected_texts, bank["texts"]["references"]
                    )
                ]
            )
        ),
    )
    fixed_quality = bank["aggregate"][f"trigger_{bank['fixed_trigger']:02d}"]
    return {
        "selected_ppl": selected_ppl,
        "fixed_ppl": aggregate_ppl(fixed_nll, fixed_counts),
        "mean_delta_nats": float(delta.mean()),
        "mean_delta_ci95": bootstrap(delta, bootstrap_samples, seed),
        "better_fraction": float((delta < 0).double().mean()),
        "pairwise_accuracy": pairwise_accuracy(score, data["nll"]),
        "mean_spearman": mean_spearman(score, data["nll"]),
        "selected_trigger_histogram": {
            str(step): int((selected_index == index).sum())
            for index, step in enumerate(bank["triggers"])
        },
        "quality": quality,
        "quality_delta": {
            "d1": quality["d1"] - fixed_quality["d1"],
            "d2": quality["d2"] - fixed_quality["d2"],
            "rep4": quality["rep4"] - fixed_quality["rep4"],
            "degeneration_rate": quality["degeneration_rate"]
            - fixed_quality["degeneration_rate"],
            "prompt_gain_nats": quality["prompt_gain_nats"]
            - fixed_quality["prompt_gain_nats"],
        },
    }


def main():
    args = parse_args()
    discovery = load(args.discovery_local, args.discovery_bank)
    validation = load(args.validation_local, args.validation_bank)
    for key in ("triggers", "lookaheads", "signal_names"):
        left = discovery["local"][key]
        right = validation["local"][key]
        if left != right:
            raise ValueError(f"discovery/validation mismatch for {key}")

    candidates = []
    for lookahead in discovery["local"]["lookaheads"]:
        for signal in discovery["local"]["signal_names"]:
            for direction in DIRECTIONS[signal]:
                rule = {
                    "lookahead": lookahead,
                    "signal": signal,
                    "direction": direction,
                }
                metrics = evaluate(
                    discovery, rule, args.bootstrap_samples, args.seed
                )
                candidates.append((metrics["mean_delta_nats"], rule, metrics))
    candidates.sort(key=lambda item: item[0])
    _, rule, discovery_metrics = candidates[0]
    validation_metrics = evaluate(
        validation, rule, args.bootstrap_samples, args.seed + 1
    )
    delta = validation_metrics["quality_delta"]
    gate = (
        validation_metrics["mean_delta_ci95"][1] < 0.0
        and discovery_metrics["pairwise_accuracy"] > 0.55
        and validation_metrics["pairwise_accuracy"] > 0.55
        and delta["d1"] >= -0.005
        and delta["rep4"] <= 0.005
        and delta["degeneration_rate"] <= 0.015
        and delta["prompt_gain_nats"] >= -0.01
    )
    result = {
        **vars(args),
        "selection_used_validation": False,
        "n_candidate_signals": len(candidates),
        "frozen_rule": rule,
        "discovery": discovery_metrics,
        "validation": validation_metrics,
        "validation_gate_passed": gate,
        "top_discovery_signals": [
            {
                "mean_delta_nats": value,
                **candidate_rule,
                "pairwise_accuracy": metrics["pairwise_accuracy"],
                "mean_spearman": metrics["mean_spearman"],
            }
            for value, candidate_rule, metrics in candidates[:10]
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()
