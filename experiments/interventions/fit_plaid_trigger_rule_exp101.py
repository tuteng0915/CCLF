#!/usr/bin/env python3
"""EXP-101: freeze a one-statistic event-trigger rule on a discovery bank.

The policy class is intentionally small. It chooses the earliest candidate
whose inference-time statistic crosses a threshold, optionally for two
successive observations, and otherwise falls back to the frozen step-14 arm.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(HERE))

import eval_late_coupled_blocks as quality_base  # noqa: E402
import eval_plaid_conditional_late_coupling as conditional_base  # noqa: E402


DIRECTIONS = {
    "mean_confidence": "ge",
    "q10_confidence": "ge",
    "mean_entropy": "le",
    "mean_top12_margin": "ge",
    "lexical_revision": "le",
    "xhat_cosine_instability": "le",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery_bank", required=True)
    parser.add_argument("--validation_bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantiles", type=int, default=39)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_bank(path):
    payload = json.loads(Path(path).read_text())
    required = (
        "triggers",
        "fixed_trigger",
        "feature_names",
        "event_features",
        "per_sequence",
        "density",
        "horizon",
        "texts",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"bank {path} missing {missing}")
    if not payload.get("paired_initial_and_ancestral_noise", False):
        raise ValueError(f"bank {path} does not use paired Plaid noise")
    payload["features_tensor"] = torch.tensor(
        payload["event_features"], dtype=torch.float32
    )
    payload["nll_tensor"] = torch.tensor(
        payload["per_sequence"]["trigger_nll"], dtype=torch.float64
    ).T
    payload["counts_tensor"] = torch.tensor(
        payload["per_sequence"]["trigger_token_counts"], dtype=torch.long
    ).T
    payload["shuffled_nll_tensor"] = torch.tensor(
        payload["per_sequence"]["trigger_shuffled_nll"], dtype=torch.float64
    ).T
    expected = (len(payload["features_tensor"]), len(payload["triggers"]))
    if tuple(payload["features_tensor"].shape[:2]) != expected:
        raise ValueError(f"event feature shape mismatch in {path}")
    if tuple(payload["nll_tensor"].shape) != expected:
        raise ValueError(f"trigger NLL shape mismatch in {path}")
    return payload


def select_indices(values, direction, threshold, persistence, fallback_index):
    crossed = values >= threshold if direction == "ge" else values <= threshold
    if persistence == 2:
        crossed = crossed & torch.cat(
            (torch.zeros_like(crossed[:, :1]), crossed[:, :-1]), dim=1
        )
    candidate = torch.arange(values.shape[1])[None].expand_as(values)
    sentinel = torch.full_like(candidate, values.shape[1])
    first = torch.where(crossed, candidate, sentinel).min(dim=1).values
    return torch.where(first == values.shape[1], fallback_index, first)


def aggregate_ppl(nll, counts):
    return math.exp(float((nll * counts.double()).sum() / counts.sum()))


def evaluate_rule(bank, rule):
    feature_index = bank["feature_names"].index(rule["feature"])
    values = bank["features_tensor"][:, :, feature_index]
    fallback_index = bank["triggers"].index(bank["fixed_trigger"])
    selected_index = select_indices(
        values,
        rule["direction"],
        rule["threshold"],
        rule["persistence"],
        fallback_index,
    )
    rows = torch.arange(len(selected_index))
    selected_nll = bank["nll_tensor"][rows, selected_index]
    selected_counts = bank["counts_tensor"][rows, selected_index]
    selected_shuffled_nll = bank["shuffled_nll_tensor"][rows, selected_index]
    fixed_nll = bank["nll_tensor"][:, fallback_index]
    fixed_counts = bank["counts_tensor"][:, fallback_index]
    delta = selected_nll - fixed_nll
    selected_texts = [
        bank["texts"]["by_trigger"][str(bank["triggers"][int(index)])][row]
        for row, index in enumerate(selected_index)
    ]
    quality = quality_base.text_quality(selected_texts)
    selected_ppl = aggregate_ppl(selected_nll, selected_counts)
    shuffled_ppl = aggregate_ppl(selected_shuffled_nll, selected_counts)
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
        "selected_index": selected_index,
        "selected_nll": selected_nll,
        "selected_counts": selected_counts,
        "fixed_nll": fixed_nll,
        "selected_ppl": selected_ppl,
        "fixed_ppl": aggregate_ppl(fixed_nll, fixed_counts),
        "mean_delta_nats": float(delta.mean()),
        "better_fraction": float((delta < 0).double().mean()),
        "selected_trigger_mean": float(
            torch.tensor(bank["triggers"])[selected_index].double().mean()
        ),
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


def bootstrap(metrics, samples, seed):
    delta = metrics["selected_nll"] - metrics["fixed_nll"]
    generator = torch.Generator().manual_seed(seed + 1011019)
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(0, len(delta), (len(delta),), generator=generator)
        estimates[index] = delta[rows].mean()
    return [
        float(torch.quantile(estimates, 0.025)),
        float(torch.quantile(estimates, 0.975)),
    ]


def serializable(metrics, samples, seed):
    return {
        key: value
        for key, value in metrics.items()
        if key not in (
            "selected_index",
            "selected_nll",
            "selected_counts",
            "fixed_nll",
        )
    } | {"mean_delta_ci95": bootstrap(metrics, samples, seed)}


def main():
    args = parse_args()
    discovery = load_bank(args.discovery_bank)
    validation = load_bank(args.validation_bank)
    for key in ("triggers", "fixed_trigger", "feature_names", "density", "horizon"):
        if discovery[key] != validation[key]:
            raise ValueError(f"discovery/validation mismatch for {key}")
    if args.quantiles < 3:
        raise ValueError("at least three threshold quantiles are required")

    candidates = []
    quantile_grid = torch.linspace(0.025, 0.975, args.quantiles)
    for feature, direction in DIRECTIONS.items():
        feature_index = discovery["feature_names"].index(feature)
        values = discovery["features_tensor"][:, :, feature_index].flatten()
        thresholds = torch.quantile(values, quantile_grid).unique()
        for persistence in (1, 2):
            for threshold in thresholds.tolist():
                rule = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "persistence": persistence,
                }
                metrics = evaluate_rule(discovery, rule)
                candidates.append((metrics["mean_delta_nats"], rule, metrics))

    candidates.sort(key=lambda item: item[0])
    _, best_rule, discovery_metrics = candidates[0]
    validation_metrics = evaluate_rule(validation, best_rule)
    validation_ci = bootstrap(
        validation_metrics, args.bootstrap_samples, args.seed + 1
    )
    quality_delta = validation_metrics["quality_delta"]
    result = {
        **vars(args),
        "policy_class": "single-statistic earliest-threshold crossing",
        "selection_used_validation": False,
        "n_candidate_rules": len(candidates),
        "frozen_rule": best_rule,
        "discovery": serializable(
            discovery_metrics, args.bootstrap_samples, args.seed
        ),
        "validation": serializable(
            validation_metrics, args.bootstrap_samples, args.seed + 1
        ),
        "validation_gate_passed": (
            validation_metrics["mean_delta_nats"] < 0.0
            and validation_ci[1] < 0.0
            and quality_delta["d1"] >= -0.005
            and quality_delta["rep4"] <= 0.005
            and quality_delta["degeneration_rate"] <= 0.015
            and quality_delta["prompt_gain_nats"] >= -0.01
        ),
        "top_discovery_rules": [
            {
                "mean_delta_nats": delta,
                **rule,
                "selected_trigger_mean": metrics["selected_trigger_mean"],
            }
            for delta, rule, metrics in candidates[:10]
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()
