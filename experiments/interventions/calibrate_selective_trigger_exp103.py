#!/usr/bin/env python3
"""EXP-103: quality-constrained abstention for the frozen EXP-102 signal."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_signal", required=True)
    parser.add_argument("--calibration_local", required=True)
    parser.add_argument("--calibration_bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final_local")
    parser.add_argument("--final_bank")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def load(local_path, bank_path):
    local = json.loads(Path(local_path).read_text())
    bank = json.loads(Path(bank_path).read_text())
    for key in ("seed", "panel_offset", "triggers", "fixed_trigger"):
        if local[key] != bank[key]:
            raise ValueError(f"local/source mismatch for {key}")
    return {
        "local": local,
        "bank": bank,
        "signals": torch.tensor(local["local_signals"], dtype=torch.float64),
        "nll": torch.tensor(
            bank["per_sequence"]["trigger_nll"], dtype=torch.float64
        ).T,
        "counts": torch.tensor(
            bank["per_sequence"]["trigger_token_counts"], dtype=torch.long
        ).T,
        "shuffled": torch.tensor(
            bank["per_sequence"]["trigger_shuffled_nll"], dtype=torch.float64
        ).T,
    }


def aggregate_ppl(nll, counts):
    return math.exp(float((nll * counts.double()).sum() / counts.sum()))


def bootstrap(delta, samples, seed):
    generator = torch.Generator().manual_seed(seed + 1030033)
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(0, len(delta), (len(delta),), generator=generator)
        estimates[index] = delta[rows].mean()
    return [
        float(torch.quantile(estimates, 0.025)),
        float(torch.quantile(estimates, 0.975)),
    ]


def quality_gate(delta):
    return (
        delta["d1"] >= -0.005
        and delta["rep4"] <= 0.005
        and delta["degeneration_rate"] <= 0.015
        and delta["prompt_gain_nats"] >= -0.01
    )


def evaluate(data, rule, threshold, samples, seed):
    local, bank = data["local"], data["bank"]
    lookahead_index = local["lookaheads"].index(rule["lookahead"])
    signal_index = local["signal_names"].index(rule["signal"])
    score = data["signals"][:, :, lookahead_index, signal_index]
    if rule["direction"] == "min":
        score = -score
    fixed_index = bank["triggers"].index(bank["fixed_trigger"])
    best_index = score.argmax(dim=1)
    rows = torch.arange(len(best_index))
    advantage = score[rows, best_index] - score[:, fixed_index]
    selected_index = torch.where(
        advantage >= threshold,
        best_index,
        torch.full_like(best_index, fixed_index),
    )
    selected_nll = data["nll"][rows, selected_index]
    selected_counts = data["counts"][rows, selected_index]
    selected_shuffled = data["shuffled"][rows, selected_index]
    fixed_nll = data["nll"][:, fixed_index]
    fixed_counts = data["counts"][:, fixed_index]
    nll_delta = selected_nll - fixed_nll
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
    quality_delta = {
        "d1": quality["d1"] - fixed_quality["d1"],
        "d2": quality["d2"] - fixed_quality["d2"],
        "rep4": quality["rep4"] - fixed_quality["rep4"],
        "degeneration_rate": quality["degeneration_rate"]
        - fixed_quality["degeneration_rate"],
        "prompt_gain_nats": quality["prompt_gain_nats"]
        - fixed_quality["prompt_gain_nats"],
    }
    interval = bootstrap(nll_delta, samples, seed)
    return {
        "threshold": threshold,
        "selected_ppl": selected_ppl,
        "fixed_ppl": aggregate_ppl(fixed_nll, fixed_counts),
        "mean_delta_nats": float(nll_delta.mean()),
        "mean_delta_ci95": interval,
        "switch_fraction": float((selected_index != fixed_index).double().mean()),
        "selected_trigger_histogram": {
            str(step): int((selected_index == index).sum())
            for index, step in enumerate(bank["triggers"])
        },
        "quality": quality,
        "quality_delta": quality_delta,
        "gate_passed": interval[1] < 0.0 and quality_gate(quality_delta),
        "advantage": advantage,
    }


def serializable(metrics):
    return {key: value for key, value in metrics.items() if key != "advantage"}


def main():
    args = parse_args()
    frozen = json.loads(Path(args.frozen_signal).read_text())
    rule = frozen["frozen_rule"]
    calibration = load(args.calibration_local, args.calibration_bank)

    lookahead_index = calibration["local"]["lookaheads"].index(rule["lookahead"])
    signal_index = calibration["local"]["signal_names"].index(rule["signal"])
    score = calibration["signals"][:, :, lookahead_index, signal_index]
    if rule["direction"] == "min":
        score = -score
    fixed_index = calibration["bank"]["triggers"].index(
        calibration["bank"]["fixed_trigger"]
    )
    rows = torch.arange(len(score))
    advantage = score.max(dim=1).values - score[:, fixed_index]
    thresholds = torch.cat(
        (
            torch.tensor([-float("inf")], dtype=advantage.dtype),
            torch.quantile(
                advantage,
                torch.linspace(0.0, 1.0, 65, dtype=advantage.dtype),
            ).unique(),
            torch.tensor([float("inf")], dtype=advantage.dtype),
        )
    ).tolist()
    candidates = [
        evaluate(calibration, rule, threshold, args.bootstrap_samples, args.seed)
        for threshold in thresholds
    ]
    passing = [metrics for metrics in candidates if metrics["gate_passed"]]
    best = min(passing, key=lambda metrics: metrics["mean_delta_nats"]) if passing else None

    result = {
        **vars(args),
        "frozen_signal_rule": rule,
        "selection_used_final": False,
        "n_thresholds": len(candidates),
        "calibration_gate_passed": best is not None,
        "frozen_threshold": None if best is None else best["threshold"],
        "calibration": None if best is None else serializable(best),
        "top_passing_thresholds": [
            serializable(metrics)
            for metrics in sorted(passing, key=lambda item: item["mean_delta_nats"])[:5]
        ],
    }
    if best is not None and bool(args.final_local) != bool(args.final_bank):
        raise ValueError("final_local and final_bank must be supplied together")
    if best is not None and args.final_local:
        final = load(args.final_local, args.final_bank)
        final_metrics = evaluate(
            final,
            rule,
            best["threshold"],
            args.bootstrap_samples,
            args.seed + 1,
        )
        result["final"] = serializable(final_metrics)
        result["final_gate_passed"] = final_metrics["gate_passed"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()
