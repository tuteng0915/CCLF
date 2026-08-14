#!/usr/bin/env python3
"""EXP-105: calibrate/evaluate a causal short-lookahead response trigger."""

import argparse
import json
from pathlib import Path

import torch

import train_distilled_trigger_controller_exp104 as base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_local", required=True)
    parser.add_argument("--calibration_bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final_local")
    parser.add_argument("--final_bank")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def candidate_thresholds(score, fixed_index):
    eligible = score[:, :fixed_index].flatten()
    return torch.cat(
        (
            torch.tensor([-float("inf")], dtype=eligible.dtype),
            eligible.unique().sort().values,
            torch.tensor([float("inf")], dtype=eligible.dtype),
        )
    ).tolist()


def main():
    args = parse_args()
    if bool(args.final_local) != bool(args.final_bank):
        raise ValueError("final_local and final_bank must be supplied together")

    calibration = base.load_pair(args.calibration_bank, args.calibration_local)
    score = calibration["target"]
    fixed_index = calibration["bank"]["triggers"].index(
        calibration["bank"]["fixed_trigger"]
    )
    candidates = [
        base.evaluate_policy(
            calibration,
            score,
            threshold,
            args.bootstrap_samples,
            args.seed + index,
        )
        for index, threshold in enumerate(candidate_thresholds(score, fixed_index))
    ]
    passing = [metrics for metrics in candidates if metrics["gate_passed"]]
    best = min(passing, key=lambda metrics: metrics["mean_delta_nats"]) if passing else None

    result = {
        **vars(args),
        "selection_used_final": False,
        "signal": "lookahead-4 unresolved entropy reduction",
        "policy": "first response above threshold at steps 8/10/12; else 14",
        "n_thresholds": len(candidates),
        "calibration_gate_passed": best is not None,
        "frozen_threshold": None if best is None else best["threshold"],
        "calibration": best,
        "top_passing_thresholds": sorted(
            passing, key=lambda item: item["mean_delta_nats"]
        )[:5],
    }

    if best is not None and args.final_local:
        final = base.load_pair(args.final_bank, args.final_local)
        final_metrics = base.evaluate_policy(
            final,
            final["target"],
            best["threshold"],
            args.bootstrap_samples,
            args.seed + 100000,
        )
        result["selection_used_final"] = True
        result["final"] = final_metrics
        result["final_gate_passed"] = final_metrics["gate_passed"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()
