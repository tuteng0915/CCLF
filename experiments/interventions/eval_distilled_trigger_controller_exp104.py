#!/usr/bin/env python3
"""EXP-104: one-shot final evaluation of a frozen online trigger controller."""

import argparse
import json
from pathlib import Path

import torch

import train_distilled_trigger_controller_exp104 as train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--final_bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2028)
    return parser.parse_args()


def build_features(bank):
    features = torch.tensor(bank["event_features"], dtype=torch.float32)
    trigger_time = torch.tensor(bank["triggers"], dtype=torch.float32)
    trigger_time = (trigger_time / float(bank["n_steps"]))[None, :, None]
    return torch.cat(
        (features, trigger_time.expand(features.shape[0], -1, -1)), dim=-1
    )


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["threshold"] is None:
        raise ValueError("controller did not pass calibration; final bank must stay closed")
    bank = json.loads(Path(args.final_bank).read_text())
    for key in ("triggers", "fixed_trigger"):
        if bank[key] != checkpoint[key]:
            raise ValueError(f"checkpoint/final mismatch for {key}")
    expected_names = list(bank["feature_names"]) + ["normalized_trigger_step"]
    if expected_names != checkpoint["feature_names"]:
        raise ValueError("checkpoint/final feature names differ")

    device = torch.device(args.device)
    model = train.TriggerController(
        checkpoint["feature_dim"], checkpoint["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    features = build_features(bank)
    prediction = train.predict(
        model,
        features,
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
        device,
    ) * checkpoint["target_std"] + checkpoint["target_mean"]
    metrics = train.evaluate_policy(
        {"bank": bank},
        prediction,
        checkpoint["threshold"],
        args.bootstrap_samples,
        args.seed,
    )
    result = {
        **vars(args),
        "selection_used_final": False,
        "training_seed": checkpoint["training_seed"],
        "frozen_threshold": checkpoint["threshold"],
        "teacher_target": "lookahead-4 unresolved entropy reduction",
        "metrics": metrics,
        "final_gate_passed": metrics["gate_passed"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()

