#!/usr/bin/env python3
"""Evaluate a frozen EXP-100 selector on an unopened feature bank."""

import argparse
import json
from pathlib import Path

import torch

from train_joint_anchor_selector_exp100 import (
    JointSubsetScorer,
    evaluate,
    load_bank,
    serializable,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bank = load_bank(args.test_bank)
    for key in ("density", "trigger_step", "horizon", "prefix_length"):
        if bank[key] != checkpoint[key]:
            raise ValueError(f"frozen checkpoint/test mismatch for {key}")
    mean = checkpoint["feature_mean"].view(1, 1, -1)
    std = checkpoint["feature_std"].view(1, 1, -1)
    bank["features"] = ((bank["features"].float() - mean) / std).half()

    device = torch.device(args.device)
    model = JointSubsetScorer(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, bank, device, args.batch_size)
    result = {
        **vars(args),
        "training_seed": checkpoint["training_seed"],
        "best_epoch": checkpoint["best_epoch"],
        "test_seed": bank["seed"],
        "test_panel_offset": bank["panel_offset"],
        "metrics": serializable(metrics),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()
