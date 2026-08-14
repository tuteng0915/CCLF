#!/usr/bin/env python3
"""Merge trajectory-disjoint EXP-100 feature-bank shards."""

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    banks = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    reference = banks[0]
    invariant_keys = (
        "density",
        "trigger_step",
        "horizon",
        "prefix_length",
        "feature_names",
    )
    for index, bank in enumerate(banks[1:], start=1):
        for key in invariant_keys:
            if bank[key] != reference[key]:
                raise ValueError(f"bank {index} mismatches {key}")

    tensor_keys = (
        "features",
        "candidate_masks",
        "candidate_nll",
        "candidate_token_counts",
        "standard_nll",
        "top_confidence_nll",
        "panel_ids",
    )
    merged = {
        key: torch.cat([bank[key] for bank in banks], dim=0) for key in tensor_keys
    }
    merged.update(
        {
            key: reference[key] for key in invariant_keys
        }
    )
    merged.update(
        {
            "seed": -1,
            "panel_offset": -1,
            "source_banks": [str(Path(path).resolve()) for path in args.inputs],
            "source_seeds": [bank["seed"] for bank in banks],
            "source_offsets": [bank["panel_offset"] for bank in banks],
        }
    )
    unique_panels = torch.unique(merged["panel_ids"], dim=0)
    if len(unique_panels) != len(merged["panel_ids"]):
        raise RuntimeError("trajectory-disjoint merge contains duplicate token panels")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(
        f"Saved -> {output} trajectories={len(merged['features'])} "
        f"candidates={merged['candidate_masks'].shape[1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
