#!/usr/bin/env python3
"""Merge deterministic EXP-80 arm shards and print the paired main table."""

import argparse
import json
from pathlib import Path


SHARED_KEYS = (
    "checkpoint",
    "seed",
    "n_uncond",
    "n_cond",
    "max_length",
    "prefix_length",
    "groups",
    "n_steps",
    "noise_scale",
    "sccfg",
    "conditional_dataset",
    "owt_offset",
)
ORDER = (
    "standard32",
    "standard64",
    "standard136",
    "unlock4",
    "soft_ltr",
    "soft_random",
    "pipeline_local_refine8",
    "canonical_ltr_refine8",
)


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    if value != value:
        return "nan"
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    reference = payloads[0]
    for payload in payloads[1:]:
        for key in SHARED_KEYS:
            if payload.get(key) != reference.get(key):
                raise ValueError(f"shard mismatch for {key}")

    results = {}
    for payload in payloads:
        overlap = results.keys() & payload["results"].keys()
        if overlap:
            raise ValueError(f"duplicate arms: {sorted(overlap)}")
        results.update(payload["results"])

    header = (
        "arm",
        "u_ppl",
        "u_d1",
        "u_d2",
        "u_rep4",
        "u_deg",
        "c_prompt_ppl",
        "c_shuffle_ppl",
        "c_gain",
        "c_rl",
        "c_d1",
        "c_d2",
        "c_rep4",
        "c_deg",
        "calls",
        "readouts",
    )
    print("\t".join(header))
    for arm in ORDER:
        if arm not in results:
            continue
        u = results[arm]["unconditional"]
        c = results[arm]["conditional"]
        row = (
            arm,
            u["ppl"],
            u["d1"],
            u["d2"],
            u["rep4"],
            u["degeneration_rate"],
            c["prompt_conditioned_ppl"],
            c["shuffled_prompt_ppl"],
            c["prompt_gain_nats"],
            c["rouge_l"],
            c["d1"],
            c["d2"],
            c["rep4"],
            c["degeneration_rate"],
            c["denoiser_calls"],
            c["readout_calls"],
        )
        print("\t".join(fmt(value) for value in row))

    if args.output:
        merged = {
            key: reference[key]
            for key in reference
            if key not in ("arms", "label", "results")
        }
        merged.update(
            {
                "arms": [arm for arm in ORDER if arm in results],
                "label": "merged_p0_owt",
                "source_shards": [str(path) for path in args.inputs],
                "results": results,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
