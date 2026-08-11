#!/usr/bin/env python3
"""Print the decisive EXP-79 comparisons from one result JSON."""

import argparse
import json
from pathlib import Path


FIELDS = (
    "ppl",
    "prompt_conditioned_ppl",
    "rouge_l",
    "decoded_prefix_agreement",
    "prefix_ppl",
    "suffix_ppl",
    "boundary32_conditional_ppl",
    "d1",
    "d2",
    "rep4",
    "degeneration_rate",
    "prefix_revision",
    "suffix_revision",
    "denoiser_calls",
    "processed_token_calls",
)


def fmt(value):
    if not isinstance(value, (int, float)):
        return str(value)
    if value != value:
        return "nan"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text())
    results = payload["results"]

    names = [
        "parallel32",
        "parallel60",
        "semi_ar64",
        "late_reencoded_m24",
        "late_reencoded_m28",
        "late_reencoded_m28_freeze_a",
    ]
    names = [name for name in names if name in results]
    print("\t".join(("arm", *FIELDS)))
    for name in names:
        print("\t".join((name, *(fmt(results[name].get(field)) for field in FIELDS))))

    primary = results.get("late_reencoded_m28")
    if primary is None:
        return
    primary_fields = (
        "prompt_conditioned_ppl",
        "rouge_l",
        "boundary32_conditional_ppl",
    ) if payload.get("conditional") else (
        "ppl",
        "suffix_ppl",
        "boundary32_conditional_ppl",
    )
    print("\nPrimary deltas: late_reencoded_m28 minus comparator")
    for comparator in ("parallel32", "parallel60", "semi_ar64"):
        if comparator not in results:
            continue
        deltas = []
        for field in primary_fields:
            deltas.append(
                f"{field}={primary[field] - results[comparator][field]:+.4f}"
            )
        print(f"  vs {comparator}: " + ", ".join(deltas))

    frozen = results.get("late_reencoded_m28_freeze_a")
    if frozen is not None:
        print("\nJoint-refinement contrast: full minus freeze-A")
        for field in primary_fields:
            print(f"  {field}: {primary[field] - frozen[field]:+.4f}")
        print(f"  full prefix revision: {primary['prefix_revision']:.4f}")
        print(f"  freeze prefix revision: {frozen['prefix_revision']:.4f}")


if __name__ == "__main__":
    main()
