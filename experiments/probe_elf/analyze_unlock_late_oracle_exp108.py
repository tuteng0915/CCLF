#!/usr/bin/env python3
"""EXP-108 supplement: quality-audit a restricted late-trigger oracle.

This script does not regenerate ELF trajectories.  It selects among trigger
arms already saved by ``unlock_trigger_headroom_exp108.py``, reconstructs the
fixed OWT prompt/reference panel, and evaluates the selected continuations
with the same complete metric panel.  The selector remains an offline oracle,
not a deployable sampler.
"""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min_trigger", type=float, default=.40)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--output", type=Path)
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


def main():
    args = parse_args()
    payload = json.loads(args.input.read_text())
    fixed_name = f"trigger_{float(payload['fixed_trigger']):.2f}"
    allowed = [
        f"trigger_{float(value):.2f}"
        for value in payload["trigger_times"]
        if float(value) + 1e-9 >= args.min_trigger
    ]
    if fixed_name not in allowed or not allowed:
        raise ValueError("restricted action space must include the fixed reference")

    all_names = [f"trigger_{float(value):.2f}" for value in payload["trigger_times"]]
    rows_in_full = [all_names.index(name) for name in allowed]
    full_nll = torch.as_tensor(
        payload["per_sequence"]["trigger_nll"], dtype=torch.float64
    )
    full_counts = torch.as_tensor(
        payload["per_sequence"]["trigger_counts"], dtype=torch.long
    )
    candidate_nll = full_nll[rows_in_full]
    candidate_counts = full_counts[rows_in_full]
    best_local = candidate_nll.argmin(dim=0)
    rows = torch.arange(candidate_nll.shape[1])
    best_nll = candidate_nll[best_local, rows]
    best_counts = candidate_counts[best_local, rows]
    selected_names = [allowed[int(index)] for index in best_local]
    selected_texts = [
        payload["texts"][name][row]
        for row, name in enumerate(selected_names)
    ]

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    prompts, shuffled_prompts, references = reconstruct_panel(payload, elf_tokenizer)
    device = torch.device(args.device)
    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    evaluator = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()
    shuffled_nll, shuffled_counts = exp93.conditional_sequence_nlls(
        shuffled_prompts, selected_texts, evaluator, ppl_tokenizer, device
    )
    metrics = exp93.summarize_texts(
        selected_texts, references, best_nll, best_counts
    )
    metrics["shuffled_prompt_ppl"] = exp93.aggregate_ppl(
        shuffled_nll, shuffled_counts
    )
    metrics["prompt_gain_nats"] = math.log(metrics["shuffled_prompt_ppl"]) - math.log(
        metrics["prompt_conditioned_ppl"]
    )

    fixed = payload["summaries"][fixed_name]
    fixed_row = all_names.index(fixed_name)
    delta = best_nll - full_nll[fixed_row]
    quality_delta = exp108.quality_delta(metrics, fixed)
    improvement = 100 * (
        fixed["prompt_conditioned_ppl"] - metrics["prompt_conditioned_ppl"]
    ) / fixed["prompt_conditioned_ppl"]
    interval = exp108.bootstrap(delta, args.bootstrap_samples, int(payload["seed"]))
    result = {
        "source": str(args.input),
        "oracle_is_deployable": False,
        "min_trigger": args.min_trigger,
        "allowed_triggers": allowed,
        "fixed_reference": fixed_name,
        "fixed_metrics": fixed,
        "oracle_metrics": metrics,
        "oracle_vs_fixed": {
            "fixed_ppl": fixed["prompt_conditioned_ppl"],
            "oracle_ppl": metrics["prompt_conditioned_ppl"],
            "improvement_pct": improvement,
            "mean_delta_nats": float(delta.mean()),
            "mean_delta_ci95": interval,
            "quality_delta": quality_delta,
            "gate_passed": (
                improvement >= 5.0
                and interval[1] < 0.0
                and exp108.quality_gate(quality_delta)
            ),
        },
        "winner_histogram": {
            name: selected_names.count(name) for name in allowed
        },
        "selected_names": selected_names,
        "selected_texts": selected_texts,
    }
    output = args.output or args.input.with_name(
        args.input.stem + f"_late_ge_{args.min_trigger:.2f}.json"
    )
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "oracle_vs_fixed": result["oracle_vs_fixed"],
        "winner_histogram": result["winner_histogram"],
    }, indent=2))
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
