#!/usr/bin/env python3
"""EXP-99: paired Plaid temporary-anchor subset-headroom audit.

The Plaid sampler is stochastic.  Candidate masks therefore vary through a
dedicated ``mask_seed`` while initial noise, prompts, and every ancestral
solver noise draw remain fixed within a trajectory.  The oracle best-of-M arm
is a diagnostic upper bound selected with final conditional NLL; it is not a
deployable inference method.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GLOBAL_DIR = ROOT / "experiments" / "global_state"
for path in (HERE, GLOBAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_late_coupled_blocks as quality_base  # noqa: E402
import eval_plaid_conditional_late_coupling as conditional_base  # noqa: E402
import eval_temporary_anchor_portability_exp90 as exp90  # noqa: E402
from common import load_adapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results/exp99_plaid_subset_headroom")
    parser.add_argument("--label", default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel_offset", type=int, default=0)
    parser.add_argument("--n_cond", type=int, default=64)
    parser.add_argument("--n_masks", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--trigger_step", type=int, default=14)
    parser.add_argument("--density", type=float, choices=(0.50, 0.75), default=0.75)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--ppl_model", default="gpt2-large")
    parser.add_argument("--ppl_batch_size", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--skip_reference_gate", action="store_true")
    return parser.parse_args()


def setting_args(args):
    class Settings:
        pass

    settings = Settings()
    for name, value in vars(args).items():
        setattr(settings, name, value)
    settings.model = "plaid"
    settings.checkpoint = "baseline"
    return settings


def load_conditional_panel(adapter, n_samples, seq_len, offset):
    """Load a deterministic trajectory-disjoint panel from streaming corpora."""
    from datasets import load_dataset

    sources = (
        ("Skylion007/openwebtext", {}),
        ("stas/openwebtext-10k", {}),
        ("wikitext", {"name": "wikitext-103-raw-v1"}),
    )
    errors = []
    for name, kwargs in sources:
        sequences = []
        eligible_index = 0
        try:
            dataset = load_dataset(name, split="train", streaming=True, **kwargs)
            for example in dataset:
                text = example["text"].strip()
                if len(text) < 3 * seq_len:
                    continue
                ids = exp90.tokenize(adapter, text, seq_len)
                if ids is None:
                    continue
                if eligible_index < offset:
                    eligible_index += 1
                    continue
                sequences.append(ids)
                eligible_index += 1
                if len(sequences) == n_samples:
                    return torch.tensor(sequences, dtype=torch.long), name
        except Exception as error:  # pragma: no cover - network fallback
            errors.append(f"{name}: {error}")
    raise RuntimeError("conditional panel unavailable: " + " | ".join(errors))


@torch.no_grad()
def generate_candidates(adapter, args, grid, panel_ids):
    names = ["standard", "top_confidence"] + [
        f"random_{index:03d}" for index in range(args.n_masks)
    ]
    records = {name: {"ids": [], "stats": []} for name in names}
    reference_agreements = []
    run_args = setting_args(args)

    for batch_index, start in enumerate(range(0, args.n_cond, args.batch_size)):
        end = min(start + args.batch_size, args.n_cond)
        size = end - start
        generator = torch.Generator(device=adapter.device).manual_seed(
            args.seed * 10007 + 700001 + batch_index
        )
        eps = adapter.sample_epsilon(
            (size, args.seq_len, adapter.d_model), generator=generator
        )
        prompt_ids = panel_ids[start:end, : args.prefix_length]
        prompt_clean = adapter.encode_clean(prompt_ids).to(adapter.device)

        for name in names:
            if name.startswith("random_"):
                mask_index = int(name.rsplit("_", 1)[1])
                arm = "random"
                mask_seed = args.seed + 1009 * (mask_index + 1)
            else:
                arm = name
                mask_seed = None
            result = exp90.run_arm(
                adapter,
                eps,
                grid,
                run_args,
                arm,
                "conditional",
                batch_index,
                prompt_clean,
                mask_seed=mask_seed,
            )
            records[name]["ids"].append(result.pop("ids"))
            records[name]["stats"].append(result)

        if not args.skip_reference_gate and batch_index == 0:
            duplicate = exp90.run_arm(
                adapter,
                eps,
                grid,
                run_args,
                "standard",
                "conditional",
                batch_index,
                prompt_clean,
            )
            agreement = float(
                (duplicate["ids"] == records["standard"]["ids"][-1]).float().mean()
            )
            reference_agreements.append(agreement)
            if agreement != 1.0:
                raise RuntimeError(f"native reference agreement={agreement}")
        print(f"generation completed {end}/{args.n_cond}", flush=True)

    for record in records.values():
        record["ids"] = torch.cat(record["ids"], dim=0)
    return records, min(reference_agreements) if reference_agreements else None


@torch.no_grad()
def conditional_sequence_nlls(prefixes, suffixes, evaluator, max_length=1024):
    tokenizer = evaluator.tokenizer
    sequences, prefix_lengths = [], []
    for prefix, suffix in zip(prefixes, suffixes):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        separator = " " if prefix.strip() and suffix.strip() else ""
        suffix_ids = tokenizer.encode(separator + suffix, add_special_tokens=False)
        budget = max(max_length - len(suffix_ids), 0)
        prefix_ids = prefix_ids[-budget:] if budget else []
        sequences.append(prefix_ids + suffix_ids)
        prefix_lengths.append(len(prefix_ids))

    width = max(max(map(len, sequences), default=0), 1)
    ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    suffix_mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, (sequence, prefix_length) in enumerate(zip(sequences, prefix_lengths)):
        if sequence:
            ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention[row, : len(sequence)] = 1
            suffix_mask[row, prefix_length : len(sequence)] = True

    sums = torch.zeros(len(sequences), dtype=torch.float64)
    counts = torch.zeros(len(sequences), dtype=torch.long)
    for start in range(0, len(sequences), evaluator.batch_size):
        end = min(start + evaluator.batch_size, len(sequences))
        ids_b = ids[start:end].to(evaluator.device)
        attention_b = attention[start:end].to(evaluator.device)
        valid = suffix_mask[start:end, 1:].to(evaluator.device)
        logits = evaluator.model(
            input_ids=ids_b, attention_mask=attention_b
        ).logits[:, :-1].float()
        targets = ids_b[:, 1:]
        token_nll = -F.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        sums[start:end] = (token_nll * valid).sum(dim=1).double().cpu()
        counts[start:end] = valid.sum(dim=1).long().cpu()
    if (counts == 0).any():
        raise RuntimeError("conditional evaluator found an empty generated suffix")
    return sums / counts.double(), counts


def aggregate_ppl(nlls, counts):
    nlls = torch.as_tensor(nlls, dtype=torch.float64)
    counts = torch.as_tensor(counts, dtype=torch.float64)
    return math.exp(float((nlls * counts).sum() / counts.sum()))


def summarize_texts(texts, references, nlls, counts, shuffled_nlls=None):
    result = quality_base.text_quality(texts)
    result["prompt_conditioned_ppl"] = aggregate_ppl(nlls, counts)
    result["mean_sequence_nll"] = float(torch.as_tensor(nlls).mean())
    result["rouge_l"] = float(
        np.mean(
            [
                conditional_base.rouge_l_f1(prediction, reference)
                for prediction, reference in zip(texts, references)
            ]
        )
    )
    if shuffled_nlls is not None:
        shuffled_ppl = aggregate_ppl(shuffled_nlls, counts)
        result["shuffled_prompt_ppl"] = shuffled_ppl
        result["prompt_gain_nats"] = math.log(shuffled_ppl) - math.log(
            result["prompt_conditioned_ppl"]
        )
    return result


def bootstrap_headroom(mean_random_nll, best_nll, samples, seed):
    differences = (mean_random_nll - best_nll).double()
    generator = torch.Generator().manual_seed(seed + 990017)
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(
            0, differences.numel(), (differences.numel(),), generator=generator
        )
        estimates[index] = differences[rows].mean()
    return {
        "mean_nats": float(differences.mean()),
        "ci95_nats": [
            float(torch.quantile(estimates, 0.025)),
            float(torch.quantile(estimates, 0.975)),
        ],
        "positive_trajectory_fraction": float((differences > 0).double().mean()),
    }


def mean_stat(stats, key):
    values = [row[key] for row in stats if row[key] is not None]
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    if not 0 < args.prefix_length < args.seq_len:
        raise ValueError("prefix_length must lie inside seq_len")
    if not 0 < args.trigger_step < args.n_steps:
        raise ValueError("trigger_step must lie inside the solver grid")
    if args.trigger_step + args.horizon > args.n_steps:
        raise ValueError("anchor horizon exceeds the solver grid")
    if min(args.n_cond, args.n_masks, args.batch_size) <= 0:
        raise ValueError("sample counts, masks, and batch size must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    adapter = load_adapter("plaid", "baseline", None, device)
    adapter.seq_len = args.seq_len
    grid = np.linspace(adapter.t_eps, 0.999, args.n_steps + 1).tolist()
    panel_ids, dataset_name = load_conditional_panel(
        adapter, args.n_cond, args.seq_len, args.panel_offset
    )
    records, reference_agreement = generate_candidates(
        adapter, args, grid, panel_ids
    )

    prompts = [
        exp90.decode_ids(adapter, row[: args.prefix_length]) for row in panel_ids
    ]
    references = [
        exp90.decode_ids(adapter, row[args.prefix_length :]) for row in panel_ids
    ]
    for record in records.values():
        record["texts"] = [
            exp90.decode_ids(adapter, row[args.prefix_length :])
            for row in record["ids"]
        ]

    quality_base.release_generator(adapter)
    evaluator = quality_base.PPLEvaluator(
        args.ppl_model, device, args.ppl_batch_size
    )
    shuffled_prompts = prompts[1:] + prompts[:1]
    nlls, counts, shuffled_nlls = {}, {}, {}
    for name, record in records.items():
        print(f"evaluating {name}", flush=True)
        nlls[name], counts[name] = conditional_sequence_nlls(
            prompts, record["texts"], evaluator
        )
        shuffled_nlls[name], _ = conditional_sequence_nlls(
            shuffled_prompts, record["texts"], evaluator
        )

    random_names = [f"random_{index:03d}" for index in range(args.n_masks)]
    random_nll = torch.stack([nlls[name] for name in random_names])
    random_counts = torch.stack([counts[name] for name in random_names])
    best_indices = random_nll.argmin(dim=0)
    worst_indices = random_nll.argmax(dim=0)
    rows = torch.arange(args.n_cond)
    best_nll = random_nll[best_indices, rows]
    worst_nll = random_nll[worst_indices, rows]
    best_counts = random_counts[best_indices, rows]
    worst_counts = random_counts[worst_indices, rows]
    best_texts = [
        records[random_names[int(best_indices[row])]]["texts"][row]
        for row in range(args.n_cond)
    ]
    worst_texts = [
        records[random_names[int(worst_indices[row])]]["texts"][row]
        for row in range(args.n_cond)
    ]
    best_shuffled_nll, _ = conditional_sequence_nlls(
        shuffled_prompts, best_texts, evaluator
    )
    worst_shuffled_nll, _ = conditional_sequence_nlls(
        shuffled_prompts, worst_texts, evaluator
    )

    mean_random_nll = random_nll.mean(dim=0)
    mean_random_ppl = aggregate_ppl(random_nll.flatten(), random_counts.flatten())
    oracle_best_ppl = aggregate_ppl(best_nll, best_counts)
    oracle_improvement_pct = 100.0 * (
        mean_random_ppl - oracle_best_ppl
    ) / mean_random_ppl
    top_nll = nlls["top_confidence"]
    utility = nlls["standard"].unsqueeze(0) - random_nll
    trajectory_iqr = torch.quantile(utility, 0.75, dim=0) - torch.quantile(
        utility, 0.25, dim=0
    )

    flattened_random_texts = [
        text for name in random_names for text in records[name]["texts"]
    ]
    flattened_references = references * args.n_masks
    flattened_shuffled_nll = torch.stack(
        [shuffled_nlls[name] for name in random_names]
    ).flatten()
    aggregate = {
        "standard": summarize_texts(
            records["standard"]["texts"],
            references,
            nlls["standard"],
            counts["standard"],
            shuffled_nlls["standard"],
        ),
        "top_confidence": summarize_texts(
            records["top_confidence"]["texts"],
            references,
            top_nll,
            counts["top_confidence"],
            shuffled_nlls["top_confidence"],
        ),
        "mean_random": summarize_texts(
            flattened_random_texts,
            flattened_references,
            random_nll.flatten(),
            random_counts.flatten(),
            flattened_shuffled_nll,
        ),
        "oracle_best_of_m": summarize_texts(
            best_texts,
            references,
            best_nll,
            best_counts,
            best_shuffled_nll,
        ),
        "oracle_worst_of_m": summarize_texts(
            worst_texts,
            references,
            worst_nll,
            worst_counts,
            worst_shuffled_nll,
        ),
        "oracle_best_vs_mean_random_ppl_improvement_pct": oracle_improvement_pct,
        "probability_random_beats_top_confidence": float(
            (random_nll < top_nll.unsqueeze(0)).double().mean()
        ),
        "mean_trajectory_utility_iqr_nats": float(trajectory_iqr.mean()),
        "headroom_bootstrap": bootstrap_headroom(
            mean_random_nll, best_nll, args.bootstrap_samples, args.seed
        ),
        "winning_mask_histogram": {
            name: int((best_indices == index).sum())
            for index, name in enumerate(random_names)
        },
        "mean_anchor_revision": float(
            np.mean(
                [
                    mean_stat(records[name]["stats"], "anchor_revision")
                    for name in random_names
                ]
            )
        ),
    }
    print(json.dumps(aggregate, indent=2), flush=True)

    payload = {
        **vars(args),
        "model": "plaid",
        "dataset": dataset_name,
        "model_native_trigger_t": grid[args.trigger_step],
        "paired_initial_and_ancestral_noise": True,
        "mask_seed_is_independent_of_solver_seed": True,
        "native_reference_agreement": reference_agreement,
        "oracle_is_deployable": False,
        "headroom_gate_passed": oracle_improvement_pct >= 5.0,
        "aggregate": aggregate,
        "per_sequence": {
            "standard_nll": nlls["standard"].tolist(),
            "top_confidence_nll": top_nll.tolist(),
            "random_nll": random_nll.tolist(),
            "random_token_counts": random_counts.tolist(),
            "best_mask_index": best_indices.tolist(),
            "best_nll": best_nll.tolist(),
            "worst_nll": worst_nll.tolist(),
        },
        "texts": {
            "standard": records["standard"]["texts"],
            "top_confidence": records["top_confidence"]["texts"],
            "oracle_best": best_texts,
            "oracle_worst": worst_texts,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / (
        f"{args.label}_d{str(args.density).replace('.', 'p')}_seed{args.seed}.json"
    )
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()
