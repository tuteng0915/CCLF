#!/usr/bin/env python3
"""EXP-81: sample-level and suffix-band prompt-use decomposition.

This is analysis-only: it reuses completed EXP-80 continuations and rebuilds
their exact true/shuffled prompts.  It never regenerates model samples.
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
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import paired_conditional_revalidation_exp80 as exp80
import unified_method_eval_exp64 as common


BANDS = {
    "boundary_1_8": (0, 8),
    "middle_9_32": (8, 32),
    "late_33_plus": (32, None),
    "full": (0, None),
}
OUT_DIR = Path("results/exp81_prompt_use_decomposition")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=["standard32", "unlock4"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--label", default="p1")
    return parser.parse_args()


def resolve_inputs(patterns):
    paths = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.exists():
            paths.append(candidate)
            continue
        matches = sorted(Path().glob(pattern))
        paths.extend(matches)
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise FileNotFoundError("no EXP-80 inputs matched")
    return unique


def rebuild_prompts(payload, tokenizer):
    n_cond = int(payload["n_cond"])
    prefix_length = int(payload["prefix_length"])
    suffix_length = int(payload["max_length"]) - prefix_length
    if payload["conditional_dataset"] == "owt":
        pairs = exp80.load_owt_pairs(
            n_cond,
            prefix_length,
            suffix_length,
            int(payload["owt_offset"]),
        )
    else:
        pairs = common.get_gutenberg_pairs(
            tokenizer, n_cond, prefix_length, suffix_length
        )
    targets = exp80.prefix_targets(pairs, prefix_length, torch.device("cpu"))
    return common.decode_texts(targets, tokenizer)


def encode_sequences(prefixes, suffixes, tokenizer, max_length):
    sequences, prefix_lengths, suffix_lengths = [], [], []
    for prefix, suffix in zip(prefixes, suffixes):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        separator = " " if prefix.strip() and suffix.strip() else ""
        suffix_ids = tokenizer.encode(separator + suffix, add_special_tokens=False)
        suffix_ids = suffix_ids[:max_length]
        prefix_budget = max(max_length - len(suffix_ids), 0)
        prefix_ids = prefix_ids[-prefix_budget:] if prefix_budget else []
        sequences.append(prefix_ids + suffix_ids)
        prefix_lengths.append(len(prefix_ids))
        suffix_lengths.append(len(suffix_ids))
    return sequences, prefix_lengths, suffix_lengths


@torch.no_grad()
def per_sample_band_nll(
    prefixes,
    suffixes,
    evaluator,
    tokenizer,
    device,
    batch_size,
    max_length,
):
    sequences, prefix_lengths, suffix_lengths = encode_sequences(
        prefixes, suffixes, tokenizer, max_length
    )
    output = {
        name: {"nll": [float("nan")] * len(sequences), "tokens": [0] * len(sequences)}
        for name in BANDS
    }
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        batch = sequences[start:end]
        width = max(max((len(row) for row in batch), default=0), 1)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        ordinal = torch.full_like(ids, -1)
        for local, row in enumerate(batch):
            if row:
                ids[local, : len(row)] = torch.tensor(row, dtype=torch.long)
                attention[local, : len(row)] = 1
            prefix_len = prefix_lengths[start + local]
            suffix_len = suffix_lengths[start + local]
            if suffix_len:
                ordinal[local, prefix_len : prefix_len + suffix_len] = torch.arange(
                    suffix_len, dtype=torch.long
                )
        ids_device = ids.to(device)
        logits = evaluator(
            input_ids=ids_device,
            attention_mask=attention.to(device),
        ).logits[:, :-1].float()
        targets = ids_device[:, 1:]
        token_nll = -F.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        shifted_ordinal = ordinal[:, 1:].to(device)
        for name, (lo, hi) in BANDS.items():
            mask = shifted_ordinal >= lo
            if hi is not None:
                mask &= shifted_ordinal < hi
            for local in range(len(batch)):
                values = token_nll[local][mask[local]]
                count = int(values.numel())
                output[name]["tokens"][start + local] = count
                if count:
                    output[name]["nll"][start + local] = float(
                        values.mean().item()
                    )
    return output


def finite_pair(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    return left[valid], right[valid]


def bootstrap_delta(standard, unlock, draws, rng):
    standard, unlock = finite_pair(standard, unlock)
    delta = unlock - standard
    if delta.size == 0:
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan")] * 2}
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        resample = rng.integers(0, delta.size, size=delta.size)
        samples[index] = delta[resample].mean()
    return {
        "n": int(delta.size),
        "mean": float(delta.mean()),
        "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "positive_fraction": float((delta > 0).mean()),
    }


def score_arm(texts, prompts, shuffled, evaluator, tokenizer, args):
    empty = [""] * len(texts)
    true = per_sample_band_nll(
        prompts, texts, evaluator, tokenizer, args.device,
        args.batch_size, args.max_length,
    )
    shuffled_score = per_sample_band_nll(
        shuffled, texts, evaluator, tokenizer, args.device,
        args.batch_size, args.max_length,
    )
    standalone = per_sample_band_nll(
        empty, texts, evaluator, tokenizer, args.device,
        args.batch_size, args.max_length,
    )
    result = {}
    for band in BANDS:
        true_nll = np.asarray(true[band]["nll"], dtype=np.float64)
        shuffled_nll = np.asarray(shuffled_score[band]["nll"], dtype=np.float64)
        result[band] = {
            "true_nll": true_nll.tolist(),
            "shuffled_nll": shuffled_nll.tolist(),
            "prompt_gain": (shuffled_nll - true_nll).tolist(),
            "standalone_nll": standalone[band]["nll"],
            "true_tokens": true[band]["tokens"],
            "shuffled_tokens": shuffled_score[band]["tokens"],
        }
    return result


def compact_panel(panel, draws, rng):
    summary = {}
    for band in BANDS:
        summary[band] = {}
        for metric in ("true_nll", "prompt_gain", "standalone_nll"):
            summary[band][f"unlock_minus_standard_{metric}"] = bootstrap_delta(
                panel["standard32"][band][metric],
                panel["unlock4"][band][metric],
                draws,
                rng,
            )
    return summary


def main():
    args = parse_args()
    paths = resolve_inputs(args.inputs)
    if args.bootstrap <= 0 or args.batch_size <= 0:
        raise ValueError("bootstrap and batch_size must be positive")
    device = torch.device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    prompt_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    scorer_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if scorer_tokenizer.pad_token is None:
        scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
        scorer_tokenizer.pad_token_id = scorer_tokenizer.eos_token_id
    evaluator = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()

    rng = np.random.default_rng(args.seed)
    panels, summaries = {}, {}
    pooled = {arm: {band: {metric: [] for metric in ("true_nll", "prompt_gain", "standalone_nll")} for band in BANDS} for arm in args.arms}
    for path in paths:
        payload = json.loads(path.read_text())
        prompts = rebuild_prompts(payload, prompt_tokenizer)
        shuffled = prompts[1:] + prompts[:1]
        panel_name = f"{payload['conditional_dataset']}_seed{payload['seed']}_offset{payload.get('owt_offset', 'na')}"
        panels[panel_name] = {}
        for arm in args.arms:
            if arm not in payload["results"]:
                raise KeyError(f"{arm} missing from {path}")
            texts = payload["results"][arm]["conditional"]["texts"]
            if len(texts) != len(prompts):
                raise RuntimeError(f"prompt/text mismatch for {path}: {len(prompts)} != {len(texts)}")
            panels[panel_name][arm] = score_arm(
                texts, prompts, shuffled, evaluator, scorer_tokenizer, args
            )
            for band in BANDS:
                for metric in pooled[arm][band]:
                    pooled[arm][band][metric].extend(
                        panels[panel_name][arm][band][metric]
                    )
        summaries[panel_name] = compact_panel(
            panels[panel_name], args.bootstrap, rng
        )
        print(panel_name, json.dumps(summaries[panel_name]["full"], indent=2))

    summaries["pooled"] = compact_panel(pooled, args.bootstrap, rng)
    print("pooled", json.dumps(summaries["pooled"], indent=2))
    output = {
        **vars(args),
        "inputs": [str(path) for path in paths],
        "bands_gpt2_suffix_tokens": BANDS,
        "panels": panels,
        "bootstrap_summary": summaries,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.label}_seed{args.seed}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
