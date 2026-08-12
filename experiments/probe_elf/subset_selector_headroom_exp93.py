#!/usr/bin/env python3
"""EXP-93: paired best-of-M random-subset selector headroom on ELF.

This is an oracle diagnostic, not a deployable decoding method. Every mask is
evaluated with the same prompt, initial latent noise, trigger, density, anchor
content, and lock horizon. Only the random subset identity changes.
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import late_coupled_blocks_exp79 as exp79
import paired_conditional_revalidation_exp80 as exp80
import transition_unlock_pareto_exp82 as exp82
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from modules.t5_encoder import get_encoder


OUT_DIR = Path("results/exp93_subset_selector_headroom")
CHECKPOINTS = {
    name: common.CHECKPOINTS[name]
    for name in ("baseline", "ct_control", "kd_early")
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="baseline")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_cond", type=int, default=64)
    parser.add_argument("--n_masks", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--conditional_dataset", choices=("owt", "gutenberg"), default="owt")
    parser.add_argument("--owt_offset", type=int, default=30000)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--label", default="p0")
    return parser.parse_args()


@torch.no_grad()
def conditional_sequence_nlls(
    prefix_texts, suffix_texts, evaluator, tokenizer, device, max_length=1024
):
    """Return suffix-only mean NLL and token count for each continuation."""
    sequences, prefix_lengths = [], []
    for prefix, suffix in zip(prefix_texts, suffix_texts):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        separator = " " if prefix.strip() and suffix.strip() else ""
        suffix_ids = tokenizer.encode(separator + suffix, add_special_tokens=False)
        prefix_budget = max(max_length - len(suffix_ids), 0)
        prefix_ids = prefix_ids[-prefix_budget:] if prefix_budget else []
        sequences.append(prefix_ids + suffix_ids)
        prefix_lengths.append(len(prefix_ids))

    width = max((len(sequence) for sequence in sequences), default=1)
    ids = torch.full(
        (len(sequences), width), tokenizer.pad_token_id, dtype=torch.long
    )
    attention = torch.zeros_like(ids)
    suffix_mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, (sequence, prefix_length) in enumerate(zip(sequences, prefix_lengths)):
        if not sequence:
            continue
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention[row, : len(sequence)] = 1
        suffix_mask[row, prefix_length : len(sequence)] = True

    sums = torch.zeros(len(sequences), dtype=torch.float64)
    counts = torch.zeros(len(sequences), dtype=torch.long)
    for start in range(0, len(sequences), 8):
        end = min(start + 8, len(sequences))
        ids_b = ids[start:end].to(device)
        attention_b = attention[start:end].to(device)
        valid = suffix_mask[start:end, 1:].to(device)
        logits = evaluator(input_ids=ids_b, attention_mask=attention_b).logits[:, :-1].float()
        targets = ids_b[:, 1:]
        token_nll = -F.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        sums[start:end] = (token_nll * valid).sum(dim=1).double().cpu()
        counts[start:end] = valid.sum(dim=1).long().cpu()
    if (counts == 0).any():
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"conditional evaluator found empty suffixes at rows {empty}")
    return sums / counts.double(), counts


def aggregate_ppl(nlls, counts):
    nlls = torch.as_tensor(nlls, dtype=torch.float64)
    counts = torch.as_tensor(counts, dtype=torch.float64)
    return math.exp(float((nlls * counts).sum() / counts.sum()))


def quantile(values, q):
    return float(torch.quantile(torch.as_tensor(values, dtype=torch.float64), q).item())


def summarize_texts(texts, references, nlls, counts):
    metrics = exp79.text_metrics_without_ppl(texts)
    metrics["prompt_conditioned_ppl"] = aggregate_ppl(nlls, counts)
    metrics["mean_sequence_nll"] = float(torch.as_tensor(nlls).double().mean().item())
    metrics["rouge_l"] = sum(
        common.rouge_l_f1(hypothesis, reference)
        for hypothesis, reference in zip(texts, references)
    ) / len(references)
    metrics["samples"] = texts[:4]
    return metrics


def bootstrap_headroom(mean_random_nll, best_nll, samples, seed):
    per_sequence = (mean_random_nll - best_nll).double()
    generator = torch.Generator().manual_seed(seed + 930017)
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(
            0, per_sequence.numel(), (per_sequence.numel(),), generator=generator
        )
        estimates[index] = per_sequence[rows].mean()
    return {
        "mean_nats": float(per_sequence.mean().item()),
        "ci95_nats": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
        "positive_trajectory_fraction": float((per_sequence > 0).double().mean().item()),
    }


def main():
    args = parse_args()
    if args.n_steps != 32:
        raise ValueError("EXP-93 fixes ODE-32")
    if not 0 < args.prefix_length < args.max_length:
        raise ValueError("prefix_length must lie inside max_length")
    if min(args.n_cond, args.n_masks, args.batch_size) <= 0:
        raise ValueError("sample counts, masks, and batch size must be positive")
    if args.skip_ppl:
        raise ValueError("EXP-93 oracle ranking requires conditional GPT-2 NLL")

    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    from transformers import T5Tokenizer

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else REPO_ROOT / CHECKPOINTS[args.checkpoint]
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(args.max_length))
    missing, unexpected = model.load_state_dict(common.load_weights(checkpoint), strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder.to(device).eval()

    pairs = exp80.load_pairs(args, tokenizer)
    cond_seq, cond_mask, references = common.build_condition_data(
        pairs, tokenizer, encoder, device, args.max_length, args.prefix_length
    )
    targets = exp80.prefix_targets(pairs, args.prefix_length, device)
    prompts = common.decode_texts(targets.cpu(), tokenizer)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cond_noise = args.noise_scale * torch.randn(
        args.n_cond,
        args.max_length,
        common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    cond_noise[:, : args.prefix_length] = cond_seq[:, : args.prefix_length]

    generated, rollout_info = {}, {}
    fixed_arms = ("standard32", "topq_t30_q50_h4")
    for arm in fixed_arms:
        print(f"[{arm}] conditional", flush=True)
        texts, info = exp82.generate_scope(
            arm, cond_noise, model, tokenizer, args, cond_seq, cond_mask, targets
        )
        generated[arm] = texts
        rollout_info[arm] = info
    random_names = []
    for mask_index in range(args.n_masks):
        name = f"random_{mask_index:02d}"
        mask_args = copy.copy(args)
        mask_args.seed = args.seed + (mask_index + 1) * 1009
        print(f"[{name}] conditional", flush=True)
        texts, info = exp82.generate_scope(
            "random_t30_q50_h4",
            cond_noise,
            model,
            tokenizer,
            mask_args,
            cond_seq,
            cond_mask,
            targets,
        )
        generated[name] = texts
        rollout_info[name] = info
        random_names.append(name)

    for name, info in rollout_info.items():
        if info["max_prompt_clamp_error"] > 1e-6:
            raise RuntimeError(f"prompt clamp failed for {name}")
        if name.startswith("random_") or name.startswith("topq_"):
            if abs(info["anchor_fraction"] - 0.5) > 1e-6:
                raise RuntimeError(f"anchor density drifted for {name}: {info['anchor_fraction']}")

    model.cpu()
    encoder.cpu()
    del model, encoder, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    evaluator = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()

    nlls, token_counts = {}, {}
    summaries = {}
    for name, texts in generated.items():
        print(f"[{name}] conditional NLL", flush=True)
        nll, counts = conditional_sequence_nlls(
            prompts, texts, evaluator, ppl_tokenizer, device
        )
        nlls[name], token_counts[name] = nll, counts
        summaries[name] = summarize_texts(texts, references, nll, counts)
        summaries[name].update(rollout_info[name])

    random_nll = torch.stack([nlls[name] for name in random_names])
    random_counts = torch.stack([token_counts[name] for name in random_names])
    best_indices = random_nll.argmin(dim=0)
    worst_indices = random_nll.argmax(dim=0)
    rows = torch.arange(args.n_cond)
    best_nll = random_nll[best_indices, rows]
    worst_nll = random_nll[worst_indices, rows]
    best_counts = random_counts[best_indices, rows]
    worst_counts = random_counts[worst_indices, rows]
    best_texts = [generated[random_names[int(best_indices[i])]][i] for i in range(args.n_cond)]
    worst_texts = [generated[random_names[int(worst_indices[i])]][i] for i in range(args.n_cond)]
    mean_random_nll = random_nll.mean(dim=0)
    median_random_nll = random_nll.median(dim=0).values

    mean_random_ppl = aggregate_ppl(random_nll.flatten(), random_counts.flatten())
    median_random_ppl = math.exp(float(median_random_nll.mean().item()))
    oracle_best_ppl = aggregate_ppl(best_nll, best_counts)
    oracle_worst_ppl = aggregate_ppl(worst_nll, worst_counts)
    oracle_improvement_pct = 100 * (mean_random_ppl - oracle_best_ppl) / mean_random_ppl
    topconf_nll = nlls["topq_t30_q50_h4"]
    random_beats_topconf = float((random_nll < topconf_nll.unsqueeze(0)).double().mean().item())
    utility = nlls["standard32"].unsqueeze(0) - random_nll
    trajectory_iqr = torch.quantile(utility, 0.75, dim=0) - torch.quantile(utility, 0.25, dim=0)

    aggregate = {
        "standard": summaries["standard32"],
        "top_confidence": summaries["topq_t30_q50_h4"],
        "mean_random": {
            "prompt_conditioned_ppl": mean_random_ppl,
            "mean_sequence_nll": float(mean_random_nll.mean().item()),
            "mean_anchor_revision_rate": sum(
                rollout_info[name]["anchor_revision_rate"] for name in random_names
            ) / len(random_names),
        },
        "median_random": {
            "sequence_median_ppl_proxy": median_random_ppl,
            "mean_sequence_nll": float(median_random_nll.mean().item()),
        },
        "oracle_best_of_m": summarize_texts(
            best_texts, references, best_nll, best_counts
        ),
        "oracle_worst_of_m": summarize_texts(
            worst_texts, references, worst_nll, worst_counts
        ),
        "oracle_worst_prompt_conditioned_ppl": oracle_worst_ppl,
        "oracle_best_vs_mean_random_ppl_improvement_pct": oracle_improvement_pct,
        "probability_random_beats_top_confidence": random_beats_topconf,
        "mean_trajectory_utility_iqr_nats": float(trajectory_iqr.mean().item()),
        "headroom_bootstrap": bootstrap_headroom(
            mean_random_nll, best_nll, args.bootstrap_samples, args.seed
        ),
        "winning_mask_histogram": {
            random_names[index]: int((best_indices == index).sum().item())
            for index in range(args.n_masks)
        },
    }
    print(json.dumps(aggregate, indent=2))

    output = {
        **vars(args),
        "checkpoint_path": str(checkpoint_path),
        "oracle_is_deployable": False,
        "paired_suffix_noise": True,
        "fixed_policy": {"trigger": 0.30, "density": 0.50, "hold_horizon": 4},
        "aggregate": aggregate,
        "per_mask": {name: summaries[name] for name in random_names},
        "rollout_info": rollout_info,
        "per_sequence": {
            "standard_nll": nlls["standard32"].tolist(),
            "top_confidence_nll": topconf_nll.tolist(),
            "random_nll": random_nll.tolist(),
            "random_token_counts": random_counts.tolist(),
            "best_mask_index": best_indices.tolist(),
            "best_nll": best_nll.tolist(),
            "worst_nll": worst_nll.tolist(),
        },
        "texts": generated,
        "oracle_best_texts": best_texts,
        "oracle_worst_texts": worst_texts,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
