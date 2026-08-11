"""EXP-86: fixed-prefix conditional verification of Plaid late coupling."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GLOBAL_DIR = ROOT / "experiments" / "global_state"
PHASE_DIR = ROOT / "experiments" / "phase_transition"
for path in (HERE, GLOBAL_DIR, PHASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_late_coupled_blocks as base  # noqa: E402
from common import decode_text, load_adapter  # noqa: E402


ARMS = ("full_parallel", "block_sar", "late_raw_m24", "late_continuous_m24", "late_hard_m24")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results/exp86_plaid_conditional_late_coupling")
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--split", type=int, default=24)
    parser.add_argument("--ppl_model", default="gpt2-large")
    parser.add_argument("--ppl_batch_size", type=int, default=4)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_reference_gate", action="store_true")
    return parser.parse_args()


def decode_ids(adapter, ids):
    return adapter.tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def rouge_l_f1(prediction, reference):
    pred = prediction.split()
    ref = reference.split()
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for token in pred:
        current = [0]
        for index, ref_token in enumerate(ref, start=1):
            if token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall + 1e-12)


def load_gutenberg_panel(adapter, n_samples, total_length):
    import nltk

    nltk.download("gutenberg", quiet=True)
    from nltk.corpus import gutenberg

    sequences = []
    for filename in gutenberg.fileids():
        words = re.split(r"\s+", gutenberg.raw(filename).strip())
        window_words = total_length * 2
        for start in range(0, max(len(words) - window_words, 0), window_words):
            text = " ".join(words[start : start + window_words])
            ids = adapter.tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) >= total_length:
                sequences.append(ids[:total_length])
            if len(sequences) >= n_samples:
                return torch.tensor(sequences, dtype=torch.long)
    raise RuntimeError(f"requested {n_samples} Gutenberg sequences, found {len(sequences)}")


@torch.no_grad()
def advance_clamped(
    adapter,
    z,
    sc,
    grid,
    start,
    end,
    seed,
    batch_index,
    phase,
    prompt_clean,
    prefix_length,
):
    max_error = 0.0
    if sc is None:
        sc = torch.zeros_like(z)
    z[:, :prefix_length] = prompt_clean
    sc[:, :prefix_length] = prompt_clean
    for step in range(start, end):
        z, sc = base.native_step(
            adapter,
            z,
            sc,
            grid[step],
            grid[step + 1],
            base.phase_seed(seed, batch_index, phase, step),
        )
        z[:, :prefix_length] = prompt_clean
        sc[:, :prefix_length] = prompt_clean
        max_error = max(
            max_error,
            float((z[:, :prefix_length] - prompt_clean).abs().max().item()),
        )
    return z, sc, max_error


@torch.no_grad()
def run_batch(adapter, eps_a, eps_b, prompt_ids, prompt_clean, grid, args, batch_index):
    outputs = {}
    total_eps = torch.cat([eps_a, eps_b], dim=1)
    total_eps[:, : args.prefix_length] = prompt_clean

    z_full, sc_full, full_error = advance_clamped(
        adapter,
        total_eps.clone(),
        None,
        grid,
        0,
        args.n_steps,
        args.seed,
        batch_index,
        "full",
        prompt_clean,
        args.prefix_length,
    )
    full_ids, _ = base.readout(adapter, z_full, sc_full, grid[-1], args.batch_size)
    outputs["full_parallel"] = {
        "ids": full_ids,
        "calls": args.n_steps,
        "prefix_revision": 0.0,
        "suffix_revision": 0.0,
        "clamp_error": full_error,
    }
    agreement = 1.0
    if not args.skip_reference_gate and batch_index == 0:
        z_ref, sc_ref, _ = advance_clamped(
            adapter,
            total_eps.clone(),
            None,
            grid,
            0,
            args.n_steps,
            args.seed,
            batch_index,
            "full",
            prompt_clean,
            args.prefix_length,
        )
        ref_ids, _ = base.readout(adapter, z_ref, sc_ref, grid[-1], args.batch_size)
        agreement = float((ref_ids == full_ids).float().mean())
        if agreement != 1.0:
            raise RuntimeError(f"native reference agreement={agreement}")

    # First generated block contains the observed prompt followed by region A.
    z_a, sc_a = eps_a.clone(), None
    z_a[:, : args.prefix_length] = prompt_clean
    a_cache = {}
    for step in range(args.n_steps):
        z_a, sc_a, _ = advance_clamped(
            adapter,
            z_a,
            sc_a,
            grid,
            step,
            step + 1,
            args.seed,
            batch_index,
            "a",
            prompt_clean,
            args.prefix_length,
        )
        reached = step + 1
        if reached in (args.split, args.n_steps):
            ids_a, xhat_a = base.readout(
                adapter, z_a, sc_a, grid[reached], args.batch_size
            )
            a_cache[reached] = {
                "z": z_a.clone(),
                "sc": sc_a.clone(),
                "ids": ids_a.clone(),
                "xhat": xhat_a.clone(),
            }

    a_final = a_cache[args.n_steps]
    hard_a = adapter.encode_clean(a_final["ids"]).to(adapter.device)
    hard_a[:, : args.prefix_length] = prompt_clean
    z_b, sc_b, sar_error = base.mature_suffix(
        adapter,
        eps_b,
        hard_a,
        hard_a,
        grid,
        args.n_steps,
        args.seed,
        batch_index,
    )
    sar_read_ids, _ = base.readout(
        adapter,
        torch.cat([hard_a, z_b], dim=1),
        torch.cat([hard_a, sc_b], dim=1),
        grid[-1],
        args.batch_size,
    )
    sar_ids = torch.cat(
        [a_final["ids"], sar_read_ids[:, args.block_size :]], dim=1
    )
    outputs["block_sar"] = {
        "ids": sar_ids,
        "calls": 2 * args.n_steps,
        "prefix_revision": 0.0,
        "suffix_revision": 0.0,
        "clamp_error": sar_error,
    }

    a = a_cache[args.split]
    for kind in ("raw", "continuous", "hard"):
        context_z, context_sc = base.condition_tensor(
            adapter, kind, a["z"], a["sc"], a["ids"], a["xhat"]
        )
        context_z[:, : args.prefix_length] = prompt_clean
        context_sc[:, : args.prefix_length] = prompt_clean
        z_b, sc_b, condition_error = base.mature_suffix(
            adapter,
            eps_b,
            context_z,
            context_sc,
            grid,
            args.split,
            args.seed,
            batch_index,
        )
        before_ids, _ = base.readout(
            adapter,
            torch.cat([context_z, z_b], dim=1),
            torch.cat([context_sc, sc_b], dim=1),
            grid[args.split],
            args.batch_size,
        )
        before_suffix = before_ids[:, args.block_size :]

        joint_z = torch.cat([a["z"], z_b], dim=1)
        joint_sc = torch.cat([a["sc"], sc_b], dim=1)
        joint_z, joint_sc, joint_error = advance_clamped(
            adapter,
            joint_z,
            joint_sc,
            grid,
            args.split,
            args.n_steps,
            args.seed,
            batch_index,
            "joint",
            prompt_clean,
            args.prefix_length,
        )
        final_ids, _ = base.readout(
            adapter, joint_z, joint_sc, grid[-1], args.batch_size
        )
        outputs[f"late_{kind}_m{args.split}"] = {
            "ids": final_ids,
            "calls": args.n_steps + args.split,
            "prefix_revision": float(
                (
                    final_ids[:, args.prefix_length : args.block_size]
                    != a["ids"][:, args.prefix_length : args.block_size]
                ).float().mean()
            ),
            "suffix_revision": float(
                (final_ids[:, args.block_size :] != before_suffix).float().mean()
            ),
            "clamp_error": max(condition_error, joint_error),
        }

    for output in outputs.values():
        output["decoded_prompt_agreement"] = float(
            (output["ids"][:, : args.prefix_length] == prompt_ids).float().mean()
        )
    return outputs, agreement


@torch.no_grad()
def conditional_ppl(prefixes, suffixes, evaluator, suffix_tokens=None, max_length=1024):
    tokenizer = evaluator.tokenizer
    sequences, prefix_lengths = [], []
    for prefix, suffix in zip(prefixes, suffixes):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        separator = " " if prefix.strip() and suffix.strip() else ""
        suffix_ids = tokenizer.encode(separator + suffix, add_special_tokens=False)
        if suffix_tokens is not None:
            suffix_ids = suffix_ids[:suffix_tokens]
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
            ids[row, : len(sequence)] = torch.tensor(sequence)
            attention[row, : len(sequence)] = 1
            suffix_mask[row, prefix_length : len(sequence)] = True
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(sequences), evaluator.batch_size):
        ids_b = ids[start : start + evaluator.batch_size].to(evaluator.device)
        mask_b = attention[start : start + evaluator.batch_size].to(evaluator.device)
        valid = suffix_mask[start : start + evaluator.batch_size, 1:].to(evaluator.device)
        logits = evaluator.model(input_ids=ids_b, attention_mask=mask_b).logits[:, :-1].float()
        targets = ids_b[:, 1:]
        nll = -F.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        total_nll += float(nll[valid].sum())
        total_tokens += int(valid.sum())
    return math.exp(total_nll / total_tokens) if total_tokens else float("nan")


def main():
    args = parse_args()
    if not 0 < args.prefix_length < args.block_size:
        raise ValueError("prefix_length must lie inside the first block")
    if not 0 < args.split < args.n_steps:
        raise ValueError("split must lie inside the solver grid")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    adapter = load_adapter("plaid", "baseline", None, device)
    adapter.seq_len = 2 * args.block_size
    grid = np.linspace(adapter.t_eps, 0.999, args.n_steps + 1).tolist()
    panel_ids = load_gutenberg_panel(adapter, args.n_samples, 2 * args.block_size)
    prompt_ids_all = panel_ids[:, : args.prefix_length]
    prompt_clean_all = adapter.encode_clean(prompt_ids_all).to(device)

    records = {}
    reference_agreements = []
    for batch_index, start in enumerate(range(0, args.n_samples, args.batch_size)):
        end = min(start + args.batch_size, args.n_samples)
        size = end - start
        generator = torch.Generator(device=device).manual_seed(
            args.seed * 100003 + batch_index
        )
        eps_a = adapter.sample_epsilon(
            (size, args.block_size, adapter.d_model), generator=generator
        )
        eps_b = adapter.sample_epsilon(
            (size, args.block_size, adapter.d_model), generator=generator
        )
        outputs, agreement = run_batch(
            adapter,
            eps_a,
            eps_b,
            prompt_ids_all[start:end].to(device),
            prompt_clean_all[start:end],
            grid,
            args,
            batch_index,
        )
        reference_agreements.append(agreement)
        for name, output in outputs.items():
            target = records.setdefault(
                name,
                {
                    "ids": [],
                    "calls": output["calls"],
                    "prefix_revision": [],
                    "suffix_revision": [],
                    "clamp_error": [],
                    "decoded_prompt_agreement": [],
                },
            )
            target["ids"].append(output["ids"].cpu())
            for metric in (
                "prefix_revision",
                "suffix_revision",
                "clamp_error",
                "decoded_prompt_agreement",
            ):
                target[metric].append(output[metric])
        print(f"completed batch {batch_index + 1}: {end}/{args.n_samples}")

    prompts = [decode_ids(adapter, ids) for ids in prompt_ids_all]
    references = [
        decode_ids(adapter, ids[args.prefix_length :]) for ids in panel_ids
    ]
    for record in records.values():
        record["ids"] = torch.cat(record["ids"], dim=0)
        record["continuations"] = [
            decode_ids(adapter, ids[args.prefix_length :]) for ids in record["ids"]
        ]
        record["a_texts"] = [
            decode_ids(adapter, ids[args.prefix_length : args.block_size])
            for ids in record["ids"]
        ]
        record["b_texts"] = [
            decode_ids(adapter, ids[args.block_size :]) for ids in record["ids"]
        ]

    base.release_generator(adapter)
    evaluator = None if args.skip_ppl else base.PPLEvaluator(
        args.ppl_model, device, args.ppl_batch_size
    )
    shuffled_prompts = prompts[1:] + prompts[:1]
    results = {}
    for name, record in records.items():
        quality = base.text_quality(record["continuations"])
        quality["continuation_ppl"] = (
            float("nan")
            if evaluator is None
            else evaluator(record["continuations"], 2 * args.block_size)
        )
        if evaluator is None:
            prompt_ppl = shuffled_ppl = boundary_ppl = float("nan")
        else:
            prompt_ppl = conditional_ppl(prompts, record["continuations"], evaluator, None)
            shuffled_ppl = conditional_ppl(
                shuffled_prompts, record["continuations"], evaluator, None
            )
            prefix_for_b = [
                prompt + (" " if prompt.strip() and a.strip() else "") + a
                for prompt, a in zip(prompts, record["a_texts"])
            ]
            boundary_ppl = conditional_ppl(
                prefix_for_b, record["b_texts"], evaluator, suffix_tokens=32
            )
        quality.update(
            {
                "prompt_conditioned_ppl": prompt_ppl,
                "shuffled_prompt_ppl": shuffled_ppl,
                "prompt_gain_nats": (
                    math.log(shuffled_ppl) - math.log(prompt_ppl)
                    if prompt_ppl == prompt_ppl and shuffled_ppl == shuffled_ppl
                    else float("nan")
                ),
                "boundary_ppl": boundary_ppl,
                "rouge_l": float(
                    np.mean(
                        [
                            rouge_l_f1(prediction, reference)
                            for prediction, reference in zip(
                                record["continuations"], references
                            )
                        ]
                    )
                ),
                "calls": record["calls"],
                "prefix_revision_rate": float(np.mean(record["prefix_revision"])),
                "suffix_revision_rate": float(np.mean(record["suffix_revision"])),
                "max_clamp_restore_error": float(max(record["clamp_error"])),
                "decoded_prompt_agreement": float(
                    np.mean(record["decoded_prompt_agreement"])
                ),
                "texts": record["continuations"],
            }
        )
        results[name] = quality
        print(
            f"{name:24s} calls={quality['calls']:2d} "
            f"C-PPL={quality['prompt_conditioned_ppl']:.2f} "
            f"boundary={quality['boundary_ppl']:.2f} RL={quality['rouge_l']:.4f} "
            f"D2={quality['d2']:.4f} deg={quality['degeneration_rate']:.3f}"
        )

    payload = {
        **vars(args),
        "model": "plaid",
        "dataset": "gutenberg",
        "paired_suffix_and_step_noise": True,
        "native_reference_agreement": min(reference_agreements),
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.label}_plaid_seed{args.seed}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
