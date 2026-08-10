#!/usr/bin/env python3
"""EXP-79: late-coupled two-block decoding on LangFlow and Plaid."""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GLOBAL_DIR = ROOT / "experiments" / "global_state"
PHASE_DIR = ROOT / "experiments" / "phase_transition"
for path in (HERE, GLOBAL_DIR, PHASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import decode_text, load_adapter  # noqa: E402


CONTEXTS = ("neutral", "raw", "continuous", "hard")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("langflow", "plaid"), required=True)
    parser.add_argument("--checkpoint", default="baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results/exp79_late_coupled_blocks")
    parser.add_argument("--label", default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--splits", type=int, nargs="+", default=[24, 28, 30])
    parser.add_argument("--contexts", choices=CONTEXTS, nargs="+", default=list(CONTEXTS))
    parser.add_argument("--ppl_model", default="gpt2-large")
    parser.add_argument("--ppl_batch_size", type=int, default=4)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_reference_gate", action="store_true")
    return parser.parse_args()


def phase_seed(seed, batch_index, phase, step):
    phase_code = {"full": 11, "a": 23, "b": 37, "joint": 53}[phase]
    return seed * 1000003 + batch_index * 10007 + phase_code * 1009 + step


def native_step(adapter, z, sc, t, t_next, seed):
    if adapter.name == "plaid":
        generator = torch.Generator(device=adapter.device).manual_seed(seed)
        return adapter.solver_step(z, sc, t, t_next, generator=generator)
    return adapter.solver_step(z, sc, t, t_next)


@torch.no_grad()
def advance(adapter, z, sc, grid, start, end, seed, batch_index, phase):
    for step in range(start, end):
        z, sc = native_step(
            adapter,
            z,
            sc,
            grid[step],
            grid[step + 1],
            phase_seed(seed, batch_index, phase, step),
        )
    return z, sc


@torch.no_grad()
def readout(adapter, z, sc, t, batch_size):
    out = adapter.forward_state(z, sc, t, batch_size=batch_size)
    return (
        out["logits"].argmax(-1),
        out["predicted_clean"].to(adapter.device),
    )


def neutral_token_id(adapter):
    if adapter.name == "langflow":
        return int(adapter.tokenizer.eos_token_id)
    return 0


@torch.no_grad()
def condition_tensor(adapter, kind, z_a, sc_a, ids_a, xhat_a):
    if kind == "raw":
        return z_a, sc_a
    if kind == "continuous":
        return xhat_a, xhat_a
    if kind == "hard":
        clean = adapter.encode_clean(ids_a).to(adapter.device)
        return clean, clean
    if kind == "neutral":
        ids = torch.full_like(ids_a, neutral_token_id(adapter))
        clean = adapter.encode_clean(ids).to(adapter.device)
        return clean, clean
    raise ValueError(kind)


@torch.no_grad()
def mature_suffix(
    adapter,
    eps_b,
    context_z,
    context_sc,
    grid,
    end,
    seed,
    batch_index,
):
    z_b, sc_b = eps_b.clone(), None
    max_restore_error = 0.0
    for step in range(end):
        if sc_b is None:
            sc_b = torch.zeros_like(z_b)
        sc_prefix = context_sc if context_sc is not None else torch.zeros_like(context_z)
        z_full = torch.cat([context_z, z_b], dim=1)
        sc_full = torch.cat([sc_prefix, sc_b], dim=1)
        z_next, sc_next = native_step(
            adapter,
            z_full,
            sc_full,
            grid[step],
            grid[step + 1],
            phase_seed(seed, batch_index, "b", step),
        )
        # Explicit clamping: model updates at prefix positions are discarded.
        restored = torch.cat([context_z, z_next[:, context_z.shape[1] :]], dim=1)
        max_restore_error = max(
            max_restore_error,
            float((restored[:, : context_z.shape[1]] - context_z).abs().max().item()),
        )
        z_b = z_next[:, context_z.shape[1] :]
        sc_b = sc_next[:, context_z.shape[1] :]
    return z_b, sc_b, max_restore_error


@torch.no_grad()
def run_batch(adapter, eps_a, eps_b, grid, args, batch_index):
    outputs = {}
    batch_size = eps_a.shape[0]
    total_eps = torch.cat([eps_a, eps_b], dim=1)

    # Full-parallel baseline and exact duplicate correctness gate.
    z_full, sc_full = advance(
        adapter, total_eps.clone(), None, grid, 0, args.n_steps,
        args.seed, batch_index, "full",
    )
    full_ids, _ = readout(adapter, z_full, sc_full, grid[-1], args.batch_size)
    outputs["full_parallel"] = {
        "ids": full_ids,
        "calls": args.n_steps,
        "prefix_revision": 0.0,
        "suffix_revision": 0.0,
        "clamp_error": 0.0,
    }
    reference_agreement = 1.0
    if not args.skip_reference_gate and batch_index == 0:
        z_ref, sc_ref = advance(
            adapter, total_eps.clone(), None, grid, 0, args.n_steps,
            args.seed, batch_index, "full",
        )
        ref_ids, _ = readout(adapter, z_ref, sc_ref, grid[-1], args.batch_size)
        reference_agreement = float((ref_ids == full_ids).float().mean().item())
        if reference_agreement != 1.0:
            raise RuntimeError(
                f"native reference gate failed: agreement={reference_agreement:.6f}"
            )

    # One shared A trajectory supplies every split and the finished SAR prefix.
    split_set = set(args.splits)
    a_cache = {}
    z_a, sc_a = eps_a.clone(), None
    for step in range(args.n_steps):
        z_a, sc_a = native_step(
            adapter,
            z_a,
            sc_a,
            grid[step],
            grid[step + 1],
            phase_seed(args.seed, batch_index, "a", step),
        )
        reached = step + 1
        if reached in split_set or reached == args.n_steps:
            ids_a, xhat_a = readout(
                adapter, z_a, sc_a, grid[reached], args.batch_size
            )
            a_cache[reached] = {
                "z": z_a.clone(),
                "sc": sc_a.clone() if sc_a is not None else None,
                "ids": ids_a.clone(),
                "xhat": xhat_a.clone(),
            }

    # Ordinary hard Block-SAR: A is final and cannot revise.
    a_final = a_cache[args.n_steps]
    hard_a = adapter.encode_clean(a_final["ids"]).to(adapter.device)
    z_b, sc_b, clamp_error = mature_suffix(
        adapter, eps_b, hard_a, hard_a, grid, args.n_steps,
        args.seed, batch_index,
    )
    sar_full_z = torch.cat([hard_a, z_b], dim=1)
    sar_full_sc = torch.cat([hard_a, sc_b], dim=1)
    sar_read_ids, _ = readout(
        adapter, sar_full_z, sar_full_sc, grid[-1], args.batch_size
    )
    sar_ids = torch.cat(
        [a_final["ids"], sar_read_ids[:, args.block_size :]], dim=1
    )
    outputs["block_sar"] = {
        "ids": sar_ids,
        "calls": 2 * args.n_steps,
        "prefix_revision": 0.0,
        "suffix_revision": 0.0,
        "clamp_error": clamp_error,
    }

    for split in args.splits:
        a = a_cache[split]
        for kind in args.contexts:
            context_z, context_sc = condition_tensor(
                adapter, kind, a["z"], a["sc"], a["ids"], a["xhat"]
            )
            z_b, sc_b, clamp_error = mature_suffix(
                adapter, eps_b, context_z, context_sc, grid, split,
                args.seed, batch_index,
            )
            suffix_ids_before, _ = readout(
                adapter,
                torch.cat([context_z, z_b], dim=1),
                torch.cat([
                    context_sc if context_sc is not None else torch.zeros_like(context_z),
                    sc_b,
                ], dim=1),
                grid[split],
                args.batch_size,
            )
            suffix_ids_before = suffix_ids_before[:, args.block_size :]

            # Raw A and B states are now synchronized at the same grid point.
            joint_z = torch.cat([a["z"], z_b], dim=1)
            a_sc = a["sc"] if a["sc"] is not None else torch.zeros_like(a["z"])
            joint_sc = torch.cat([a_sc, sc_b], dim=1)
            joint_z, joint_sc = advance(
                adapter, joint_z, joint_sc, grid, split, args.n_steps,
                args.seed, batch_index, "joint",
            )
            final_ids, _ = readout(
                adapter, joint_z, joint_sc, grid[-1], args.batch_size
            )
            outputs[f"late_{kind}_m{split}"] = {
                "ids": final_ids,
                "calls": args.n_steps + split,
                "prefix_revision": float(
                    (final_ids[:, : args.block_size] != a["ids"]).float().mean().item()
                ),
                "suffix_revision": float(
                    (final_ids[:, args.block_size :] != suffix_ids_before).float().mean().item()
                ),
                "clamp_error": clamp_error,
            }

    return outputs, reference_agreement


def distinct_n(texts, n):
    unique, total = set(), 0
    for text in texts:
        words = text.split()
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        unique.update(grams)
        total += len(grams)
    return len(unique) / total if total else 0.0


def repetition_rate(texts, n=4):
    values = []
    for text in texts:
        words = text.split()
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        values.append(1.0 - len(set(grams)) / len(grams) if grams else 0.0)
    return float(np.mean(values)) if values else 0.0


def text_quality(texts):
    degeneration, max_shares, unique_ratios, word_counts = [], [], [], []
    for text in texts:
        words = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
        word_counts.append(len(words))
        counts = Counter(words)
        max_share = max(counts.values()) / len(words) if words else 1.0
        max_shares.append(max_share)
        unique_ratios.append(len(counts) / len(words) if words else 0.0)
        non_ascii = sum(ord(char) > 127 for char in text) / len(text) if text else 1.0
        degeneration.append((not text.strip()) or max_share > 0.20 or non_ascii > 0.02)
    return {
        "d1": distinct_n(texts, 1),
        "d2": distinct_n(texts, 2),
        "rep4": repetition_rate(texts, 4),
        "degeneration_rate": float(np.mean(degeneration)),
        "mean_words": float(np.mean(word_counts)),
        "mean_max_word_fraction": float(np.mean(max_shares)),
        "mean_unique_word_ratio": float(np.mean(unique_ratios)),
        "unigram_collapse_rate": float(np.mean(np.asarray(max_shares) > 0.20)),
    }


class PPLEvaluator:
    def __init__(self, name, device, batch_size):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, texts, max_length):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        total_nll, total_tokens = 0.0, 0
        for start in range(0, len(texts), self.batch_size):
            ids = encoded["input_ids"][start : start + self.batch_size].to(self.device)
            mask = encoded["attention_mask"][start : start + self.batch_size].to(self.device)
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            targets = ids[:, 1:]
            valid = mask[:, 1:].bool()
            nll = torch.logsumexp(logits, dim=-1) - logits.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            total_nll += float(nll[valid].sum().item())
            total_tokens += int(valid.sum().item())
        return math.exp(total_nll / max(total_tokens, 1))


def release_generator(adapter):
    if adapter.name == "langflow":
        adapter.model.cpu()
    else:
        for module in adapter.modules.values():
            module.cpu()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    if any(split <= 0 or split >= args.n_steps for split in args.splits):
        raise ValueError("every split must lie strictly between 0 and n_steps")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    adapter = load_adapter(args.model, args.checkpoint, None, device)
    adapter.seq_len = 2 * args.block_size
    grid = np.linspace(adapter.t_eps, 0.999, args.n_steps + 1).tolist()

    records = {}
    reference_agreements = []
    for batch_index, start in enumerate(range(0, args.n_samples, args.batch_size)):
        size = min(args.batch_size, args.n_samples - start)
        generator = torch.Generator(device=device).manual_seed(
            args.seed * 100003 + batch_index
        )
        eps_a = adapter.sample_epsilon(
            (size, args.block_size, adapter.d_model), generator=generator
        )
        eps_b = adapter.sample_epsilon(
            (size, args.block_size, adapter.d_model), generator=generator
        )
        batch_outputs, agreement = run_batch(
            adapter, eps_a, eps_b, grid, args, batch_index
        )
        reference_agreements.append(agreement)
        for name, output in batch_outputs.items():
            target = records.setdefault(
                name,
                {"ids": [], "calls": output["calls"], "prefix_revision": [],
                 "suffix_revision": [], "clamp_error": []},
            )
            target["ids"].append(output["ids"].cpu())
            for metric in ("prefix_revision", "suffix_revision", "clamp_error"):
                target[metric].append(output[metric])
        print(f"completed batch {batch_index + 1}: {start + size}/{args.n_samples}")

    mask = torch.ones(2 * args.block_size, dtype=torch.long)
    for record in records.values():
        record["ids"] = torch.cat(record["ids"], dim=0)
        record["texts"] = [
            decode_text(adapter.tokenizer, ids, mask) for ids in record["ids"]
        ]

    release_generator(adapter)
    ppl = None if args.skip_ppl else PPLEvaluator(
        args.ppl_model, device, args.ppl_batch_size
    )
    results = {}
    for name, record in records.items():
        quality = text_quality(record["texts"])
        quality["ppl"] = (
            float("nan") if ppl is None else ppl(record["texts"], 2 * args.block_size)
        )
        quality.update({
            "calls": record["calls"],
            "prefix_revision_rate": float(np.mean(record["prefix_revision"])),
            "suffix_revision_rate": float(np.mean(record["suffix_revision"])),
            "max_clamp_restore_error": float(max(record["clamp_error"])),
            "texts": record["texts"],
        })
        results[name] = quality
        print(
            f"{name:24s} calls={quality['calls']:2d} ppl={quality['ppl']:.2f} "
            f"d2={quality['d2']:.4f} rep4={quality['rep4']:.4f} "
            f"deg={quality['degeneration_rate']:.3f} "
            f"prefix_rev={quality['prefix_revision_rate']:.3f}"
        )

    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "n_steps": args.n_steps,
        "splits": args.splits,
        "contexts": args.contexts,
        "grid_start": grid[0],
        "grid_end": grid[-1],
        "ppl_model": None if args.skip_ppl else args.ppl_model,
        "native_reference_agreement": min(reference_agreements),
        "paired_plaid_step_noise": args.model == "plaid",
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{args.label}_{args.model}_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
