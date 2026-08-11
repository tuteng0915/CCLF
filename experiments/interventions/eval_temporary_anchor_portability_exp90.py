#!/usr/bin/env python3
"""EXP-90: paired U/C portability of temporary predicted-clean anchors.

This runner deliberately uses model-native solver steps and calibrates the
trigger by solver-step index rather than reusing ELF's nominal diffusion time.
Plaid arms share both initial and per-step ancestral noise.  Observed prompt
embeddings are restored after every solver call in conditional generation.
"""

import argparse
import json
import math
import sys
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

import eval_late_coupled_blocks as quality_base  # noqa: E402
import eval_plaid_conditional_late_coupling as conditional_base  # noqa: E402
from common import load_adapter  # noqa: E402


ARMS = ("standard", "random", "top_confidence", "shuffled_content")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("langflow", "plaid"), required=True)
    parser.add_argument("--checkpoint", default="baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results/exp90_anchor_portability")
    parser.add_argument("--label", default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_uncond", type=int, default=32)
    parser.add_argument("--n_cond", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--trigger_step", type=int, required=True)
    parser.add_argument("--density", type=float, default=0.50)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--ppl_model", default="gpt2-large")
    parser.add_argument("--ppl_batch_size", type=int, default=4)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_reference_gate", action="store_true")
    return parser.parse_args()


def step_seed(seed, scope_code, batch_index, step):
    return seed * 1000003 + scope_code * 100003 + batch_index * 1009 + step


def native_step(adapter, z, sc, t, t_next, seed):
    if adapter.name == "plaid":
        generator = torch.Generator(device=adapter.device).manual_seed(seed)
        return adapter.solver_step(z, sc, t, t_next, generator=generator)
    return adapter.solver_step(z, sc, t, t_next)


def decode_ids(adapter, ids):
    values = ids.detach().cpu().tolist()
    if adapter.name == "plaid":
        return adapter.tokenizer.decode(values, skip_special_tokens=True)
    return adapter.tokenizer.decode(values, skip_special_tokens=True)


def tokenize(adapter, text, seq_len):
    if adapter.name == "plaid":
        ids = adapter.tokenizer.encode(text, add_special_tokens=False).ids
    else:
        ids = adapter.tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < seq_len:
        return None
    return ids[:seq_len]


def load_conditional_panel(adapter, n_samples, seq_len):
    from datasets import load_dataset

    sources = (
        ("Skylion007/openwebtext", {}),
        ("stas/openwebtext-10k", {}),
        ("wikitext", {"name": "wikitext-103-raw-v1"}),
    )
    errors = []
    for name, kwargs in sources:
        sequences = []
        try:
            dataset = load_dataset(name, split="train", streaming=True, **kwargs)
            for example in dataset:
                text = example["text"].strip()
                if len(text) < 3 * seq_len:
                    continue
                ids = tokenize(adapter, text, seq_len)
                if ids is None:
                    continue
                sequences.append(ids)
                if len(sequences) == n_samples:
                    return torch.tensor(sequences, dtype=torch.long), name
        except Exception as error:
            errors.append(f"{name}: {error}")
    raise RuntimeError("conditional panel unavailable: " + " | ".join(errors))


def exact_mask(scores, eligible, density):
    """Select exactly ceil(density * eligible_count) positions per sample."""
    mask = torch.zeros_like(eligible, dtype=torch.bool)
    for row in range(scores.shape[0]):
        indices = torch.nonzero(eligible[row], as_tuple=False).flatten()
        count = max(1, int(math.ceil(float(density) * len(indices))))
        chosen = indices[torch.topk(scores[row, indices], k=count).indices]
        mask[row, chosen] = True
    return mask


def build_anchor(adapter, logits, predicted_clean, arm, eligible, density, seed):
    probs = torch.softmax(logits.float(), dim=-1)
    confidence, ids = probs.max(-1)
    if arm == "top_confidence":
        scores = confidence
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        scores = torch.rand(confidence.shape, generator=generator).to(confidence.device)
    mask = exact_mask(scores, eligible, density)
    content_ids = ids.clone()
    clean = predicted_clean.to(adapter.device).clone()
    if arm == "shuffled_content":
        for row in range(content_ids.shape[0]):
            selected = torch.nonzero(mask[row], as_tuple=False).flatten()
            if len(selected) > 1:
                content_ids[row, selected] = torch.roll(content_ids[row, selected], 1)
                clean[row, selected] = torch.roll(clean[row, selected], 1, dims=0)
    selected_conf = confidence[mask]
    return mask.to(adapter.device), content_ids.to(adapter.device), clean, float(selected_conf.mean())


@torch.no_grad()
def run_arm(adapter, eps, grid, args, arm, scope, batch_index, prompt_clean=None):
    z = eps.clone().to(adapter.device)
    sc = torch.zeros_like(z)
    prefix_length = 0 if prompt_clean is None else args.prefix_length
    if prompt_clean is not None:
        z[:, :prefix_length] = prompt_clean
        sc[:, :prefix_length] = prompt_clean

    eligible = torch.ones(z.shape[:2], dtype=torch.bool, device=adapter.device)
    eligible[:, :prefix_length] = False
    scope_code = 11 if scope == "unconditional" else 29
    anchor_mask = None
    anchor_ids = None
    anchor_clean = None
    anchor_confidence = None
    max_clamp_error = 0.0

    for step in range(args.n_steps):
        if step == args.trigger_step and arm != "standard":
            out = adapter.forward_state(
                z, sc, grid[step], batch_size=args.batch_size
            )
            anchor_mask, anchor_ids, anchor_clean, anchor_confidence = build_anchor(
                adapter,
                out["logits"].to(adapter.device),
                out["predicted_clean"].to(adapter.device),
                arm,
                eligible,
                args.density,
                step_seed(args.seed + 17, scope_code, batch_index, step),
            )

        active = (
            anchor_mask is not None
            and args.trigger_step <= step < args.trigger_step + args.horizon
        )
        if active:
            z[anchor_mask] = anchor_clean[anchor_mask]
            sc[anchor_mask] = anchor_clean[anchor_mask]

        z, sc = native_step(
            adapter,
            z,
            sc,
            grid[step],
            grid[step + 1],
            step_seed(args.seed, scope_code, batch_index, step),
        )
        if active:
            z[anchor_mask] = anchor_clean[anchor_mask]
            sc[anchor_mask] = anchor_clean[anchor_mask]
        if prompt_clean is not None:
            z[:, :prefix_length] = prompt_clean
            sc[:, :prefix_length] = prompt_clean
            max_clamp_error = max(
                max_clamp_error,
                float((z[:, :prefix_length] - prompt_clean).abs().max()),
            )

    final = adapter.forward_state(z, sc, grid[-1], batch_size=args.batch_size)
    final_ids = final["logits"].argmax(-1).to(adapter.device)
    if prompt_clean is not None:
        decoded_prompt = adapter.forward_state(
            prompt_clean,
            prompt_clean,
            grid[-1],
            batch_size=args.batch_size,
        )["logits"].argmax(-1).to(adapter.device)
        final_ids[:, :prefix_length] = decoded_prompt

    if anchor_mask is None:
        fraction = revision = 0.0
    else:
        fraction = float(anchor_mask.float().sum() / eligible.float().sum())
        revision = float((final_ids[anchor_mask] != anchor_ids[anchor_mask]).float().mean())
    return {
        "ids": final_ids.cpu(),
        "anchor_fraction": fraction,
        "anchor_confidence": anchor_confidence,
        "anchor_revision": revision,
        "max_prompt_clamp_error": max_clamp_error,
        "denoiser_calls": args.n_steps,
        "readout_calls": int(arm != "standard"),
    }


def evaluate_scope(adapter, args, grid, scope, panel_ids=None):
    count = args.n_uncond if scope == "unconditional" else args.n_cond
    records = {arm: {"ids": [], "stats": []} for arm in args.arms}
    reference_agreements = []
    for batch_index, start in enumerate(range(0, count, args.batch_size)):
        end = min(start + args.batch_size, count)
        size = end - start
        generator = torch.Generator(device=adapter.device).manual_seed(
            args.seed * 10007 + (0 if scope == "unconditional" else 700001) + batch_index
        )
        eps = adapter.sample_epsilon(
            (size, args.seq_len, adapter.d_model), generator=generator
        )
        prompt_clean = None
        if scope == "conditional":
            prompt_ids = panel_ids[start:end, : args.prefix_length]
            prompt_clean = adapter.encode_clean(prompt_ids).to(adapter.device)
        for arm in args.arms:
            result = run_arm(
                adapter, eps, grid, args, arm, scope, batch_index, prompt_clean
            )
            if scope == "conditional":
                result["decoded_prompt_agreement"] = float(
                    (
                        result["ids"][:, : args.prefix_length]
                        == panel_ids[start:end, : args.prefix_length]
                    )
                    .float()
                    .mean()
                )
            records[arm]["ids"].append(result.pop("ids"))
            records[arm]["stats"].append(result)
        if batch_index == 0 and "standard" in args.arms and not args.skip_reference_gate:
            duplicate = run_arm(
                adapter, eps, grid, args, "standard", scope, batch_index, prompt_clean
            )
            agreement = float(
                (duplicate["ids"] == records["standard"]["ids"][-1]).float().mean()
            )
            reference_agreements.append(agreement)
            if agreement != 1.0:
                raise RuntimeError(
                    f"{scope} native reference agreement={agreement:.8f}"
                )
        print(f"{scope}: completed {end}/{count}")

    for record in records.values():
        record["ids"] = torch.cat(record["ids"], dim=0)
    return records, min(reference_agreements) if reference_agreements else None


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
        raise ValueError("anchor horizon exceeds solver grid")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    adapter = load_adapter(args.model, args.checkpoint, None, device)
    adapter.seq_len = args.seq_len
    grid = np.linspace(adapter.t_eps, 0.999, args.n_steps + 1).tolist()

    panel_ids, dataset_name = load_conditional_panel(adapter, args.n_cond, args.seq_len)
    unconditional, u_reference = evaluate_scope(
        adapter, args, grid, "unconditional"
    )
    conditional, c_reference = evaluate_scope(
        adapter, args, grid, "conditional", panel_ids
    )

    prompts = [decode_ids(adapter, row[: args.prefix_length]) for row in panel_ids]
    references = [decode_ids(adapter, row[args.prefix_length :]) for row in panel_ids]
    for record in unconditional.values():
        record["texts"] = [decode_ids(adapter, row) for row in record["ids"]]
    for record in conditional.values():
        record["texts"] = [
            decode_ids(adapter, row[args.prefix_length :]) for row in record["ids"]
        ]

    quality_base.release_generator(adapter)
    evaluator = None if args.skip_ppl else quality_base.PPLEvaluator(
        args.ppl_model, device, args.ppl_batch_size
    )
    shuffled_prompts = prompts[1:] + prompts[:1]
    results = {}
    for arm in args.arms:
        u_record = unconditional[arm]
        c_record = conditional[arm]
        u_quality = quality_base.text_quality(u_record["texts"])
        c_quality = quality_base.text_quality(c_record["texts"])
        if evaluator is None:
            u_ppl = c_ppl = shuffled_ppl = float("nan")
        else:
            u_ppl = evaluator(u_record["texts"], args.seq_len)
            c_ppl = conditional_base.conditional_ppl(
                prompts, c_record["texts"], evaluator, None
            )
            shuffled_ppl = conditional_base.conditional_ppl(
                shuffled_prompts, c_record["texts"], evaluator, None
            )
        u_quality["ppl"] = u_ppl
        c_quality.update(
            {
                "prompt_conditioned_ppl": c_ppl,
                "shuffled_prompt_ppl": shuffled_ppl,
                "prompt_gain_nats": (
                    math.log(shuffled_ppl) - math.log(c_ppl)
                    if c_ppl == c_ppl and shuffled_ppl == shuffled_ppl
                    else float("nan")
                ),
                "rouge_l": float(
                    np.mean(
                        [
                            conditional_base.rouge_l_f1(pred, ref)
                            for pred, ref in zip(c_record["texts"], references)
                        ]
                    )
                ),
            }
        )
        stats = c_record["stats"]
        results[arm] = {
            "unconditional": u_quality,
            "conditional": c_quality,
            "anchor_fraction": mean_stat(stats, "anchor_fraction"),
            "anchor_confidence": mean_stat(stats, "anchor_confidence"),
            "anchor_revision": mean_stat(stats, "anchor_revision"),
            "max_prompt_clamp_error": max(
                row["max_prompt_clamp_error"] for row in stats
            ),
            "decoded_prompt_agreement": mean_stat(
                stats, "decoded_prompt_agreement"
            ),
            "denoiser_calls": args.n_steps,
            "readout_calls": int(arm != "standard"),
        }
        print(
            f"{arm:18s} U-PPL={u_ppl:.2f} C-PPL={c_ppl:.2f} "
            f"gain={results[arm]['conditional']['prompt_gain_nats']:.4f} "
            f"C-D1={c_quality['d1']:.4f} C-Rep4={c_quality['rep4']:.4f}"
        )

    payload = {
        **vars(args),
        "dataset": dataset_name,
        "model_native_trigger_t": grid[args.trigger_step],
        "paired_initial_noise": True,
        "paired_plaid_step_noise": args.model == "plaid",
        "native_reference_agreement": {
            "unconditional": u_reference,
            "conditional": c_reference,
        },
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.label}_{args.model}_seed{args.seed}.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
