#!/usr/bin/env python3
"""EXP-107: select a short anchor/control shadow under realized Plaid noise."""

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
for path in (HERE, GLOBAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_late_coupled_blocks as quality_base  # noqa: E402
import eval_plaid_conditional_late_coupling as conditional_base  # noqa: E402
import eval_plaid_subset_headroom_exp99 as exp99  # noqa: E402
import eval_temporary_anchor_portability_exp90 as exp90  # noqa: E402
from common import load_adapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results/exp107_plaid_pathwise_shadow")
    parser.add_argument("--label", default="pilot")
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--panel_offset", type=int, default=9000)
    parser.add_argument("--n_cond", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--trigger_step", type=int, default=14)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--density", type=float, default=0.75)
    parser.add_argument("--ppl_model", default="gpt2-large")
    parser.add_argument("--ppl_batch_size", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    return parser.parse_args()


def setting_args(args):
    class Settings:
        pass

    settings = Settings()
    for name, value in vars(args).items():
        setattr(settings, name, value)
    settings.model = "plaid"
    settings.checkpoint = "baseline"
    settings.horizon = 1
    return settings


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def run_shadow(adapter, eps, grid, args, batch_index, prompt_clean):
    prefix = args.prefix_length
    route = 29
    z = eps.clone().to(adapter.device)
    sc = torch.zeros_like(z)
    z[:, :prefix] = prompt_clean
    sc[:, :prefix] = prompt_clean

    for step in range(args.trigger_step):
        z, sc = exp90.native_step(
            adapter,
            z,
            sc,
            grid[step],
            grid[step + 1],
            exp90.step_seed(args.seed, route, batch_index, step),
        )
        z[:, :prefix] = prompt_clean
        sc[:, :prefix] = prompt_clean

    trigger_out = adapter.forward_state(
        z, sc, grid[args.trigger_step], batch_size=args.batch_size
    )
    eligible = torch.ones(z.shape[:2], dtype=torch.bool, device=adapter.device)
    eligible[:, :prefix] = False
    anchor_mask, anchor_ids, anchor_clean, anchor_confidence = exp90.build_anchor(
        adapter,
        trigger_out["logits"].to(adapter.device),
        trigger_out["predicted_clean"].to(adapter.device),
        "top_confidence",
        eligible,
        args.density,
        seed=0,
    )
    unresolved = eligible & ~anchor_mask
    control_z, control_sc = z.clone(), sc.clone()
    anchor_z, anchor_sc = z.clone(), sc.clone()

    decision_step = args.trigger_step + 1 + args.lookahead
    for step in range(args.trigger_step, decision_step):
        if step == args.trigger_step:
            anchor_z[anchor_mask] = anchor_clean[anchor_mask]
            anchor_sc[anchor_mask] = anchor_clean[anchor_mask]
        seed = exp90.step_seed(args.seed, route, batch_index, step)
        control_z, control_sc = exp90.native_step(
            adapter, control_z, control_sc, grid[step], grid[step + 1], seed
        )
        anchor_z, anchor_sc = exp90.native_step(
            adapter, anchor_z, anchor_sc, grid[step], grid[step + 1], seed
        )
        if step == args.trigger_step:
            anchor_z[anchor_mask] = anchor_clean[anchor_mask]
            anchor_sc[anchor_mask] = anchor_clean[anchor_mask]
        for state in (control_z, control_sc, anchor_z, anchor_sc):
            state[:, :prefix] = prompt_clean

    anchor_out = adapter.forward_state(
        anchor_z, anchor_sc, grid[decision_step], batch_size=args.batch_size
    )
    control_out = adapter.forward_state(
        control_z, control_sc, grid[decision_step], batch_size=args.batch_size
    )
    anchor_prob = torch.softmax(anchor_out["logits"].float(), dim=-1)
    control_prob = torch.softmax(control_out["logits"].float(), dim=-1)
    anchor_entropy = -(anchor_prob * anchor_prob.clamp_min(1e-12).log()).sum(-1)
    control_entropy = -(control_prob * control_prob.clamp_min(1e-12).log()).sum(-1)
    response = masked_mean(control_entropy - anchor_entropy, unresolved)
    choose_anchor = response > 0.0
    row_mask = choose_anchor[:, None, None]
    z = torch.where(row_mask, anchor_z, control_z)
    sc = torch.where(row_mask, anchor_sc, control_sc)

    for step in range(decision_step, args.n_steps):
        z, sc = exp90.native_step(
            adapter,
            z,
            sc,
            grid[step],
            grid[step + 1],
            exp90.step_seed(args.seed, route, batch_index, step),
        )
        z[:, :prefix] = prompt_clean
        sc[:, :prefix] = prompt_clean

    final = adapter.forward_state(z, sc, grid[-1], batch_size=args.batch_size)
    final_ids = final["logits"].argmax(-1)
    decoded_prompt = adapter.forward_state(
        prompt_clean, prompt_clean, grid[-1], batch_size=args.batch_size
    )["logits"].argmax(-1)
    final_ids[:, :prefix] = decoded_prompt

    selected_mask = anchor_mask & choose_anchor[:, None]
    revision = (
        float((final_ids[selected_mask] != anchor_ids[selected_mask]).float().mean())
        if selected_mask.any()
        else 0.0
    )
    return {
        "ids": final_ids.cpu(),
        "response": response.cpu(),
        "choose_anchor": choose_anchor.cpu(),
        "anchor_confidence": anchor_confidence,
        "anchor_revision": revision,
        "denoiser_calls": args.n_steps + 1 + args.lookahead,
        "readout_calls": 3,
    }


def bootstrap(delta, samples, seed):
    generator = torch.Generator().manual_seed(seed + 1070039)
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(0, len(delta), (len(delta),), generator=generator)
        estimates[index] = delta[rows].mean()
    return [
        float(torch.quantile(estimates, 0.025)),
        float(torch.quantile(estimates, 0.975)),
    ]


def main():
    args = parse_args()
    if args.trigger_step + 1 + args.lookahead > args.n_steps:
        raise ValueError("shadow horizon exceeds solver grid")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    adapter = load_adapter("plaid", "baseline", None, device)
    adapter.seq_len = args.seq_len
    grid = np.linspace(adapter.t_eps, 0.999, args.n_steps + 1).tolist()
    panel_ids, dataset_name = exp99.load_conditional_panel(
        adapter, args.n_cond, args.seq_len, args.panel_offset
    )
    records = {
        name: {"ids": [], "stats": []}
        for name in ("standard", "fixed_anchor", "pathwise_shadow")
    }
    settings = setting_args(args)
    native_reference_agreement = None

    for batch_index, start in enumerate(range(0, args.n_cond, args.batch_size)):
        end = min(start + args.batch_size, args.n_cond)
        generator = torch.Generator(device=adapter.device).manual_seed(
            args.seed * 10007 + 700001 + batch_index
        )
        eps = adapter.sample_epsilon(
            (end - start, args.seq_len, adapter.d_model), generator=generator
        )
        prompt_clean = adapter.encode_clean(
            panel_ids[start:end, : args.prefix_length]
        ).to(adapter.device)
        standard = exp90.run_arm(
            adapter, eps, grid, settings, "standard", "conditional", batch_index, prompt_clean
        )
        if batch_index == 0:
            duplicate = exp90.run_arm(
                adapter,
                eps,
                grid,
                settings,
                "standard",
                "conditional",
                batch_index,
                prompt_clean,
            )
            native_reference_agreement = float(
                (duplicate["ids"] == standard["ids"]).float().mean()
            )
            if native_reference_agreement != 1.0:
                raise RuntimeError(
                    f"native reference agreement={native_reference_agreement}"
                )
        fixed = exp90.run_arm(
            adapter,
            eps,
            grid,
            settings,
            "top_confidence",
            "conditional",
            batch_index,
            prompt_clean,
        )
        shadow = run_shadow(adapter, eps, grid, args, batch_index, prompt_clean)
        for name, result in (
            ("standard", standard),
            ("fixed_anchor", fixed),
            ("pathwise_shadow", shadow),
        ):
            records[name]["ids"].append(result.pop("ids"))
            records[name]["stats"].append(result)
        print(f"generation completed {end}/{args.n_cond}", flush=True)

    prompts = [
        exp90.decode_ids(adapter, row[: args.prefix_length]) for row in panel_ids
    ]
    references = [
        exp90.decode_ids(adapter, row[args.prefix_length :]) for row in panel_ids
    ]
    for record in records.values():
        record["ids"] = torch.cat(record["ids"])
        record["texts"] = [
            exp90.decode_ids(adapter, row[args.prefix_length :]) for row in record["ids"]
        ]

    quality_base.release_generator(adapter)
    evaluator = quality_base.PPLEvaluator(args.ppl_model, device, args.ppl_batch_size)
    shuffled_prompts = prompts[1:] + prompts[:1]
    aggregate, nlls, counts = {}, {}, {}
    for name, record in records.items():
        nlls[name], counts[name] = exp99.conditional_sequence_nlls(
            prompts, record["texts"], evaluator
        )
        shuffled_nll, _ = exp99.conditional_sequence_nlls(
            shuffled_prompts, record["texts"], evaluator
        )
        aggregate[name] = exp99.summarize_texts(
            record["texts"], references, nlls[name], counts[name], shuffled_nll
        )

    delta = nlls["pathwise_shadow"] - nlls["fixed_anchor"]
    interval = bootstrap(delta, args.bootstrap_samples, args.seed)
    fixed_quality = aggregate["fixed_anchor"]
    shadow_quality = aggregate["pathwise_shadow"]
    quality_delta = {
        "d1": shadow_quality["d1"] - fixed_quality["d1"],
        "d2": shadow_quality["d2"] - fixed_quality["d2"],
        "rep4": shadow_quality["rep4"] - fixed_quality["rep4"],
        "degeneration_rate": shadow_quality["degeneration_rate"]
        - fixed_quality["degeneration_rate"],
        "prompt_gain_nats": shadow_quality["prompt_gain_nats"]
        - fixed_quality["prompt_gain_nats"],
    }
    response = torch.cat([row["response"] for row in records["pathwise_shadow"]["stats"]])
    chosen = torch.cat(
        [row["choose_anchor"] for row in records["pathwise_shadow"]["stats"]]
    )
    gate = (
        interval[1] < 0.0
        and quality_delta["d1"] >= -0.005
        and quality_delta["rep4"] <= 0.005
        and quality_delta["degeneration_rate"] <= 0.015
        and quality_delta["prompt_gain_nats"] >= -0.01
    )
    result = {
        **vars(args),
        "dataset": dataset_name,
        "paired_initial_and_ancestral_noise": True,
        "native_reference_agreement": native_reference_agreement,
        "pathwise_noise_route": 29,
        "aggregate": aggregate,
        "shadow_vs_fixed": {
            "mean_delta_nats": float(delta.double().mean()),
            "mean_delta_ci95": interval,
            "candidate_better_fraction": float((delta < 0).double().mean()),
            "quality_delta": quality_delta,
        },
        "choose_anchor_fraction": float(chosen.float().mean()),
        "response_mean": float(response.mean()),
        "response_std": float(response.std(unbiased=True)),
        "model_calls": {"standard": 32, "fixed_anchor": 33, "pathwise_shadow": 40},
        "pilot_gate_passed": gate,
        "per_sequence": {
            "standard_nll": nlls["standard"].tolist(),
            "fixed_anchor_nll": nlls["fixed_anchor"].tolist(),
            "pathwise_shadow_nll": nlls["pathwise_shadow"].tolist(),
            "response": response.tolist(),
            "choose_anchor": chosen.tolist(),
        },
        "texts": {name: record["texts"] for name, record in records.items()},
    }
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.label}_seed{args.seed}.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "aggregate": aggregate,
        "shadow_vs_fixed": result["shadow_vs_fixed"],
        "choose_anchor_fraction": result["choose_anchor_fraction"],
        "response_mean": result["response_mean"],
        "response_std": result["response_std"],
        "pilot_gate_passed": gate,
    }, indent=2), flush=True)
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()
