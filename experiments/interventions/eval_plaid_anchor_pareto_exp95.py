#!/usr/bin/env python3
"""EXP-95 Stage 1: Plaid temporary-anchor density/horizon/trigger screen."""

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
import eval_temporary_anchor_portability_exp90 as exp90  # noqa: E402
from common import load_adapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results/exp95_plaid_anchor_pareto")
    parser.add_argument("--label", default="screen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_uncond", type=int, default=16)
    parser.add_argument("--n_cond", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--trigger_steps", default="14,18,22")
    parser.add_argument("--densities", default=".125,.25,.5,.75")
    parser.add_argument("--horizons", default="1,2,4,8")
    parser.add_argument("--ppl_model", default="gpt2-large")
    parser.add_argument("--ppl_batch_size", type=int, default=4)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_reference_gate", action="store_true")
    return parser.parse_args()


def parse_csv(value, cast):
    return tuple(cast(item) for item in value.split(",") if item.strip())


def key(trigger, density, horizon):
    density_tag = str(density).replace(".", "p")
    return f"random_t{trigger}_d{density_tag}_h{horizon}"


def setting_args(args, trigger, density, horizon):
    class Settings:
        pass

    settings = Settings()
    for name, value in vars(args).items():
        setattr(settings, name, value)
    settings.model = "plaid"
    settings.checkpoint = "baseline"
    settings.trigger_step = trigger
    settings.density = density
    settings.horizon = horizon
    return settings


@torch.no_grad()
def evaluate_grid_scope(adapter, args, grid, scope, settings, panel_ids=None):
    count = args.n_uncond if scope == "unconditional" else args.n_cond
    names = ["standard"] + [key(*setting) for setting in settings]
    records = {name: {"ids": [], "stats": []} for name in names}
    reference_agreements = []
    for batch_index, start in enumerate(range(0, count, args.batch_size)):
        end = min(start + args.batch_size, count)
        size = end - start
        generator = torch.Generator(device=adapter.device).manual_seed(
            args.seed * 10007
            + (0 if scope == "unconditional" else 700001)
            + batch_index
        )
        eps = adapter.sample_epsilon(
            (size, args.seq_len, adapter.d_model), generator=generator
        )
        prompt_clean = None
        if scope == "conditional":
            prompt_ids = panel_ids[start:end, : args.prefix_length]
            prompt_clean = adapter.encode_clean(prompt_ids).to(adapter.device)

        standard_args = setting_args(args, settings[0][0], settings[0][1], settings[0][2])
        standard = exp90.run_arm(
            adapter, eps, grid, standard_args, "standard", scope, batch_index, prompt_clean
        )
        standard_ids = standard.pop("ids")
        records["standard"]["ids"].append(standard_ids)
        records["standard"]["stats"].append(standard)
        if not args.skip_reference_gate and batch_index == 0:
            duplicate = exp90.run_arm(
                adapter,
                eps,
                grid,
                standard_args,
                "standard",
                scope,
                batch_index,
                prompt_clean,
            )
            agreement = float((duplicate["ids"] == standard_ids).float().mean())
            reference_agreements.append(agreement)
            if agreement != 1.0:
                raise RuntimeError(f"{scope} reference agreement={agreement}")

        for trigger, density, horizon in settings:
            run_args = setting_args(args, trigger, density, horizon)
            result = exp90.run_arm(
                adapter, eps, grid, run_args, "random", scope, batch_index, prompt_clean
            )
            name = key(trigger, density, horizon)
            records[name]["ids"].append(result.pop("ids"))
            records[name]["stats"].append(result)
        print(f"{scope}: completed {end}/{count}", flush=True)

    for record in records.values():
        record["ids"] = torch.cat(record["ids"], dim=0)
    return records, min(reference_agreements) if reference_agreements else None


def main():
    args = parse_args()
    triggers = parse_csv(args.trigger_steps, int)
    densities = parse_csv(args.densities, float)
    horizons = parse_csv(args.horizons, int)
    settings = tuple(
        (trigger, density, horizon)
        for trigger in triggers
        for density in densities
        for horizon in horizons
        if trigger + horizon <= args.n_steps
    )
    if not settings:
        raise ValueError("empty setting grid")
    if not 0 < args.prefix_length < args.seq_len:
        raise ValueError("prefix_length must lie inside seq_len")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    adapter = load_adapter("plaid", "baseline", None, device)
    adapter.seq_len = args.seq_len
    grid = np.linspace(adapter.t_eps, 0.999, args.n_steps + 1).tolist()
    panel_ids, dataset_name = exp90.load_conditional_panel(
        adapter, args.n_cond, args.seq_len
    )
    unconditional, u_reference = evaluate_grid_scope(
        adapter, args, grid, "unconditional", settings
    )
    conditional, c_reference = evaluate_grid_scope(
        adapter, args, grid, "conditional", settings, panel_ids
    )

    prompts = [exp90.decode_ids(adapter, row[: args.prefix_length]) for row in panel_ids]
    references = [exp90.decode_ids(adapter, row[args.prefix_length :]) for row in panel_ids]
    for record in unconditional.values():
        record["texts"] = [exp90.decode_ids(adapter, row) for row in record["ids"]]
    for record in conditional.values():
        record["texts"] = [
            exp90.decode_ids(adapter, row[args.prefix_length :]) for row in record["ids"]
        ]

    quality_base.release_generator(adapter)
    evaluator = (
        None
        if args.skip_ppl
        else quality_base.PPLEvaluator(args.ppl_model, device, args.ppl_batch_size)
    )
    shuffled_prompts = prompts[1:] + prompts[:1]
    results = {}
    for name in unconditional:
        u_record, c_record = unconditional[name], conditional[name]
        u_quality = quality_base.text_quality(u_record["texts"])
        c_quality = quality_base.text_quality(c_record["texts"])
        if evaluator is None:
            u_ppl = c_ppl = shuffled_ppl = float("nan")
        else:
            u_ppl = evaluator(u_record["texts"], args.seq_len)
            c_ppl = conditional_base.conditional_ppl(
                prompts, c_record["texts"], evaluator
            )
            shuffled_ppl = conditional_base.conditional_ppl(
                shuffled_prompts, c_record["texts"], evaluator
            )
        c_quality.update(
            prompt_conditioned_ppl=c_ppl,
            shuffled_prompt_ppl=shuffled_ppl,
            prompt_gain_nats=(
                math.log(shuffled_ppl) - math.log(c_ppl)
                if c_ppl == c_ppl and shuffled_ppl == shuffled_ppl
                else float("nan")
            ),
            rouge_l=float(
                np.mean(
                    [
                        conditional_base.rouge_l_f1(pred, ref)
                        for pred, ref in zip(c_record["texts"], references)
                    ]
                )
            ),
        )
        u_quality["ppl"] = u_ppl
        stats = c_record["stats"]
        results[name] = {
            "unconditional": u_quality,
            "conditional": c_quality,
            "anchor_fraction": exp90.mean_stat(stats, "anchor_fraction"),
            "anchor_confidence": exp90.mean_stat(stats, "anchor_confidence"),
            "anchor_revision": exp90.mean_stat(stats, "anchor_revision"),
            "max_prompt_clamp_error": max(
                row["max_prompt_clamp_error"] for row in stats
            ),
        }
        print(
            f"{name:24s} U={u_ppl:.2f} C={c_ppl:.2f} "
            f"D1={c_quality['d1']:.4f} Rep4={c_quality['rep4']:.4f} "
            f"Deg={c_quality['degeneration_rate']:.4f}",
            flush=True,
        )

    standard = results["standard"]
    ranking = []
    for trigger, density, horizon in settings:
        name = key(trigger, density, horizon)
        row = results[name]
        ranking.append(
            {
                "name": name,
                "trigger_step": trigger,
                "trigger_t": grid[trigger],
                "density": density,
                "horizon": horizon,
                "delta_u_ppl": row["unconditional"]["ppl"]
                - standard["unconditional"]["ppl"],
                "delta_c_ppl": row["conditional"]["prompt_conditioned_ppl"]
                - standard["conditional"]["prompt_conditioned_ppl"],
                "delta_c_d1": row["conditional"]["d1"]
                - standard["conditional"]["d1"],
                "delta_c_rep4": row["conditional"]["rep4"]
                - standard["conditional"]["rep4"],
                "delta_c_degeneration": row["conditional"]["degeneration_rate"]
                - standard["conditional"]["degeneration_rate"],
                "delta_prompt_gain": row["conditional"]["prompt_gain_nats"]
                - standard["conditional"]["prompt_gain_nats"],
            }
        )
    ranking.sort(key=lambda row: row["delta_c_ppl"])

    payload = {
        **vars(args),
        "model": "plaid",
        "dataset": dataset_name,
        "settings": [
            {"trigger": t, "density": d, "horizon": h} for t, d, h in settings
        ],
        "paired_initial_and_ancestral_noise": True,
        "native_reference_agreement": {
            "unconditional": u_reference,
            "conditional": c_reference,
        },
        "ranking": ranking,
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.label}_plaid_seed{args.seed}.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
