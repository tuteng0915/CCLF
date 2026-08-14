#!/usr/bin/env python3
"""EXP-101: paired Plaid temporary-anchor trigger-time headroom.

Only trigger time changes across candidates. Prompts, initial latents, and all
ancestral solver noise are paired. The per-trajectory best trigger is a
diagnostic upper bound and is never presented as a deployable sampler.
"""

import argparse
import json
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
import eval_plaid_subset_headroom_exp99 as exp99  # noqa: E402
import eval_temporary_anchor_portability_exp90 as exp90  # noqa: E402
from common import load_adapter  # noqa: E402


FEATURE_NAMES = (
    "mean_confidence",
    "q10_confidence",
    "mean_entropy",
    "mean_top12_margin",
    "lexical_revision",
    "xhat_cosine_instability",
    "confidence_change",
    "entropy_change",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results/exp101_plaid_adaptive_trigger")
    parser.add_argument("--label", default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel_offset", type=int, default=0)
    parser.add_argument("--n_cond", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--trigger_steps", default="8,10,12,14,16,18,20,22")
    parser.add_argument("--fixed_trigger", type=int, default=14)
    parser.add_argument("--density", type=float, default=0.75)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--ppl_model", default="gpt2-large")
    parser.add_argument("--ppl_batch_size", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--skip_reference_gate", action="store_true")
    return parser.parse_args()


def setting_args(args, trigger_step):
    class Settings:
        pass

    settings = Settings()
    for name, value in vars(args).items():
        setattr(settings, name, value)
    settings.model = "plaid"
    settings.checkpoint = "baseline"
    settings.trigger_step = trigger_step
    return settings


def parse_triggers(value):
    triggers = tuple(int(item) for item in value.split(",") if item.strip())
    if len(set(triggers)) != len(triggers):
        raise ValueError("trigger steps must be unique")
    return tuple(sorted(triggers))


@torch.no_grad()
def replay_event_features(adapter, args, grid, eps, prompt_clean, batch_index, triggers):
    """Record summaries on the unmodified native trajectory.

    Changes are measured against the immediately preceding native solver step,
    not the preceding item in the coarser trigger grid.
    """
    z = eps.clone().to(adapter.device)
    sc = torch.zeros_like(z)
    prefix = args.prefix_length
    z[:, :prefix] = prompt_clean
    sc[:, :prefix] = prompt_clean
    wanted = set(triggers)
    feature_rows = {}
    previous = None

    for step in range(max(triggers) + 1):
        if step >= min(triggers) - 1:
            out = adapter.forward_state(z, sc, grid[step], batch_size=args.batch_size)
            logits = out["logits"].float()[:, prefix:]
            xhat = out["predicted_clean"].float()[:, prefix:]
            probabilities = torch.softmax(logits, dim=-1)
            confidence, top1 = probabilities.max(dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
            top2 = probabilities.topk(k=2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]

            if step in wanted:
                if previous is None:
                    raise RuntimeError("missing immediately preceding readout")
                previous_top1, previous_xhat, previous_conf, previous_entropy = previous
                lexical_revision = (top1 != previous_top1).float().mean(dim=1)
                cosine = F.cosine_similarity(xhat, previous_xhat, dim=-1).mean(dim=1)
                features = torch.stack(
                    (
                        confidence.mean(dim=1),
                        torch.quantile(confidence, 0.10, dim=1),
                        entropy.mean(dim=1),
                        margin.mean(dim=1),
                        lexical_revision,
                        1.0 - cosine,
                        confidence.mean(dim=1) - previous_conf.mean(dim=1),
                        entropy.mean(dim=1) - previous_entropy.mean(dim=1),
                    ),
                    dim=-1,
                )
                feature_rows[step] = features.cpu()
            previous = (top1, xhat, confidence, entropy)

        if step < max(triggers):
            z, sc = exp90.native_step(
                adapter,
                z,
                sc,
                grid[step],
                grid[step + 1],
                exp90.step_seed(args.seed, 29, batch_index, step),
            )
            z[:, :prefix] = prompt_clean
            sc[:, :prefix] = prompt_clean

    return torch.stack([feature_rows[step] for step in triggers], dim=1)


@torch.no_grad()
def generate_bank(adapter, args, grid, panel_ids, triggers):
    names = ["standard"] + [f"trigger_{step:02d}" for step in triggers]
    records = {name: {"ids": [], "stats": []} for name in names}
    feature_batches = []
    agreements = []

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
        feature_batches.append(
            replay_event_features(
                adapter, args, grid, eps, prompt_clean, batch_index, triggers
            )
        )

        standard_args = setting_args(args, args.fixed_trigger)
        standard = exp90.run_arm(
            adapter,
            eps,
            grid,
            standard_args,
            "standard",
            "conditional",
            batch_index,
            prompt_clean,
        )
        records["standard"]["ids"].append(standard.pop("ids"))
        records["standard"]["stats"].append(standard)

        for step in triggers:
            run_args = setting_args(args, step)
            result = exp90.run_arm(
                adapter,
                eps,
                grid,
                run_args,
                "top_confidence",
                "conditional",
                batch_index,
                prompt_clean,
            )
            name = f"trigger_{step:02d}"
            records[name]["ids"].append(result.pop("ids"))
            records[name]["stats"].append(result)

        if not args.skip_reference_gate and batch_index == 0:
            duplicate = exp90.run_arm(
                adapter,
                eps,
                grid,
                standard_args,
                "standard",
                "conditional",
                batch_index,
                prompt_clean,
            )
            agreement = float(
                (duplicate["ids"] == records["standard"]["ids"][-1]).float().mean()
            )
            agreements.append(agreement)
            if agreement != 1.0:
                raise RuntimeError(f"native reference agreement={agreement}")
        print(f"generation completed {end}/{args.n_cond}", flush=True)

    for record in records.values():
        record["ids"] = torch.cat(record["ids"], dim=0)
    return records, torch.cat(feature_batches), min(agreements) if agreements else None


def summarize(adapter, records, names, prompts, evaluator, args):
    shuffled_prompts = prompts[1:] + prompts[:1]
    nlls, counts, shuffled_nlls = {}, {}, {}
    for name in names:
        record = records[name]
        record["texts"] = [
            exp90.decode_ids(adapter, row[args.prefix_length :])
            for row in record["ids"]
        ]
        print(f"evaluating {name}", flush=True)
        nlls[name], counts[name] = exp99.conditional_sequence_nlls(
            prompts, record["texts"], evaluator
        )
        shuffled_nlls[name], _ = exp99.conditional_sequence_nlls(
            shuffled_prompts, record["texts"], evaluator
        )
    return nlls, counts, shuffled_nlls


def bootstrap_delta(reference_nll, candidate_nll, samples, seed):
    delta = (candidate_nll - reference_nll).double()
    generator = torch.Generator().manual_seed(seed + 1010017)
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(0, len(delta), (len(delta),), generator=generator)
        estimates[index] = delta[rows].mean()
    return {
        "mean_nats": float(delta.mean()),
        "ci95_nats": [
            float(torch.quantile(estimates, 0.025)),
            float(torch.quantile(estimates, 0.975)),
        ],
        "candidate_better_fraction": float((delta < 0).double().mean()),
    }


def main():
    args = parse_args()
    triggers = parse_triggers(args.trigger_steps)
    if args.fixed_trigger not in triggers:
        raise ValueError("fixed trigger must be one of the trigger candidates")
    if min(triggers) < 1 or max(triggers) + args.horizon > args.n_steps:
        raise ValueError("candidate trigger outside the solver grid")
    if not 0.0 < args.density <= 1.0:
        raise ValueError("density must lie in (0, 1]")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    adapter = load_adapter("plaid", "baseline", None, device)
    adapter.seq_len = args.seq_len
    grid = np.linspace(adapter.t_eps, 0.999, args.n_steps + 1).tolist()
    panel_ids, dataset_name = exp99.load_conditional_panel(
        adapter, args.n_cond, args.seq_len, args.panel_offset
    )
    records, features, reference_agreement = generate_bank(
        adapter, args, grid, panel_ids, triggers
    )

    prompts = [
        exp90.decode_ids(adapter, row[: args.prefix_length]) for row in panel_ids
    ]
    references = [
        exp90.decode_ids(adapter, row[args.prefix_length :]) for row in panel_ids
    ]
    quality_base.release_generator(adapter)
    evaluator = quality_base.PPLEvaluator(
        args.ppl_model, device, args.ppl_batch_size
    )
    names = ["standard"] + [f"trigger_{step:02d}" for step in triggers]
    nlls, counts, shuffled_nlls = summarize(
        adapter, records, names, prompts, evaluator, args
    )

    trigger_nll = torch.stack([nlls[f"trigger_{step:02d}"] for step in triggers])
    trigger_counts = torch.stack([counts[f"trigger_{step:02d}"] for step in triggers])
    best_index = trigger_nll.argmin(dim=0)
    rows = torch.arange(args.n_cond)
    best_nll = trigger_nll[best_index, rows]
    best_counts = trigger_counts[best_index, rows]
    fixed_index = triggers.index(args.fixed_trigger)
    fixed_nll = trigger_nll[fixed_index]
    fixed_counts = trigger_counts[fixed_index]
    best_texts = [
        records[f"trigger_{triggers[int(best_index[row])]:02d}"]["texts"][row]
        for row in range(args.n_cond)
    ]
    shuffled_prompts = prompts[1:] + prompts[:1]
    best_shuffled_nll, _ = exp99.conditional_sequence_nlls(
        shuffled_prompts, best_texts, evaluator
    )

    aggregate = {}
    for name in names:
        aggregate[name] = exp99.summarize_texts(
            records[name]["texts"],
            references,
            nlls[name],
            counts[name],
            shuffled_nlls[name],
        )
        aggregate[name]["anchor_revision"] = exp99.mean_stat(
            records[name]["stats"], "anchor_revision"
        )
    aggregate["oracle_best_trigger"] = exp99.summarize_texts(
        best_texts, references, best_nll, best_counts, best_shuffled_nll
    )
    fixed_ppl = exp99.aggregate_ppl(fixed_nll, fixed_counts)
    oracle_ppl = exp99.aggregate_ppl(best_nll, best_counts)
    improvement_pct = 100.0 * (fixed_ppl - oracle_ppl) / fixed_ppl
    headroom = bootstrap_delta(
        fixed_nll, best_nll, args.bootstrap_samples, args.seed
    )
    aggregate["oracle_vs_fixed_improvement_pct"] = improvement_pct
    aggregate["oracle_vs_fixed_bootstrap"] = headroom
    aggregate["winning_trigger_histogram"] = {
        str(step): int((best_index == index).sum())
        for index, step in enumerate(triggers)
    }
    print(json.dumps(aggregate, indent=2), flush=True)

    payload = {
        **vars(args),
        "model": "plaid",
        "dataset": dataset_name,
        "triggers": list(triggers),
        "trigger_times": [grid[step] for step in triggers],
        "feature_names": FEATURE_NAMES,
        "event_features": features.tolist(),
        "paired_initial_and_ancestral_noise": True,
        "native_reference_agreement": reference_agreement,
        "oracle_is_deployable": False,
        "trigger_headroom_gate_passed": (
            improvement_pct >= 5.0 and headroom["ci95_nats"][1] < 0.0
        ),
        "aggregate": aggregate,
        "per_sequence": {
            "trigger_nll": trigger_nll.tolist(),
            "trigger_token_counts": trigger_counts.tolist(),
            "fixed_nll": fixed_nll.tolist(),
            "best_trigger_index": best_index.tolist(),
            "best_nll": best_nll.tolist(),
        },
        "texts": {
            "fixed": records[f"trigger_{args.fixed_trigger:02d}"]["texts"],
            "oracle_best": best_texts,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.label}_seed{args.seed}.json"
    output.write_text(json.dumps(payload, indent=2))
    print(f"Saved -> {output}", flush=True)


if __name__ == "__main__":
    main()
