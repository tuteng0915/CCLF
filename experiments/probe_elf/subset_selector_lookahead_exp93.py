#!/usr/bin/env python3
"""EXP-93 Stage 2b: multi-step future-context subset lookahead features."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import paired_conditional_revalidation_exp80 as exp80
import subset_selector_features_exp93 as features
import transition_unlock_pareto_exp82 as exp82
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from modules.t5_encoder import get_encoder
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


OUT_DIR = Path("results/exp93_subset_selector_headroom")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--utility_json",
        default="results/exp93_subset_selector_headroom/p0_baseline_seed42.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizons", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--label", default="lookahead_discovery")
    return parser.parse_args()


@torch.no_grad()
def ode_step(z, x_pred, model, grid, index, sccfg, cond_seq, cond_mask):
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        return _ode_step(
            model=model,
            z=z,
            t=grid[index].item(),
            t_next=grid[index + 1].item(),
            x_pred_prev=x_pred,
            config=common.SamplingConfig(),
            cfg_scale=1.0,
            self_cond_cfg_scale=sccfg,
            cond_seq=cond_seq,
            cond_seq_mask=cond_mask,
        )


def horizon_features(
    trigger_ids,
    selected,
    base_ids,
    base_conf,
    base_entropy,
    anchor_ids,
    anchor_conf,
    anchor_entropy,
    release_ids,
    release_conf,
    release_entropy,
    base_release_ids,
    base_release_conf,
    base_release_entropy,
):
    unresolved = ~selected
    return {
        "anchor_selected_consistency": float(
            (anchor_ids[selected] == trigger_ids[selected]).float().mean().item()
        ),
        "anchor_selected_confidence": float(anchor_conf[selected].mean().item()),
        "anchor_unresolved_confidence_gain": float(
            (anchor_conf[unresolved] - base_conf[unresolved]).mean().item()
        ),
        "anchor_unresolved_entropy_reduction": float(
            (base_entropy[unresolved] - anchor_entropy[unresolved]).mean().item()
        ),
        "anchor_unresolved_top1_flip": float(
            (anchor_ids[unresolved] != base_ids[unresolved]).float().mean().item()
        ),
        "release_selected_consistency": float(
            (release_ids[selected] == trigger_ids[selected]).float().mean().item()
        ),
        "release_selected_confidence": float(release_conf[selected].mean().item()),
        "release_unresolved_confidence_gain": float(
            (release_conf[unresolved] - base_release_conf[unresolved]).mean().item()
        ),
        "release_unresolved_entropy_reduction": float(
            (base_release_entropy[unresolved] - release_entropy[unresolved]).mean().item()
        ),
        "release_unresolved_top1_flip": float(
            (release_ids[unresolved] != base_release_ids[unresolved]).float().mean().item()
        ),
    }


@torch.no_grad()
def extract(utility, model, cond_noise, cond_seq, cond_mask, horizons):
    device = cond_noise.device
    grid = get_sampling_steps(utility["n_steps"], "uniform", device=device)
    maximum = max(horizons)
    random_nll = torch.tensor(utility["per_sequence"]["random_nll"], dtype=torch.float64)
    standard_nll = torch.tensor(utility["per_sequence"]["standard_nll"], dtype=torch.float64)
    rows = {horizon: [] for horizon in horizons}

    for batch_index, start in enumerate(
        range(0, utility["n_cond"], utility["batch_size"])
    ):
        end = min(start + utility["batch_size"], utility["n_cond"])
        base_seq = cond_seq[start:end]
        base_mask = cond_mask[start:end]
        z = restore_cond(cond_noise[start:end].clone(), base_seq, base_mask)
        x_pred = restore_cond(torch.zeros_like(z), base_seq, base_mask)
        trigger_index = None
        for index in range(grid.shape[0] - 1):
            z, x_pred = ode_step(
                z, x_pred, model, grid, index, utility["sccfg"], base_seq, base_mask
            )
            if grid[index + 1].item() >= utility["fixed_policy"]["trigger"]:
                trigger_index = index
                break
        if trigger_index is None or trigger_index + maximum + 2 >= grid.shape[0]:
            raise RuntimeError("lookahead horizon exceeds the post-trigger grid")
        trigger_ids, trigger_conf, _ = features.lexical_stats(x_pred, model)

        base_states = {}
        base_z, base_x = z.clone(), x_pred.clone()
        for step in range(1, maximum + 2):
            grid_index = trigger_index + step
            base_z, base_x = ode_step(
                base_z,
                base_x,
                model,
                grid,
                grid_index,
                utility["sccfg"],
                base_seq,
                base_mask,
            )
            if step in horizons or step - 1 in horizons:
                base_states[step] = features.lexical_stats(base_x, model)

        for mask_index in range(utility["n_masks"]):
            mask_seed = utility["seed"] + (mask_index + 1) * 1009
            rng_seed = mask_seed * 100003 + batch_index * 7919
            generator = torch.Generator(device=device).manual_seed(rng_seed)
            scores = torch.rand(trigger_conf.shape, generator=generator, device=device)
            selected_full = exp82.exact_budget_mask(
                scores, base_mask < 0.5, utility["fixed_policy"]["density"]
            )
            anchor_seq = torch.where(selected_full.unsqueeze(-1), x_pred, base_seq)
            anchor_mask = torch.maximum(base_mask, selected_full.to(base_mask.dtype))
            anchor_z = restore_cond(z.clone(), anchor_seq, anchor_mask)
            anchor_x = restore_cond(x_pred.clone(), anchor_seq, anchor_mask)
            anchor_states = {}
            for step in range(1, maximum + 1):
                grid_index = trigger_index + step
                anchor_z, anchor_x = ode_step(
                    anchor_z,
                    anchor_x,
                    model,
                    grid,
                    grid_index,
                    utility["sccfg"],
                    anchor_seq,
                    anchor_mask,
                )
                if step in horizons:
                    anchor_states[step] = (
                        anchor_z.clone(),
                        anchor_x.clone(),
                        features.lexical_stats(anchor_x, model),
                    )

            for horizon in horizons:
                horizon_z, horizon_x, anchor_lexical = anchor_states[horizon]
                release_z, release_x = ode_step(
                    horizon_z,
                    horizon_x,
                    model,
                    grid,
                    trigger_index + horizon + 1,
                    utility["sccfg"],
                    base_seq,
                    base_mask,
                )
                release_lexical = features.lexical_stats(release_x, model)
                base_lexical = base_states[horizon]
                base_release_lexical = base_states[horizon + 1]
                for local_row, trajectory in enumerate(range(start, end)):
                    suffix = slice(utility["prefix_length"], utility["max_length"])
                    selected = selected_full[local_row, suffix]
                    feature_values = horizon_features(
                        trigger_ids[local_row, suffix],
                        selected,
                        *(value[local_row, suffix] for value in base_lexical),
                        *(value[local_row, suffix] for value in anchor_lexical),
                        *(value[local_row, suffix] for value in release_lexical),
                        *(value[local_row, suffix] for value in base_release_lexical),
                    )
                    rows[horizon].append({
                        "trajectory": trajectory,
                        "mask_index": mask_index,
                        "utility_nats": float(
                            (standard_nll[trajectory] - random_nll[mask_index, trajectory]).item()
                        ),
                        "features": feature_values,
                    })
    return rows


def analyze_rows(rows, utility, folds, ridge):
    feature_names = sorted(rows[0]["features"])
    x = torch.tensor(
        [[row["features"][name] for name in feature_names] for row in rows],
        dtype=torch.float64,
    )
    y = torch.tensor([row["utility_nats"] for row in rows], dtype=torch.float64)
    trajectories = torch.tensor([row["trajectory"] for row in rows], dtype=torch.long)
    predictions, coefficients = features.grouped_oof_ridge(
        x, y, trajectories, folds, ridge, utility["seed"]
    )
    screens = {
        "grouped_oof_ridge": features.ranking_metrics(predictions, y, trajectories)
    }
    screens["grouped_oof_ridge"]["selected_prompt_conditioned_ppl"] = features.selected_ppl(
        predictions, rows, utility
    )
    for name in feature_names:
        score = x[:, feature_names.index(name)]
        item = features.ranking_metrics(score, y, trajectories)
        item["selected_prompt_conditioned_ppl"] = features.selected_ppl(
            score, rows, utility
        )
        screens[name] = item
    return {
        "feature_names": feature_names,
        "screens": screens,
        "mean_standardized_coefficients": {
            name: float(value)
            for name, value in zip(feature_names, coefficients.tolist())
        },
        "rows": rows,
        "oof_predictions": predictions.tolist(),
    }


def main():
    args = parse_args()
    utility_path = Path(args.utility_json)
    utility = json.loads(utility_path.read_text())
    horizons = sorted(set(args.horizons))
    if not horizons or min(horizons) < 1:
        raise ValueError("lookahead horizons must be positive")
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = utility["noise_scale"]
    from transformers import T5Tokenizer

    checkpoint_path = Path(utility["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(utility["max_length"]))
    model.load_state_dict(common.load_weights(checkpoint), strict=False)
    model.to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder.to(device).eval()

    class Namespace:
        pass

    data_args = Namespace()
    for key in ("n_cond", "max_length", "prefix_length", "conditional_dataset", "owt_offset"):
        setattr(data_args, key, utility[key])
    pairs = exp80.load_pairs(data_args, tokenizer)
    cond_seq, cond_mask, _ = common.build_condition_data(
        pairs, tokenizer, encoder, device, utility["max_length"], utility["prefix_length"]
    )
    generator = torch.Generator(device=device).manual_seed(utility["seed"])
    cond_noise = utility["noise_scale"] * torch.randn(
        utility["n_cond"],
        utility["max_length"],
        common.model_config(utility["max_length"])["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    cond_noise[:, : utility["prefix_length"]] = cond_seq[:, : utility["prefix_length"]]
    by_horizon = extract(
        utility, model, cond_noise, cond_seq, cond_mask, horizons
    )
    analyses = {
        str(horizon): analyze_rows(
            by_horizon[horizon], utility, args.folds, args.ridge
        )
        for horizon in horizons
    }
    compact = {
        horizon: analysis["screens"] for horizon, analysis in analyses.items()
    }
    print(json.dumps(compact, indent=2))
    result = {
        "utility_json": str(utility_path),
        "horizons": horizons,
        "folds": args.folds,
        "ridge": args.ridge,
        "extra_denoiser_calls_per_candidate": {
            str(horizon): horizon + 1 for horizon in horizons
        },
        "analyses": analyses,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.label}_{utility['checkpoint']}_seed{utility['seed']}.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
