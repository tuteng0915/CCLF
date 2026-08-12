#!/usr/bin/env python3
"""EXP-93 Stage 2c: pairwise causal influence graph for subset utility."""

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
import subset_selector_features_exp93 as feature_common
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
    parser.add_argument("--n_trajectories", type=int, default=16)
    parser.add_argument("--probe_batch_size", type=int, default=16)
    parser.add_argument("--hold_horizon", type=int, default=4)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--label", default="dependency_pilot")
    return parser.parse_args()


@torch.no_grad()
def step(z, x_pred, model, grid, index, sccfg, cond_seq, cond_mask):
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


@torch.no_grad()
def trigger_state(z0, model, grid, sccfg, cond_seq, cond_mask, trigger):
    z = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    for index in range(grid.shape[0] - 1):
        z, x_pred = step(
            z, x_pred, model, grid, index, sccfg, cond_seq, cond_mask
        )
        if grid[index + 1].item() >= trigger:
            return z, x_pred, index
    raise RuntimeError("trigger was not reached")


@torch.no_grad()
def base_future(
    z, x_pred, model, grid, trigger_index, horizon, sccfg, cond_seq, cond_mask
):
    for offset in range(1, horizon + 1):
        z, x_pred = step(
            z,
            x_pred,
            model,
            grid,
            trigger_index + offset,
            sccfg,
            cond_seq,
            cond_mask,
        )
    return feature_common.lexical_stats(x_pred, model)


@torch.no_grad()
def influence_graph(
    z,
    x_pred,
    model,
    grid,
    trigger_index,
    horizon,
    sccfg,
    cond_seq,
    cond_mask,
    prefix_length,
    probe_batch_size,
):
    """Return source-by-target changes from single-position temporary anchors."""
    if z.shape[0] != 1:
        raise ValueError("influence_graph expects one trajectory")
    suffix_length = z.shape[1] - prefix_length
    base_ids, base_conf, base_entropy = base_future(
        z.clone(),
        x_pred.clone(),
        model,
        grid,
        trigger_index,
        horizon,
        sccfg,
        cond_seq,
        cond_mask,
    )
    confidence_rows, entropy_rows, flip_rows = [], [], []
    for source_start in range(0, suffix_length, probe_batch_size):
        source_end = min(source_start + probe_batch_size, suffix_length)
        count = source_end - source_start
        z_batch = z.expand(count, -1, -1).clone()
        x_batch = x_pred.expand(count, -1, -1).clone()
        seq_batch = cond_seq.expand(count, -1, -1).clone()
        mask_batch = cond_mask.expand(count, -1).clone()
        for local, source in enumerate(range(source_start, source_end)):
            position = prefix_length + source
            seq_batch[local, position] = x_pred[0, position]
            mask_batch[local, position] = 1
        z_batch = restore_cond(z_batch, seq_batch, mask_batch)
        x_batch = restore_cond(x_batch, seq_batch, mask_batch)
        for offset in range(1, horizon + 1):
            z_batch, x_batch = step(
                z_batch,
                x_batch,
                model,
                grid,
                trigger_index + offset,
                sccfg,
                seq_batch,
                mask_batch,
            )
        ids, confidence, entropy = feature_common.lexical_stats(x_batch, model)
        suffix = slice(prefix_length, z.shape[1])
        confidence_rows.append(
            confidence[:, suffix].float() - base_conf[0, suffix].float()
        )
        entropy_rows.append(
            base_entropy[0, suffix].float() - entropy[:, suffix].float()
        )
        flip_rows.append(
            (ids[:, suffix] != base_ids[0, suffix]).float()
        )
    confidence_graph = torch.cat(confidence_rows, dim=0)
    entropy_graph = torch.cat(entropy_rows, dim=0)
    flip_graph = torch.cat(flip_rows, dim=0)
    diagonal = torch.arange(suffix_length, device=z.device)
    for graph in (confidence_graph, entropy_graph, flip_graph):
        graph[diagonal, diagonal] = 0
    return confidence_graph, entropy_graph, flip_graph


def graph_features(graph, selected):
    unresolved = ~selected
    cross = graph[selected][:, unresolved]
    within = graph[selected][:, selected]
    positive = cross.clamp_min(0)
    negative = (-cross).clamp_min(0)
    max_coverage = cross.max(dim=0).values
    positive_coverage = positive.max(dim=0).values
    return {
        "cross_mean": float(cross.mean().item()),
        "cross_positive_mean": float(positive.mean().item()),
        "cross_negative_mean": float(negative.mean().item()),
        "coverage_mean": float(max_coverage.mean().item()),
        "coverage_q25": float(torch.quantile(max_coverage, 0.25).item()),
        "positive_coverage_mean": float(positive_coverage.mean().item()),
        "positive_coverage_q25": float(torch.quantile(positive_coverage, 0.25).item()),
        "within_mean": float(within.mean().item()),
        "within_abs_mean": float(within.abs().mean().item()),
    }


def rebuild_masks(utility, trajectory, batch_index, device):
    suffix_length = utility["max_length"] - utility["prefix_length"]
    local_row = trajectory - batch_index * utility["batch_size"]
    masks = []
    for mask_index in range(utility["n_masks"]):
        mask_seed = utility["seed"] + (mask_index + 1) * 1009
        rng_seed = mask_seed * 100003 + batch_index * 7919
        generator = torch.Generator(device=device).manual_seed(rng_seed)
        scores = torch.rand(
            utility["batch_size"], utility["max_length"],
            generator=generator, device=device,
        )
        eligible = torch.zeros_like(scores, dtype=torch.bool)
        eligible[:, utility["prefix_length"] :] = True
        selected = exp82.exact_budget_mask(
            scores, eligible, utility["fixed_policy"]["density"]
        )
        masks.append(selected[local_row, utility["prefix_length"] :])
    for mask in masks:
        if int(mask.sum().item()) != suffix_length // 2:
            raise RuntimeError("reconstructed mask violates exact density")
    return masks


def selected_ppl(scores, rows, utility):
    nll = torch.tensor(utility["per_sequence"]["random_nll"], dtype=torch.float64)
    counts = torch.tensor(
        utility["per_sequence"]["random_token_counts"], dtype=torch.float64
    )
    chosen_nll, chosen_counts = [], []
    for trajectory in sorted({row["trajectory"] for row in rows}):
        candidates = [
            (index, row) for index, row in enumerate(rows)
            if row["trajectory"] == trajectory
        ]
        index, row = max(candidates, key=lambda item: scores[item[0]])
        mask = row["mask_index"]
        chosen_nll.append(nll[mask, trajectory])
        chosen_counts.append(counts[mask, trajectory])
    chosen_nll = torch.stack(chosen_nll)
    chosen_counts = torch.stack(chosen_counts)
    return math.exp(float((chosen_nll * chosen_counts).sum() / chosen_counts.sum()))


def analyze(rows, utility, folds, ridge):
    names = sorted(rows[0]["features"])
    x = torch.tensor(
        [[row["features"][name] for name in names] for row in rows],
        dtype=torch.float64,
    )
    y = torch.tensor([row["utility_nats"] for row in rows], dtype=torch.float64)
    trajectories = torch.tensor([row["trajectory"] for row in rows], dtype=torch.long)
    predictions, coefficients = feature_common.grouped_oof_ridge(
        x, y, trajectories, folds, ridge, utility["seed"]
    )
    screens = {
        "grouped_oof_ridge": feature_common.ranking_metrics(
            predictions, y, trajectories
        )
    }
    screens["grouped_oof_ridge"]["selected_prompt_conditioned_ppl"] = selected_ppl(
        predictions.tolist(), rows, utility
    )
    for index, name in enumerate(names):
        score = x[:, index]
        screens[name] = feature_common.ranking_metrics(score, y, trajectories)
        screens[name]["selected_prompt_conditioned_ppl"] = selected_ppl(
            score.tolist(), rows, utility
        )
    return names, screens, {
        name: float(value) for name, value in zip(names, coefficients.tolist())
    }, predictions


def main():
    args = parse_args()
    utility_path = Path(args.utility_json)
    utility = json.loads(utility_path.read_text())
    n_trajectories = min(args.n_trajectories, utility["n_cond"])
    if n_trajectories < args.folds:
        raise ValueError("n_trajectories must be at least the number of folds")
    if args.hold_horizon != utility["fixed_policy"]["hold_horizon"]:
        raise ValueError("pilot horizon must match the utility-generating policy")
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
    for key in ("max_length", "prefix_length", "conditional_dataset", "owt_offset"):
        setattr(data_args, key, utility[key])
    data_args.n_cond = n_trajectories
    pairs = exp80.load_pairs(data_args, tokenizer)
    cond_seq, cond_mask, _ = common.build_condition_data(
        pairs, tokenizer, encoder, device,
        utility["max_length"], utility["prefix_length"],
    )
    generator = torch.Generator(device=device).manual_seed(utility["seed"])
    cond_noise = utility["noise_scale"] * torch.randn(
        utility["n_cond"], utility["max_length"],
        common.model_config(utility["max_length"])["text_encoder_dim"],
        generator=generator, device=device,
    )
    cond_noise = cond_noise[:n_trajectories]
    cond_noise[:, : utility["prefix_length"]] = cond_seq[:, : utility["prefix_length"]]
    grid = get_sampling_steps(utility["n_steps"], "uniform", device=device)
    standard_nll = torch.tensor(
        utility["per_sequence"]["standard_nll"], dtype=torch.float64
    )
    random_nll = torch.tensor(
        utility["per_sequence"]["random_nll"], dtype=torch.float64
    )
    rows = []
    for trajectory in range(n_trajectories):
        print(f"[trajectory {trajectory + 1}/{n_trajectories}]", flush=True)
        seq = cond_seq[trajectory : trajectory + 1]
        mask = cond_mask[trajectory : trajectory + 1]
        z, x_pred, trigger_index = trigger_state(
            cond_noise[trajectory : trajectory + 1], model, grid,
            utility["sccfg"], seq, mask, utility["fixed_policy"]["trigger"],
        )
        graphs = influence_graph(
            z, x_pred, model, grid, trigger_index, args.hold_horizon,
            utility["sccfg"], seq, mask, utility["prefix_length"],
            args.probe_batch_size,
        )
        batch_index = trajectory // utility["batch_size"]
        masks = rebuild_masks(utility, trajectory, batch_index, device)
        for mask_index, selected in enumerate(masks):
            values = {}
            for prefix, graph in zip(("confidence", "entropy", "flip"), graphs):
                values.update({
                    f"{prefix}_{name}": value
                    for name, value in graph_features(graph, selected).items()
                })
            rows.append({
                "trajectory": trajectory,
                "mask_index": mask_index,
                "utility_nats": float(
                    (standard_nll[trajectory] - random_nll[mask_index, trajectory]).item()
                ),
                "features": values,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"{args.label}_{utility['checkpoint']}_seed{utility['seed']}_raw.json"
    raw_path.write_text(json.dumps({
        "utility_json": str(utility_path),
        "n_trajectories": n_trajectories,
        "masks_per_trajectory": utility["n_masks"],
        "hold_horizon": args.hold_horizon,
        "rows": rows,
    }, indent=2))
    print(f"Saved raw rows -> {raw_path}", flush=True)

    names, screens, coefficients, predictions = analyze(
        rows, utility, args.folds, args.ridge
    )
    compact = dict(sorted(
        screens.items(),
        key=lambda item: item[1]["pairwise_ranking_accuracy"],
        reverse=True,
    ))
    print(json.dumps(compact, indent=2))
    result = {
        "utility_json": str(utility_path),
        "n_trajectories": n_trajectories,
        "masks_per_trajectory": utility["n_masks"],
        "hold_horizon": args.hold_horizon,
        "single_position_probes_per_trajectory": utility["max_length"] - utility["prefix_length"],
        "feature_names": names,
        "screens": screens,
        "mean_standardized_coefficients": coefficients,
        "rows": rows,
        "oof_predictions": predictions.tolist(),
    }
    output_path = OUT_DIR / f"{args.label}_{utility['checkpoint']}_seed{utility['seed']}.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
