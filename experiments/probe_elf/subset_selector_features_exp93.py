#!/usr/bin/env python3
"""EXP-93 Stage 2: inference-time subset features and grouped OOF ranking."""

import argparse
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

import paired_conditional_revalidation_exp80 as exp80
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
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--label", default="stage2_oof")
    return parser.parse_args()


@torch.no_grad()
def lexical_stats(x_pred, model):
    batch = x_pred.shape[0]
    z_in = torch.cat([x_pred.detach(), torch.zeros_like(x_pred)], dim=-1)
    ones = torch.ones(batch, dtype=x_pred.dtype, device=x_pred.device)
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=x_pred.device.type == "cuda"
    ):
        _, logits, _ = model(
            z_in,
            ones,
            deterministic=True,
            self_cond_cfg_scale=ones,
            decoder_step_active=True,
        )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    confidence, token_ids = probs.max(dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return token_ids, confidence, entropy


def run_lengths(mask):
    longest = current = 0
    for value in mask.tolist():
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def static_features(x_row, confidence, token_ids, selected):
    eligible = torch.ones_like(selected, dtype=torch.bool)
    unresolved = eligible & ~selected
    positions = torch.nonzero(selected, as_tuple=False).flatten().float()
    selected_conf = confidence[selected].float()
    selected_x = F.normalize(x_row[selected].float(), dim=-1)
    unresolved_x = F.normalize(x_row[unresolved].float(), dim=-1)
    coverage = unresolved_x @ selected_x.T
    facility = coverage.max(dim=1).values
    pairwise = selected_x @ selected_x.T
    upper = pairwise[torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)]
    nearest = torch.cdist(
        torch.arange(selected.numel(), device=selected.device).float().unsqueeze(-1),
        positions.unsqueeze(-1),
    ).min(dim=1).values
    block_ids = torch.div(positions.long() * 8, selected.numel(), rounding_mode="floor")
    occupied = torch.bincount(block_ids, minlength=8) > 0
    q = torch.quantile(selected_conf, torch.tensor([0.25, 0.5, 0.75], device=x_row.device))
    return {
        "reliability_mean": float(selected_conf.mean().item()),
        "reliability_min": float(selected_conf.min().item()),
        "reliability_q25": float(q[0].item()),
        "reliability_q50": float(q[1].item()),
        "reliability_q75": float(q[2].item()),
        "spatial_max_uncovered_gap": run_lengths(selected) / selected.numel(),
        "spatial_mean_nearest": float(nearest.mean().item() / selected.numel()),
        "spatial_span": float((positions.max() - positions.min()).item() / max(selected.numel() - 1, 1)),
        "spatial_block_occupancy": float(occupied.float().mean().item()),
        "latent_coverage_mean": float(facility.mean().item()),
        "latent_coverage_min": float(facility.min().item()),
        "latent_redundancy_mean": float(upper.mean().item()),
        "latent_redundancy_max": float(upper.max().item()),
        "lexical_unique_ratio": float(token_ids[selected].unique().numel() / selected.sum().item()),
        "_selected_token_ids": token_ids[selected].detach().cpu().tolist(),
    }


@torch.no_grad()
def extract_rows(utility, model, cond_noise, cond_seq, cond_mask):
    device = cond_noise.device
    grid = get_sampling_steps(utility["n_steps"], "uniform", device=device)
    random_nll = torch.tensor(utility["per_sequence"]["random_nll"], dtype=torch.float64)
    standard_nll = torch.tensor(utility["per_sequence"]["standard_nll"], dtype=torch.float64)
    rows = []
    global_frequency = torch.zeros(32100, dtype=torch.long)
    cfg = common.SamplingConfig()

    for batch_index, start in enumerate(range(0, utility["n_cond"], utility["batch_size"])):
        end = min(start + utility["batch_size"], utility["n_cond"])
        base_seq = cond_seq[start:end]
        base_mask = cond_mask[start:end]
        z = restore_cond(cond_noise[start:end].clone(), base_seq, base_mask)
        x_pred = restore_cond(torch.zeros_like(z), base_seq, base_mask)
        trigger_index = None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for index in range(grid.shape[0] - 1):
                z, x_pred = _ode_step(
                    model=model,
                    z=z,
                    t=grid[index].item(),
                    t_next=grid[index + 1].item(),
                    x_pred_prev=x_pred,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=utility["sccfg"],
                    cond_seq=base_seq,
                    cond_seq_mask=base_mask,
                )
                if grid[index + 1].item() >= utility["fixed_policy"]["trigger"]:
                    trigger_index = index
                    break
        if trigger_index is None or trigger_index + 2 >= grid.shape[0]:
            raise RuntimeError("failed to locate a valid trigger step")

        trigger_ids, trigger_conf, _ = lexical_stats(x_pred, model)
        suffix_ids = trigger_ids[:, utility["prefix_length"] :]
        global_frequency += torch.bincount(suffix_ids.cpu().flatten(), minlength=32100)
        next_index = trigger_index + 1
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, base_next_x = _ode_step(
                model=model,
                z=z,
                t=grid[next_index].item(),
                t_next=grid[next_index + 1].item(),
                x_pred_prev=x_pred,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=utility["sccfg"],
                cond_seq=base_seq,
                cond_seq_mask=base_mask,
            )
        base_ids, base_conf, base_entropy = lexical_stats(base_next_x, model)

        for mask_index in range(utility["n_masks"]):
            mask_seed = utility["seed"] + (mask_index + 1) * 1009
            rng_seed = mask_seed * 100003 + batch_index * 7919
            generator = torch.Generator(device=device).manual_seed(rng_seed)
            scores = torch.rand(trigger_conf.shape, generator=generator, device=device)
            selected_full = exp82.exact_budget_mask(
                scores, base_mask < 0.5, utility["fixed_policy"]["density"]
            )
            selected_count = int(selected_full.sum().item())
            if selected_count != (end - start) * (utility["max_length"] - utility["prefix_length"]) // 2:
                raise RuntimeError("reconstructed mask violates exact density")

            anchor_seq = torch.where(selected_full.unsqueeze(-1), x_pred, base_seq)
            anchor_mask = torch.maximum(base_mask, selected_full.to(base_mask.dtype))
            anchor_z = restore_cond(z.clone(), anchor_seq, anchor_mask)
            anchor_x = restore_cond(x_pred.clone(), anchor_seq, anchor_mask)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, shadow_x = _ode_step(
                    model=model,
                    z=anchor_z,
                    t=grid[next_index].item(),
                    t_next=grid[next_index + 1].item(),
                    x_pred_prev=anchor_x,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=utility["sccfg"],
                    cond_seq=anchor_seq,
                    cond_seq_mask=anchor_mask,
                )
            shadow_ids, shadow_conf, shadow_entropy = lexical_stats(shadow_x, model)

            for local_row, trajectory in enumerate(range(start, end)):
                suffix = slice(utility["prefix_length"], utility["max_length"])
                selected = selected_full[local_row, suffix]
                unresolved = ~selected
                features = static_features(
                    x_pred[local_row, suffix],
                    trigger_conf[local_row, suffix],
                    trigger_ids[local_row, suffix],
                    selected,
                )
                base_conf_row = base_conf[local_row, suffix][unresolved]
                shadow_conf_row = shadow_conf[local_row, suffix][unresolved]
                base_entropy_row = base_entropy[local_row, suffix][unresolved]
                shadow_entropy_row = shadow_entropy[local_row, suffix][unresolved]
                features.update({
                    "shadow_delta_confidence": float((shadow_conf_row - base_conf_row).mean().item()),
                    "shadow_entropy_reduction": float((base_entropy_row - shadow_entropy_row).mean().item()),
                    "shadow_top1_flip_fraction": float(
                        (shadow_ids[local_row, suffix][unresolved] != base_ids[local_row, suffix][unresolved])
                        .float().mean().item()
                    ),
                })
                rows.append({
                    "trajectory": trajectory,
                    "mask_index": mask_index,
                    "utility_nats": float((standard_nll[trajectory] - random_nll[mask_index, trajectory]).item()),
                    "final_nll": float(random_nll[mask_index, trajectory].item()),
                    "features": features,
                })

    for row in rows:
        token_ids = torch.tensor(row["features"].pop("_selected_token_ids"), dtype=torch.long)
        frequency = global_frequency[token_ids].float().clamp_min(1).log()
        q = torch.quantile(frequency, torch.tensor([0.25, 0.5, 0.75]))
        row["features"].update({
            "lexical_logfreq_mean": float(frequency.mean().item()),
            "lexical_logfreq_q25": float(q[0].item()),
            "lexical_logfreq_q50": float(q[1].item()),
            "lexical_logfreq_q75": float(q[2].item()),
        })

    # Reconstructed masks must reproduce Stage-1 aggregate confidence exactly.
    for mask_index in range(utility["n_masks"]):
        observed = sum(
            row["features"]["reliability_mean"]
            for row in rows if row["mask_index"] == mask_index
        ) / utility["n_cond"]
        expected = utility["rollout_info"][f"random_{mask_index:02d}"]["anchor_confidence"]
        if abs(observed - expected) > 2e-5:
            raise RuntimeError(
                f"mask reconstruction mismatch m={mask_index}: {observed} != {expected}"
            )
    return rows


def rankdata(values):
    """Average ranks for ties, matching the usual Spearman definition."""
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def ranking_metrics(scores, utilities, trajectories):
    spearman, pair_correct, pair_total, top1 = [], 0.0, 0, 0
    for trajectory in trajectories.unique(sorted=True):
        chosen = trajectories == trajectory
        score = scores[chosen]
        utility = utilities[chosen]
        sr, ur = rankdata(score), rankdata(utility)
        if sr.std() > 0 and ur.std() > 0:
            spearman.append(float(torch.corrcoef(torch.stack([sr, ur]))[0, 1].item()))
        diff_s = score[:, None] - score[None, :]
        diff_u = utility[:, None] - utility[None, :]
        upper = torch.triu(torch.ones_like(diff_s, dtype=torch.bool), diagonal=1)
        products = diff_s[upper] * diff_u[upper]
        pair_correct += float((products > 0).sum().item())
        pair_correct += 0.5 * float((products == 0).sum().item())
        pair_total += int(products.numel())
        top1 += int(score.argmax().item() == utility.argmax().item())
    count = int(trajectories.unique().numel())
    return {
        "mean_trajectory_spearman": sum(spearman) / len(spearman),
        "pairwise_ranking_accuracy": pair_correct / pair_total,
        "oracle_mask_top1_accuracy": top1 / count,
    }


def grouped_oof_ridge(x, y, trajectories, folds, ridge, seed):
    unique = trajectories.unique(sorted=True)
    generator = torch.Generator().manual_seed(seed + 930029)
    shuffled = unique[torch.randperm(unique.numel(), generator=generator)]
    fold_id = torch.empty(int(unique.max().item()) + 1, dtype=torch.long)
    for index, trajectory in enumerate(shuffled.tolist()):
        fold_id[trajectory] = index % folds
    predictions = torch.empty_like(y)
    coefficients = []
    for fold in range(folds):
        test = fold_id[trajectories] == fold
        train = ~test
        mean = x[train].mean(dim=0)
        std = x[train].std(dim=0).clamp_min(1e-6)
        x_train = (x[train] - mean) / std
        y_mean = y[train].mean()
        y_train = y[train] - y_mean
        eye = torch.eye(x.shape[1], dtype=torch.float64)
        beta = torch.linalg.solve(x_train.T @ x_train + ridge * eye, x_train.T @ y_train)
        predictions[test] = ((x[test] - mean) / std) @ beta + y_mean
        coefficients.append(beta)
    return predictions, torch.stack(coefficients).mean(dim=0)


def selected_ppl(scores, rows, utility):
    nll = torch.tensor(utility["per_sequence"]["random_nll"], dtype=torch.float64)
    counts = torch.tensor(utility["per_sequence"]["random_token_counts"], dtype=torch.float64)
    trajectories = torch.tensor([row["trajectory"] for row in rows], dtype=torch.long)
    mask_indices = torch.tensor([row["mask_index"] for row in rows], dtype=torch.long)
    chosen_nll, chosen_counts = [], []
    for trajectory in range(utility["n_cond"]):
        candidate_rows = torch.nonzero(trajectories == trajectory, as_tuple=False).flatten()
        best_row = candidate_rows[scores[candidate_rows].argmax()]
        mask = int(mask_indices[best_row].item())
        chosen_nll.append(nll[mask, trajectory])
        chosen_counts.append(counts[mask, trajectory])
    chosen_nll = torch.stack(chosen_nll)
    chosen_counts = torch.stack(chosen_counts)
    return math.exp(float((chosen_nll * chosen_counts).sum() / chosen_counts.sum()))


def main():
    args = parse_args()
    utility_path = Path(args.utility_json)
    utility = json.loads(utility_path.read_text())
    if args.folds < 2 or args.folds > utility["n_cond"]:
        raise ValueError("folds must lie between 2 and n_cond")
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
        pairs,
        tokenizer,
        encoder,
        device,
        utility["max_length"],
        utility["prefix_length"],
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
    rows = extract_rows(utility, model, cond_noise, cond_seq, cond_mask)

    feature_names = sorted(rows[0]["features"])
    x = torch.tensor(
        [[row["features"][name] for name in feature_names] for row in rows],
        dtype=torch.float64,
    )
    y = torch.tensor([row["utility_nats"] for row in rows], dtype=torch.float64)
    trajectories = torch.tensor([row["trajectory"] for row in rows], dtype=torch.long)
    predictions, coefficients = grouped_oof_ridge(
        x, y, trajectories, args.folds, args.ridge, utility["seed"]
    )
    screens = {
        "grouped_oof_ridge": ranking_metrics(predictions, y, trajectories),
        "mean_confidence": ranking_metrics(
            x[:, feature_names.index("reliability_mean")], y, trajectories
        ),
        "latent_coverage": ranking_metrics(
            x[:, feature_names.index("latent_coverage_mean")], y, trajectories
        ),
        "negative_redundancy": ranking_metrics(
            -x[:, feature_names.index("latent_redundancy_mean")], y, trajectories
        ),
        "shadow_confidence_gain": ranking_metrics(
            x[:, feature_names.index("shadow_delta_confidence")], y, trajectories
        ),
        "shadow_entropy_reduction": ranking_metrics(
            x[:, feature_names.index("shadow_entropy_reduction")], y, trajectories
        ),
    }
    screens["grouped_oof_ridge"]["selected_prompt_conditioned_ppl"] = selected_ppl(
        predictions, rows, utility
    )
    screens["grouped_oof_ridge"]["mean_random_prompt_conditioned_ppl"] = utility["aggregate"]["mean_random"]["prompt_conditioned_ppl"]
    screens["grouped_oof_ridge"]["oracle_prompt_conditioned_ppl"] = utility["aggregate"]["oracle_best_of_m"]["prompt_conditioned_ppl"]
    screens["grouped_oof_ridge"]["top_confidence_prompt_conditioned_ppl"] = utility["aggregate"]["top_confidence"]["prompt_conditioned_ppl"]

    coefficient_map = {
        name: float(value) for name, value in zip(feature_names, coefficients.tolist())
    }
    result = {
        "utility_json": str(utility_path),
        "n_trajectories": utility["n_cond"],
        "masks_per_trajectory": utility["n_masks"],
        "folds": args.folds,
        "ridge": args.ridge,
        "feature_names": feature_names,
        "screens": screens,
        "mean_standardized_coefficients": coefficient_map,
        "rows": rows,
        "oof_predictions": predictions.tolist(),
    }
    print(json.dumps({"screens": screens, "coefficients": coefficient_map}, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.label}_{utility['checkpoint']}_seed{utility['seed']}.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
