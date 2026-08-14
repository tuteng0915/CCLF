#!/usr/bin/env python3
"""EXP-104: distill the native entropy-response teacher into an online rule."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_late_coupled_blocks as quality_base  # noqa: E402
import eval_plaid_conditional_late_coupling as conditional_base  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_banks", nargs="+", required=True)
    parser.add_argument("--train_locals", nargs="+", required=True)
    parser.add_argument("--calibration_bank", required=True)
    parser.add_argument("--calibration_local", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="controller")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--ranking_weight", type=float, default=0.25)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    return parser.parse_args()


class TriggerController(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def load_pair(bank_path, local_path):
    bank = json.loads(Path(bank_path).read_text())
    local = json.loads(Path(local_path).read_text())
    for key in ("seed", "panel_offset", "triggers", "fixed_trigger"):
        if bank[key] != local[key]:
            raise ValueError(f"bank/local mismatch for {key}: {bank_path}")
    feature_names = list(bank["feature_names"]) + ["normalized_trigger_step"]
    features = torch.tensor(bank["event_features"], dtype=torch.float32)
    trigger_time = torch.tensor(bank["triggers"], dtype=torch.float32)
    trigger_time = (trigger_time / float(bank["n_steps"]))[None, :, None]
    trigger_time = trigger_time.expand(features.shape[0], -1, -1)
    features = torch.cat((features, trigger_time), dim=-1)
    lookahead_index = local["lookaheads"].index(4)
    signal_index = local["signal_names"].index("entropy_reduction")
    target = torch.tensor(local["local_signals"], dtype=torch.float32)[
        :, :, lookahead_index, signal_index
    ]
    return {
        "bank": bank,
        "local": local,
        "features": features,
        "target": target,
        "feature_names": feature_names,
    }


def quality_gate(delta):
    return (
        delta["d1"] >= -0.005
        and delta["rep4"] <= 0.005
        and delta["degeneration_rate"] <= 0.015
        and delta["prompt_gain_nats"] >= -0.01
    )


def aggregate_ppl(nll, counts):
    return math.exp(float((nll * counts.double()).sum() / counts.sum()))


def bootstrap(delta, samples, seed):
    generator = torch.Generator().manual_seed(seed + 1040037)
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(0, len(delta), (len(delta),), generator=generator)
        estimates[index] = delta[rows].mean()
    return [
        float(torch.quantile(estimates, 0.025)),
        float(torch.quantile(estimates, 0.975)),
    ]


def pairwise_accuracy(prediction, target):
    pred_diff = prediction[:, :, None] - prediction[:, None, :]
    target_diff = target[:, :, None] - target[:, None, :]
    candidates = prediction.shape[1]
    upper = torch.triu(torch.ones(candidates, candidates, dtype=torch.bool), diagonal=1)
    valid = upper[None] & (target_diff.abs() > 1e-8)
    return float(((pred_diff * target_diff) > 0)[valid].float().mean())


def pearson(prediction, target):
    left = prediction.flatten().double()
    right = target.flatten().double()
    left -= left.mean()
    right -= right.mean()
    return float(
        (left * right).sum()
        / (left.square().sum().sqrt() * right.square().sum().sqrt()).clamp_min(1e-12)
    )


def select_online(prediction, triggers, fixed_trigger, threshold):
    fixed_index = triggers.index(fixed_trigger)
    allowed = [index for index, step in enumerate(triggers) if step <= fixed_trigger]
    selected = torch.full((len(prediction),), fixed_index, dtype=torch.long)
    checks = torch.full((len(prediction),), len(allowed), dtype=torch.long)
    undecided = torch.ones(len(prediction), dtype=torch.bool)
    for order, index in enumerate(allowed[:-1], start=1):
        fire = undecided & (prediction[:, index] >= threshold)
        selected[fire] = index
        checks[fire] = order
        undecided &= ~fire
    return selected, checks


def evaluate_policy(data, prediction, threshold, samples, seed):
    bank = data["bank"]
    triggers = bank["triggers"]
    fixed_index = triggers.index(bank["fixed_trigger"])
    selected_index, checks = select_online(
        prediction, triggers, bank["fixed_trigger"], threshold
    )
    rows = torch.arange(len(selected_index))
    nll = torch.tensor(bank["per_sequence"]["trigger_nll"], dtype=torch.float64).T
    counts = torch.tensor(
        bank["per_sequence"]["trigger_token_counts"], dtype=torch.long
    ).T
    shuffled = torch.tensor(
        bank["per_sequence"]["trigger_shuffled_nll"], dtype=torch.float64
    ).T
    selected_nll = nll[rows, selected_index]
    selected_counts = counts[rows, selected_index]
    selected_shuffled = shuffled[rows, selected_index]
    fixed_nll = nll[:, fixed_index]
    fixed_counts = counts[:, fixed_index]
    delta = selected_nll - fixed_nll
    selected_texts = [
        bank["texts"]["by_trigger"][str(triggers[int(index)])][row]
        for row, index in enumerate(selected_index)
    ]
    quality = quality_base.text_quality(selected_texts)
    selected_ppl = aggregate_ppl(selected_nll, selected_counts)
    shuffled_ppl = aggregate_ppl(selected_shuffled, selected_counts)
    quality.update(
        prompt_conditioned_ppl=selected_ppl,
        shuffled_prompt_ppl=shuffled_ppl,
        prompt_gain_nats=math.log(shuffled_ppl) - math.log(selected_ppl),
        rouge_l=float(
            np.mean(
                [
                    conditional_base.rouge_l_f1(prediction_text, reference)
                    for prediction_text, reference in zip(
                        selected_texts, bank["texts"]["references"]
                    )
                ]
            )
        ),
    )
    fixed_quality = bank["aggregate"][f"trigger_{bank['fixed_trigger']:02d}"]
    quality_delta = {
        "d1": quality["d1"] - fixed_quality["d1"],
        "d2": quality["d2"] - fixed_quality["d2"],
        "rep4": quality["rep4"] - fixed_quality["rep4"],
        "degeneration_rate": quality["degeneration_rate"]
        - fixed_quality["degeneration_rate"],
        "prompt_gain_nats": quality["prompt_gain_nats"]
        - fixed_quality["prompt_gain_nats"],
    }
    interval = bootstrap(delta, samples, seed)
    early_fraction = float((selected_index != fixed_index).double().mean())
    return {
        "threshold": threshold,
        "selected_ppl": selected_ppl,
        "fixed_ppl": aggregate_ppl(fixed_nll, fixed_counts),
        "mean_delta_nats": float(delta.mean()),
        "mean_delta_ci95": interval,
        "early_trigger_fraction": early_fraction,
        "mean_controller_checks": float(checks.double().mean()),
        "selected_trigger_histogram": {
            str(step): int((selected_index == index).sum())
            for index, step in enumerate(triggers)
        },
        "quality": quality,
        "quality_delta": quality_delta,
        "gate_passed": (
            interval[1] < 0.0
            and quality_gate(quality_delta)
            and 0.0 < early_fraction < 1.0
        ),
    }


@torch.no_grad()
def predict(model, features, mean, std, device):
    model.eval()
    return model(((features - mean) / std).to(device)).cpu()


def main():
    args = parse_args()
    if len(args.train_banks) != len(args.train_locals):
        raise ValueError("train_banks and train_locals must have equal length")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_parts = [
        load_pair(bank, local)
        for bank, local in zip(args.train_banks, args.train_locals)
    ]
    calibration = load_pair(args.calibration_bank, args.calibration_local)
    reference = train_parts[0]
    for part in train_parts[1:] + [calibration]:
        for key in ("triggers", "fixed_trigger", "feature_names"):
            left = reference["bank"].get(key, reference.get(key))
            right = part["bank"].get(key, part.get(key))
            if left != right:
                raise ValueError(f"bank mismatch for {key}")

    train_features = torch.cat([part["features"] for part in train_parts])
    train_target = torch.cat([part["target"] for part in train_parts])
    feature_mean = train_features.mean(dim=(0, 1), keepdim=True)
    feature_std = train_features.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    target_mean = train_target.mean()
    target_std = train_target.std().clamp_min(1e-4)
    normalized_features = (train_features - feature_mean) / feature_std
    normalized_target = (train_target - target_mean) / target_std

    model = TriggerController(train_features.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed + 1049)
    upper = torch.triu(
        torch.ones(train_target.shape[1], train_target.shape[1], dtype=torch.bool),
        diagonal=1,
    )
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_features), generator=generator)
        losses = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            features = normalized_features[rows].to(device)
            target = normalized_target[rows].to(device)
            prediction = model(features)
            mse = F.mse_loss(prediction, target)
            pred_diff = prediction[:, :, None] - prediction[:, None, :]
            target_diff = target[:, :, None] - target[:, None, :]
            valid = upper.to(device)[None] & (target_diff.abs() > 1e-6)
            sign = target_diff.sign()
            ranking = F.softplus(-(pred_diff * sign))[valid].mean()
            loss = mse + args.ranking_weight * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch:03d} loss={np.mean(losses):.5f}", flush=True)

    train_prediction = predict(
        model, train_features, feature_mean, feature_std, device
    ) * target_std + target_mean
    calibration_prediction = predict(
        model, calibration["features"], feature_mean, feature_std, device
    ) * target_std + target_mean
    calibration_pair = pairwise_accuracy(
        calibration_prediction, calibration["target"]
    )

    fixed_index = calibration["bank"]["triggers"].index(
        calibration["bank"]["fixed_trigger"]
    )
    candidate_scores = calibration_prediction[:, : fixed_index + 1].flatten()
    thresholds = torch.cat(
        (
            torch.tensor([-float("inf")]),
            torch.quantile(candidate_scores, torch.linspace(0.0, 1.0, 65)).unique(),
            torch.tensor([float("inf")]),
        )
    ).tolist()
    candidates = [
        evaluate_policy(
            calibration,
            calibration_prediction,
            threshold,
            args.bootstrap_samples,
            args.seed,
        )
        for threshold in thresholds
    ]
    passing = [
        row
        for row in candidates
        if row["gate_passed"] and calibration_pair > 0.55
    ]
    best = min(passing, key=lambda row: row["mean_delta_nats"]) if passing else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{args.label}_seed{args.seed}.pt"
    torch.save(
        {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "feature_dim": train_features.shape[-1],
            "hidden_dim": args.hidden_dim,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "feature_names": reference["feature_names"],
            "triggers": reference["bank"]["triggers"],
            "fixed_trigger": reference["bank"]["fixed_trigger"],
            "threshold": None if best is None else best["threshold"],
            "training_seed": args.seed,
        },
        checkpoint,
    )
    result = {
        **vars(args),
        "checkpoint": str(checkpoint),
        "n_train_trajectories": len(train_features),
        "teacher_target": "lookahead-4 unresolved entropy reduction",
        "train_teacher_pairwise_accuracy": pairwise_accuracy(
            train_prediction, train_target
        ),
        "train_teacher_pearson": pearson(train_prediction, train_target),
        "calibration_teacher_pairwise_accuracy": calibration_pair,
        "calibration_teacher_pearson": pearson(
            calibration_prediction, calibration["target"]
        ),
        "calibration_gate_passed": best is not None,
        "frozen_threshold": None if best is None else best["threshold"],
        "calibration": best,
        "top_passing_thresholds": sorted(
            passing, key=lambda row: row["mean_delta_nats"]
        )[:5],
    }
    result_path = output_dir / f"{args.label}_seed{args.seed}.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved -> {checkpoint} and {result_path}", flush=True)


if __name__ == "__main__":
    main()

