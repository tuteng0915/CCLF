#!/usr/bin/env python3
"""EXP-100: train a non-additive Plaid temporary-anchor subset scorer."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_bank", required=True)
    parser.add_argument("--validation_bank", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--target_temperature", type=float, default=0.20)
    return parser.parse_args()


class JointSubsetScorer(nn.Module):
    def __init__(self, feature_dim, seq_len, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.membership_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 1),
        )

    @staticmethod
    def masked_mean(values, mask):
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, features, candidate_masks, prefix_length):
        batch, candidates, seq_len = candidate_masks.shape
        encoded = self.input_projection(features.float())
        encoded = encoded[:, None].expand(-1, candidates, -1, -1)
        positions = self.position_embedding(
            torch.arange(seq_len, device=features.device)
        )[None, None]
        encoded = encoded + positions
        encoded = encoded + self.membership_embedding(candidate_masks.long())
        encoded = self.encoder(encoded.reshape(batch * candidates, seq_len, -1))

        membership = candidate_masks.reshape(batch * candidates, seq_len)
        prefix = torch.arange(seq_len, device=features.device) < prefix_length
        prefix = prefix[None].expand(batch * candidates, -1)
        eligible = ~prefix
        selected = membership & eligible
        unresolved = (~membership) & eligible
        pooled = torch.cat(
            (
                self.masked_mean(encoded, selected),
                self.masked_mean(encoded, unresolved),
                self.masked_mean(encoded, prefix),
                encoded.mean(dim=1),
            ),
            dim=-1,
        )
        return self.head(pooled).reshape(batch, candidates)


def load_bank(path):
    bank = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "features",
        "candidate_masks",
        "candidate_nll",
        "candidate_token_counts",
        "standard_nll",
        "top_confidence_nll",
        "prefix_length",
        "density",
    )
    missing = [key for key in required if key not in bank]
    if missing:
        raise ValueError(f"bank {path} missing {missing}")
    return bank


def normalize_features(train, validation):
    values = train["features"].float()
    mean = values.mean(dim=(0, 1), keepdim=True)
    std = values.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    train["features"] = ((values - mean) / std).half()
    validation["features"] = (
        (validation["features"].float() - mean) / std
    ).half()
    return mean.squeeze().cpu(), std.squeeze().cpu()


def pairwise_accuracy(scores, nll):
    score_diff = scores[:, :, None] - scores[:, None, :]
    utility_diff = -nll[:, :, None] + nll[:, None, :]
    upper = torch.triu(
        torch.ones(score_diff.shape[1:], dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    valid = upper[None] & (utility_diff.abs() > 1e-8)
    return float(((score_diff * utility_diff) > 0)[valid].float().mean().cpu())


def mean_spearman(scores, nll):
    score_rank = scores.argsort(dim=1).argsort(dim=1).float()
    utility_rank = (-nll).argsort(dim=1).argsort(dim=1).float()
    score_rank = score_rank - score_rank.mean(dim=1, keepdim=True)
    utility_rank = utility_rank - utility_rank.mean(dim=1, keepdim=True)
    numerator = (score_rank * utility_rank).sum(dim=1)
    denominator = score_rank.square().sum(dim=1).sqrt() * utility_rank.square().sum(
        dim=1
    ).sqrt()
    return float((numerator / denominator.clamp_min(1e-12)).mean().cpu())


def aggregate_ppl(nll, counts):
    return math.exp(float((nll.double() * counts.double()).sum() / counts.sum()))


@torch.no_grad()
def evaluate(model, bank, device, batch_size):
    model.eval()
    all_scores = []
    count = bank["features"].shape[0]
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        scores = model(
            bank["features"][start:end].to(device),
            bank["candidate_masks"][start:end].to(device),
            bank["prefix_length"],
        )
        all_scores.append(scores.cpu())
    scores = torch.cat(all_scores)
    nll = bank["candidate_nll"].float()
    counts = bank["candidate_token_counts"].long()
    selected_index = scores.argmax(dim=1)
    rows = torch.arange(len(selected_index))
    selected_nll = nll[rows, selected_index]
    selected_counts = counts[rows, selected_index]
    mean_random_nll = nll.mean(dim=1)
    oracle_nll = nll.min(dim=1).values
    delta = selected_nll - mean_random_nll
    generator = torch.Generator().manual_seed(100003 + int(bank["seed"]))
    bootstrap = torch.empty(2000)
    for index in range(len(bootstrap)):
        sampled = torch.randint(0, len(delta), (len(delta),), generator=generator)
        bootstrap[index] = delta[sampled].mean()
    return {
        "scores": scores,
        "selected_index": selected_index,
        "selected_nll": selected_nll,
        "selected_ppl": aggregate_ppl(selected_nll, selected_counts),
        "mean_random_ppl": aggregate_ppl(nll.flatten(), counts.flatten()),
        "oracle_ppl": aggregate_ppl(
            oracle_nll, counts[rows, nll.argmin(dim=1)]
        ),
        "top_confidence_ppl_proxy": math.exp(
            float(bank["top_confidence_nll"].double().mean())
        ),
        "pairwise_accuracy": pairwise_accuracy(scores, nll),
        "mean_spearman": mean_spearman(scores, nll),
        "selected_beats_mean_random_fraction": float((delta < 0).float().mean()),
        "selected_minus_mean_random_nll": float(delta.mean()),
        "selected_minus_mean_random_ci95": [
            float(torch.quantile(bootstrap, 0.025)),
            float(torch.quantile(bootstrap, 0.975)),
        ],
    }


def serializable(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in ("scores", "selected_index", "selected_nll")
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train = load_bank(args.train_bank)
    validation = load_bank(args.validation_bank)
    for key in ("density", "trigger_step", "horizon", "prefix_length"):
        if train[key] != validation[key]:
            raise ValueError(f"train/validation mismatch for {key}")
    feature_mean, feature_std = normalize_features(train, validation)

    model_args = {
        "feature_dim": train["features"].shape[-1],
        "seq_len": train["features"].shape[1],
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
    }
    model = JointSubsetScorer(**model_args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed + 1009)
    best_state, best_metrics, best_epoch = None, None, -1
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(train["features"].shape[0], generator=generator)
        losses = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            features = train["features"][rows].to(device)
            masks = train["candidate_masks"][rows].to(device)
            nll = train["candidate_nll"][rows].to(device)
            scores = model(features, masks, train["prefix_length"])
            target = torch.softmax(-nll / args.target_temperature, dim=1)
            loss = -(target * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(model, validation, device, args.batch_size)
        criterion = metrics["selected_minus_mean_random_nll"]
        if best_metrics is None or criterion < best_metrics[
            "selected_minus_mean_random_nll"
        ]:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = metrics
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
                f"val_pair={metrics['pairwise_accuracy']:.4f} "
                f"val_delta={metrics['selected_minus_mean_random_nll']:+.4f}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            break

    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train, device, args.batch_size)
    validation_metrics = evaluate(model, validation, device, args.batch_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.label}_selector_seed{args.seed}.pt"
    torch.save(
        {
            "model_state": best_state,
            "model_args": model_args,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "prefix_length": train["prefix_length"],
            "density": train["density"],
            "trigger_step": train["trigger_step"],
            "horizon": train["horizon"],
            "training_seed": args.seed,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    result = {
        **vars(args),
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "model_args": model_args,
        "train": serializable(train_metrics),
        "validation": serializable(validation_metrics),
    }
    result_path = output_dir / f"{args.label}_selector_seed{args.seed}.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved -> {checkpoint_path} and {result_path}", flush=True)


if __name__ == "__main__":
    main()
