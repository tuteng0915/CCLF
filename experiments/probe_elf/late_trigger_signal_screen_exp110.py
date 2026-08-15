#!/usr/bin/env python3
"""EXP-110 Stage A: screen deterministic signals for .40 vs .45 Unlock-4."""

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

import paired_conditional_revalidation_exp80 as exp80  # noqa: E402
import robust_revisable_commit_exp78 as exp78  # noqa: E402
import unified_method_eval_exp64 as common  # noqa: E402
from modules.model import ELF_B  # noqa: E402
from modules.t5_encoder import get_encoder  # noqa: E402
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond  # noqa: E402


SIGN_ORIENTATION = {
    # Positive oriented score should mean "wait until .45".
    "current_entropy_mean": 1.0,
    "current_confidence_mean": -1.0,
    "current_margin_mean": -1.0,
    "current_anchor_fraction": -1.0,
    "current_stability": -1.0,
    "current_displacement": 1.0,
    "shadow_entropy_response": 1.0,
    "shadow_confidence_response": 1.0,
    "shadow_top1_disagreement": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare_time", type=float, default=.625)
    return parser.parse_args()


@torch.no_grad()
def lexical_stats(x_pred, model):
    batch = x_pred.shape[0]
    model_input = torch.cat([x_pred.detach(), torch.zeros_like(x_pred)], dim=-1)
    ones = torch.ones(batch, dtype=x_pred.dtype, device=x_pred.device)
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=x_pred.device.type == "cuda"
    ):
        _, logits, _ = model(
            model_input,
            ones,
            deterministic=True,
            self_cond_cfg_scale=ones,
            decoder_step_active=True,
        )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    top = probabilities.topk(2, dim=-1)
    confidence = top.values[..., 0]
    margin = confidence - top.values[..., 1]
    entropy = -(probabilities * log_probs).sum(dim=-1)
    return top.indices[..., 0], confidence, margin, entropy


def row_means(values, suffix):
    return values[:, suffix].float().mean(dim=1)


@torch.no_grad()
def trace_arm(z0, model, grid, payload, cond_seq, cond_mask, commit_time, compare_time):
    cfg = common.SamplingConfig()
    base_seq, base_mask = cond_seq.clone(), cond_mask.clone()
    active_seq, active_mask = cond_seq.clone(), cond_mask.clone()
    z = restore_cond(z0.clone(), active_seq, active_mask)
    x_pred = restore_cond(torch.zeros_like(z), active_seq, active_mask)
    suffix = slice(int(payload["prefix_length"]), int(payload["max_length"]))
    previous_ids = None
    current_features = None
    compare = None
    committed = False
    release_index = None

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for index in range(grid.shape[0] - 1):
            if release_index is not None and index >= release_index:
                active_seq = base_seq.clone()
                active_mask = base_mask.clone()
                release_index = None

            x_before = x_pred
            z, x_pred = _ode_step(
                model=model,
                z=z,
                t=grid[index].item(),
                t_next=grid[index + 1].item(),
                x_pred_prev=x_pred,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=float(payload["sccfg"]),
                cond_seq=active_seq,
                cond_seq_mask=active_mask,
            )
            t_next = float(grid[index + 1])

            if previous_ids is None and t_next + 1e-9 >= .375:
                previous_ids, _, _, _ = lexical_stats(x_pred, model)

            if current_features is None and t_next + 1e-9 >= .40:
                token_ids, confidence, margin, entropy = lexical_stats(x_pred, model)
                free = base_mask[:, suffix] < .5
                displacement = (
                    (x_pred[:, suffix].float() - x_before[:, suffix].float()).norm(dim=-1)
                    / x_before[:, suffix].float().norm(dim=-1).clamp_min(1e-6)
                )
                current_features = {
                    "current_entropy_mean": row_means(entropy, suffix),
                    "current_confidence_mean": row_means(confidence, suffix),
                    "current_margin_mean": row_means(margin, suffix),
                    "current_anchor_fraction": (
                        ((confidence[:, suffix] >= float(payload["high_confidence"])) & free)
                        .float().sum(dim=1)
                        / free.float().sum(dim=1).clamp_min(1)
                    ),
                    "current_stability": (
                        ((token_ids[:, suffix] == previous_ids[:, suffix]) & free)
                        .float().sum(dim=1)
                        / free.float().sum(dim=1).clamp_min(1)
                    ),
                    "current_displacement": displacement.mean(dim=1),
                }

            if not committed and t_next + 1e-9 >= commit_time:
                token_ids, confidence, _, _ = lexical_stats(x_pred, model)
                selected = (confidence >= float(payload["high_confidence"])) & (base_mask < .5)
                active_seq = torch.where(selected.unsqueeze(-1), x_pred.detach(), active_seq)
                active_mask = torch.maximum(active_mask, selected.to(active_mask.dtype))
                z = restore_cond(z, active_seq, active_mask)
                x_pred = restore_cond(x_pred, active_seq, active_mask)
                committed = True
                release_index = index + 1 + 4

            if compare is None and t_next + 1e-9 >= compare_time:
                ids, confidence, _, entropy = lexical_stats(x_pred, model)
                compare = {
                    "ids": ids[:, suffix].detach().cpu(),
                    "entropy": row_means(entropy, suffix).detach().cpu(),
                    "confidence": row_means(confidence, suffix).detach().cpu(),
                    "actual_time": t_next,
                }

    if current_features is None or compare is None or not committed:
        raise RuntimeError("failed to capture trigger/compare states")
    return z, {
        "current": {key: value.detach().cpu() for key, value in current_features.items()},
        "compare": compare,
    }


def rankdata(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(score, target):
    x, y = rankdata(score), rankdata(target)
    x, y = x - x.mean(), y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y / denom).item()) if denom > 0 else 0.0


def sign_auc(score, target):
    score = torch.as_tensor(score, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    positive, negative = score[target > 0], score[target <= 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).float() + .5 * (comparisons == 0).float()).mean())


def bank_metrics(features, utility):
    result = {}
    for name, orientation in SIGN_ORIENTATION.items():
        score = orientation * torch.as_tensor(features[name], dtype=torch.float64)
        result[name] = {
            "orientation": orientation,
            "spearman": spearman(score, utility),
            "sign_auc": sign_auc(score, utility),
        }
    return result


@torch.no_grad()
def run_bank(payload, model, encoder, tokenizer, device, compare_time):
    args = type("Args", (), {})()
    for key in ("n_cond", "max_length", "prefix_length", "owt_offset"):
        setattr(args, key, int(payload[key]))
    args.conditional_dataset = "owt"
    pairs = exp80.load_pairs(args, tokenizer)
    cond_seq, cond_mask, _ = common.build_condition_data(
        pairs, tokenizer, encoder, device, args.max_length, args.prefix_length
    )
    generator = torch.Generator(device=device).manual_seed(int(payload["seed"]))
    z0 = float(payload["noise_scale"]) * torch.randn(
        args.n_cond,
        args.max_length,
        common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    z0[:, : args.prefix_length] = cond_seq[:, : args.prefix_length].to(z0.dtype)
    grid = get_sampling_steps(int(payload["n_steps"]), "uniform", device=device)

    final_z = {"trigger_0.40": [], "trigger_0.45": []}
    current = {}
    compare = {"trigger_0.40": {}, "trigger_0.45": {}}
    for start in range(0, args.n_cond, int(payload["batch_size"])):
        end = min(start + int(payload["batch_size"]), args.n_cond)
        for name, trigger in (("trigger_0.40", .40), ("trigger_0.45", .45)):
            z, trace = trace_arm(
                z0[start:end], model, grid, payload,
                cond_seq[start:end], cond_mask[start:end], trigger, compare_time,
            )
            final_z[name].append(z.cpu())
            if name == "trigger_0.40":
                for key, value in trace["current"].items():
                    current.setdefault(key, []).extend(value.tolist())
            for key, value in trace["compare"].items():
                if key == "actual_time":
                    compare[name][key] = value
                else:
                    compare[name].setdefault(key, []).extend(value.tolist())

    agreement = {}
    for name in final_z:
        ids = common.decode(torch.cat(final_z[name]).to(device), model, device)
        texts = common.decode_texts(ids.cpu(), tokenizer, args.prefix_length)
        saved = payload["texts"][name]
        agreement[name] = sum(a == b for a, b in zip(texts, saved)) / len(saved)
        if agreement[name] != 1.0:
            raise RuntimeError(f"{name} output agreement={agreement[name]}")

    entropy40 = torch.tensor(compare["trigger_0.40"]["entropy"])
    entropy45 = torch.tensor(compare["trigger_0.45"]["entropy"])
    confidence40 = torch.tensor(compare["trigger_0.40"]["confidence"])
    confidence45 = torch.tensor(compare["trigger_0.45"]["confidence"])
    ids40 = torch.tensor(compare["trigger_0.40"]["ids"])
    ids45 = torch.tensor(compare["trigger_0.45"]["ids"])
    current.update({
        "shadow_entropy_response": (entropy40 - entropy45).tolist(),
        "shadow_confidence_response": (confidence45 - confidence40).tolist(),
        "shadow_top1_disagreement": (ids40 != ids45).float().mean(dim=1).tolist(),
    })

    names = [f"trigger_{float(value):.2f}" for value in payload["trigger_times"]]
    nll = torch.tensor(payload["per_sequence"]["trigger_nll"], dtype=torch.float64)
    utility = nll[names.index("trigger_0.40")] - nll[names.index("trigger_0.45")]
    return {
        "seed": payload["seed"],
        "owt_offset": payload["owt_offset"],
        "output_agreement": agreement,
        "compare_actual_times": {
            name: compare[name]["actual_time"] for name in compare
        },
        "wait_better_count": int((utility > 0).sum()),
        "utility": utility.tolist(),
        "features": current,
        "metrics": bank_metrics(current, utility),
    }


def main():
    args = parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    if any(int(payload["n_steps"]) != 32 for payload in payloads):
        raise ValueError("EXP-110 Stage A requires the ODE-32 EXP-108 banks")
    if any(payload["checkpoint"] != "baseline" for payload in payloads):
        raise ValueError("EXP-110 Stage A is frozen to ELF baseline")

    device = torch.device(args.device)
    noise_scales = {float(payload["noise_scale"]) for payload in payloads}
    if len(noise_scales) != 1:
        raise ValueError("all Stage-A banks must use the same denoiser noise scale")
    common.SamplingConfig.denoiser_noise_scale = noise_scales.pop()
    checkpoint_path = REPO_ROOT / exp78.CHECKPOINTS["baseline"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(int(payloads[0]["max_length"])))
    model.load_state_dict(common.load_weights(checkpoint), strict=False)
    model.to(device).eval()

    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    banks = [
        run_bank(payload, model, encoder, tokenizer, device, args.compare_time)
        for payload in payloads
    ]
    pooled = {}
    for name in SIGN_ORIENTATION:
        scores, utilities = [], []
        for bank in banks:
            scores.extend(bank["features"][name])
            utilities.extend(bank["utility"])
        oriented = SIGN_ORIENTATION[name] * torch.tensor(scores, dtype=torch.float64)
        utility = torch.tensor(utilities, dtype=torch.float64)
        pooled[name] = {
            "spearman": spearman(oriented, utility),
            "sign_auc": sign_auc(oriented, utility),
            "direction_agrees": all(
                bank["metrics"][name]["spearman"] > 0 for bank in banks
            ),
        }
        pooled[name]["gate_passed"] = (
            pooled[name]["direction_agrees"]
            and pooled[name]["sign_auc"] is not None
            and pooled[name]["sign_auc"] >= .60
        )
    survivors = [name for name, item in pooled.items() if item["gate_passed"]]
    preferred = None
    if survivors:
        current = [name for name in survivors if name.startswith("current_")]
        pool = current or survivors
        preferred = max(pool, key=lambda name: pooled[name]["sign_auc"])
    result = {
        "inputs": [str(path) for path in args.inputs],
        "oracle_is_deployable": False,
        "compare_time": args.compare_time,
        "signal_orientations": SIGN_ORIENTATION,
        "banks": banks,
        "pooled": pooled,
        "survivors": survivors,
        "preferred_signal": preferred,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "pooled": pooled,
        "survivors": survivors,
        "preferred_signal": preferred,
    }, indent=2))
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
