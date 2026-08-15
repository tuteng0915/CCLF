#!/usr/bin/env python3
"""EXP-111: overlapping token-level Unlock waves on deterministic ELF ODE."""

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import paired_conditional_revalidation_exp80 as exp80  # noqa: E402
import robust_revisable_commit_exp78 as exp78  # noqa: E402
import unlock_trigger_headroom_exp108 as exp108  # noqa: E402
import unified_method_eval_exp64 as common  # noqa: E402
from modules.model import ELF_B  # noqa: E402
from modules.t5_encoder import get_encoder  # noqa: E402
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond  # noqa: E402


ARMS = (
    "standard",
    "fixed40",
    "fixed45",
    "fixed40_sham45",
    "two_wave_new",
    "two_wave_refresh",
)
OUT_DIR = Path("results/exp111_two_wave_unlock")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--owt_offset", type=int, required=True)
    parser.add_argument("--n_cond", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--high_confidence", type=float, default=.90)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def event_for_arm(arm, event):
    if event == 1:
        return arm in ("fixed40", "fixed40_sham45", "two_wave_new", "two_wave_refresh")
    return arm in ("fixed45", "two_wave_new", "two_wave_refresh")


@torch.no_grad()
def rollout(z0, model, grid, args, arm, cond_seq, cond_mask):
    cfg = common.SamplingConfig()
    base_seq, base_mask = cond_seq.clone(), cond_mask.clone()
    active_seq, active_mask = cond_seq.clone(), cond_mask.clone()
    z = restore_cond(z0.clone(), active_seq, active_mask)
    x_pred = restore_cond(torch.zeros_like(z), active_seq, active_mask)
    batch, length = base_mask.shape
    expiry = torch.full((batch, length), -1, dtype=torch.long, device=z.device)
    ever_selected = torch.zeros((batch, length), dtype=torch.bool, device=z.device)
    anchor_ids = torch.full((batch, length), -1, dtype=torch.long, device=z.device)
    wave_counts = [torch.zeros(batch, device=z.device), torch.zeros(batch, device=z.device)]
    overlap_counts = torch.zeros(batch, device=z.device)
    readout_calls = 0
    fired = [False, False]
    actual_events = [None, None]
    free_count = (base_mask < .5).sum(dim=1).clamp_min(1)

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for index in range(grid.shape[0] - 1):
            expired = (expiry >= 0) & (expiry <= index)
            if expired.any():
                still_active = (expiry >= 0) & ~expired
                if still_active.any():
                    active_seq = torch.where(expired.unsqueeze(-1), base_seq, active_seq)
                    active_mask = torch.where(expired, base_mask, active_mask)
                else:
                    # Match EXP-78's exact numerical path when one complete
                    # wave expires at once.
                    active_seq = base_seq.clone()
                    active_mask = base_mask.clone()
                expiry[expired] = -1

            z, x_pred = _ode_step(
                model=model,
                z=z,
                t=grid[index].item(),
                t_next=grid[index + 1].item(),
                x_pred_prev=x_pred,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=args.sccfg,
                cond_seq=active_seq,
                cond_seq_mask=active_mask,
            )
            t_next = float(grid[index + 1])

            for event, requested in enumerate((.40, .45)):
                if fired[event] or t_next + 1e-9 < requested:
                    continue
                fired[event] = True
                actual_events[event] = t_next
                sham = arm == "fixed40_sham45" and event == 1
                if not event_for_arm(arm, event) and not sham:
                    continue
                token_ids, confidence = exp78.lexical_readout(x_pred, model)
                readout_calls += 1
                if sham:
                    continue
                selected = (confidence >= args.high_confidence) & (base_mask < .5)
                if arm == "two_wave_new" and event == 1:
                    selected &= ~ever_selected
                overlap = selected & ever_selected
                active_seq = torch.where(selected.unsqueeze(-1), x_pred.detach(), active_seq)
                active_mask = torch.maximum(active_mask, selected.to(active_mask.dtype))
                z = restore_cond(z, active_seq, active_mask)
                x_pred = restore_cond(x_pred, active_seq, active_mask)
                overlap_counts += overlap.sum(dim=1)
                wave_counts[event] += selected.sum(dim=1)
                expiry[selected] = index + 1 + 4
                anchor_ids[selected] = token_ids[selected]
                ever_selected |= selected

    return z, {
        "wave1_fraction": (wave_counts[0] / free_count).cpu().tolist(),
        "wave2_fraction": (wave_counts[1] / free_count).cpu().tolist(),
        "overlap_fraction": (overlap_counts / free_count).cpu().tolist(),
        "ever_selected": ever_selected.cpu(),
        "anchor_ids": anchor_ids.cpu(),
        "readout_calls": readout_calls,
        "actual_events": actual_events,
    }


@torch.no_grad()
def generate(model, tokenizer, z0, cond_seq, cond_mask, args, grid):
    records = {
        arm: {
            "texts": [], "wave1": [], "wave2": [], "overlap": [],
            "revision_numerator": 0, "revision_denominator": 0, "calls": [],
        }
        for arm in ARMS
    }
    native_reference_agreement = None
    actual_events = None
    for batch_index, start in enumerate(range(0, args.n_cond, args.batch_size)):
        end = min(start + args.batch_size, args.n_cond)
        batch_z0 = z0[start:end]
        batch_seq = cond_seq[start:end]
        batch_mask = cond_mask[start:end]
        for arm in ARMS:
            z, info = rollout(
                batch_z0, model, grid, args, arm, batch_seq, batch_mask
            )
            ids = common.decode(z, model, z.device)
            records[arm]["texts"].extend(
                common.decode_texts(ids.cpu(), tokenizer, args.prefix_length)
            )
            records[arm]["wave1"].extend(info["wave1_fraction"])
            records[arm]["wave2"].extend(info["wave2_fraction"])
            records[arm]["overlap"].extend(info["overlap_fraction"])
            records[arm]["calls"].append(info["readout_calls"])
            selected = info["ever_selected"]
            if selected.any():
                records[arm]["revision_numerator"] += int(
                    (ids.cpu()[selected] != info["anchor_ids"][selected]).sum()
                )
                records[arm]["revision_denominator"] += int(selected.sum())
            if actual_events is None and arm == "two_wave_new":
                actual_events = info["actual_events"]

            if batch_index == 0 and arm == "fixed40":
                args.sampler = "ode"
                args.commit_time = .40
                args.read_time = .30
                args.stable_confidence = .60
                reference_z, _ = exp78.rollout(
                    batch_z0, model, grid, args, "unlock4", 0,
                    batch_seq, batch_mask,
                )
                reference_ids = common.decode(reference_z, model, z.device)
                native_reference_agreement = float((reference_ids == ids).float().mean())
                if native_reference_agreement != 1.0:
                    raise RuntimeError(
                        f"fixed40 native reference agreement={native_reference_agreement}"
                    )
        print(f"generation completed {end}/{args.n_cond}", flush=True)
    return records, native_reference_agreement, actual_events


def mean(values):
    return sum(values) / max(len(values), 1)


def main():
    args = parse_args()
    if args.n_steps != 32:
        raise ValueError("EXP-111 freezes uniform ODE-32")
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    checkpoint_path = REPO_ROOT / exp78.CHECKPOINTS["baseline"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(args.max_length))
    model.load_state_dict(common.load_weights(checkpoint), strict=False)
    model.to(device).eval()

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    args.conditional_dataset = "owt"
    pairs = exp80.load_pairs(args, tokenizer)
    cond_seq, cond_mask, references = common.build_condition_data(
        pairs, tokenizer, encoder, device, args.max_length, args.prefix_length
    )
    prefix_ids = exp80.prefix_targets(pairs, args.prefix_length, device)
    prompts = common.decode_texts(prefix_ids.cpu(), tokenizer)
    shuffled_prompts = prompts[1:] + prompts[:1]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z0 = args.noise_scale * torch.randn(
        args.n_cond,
        args.max_length,
        common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator,
        device=device,
    )
    z0[:, : args.prefix_length] = cond_seq[:, : args.prefix_length].to(z0.dtype)
    grid = get_sampling_steps(args.n_steps, "uniform", device=device)
    records, native_agreement, actual_events = generate(
        model, tokenizer, z0, cond_seq, cond_mask, args, grid
    )

    model.cpu(); encoder.cpu(); del model, encoder, checkpoint
    torch.cuda.empty_cache()
    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    evaluator = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()
    summaries, nlls, counts = {}, {}, {}
    for arm, record in records.items():
        print(f"evaluating {arm}", flush=True)
        summaries[arm], nlls[arm], counts[arm], _ = exp108.summarize(
            record["texts"], references, prompts, shuffled_prompts,
            evaluator, ppl_tokenizer, device,
        )
        summaries[arm].update({
            "wave1_fraction": mean(record["wave1"]),
            "wave2_fraction": mean(record["wave2"]),
            "overlap_fraction": mean(record["overlap"]),
            "revision_rate": (
                record["revision_numerator"] / max(record["revision_denominator"], 1)
            ),
            "mean_readout_calls": mean(record["calls"]),
            "denoiser_calls": args.n_steps,
        })

    sham_agreement = sum(
        a == b for a, b in zip(records["fixed40"]["texts"], records["fixed40_sham45"]["texts"])
    ) / args.n_cond
    comparisons = {}
    fixed = summaries["fixed40"]
    for arm in ARMS:
        if arm == "fixed40":
            continue
        delta = nlls[arm] - nlls["fixed40"]
        interval = exp108.bootstrap(delta, args.bootstrap_samples, args.seed + len(arm))
        quality_delta = exp108.quality_delta(summaries[arm], fixed)
        improvement = 100 * (
            fixed["prompt_conditioned_ppl"] - summaries[arm]["prompt_conditioned_ppl"]
        ) / fixed["prompt_conditioned_ppl"]
        comparisons[arm] = {
            "improvement_pct": improvement,
            "mean_delta_nats": float(delta.mean()),
            "mean_delta_ci95": interval,
            "quality_delta": quality_delta,
            "gate_passed": (
                arm in ("two_wave_new", "two_wave_refresh")
                and improvement >= 2.0
                and interval[1] < 0.0
                and exp108.quality_gate(quality_delta)
                and summaries[arm]["wave2_fraction"] >= .05
                and sham_agreement == 1.0
            ),
        }

    payload = {
        **vars(args),
        "checkpoint": "baseline",
        "checkpoint_path": str(checkpoint_path),
        "actual_event_times": actual_events,
        "native_fixed40_reference_agreement": native_agreement,
        "fixed40_sham_text_agreement": sham_agreement,
        "summaries": summaries,
        "comparisons_vs_fixed40": comparisons,
        "per_sequence": {
            "nll": {arm: nlls[arm].tolist() for arm in ARMS},
            "counts": {arm: counts[arm].tolist() for arm in ARMS},
        },
        "texts": {arm: records[arm]["texts"] for arm in ARMS},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_seed{args.seed}.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "sham_agreement": sham_agreement,
        "summaries": {
            arm: {
                "c_ppl": summaries[arm]["prompt_conditioned_ppl"],
                "d1": summaries[arm]["d1"],
                "rep4": summaries[arm]["rep4"],
                "degeneration": summaries[arm]["degeneration_rate"],
                "prompt_gain": summaries[arm]["prompt_gain_nats"],
                "wave2_fraction": summaries[arm]["wave2_fraction"],
            }
            for arm in ARMS
        },
        "comparisons_vs_fixed40": comparisons,
    }, indent=2), flush=True)
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
