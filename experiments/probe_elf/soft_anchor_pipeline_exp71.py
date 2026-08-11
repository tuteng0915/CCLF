#!/usr/bin/env python3
"""EXP-71: synchronized, revisable soft-anchor Pipeline screen.

All positions retain one global ODE time.  A same-time first pass refreshes the
self-conditioning state for a growing leader set; a second pass supplies the
velocity used by the solver.  No latent is frozen or discretized.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import unified_method_eval_exp64 as common
from modules.model import ELF_B
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


CHECKPOINTS = {
    name: common.CHECKPOINTS[name]
    for name in ("baseline", "ct_control", "kd_early")
}
ARMS = (
    "standard32",
    "standard64",
    "two_forward_none",
    "two_forward_all",
    "soft_ltr",
    "soft_rtl",
    "soft_random",
    "soft_confidence",
    "soft_shuffled",
)
OUT_DIR = Path("results/exp71_soft_anchor_pipeline")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seq", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--wave_start", type=float, default=0.15)
    parser.add_argument("--wave_end", type=float, default=0.75)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--label", default="screen")
    return parser.parse_args()


def empty_condition(z0):
    batch, length, _ = z0.shape
    return (
        torch.zeros_like(z0),
        torch.zeros(batch, length, dtype=z0.dtype, device=z0.device),
    )


def leader_count(progress, length, start, end):
    fraction = min(max((progress - start) / max(end - start, 1e-8), 0.0), 1.0)
    return int(fraction * length)


def position_mask(
    kind,
    batch,
    length,
    count,
    device,
    random_rank=None,
    eligible_mask=None,
):
    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    if count <= 0:
        return mask
    if eligible_mask is None:
        eligible_mask = torch.ones(length, dtype=torch.bool, device=device)
    eligible_positions = torch.arange(length, device=device)[eligible_mask]
    if kind == "rtl":
        eligible_positions = eligible_positions.flip(0)
    elif kind == "random":
        eligible_positions = eligible_positions[
            random_rank[eligible_positions].argsort()
        ]
    elif kind != "ltr":
        raise ValueError(kind)
    chosen = eligible_positions[:count]
    mask[:, chosen] = True
    return mask


@torch.no_grad()
def lexical_confidence(x_pred, model):
    batch = x_pred.shape[0]
    model_input = torch.cat([x_pred, torch.zeros_like(x_pred)], dim=-1)
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
    return F.softmax(logits.float(), dim=-1).amax(-1)


def confidence_mask(confidence, count, eligible_mask=None):
    batch, length = confidence.shape
    if count <= 0:
        return torch.zeros_like(confidence, dtype=torch.bool)
    if count >= length:
        return torch.ones_like(confidence, dtype=torch.bool)
    scores = confidence
    if eligible_mask is not None:
        scores = confidence.masked_fill(~eligible_mask[None, :], float("-inf"))
    indices = scores.topk(count, dim=1).indices
    mask = torch.zeros_like(confidence, dtype=torch.bool)
    return mask.scatter(1, indices, True)


@torch.no_grad()
def two_forward_ode(
    z0, model, t_steps, sccfg, arm, args, cond_seq=None, cond_mask=None
):
    cfg = common.SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = empty_condition(z0)
    if not torch.equal(cond_mask, cond_mask[:1].expand_as(cond_mask)):
        raise ValueError("all examples must share the same condition mask")
    eligible_mask = cond_mask[0] < 0.5
    eligible_count = int(eligible_mask.sum().item())
    z = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    length = z.shape[1]
    generator = torch.Generator(device=z.device).manual_seed(args.seed)
    permutation = torch.randperm(length, generator=generator, device=z.device)
    random_rank = torch.empty_like(permutation)
    random_rank[permutation] = torch.arange(length, device=z.device)
    model_calls = 0
    readout_calls = 0
    leader_fractions = []

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for index in range(t_steps.shape[0] - 1):
            t = t_steps[index].item()
            t_next = t_steps[index + 1].item()
            _, current_pred = _ode_step(
                model=model,
                z=z,
                t=t,
                t_next=t,
                x_pred_prev=x_pred,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            model_calls += 1

            progress = index / max(t_steps.shape[0] - 2, 1)
            count = leader_count(
                progress, eligible_count, args.wave_start, args.wave_end
            )
            if arm == "two_forward_none":
                leaders = torch.zeros(
                    z.shape[:2], dtype=torch.bool, device=z.device
                )
            elif arm == "two_forward_all":
                leaders = eligible_mask[None, :].expand(z.shape[0], -1)
            elif arm in ("soft_ltr", "soft_shuffled"):
                leaders = position_mask(
                    "ltr",
                    z.shape[0],
                    length,
                    count,
                    z.device,
                    eligible_mask=eligible_mask,
                )
            elif arm == "soft_rtl":
                leaders = position_mask(
                    "rtl",
                    z.shape[0],
                    length,
                    count,
                    z.device,
                    eligible_mask=eligible_mask,
                )
            elif arm == "soft_random":
                leaders = position_mask(
                    "random",
                    z.shape[0],
                    length,
                    count,
                    z.device,
                    random_rank,
                    eligible_mask,
                )
            elif arm == "soft_confidence":
                leaders = confidence_mask(
                    lexical_confidence(current_pred, model),
                    count,
                    eligible_mask,
                )
                readout_calls += 1
            else:
                raise ValueError(arm)

            fresh = current_pred
            if arm == "soft_shuffled":
                if z.shape[0] > 1:
                    fresh = torch.roll(current_pred, shifts=1, dims=0)
                else:
                    # Batch-size-one smoke: use a deterministic position roll,
                    # while formal runs use cross-sequence shuffling.
                    fresh = torch.roll(current_pred, shifts=max(length // 4, 1), dims=1)
            sc_state = torch.where(leaders.unsqueeze(-1), fresh, x_pred)
            z, x_pred = _ode_step(
                model=model,
                z=z,
                t=t,
                t_next=t_next,
                x_pred_prev=sc_state,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            z = restore_cond(z, cond_seq, cond_mask)
            x_pred = restore_cond(x_pred, cond_seq, cond_mask)
            model_calls += 1
            leader_fractions.append(
                float(leaders[:, eligible_mask].float().mean().item())
            )
    return z, x_pred, {
        "denoiser_calls": model_calls,
        "lexical_readout_calls": readout_calls,
        "mean_leader_fraction": sum(leader_fractions) / len(leader_fractions),
    }


def run_arm(name, z0, model, args):
    if name in ("standard32", "standard64"):
        steps = 32 if name == "standard32" else 64
        grid = get_sampling_steps(steps, time_schedule="uniform", device=z0.device)
        z, x_pred = common.standard_ode(z0, model, grid, args.sccfg)
        return z, x_pred, {
            "denoiser_calls": steps,
            "lexical_readout_calls": 0,
            "mean_leader_fraction": None,
        }
    grid = get_sampling_steps(args.n_steps, time_schedule="uniform", device=z0.device)
    return two_forward_ode(z0, model, grid, args.sccfg, name, args)


def main():
    args = parse_args()
    if not 0 <= args.wave_start < args.wave_end <= 1:
        raise ValueError("wave_start/end must satisfy 0 <= start < end <= 1")
    device = torch.device(args.device)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(args.max_length))
    missing, unexpected = model.load_state_dict(common.load_weights(checkpoint), strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device).eval()

    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    ppl_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()
    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    z0 = args.noise_scale * torch.randn(
        args.n_seq,
        args.max_length,
        common.model_config(args.max_length)["text_encoder_dim"],
        generator=generator,
        device=device,
    )

    results = {}
    for arm in args.arms:
        texts = []
        call_info = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        for start in range(0, args.n_seq, args.batch_size):
            z, _, info = run_arm(arm, z0[start:start + args.batch_size], model, args)
            if call_info is None:
                call_info = info
            ids = common.decode(z, model, device)
            texts.extend(common.decode_texts(ids.cpu(), elf_tokenizer))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        metrics = common.text_metrics(
            texts, ppl_model, ppl_tokenizer, device, max_length=args.max_length
        )
        metrics.update(call_info)
        metrics.update(
            {
                "wall_seconds": elapsed,
                "seconds_per_sequence": elapsed / args.n_seq,
                "samples": texts[:4],
                "texts": texts,
            }
        )
        results[arm] = metrics
        print(
            f"{arm:<20} PPL={metrics['ppl']:.1f} D1={metrics['d1']:.3f} "
            f"D2={metrics['d2']:.3f} deg={metrics['degeneration_rate']:.3f} "
            f"calls={metrics['denoiser_calls']}+{metrics['lexical_readout_calls']} "
            f"sec={elapsed:.1f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "seed": args.seed,
                "n_seq": args.n_seq,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "n_steps": args.n_steps,
                "noise_scale": args.noise_scale,
                "sccfg": args.sccfg,
                "wave_start": args.wave_start,
                "wave_end": args.wave_end,
                "arms": args.arms,
                "results": results,
            },
            handle,
            indent=2,
        )
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
