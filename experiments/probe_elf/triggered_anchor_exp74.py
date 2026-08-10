#!/usr/bin/env python3
"""EXP-74: sparse, reversible self-conditioning anchors.

Unlike EXP-71, this sampler makes only one denoiser call per ODE interval.
Selected predicted-clean vectors are retained in self-conditioning memory for
a short horizon while every latent remains continuously updated.
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
    "trigger_t30",
    "trigger_t40",
    "trigger_two",
    "trigger_stable",
    "trigger_shuffled",
    "hard_t30",
    "hard_persistent",
    "hard_highconf",
    "hard_stable",
)
OUT_DIR = Path("results/exp74_triggered_anchor")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seq", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--label", default="screen")
    return parser.parse_args()


def empty_condition(z0):
    batch, length, _ = z0.shape
    return (
        torch.zeros_like(z0),
        torch.zeros(batch, length, dtype=z0.dtype, device=z0.device),
    )


@torch.no_grad()
def lexical_readout(x_pred, model):
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
    probabilities = F.softmax(logits.float(), dim=-1)
    confidence, token_ids = probabilities.max(-1)
    return token_ids, confidence


def nearest_index(grid, target):
    return int(torch.argmin((grid - target).abs()).item())


@torch.no_grad()
def triggered_ode(z0, model, grid, sccfg, arm, args):
    cfg = common.SamplingConfig()
    cond_seq, cond_mask = empty_condition(z0)
    z = z0.clone()
    x_pred = torch.zeros_like(z)
    anchor_value = torch.zeros_like(z)
    anchor_mask = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)
    expiry = torch.zeros(z.shape[:2], dtype=torch.long, device=z.device)
    index30 = nearest_index(grid, 0.30)
    index40 = nearest_index(grid, 0.40)
    previous_ids = None
    readout_calls = 0
    trigger_records = []

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for index in range(grid.shape[0] - 1):
            t = grid[index].item()
            t_next = grid[index + 1].item()
            trigger_now = (
                (arm == "trigger_t30" and index == index30)
                or (arm == "hard_t30" and index == index30)
                or (arm in ("trigger_t40", "hard_persistent", "hard_highconf") and index == index40)
                or (arm in ("trigger_two", "trigger_shuffled") and index in (index30, index40))
                or (arm in ("trigger_stable", "hard_stable") and index in (index30, index40))
            )
            if trigger_now:
                token_ids, confidence = lexical_readout(x_pred, model)
                readout_calls += 1
                selected = confidence >= args.confidence
                if arm in ("trigger_stable", "hard_stable"):
                    if index == index30:
                        previous_ids = token_ids
                        selected = torch.zeros_like(selected)
                    else:
                        selected = selected & (token_ids == previous_ids)
                if arm == "hard_highconf":
                    selected = confidence >= max(args.confidence, 0.90)
                if selected.any():
                    fresh = x_pred
                    if arm == "trigger_shuffled":
                        fresh = (
                            torch.roll(x_pred, shifts=1, dims=0)
                            if x_pred.shape[0] > 1
                            else torch.roll(x_pred, shifts=max(x_pred.shape[1] // 4, 1), dims=1)
                        )
                    anchor_value = torch.where(selected.unsqueeze(-1), fresh, anchor_value)
                    anchor_mask = anchor_mask | selected
                    if arm.startswith("hard_") or arm == "hard_persistent":
                        expiry = torch.where(
                            selected, torch.full_like(expiry, grid.shape[0] + 1), expiry
                        )
                    else:
                        expiry = torch.where(
                            selected, torch.full_like(expiry, index + args.horizon), expiry
                        )
                trigger_records.append(
                    {
                        "index": index,
                        "time": t,
                        "selected_fraction": float(selected.float().mean()),
                        "mean_confidence": float(confidence.mean()),
                    }
                )

            active = anchor_mask & (index < expiry)
            if arm.startswith("hard_") or arm == "hard_persistent":
                active_cond = active.to(z.dtype)
                z = restore_cond(z, anchor_value, active_cond)
                x_pred = restore_cond(x_pred, anchor_value, active_cond)
                z, x_pred = _ode_step(
                    model=model,
                    z=z,
                    t=t,
                    t_next=t_next,
                    x_pred_prev=x_pred,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=sccfg,
                    cond_seq=anchor_value,
                    cond_seq_mask=active_cond,
                )
            else:
                anchored_memory = (1.0 - args.alpha) * x_pred + args.alpha * anchor_value
                sc_state = torch.where(active.unsqueeze(-1), anchored_memory, x_pred)
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
            anchor_mask = anchor_mask & (index + 1 < expiry)

    return z, x_pred, {
        "denoiser_calls": grid.shape[0] - 1,
        "lexical_readout_calls": readout_calls,
        "triggers": trigger_records,
    }


def run_arm(name, z0, model, args):
    if name in ("standard32", "standard64"):
        steps = 32 if name == "standard32" else 64
        grid = get_sampling_steps(steps, time_schedule="uniform", device=z0.device)
        z, x_pred = common.standard_ode(z0, model, grid, args.sccfg)
        return z, x_pred, {
            "denoiser_calls": steps,
            "lexical_readout_calls": 0,
            "triggers": [],
        }
    grid = get_sampling_steps(args.n_steps, time_schedule="uniform", device=z0.device)
    return triggered_ode(z0, model, grid, args.sccfg, name, args)


def main():
    args = parse_args()
    if args.horizon <= 0 or not 0.0 <= args.alpha <= 1.0:
        raise ValueError("horizon must be positive and alpha must be in [0,1]")
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
        texts, trigger_records = [], []
        call_info = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for start in range(0, args.n_seq, args.batch_size):
            z, _, info = run_arm(arm, z0[start:start + args.batch_size], model, args)
            if call_info is None:
                call_info = {k: info[k] for k in ("denoiser_calls", "lexical_readout_calls")}
            trigger_records.extend(info["triggers"])
            ids = common.decode(z, model, device)
            texts.extend(common.decode_texts(ids.cpu(), elf_tokenizer))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        metrics = common.text_metrics(
            texts, ppl_model, ppl_tokenizer, device, max_length=args.max_length
        )
        metrics.update(call_info)
        metrics.update(
            {
                "wall_seconds": elapsed,
                "seconds_per_sequence": elapsed / args.n_seq,
                "trigger_records": trigger_records,
                "samples": texts[:4],
                "texts": texts,
            }
        )
        results[arm] = metrics
        print(
            f"{arm:<20} PPL={metrics['ppl']:.1f} D1={metrics['d1']:.3f} "
            f"D2={metrics['d2']:.3f} deg={metrics['degeneration_rate']:.3f} "
            f"calls={metrics['denoiser_calls']}+{metrics['lexical_readout_calls']}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({**vars(args), "results": results}, handle, indent=2)
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
