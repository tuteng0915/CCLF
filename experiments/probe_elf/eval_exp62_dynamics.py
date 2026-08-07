#!/usr/bin/env python3
"""True-rollout commitment fingerprint for an EXP-62 checkpoint.

The endpoint is each trajectory's own final decoded token.  At every native
ODE step we decode the current predicted-clean state and measure when each
position first matches that endpoint, when it matches permanently, and how
often its top-1 proposal changes.  All checkpoints use the same initial-noise
bank, so the resulting arrays support paired checkpoint comparisons.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import _ode_step, get_sampling_steps

from eval_wff_pilot import MAX_LENGTH, MODEL_CFG as WFF_MODEL_CFG, _Cfg

OUT_DIR = Path("results/exp62_checkpoint_panel/dynamics")

MODEL_CFG = dict(WFF_MODEL_CFG)
MODEL_CFG["per_token_time_conditioning"] = False


@torch.no_grad()
def decode_predicted_clean(x_pred, model, device):
    batch_size = x_pred.shape[0]
    z_input = torch.cat([x_pred, torch.zeros_like(x_pred)], dim=-1)
    t_final = torch.ones(batch_size, dtype=x_pred.dtype, device=device)
    sc = torch.ones(batch_size, dtype=x_pred.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(
            z_input,
            t_final,
            deterministic=True,
            self_cond_cfg_scale=sc,
            decoder_step_active=True,
        )
    top2 = logits.float().topk(2, dim=-1)
    return top2.indices[..., 0], top2.values[..., 0] - top2.values[..., 1]


@torch.no_grad()
def run_trajectory(z0, model, t_steps, device, sccfg):
    z = z0.clone()
    x_pred = torch.zeros_like(z0)
    predictions = []
    margins = []
    scored_times = []
    cfg = _Cfg()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for index in range(t_steps.shape[0] - 1):
            z, x_pred = _ode_step(
                z=z,
                t=t_steps[index].item(),
                t_next=t_steps[index + 1].item(),
                x_pred_prev=x_pred,
                model=model,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=None,
                cond_seq_mask=None,
            )
            pred, margin = decode_predicted_clean(x_pred, model, device)
            predictions.append(pred.cpu())
            margins.append(margin.cpu())
            scored_times.append(float(t_steps[index + 1].item()))

    return torch.stack(predictions), torch.stack(margins), scored_times


def timing_arrays(predictions):
    """Return first endpoint hit, permanent endpoint hit, and revisions."""
    n_steps, batch_size, length = predictions.shape
    endpoint = predictions[-1]
    matches = predictions.eq(endpoint.unsqueeze(0))

    first = torch.full((batch_size, length), n_steps - 1, dtype=torch.long)
    for step in range(n_steps):
        mask = matches[step] & first.eq(n_steps - 1)
        first[mask] = step

    # A position is stable at step k iff every prediction from k onward equals
    # the final endpoint.  Reverse cumulative AND implements this exactly.
    suffix_all = torch.ones_like(matches)
    suffix_all[-1] = matches[-1]
    for step in range(n_steps - 2, -1, -1):
        suffix_all[step] = matches[step] & suffix_all[step + 1]
    stable = torch.full((batch_size, length), n_steps - 1, dtype=torch.long)
    for step in range(n_steps):
        mask = suffix_all[step] & stable.eq(n_steps - 1)
        stable[mask] = step

    revisions = predictions[1:].ne(predictions[:-1]).sum(dim=0)
    return endpoint, first, stable, revisions, matches


def quantiles(values):
    values = values.float().reshape(-1)
    q = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75]))
    return {
        "mean": float(values.mean()),
        "q25": float(q[0]),
        "median": float(q[1]),
        "q75": float(q[2]),
    }


def summarize(predictions, margins, scored_times):
    endpoint, first, stable, revisions, matches = timing_arrays(predictions)
    denom = max(len(scored_times) - 1, 1)
    first_progress = first.float() / denom
    stable_progress = stable.float() / denom

    thresholds = {}
    for progress in (0.25, 0.50, 0.75):
        thresholds[f"stable_by_{progress:.2f}"] = float(
            stable_progress.le(progress).float().mean()
        )

    flat_revision = revisions.reshape(-1)
    result = {
        "n_steps": len(scored_times),
        "n_positions": int(first.numel()),
        "scored_times": scored_times,
        "first_progress": quantiles(first_progress),
        "stable_progress": quantiles(stable_progress),
        "stable_minus_first": quantiles(stable_progress - first_progress),
        "mean_revisions": float(flat_revision.float().mean()),
        "frac_any_revision": float(flat_revision.gt(0).float().mean()),
        "frac_three_plus_revisions": float(flat_revision.ge(3).float().mean()),
        "endpoint_agreement_curve": matches.reshape(matches.shape[0], -1).float().mean(1).tolist(),
        "final_margin": quantiles(margins[-1]),
        **thresholds,
    }
    arrays = {
        "endpoint": endpoint,
        "first_step": first,
        "stable_step": stable,
        "revisions": revisions,
        "scored_times": torch.tensor(scored_times),
    }
    return result, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seq", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--sccfg", type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights_key = "ema_params1" if "ema_params1" in checkpoint else "params"
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(checkpoint[weights_key], strict=True)
    model.eval().to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    all_z0 = torch.randn(
        args.n_seq, MAX_LENGTH, 512, generator=generator, device=device
    ) * _Cfg.denoiser_noise_scale
    t_steps = get_sampling_steps(args.n_steps, time_schedule="uniform", device=device)

    predictions = []
    margins = []
    scored_times = None
    for start in range(0, args.n_seq, args.batch_size):
        pred, margin, batch_times = run_trajectory(
            all_z0[start:start + args.batch_size], model, t_steps, device, args.sccfg
        )
        predictions.append(pred)
        margins.append(margin)
        scored_times = batch_times
        print(f"{args.label}: {min(start + args.batch_size, args.n_seq)}/{args.n_seq}", flush=True)

    all_predictions = torch.cat(predictions, dim=1)
    all_margins = torch.cat(margins, dim=1)
    result, arrays = summarize(all_predictions, all_margins, scored_times)

    print(
        f"{args.label}: first={result['first_progress']['mean']:.3f} "
        f"stable={result['stable_progress']['mean']:.3f} "
        f"revisions={result['mean_revisions']:.2f} "
        f"stable@50={result['stable_by_0.50']:.3f}"
    )

    metadata = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "weights": weights_key,
        "seed": args.seed,
        "n_seq": args.n_seq,
        "length": MAX_LENGTH,
        "ode_steps": args.n_steps,
        "noise_scale": _Cfg.denoiser_noise_scale,
        "sccfg": args.sccfg,
        "result": result,
    }
    json_path = OUT_DIR / f"{args.label}_seed{args.seed}.json"
    arrays_path = OUT_DIR / f"{args.label}_seed{args.seed}_arrays.pt"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    torch.save(arrays, arrays_path)
    print(f"Saved -> {json_path} and {arrays_path}")


if __name__ == "__main__":
    main()
