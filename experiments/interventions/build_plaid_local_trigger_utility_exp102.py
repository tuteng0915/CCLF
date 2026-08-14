#!/usr/bin/env python3
"""EXP-102: build paired native short-horizon trigger-utility signals."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GLOBAL_DIR = ROOT / "experiments" / "global_state"
for path in (HERE, GLOBAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_plaid_subset_headroom_exp99 as exp99  # noqa: E402
import eval_temporary_anchor_portability_exp90 as exp90  # noqa: E402
from common import load_adapter  # noqa: E402


SIGNAL_NAMES = (
    "confidence_gain",
    "entropy_reduction",
    "margin_gain",
    "xhat_control_cosine_distance",
    "lexical_disagreement",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp101_bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lookaheads", default="0,1,2,4")
    parser.add_argument("--probe_replicates", type=int, default=1)
    parser.add_argument("--probe_seed_base", type=int, default=29)
    return parser.parse_args()


def readout(adapter, z, sc, t, batch_size):
    out = adapter.forward_state(z, sc, t, batch_size=batch_size)
    logits = out["logits"].float().to(adapter.device)
    probabilities = torch.softmax(logits, dim=-1)
    confidence, top1 = probabilities.max(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    top2 = probabilities.topk(k=2, dim=-1).values
    return {
        "logits": logits,
        "xhat": out["predicted_clean"].float().to(adapter.device),
        "confidence": confidence,
        "entropy": entropy,
        "margin": top2[..., 0] - top2[..., 1],
        "top1": top1,
    }


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def compare(anchor, control, unresolved):
    cosine_distance = 1.0 - F.cosine_similarity(
        anchor["xhat"], control["xhat"], dim=-1
    )
    return torch.stack(
        (
            masked_mean(anchor["confidence"] - control["confidence"], unresolved),
            masked_mean(control["entropy"] - anchor["entropy"], unresolved),
            masked_mean(anchor["margin"] - control["margin"], unresolved),
            masked_mean(cosine_distance, unresolved),
            masked_mean((anchor["top1"] != control["top1"]).float(), unresolved),
        ),
        dim=-1,
    )


@torch.no_grad()
def replay_trigger(
    adapter,
    payload,
    grid,
    eps,
    prompt_clean,
    batch_index,
    trigger,
    lookaheads,
    probe_replicates=1,
    probe_seed_base=29,
):
    prefix = payload["prefix_length"]
    z = eps.clone().to(adapter.device)
    sc = torch.zeros_like(z)
    z[:, :prefix] = prompt_clean
    sc[:, :prefix] = prompt_clean
    for step in range(trigger):
        z, sc = exp90.native_step(
            adapter,
            z,
            sc,
            grid[step],
            grid[step + 1],
            exp90.step_seed(payload["seed"], 29, batch_index, step),
        )
        z[:, :prefix] = prompt_clean
        sc[:, :prefix] = prompt_clean

    trigger_out = adapter.forward_state(
        z, sc, grid[trigger], batch_size=payload["batch_size"]
    )
    eligible = torch.ones(z.shape[:2], dtype=torch.bool, device=adapter.device)
    eligible[:, :prefix] = False
    anchor_mask, _, anchor_clean, _ = exp90.build_anchor(
        adapter,
        trigger_out["logits"].to(adapter.device),
        trigger_out["predicted_clean"].to(adapter.device),
        "top_confidence",
        eligible,
        payload["density"],
        seed=0,
    )
    unresolved = eligible & ~anchor_mask

    replicate_signals = []
    for replicate in range(probe_replicates):
        route = probe_seed_base + replicate
        control_z, control_sc = z.clone(), sc.clone()
        anchor_z, anchor_sc = z.clone(), sc.clone()
        anchor_z[anchor_mask] = anchor_clean[anchor_mask]
        anchor_sc[anchor_mask] = anchor_clean[anchor_mask]
        seed = exp90.step_seed(payload["seed"], route, batch_index, trigger)
        control_z, control_sc = exp90.native_step(
            adapter, control_z, control_sc, grid[trigger], grid[trigger + 1], seed
        )
        anchor_z, anchor_sc = exp90.native_step(
            adapter, anchor_z, anchor_sc, grid[trigger], grid[trigger + 1], seed
        )
        anchor_z[anchor_mask] = anchor_clean[anchor_mask]
        anchor_sc[anchor_mask] = anchor_clean[anchor_mask]
        for state in (control_z, control_sc, anchor_z, anchor_sc):
            state[:, :prefix] = prompt_clean

        signal_by_lookahead = {}
        max_lookahead = max(lookaheads)
        for extra in range(max_lookahead + 1):
            if extra in lookaheads:
                time_index = trigger + 1 + extra
                anchor_out = readout(
                    adapter,
                    anchor_z,
                    anchor_sc,
                    grid[time_index],
                    payload["batch_size"],
                )
                control_out = readout(
                    adapter,
                    control_z,
                    control_sc,
                    grid[time_index],
                    payload["batch_size"],
                )
                signal_by_lookahead[extra] = compare(
                    anchor_out, control_out, unresolved
                ).cpu()
            if extra < max_lookahead:
                step = trigger + 1 + extra
                seed = exp90.step_seed(payload["seed"], route, batch_index, step)
                control_z, control_sc = exp90.native_step(
                    adapter, control_z, control_sc, grid[step], grid[step + 1], seed
                )
                anchor_z, anchor_sc = exp90.native_step(
                    adapter, anchor_z, anchor_sc, grid[step], grid[step + 1], seed
                )
                for state in (control_z, control_sc, anchor_z, anchor_sc):
                    state[:, :prefix] = prompt_clean
        replicate_signals.append(
            torch.stack([signal_by_lookahead[value] for value in lookaheads], dim=1)
        )
    replicate_signals = torch.stack(replicate_signals, dim=1)
    return replicate_signals.mean(dim=1), replicate_signals


def main():
    args = parse_args()
    if args.probe_replicates < 1:
        raise ValueError("probe_replicates must be positive")
    payload = json.loads(Path(args.exp101_bank).read_text())
    lookaheads = tuple(sorted({int(item) for item in args.lookaheads.split(",")}))
    if min(lookaheads) < 0:
        raise ValueError("lookahead must be non-negative")
    if max(payload["triggers"]) + 1 + max(lookaheads) > payload["n_steps"]:
        raise ValueError("lookahead exceeds the native solver grid")

    device = torch.device(args.device)
    torch.manual_seed(payload["seed"])
    np.random.seed(payload["seed"])
    adapter = load_adapter("plaid", "baseline", None, device)
    adapter.seq_len = payload["seq_len"]
    grid = np.linspace(adapter.t_eps, 0.999, payload["n_steps"] + 1).tolist()
    panel_ids, dataset_name = exp99.load_conditional_panel(
        adapter,
        payload["n_cond"],
        payload["seq_len"],
        payload["panel_offset"],
    )

    batches = []
    replicate_batches = []
    for batch_index, start in enumerate(
        range(0, payload["n_cond"], payload["batch_size"])
    ):
        end = min(start + payload["batch_size"], payload["n_cond"])
        size = end - start
        generator = torch.Generator(device=adapter.device).manual_seed(
            payload["seed"] * 10007 + 700001 + batch_index
        )
        eps = adapter.sample_epsilon(
            (size, payload["seq_len"], adapter.d_model), generator=generator
        )
        prompt_clean = adapter.encode_clean(
            panel_ids[start:end, : payload["prefix_length"]]
        ).to(adapter.device)
        trigger_results = [
            replay_trigger(
                adapter,
                payload,
                grid,
                eps,
                prompt_clean,
                batch_index,
                trigger,
                lookaheads,
                args.probe_replicates,
                args.probe_seed_base,
            )
            for trigger in payload["triggers"]
        ]
        batches.append(torch.stack([result[0] for result in trigger_results], dim=1))
        replicate_batches.append(
            torch.stack([result[1] for result in trigger_results], dim=1)
        )
        print(f"replayed {end}/{payload['n_cond']}", flush=True)

    output = {
        "source_bank": str(Path(args.exp101_bank).resolve()),
        "dataset": dataset_name,
        "seed": payload["seed"],
        "panel_offset": payload["panel_offset"],
        "triggers": payload["triggers"],
        "fixed_trigger": payload["fixed_trigger"],
        "lookaheads": list(lookaheads),
        "signal_names": SIGNAL_NAMES,
        "local_signals": torch.cat(batches).tolist(),
        "local_signals_by_replicate": torch.cat(replicate_batches).tolist(),
        "probe_replicates": args.probe_replicates,
        "probe_seed_routes": [
            args.probe_seed_base + index for index in range(args.probe_replicates)
        ],
        "final_trigger_nll": payload["per_sequence"]["trigger_nll"],
        "final_trigger_token_counts": payload["per_sequence"]["trigger_token_counts"],
        "paired_native_counterfactual_noise": True,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output))
    print(
        f"Saved -> {output_path} signals="
        f"{tuple(torch.cat(batches).shape)} replicates="
        f"{tuple(torch.cat(replicate_batches).shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
