#!/usr/bin/env python3
"""EXP-100: replay a Plaid headroom bank and save trigger-time features.

Final-NLL labels come from EXP-99.  This script performs only the shared
pre-trigger rollout, reconstructs candidate masks from their independent mask
seeds, and stores inference-available compact features.  It never rolls out or
uses final text to construct features.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GLOBAL_DIR = ROOT / "experiments" / "global_state"
for path in (HERE, GLOBAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_plaid_subset_headroom_exp99 as exp99  # noqa: E402
import eval_temporary_anchor_portability_exp90 as exp90  # noqa: E402
from common import load_adapter  # noqa: E402


FEATURE_NAMES = (
    tuple(f"z_{index}" for index in range(16))
    + tuple(f"sc_{index}" for index in range(16))
    + tuple(f"xhat_{index}" for index in range(16))
    + ("confidence", "entropy", "top12_margin", "position", "is_prefix")
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headroom_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.no_grad()
def replay_batch(adapter, payload, grid, panel_ids, batch_index, start, end):
    batch_size = end - start
    seq_len = payload["seq_len"]
    prefix_length = payload["prefix_length"]
    trigger_step = payload["trigger_step"]
    seed = payload["seed"]

    generator = torch.Generator(device=adapter.device).manual_seed(
        seed * 10007 + 700001 + batch_index
    )
    z = adapter.sample_epsilon(
        (batch_size, seq_len, adapter.d_model), generator=generator
    )
    prompt_ids = panel_ids[start:end, :prefix_length]
    prompt_clean = adapter.encode_clean(prompt_ids).to(adapter.device)
    sc = torch.zeros_like(z)
    z[:, :prefix_length] = prompt_clean
    sc[:, :prefix_length] = prompt_clean

    for step in range(trigger_step):
        z, sc = exp90.native_step(
            adapter,
            z,
            sc,
            grid[step],
            grid[step + 1],
            exp90.step_seed(seed, 29, batch_index, step),
        )
        z[:, :prefix_length] = prompt_clean
        sc[:, :prefix_length] = prompt_clean

    out = adapter.forward_state(
        z, sc, grid[trigger_step], batch_size=payload["batch_size"]
    )
    logits = out["logits"].float()
    predicted_clean = out["predicted_clean"].float()
    probabilities = torch.softmax(logits, dim=-1)
    confidence = probabilities.max(dim=-1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    top2 = probabilities.topk(k=2, dim=-1).values
    top12_margin = top2[..., 0] - top2[..., 1]

    position = torch.linspace(0.0, 1.0, seq_len).view(1, seq_len).expand(batch_size, -1)
    is_prefix = torch.zeros(batch_size, seq_len)
    is_prefix[:, :prefix_length] = 1.0
    features = torch.cat(
        (
            z.float().cpu(),
            sc.float().cpu(),
            predicted_clean.cpu(),
            confidence.unsqueeze(-1).cpu(),
            entropy.unsqueeze(-1).cpu(),
            top12_margin.unsqueeze(-1).cpu(),
            position.unsqueeze(-1),
            is_prefix.unsqueeze(-1),
        ),
        dim=-1,
    )
    del logits, probabilities, top2

    eligible = torch.ones(batch_size, seq_len, dtype=torch.bool)
    eligible[:, :prefix_length] = False
    masks = []
    for mask_index in range(payload["n_masks"]):
        mask_seed = seed + 1009 * (mask_index + 1)
        mask_generator = torch.Generator(device="cpu").manual_seed(
            exp90.step_seed(mask_seed, 29, batch_index, trigger_step)
        )
        scores = torch.rand(confidence.shape, generator=mask_generator)
        masks.append(exp90.exact_mask(scores, eligible, payload["density"]))
    return features, torch.stack(masks, dim=1), panel_ids[start:end]


def main():
    args = parse_args()
    payload = json.loads(Path(args.headroom_json).read_text())
    required = (
        "seed",
        "panel_offset",
        "n_cond",
        "n_masks",
        "batch_size",
        "seq_len",
        "prefix_length",
        "n_steps",
        "trigger_step",
        "density",
        "per_sequence",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"headroom JSON missing keys: {missing}")
    if not payload.get("paired_initial_and_ancestral_noise", False):
        raise ValueError("EXP-100 requires the corrected paired Plaid protocol")

    labels = torch.tensor(payload["per_sequence"]["random_nll"], dtype=torch.float32).T
    counts = torch.tensor(
        payload["per_sequence"]["random_token_counts"], dtype=torch.long
    ).T
    expected = (payload["n_cond"], payload["n_masks"])
    if tuple(labels.shape) != expected or tuple(counts.shape) != expected:
        raise ValueError(
            f"candidate label shape mismatch: labels={labels.shape}, counts={counts.shape}, "
            f"expected={expected}"
        )

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

    feature_rows, mask_rows, panel_rows = [], [], []
    for batch_index, start in enumerate(
        range(0, payload["n_cond"], payload["batch_size"])
    ):
        end = min(start + payload["batch_size"], payload["n_cond"])
        features, masks, ids = replay_batch(
            adapter, payload, grid, panel_ids, batch_index, start, end
        )
        feature_rows.append(features.half())
        mask_rows.append(masks)
        panel_rows.append(ids)
        print(f"replayed {end}/{payload['n_cond']}", flush=True)

    features = torch.cat(feature_rows)
    masks = torch.cat(mask_rows)
    panel_ids = torch.cat(panel_rows)
    suffix_mask = torch.arange(payload["seq_len"]) >= payload["prefix_length"]
    density = masks[:, :, suffix_mask].float().mean(dim=-1)
    if not torch.allclose(
        density,
        torch.full_like(density, float(payload["density"])),
        atol=1.0 / (payload["seq_len"] - payload["prefix_length"]),
    ):
        raise RuntimeError("reconstructed candidate density drifted")

    output = {
        "source_json": str(Path(args.headroom_json).resolve()),
        "dataset": dataset_name,
        "seed": payload["seed"],
        "panel_offset": payload["panel_offset"],
        "density": payload["density"],
        "trigger_step": payload["trigger_step"],
        "horizon": payload["horizon"],
        "prefix_length": payload["prefix_length"],
        "feature_names": FEATURE_NAMES,
        "features": features,
        "candidate_masks": masks,
        "candidate_nll": labels,
        "candidate_token_counts": counts,
        "standard_nll": torch.tensor(
            payload["per_sequence"]["standard_nll"], dtype=torch.float32
        ),
        "top_confidence_nll": torch.tensor(
            payload["per_sequence"]["top_confidence_nll"], dtype=torch.float32
        ),
        "panel_ids": panel_ids,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        f"Saved -> {output_path} features={tuple(features.shape)} "
        f"masks={tuple(masks.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
