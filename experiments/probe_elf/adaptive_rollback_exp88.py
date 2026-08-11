#!/usr/bin/env python3
"""EXP-88: one-shot unmasked shadow checks for adaptive anchor rollback."""

import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import robust_revisable_commit_exp78 as exp78
import transition_unlock_pareto_exp82 as base
import unified_method_eval_exp64 as common
from utils.sampling_utils import _ode_step, restore_cond


ORIGINAL_PARSE_ARM = base.parse_arm
ORIGINAL_ROLLOUT = base.rollout
ROLLBACK_ARMS = (
    "shadow_null_t30",
    "random_shadowkeep_t30_q50_h4",
    "rollback_identity_t30_q50_h4",
    "rollback_conf10_t30_q50_h4",
    "rollback_combined_t30_q50_h4",
)
DEFAULT_ARMS = (
    "standard32",
    "random_t30_q50_h4",
    *ROLLBACK_ARMS,
)
OUT_DIR = Path("results/exp88_adaptive_rollback")


def parse_arm(arm):
    if arm not in ROLLBACK_ARMS:
        return ORIGINAL_PARSE_ARM(arm)
    if arm == "shadow_null_t30":
        mode = "shadow_null"
    elif arm.startswith("random_shadowkeep"):
        mode = "shadow_keep"
    elif arm.startswith("rollback_identity"):
        mode = "identity"
    elif arm.startswith("rollback_conf10"):
        mode = "confidence"
    else:
        mode = "combined"
    return {
        "mode": mode,
        "time": 0.30,
        "density": 0.50,
        "horizon": 4,
    }


@torch.no_grad()
def rollout(z0, model, grid, args, arm, rng_seed, cond_seq=None, cond_mask=None):
    if arm not in ROLLBACK_ARMS:
        return ORIGINAL_ROLLOUT(
            z0, model, grid, args, arm, rng_seed, cond_seq, cond_mask
        )
    cell = parse_arm(arm)
    cfg = common.SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = common.empty_condition(z0)
    base_seq, base_mask = cond_seq.clone(), cond_mask.clone()
    active_seq, active_mask = cond_seq.clone(), cond_mask.clone()
    z = restore_cond(z0.clone(), active_seq, active_mask)
    x_pred = restore_cond(torch.zeros_like(z), active_seq, active_mask)
    triggered = False
    trigger_index = release_index = shadow_index = None
    selected = torch.zeros_like(base_mask, dtype=torch.bool)
    trigger_ids = torch.zeros_like(base_mask, dtype=torch.long)
    trigger_confidence = torch.zeros_like(base_mask)
    selected_total = released_total = 0
    eligible_total = int((base_mask < 0.5).sum().item())
    selected_confidence_sum = 0.0
    readout_calls = shadow_denoiser_calls = 0
    random_generator = torch.Generator(device=z.device).manual_seed(rng_seed)

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for index in range(grid.shape[0] - 1):
            if release_index is not None and index >= release_index:
                active_seq = base_seq.clone()
                active_mask = base_mask.clone()
                release_index = None

            if shadow_index is not None and index == shadow_index:
                current_t = grid[index].item()
                _, shadow_x = _ode_step(
                    model=model,
                    z=z,
                    t=current_t,
                    t_next=current_t,
                    x_pred_prev=x_pred,
                    config=cfg,
                    cfg_scale=1.0,
                    self_cond_cfg_scale=args.sccfg,
                    cond_seq=base_seq,
                    cond_seq_mask=base_mask,
                )
                shadow_denoiser_calls += 1
                shadow_ids, shadow_confidence = exp78.lexical_readout(
                    shadow_x, model
                )
                readout_calls += 1
                release = torch.zeros_like(selected)
                if cell["mode"] in ("identity", "combined"):
                    release |= selected & (shadow_ids != trigger_ids)
                if cell["mode"] in ("confidence", "combined"):
                    release |= selected & (
                        shadow_confidence < trigger_confidence - 0.10
                    )
                released_total = int(release.sum().item())
                if released_total:
                    active_seq = torch.where(
                        release.unsqueeze(-1), base_seq, active_seq
                    )
                    active_mask = torch.where(
                        release, base_mask, active_mask
                    )
                    z = restore_cond(z, active_seq, active_mask)
                    x_pred = restore_cond(x_pred, active_seq, active_mask)

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
            t_next = grid[index + 1].item()
            if not triggered and t_next >= cell["time"]:
                trigger_ids, trigger_confidence = exp78.lexical_readout(
                    x_pred, model
                )
                readout_calls += 1
                triggered = True
                trigger_index = index
                shadow_index = index + 1 + 2
                release_index = index + 1 + cell["horizon"]
                if cell["mode"] != "shadow_null":
                    eligible = base_mask < 0.5
                    scores = torch.rand(
                        trigger_confidence.shape,
                        generator=random_generator,
                        device=trigger_confidence.device,
                    )
                    selected = base.exact_budget_mask(
                        scores, eligible, cell["density"]
                    )
                    selected_total = int(selected.sum().item())
                    selected_confidence_sum = float(
                        trigger_confidence[selected].sum().item()
                    )
                    active_seq = torch.where(
                        selected.unsqueeze(-1), x_pred.detach(), active_seq
                    )
                    active_mask = torch.maximum(
                        active_mask, selected.to(active_mask.dtype)
                    )
                    z = restore_cond(z, active_seq, active_mask)
                    x_pred = restore_cond(x_pred, active_seq, active_mask)

    del trigger_index
    return z, {
        "denoiser_calls": args.n_steps + shadow_denoiser_calls,
        "readout_calls": readout_calls,
        "anchor_fraction": selected_total / max(eligible_total, 1),
        "anchor_confidence": (
            selected_confidence_sum / selected_total
            if selected_total else float("nan")
        ),
        "release_fraction": released_total / max(selected_total, 1),
        "trigger_time": cell["time"],
        "lock_horizon": cell["horizon"],
        "selection_mode": cell["mode"],
    }


def main():
    base.DEFAULT_ARMS = DEFAULT_ARMS
    base.OUT_DIR = OUT_DIR
    base.parse_arm = parse_arm
    base.rollout = rollout
    base.main()


if __name__ == "__main__":
    main()
