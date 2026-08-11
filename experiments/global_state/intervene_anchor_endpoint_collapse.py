"""EXP-85: does cross-position anchoring causally reduce endpoint uncertainty?

This is a mechanism runner, not a replacement sampler.  It reuses a fixed
GS16 endpoint bank and the same base trajectories.  At calibrated checkpoints,
it writes several matched anchor contents into self-conditioning memory for a
short horizon, then measures immediate endpoint entropy on unanchored positions
and final endpoint capture after the anchors are released.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PT_DIR = HERE.parent / "phase_transition"
for path in (HERE, PT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_endpoint_specificity import entropy_and_neff  # noqa: E402
from branch_true_trajectory import (  # noqa: E402
    rollout_branches_from_state,
    rollout_with_checkpoints_and_sc,
)
from common import frobenius_cosine, load_adapter  # noqa: E402


ARMS = ("correct", "position_shuffled", "alternative", "random_endpoint", "sham")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("elf", "langflow"), default="elf")
    parser.add_argument("--checkpoint", default="baseline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--endpoint_bank_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--times", type=float, nargs="+", default=[0.30, 0.36, 0.43])
    parser.add_argument("--densities", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--n_traj", type=int, default=None)
    parser.add_argument("--full_n_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def centered_numpy(z):
    return z - z.mean(axis=-2, keepdims=True)


def exact_top_mask(confidence, density):
    selected = torch.zeros_like(confidence, dtype=torch.bool)
    length = confidence.shape[1]
    count = max(1, min(length - 1, round(density * length)))
    selected.scatter_(1, confidence.topk(count, dim=1).indices, True)
    return selected


def affinity_stats(states, masks, z_bank, n_candidates):
    """Centered endpoint affinity restricted to each row's unanchored mask."""
    states = states.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy().astype(bool)
    rows = []
    for traj, state in enumerate(states):
        keep = ~masks[traj]
        if keep.sum() < 2:
            rows.append({"S_self": float("nan"), "rank_self": None, "H_end": float("nan")})
            continue
        residual = centered_numpy(state[keep])
        scores = []
        for candidate in range(int(n_candidates[traj])):
            endpoint = centered_numpy(z_bank[traj, candidate, keep])
            scores.append(frobenius_cosine(residual, endpoint))
        scores = np.asarray(scores)
        h_raw, h_norm, n_eff = entropy_and_neff(scores, np.ones(len(scores)), 1.0)
        rows.append(
            {
                "S_self": float(scores[0] - scores[1:].mean()) if len(scores) > 1 else float("nan"),
                "rank_self": int((scores > scores[0]).sum() + 1),
                "H_end": h_norm,
                "N_eff": n_eff,
                "affinities": scores.tolist(),
            }
        )
    return rows


def endpoint_assignment(states, z_bank, n_candidates):
    masks = torch.zeros(states.shape[:2], dtype=torch.bool)
    stats = affinity_stats(states, masks, z_bank, n_candidates)
    return [int(np.argmax(row["affinities"])) for row in stats]


@torch.no_grad()
def main():
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if any(not 0 < density < 1 for density in args.densities):
        raise ValueError("densities must lie inside (0,1)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bank_npz = np.load(args.endpoint_bank_npz)
    if int(bank_npz["seed"]) != args.seed:
        raise ValueError("endpoint bank seed must match --seed")
    bank_n = int(bank_npz["n_traj"])
    n_traj = bank_n if args.n_traj is None else args.n_traj
    if n_traj > bank_n:
        raise ValueError("requested trajectories exceed endpoint bank")
    z_bank = bank_npz["z_bank"][:n_traj]
    hamming = bank_npz["hamming"][:n_traj]
    n_candidates = bank_npz["n_candidates_per_traj"][:n_traj]
    t_end = float(bank_npz["t_end"])
    t_bank = float(bank_npz["t_bank"])
    times = sorted(set(round(float(t), 6) for t in args.times if t >= t_bank))

    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    length, dim = adapter.seq_len, adapter.d_model
    if z_bank.shape[-2:] != (length, dim):
        raise ValueError("endpoint bank and adapter shapes do not match")

    alt_index = []
    random_index = []
    for traj in range(n_traj):
        count = int(n_candidates[traj])
        alt_index.append(
            int(np.argmax(hamming[traj, 1:count]) + 1) if count > 1 else 0
        )
        random_index.append((traj + 1) % n_traj)

    epsilon = adapter.sample_epsilon((n_traj, length, dim))
    checkpoints = sorted(set(times + [round(t_end, 6)]))
    merged_grid = sorted(
        set(np.linspace(adapter.t_eps, t_end, args.full_n_steps + 1).tolist())
        | set(checkpoints)
    )
    saved = rollout_with_checkpoints_and_sc(
        adapter,
        epsilon,
        adapter.t_eps,
        checkpoints,
        args.full_n_steps,
        device,
    )

    records = []
    for time_value in times:
        z_time, sc_time = saved[time_value]
        readout = adapter.forward_state(
            z_time, sc_time, time_value, batch_size=args.batch_size
        )
        confidence = torch.softmax(readout["logits"].float(), dim=-1).amax(-1)
        predicted_clean = readout["predicted_clean"].cpu()
        # Reuse exactly the suffix of the merged base grid so the sham arm is
        # an exact paired continuation rather than a slightly different ODE.
        local_grid = [value for value in merged_grid if value >= time_value - 1e-9]
        remaining = len(local_grid) - 1

        for density in args.densities:
            selected = exact_top_mask(confidence, density)
            before = affinity_stats(predicted_clean, selected, z_bank, n_candidates)
            for arm in args.arms:
                anchor = predicted_clean.clone()
                if arm == "position_shuffled":
                    for traj in range(n_traj):
                        permutation = torch.randperm(
                            length,
                            generator=torch.Generator().manual_seed(
                                args.seed + 1009 * traj + int(time_value * 1000)
                            ),
                        )
                        anchor[traj] = predicted_clean[traj, permutation]
                elif arm == "alternative":
                    anchor = torch.stack(
                        [torch.from_numpy(z_bank[n, alt_index[n]]) for n in range(n_traj)]
                    ).float()
                elif arm == "random_endpoint":
                    anchor = torch.stack(
                        [torch.from_numpy(z_bank[random_index[n], 0]) for n in range(n_traj)]
                    ).float()

                z = z_time.to(device).clone()
                sc = sc_time.to(device).clone()
                first_xhat = None
                for step in range(remaining):
                    if arm != "sham" and step < args.horizon:
                        sc = torch.where(
                            selected.to(device).unsqueeze(-1),
                            anchor.to(device),
                            sc,
                        )
                    z, sc = adapter.solver_step(
                        z, sc, local_grid[step], local_grid[step + 1]
                    )
                    if first_xhat is None:
                        first_xhat = sc.detach().cpu()

                after = affinity_stats(first_xhat, selected, z_bank, n_candidates)
                assigned = endpoint_assignment(z.detach().cpu(), z_bank, n_candidates)
                per_traj = []
                for traj in range(n_traj):
                    per_traj.append(
                        {
                            "traj": traj,
                            "before": before[traj],
                            "after_one_step": after[traj],
                            "delta_H_end": after[traj]["H_end"] - before[traj]["H_end"],
                            "delta_S_self": after[traj]["S_self"] - before[traj]["S_self"],
                            "final_endpoint": assigned[traj],
                            "captured_alternative": assigned[traj] == alt_index[traj],
                            "retained_self": assigned[traj] == 0,
                        }
                    )
                record = {
                    "t": time_value,
                    "density": float(density),
                    "arm": arm,
                    "horizon": args.horizon,
                    "selected_fraction": float(selected.float().mean()),
                    "mean_delta_H_end": float(np.nanmean([r["delta_H_end"] for r in per_traj])),
                    "mean_delta_S_self": float(np.nanmean([r["delta_S_self"] for r in per_traj])),
                    "alternative_capture_rate": float(np.mean([r["captured_alternative"] for r in per_traj])),
                    "self_retention_rate": float(np.mean([r["retained_self"] for r in per_traj])),
                    "per_traj": per_traj,
                }
                records.append(record)
                print(
                    f"t={time_value:.3f} density={density:.2f} {arm:<18} "
                    f"dH={record['mean_delta_H_end']:+.4f} "
                    f"alt={record['alternative_capture_rate']:.3f}"
                )

    output = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "label": args.label,
        "seed": args.seed,
        "n_traj": n_traj,
        "t_bank": t_bank,
        "t_end": t_end,
        "times": times,
        "densities": args.densities,
        "horizon": args.horizon,
        "endpoint_bank_npz": args.endpoint_bank_npz,
        "records": records,
        "notes": [
            "Anchors replace self-conditioning memory only; the latent remains free.",
            "Endpoint entropy is computed on unanchored positions only.",
            "Alternative content comes from the maximum-Hamming reachable endpoint.",
            "This mechanism runner omits PPL; EXP-82 supplies the matched quality panel.",
        ],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"anchor_endpoint_collapse_{args.label}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
