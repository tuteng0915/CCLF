"""EXP-84: counterfactual endpoint steering and basin rigidity.

Reuses a fixed EXP-GS16 endpoint bank.  At several checkpoints on the same
base trajectories, perturb the centered residual along a contrast between the
self endpoint and a reachable alternative endpoint, then continue the native
rollout.  Matched opposite, random-orthogonal, and position-shuffled controls
distinguish endpoint-specific steering from generic trajectory fragility.
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

from branch_true_trajectory import (  # noqa: E402
    rollout_with_checkpoints_and_sc,
)
from common import frobenius_cosine, load_adapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["elf", "langflow"], default="elf")
    parser.add_argument("--checkpoint", default="baseline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--endpoint_bank_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--times", type=float, nargs="+", default=[0.20, 0.30, 0.36, 0.43, 0.55])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.03, 0.10, 0.30])
    parser.add_argument("--n_traj", type=int, default=None)
    parser.add_argument("--full_n_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def centered(z):
    return z - z.mean(dim=-2, keepdim=True)


def unit_frobenius(z):
    return z / z.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)


def endpoint_assignment(z_final, z_bank, n_candidates):
    """Return nearest centered-residual endpoint and all cosine scores."""
    z_np = z_final.detach().cpu().numpy()
    assigned, scores = [], []
    for n, state in enumerate(z_np):
        residual = state - state.mean(axis=0, keepdims=True)
        candidate_scores = []
        for j in range(int(n_candidates[n])):
            endpoint = z_bank[n, j]
            endpoint = endpoint - endpoint.mean(axis=0, keepdims=True)
            candidate_scores.append(frobenius_cosine(residual, endpoint))
        scores.append(candidate_scores)
        assigned.append(int(np.argmax(candidate_scores)))
    return assigned, scores


def gs16_grid(bank_path, t_start, t_end, full_n_steps):
    bank_path = Path(bank_path)
    json_path = Path(str(bank_path).replace("_bank.npz", ".json"))
    if not json_path.exists():
        raise FileNotFoundError(
            f"matching GS16 JSON is required to reproduce the bank trajectory: {json_path}"
        )
    payload = json.load(open(json_path, encoding="utf-8"))
    checkpoints = [round(float(value), 6) for value in payload["checkpoint_ts"]]
    merged = sorted(
        set(np.linspace(t_start, t_end, full_n_steps + 1).tolist())
        | set(checkpoints)
    )
    return checkpoints, merged, str(json_path)


@torch.no_grad()
def continue_on_grid(adapter, z, sc, local_grid, device):
    z = z.to(device)
    sc = sc.to(device)
    for step in range(len(local_grid) - 1):
        z, sc = adapter.solver_step(
            z, sc, local_grid[step], local_grid[step + 1]
        )
    return z.cpu(), sc.cpu(), len(local_grid) - 1


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bank_npz = np.load(args.endpoint_bank_npz)
    bank_seed = int(bank_npz["seed"])
    if bank_seed != args.seed:
        raise ValueError(f"bank seed={bank_seed} but --seed={args.seed}")
    bank_n = int(bank_npz["n_traj"])
    n_traj = bank_n if args.n_traj is None else args.n_traj
    if n_traj > bank_n:
        raise ValueError(f"requested n_traj={n_traj}, bank has only {bank_n}")

    z_bank = bank_npz["z_bank"][:n_traj]
    hamming = bank_npz["hamming"][:n_traj]
    n_candidates = bank_npz["n_candidates_per_traj"][:n_traj]
    t_bank = float(bank_npz["t_bank"])
    t_end = float(bank_npz["t_end"])

    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    length, dim = adapter.seq_len, adapter.d_model
    if z_bank.shape[-2:] != (length, dim):
        raise ValueError(
            f"bank shape {z_bank.shape[-2:]} does not match adapter {(length, dim)}"
        )

    bank_grid, merged_grid, gs16_json = gs16_grid(
        args.endpoint_bank_npz, adapter.t_eps, t_end, args.full_n_steps
    )
    times = sorted(
        set(
            min(bank_grid, key=lambda value: abs(value - float(requested)))
            for requested in args.times
            if requested >= t_bank
        )
    )
    epsilon = adapter.sample_epsilon((n_traj, length, dim))
    saved = rollout_with_checkpoints_and_sc(
        adapter,
        epsilon,
        adapter.t_eps,
        bank_grid,
        args.full_n_steps,
        device,
    )

    # Choose the lexically furthest reachable endpoint per trajectory.  A
    # trajectory without a distinct alternative is retained for bookkeeping
    # but marked invalid and omitted from aggregate redirect rates.
    alt_index = []
    valid = []
    for n in range(n_traj):
        m = int(n_candidates[n])
        if m <= 1:
            alt_index.append(0)
            valid.append(False)
        else:
            j = int(np.argmax(hamming[n, 1:m]) + 1)
            alt_index.append(j)
            valid.append(True)
    bank_t = torch.from_numpy(z_bank).float()
    self_endpoint = bank_t[:, 0]
    alt_endpoint = torch.stack(
        [bank_t[n, alt_index[n]] for n in range(n_traj)], dim=0
    )
    contrast = centered(alt_endpoint) - centered(self_endpoint)
    alt_direction = unit_frobenius(contrast)
    self_direction = -alt_direction

    shuffled_direction = torch.empty_like(alt_direction)
    for n in range(n_traj):
        generator = torch.Generator().manual_seed(args.seed + 1009 + n)
        permutation = torch.randperm(length, generator=generator)
        shuffled_direction[n] = alt_direction[n, permutation]
    shuffled_direction = unit_frobenius(centered(shuffled_direction))

    random = torch.randn(
        n_traj,
        length,
        dim,
        generator=torch.Generator().manual_seed(args.seed + 2027),
    )
    random = centered(random)
    projection = (
        (random * alt_direction).flatten(1).sum(1).view(-1, 1, 1)
        * alt_direction
    )
    random_orthogonal = unit_frobenius(random - projection)

    directions = {
        "no_perturbation": torch.zeros_like(alt_direction),
        "alternative": alt_direction,
        "self": self_direction,
        "random_orthogonal": random_orthogonal,
        "position_shuffled": shuffled_direction,
    }

    bank_self_tokens = adapter.forward_state(
        self_endpoint, None, t_end, batch_size=args.batch_size
    )["logits"].argmax(-1)
    bank_alt_tokens = adapter.forward_state(
        alt_endpoint, None, t_end, batch_size=args.batch_size
    )["logits"].argmax(-1)

    records = []
    for time_value in times:
        z_time, sc_time = saved[time_value]
        residual_norm = centered(z_time).flatten(1).norm(dim=1).view(-1, 1, 1)
        local_grid = [value for value in merged_grid if value >= time_value - 1e-9]
        for epsilon_value in args.epsilons:
            for arm, direction in directions.items():
                if arm == "no_perturbation" and epsilon_value != args.epsilons[0]:
                    continue
                perturbed = z_time + float(epsilon_value) * residual_norm * direction
                z_final, sc_final, n_steps = continue_on_grid(
                    adapter, perturbed, sc_time, local_grid, device
                )
                assigned, cosine_scores = endpoint_assignment(
                    z_final, z_bank, n_candidates
                )
                out = adapter.forward_state(
                    z_final, sc_final, t_end, batch_size=args.batch_size
                )
                final_tokens = out["logits"].argmax(-1).cpu()

                per_traj = []
                for n in range(n_traj):
                    self_agreement = float(
                        (final_tokens[n] == bank_self_tokens[n]).float().mean()
                    )
                    alt_agreement = float(
                        (final_tokens[n] == bank_alt_tokens[n]).float().mean()
                    )
                    per_traj.append(
                        {
                            "traj": n,
                            "valid_alternative": bool(valid[n]),
                            "alternative_index": int(alt_index[n]),
                            "assigned_endpoint": int(assigned[n]),
                            "captured_alternative": bool(
                                valid[n] and assigned[n] == alt_index[n]
                            ),
                            "retained_self": bool(assigned[n] == 0),
                            "self_token_agreement": self_agreement,
                            "alternative_token_agreement": alt_agreement,
                            "endpoint_cosines": cosine_scores[n],
                        }
                    )

                valid_rows = [row for row in per_traj if row["valid_alternative"]]
                redirect_rate = float(
                    np.mean([row["captured_alternative"] for row in valid_rows])
                ) if valid_rows else float("nan")
                self_retention = float(
                    np.mean([row["retained_self"] for row in valid_rows])
                ) if valid_rows else float("nan")
                record = {
                    "t": time_value,
                    "epsilon": float(epsilon_value),
                    "arm": arm,
                    "n_steps_remaining": n_steps,
                    "n_valid": len(valid_rows),
                    "redirect_rate": redirect_rate,
                    "self_retention_rate": self_retention,
                    "mean_self_token_agreement": float(
                        np.mean([row["self_token_agreement"] for row in valid_rows])
                    ) if valid_rows else float("nan"),
                    "mean_alternative_token_agreement": float(
                        np.mean([row["alternative_token_agreement"] for row in valid_rows])
                    ) if valid_rows else float("nan"),
                    "per_traj": per_traj,
                }
                records.append(record)
                print(
                    f"t={time_value:.3f} eps={epsilon_value:.3g} "
                    f"{arm:<19} redirect={redirect_rate:.3f} "
                    f"self={self_retention:.3f}"
                )

        sham = [
            row for row in records
            if row["t"] == time_value and row["arm"] == "no_perturbation"
        ][0]
        if sham["self_retention_rate"] != 1.0:
            raise RuntimeError(
                f"paired continuation gate failed at t={time_value}: "
                f"self retention={sham['self_retention_rate']}"
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
        "epsilons": args.epsilons,
        "endpoint_bank_npz": args.endpoint_bank_npz,
        "gs16_json": gs16_json,
        "n_candidates_per_traj": n_candidates.tolist(),
        "alternative_index": alt_index,
        "records": records,
        "notes": [
            "Perturbation magnitude is epsilon times the centered-state Frobenius norm.",
            "Alternative endpoint is the maximum-Hamming distinct bank endpoint.",
            "Assignment uses centered-residual Frobenius cosine to the fixed bank.",
            "The exact GS16 merged ODE grid is reused; no-perturbation self retention must be 1.0.",
            "ELF/LangFlow deterministic solvers are required for exact paired continuation.",
        ],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"endpoint_steering_{args.label}.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
