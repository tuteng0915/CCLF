"""EXP-84: antithetic Monte Carlo estimate of Plaid's conditional mean drift.

Plaid's native ancestral transition injects independent Gaussian noise at each
step.  A finite difference along one sampled path conflates this diffusion
term with the learned conditional drift.  This runner repeatedly advances the
same saved state with paired xi/-xi noise, then recomputes endpoint alignment
and endpoint specificity from the mean increment.
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

from common import frobenius_cosine, load_adapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint_bank_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--n_traj", type=int, default=None)
    parser.add_argument("--n_states", type=int, default=17)
    parser.add_argument("--k_draws", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def centered_numpy(z):
    return z - z.mean(axis=-2, keepdims=True)


@torch.no_grad()
def main():
    args = parse_args()
    if args.k_draws < 2 or args.k_draws % 2:
        raise ValueError("--k_draws must be a positive even number")
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
    n_candidates = bank_npz["n_candidates_per_traj"][:n_traj]

    adapter = load_adapter("plaid", "baseline", None, device)
    length, dim = adapter.seq_len, adapter.d_model
    if z_bank.shape[-2:] != (length, dim):
        raise ValueError("endpoint bank and Plaid adapter shapes do not match")

    grid = np.linspace(adapter.t_eps, 0.99, args.n_states).round(6).tolist()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z = adapter.sample_epsilon((n_traj, length, dim), generator=generator)
    sc = torch.zeros_like(z)
    saved = {grid[0]: (z.detach().cpu(), sc.detach().cpu())}
    for index in range(len(grid) - 1):
        z, sc = adapter.solver_step(
            z, sc, grid[index], grid[index + 1], generator=generator
        )
        saved[grid[index + 1]] = (z.detach().cpu(), sc.detach().cpu())

    records = []
    for index in range(len(grid) - 1):
        time_value, next_time = grid[index], grid[index + 1]
        delta_t = next_time - time_value
        z_time, sc_time = saved[time_value]
        increments = []
        for pair in range(args.k_draws // 2):
            noise = torch.randn(
                n_traj,
                length,
                dim,
                generator=torch.Generator(device=device).manual_seed(
                    args.seed * 100003 + index * 1009 + pair
                ),
                device=device,
                dtype=torch.float64,
            )
            for sign in (1.0, -1.0):
                z_next, _ = adapter.solver_step(
                    z_time.to(device),
                    sc_time.to(device),
                    time_value,
                    next_time,
                    noise=sign * noise,
                )
                increments.append(((z_next.cpu() - z_time) / delta_t).numpy())

        draws = np.stack(increments, axis=0)  # (K,N,L,d)
        mean_drift = draws.mean(axis=0)
        centered_drift = centered_numpy(mean_drift)
        centered_draws = centered_numpy(draws)
        noise_component = centered_draws - centered_drift[None]

        for traj in range(n_traj):
            state_residual = centered_numpy(z_time[traj].numpy())
            m = int(n_candidates[traj])
            endpoint_residuals = centered_numpy(z_bank[traj, :m])
            own_direction = endpoint_residuals[0] - state_residual
            candidate_cosines = [
                frobenius_cosine(centered_drift[traj], endpoint_residuals[j] - state_residual)
                for j in range(m)
            ]
            v_self = (
                float(candidate_cosines[0] - np.mean(candidate_cosines[1:]))
                if m > 1 else float("nan")
            )
            drift_norm = float(np.linalg.norm(centered_drift[traj].reshape(-1)))
            noise_rms = float(
                np.sqrt(np.mean(np.sum(noise_component[:, traj].reshape(args.k_draws, -1) ** 2, axis=1)))
            )
            single_cosines = [
                frobenius_cosine(centered_draws[k, traj], own_direction)
                for k in range(args.k_draws)
            ]
            records.append(
                {
                    "traj": traj,
                    "t": time_value,
                    "t_next": next_time,
                    "logsnr": adapter.native_logsnr(time_value),
                    "cos_endpoint_mean_drift": frobenius_cosine(
                        centered_drift[traj], own_direction
                    ),
                    "cos_endpoint_single_mean": float(np.mean(single_cosines)),
                    "cos_endpoint_single_std": float(np.std(single_cosines)),
                    "candidate_cosines_mean_drift": candidate_cosines,
                    "V_self_mean_drift": v_self,
                    "drift_norm": drift_norm,
                    "diffusion_noise_rms": noise_rms,
                    "drift_to_noise_ratio": drift_norm / (noise_rms + 1e-12),
                    "mc_standard_error_ratio": (
                        noise_rms / np.sqrt(args.k_draws) / (drift_norm + 1e-12)
                    ),
                }
            )
        print(
            f"t={time_value:.3f} mean drift/noise="
            f"{np.mean([r['drift_to_noise_ratio'] for r in records if r['t'] == time_value]):.4f}"
        )

    timeline = []
    for traj in range(n_traj):
        rows = [row for row in records if row["traj"] == traj]
        curve = np.array([row["V_self_mean_drift"] for row in rows])
        times = np.array([row["t"] for row in rows])
        if len(curve) > 1 and np.isfinite(curve).sum() > 1:
            derivative = np.diff(curve) / np.diff(times)
            tau_velocity = float(times[int(np.nanargmax(derivative))])
        else:
            tau_velocity = None
        timeline.append({"traj": traj, "tau_velocity_debiased": tau_velocity})

    output = {
        "model": "plaid",
        "label": args.label,
        "seed": args.seed,
        "n_traj": n_traj,
        "n_states": args.n_states,
        "k_draws": args.k_draws,
        "grid": grid,
        "endpoint_bank_npz": args.endpoint_bank_npz,
        "records": records,
        "timeline": timeline,
        "notes": [
            "Each conditional drift estimate uses exact antithetic xi/-xi pairs.",
            "The endpoint bank and base trajectory must share seed and trajectory order.",
            "GS18-B collective susceptibility remains a separate second-stage analysis.",
        ],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"plaid_debiased_drift_{args.label}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
