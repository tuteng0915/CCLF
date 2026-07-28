"""EXP-GS15: Residual Organization Trajectory.

Tracks the high-rank, position-specific centered residual R_t = Z_t - mean(Z_t)
along a REAL free-running trajectory, asking whether it approaches its own
final value R_star smoothly or accelerates in some window. Compares against:
  - a paired oracle path (same initial noise, same eventual "clean" endpoint)
  - a pure linear-interpolation control (no model dynamics at all) -- this is
    the key control that rules out the tautology "later t is just closer to
    the endpoint by construction"
  - both raw z_t and the model's own predicted_clean (EXP-GS11 showed these
    behave very differently)

See docs/specs/EXP-GS15-spec.md.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_residual_organization.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_traj 4 --label pilot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
_PT_DIR = _THIS_DIR.parent / "phase_transition"
for p in (_THIS_DIR, _PT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_low_rank_modes import effective_rank, linear_cka  # noqa: E402
from branch_true_trajectory import rollout_with_checkpoints_and_sc  # noqa: E402
from common import load_adapter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_traj", type=int, default=4)
    p.add_argument("--checkpoint_ts", type=float, nargs="+",
                    default=[0.05, 0.20, 0.28, 0.38, 0.50, 0.65, 0.85, 0.99])
    p.add_argument("--full_n_steps", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def centered_residual(Z):
    """Z: (L,d) numpy -> (mu (d,), R (L,d))."""
    mu = Z.mean(axis=0)
    return mu, Z - mu[None, :]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS15] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)

    N, L, d = args.n_traj, adapter.seq_len, adapter.d_model
    t_start = adapter.t_eps
    t_grid = sorted(args.checkpoint_ts)
    t0, t_end = t_grid[0], t_grid[-1]
    print(f"[GS15] {N} genuine free-running trajectories, L={L}, d={d}, "
          f"checkpoints={t_grid}")

    eps = adapter.sample_epsilon((N, L, d))
    saved = rollout_with_checkpoints_and_sc(adapter, eps, t_start, t_grid,
                                             args.full_n_steps, device)
    print(f"[GS15] Stage 1 (free-running rollout) done")

    # predicted_clean at every checkpoint (one extra forward pass per checkpoint)
    predicted_clean_by_t = {}
    for t in t_grid:
        z_t, sc_t = saved[round(t, 6)]
        out = adapter.forward_state(z_t, sc_t, t, batch_size=args.batch_size)
        predicted_clean_by_t[round(t, 6)] = out["predicted_clean"]
    x_clean_rollout = saved[round(t_end, 6)][0]  # raw Z at t_end, used as "clean" proxy

    # paired oracle path, same eps + same eventual endpoint
    oracle_by_t = {}
    for t in t_grid:
        z_oracle = adapter.make_oracle_state(x_clean_rollout.to(device), eps, t).cpu()
        oracle_by_t[round(t, 6)] = z_oracle

    per_traj_records = []
    for n in range(N):
        # residuals for each representation, all checkpoints
        R_raw, R_model, R_oracle = {}, {}, {}
        reff_raw, reff_model = {}, {}
        for t in t_grid:
            tk = round(t, 6)
            _, r = centered_residual(saved[tk][0][n].numpy())
            R_raw[tk] = r
            _, s, _ = np.linalg.svd(r, full_matrices=False)
            reff_raw[tk] = effective_rank(s)

            _, r_m = centered_residual(predicted_clean_by_t[tk][n].numpy())
            R_model[tk] = r_m
            _, s_m, _ = np.linalg.svd(r_m, full_matrices=False)
            reff_model[tk] = effective_rank(s_m)

            _, r_o = centered_residual(oracle_by_t[tk][n].numpy())
            R_oracle[tk] = r_o

        R_star_raw = R_raw[round(t_end, 6)]
        R_star_model = R_model[round(t_end, 6)]
        R_t0_raw = R_raw[round(t0, 6)]
        R_t0_model = R_model[round(t0, 6)]

        traj_rec = {"traj": n, "per_t": []}
        for t in t_grid:
            tk = round(t, 6)
            frac = (t - t0) / (t_end - t0) if t_end > t0 else 1.0
            R_linear = R_t0_raw + frac * (R_star_raw - R_t0_raw)
            R_linear_model = R_t0_model + frac * (R_star_model - R_t0_model)

            a_rollout = linear_cka(R_raw[tk], R_star_raw)
            a_model = linear_cka(R_model[tk], R_star_model)
            a_oracle = linear_cka(R_oracle[tk], R_star_raw)
            a_linear = linear_cka(R_linear, R_star_raw)
            a_linear_model = linear_cka(R_linear_model, R_star_model)

            traj_rec["per_t"].append({
                "t": t, "A_rollout_raw": a_rollout, "A_rollout_model": a_model,
                "A_oracle": a_oracle, "A_linear": a_linear, "A_linear_model": a_linear_model,
                "O_R": a_rollout - a_linear,
                "O_R_model": a_model - a_linear_model,
                "r_eff_raw": reff_raw[tk], "r_eff_model": reff_model[tk],
            })
        per_traj_records.append(traj_rec)
        print(f"  [GS15] traj={n} done")

    def agg(key, t):
        vals = [r for traj in per_traj_records for r in traj["per_t"] if r["t"] == t]
        return float(np.mean([v[key] for v in vals]))

    print("\n[GS15] Summary (mean across trajectories):")
    print(f"{'t':>6} | {'A_rollout':>9} | {'A_model':>9} | {'A_oracle':>9} | "
          f"{'A_linear':>9} | {'O_R':>7} | {'A_lin_mdl':>9} | {'O_R_mdl':>7} | "
          f"{'r_eff_raw':>9} | {'r_eff_model':>11}")
    for t in t_grid:
        print(f"{t:6.3f} | {agg('A_rollout_raw', t):9.3f} | {agg('A_rollout_model', t):9.3f} | "
              f"{agg('A_oracle', t):9.3f} | {agg('A_linear', t):9.3f} | "
              f"{agg('O_R', t):7.3f} | {agg('A_linear_model', t):9.3f} | "
              f"{agg('O_R_model', t):7.3f} | "
              f"{agg('r_eff_raw', t):9.1f} | {agg('r_eff_model', t):11.1f}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_traj": N, "checkpoint_ts": t_grid,
        "per_traj_records": per_traj_records,
        "notes": [
            "R_star is the trajectory's OWN final-checkpoint residual (no external "
            "ground truth -- free-running generation has none), matching EXP-GS7's "
            "convention.",
            "A_linear is a pure straight-line interpolation between the trajectory's "
            "own first- and last-checkpoint residuals -- rules out the tautology "
            "'later t is closer to the endpoint by construction'. O_R = A_rollout - "
            "A_linear is the key excess-organization signal.",
            "A_linear_model / O_R_model are the same construction but using the "
            "model's predicted_clean residual instead of raw z_t -- added because "
            "EXP-GS15's original raw-based O_R saturates near-uninformative on "
            "LangFlow (raw-state CKA >=0.99 from t=0.05 onward); O_R_model is the "
            "metric to use for testing the 'late collapse' hypothesis cross-model.",
            "A_oracle uses the SAME initial eps and the rollout's own endpoint as "
            "x_clean, compared against the RAW representation's R_star (not a "
            "separate oracle R_star) so it is directly comparable to A_rollout_raw.",
            "Pilot scale (n_traj=%d) -- see EXP-GS15-spec.md before citing numbers." % N,
        ],
    }
    json_path = out_dir / f"residual_organization_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS15] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
