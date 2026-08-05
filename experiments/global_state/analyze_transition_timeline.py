"""EXP-GS17: Local Residual Dynamics and Unified Transition Timing.

GS15 only compares rollout CKA against a direct chord -- a curved but
progressive path can remain below that chord at every intermediate point
without telling us anything about the LOCAL mechanism. This experiment asks
directly: at each state, how much of the actual local velocity reduces
distance to the trajectory's own endpoint, and how much moves in
endpoint-orthogonal directions? It also builds the unified per-position and
per-trajectory transition timeline (tau_first/tau_stable/tau_margin/
tau_50_stable/tau_velocity/tau_affinity) on the SAME true rollouts. See
docs/specs/EXP-GS17-spec.md (Section 8 absorbs the former GS18 spec).

Implementation notes / deviations from the spec:
  - Combines the spec's two proposed scripts (analyze_residual_velocity.py +
    analyze_transition_timeline.py) into one, so the expensive dense-rollout
    collection is only done once.
  - Velocity is estimated ONLY via central finite difference on the saved
    dense trajectory (the spec's "Robustness" method), used uniformly for
    both ELF and LangFlow, because LangFlow's `_euler_edm_step` is a
    gamma-space exponential/lerp update with no simple closed-form drift
    (audited directly against models/LangFlow/langflow/model.py). For ELF,
    the analytic v=(xhat-z)/(1-t) (audited against
    models/ELF-torch/src/utils/sampling_utils.py:net_out_to_v_x, confirmed
    matching the spec's stated convention exactly) is ALSO computed and
    reported alongside the finite-difference estimate as a cross-check, per
    Section 4's "verify sign/location do not depend on method".
  - Section 5 alternative-endpoint control (reusing GS16's fixed bank) is
    computed on the RAW-state representation only; the bank's branch
    endpoints do not have a saved predicted-clean version.
  - tau_affinity / tau_branch are read from a GS16 JSON if --gs16_json is
    given (on GS16's own, sparser checkpoint grid); GS16 defines only one
    kind of endpoint-affinity entropy, so tau_branch and tau_affinity are
    treated as the same underlying quantity (both = steepest H_end drop).
  - tau_collective (from the still-conditional GS18 Part B) is not included.
  - No plotting; this repo's GS scripts are numbers-only, consistent with
    all prior EXP-GSx analysis scripts.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_transition_timeline.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_traj 16 --n_states 65 \\
        --endpoint_bank_npz results/global_state/elf/baseline/endpoint_specificity_pilot_bank.npz \\
        --label pilot
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

from branch_true_trajectory import rollout_with_checkpoints_and_sc  # noqa: E402
from common import frobenius_cosine, load_adapter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow", "plaid"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_traj", type=int, default=16)
    p.add_argument("--n_states", type=int, default=65)
    p.add_argument("--endpoint_bank_npz", default=None,
                    help="from EXP-GS16 (--seed and --n_traj must match to align trajectories)")
    p.add_argument("--gs16_json", default=None,
                    help="from EXP-GS16, used only to read tau_affinity/tau_branch")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def centered_residual(Z):
    mu = Z.mean(axis=0)
    return mu, Z - mu[None, :]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS17] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)

    N = args.n_traj
    L, d = adapter.seq_len, adapter.d_model
    t_start = adapter.t_eps
    t_end = 0.99
    n_steps = args.n_states - 1
    grid = np.linspace(t_start, t_end, args.n_states).round(6).tolist()
    print(f"[GS17] N={N}, L={L}, d={d}, {len(grid)} dense states "
          f"t=[{grid[0]:.4f},{grid[-1]:.4f}]")

    # ---------- dense true-rollout collection (same seed as GS16 -> same base trajectories) ----------
    eps = adapter.sample_epsilon((N, L, d))
    saved = rollout_with_checkpoints_and_sc(adapter, eps, t_start, grid, n_steps, device)
    print("[GS17] dense rollout done")

    logits_by_t, xhat_by_t = {}, {}
    for t in grid:
        z_t, sc_t = saved[t]
        out = adapter.forward_state(z_t, sc_t, t, batch_size=args.batch_size)
        logits_by_t[t] = out["logits"]
        xhat_by_t[t] = out["predicted_clean"]
    print("[GS17] logits + predicted_clean at all checkpoints done")

    # ---------- optional: GS16 endpoint bank for the alternative-endpoint control ----------
    bank = None
    if args.endpoint_bank_npz:
        npz = np.load(args.endpoint_bank_npz)
        assert int(npz["n_traj"]) == N and int(npz["seed"]) == args.seed, (
            "GS16 bank was built with a different --n_traj/--seed; base trajectories "
            "would not align. Rerun GS16 with matching --n_traj/--seed first.")
        bank = {"z_bank": npz["z_bank"], "n_cand": npz["n_candidates_per_traj"],
                "t_bank": float(npz["t_bank"])}
        print(f"[GS17] loaded endpoint bank from {args.endpoint_bank_npz} "
              f"(t_bank={bank['t_bank']:.3f}, up to {bank['z_bank'].shape[1]} candidates)")

    tau_affinity_by_traj = None
    if args.gs16_json:
        g16 = json.load(open(args.gs16_json))
        tau_aff = {}
        for n in range(N):
            recs = [r for r in g16["records"]
                    if r["traj"] == n and r["rep"] == "raw"]
            recs = sorted(recs, key=lambda r: r["t"])
            ts = [r["t"] for r in recs]
            h = [r.get("H_end_unique_beta1", np.nan) for r in recs]
            if len(ts) >= 2:
                dh = np.diff(h) / np.diff(ts)
                tau_aff[n] = float(ts[int(np.argmin(dh))])  # steepest drop
            else:
                tau_aff[n] = None
        tau_affinity_by_traj = tau_aff
        print(f"[GS17] loaded tau_affinity from {args.gs16_json}")

    # ---------- per-trajectory local dynamics + timeline ----------
    per_traj = []
    for n in range(N):
        R_raw = {}
        R_model = {}
        for t in grid:
            _, r = centered_residual(saved[t][0][n].numpy())
            R_raw[t] = r
            _, r_m = centered_residual(xhat_by_t[t][n].numpy())
            R_model[t] = r_m
        R_star_raw = R_raw[grid[-1]]
        R_star_model = R_model[grid[-1]]

        rec = {"traj": n, "grid": grid, "per_state": []}

        # log-SNR percentile (rank-normalized across this trajectory's own grid)
        if args.model == "elf":
            logsnr = np.array([adapter.native_logsnr(t) for t in grid])
        else:
            logsnr = np.array([adapter.native_logsnr(t) for t in grid])
        logsnr_pct = (np.argsort(np.argsort(logsnr)) / (len(logsnr) - 1)).tolist()

        for rep_name, R in [("raw", R_raw), ("model", R_model)]:
            R_star = R_star_raw if rep_name == "raw" else R_star_model
            R_arr = np.stack([R[t] for t in grid], axis=0)  # (S,L,d)
            S = len(grid)

            # finite-difference velocity (central; forward/backward at ends)
            v = np.zeros_like(R_arr)
            for s in range(S):
                if s == 0:
                    v[s] = (R_arr[1] - R_arr[0]) / (grid[1] - grid[0])
                elif s == S - 1:
                    v[s] = (R_arr[s] - R_arr[s - 1]) / (grid[s] - grid[s - 1])
                else:
                    v[s] = (R_arr[s + 1] - R_arr[s - 1]) / (grid[s + 1] - grid[s - 1])

            d_vec = R_star[None, :, :] - R_arr  # (S,L,d), d_s
            dist2 = np.array([np.sum((R_star - R_arr[s]) ** 2) for s in range(S)])
            progress = np.zeros(S)
            for s in range(S):
                if s == 0:
                    progress[s] = -(dist2[1] - dist2[0]) / (grid[1] - grid[0])
                elif s == S - 1:
                    progress[s] = -(dist2[s] - dist2[s - 1]) / (grid[s] - grid[s - 1])
                else:
                    progress[s] = -(dist2[s + 1] - dist2[s - 1]) / (grid[s + 1] - grid[s - 1])

            cos_endpoint = np.zeros(S)
            rho_parallel = np.zeros(S)
            for s in range(S):
                vs, ds = v[s].reshape(-1), d_vec[s].reshape(-1)
                nv, nd = np.linalg.norm(vs), np.linalg.norm(ds)
                cos_endpoint[s] = float(vs @ ds / (nv * nd + 1e-12))
                v_par_norm2 = (vs @ ds) ** 2 / (nd ** 2 + 1e-12)
                rho_parallel[s] = float(v_par_norm2 / (nv ** 2 + 1e-12))

            turning = np.zeros(S)
            for s in range(S - 1):
                a, b = v[s].reshape(-1), v[s + 1].reshape(-1)
                c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                turning[s] = float(np.arccos(np.clip(c, -1.0, 1.0)))
            turning[-1] = float("nan")

            step_disp = [float(np.linalg.norm((R_arr[s] - R_arr[s - 1]).reshape(-1)))
                         for s in range(1, S)]
            arc_len = np.concatenate([[0.0], np.cumsum(step_disp)])
            arc_len_norm = (arc_len / (arc_len[-1] + 1e-12)).tolist()
            chord_disp = [float(np.linalg.norm((R_arr[s] - R_arr[0]).reshape(-1)))
                          for s in range(S)]
            path_eff = [chord_disp[s] / (arc_len[s] + 1e-12) if s > 0 else float("nan")
                        for s in range(S)]

            # Section 5: alternative-endpoint control (raw representation only)
            v_self = [None] * S
            cos_self_list = [None] * S
            if bank is not None and rep_name == "raw":
                m = int(bank["n_cand"][n])
                z_cands = bank["z_bank"][n, :m]  # (m,L,d)
                R_cands = np.stack([centered_residual(z_cands[j])[1] for j in range(m)], axis=0)
                for s in range(S):
                    vs = v[s]
                    cos_j = []
                    for j in range(m):
                        dj = R_cands[j] - R_arr[s]
                        vsf, djf = vs.reshape(-1), dj.reshape(-1)
                        cos_j.append(float(vsf @ djf / (
                            np.linalg.norm(vsf) * np.linalg.norm(djf) + 1e-12)))
                    cos_self_list[s] = cos_j[0]
                    v_self[s] = float(cos_j[0] - np.mean(cos_j[1:])) if m > 1 else float("nan")

            for s, t in enumerate(grid):
                entry = {
                    "t": t, "rep": rep_name, "s_idx": s,
                    "logsnr_percentile": logsnr_pct[s], "arc_length_norm": arc_len_norm[s],
                    "cos_endpoint": cos_endpoint[s], "rho_parallel": rho_parallel[s],
                    "progress": progress[s], "turning": turning[s], "path_eff": path_eff[s],
                    "V_self": v_self[s], "cos_self_to_bank": cos_self_list[s],
                }
                if rep_name == "raw" and args.model == "elf":
                    z_s = saved[t][0][n].numpy()
                    xhat_s = xhat_by_t[t][n].numpy()
                    denom = max(1.0 - t, adapter.t_eps)
                    v_analytic = (xhat_s - z_s) / denom
                    _, v_analytic_c = centered_residual(v_analytic)
                    vf = v[s].reshape(-1)
                    vaf = v_analytic_c.reshape(-1)
                    entry["cos_v_finite_vs_analytic"] = float(
                        vf @ vaf / (np.linalg.norm(vf) * np.linalg.norm(vaf) + 1e-12))
                rec["per_state"].append(entry)

        per_traj.append(rec)
        print(f"  [GS17] traj={n} local dynamics done ({len(grid)} states x 2 reps)")

    # ---------- unified transition timeline (per-position + per-trajectory) ----------
    timeline = []
    for n in range(N):
        logits_seq = torch.stack([logits_by_t[t][n] for t in grid], dim=0)  # (S,L,V)
        top1 = logits_seq.argmax(-1)  # (S,L)
        terminal_token = top1[-1]  # (L,)
        logp = torch.log_softmax(logits_seq.float(), dim=-1)  # (S,L,V)
        f_i = top1[0]  # default competitor fixed at first checkpoint (GS5/GS9 convention)
        ell_y = logp.gather(-1, terminal_token.view(1, -1, 1).expand(len(grid), -1, -1)).squeeze(-1)
        ell_f = logp.gather(-1, f_i.view(1, -1, 1).expand(len(grid), -1, -1)).squeeze(-1)
        margin = (ell_y - ell_f).numpy()  # (S,L)

        S, L_ = top1.shape
        match = (top1 == terminal_token.unsqueeze(0)).numpy()  # (S,L)
        tau_first = np.full(L_, np.nan)
        tau_stable = np.full(L_, np.nan)
        tau_margin = np.full(L_, np.nan)
        for i in range(L_):
            first_idx = np.argmax(match[:, i]) if match[:, i].any() else None
            if first_idx is not None:
                tau_first[i] = grid[first_idx]
            stable_from = None
            for s in range(S):
                if match[s:, i].all():
                    stable_from = s
                    break
            if stable_from is not None:
                tau_stable[i] = grid[stable_from]
            pos_margin = margin[:, i] > 0
            stable_from_m = None
            for s in range(S):
                if pos_margin[s:].all():
                    stable_from_m = s
                    break
            if stable_from_m is not None:
                tau_margin[i] = grid[stable_from_m]

        frac_stable = [(float(np.mean(tau_stable <= t)) if not np.all(np.isnan(tau_stable))
                        else 0.0) for t in grid]
        tau_50_idx = next((s for s, f in enumerate(frac_stable) if f >= 0.5), None)
        tau_50_stable = grid[tau_50_idx] if tau_50_idx is not None else None

        v_self_curve = [e["V_self"] for e in per_traj[n]["per_state"]
                         if e["rep"] == "raw" and e["V_self"] is not None]
        tau_velocity = None
        if len(v_self_curve) >= 2:
            dv = np.diff(v_self_curve) / np.diff(grid)
            tau_velocity = float(grid[int(np.argmax(dv))])

        tau_affinity = tau_affinity_by_traj.get(n) if tau_affinity_by_traj else None

        timeline.append({
            "traj": n,
            "tau_first_per_pos": tau_first.tolist(),
            "tau_stable_per_pos": tau_stable.tolist(),
            "tau_margin_per_pos": tau_margin.tolist(),
            "frac_stable_curve": frac_stable,
            "tau_50_stable": tau_50_stable,
            "tau_velocity": tau_velocity,
            "tau_affinity": tau_affinity,
            "tau_branch": tau_affinity,  # same underlying quantity, see module docstring
            "never_stable_frac": float(np.mean(np.isnan(tau_stable))),
        })
        print(f"  [GS17] traj={n} timeline: tau_50_stable={tau_50_stable} "
              f"tau_velocity={tau_velocity} tau_affinity={tau_affinity}")

    # event ordering across trajectories (fraction where A precedes B)
    def frac_precedes(key_a, key_b):
        pairs = [(t[key_a], t[key_b]) for t in timeline
                 if t[key_a] is not None and t[key_b] is not None]
        if not pairs:
            return None
        return float(np.mean([a <= b for a, b in pairs]))

    event_order = {
        "P(tau_velocity <= tau_50_stable)": frac_precedes("tau_velocity", "tau_50_stable"),
        "P(tau_affinity <= tau_50_stable)": frac_precedes("tau_affinity", "tau_50_stable"),
        "P(tau_velocity <= tau_affinity)": frac_precedes("tau_velocity", "tau_affinity"),
    }

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_traj": N, "n_states": args.n_states, "grid": grid,
        "endpoint_bank_used": args.endpoint_bank_npz,
        "local_dynamics": per_traj, "timeline": timeline, "event_order": event_order,
        "notes": [
            "t=t_end is TAUTOLOGICAL for d_s=R_star-R_s: R_star is defined as "
            "the trajectory's own last saved state, so d_s=0 exactly at the "
            "last grid point and cos_endpoint/rho_parallel there are a "
            "degenerate 0/0 (clamped to 0, not a real 'moving orthogonally' "
            "signal) -- same class of issue as GS16's t_end note and GS15's "
            "A_linear(t_end)=1 artifact. Exclude the last grid point from "
            "local-mechanism claims.",
            "cos_v_finite_vs_analytic (ELF) degrades sharply near t_end (0.99 "
            "at mid-trajectory down to negative at the very last point) -- "
            "finite-difference velocity becomes numerically unstable once "
            "consecutive states are nearly identical, while the analytic "
            "v=(xhat-z)/(1-t) has its own 1/(1-t) blowup there (clamped by "
            "t_eps). Do not over-interpret local-dynamics numbers in the "
            "last 1-2 grid points near t_end for either velocity estimate.",
            "Velocity = central finite difference on the saved dense trajectory "
            "(uniform across ELF/LangFlow); ELF additionally reports "
            "cos_v_finite_vs_analytic (finite-diff vs v=(xhat-z)/(1-t)) as a "
            "cross-check on raw representation only.",
            "Section 5 alternative-endpoint control (V_self/cos_self_to_bank) "
            "uses raw representation only, and only if --endpoint_bank_npz given.",
            "tau_branch is reported identical to tau_affinity (GS16 defines a "
            "single endpoint-affinity entropy; there is no separate 'branch "
            "entropy' construct to distinguish them).",
            "tau_collective (GS18 Part B) not included -- GS18 is conditional "
            "and has not been run.",
            f"Pilot scale (n_traj={N}) -- see EXP-GS17-spec.md before citing.",
        ],
    }
    json_path = out_dir / f"transition_dynamics_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS17] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
