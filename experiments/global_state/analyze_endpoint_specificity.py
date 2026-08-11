"""EXP-GS16: Calibrated Endpoint Bank, Specificity, and Affinity Collapse.

GS15 only shows a rollout is less similar to its endpoint than a direct
chord. It does not tell us whether the endpoint is already selected at
intermediate t. This experiment builds one FIXED candidate endpoint bank per
base trajectory (the trajectory's own unperturbed endpoint plus K calibrated
branch endpoints split at t_bank), then scores every later checkpoint of the
base trajectory against that same fixed bank -- distinguishing "curved
transport toward an already-specific endpoint" from "exploration then late
collapse". See docs/specs/EXP-GS16-spec.md.

Implementation notes / deviations from the spec (kept honest per this repo's
convention):
  - Stage 0 implements only the "one-step matched impact" calibration
    protocol (bisect a scalar eta so the median relative divergence after
    ONE native solver step equals kappa_step). The "terminal-linearized
    matched impact" protocol (JVP / finite-diff amplification estimate) is
    NOT implemented -- Control 6 (perturbation-calibration robustness across
    both protocols) is therefore not available yet.
  - Controls implemented: position-shuffled endpoint (4), mean-only (5),
    multiplicity-aware unique vs basin-mass entropy (3), cross-trajectory
    endpoint null (1, approximate -- matched by sequence length only, not
    terminal token entropy), lexical-distance stratification (2, via the
    saved Hamming distance -- can be stratified post-hoc from the JSON).
  - Cross-trajectory null and position-shuffle controls are computed for the
    cosine metric only (the primary metric per spec Section 4); CKA is
    reported for the primary a_j(t) only, as a robustness check.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_endpoint_specificity.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_traj 16 --k_branches 8 --label pilot
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

from analyze_low_rank_modes import linear_cka  # noqa: E402
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
    p.add_argument("--k_branches", type=int, default=8)
    p.add_argument("--t_bank", type=float, default=0.20)
    p.add_argument("--checkpoint_ts", type=float, nargs="+", default=None,
                    help="defaults to 17 points spanning [t_eps, 0.99] incl. t_bank")
    p.add_argument("--kappa_steps", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--diversity_n_traj", type=int, default=6,
                    help="subset used to pick the calibration kappa_step before "
                         "committing compute to the full n_traj x K bank")
    p.add_argument("--min_unique_endpoints", type=int, default=4)
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--full_n_steps", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def centered_residual(Z):
    """Z: (L,d) numpy -> (mu (d,), R (L,d))."""
    mu = Z.mean(axis=0)
    return mu, Z - mu[None, :]


def _solver_step(adapter, z, sc, t, t_next, noise=None):
    """Call a native step, supplying explicit noise only for Plaid.

    Plaid's ancestral sampler is stochastic.  Common random numbers are
    required when the purpose of a comparison is to isolate a state
    perturbation rather than ordinary sampler noise.
    """
    if adapter.name == "plaid":
        return adapter.solver_step(z, sc, t, t_next, noise=noise)
    return adapter.solver_step(z, sc, t, t_next)


def _batched_solver_step(adapter, z, sc, t, t_next, batch_size, noise=None):
    """Memory-bounded native step without changing per-example dynamics."""
    if batch_size is None or z.shape[0] <= batch_size:
        return _solver_step(adapter, z, sc, t, t_next, noise=noise)
    z_parts, sc_parts = [], []
    for start in range(0, z.shape[0], batch_size):
        end = min(start + batch_size, z.shape[0])
        z_next, sc_next = _solver_step(
            adapter,
            z[start:end],
            sc[start:end],
            t,
            t_next,
            noise=(noise[start:end] if noise is not None else None),
        )
        z_parts.append(z_next)
        sc_parts.append(sc_next)
    return torch.cat(z_parts, dim=0), torch.cat(sc_parts, dim=0)


@torch.no_grad()
def rollout_base_with_checkpoints(adapter, z_start, t_start, checkpoint_ts,
                                  n_steps, device, seed, batch_size=None):
    """Base rollout plus the exact Plaid noise schedule after every step.

    For deterministic adapters this is the usual checkpoint rollout.  For
    Plaid we own the ancestral noise draws so the same future randomness can
    be replayed for every counterfactual branch.
    """
    if adapter.name != "plaid":
        saved = rollout_with_checkpoints_and_sc(
            adapter, z_start, t_start, checkpoint_ts, n_steps, device)
        return saved, None, None

    t_end = max(checkpoint_ts)
    merged = sorted(set(np.linspace(t_start, t_end, n_steps + 1).tolist()) |
                    set(checkpoint_ts))
    checkpoint_set = set(round(t, 6) for t in checkpoint_ts)
    gen = torch.Generator().manual_seed(seed)
    z = z_start.to(device)
    sc = torch.zeros_like(z)
    saved = {}
    step_noises = []
    if round(merged[0], 6) in checkpoint_set:
        saved[round(merged[0], 6)] = (z.cpu(), sc.cpu())
    for i in range(len(merged) - 1):
        t, t_next = merged[i], merged[i + 1]
        noise = torch.randn(z.shape, generator=gen)
        z, sc = _batched_solver_step(
            adapter, z, sc, t, t_next, batch_size, noise=noise)
        step_noises.append(noise.cpu())
        if round(t_next, 6) in checkpoint_set:
            saved[round(t_next, 6)] = (z.cpu(), sc.cpu())
    return saved, merged, step_noises


@torch.no_grad()
def one_step_relative_divergence(adapter, z, sc, t, t_next, eta, u, device,
                                 paired_noise=None):
    """z, sc, u: (L,d) cpu (sc may be None). Returns scalar relative Frobenius
    divergence between one native solver step from z and from z+eta*u."""
    z_b = z.unsqueeze(0).to(device)
    sc_b = (sc.unsqueeze(0).to(device) if sc is not None else torch.zeros_like(z_b))
    z_pert_b = z_b + eta * u.unsqueeze(0).to(device)
    noise_b = paired_noise.unsqueeze(0) if paired_noise is not None else None
    z_next, _ = _solver_step(adapter, z_b, sc_b, t, t_next, noise=noise_b)
    z_next_pert, _ = _solver_step(
        adapter, z_pert_b, sc_b, t, t_next, noise=noise_b)
    diff = (z_next_pert - z_next).reshape(-1)
    base = z_next.reshape(-1)
    return float(diff.norm() / (base.norm() + 1e-12))


def calibrate_eta(adapter, z_bank, sc_bank, t_bank, t_next, kappa_step, device,
                  seed, paired_noise=None):
    """z_bank: (N,L,d) cpu states at t_bank for N base trajectories.
    Bisects a scalar eta so median one-step relative divergence == kappa_step.
    Returns (eta, u) where u: (N,L,d) is the (single, magnitude-probing) unit
    direction per trajectory used during calibration."""
    N, L, d = z_bank.shape
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(N, L, d, generator=gen)
    u = u / u.reshape(N, -1).norm(dim=1).view(N, 1, 1).clamp(min=1e-12)

    def median_div(eta):
        divs = []
        for n in range(N):
            sc_n = sc_bank[n] if sc_bank is not None else None
            divs.append(one_step_relative_divergence(
                adapter, z_bank[n], sc_n, t_bank, t_next, eta, u[n], device,
                paired_noise[n] if paired_noise is not None else None))
        return float(np.median(divs))

    lo, hi = 1e-6, 1.0
    for _ in range(6):
        if median_div(hi) > kappa_step:
            break
        hi *= 3.0
    for _ in range(14):
        mid = (lo + hi) / 2.0
        if median_div(mid) > kappa_step:
            hi = mid
        else:
            lo = mid
    eta = (lo + hi) / 2.0
    return eta, u


@torch.no_grad()
def rollout_k_branches(adapter, z_bank, sc_bank, t_bank, t_end, K, eta, full_n_steps,
                        device, seed, t_steps=None, paired_step_noises=None,
                        solver_batch_size=None):
    """z_bank, sc_bank: (N,L,d) cpu -> z_final (N,K,L,d) cpu, decoded tokens."""
    N, L, d = z_bank.shape
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(N * K, L, d, generator=gen)
    u = u / u.reshape(N * K, -1).norm(dim=1).view(N * K, 1, 1).clamp(min=1e-12)

    z_rep = z_bank.unsqueeze(1).expand(N, K, L, d).reshape(N * K, L, d).clone()
    sc_rep = (sc_bank.unsqueeze(1).expand(N, K, L, d).reshape(N * K, L, d).clone()
              if sc_bank is not None else torch.zeros(N * K, L, d))
    z_rep = z_rep + eta * u

    if t_steps is None:
        n_steps = max(4, round(full_n_steps * (t_end - t_bank)))
        t_steps = torch.linspace(t_bank, t_end, n_steps + 1).tolist()
    if paired_step_noises is not None and len(paired_step_noises) != len(t_steps) - 1:
        raise ValueError("paired Plaid noise schedule does not match continuation grid")
    z = z_rep.to(device)
    sc = sc_rep.to(device)
    for i in range(len(t_steps) - 1):
        noise = None
        if paired_step_noises is not None:
            # Same ancestral draw for all K arms belonging to one trajectory.
            noise = (paired_step_noises[i].unsqueeze(1)
                     .expand(N, K, L, d).reshape(N * K, L, d))
        z, sc = _batched_solver_step(
            adapter, z, sc, t_steps[i], t_steps[i + 1],
            solver_batch_size, noise=noise)
    return z.cpu().reshape(N, K, L, d), sc.cpu().reshape(N, K, L, d)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mu, sd = x.mean(), x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def entropy_and_neff(a_vals, weights, beta):
    """a_vals: (K+1,) affinities, weights: (K+1,) multiplicities (unique-endpoint
    entropy uses weights=1s; basin-mass entropy uses real multiplicities as an
    ADDITIVE log-count bonus to logits, see Section 6 point 3)."""
    z = zscore(a_vals)
    logits = beta * z + np.log(np.asarray(weights, dtype=np.float64) + 1e-12)
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / p.sum()
    K1 = len(a_vals)
    h_raw = float(-(p * np.log(p + 1e-12)).sum())
    h_norm = h_raw / np.log(K1) if K1 > 1 else 0.0
    n_eff = float(np.exp(h_raw))
    return h_raw, h_norm, n_eff


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS16] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    tokenizer = adapter.tokenizer

    N, K = args.n_traj, args.k_branches
    L, d = adapter.seq_len, adapter.d_model
    t_start = adapter.t_eps
    t_end = 0.99
    if args.checkpoint_ts is None:
        # 17 points spanning [t_start, t_end], guaranteed to include t_bank
        grid = sorted(set(np.linspace(t_start, t_end, 16).round(6).tolist()
                           + [round(args.t_bank, 6)]))
    else:
        grid = sorted(set(round(t, 6) for t in args.checkpoint_ts) | {round(args.t_bank, 6)})
    t_bank = round(args.t_bank, 6)
    print(f"[GS16] N={N}, K={K}, L={L}, d={d}, t_bank={t_bank}, "
          f"{len(grid)} checkpoints={grid}")

    # ---------- Stage 1: dense unperturbed base rollout (all N) ----------
    eps = adapter.sample_epsilon((N, L, d))
    saved, base_steps, base_step_noises = rollout_base_with_checkpoints(
        adapter, eps, t_start, grid, args.full_n_steps, device,
        seed=args.seed + 10_000, batch_size=args.batch_size)
    print("[GS16] Stage 1 done: dense base rollout saved at all checkpoints")

    z_bank_all, sc_bank_all = saved[t_bank]
    t_next_for_calib = grid[grid.index(t_bank) + 1] if t_bank != grid[-1] else t_end
    continuation_steps = None
    continuation_noises = None
    calibration_noise = None
    if adapter.name == "plaid":
        bank_step_idx = next(
            i for i, t in enumerate(base_steps) if round(t, 6) == t_bank)
        continuation_steps = base_steps[bank_step_idx:]
        continuation_noises = base_step_noises[bank_step_idx:]
        t_next_for_calib = continuation_steps[1]
        calibration_noise = continuation_noises[0]

    # ---------- Stage 0: calibration + diversity check ----------
    n_div = min(args.diversity_n_traj, N)
    print(f"[GS16] Stage 0: calibrating eta over kappa_steps={args.kappa_steps} "
          f"using {n_div} trajectories for the diversity check")
    chosen = None
    calib_records = []
    for kappa in sorted(args.kappa_steps):
        eta, _ = calibrate_eta(adapter, z_bank_all[:n_div], sc_bank_all[:n_div],
                                t_bank, t_next_for_calib, kappa, device, args.seed,
                                paired_noise=(calibration_noise[:n_div]
                                              if calibration_noise is not None else None))
        z_final_div, _ = rollout_k_branches(adapter, z_bank_all[:n_div], sc_bank_all[:n_div],
                                             t_bank, t_end, K, eta, args.full_n_steps,
                                             device, args.seed + 1,
                                             t_steps=continuation_steps,
                                             paired_step_noises=(
                                                 [x[:n_div] for x in continuation_noises]
                                                 if continuation_noises is not None else None),
                                             solver_batch_size=args.batch_size)
        out = adapter.forward_state(z_final_div.reshape(n_div * K, L, d), None, t_end,
                                     batch_size=args.batch_size)
        toks = out["logits"].argmax(-1).reshape(n_div, K, L)
        unique_counts = []
        for n in range(n_div):
            seqs = {tuple(toks[n, k].tolist()) for k in range(K)}
            unique_counts.append(len(seqs))
        frac_ok = float(np.mean([u >= args.min_unique_endpoints for u in unique_counts]))
        calib_records.append({"kappa_step": kappa, "eta": eta,
                               "unique_counts": unique_counts, "frac_ok": frac_ok})
        print(f"  [GS16] kappa_step={kappa:g} -> eta={eta:.5f}  "
              f"unique_endpoints={unique_counts}  frac_ok={frac_ok:.2f}")
        if frac_ok >= 0.5 and chosen is None:
            chosen = (kappa, eta)
    if chosen is None:
        # fallback: use the kappa_step with the highest frac_ok (largest eta tried)
        best = max(calib_records, key=lambda r: r["frac_ok"])
        chosen = (best["kappa_step"], best["eta"])
        print(f"  [GS16] WARNING: no kappa_step reached frac_ok>=0.5 against "
              f"min_unique_endpoints={args.min_unique_endpoints}; falling back to "
              f"kappa_step={chosen[0]:g} (frac_ok={best['frac_ok']:.2f})")
    kappa_used, eta_used = chosen
    print(f"[GS16] Stage 0 done: using kappa_step={kappa_used:g}, eta={eta_used:.5f}")

    # ---------- Stage 2: full K-branch bank for all N trajectories ----------
    z_branch_end, _ = rollout_k_branches(adapter, z_bank_all, sc_bank_all, t_bank, t_end,
                                          K, eta_used, args.full_n_steps, device,
                                          args.seed + 2, t_steps=continuation_steps,
                                          paired_step_noises=continuation_noises,
                                          solver_batch_size=args.batch_size)
    out_branch = adapter.forward_state(z_branch_end.reshape(N * K, L, d), None, t_end,
                                        batch_size=args.batch_size)
    branch_toks = out_branch["logits"].argmax(-1).reshape(N, K, L)
    z_base_end = saved[t_end][0]  # (N,L,d), candidate j=0 (unperturbed endpoint)
    out_base = adapter.forward_state(z_base_end, None, t_end, batch_size=args.batch_size)
    base_toks = out_base["logits"].argmax(-1)  # (N,L)
    print("[GS16] Stage 2 done: fixed endpoint bank built for all trajectories")

    mask_full = torch.ones(L, dtype=torch.bool)  # free-running, no padding

    # dedup per trajectory: candidate 0 = self, candidates 1..M = unique branch endpoints
    banks = []  # list over n of {"z": (M+1,L,d) np, "mult": (M+1,) np, "hamming": (M+1,)}
    for n in range(N):
        self_seq = tuple(base_toks[n].tolist())
        seen = {self_seq: {"z": z_base_end[n].numpy(), "mult": 1, "source": "self"}}
        order = [self_seq]
        for k in range(K):
            seq = tuple(branch_toks[n, k].tolist())
            if seq == self_seq:
                seen[self_seq]["mult"] += 1
                continue
            if seq not in seen:
                seen[seq] = {"z": z_branch_end[n, k].numpy(), "mult": 1, "source": "branch"}
                order.append(seq)
            else:
                seen[seq]["mult"] += 1
        z_stack = np.stack([seen[s]["z"] for s in order], axis=0)
        mult = np.array([seen[s]["mult"] for s in order], dtype=np.float64)
        self_arr = np.array(self_seq)
        hamming = np.array([float((np.array(s) != self_arr).mean()) for s in order])
        banks.append({"tokens": order, "z": z_stack, "mult": mult, "hamming": hamming})

    n_unique_summary = [len(b["tokens"]) for b in banks]
    print(f"[GS16] unique endpoints per trajectory (incl. self): {n_unique_summary}")

    # ---------- Stage 3: score every checkpoint >= t_bank against fixed bank ----------
    score_ts = [t for t in grid if t >= t_bank]
    predicted_clean_by_t = {}
    for t in score_ts:
        z_t, sc_t = saved[t]
        out = adapter.forward_state(z_t, sc_t, t, batch_size=args.batch_size)
        predicted_clean_by_t[t] = out["predicted_clean"]

    per_traj_records = []
    for n in range(N):
        bank = banks[n]
        Mp1 = bank["z"].shape[0]
        R_star = np.stack([centered_residual(bank["z"][j])[1] for j in range(Mp1)], axis=0)
        R_star_shuf = np.stack(
            [np.random.RandomState(1000 + n * 97 + j).permutation(R_star[j])
             for j in range(Mp1)], axis=0)
        mu_star = np.stack([bank["z"][j].mean(axis=0) for j in range(Mp1)], axis=0)

        t_records = []
        for t in score_ts:
            z_raw = saved[t][0][n].numpy()
            z_model = predicted_clean_by_t[t][n].numpy()
            mu_raw, R_raw = centered_residual(z_raw)
            mu_model, R_model = centered_residual(z_model)

            for rep_name, R_t, mu_t in [("raw", R_raw, mu_raw), ("model", R_model, mu_model)]:
                a_cos = np.array([frobenius_cosine(R_t, R_star[j]) for j in range(Mp1)])
                a_cka = np.array([linear_cka(R_t, R_star[j]) for j in range(Mp1)])
                a_cos_shuf = np.array([frobenius_cosine(R_t, R_star_shuf[j])
                                        for j in range(Mp1)])
                a_mean_only = np.array([frobenius_cosine(np.broadcast_to(mu_t, R_t.shape),
                                                           np.broadcast_to(mu_star[j], R_t.shape))
                                         for j in range(Mp1)])

                s_self_cos = float(a_cos[0] - a_cos[1:].mean()) if Mp1 > 1 else float("nan")
                rank_self_cos = int((a_cos > a_cos[0]).sum() + 1)  # 1 = best

                rec = {
                    "traj": n, "t": t, "rep": rep_name, "n_candidates": Mp1,
                    "a_cos": a_cos.tolist(), "a_cka": a_cka.tolist(),
                    "a_cos_shuffled": a_cos_shuf.tolist(),
                    "a_mean_only": a_mean_only.tolist(),
                    "hamming": bank["hamming"].tolist(), "mult": bank["mult"].tolist(),
                    "S_self_cos": s_self_cos, "rank_self_cos": rank_self_cos,
                }
                for beta in args.betas:
                    h_raw_u, h_norm_u, neff_u = entropy_and_neff(
                        a_cos, np.ones(Mp1), beta)
                    h_raw_m, h_norm_m, neff_m = entropy_and_neff(
                        a_cos, bank["mult"], beta)
                    rec[f"H_end_unique_beta{beta:g}"] = h_norm_u
                    rec[f"N_eff_unique_beta{beta:g}"] = neff_u
                    rec[f"H_end_basinmass_beta{beta:g}"] = h_norm_m
                    rec[f"N_eff_basinmass_beta{beta:g}"] = neff_m
                t_records.append(rec)
        per_traj_records.append(t_records)
        print(f"  [GS16] traj={n} scored at {len(score_ts)} checkpoints "
              f"({Mp1} candidates)")

    # baseline-subtracted Delta a_j(t) = a_j(t) - a_j(t_bank)
    for n in range(N):
        base_by_rep = {}
        for r in per_traj_records[n]:
            if r["t"] == t_bank:
                base_by_rep[r["rep"]] = np.array(r["a_cos"])
        for r in per_traj_records[n]:
            base = base_by_rep.get(r["rep"])
            if base is not None:
                r["delta_a_cos"] = (np.array(r["a_cos"]) - base).tolist()

    # ---------- cross-trajectory endpoint null (Control 1, approximate) ----------
    rng_null = np.random.RandomState(args.seed + 3)
    null_records = []
    for n in range(N):
        other = rng_null.choice([m for m in range(N) if m != n])
        bank_null = banks[other]
        Mp1n = bank_null["z"].shape[0]
        R_star_null = np.stack([centered_residual(bank_null["z"][j])[1]
                                 for j in range(Mp1n)], axis=0)
        z_t_end = saved[t_end][0][n].numpy()
        _, R_end = centered_residual(z_t_end)
        a_null = np.array([frobenius_cosine(R_end, R_star_null[j]) for j in range(Mp1n)])
        null_records.append({"traj": n, "vs_traj": int(other),
                              "a_null_at_t_end": a_null.tolist()})

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_traj": N, "k_branches": K, "t_bank": t_bank, "t_end": t_end,
        "checkpoint_ts": grid, "score_ts": score_ts,
        "kappa_step_used": kappa_used, "eta_used": eta_used,
        "paired_plaid_solver_noise": adapter.name == "plaid",
        "calibration_records": calib_records,
        "n_unique_endpoints_per_traj": n_unique_summary,
        "records": [r for traj_recs in per_traj_records for r in traj_recs],
        "cross_traj_null": null_records,
        "notes": [
            "t=t_end is TAUTOLOGICAL for candidate j=0: R_t at t_end is literally "
            "the same state used to build R_star_0 (the trajectory's own endpoint "
            "is candidate 0 by construction), so a_cos[0]=1.0 exactly at t_end "
            "regardless of any real specificity -- same class of issue as GS15's "
            "A_linear(t_end)=1 artifact. Do not use the t=t_end point to support "
            "an 'early self-specificity' claim; treat it only as a sanity-check "
            "upper bound that the pipeline recovers the trivial answer.",
            "Only the one-step-matched-impact calibration protocol is implemented "
            "(Control 6 / terminal-linearized protocol is NOT implemented).",
            "For Plaid, the base rollout owns the ancestral solver-noise schedule. "
            "Calibration compares base/perturbed states under the same draw, and "
            "every K-way branch replays the base trajectory's exact future draws. "
            "Thus branch differences measure perturbation sensitivity rather than "
            "uncontrolled sampler noise.",
            "Cross-trajectory null (Control 1) matches by sequence length only "
            "(all trajectories share seq_len here), not terminal token entropy.",
            "a_mean_only (Control 5) broadcasts the position-mean back to (L,d) "
            "before taking the same Frobenius cosine, so it is on the same scale "
            "as a_cos.",
            "H_end/N_eff computed two ways: unique-endpoint (weights=1) and "
            "basin-mass (weights=multiplicity, Control 3) -- multiplicity enters "
            "as an additive log-count term in the softmax logits.",
            f"Pilot scale (n_traj={N}, K={K}) -- see EXP-GS16-spec.md before citing.",
        ],
    }
    json_path = out_dir / f"endpoint_specificity_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS16] Saved summary to {json_path}")

    # Save the fixed endpoint bank itself (z-vectors) so EXP-GS17 can reuse the
    # SAME candidates for its alternative-endpoint control (Section 5). Padded
    # to max_Mp1 candidates across trajectories; n_candidates_per_traj gives
    # the real count (rest is padding, mask with valid<n_candidates_per_traj).
    max_Mp1 = max(b["z"].shape[0] for b in banks)
    z_bank_padded = np.zeros((N, max_Mp1, L, d), dtype=np.float32)
    mult_padded = np.zeros((N, max_Mp1), dtype=np.float64)
    hamming_padded = np.zeros((N, max_Mp1), dtype=np.float64)
    n_cand_per_traj = np.zeros((N,), dtype=np.int64)
    for n, b in enumerate(banks):
        m = b["z"].shape[0]
        z_bank_padded[n, :m] = b["z"]
        mult_padded[n, :m] = b["mult"]
        hamming_padded[n, :m] = b["hamming"]
        n_cand_per_traj[n] = m
    npz_path = out_dir / f"endpoint_specificity_{args.label}_bank.npz"
    np.savez_compressed(
        npz_path, z_bank=z_bank_padded, mult=mult_padded, hamming=hamming_padded,
        n_candidates_per_traj=n_cand_per_traj, t_bank=t_bank, t_end=t_end,
        eta_used=eta_used, seed=args.seed, n_traj=N, k_branches=K,
    )
    print(f"[GS16] Saved fixed endpoint bank to {npz_path} "
          f"(reuse with the same --seed/--n_traj in EXP-GS17 for the "
          f"alternative-endpoint control)")


if __name__ == "__main__":
    main()
