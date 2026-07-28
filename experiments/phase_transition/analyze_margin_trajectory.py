"""EXP-PT2: True-vs-Default Margin Trajectories.

Computes, on a dense oracle t-grid, the per-position/per-t quantities needed
to test whether the "commitment cliff" is a discrete decoder-boundary
crossing produced by a smoothly accumulating margin:
  m_raw(t) = ell(y) - ell(f)      (raw true-vs-default margin)
  m_res(t) = e(y) - e(f)          (prior-subtracted residual margin)
plus true-token rank, top1/top2 margin, entropy, and native top-1 identity.

Default competitor f_i: this script implements definitions 1 and 2 from the
suite doc (definition 3, "reference-prior top-1 from EXP-PT1", requires
loading a specific EXP-PT1 run's reference and is left as a follow-up --
see docs/specs/EXP-PT2-spec.md):
  f1 = earliest-time native top-1 (same convention as EXP-PT1's f_i)
  f2 = modal raw top-1 token over the first 10% of the t-grid

Reference prior for m_res: reuses the EXP-05v3 "global null" convention
(z_t_null = (1-t)*eps, zero-signal) rather than EXP-PT1's more expensive
per-channel-matched Gaussian -- PT2 runs a much denser t-grid, so the
cheaper reference keeps this tractable. EXP-05v3 already found
G_debias(null) ~= G_debias(matched-Gaussian) in aggregate, so this is a
reasonable stand-in for a first pass (see spec for caveats).

After the dense pass, also fits (at the population/mean-curve level, not
per-position -- per-position isotonic/changepoint fitting over 100k+
positions is not tractable in a first pass):
  - an isotonic (monotone) regression to mean m_res(t)
  - a brute-force best piecewise-linear fit with 1-3 breakpoints
and reports per-position tau_e / tau_b / tau_s, pre/post-crossing slope,
zero-crossing counts, and margin-zero-vs-switch distance.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/analyze_margin_trajectory.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 128 --n_t_steps 21 --label full
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from estimate_reference_prior import rank_of_gt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_steps", type=int, default=21)
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--n_null", type=int, default=4, help="noise seeds for the null reference")
    p.add_argument("--first10pct_n", type=int, default=3,
                    help="number of grid points (from t_min) treated as 'first 10%%' for f2")
    p.add_argument("--k_e", type=int, default=3, help="consecutive-hit window for tau_e (delta_e=0)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def backbone_probs(adapter, z, t, batch_size):
    out = adapter.forward_state(z.cpu(), None, t, batch_size=batch_size)
    return F.softmax(out["logits"], dim=-1)


def null_probs(adapter, x_clean, t, n_null, batch_size, N, L, d, device):
    probs_sum = None
    for _ in range(n_null):
        eps = adapter.sample_epsilon((N, L, d))
        z_null = adapter.make_null_state(eps, t)
        probs = backbone_probs(adapter, z_null, t, batch_size)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / n_null


def oracle_probs(adapter, x_clean, t, n_oracle, batch_size, N, L, d, device):
    probs_sum = None
    for _ in range(n_oracle):
        eps = adapter.sample_epsilon((N, L, d))
        z = adapter.make_oracle_state(x_clean.to(device), eps, t)
        probs = backbone_probs(adapter, z, t, batch_size)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / n_oracle


def top2_margin_and_entropy(log_p, p):
    top2 = log_p.topk(2, dim=-1).values  # (N,L,2)
    margin_top12 = (top2[..., 0] - top2[..., 1])
    entropy = -(p * log_p).sum(-1)
    return margin_top12, entropy


def isotonic_fit_quality(t_grid, y_mean):
    """Fits an isotonic (monotone non-decreasing) regression to the mean
    curve; returns (fitted_curve, R^2 vs raw mean curve)."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    fitted = ir.fit_transform(t_grid, y_mean)
    ss_res = float(np.sum((y_mean - fitted) ** 2))
    ss_tot = float(np.sum((y_mean - y_mean.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return fitted.tolist(), r2


def best_piecewise_linear(t_grid, y_mean, max_breaks=3):
    """Brute-force search over 1..max_breaks interior breakpoints (indices),
    fitting independent least-squares lines per segment, minimizing total
    SSE with a BIC-style complexity penalty (k breakpoints -> k+1 segments,
    2*(k+1) params). T is small (grid size), so brute force is fine."""
    from itertools import combinations
    T = len(t_grid)
    best = {"n_breaks": 0, "breakpoints": [], "sse": None, "bic": None}

    def fit_segments(idx_breaks):
        bounds = [0] + list(idx_breaks) + [T]
        sse = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 2:
                seg_y = y_mean[a:b]
                sse += float(np.sum((seg_y - seg_y.mean()) ** 2)) if b > a else 0.0
                continue
            x_seg, y_seg = t_grid[a:b], y_mean[a:b]
            A = np.vstack([x_seg, np.ones_like(x_seg)]).T
            coef, res, _, _ = np.linalg.lstsq(A, y_seg, rcond=None)
            pred = A @ coef
            sse += float(np.sum((y_seg - pred) ** 2))
        return sse

    candidates = list(range(2, T - 1))  # interior indices, need >=2 points per segment roughly
    results = []
    for n_breaks in range(0, max_breaks + 1):
        if n_breaks == 0:
            sse = fit_segments([])
            k_params = 2
            combo = []
            bic = T * np.log(sse / T + 1e-12) + k_params * np.log(T)
            results.append((bic, sse, combo))
            continue
        if len(candidates) < n_breaks:
            continue
        for combo in combinations(candidates, n_breaks):
            sse = fit_segments(list(combo))
            k_params = 2 * (n_breaks + 1)
            bic = T * np.log(sse / T + 1e-12) + k_params * np.log(T)
            results.append((bic, sse, list(combo)))

    results.sort(key=lambda r: r[0])
    bic, sse, combo = results[0]
    best = {
        "n_breaks": len(combo),
        "breakpoints_t": [float(t_grid[i]) for i in combo],
        "sse": sse, "bic": float(bic),
    }
    return best


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[PT2] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        if args.seq_len != adapter.seq_len:
            print(f"[PT2] NOTE: --seq_len ignored for ELF; using config.max_length={adapter.seq_len}")
        ids, mask = adapter.load_owt_sequences(args.n_samples, seq_len=adapter.seq_len)
        x_clean = adapter.encode_clean(ids, mask).cpu()
        gt_ids = ids
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        ids, mask, x_clean = adapter.load_owt_sequences(args.n_samples, seq_len=args.seq_len)
        gt_ids = ids

    N, L, d = x_clean.shape
    t_grid = np.linspace(args.t_min, args.t_max, args.n_t_steps)
    print(f"[PT2] {N} sequences, L={L}, T={len(t_grid)} t-points")

    n_oracle = 4

    # NOTE (post-mortem fix): an earlier version of this loop cached a full
    # (N,L,V) p_probs tensor per t in a dict across the WHOLE t-grid before
    # doing anything else with it ("avoid a 3rd forward pass"). At ELF scale
    # (N=128, L=1024, V=32100) a single such tensor is ~16.8GB; caching all
    # 21 grid points is ~350GB, which OOM-killed one run and forced killing
    # two others that were thrashing into swap. Fixed by (a) computing f1/f2
    # from a small early-t subset only, discarding those tensors immediately,
    # and (b) computing p_probs fresh (not cached) inside the single main
    # loop below, explicitly `del`-ing every big tensor before the next t.
    first10_idx = list(range(min(args.first10pct_n, len(t_grid))))
    early_top1 = []
    for i in first10_idx:
        t = float(t_grid[i])
        p = oracle_probs(adapter, x_clean, t, n_oracle, args.batch_size, N, L, d, device)
        early_top1.append(p.argmax(-1))
        del p
    gc.collect()
    f1 = early_top1[0]  # (N,L) -- earliest-time native top-1
    f2 = torch.mode(torch.stack(early_top1, dim=0), dim=0).values  # (N,L) modal token over first10pct_n
    del early_top1

    records = []
    for t in t_grid:
        t = float(t)
        p_probs = oracle_probs(adapter, x_clean, t, n_oracle, args.batch_size, N, L, d, device)
        log_p = torch.log(p_probs + 1e-12)
        q_probs = null_probs(adapter, x_clean, t, args.n_null, args.batch_size, N, L, d, device)
        log_q = torch.log(q_probs + 1e-12)
        e = log_p - log_q

        rank_raw = rank_of_gt(p_probs, gt_ids)
        rank_res = rank_of_gt(e, gt_ids)
        margin_top12, entropy = top2_margin_and_entropy(log_p, p_probs)
        raw_top1 = p_probs.argmax(-1)
        residual_top1 = e.argmax(-1)

        ell_gt = log_p.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)
        ell_f1 = log_p.gather(-1, f1.unsqueeze(-1)).squeeze(-1)
        ell_f2 = log_p.gather(-1, f2.unsqueeze(-1)).squeeze(-1)
        e_gt = e.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)
        e_f1 = e.gather(-1, f1.unsqueeze(-1)).squeeze(-1)
        e_f2 = e.gather(-1, f2.unsqueeze(-1)).squeeze(-1)

        records.append({
            "t": t,
            "rank_raw": rank_raw.numpy().astype(np.int32),
            "rank_res": rank_res.numpy().astype(np.int32),
            "raw_top1": raw_top1.numpy().astype(np.int32),
            "residual_top1": residual_top1.numpy().astype(np.int32),
            "margin_top12": margin_top12.numpy().astype(np.float32),
            "entropy": entropy.numpy().astype(np.float32),
            "ell_gt": ell_gt.numpy().astype(np.float32),
            "ell_f1": ell_f1.numpy().astype(np.float32),
            "ell_f2": ell_f2.numpy().astype(np.float32),
            "e_gt": e_gt.numpy().astype(np.float32),
            "e_f1": e_f1.numpy().astype(np.float32),
            "e_f2": e_f2.numpy().astype(np.float32),
        })
        print(f"  [pass] t={t:.3f} G_oracle={float((raw_top1==gt_ids).float().mean()):.4f} "
              f"m_raw(f1)={float((ell_gt-ell_f1).mean()):+.3f} "
              f"m_res(f1)={float((e_gt-e_f1).mean()):+.3f}")

        del p_probs, log_p, q_probs, log_q, e, rank_raw, rank_res, margin_top12, entropy
        del raw_top1, residual_top1, ell_gt, ell_f1, ell_f2, e_gt, e_f1, e_f2
        gc.collect()

    npz_path = out_dir / f"margin_trajectory_raw_{args.label}.npz"
    save_dict = {"t_grid": t_grid, "gt_ids": gt_ids.numpy().astype(np.int32),
                 "f1": f1.numpy().astype(np.int32), "f2": f2.numpy().astype(np.int32)}
    for i, rec in enumerate(records):
        for k, v in rec.items():
            save_dict[f"t{i}_{k}"] = v if isinstance(v, np.ndarray) else np.asarray(v)
    np.savez_compressed(npz_path, **save_dict)
    print(f"[PT2] Saved per-position arrays to {npz_path}")

    # Population-level transition analysis (mean curves)
    m_raw_mean = np.array([float((r["ell_gt"] - r["ell_f1"]).mean()) for r in records])
    m_res_mean = np.array([float((r["e_gt"] - r["e_f1"]).mean()) for r in records])

    iso_fitted, iso_r2 = isotonic_fit_quality(t_grid, m_res_mean)
    pw_fit = best_piecewise_linear(t_grid, m_res_mean, max_breaks=3)

    zero_crossings = int(np.sum(np.diff(np.sign(m_res_mean)) != 0))

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "seq_len": L, "t_grid": t_grid.tolist(),
        "m_raw_mean": m_raw_mean.tolist(), "m_res_mean": m_res_mean.tolist(),
        "isotonic_fit_r2": iso_r2, "isotonic_fitted": iso_fitted,
        "piecewise_linear_best": pw_fit,
        "population_zero_crossings_m_res": zero_crossings,
        "raw_npz": str(npz_path),
        "notes": [
            "m_res uses the EXP-05v3-style global null reference (cheap, single reference), "
            "not EXP-PT1's per-channel Gaussian -- see EXP-PT2-spec.md.",
            "Default competitor f1 = earliest-t native top-1 (matches EXP-PT1's f_i); "
            "f2 = modal top-1 over the first first10pct_n grid points. "
            "Definition 3 (EXP-PT1 reference-prior top-1) not yet wired in.",
            "isotonic/piecewise-linear fits are at the population mean-curve level, "
            "not per-position (100k+ position regressions not tractable in a first pass).",
            "Independent-probe score (doc's 3rd measurement) not implemented in this pass.",
        ],
    }
    json_path = out_dir / f"margin_trajectory_summary_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT2] isotonic R^2 (monotone fit to mean m_res(t)) = {iso_r2:.4f}")
    print(f"[PT2] best piecewise-linear: {pw_fit['n_breaks']} breakpoint(s) at t={pw_fit['breakpoints_t']}")
    print(f"[PT2] population-level zero crossings of mean m_res(t): {zero_crossings}")
    print(f"[PT2] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
