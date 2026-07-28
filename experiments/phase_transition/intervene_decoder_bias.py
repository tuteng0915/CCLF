"""EXP-PT5: Decoder-Bias Intervention (readout-only diagnostic).

Causally tests whether the visible top-1 "commitment cliff" is mostly a
decoder-boundary phenomenon (small logit corrections move tau_b a lot while
representation-level evidence tau_e stays fixed) versus a representation-
formation bottleneck (crossing time barely moves under small corrections).

Two interventions, both applied ONLY to the readout logits -- never fed back
into the trajectory (no re-simulation), per the suite doc section 7:

  A. Prior-debiasing sweep:  ell'_t(v) = ell_t(v) - lambda * log q_t(v)
     for lambda in {0, 0.25, 0.5, 0.75, 1.0}. lambda=0 is the untouched raw
     decode; lambda=1 is the same full prior-subtraction used to define
     tau_e in EXP-PT1/PT2 (reuses the EXP-05v3-style null reference q_t,
     same choice as EXP-PT2 for consistency and compute cost).

  B. Additive offset sweep: ell'_t(y) = ell_t(y) + beta,
     ell'_t(f) = ell_t(f) - beta, all other logits untouched, for small beta.
     Reports the fraction of not-yet-correct positions that flip to
     correct at each beta -- i.e. how close the raw decode already is to
     the boundary in log-space.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/intervene_decoder_bias.py \\
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

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from analyze_margin_trajectory import null_probs, oracle_probs  # noqa: E402
from estimate_reference_prior import rank_of_gt  # noqa: E402

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BETAS = [0.5, 1.0, 2.0, 4.0, 8.0]


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
    p.add_argument("--n_oracle", type=int, default=4)
    p.add_argument("--n_null", type=int, default=4)
    p.add_argument("--k_e", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def first_stable_run_time(bool_over_t, t_grid, k):
    T = bool_over_t.shape[0]
    if T < k:
        return np.full(bool_over_t.shape[1:], np.inf)
    result = np.full(bool_over_t.shape[1:], np.inf)
    for start in range(T - k + 1):
        window_all = bool_over_t[start:start + k].all(axis=0)
        newly = window_all & (result == np.inf)
        result = np.where(newly, t_grid[start], result)
    return result


def first_true_time(bool_over_t, t_grid):
    T = bool_over_t.shape[0]
    result = np.full(bool_over_t.shape[1:], np.inf)
    for k in range(T - 1, -1, -1):
        result = np.where(bool_over_t[k], t_grid[k], result)
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[PT5] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        if args.seq_len != adapter.seq_len:
            print(f"[PT5] NOTE: --seq_len ignored for ELF; using config.max_length={adapter.seq_len}")
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
    T = len(t_grid)
    print(f"[PT5] {N} sequences, L={L}, T={T} t-points")

    # correct_lambda[li, ti] : (N,L) bool -- whether gt is argmax of
    # (log_p - lambda*log_q) at t_grid[ti]. correct_beta[bi, ti] similarly
    # for the additive-offset intervention (whether beta is enough to flip
    # gt into the argmax spot, given the *untouched* other logits).
    correct_lambda = np.zeros((len(LAMBDAS), T, N, L), dtype=bool)
    correct_beta = np.zeros((len(BETAS), T, N, L), dtype=bool)
    raw_rank_over_t = np.zeros((T, N, L), dtype=np.int32)
    m_res_over_t = np.zeros((T, N, L), dtype=np.float32)  # for tau_e (lambda=1 definition)

    for ti, t in enumerate(t_grid):
        t = float(t)
        p_probs = oracle_probs(adapter, x_clean, t, args.n_oracle, args.batch_size, N, L, d, device)
        q_probs = null_probs(adapter, x_clean, t, args.n_null, args.batch_size, N, L, d, device)
        log_p = torch.log(p_probs + 1e-12)
        log_q = torch.log(q_probs + 1e-12)

        rank_raw = rank_of_gt(p_probs, gt_ids)
        raw_rank_over_t[ti] = rank_raw.numpy()

        for li, lam in enumerate(LAMBDAS):
            e_lambda = log_p - lam * log_q
            argmax_lambda = e_lambda.argmax(-1)
            correct_lambda[li, ti] = (argmax_lambda == gt_ids).numpy()

        # m_res(t) against f1 (earliest-t native top-1), needed for tau_e.
        if ti == 0:
            f1 = p_probs.argmax(-1)  # (N,L), fixed default competitor
        e1 = log_p - 1.0 * log_q
        e_gt = e1.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)
        e_f1 = e1.gather(-1, f1.unsqueeze(-1)).squeeze(-1)
        m_res_over_t[ti] = (e_gt - e_f1).numpy()

        # Additive offset sweep: ell'(y)=ell(y)+beta, ell'(f1)=ell(f1)-beta,
        # all other entries untouched. gt becomes argmax iff
        # ell(y)+beta > max(ell(f1)-beta, max_other) where max_other is the
        # max over V \ {y, f1}.
        log_p_masked = log_p.clone()
        idx_y = gt_ids.unsqueeze(-1)
        idx_f = f1.unsqueeze(-1)
        log_p_masked.scatter_(-1, idx_y, float("-inf"))
        log_p_masked.scatter_(-1, idx_f, float("-inf"))
        max_other = log_p_masked.max(-1).values  # (N,L)
        ell_y = log_p.gather(-1, idx_y).squeeze(-1)
        ell_f = log_p.gather(-1, idx_f).squeeze(-1)
        for bi, beta in enumerate(BETAS):
            new_y = ell_y + beta
            new_f = ell_f - beta
            correct_beta[bi, ti] = (new_y > torch.maximum(new_f, max_other)).numpy()

        print(f"  t={t:.3f}  G_raw={float((rank_raw==0).float().mean()):.4f}  "
              + "  ".join(f"G(lam={lam})={correct_lambda[li,ti].mean():.4f}"
                          for li, lam in enumerate(LAMBDAS)))

        # Explicit cleanup: at ELF scale (N=128,L=1024,V=32100) each of these
        # is up to ~16.8GB; several are alive simultaneously per iteration
        # (see EXP-PT2-spec.md for the analogous leak this class of bug caused
        # in analyze_margin_trajectory.py). Not strictly required by Python's
        # refcounting here (no cross-t caching), but cheap insurance against
        # allocator fragmentation over a 21-point loop.
        del p_probs, q_probs, log_p, log_q, rank_raw, e1, e_gt, e_f1
        del log_p_masked, max_other, ell_y, ell_f
        gc.collect()

    # tau_b(lambda): first-hit time of correct_lambda[li]
    tau_b_lambda = {}
    for li, lam in enumerate(LAMBDAS):
        tau_b_lambda[lam] = first_true_time(correct_lambda[li], t_grid)

    # tau_e via lambda=1 (K_e-consecutive positive m_res, delta_e=0)
    emerged = m_res_over_t > 0.0
    tau_e = first_stable_run_time(emerged, t_grid, args.k_e)

    tau_b0 = tau_b_lambda[0.0]
    shift_stats = {}
    for lam in LAMBDAS:
        tb = tau_b_lambda[lam]
        both_finite = np.isfinite(tb) & np.isfinite(tau_b0)
        with np.errstate(invalid="ignore"):
            mean_shift = float(np.nanmean(np.where(both_finite, tau_b0 - tb, np.nan)))
        frac_changed = float(np.mean((tb != tau_b0) & (np.isfinite(tb) | np.isfinite(tau_b0))))
        frac_finite = float(np.mean(np.isfinite(tb)))
        shift_stats[lam] = {
            "mean_shift_vs_lambda0": mean_shift,  # positive = tau_b got earlier
            "frac_positions_with_different_tau_b": frac_changed,
            "frac_tau_b_finite": frac_finite,
        }

    # "positions whose crossing time changes without a change in residual
    # evidence": compare tau_b(lambda=0) -> tau_b(lambda=1) shift against
    # whether tau_e itself also shifted (tau_e is lambda=1-based already, so
    # we report how many positions' tau_b(1) == tau_e, i.e. become visible
    # exactly when representation-level evidence emerges, vs how many
    # positions still lag (tau_b(1) > tau_e) even after full debiasing).
    both = np.isfinite(tau_e) & np.isfinite(tau_b_lambda[1.0])
    with np.errstate(invalid="ignore"):
        residual_lag = np.where(both, tau_b_lambda[1.0] - tau_e, np.nan)
    frac_boundary_only = float(np.mean((tau_b0 > tau_e) & np.isfinite(tau_e) & (tau_b_lambda[1.0] <= tau_e + 1e-9)))

    wrong_raw = (raw_rank_over_t != 0)  # (T,N,L), reused below for per-seq bootstrap data
    beta_flip_rates = {}
    for bi, beta in enumerate(BETAS):
        # fraction of (t, position) cells where raw was wrong but beta-offset flips it right
        newly_correct = correct_beta[bi] & wrong_raw
        beta_flip_rates[beta] = float(newly_correct.sum() / max(1, wrong_raw.sum()))

    # Save per-position/per-sequence raw arrays for post-hoc sequence-level
    # bootstrap CI (rigor-audit follow-up -- this script previously only wrote
    # aggregate scalars to the JSON, see EXP-PT-rigor-audit.md "remaining work").
    npz_path = out_dir / f"decoder_bias_raw_{args.label}.npz"
    np.savez_compressed(
        npz_path,
        tau_e=tau_e.astype(np.float32),
        tau_b_lambda0=tau_b0.astype(np.float32),
        tau_b_lambda1=tau_b_lambda[1.0].astype(np.float32),
        correct_beta=correct_beta,  # (n_betas, T, N, L) bool
        wrong_raw=wrong_raw,  # (T, N, L) bool
        betas=np.asarray(BETAS),
        t_grid=t_grid,
    )
    print(f"[PT5] Saved raw per-position arrays to {npz_path}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "seq_len": L, "t_grid": t_grid.tolist(),
        "lambdas": LAMBDAS, "betas": BETAS,
        "tau_b_shift_by_lambda": shift_stats,
        "tau_e_mean_finite": float(np.nanmean(np.where(np.isfinite(tau_e), tau_e, np.nan))),
        "tau_b_lambda0_mean_finite": float(np.nanmean(np.where(np.isfinite(tau_b0), tau_b0, np.nan))),
        "tau_b_lambda1_mean_finite": float(np.nanmean(np.where(
            np.isfinite(tau_b_lambda[1.0]), tau_b_lambda[1.0], np.nan))),
        "mean_residual_lag_tau_b1_minus_tau_e": float(np.nanmean(residual_lag)),
        "frac_positions_boundary_explained": frac_boundary_only,
        "beta_flip_rate": beta_flip_rates,
        "notes": [
            "Readout-only diagnostic: intervened logits are NEVER fed back into "
            "the trajectory/solver -- matches suite doc section 7.",
            "q_t reference reuses the EXP-05v3-style global null (same choice as EXP-PT2), "
            "not EXP-PT1's more expensive per-channel Gaussian.",
            "'No change in independent-probe accuracy / hidden states / velocity' (doc's other "
            "two measurements) are trivially true here since this script never runs anything "
            "except a logit post-hoc transform -- not separately verified/logged.",
        ],
    }
    json_path = out_dir / f"decoder_bias_intervention_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT5] tau_e mean={summary['tau_e_mean_finite']:.3f}  "
          f"tau_b(lambda=0) mean={summary['tau_b_lambda0_mean_finite']:.3f}  "
          f"tau_b(lambda=1) mean={summary['tau_b_lambda1_mean_finite']:.3f}")
    for lam in LAMBDAS:
        s = shift_stats[lam]
        print(f"  lambda={lam:.2f}: mean_shift_vs_raw={s['mean_shift_vs_lambda0']:+.3f}  "
              f"frac_changed={s['frac_positions_with_different_tau_b']:.3f}")
    print(f"[PT5] frac positions where tau_b(0)>tau_e but tau_b(1)<=tau_e "
          f"('boundary-explained'): {frac_boundary_only:.4f}")
    for beta in BETAS:
        print(f"  beta={beta}: flip_rate(wrong->correct)={beta_flip_rates[beta]:.4f}")
    print(f"[PT5] Saved {json_path}")


if __name__ == "__main__":
    main()
