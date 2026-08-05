"""EXP-GS18 Part B: Common-Factor-Controlled Collective Dynamics.

EXP-GS5 found that margin-increment correlations between nearby positions
exceed a position-shuffle control, with a susceptibility peak just after the
commitment cliff -- interpreted as evidence of COLLECTIVE reorganization
rather than independent per-position decisions. That analysis used oracle
states and only one control. This experiment asks the harder question: does
the collective peak survive after removing shared sequence-level confounds
(overall document difficulty / confidence, logit scale, mean entropy, mean
margin, position index, current margin), run on TRUE FREE-RUNNING rollout
states (matching GS17), and compared against several matched null models --
not just one shuffle? See docs/specs/EXP-GS18-spec.md Part B.

Residualization chain (M0 -> M3), per spec Section B:
    M0 = raw margin increment dm
    M1 = M0 - mean_i(M0) within sequence           (removes per-sequence mean/global confidence)
    M2 = M1 residualized by position index + current margin (OLS, pooled)
    M3 = M2 residualized by per-sequence logit norm, mean entropy, mean margin (OLS)

Implementation notes / deviations from the spec:
  - M2 is residualized by POSITION and CURRENT MARGIN only. Frequency and
    POS covariates are NOT implemented: free-running trajectories have no
    ground-truth text to look up frequency/POS against, and building a
    reliable per-position frequency/POS table aligned to a free-running
    decode would require additional NLP alignment infrastructure not built
    here. This is a real, documented weakening of M2/M3 relative to the
    spec -- the "POS/frequency-stratified position shuffle" null is
    correspondingly implemented as a plain (unstratified) position shuffle.
  - Pilot: n_sequences=32 (spec formal minimum 128), 17 checkpoints (spec
    formal minimum 33), n_perm=200 per null (spec asks 1000).
  - Oracle-state comparison (the spec's "secondary comparison") is not run
    in this pass -- only true free-running rollout states.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_connected_coupling.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_traj 32 --label pilot
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
from common import load_adapter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow", "plaid"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_traj", type=int, default=32)
    p.add_argument("--n_states", type=int, default=17)
    p.add_argument("--max_distance", type=int, default=20)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def corr_at_distance(delta_m, d):
    """delta_m: (N,L) numpy (no mask needed, free-running has no padding)."""
    N, L = delta_m.shape
    if d >= L:
        return 0.0
    x = delta_m[:, :L - d].reshape(-1)
    y = delta_m[:, d:].reshape(-1)
    if x.size < 10 or x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def xi_from_matrix(M, max_distance):
    return float(sum(max(corr_at_distance(M, d), 0.0) for d in range(1, max_distance + 1)))


def ols_residualize(y, X):
    """y: (M,) target, X: (M,p) covariates (no intercept column needed, added here).
    Returns residuals (M,)."""
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    beta, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
    return y - Xb @ beta


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed + 13)

    print(f"[GS18b] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)

    N = args.n_traj
    L, d = adapter.seq_len, adapter.d_model
    t_start = adapter.t_eps
    t_end = 0.99
    n_steps = args.n_states - 1
    grid = np.linspace(t_start, t_end, args.n_states).round(6).tolist()
    print(f"[GS18b] N={N}, L={L}, {len(grid)} dense states")

    eps = adapter.sample_epsilon((N, L, d))
    saved = rollout_with_checkpoints_and_sc(adapter, eps, t_start, grid, n_steps, device)
    print("[GS18b] dense rollout done")

    logits_by_t = {}
    for t in grid:
        z_t, sc_t = saved[t]
        out = adapter.forward_state(z_t, sc_t, t, batch_size=args.batch_size)
        logits_by_t[t] = out["logits"]
    print("[GS18b] logits at all checkpoints done")

    logits_seq = torch.stack([logits_by_t[t] for t in grid], dim=0)  # (S,N,L,V)
    top1 = logits_seq.argmax(-1)  # (S,N,L)
    terminal_token = top1[-1]  # (N,L)
    logp = torch.log_softmax(logits_seq.float(), dim=-1)  # (S,N,L,V)
    S = len(grid)
    f_i = top1[0]  # (N,L), fixed default competitor (GS5/GS17 convention)
    ell_y = logp.gather(-1, terminal_token.unsqueeze(0).unsqueeze(-1).expand(S, N, L, 1)).squeeze(-1)
    ell_f = logp.gather(-1, f_i.unsqueeze(0).unsqueeze(-1).expand(S, N, L, 1)).squeeze(-1)
    margin = (ell_y - ell_f).numpy()  # (S,N,L)
    probs = torch.softmax(logits_seq.float(), dim=-1)
    entropy_pos = (-(probs * torch.log(probs + 1e-12)).sum(-1)).numpy()  # (S,N,L)
    logit_norm_pos = logits_seq.float().norm(dim=-1).numpy()  # (S,N,L)

    records = []
    pos_idx = np.tile(np.arange(L, dtype=np.float64) / L, (N, 1))  # (N,L), normalized

    for k in range(S - 1):
        t_k, t_next = grid[k], grid[k + 1]
        dm = margin[k + 1] - margin[k]  # M0, (N,L)

        m0 = dm.copy()
        m1 = m0 - m0.mean(axis=1, keepdims=True)

        # M2: residualize by position index + current margin (pooled OLS)
        y_flat = m1.reshape(-1)
        X_m2 = np.stack([pos_idx.reshape(-1), margin[k].reshape(-1)], axis=1)
        m2 = ols_residualize(y_flat, X_m2).reshape(N, L)

        # M3: residualize by per-sequence logit norm / mean entropy / mean margin
        seq_logit_norm = logit_norm_pos[k].mean(axis=1)  # (N,)
        seq_mean_entropy = entropy_pos[k].mean(axis=1)  # (N,)
        seq_mean_margin = margin[k].mean(axis=1)  # (N,)
        X_m3 = np.stack([
            np.repeat(seq_logit_norm, L), np.repeat(seq_mean_entropy, L),
            np.repeat(seq_mean_margin, L),
        ], axis=1)
        m3 = ols_residualize(m2.reshape(-1), X_m3).reshape(N, L)

        xi_m0 = xi_from_matrix(m0, args.max_distance)
        xi_m3 = xi_from_matrix(m3, args.max_distance)

        # ---- null models on M3 ----
        null_xis = {}
        for null_name in ["position_shuffle", "sequence_shuffle", "circular_shift",
                           "sign_flip", "gaussian_matched"]:
            xis = []
            for p_i in range(args.n_perm):
                if null_name == "position_shuffle":
                    m_null = np.stack([rng.permutation(m3[n]) for n in range(N)], axis=0)
                elif null_name == "sequence_shuffle":
                    m_null = np.stack([rng.permutation(m3[:, i]) for i in range(L)], axis=1)
                elif null_name == "circular_shift":
                    m_null = np.stack([np.roll(m3[n], rng.randint(1, L))
                                        for n in range(N)], axis=0)
                elif null_name == "sign_flip":
                    signs = rng.choice([-1.0, 1.0], size=m3.shape)
                    m_null = m3 * signs
                else:  # gaussian_matched
                    m_null = rng.normal(0.0, m3.std(), size=m3.shape)
                xis.append(xi_from_matrix(m_null, args.max_distance))
            null_xis[null_name] = {"mean": float(np.mean(xis)),
                                    "p95": float(np.percentile(xis, 95)),
                                    "max": float(np.max(xis))}

        exceeds_all = all(xi_m3 > null_xis[nm]["p95"] for nm in null_xis)
        records.append({
            "t": t_k, "t_next": t_next, "xi_M0": xi_m0, "xi_M3": xi_m3,
            "null_xi": null_xis, "xi_M3_exceeds_all_null_p95": exceeds_all,
        })
        print(f"  [GS18b] t={t_k:.3f}->{t_next:.3f}  xi_M0={xi_m0:.3f}  xi_M3={xi_m3:.3f}  "
              f"exceeds_all_null_p95={exceeds_all}  "
              f"(null p95 range {min(v['p95'] for v in null_xis.values()):.3f}-"
              f"{max(v['p95'] for v in null_xis.values()):.3f})")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_traj": N, "n_states": args.n_states, "max_distance": args.max_distance,
        "n_perm": args.n_perm, "grid": grid, "records": records,
        "notes": [
            "M2/M3 residualize by position index + current margin, then by "
            "per-sequence logit norm/mean entropy/mean margin -- frequency and "
            "POS covariates are NOT implemented (no ground-truth text for "
            "free-running trajectories to align against); the "
            "'POS/frequency-stratified position shuffle' null is correspondingly "
            "a plain (unstratified) position shuffle.",
            "Run on TRUE free-running rollout states only; the spec's oracle-state "
            "secondary comparison is not included in this pass.",
            f"Pilot scale (n_traj={N}, n_states={args.n_states}, n_perm={args.n_perm}) "
            "-- spec formal minimums are n_sequences>=128, >=33 checkpoints, "
            "1000 null permutations. See EXP-GS18-spec.md Part B decision rule "
            "before citing 'collective coordination' as a headline claim.",
        ],
    }
    json_path = out_dir / f"connected_coupling_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS18b] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
