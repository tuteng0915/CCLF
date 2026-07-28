"""EXP-GS5: Collective Coupling and Correlation Length.

Tests whether the commitment transition is collective (positions' margin
increments are spatially correlated) rather than a simple average of
independent per-position crossings. Computes C_t(d) (correlation of
Delta-margin between positions at distance d), correlation length xi(t), and
global susceptibility chi(t), with a shuffle-position control. See
docs/specs/EXP-GS5-spec.md for the operationalization.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_collective_coupling.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_samples 64 --label pilot
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

from common import load_adapter, load_owt_docs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.85)
    p.add_argument("--n_t_steps", type=int, default=15)
    p.add_argument("--max_distance", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def get_logp(adapter, z, t, batch_size):
    out = adapter.forward_state(z, None, t, batch_size=batch_size)
    return torch.log_softmax(out["logits"].float(), dim=-1)


def corr_at_distance(delta_m, mask, d):
    """delta_m: (N,L) numpy, mask: (N,L) bool numpy. Pearson corr between
    delta_m[:, i] and delta_m[:, i+d] pooled over all valid (seq,i) pairs."""
    N, L = delta_m.shape
    if d >= L:
        return 0.0
    valid = mask[:, :L - d] & mask[:, d:]
    x = delta_m[:, :L - d][valid]
    y = delta_m[:, d:][valid]
    if x.size < 10 or x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    print(f"[GS5] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    ids, mask, x_clean, _ = load_owt_docs(adapter, args.model, args.n_samples)
    gt_ids = ids

    N, L, d = x_clean.shape
    t_grid = np.linspace(args.t_min, args.t_max, args.n_t_steps)
    print(f"[GS5] {N} sequences, L={L}, {len(t_grid)} t-points, max_distance={args.max_distance}")

    eps = adapter.sample_epsilon((N, L, d))
    mask_np = mask.numpy().astype(bool)

    # f_i fixed at the first t (PT1/PT2/GS2/GS8 convention)
    z0 = adapter.make_oracle_state(x_clean.to(device), eps, t_grid[0]).cpu()
    log_p0 = get_logp(adapter, z0, t_grid[0], args.batch_size)
    f_i = log_p0.argmax(-1)  # (N,L)

    m_traj = []  # list of (N,L) numpy margin arrays, one per t
    for t in t_grid:
        t = float(t)
        z_t = adapter.make_oracle_state(x_clean.to(device), eps, t).cpu()
        log_p = get_logp(adapter, z_t, t, args.batch_size)
        ell_y = log_p.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)
        ell_f = log_p.gather(-1, f_i.unsqueeze(-1)).squeeze(-1)
        m = (ell_y - ell_f).numpy()
        m_traj.append(m)
        bar_m = m[mask_np].mean() if mask_np.any() else float("nan")
        print(f"  [GS5] t={t:.3f}  mean(m)={bar_m:+.3f}")

    records = []
    for k in range(len(t_grid) - 1):
        delta_m = m_traj[k + 1] - m_traj[k]  # (N,L)

        # shuffle-position control: permute position order independently per sequence
        delta_m_shuf = np.empty_like(delta_m)
        mask_shuf = np.empty_like(mask_np)
        for n in range(N):
            perm = rng.permutation(L)
            delta_m_shuf[n] = delta_m[n][perm]
            mask_shuf[n] = mask_np[n][perm]

        C_d, C_d_shuf = [], []
        for dist in range(1, args.max_distance + 1):
            C_d.append(corr_at_distance(delta_m, mask_np, dist))
            C_d_shuf.append(corr_at_distance(delta_m_shuf, mask_shuf, dist))
        xi = float(np.sum(np.maximum(C_d, 0)))
        xi_shuf = float(np.sum(np.maximum(C_d_shuf, 0)))

        bar_m_per_seq = np.array([m_traj[k][n][mask_np[n]].mean() for n in range(N)])
        chi = float(L * np.var(bar_m_per_seq))

        records.append({
            "t": float(t_grid[k]), "t_next": float(t_grid[k + 1]),
            "xi": xi, "xi_shuffled": xi_shuf, "chi": chi,
            "C_d": C_d, "C_d_shuffled": C_d_shuf,
        })
        print(f"  [GS5] t={t_grid[k]:.3f}->{t_grid[k+1]:.3f}  xi={xi:.3f} "
              f"(shuffled={xi_shuf:.3f})  chi={chi:.3f}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "seq_len": L, "t_grid": t_grid.tolist(),
        "max_distance": args.max_distance, "records": records,
        "notes": [
            "f_i fixed at the first t (PT1/PT2/GS2/GS8 convention), not updated per t.",
            "t_grid has 15 points, not the suite doc's suggested 101-point dense grid.",
            "Only shuffle-position control implemented (not shuffle-sequence / "
            "frequency-matched / function-word-removed / independent-position-denoiser).",
            "Pilot scale (n_samples=%d) -- see EXP-GS5-spec.md before citing numbers." % N,
        ],
    }
    json_path = out_dir / f"collective_coupling_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS5] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
