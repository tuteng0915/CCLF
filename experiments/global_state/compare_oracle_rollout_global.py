"""EXP-GS7: Oracle vs Free-Running Global Alignment.

Runs genuine free-running generation (starting from pure noise, no OWT
conditioning document, full multi-step reverse ODE) and compares it against a
paired oracle path built from the SAME initial noise and the rollout's own
final output. Unlike GS1-GS9 (which all use OWT-document oracle states as a
trajectory-point proxy), this is the first GS script that actually generates.

Metrics deliberately limited to ones already validated as non-degenerate in
GS1/GS2/GS3 (CKA, effective rank, native-decode token accuracy) -- see
docs/specs/EXP-GS7-spec.md Section 2.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/compare_oracle_rollout_global.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_samples 8 --label pilot
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
from common import load_adapter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--checkpoint_ts", type=float, nargs="+",
                    default=[0.05, 0.12, 0.20, 0.28, 0.38, 0.50, 0.65, 0.85, 0.99])
    p.add_argument("--n_steps", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def rollout_with_checkpoints(adapter, z_start, t_start, checkpoint_ts, n_steps, device):
    """Full free-running reverse-ODE rollout from t_start (pure noise) to the
    max checkpoint t. Records (cpu) state at each value in checkpoint_ts.
    Returns dict {t: (L,d)-batched cpu tensor}."""
    t_end = max(checkpoint_ts)
    base_grid = np.linspace(t_start, t_end, n_steps + 1)
    merged = sorted(set(base_grid.tolist()) | set(checkpoint_ts))
    checkpoint_set = set(round(t, 6) for t in checkpoint_ts)

    z = z_start.to(device)
    sc = torch.zeros_like(z)
    saved = {}
    if round(merged[0], 6) in checkpoint_set:
        saved[round(merged[0], 6)] = z.cpu()

    for i in range(len(merged) - 1):
        t, t_next = merged[i], merged[i + 1]
        z, sc = adapter.solver_step(z, sc, t, t_next)
        if round(t_next, 6) in checkpoint_set:
            saved[round(t_next, 6)] = z.cpu()
    return saved


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS7] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)

    N, L, d = args.n_samples, adapter.seq_len, adapter.d_model
    t_start = adapter.t_eps
    print(f"[GS7] N={N}, L={L}, d={d}, t_start(t_eps)={t_start}, "
          f"checkpoints={args.checkpoint_ts}")

    eps = adapter.sample_epsilon((N, L, d))  # pure noise, no OWT document involved

    print("[GS7] running free-running rollout (this is real generation, ~1 min)...")
    saved_rollout = rollout_with_checkpoints(
        adapter, eps, t_start, args.checkpoint_ts, args.n_steps, device)

    t_final = round(max(args.checkpoint_ts), 6)
    x_clean_rollout = saved_rollout[t_final]  # (N,L,d), used as clean endpoint approximation

    out_final = adapter.forward_state(x_clean_rollout, None, t_final, batch_size=args.batch_size)
    y_rollout = out_final["logits"].argmax(-1)  # (N,L) -- the model's own "ground truth"
    print(f"[GS7] free-running generation complete; using t={t_final} endpoint as "
          f"x_clean^rollout and its native decode as y^rollout")

    records = []
    for t in args.checkpoint_ts:
        t_key = round(t, 6)
        z_rollout = saved_rollout[t_key]
        z_oracle = adapter.make_oracle_state(x_clean_rollout.to(device), eps, t).cpu()

        cka_vals, reff_oracle_vals, reff_rollout_vals = [], [], []
        for n in range(N):
            v_oracle = z_oracle[n].numpy()
            v_rollout = z_rollout[n].numpy()
            cka_vals.append(linear_cka(v_oracle, v_rollout))
            _, s_o, _ = np.linalg.svd(v_oracle, full_matrices=False)
            _, s_r, _ = np.linalg.svd(v_rollout, full_matrices=False)
            reff_oracle_vals.append(effective_rank(s_o))
            reff_rollout_vals.append(effective_rank(s_r))

        out_oracle = adapter.forward_state(z_oracle, None, t, batch_size=args.batch_size)
        out_rollout = adapter.forward_state(z_rollout, None, t, batch_size=args.batch_size)
        token_oracle = float((out_oracle["logits"].argmax(-1) == y_rollout).float().mean())
        token_rollout = float((out_rollout["logits"].argmax(-1) == y_rollout).float().mean())

        rec = {
            "t": t, "cka_oracle_vs_rollout": float(np.mean(cka_vals)),
            "r_eff_oracle": float(np.mean(reff_oracle_vals)),
            "r_eff_rollout": float(np.mean(reff_rollout_vals)),
            "G_token_oracle": token_oracle, "G_token_rollout": token_rollout,
        }
        records.append(rec)
        print(f"  [GS7] t={t:.3f}  CKA(oracle,rollout)={rec['cka_oracle_vs_rollout']:.3f}  "
              f"r_eff(oracle/rollout)={rec['r_eff_oracle']:.1f}/{rec['r_eff_rollout']:.1f}  "
              f"G_token(oracle/rollout)={token_oracle:.3f}/{token_rollout:.3f}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "seq_len": L, "checkpoint_ts": args.checkpoint_ts,
        "records": records,
        "notes": [
            "Free-running rollout starts from pure noise (eps), no OWT conditioning "
            "document -- this is genuine generation, not an oracle-state proxy.",
            "x_clean^rollout / y^rollout are the model's OWN final state / native decode, "
            "not external ground truth -- 'G_token' here measures agreement with the "
            "model's own eventual output, not corpus text.",
            "No topic/sentence-embedding alignment tested (see EXP-GS7-spec.md Section 2 "
            "for why -- GS1/GS2 found that class of metric saturates in this space).",
            "Pilot scale (n_samples=%d) -- see EXP-GS7-spec.md before citing numbers." % N,
        ],
    }
    json_path = out_dir / f"oracle_vs_rollout_global_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS7] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
