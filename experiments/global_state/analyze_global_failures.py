"""EXP-GS10: Global Failure Predictors (scoped-down pilot).

Runs N free-running generations (reusing EXP-GS7's rollout_with_checkpoints),
computes per-trajectory features (early/mid G_token, min CKA to paired
oracle, effective-rank descent, final-text repetition rate), and buckets each
trajectory into one of 3 coarse outcome labels (healthy / degenerate /
slow-incomplete) using only signals computable without external ground truth.
Reports descriptive group statistics only -- no fitted classifier, n=16 is far
too small. See docs/specs/EXP-GS10-spec.md for the scope-reduction rationale
relative to the suite doc's original 8-label / 13-feature design.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_global_failures.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_samples 16 --label pilot
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
from compare_oracle_rollout_global import rollout_with_checkpoints  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--checkpoint_ts", type=float, nargs="+",
                    default=[0.05, 0.28, 0.65, 0.85, 0.99])
    p.add_argument("--n_steps", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--degenerate_threshold", type=float, default=0.3)
    p.add_argument("--healthy_token_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def repetition_rate(token_ids, n=4):
    ids = list(token_ids)
    if len(ids) < n + 1:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS10] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)

    N, L, d = args.n_samples, adapter.seq_len, adapter.d_model
    t_start = adapter.t_eps
    print(f"[GS10] N={N}, L={L}, d={d}, checkpoints={args.checkpoint_ts}")

    eps = adapter.sample_epsilon((N, L, d))
    print("[GS10] running free-running rollouts (batched, this is real generation)...")
    saved = rollout_with_checkpoints(adapter, eps, t_start, args.checkpoint_ts,
                                      args.n_steps, device)

    t_final = round(max(args.checkpoint_ts), 6)
    x_clean_rollout = saved[t_final]
    out_final = adapter.forward_state(x_clean_rollout, None, t_final, batch_size=args.batch_size)
    y_rollout = out_final["logits"].argmax(-1)  # (N,L)

    per_traj = [{"repetition_rate": repetition_rate(y_rollout[n].tolist())} for n in range(N)]

    for t in args.checkpoint_ts:
        t_key = round(t, 6)
        z_rollout = saved[t_key]
        z_oracle = adapter.make_oracle_state(x_clean_rollout.to(device), eps, t).cpu()

        out_rollout = adapter.forward_state(z_rollout, None, t, batch_size=args.batch_size)
        pred_rollout = out_rollout["logits"].argmax(-1)

        for n in range(N):
            v_oracle = z_oracle[n].numpy()
            v_rollout = z_rollout[n].numpy()
            cka = linear_cka(v_oracle, v_rollout)
            _, s_r, _ = np.linalg.svd(v_rollout, full_matrices=False)
            r_eff_r = effective_rank(s_r)
            g_token = float((pred_rollout[n] == y_rollout[n]).float().mean())
            per_traj[n][f"cka@{t:.2f}"] = cka
            per_traj[n][f"r_eff@{t:.2f}"] = r_eff_r
            per_traj[n][f"G_token@{t:.2f}"] = g_token
        print(f"  [GS10] t={t:.3f} done")

    t0, t_mid, t_late = args.checkpoint_ts[0], args.checkpoint_ts[len(args.checkpoint_ts) // 2], \
        args.checkpoint_ts[-2]
    records = []
    for n in range(N):
        rec = {
            "traj": n,
            "repetition_rate": per_traj[n]["repetition_rate"],
            "early_G_token": per_traj[n][f"G_token@{t0:.2f}"],
            "mid_G_token": per_traj[n][f"G_token@{t_mid:.2f}"],
            "late_G_token": per_traj[n][f"G_token@{t_late:.2f}"],
            "min_CKA": min(per_traj[n][f"cka@{t:.2f}"] for t in args.checkpoint_ts),
            "r_eff_descent_rate": (per_traj[n][f"r_eff@{t0:.2f}"]
                                    - per_traj[n][f"r_eff@{t_late:.2f}"]),
        }
        if rec["repetition_rate"] > args.degenerate_threshold:
            label = "degenerate"
        elif rec["late_G_token"] > args.healthy_token_threshold:
            label = "healthy"
        else:
            label = "slow_incomplete"
        rec["label"] = label
        records.append(rec)

    print("\n[GS10] Per-trajectory labels:")
    for r in records:
        print(f"  traj={r['traj']:2d} label={r['label']:15s} "
              f"repetition={r['repetition_rate']:.3f} early_G_token={r['early_G_token']:.3f} "
              f"late_G_token={r['late_G_token']:.3f} min_CKA={r['min_CKA']:.3f}")

    print("\n[GS10] Group means:")
    feature_keys = ["early_G_token", "mid_G_token", "late_G_token", "min_CKA",
                     "r_eff_descent_rate", "repetition_rate"]
    group_stats = {}
    for label in ["healthy", "degenerate", "slow_incomplete"]:
        members = [r for r in records if r["label"] == label]
        if not members:
            group_stats[label] = {"n": 0}
            continue
        group_stats[label] = {"n": len(members)}
        for k in feature_keys:
            vals = [m[k] for m in members]
            group_stats[label][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {label} (n={len(members)}): " +
              "  ".join(f"{k}={group_stats[label][k]['mean']:.3f}" for k in feature_keys))

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "checkpoint_ts": args.checkpoint_ts,
        "degenerate_threshold": args.degenerate_threshold,
        "healthy_token_threshold": args.healthy_token_threshold,
        "records": records, "group_stats": group_stats,
        "notes": [
            "3 coarse outcome labels (healthy/degenerate/slow_incomplete), not the suite "
            "doc's 8-category taxonomy -- no external ground truth available to support "
            "finer labels, see EXP-GS10-spec.md Section 0.",
            "Descriptive group statistics only, no fitted classifier -- n=%d is far too "
            "small for multinomial logistic regression / survival analysis as the suite "
            "doc suggests." % N,
            "repetition_rate is a 4-gram repetition heuristic added for this pilot, not "
            "in the suite doc's original feature list.",
        ],
    }
    json_path = out_dir / f"global_failures_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS10] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
