"""EXP-PT7 (part 1): Paired Oracle vs Free-Running Phase Alignment.

Tests whether free-running (Protocol B) generation failure occurs because
the sampler leaves the oracle (Protocol A) evidence-accumulation corridor.
For each generated sample: save the initial noise, generate via a REAL
free-running ODE rollout, decode+re-encode the final text as x_final, then
build a paired oracle path using the SAME initial noise:
  z_t^oracle = alpha_t * x_final + sigma_t * epsilon
so both paths share the same initial noise and the same (self-generated)
target text -- the only thing that differs is whether the state at
intermediate t comes from the real solver trajectory or from directly
noising the final answer.

This directly extends this project's existing EXP-01v3
(models/ELF-torch/experiments/probe_elf/probe_reverse_trajectory.py, DONE,
ELF-only) onto the unified FlowModelAdapter -- same core comparison
(G_reverse vs G_oracle, state distance d_t), but now runnable on LangFlow
too, and following the SAME oracle-noising convention EXP-01v3 already
established (real generation's initial noise is scaled by
config.denoiser_noise_scale for ELF; the paired oracle path re-uses that
SAME noise tensor but WITHOUT rescaling when forming z_t^oracle = t*x+(1-t)*eps
-- this asymmetry already exists in EXP-01v3 itself, not a new bug
introduced here; see docs/specs/EXP-PT7-spec.md).

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/compare_oracle_rollout.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 64 --n_gen_steps 24 --label full
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
    p.add_argument("--n_gen_steps", type=int, default=24, help="ODE steps for the free-running rollout")
    p.add_argument("--n_checkpoints", type=int, default=8, help="how many of the n_gen_steps grid points to analyze")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def free_running_rollout(adapter, N, L, d, t_eps, t_max, n_steps, batch_size, device,
                          init_noise_scale=1.0):
    """Runs a real free-running ODE rollout from pure noise, chunked over the
    batch dimension. Returns trajectory: list of {t, z (cpu)} at every step,
    plus the final decoded token ids."""
    t_steps = np.linspace(t_eps, t_max, n_steps + 1)
    all_z = [[] for _ in t_steps]
    final_ids_chunks = []

    for i in range(0, N, batch_size):
        B = min(batch_size, N - i)
        z = torch.randn(B, L, d, device=device) * init_noise_scale
        sc = None
        for k in range(n_steps):
            all_z[k].append(z.cpu())
            z, sc = adapter.solver_step(z, sc, float(t_steps[k]), float(t_steps[k + 1]))
        all_z[n_steps].append(z.cpu())
        out = adapter.forward_state(z.cpu(), sc.cpu() if sc is not None else None,
                                     float(t_steps[-1]), batch_size=B)
        final_ids_chunks.append(out["logits"].argmax(-1))
        del z, sc
        gc.collect()

    trajectory = [{"t": float(t_steps[k]), "z": torch.cat(all_z[k], dim=0)} for k in range(len(t_steps))]
    final_ids = torch.cat(final_ids_chunks, dim=0)
    return trajectory, final_ids, t_steps


@torch.no_grad()
def accuracy_and_rank(adapter, z, t, gt_ids, batch_size):
    out = adapter.forward_state(z.cpu(), None, t, batch_size=batch_size)
    probs = F.softmax(out["logits"].float(), dim=-1)
    acc = (probs.argmax(-1) == gt_ids).float().mean().item()
    rank = rank_of_gt(probs, gt_ids).float().mean().item()
    return acc, rank


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[PT7] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        seq_len = adapter.seq_len
        d = adapter.d_model
        t_eps = adapter.t_eps
        init_noise_scale = float(getattr(adapter.config, "denoiser_noise_scale", 2.0))
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        seq_len = args.seq_len
        d = adapter.d_model
        t_eps = 1e-3
        init_noise_scale = 1.0

    N, L = args.n_samples, seq_len
    print(f"[PT7] Generating {N} free-running samples, L={L}, {args.n_gen_steps} ODE steps...")
    trajectory, final_ids, t_steps = free_running_rollout(
        adapter, N, L, d, t_eps, 0.999, args.n_gen_steps, args.batch_size, device, init_noise_scale)

    # Re-encode the generated (self-produced) text as the clean target x_final.
    if args.model == "elf":
        attn_mask = torch.ones(N, L, dtype=torch.long)
        x_final = adapter.encode_clean(final_ids, attn_mask).cpu()
    else:
        x_final = adapter.encode_clean(final_ids).cpu()

    # Pick a subset of the step grid as checkpoints (evenly spaced).
    idxs = np.linspace(0, args.n_gen_steps, args.n_checkpoints).astype(int)
    idxs = sorted(set(idxs.tolist()))

    results = {"t": [], "G_reverse": [], "G_oracle": [], "rank_reverse": [], "rank_oracle": [], "dist": []}
    for k in idxs:
        t = trajectory[k]["t"]
        z_roll = trajectory[k]["z"]
        eps = torch.randn_like(x_final)
        z_oracle = float(t) * x_final + (1.0 - float(t)) * eps

        g_rev, rank_rev = accuracy_and_rank(adapter, z_roll, t, final_ids, args.batch_size)
        g_or, rank_or = accuracy_and_rank(adapter, z_oracle, t, final_ids, args.batch_size)
        dist = (z_roll - z_oracle).norm(dim=-1).mean().item()

        results["t"].append(t)
        results["G_reverse"].append(g_rev)
        results["G_oracle"].append(g_or)
        results["rank_reverse"].append(rank_rev)
        results["rank_oracle"].append(rank_or)
        results["dist"].append(dist)
        print(f"  t={t:.3f}  G_reverse={g_rev:.4f}  G_oracle={g_or:.4f}  "
              f"rank_rev={rank_rev:.1f}  rank_oracle={rank_or:.1f}  dist={dist:.3f}")

    json_path = out_dir / f"oracle_rollout_comparison_{args.label}.json"
    results["model"] = args.model
    results["checkpoint"] = args.checkpoint
    results["n_samples"] = N
    results["n_gen_steps"] = args.n_gen_steps
    results["notes"] = [
        "Extends this project's EXP-01v3 (ELF-only) onto the unified adapter framework, "
        "now runnable on LangFlow too.",
        "Oracle path re-uses z_roll's SAME initial noise tensor, but z_t^oracle = t*x+(1-t)*eps "
        "does NOT rescale by config.denoiser_noise_scale -- this asymmetry (generation starts "
        "scaled, oracle noising doesn't) already exists in EXP-01v3 itself; not a new bug.",
        "G_reverse/G_oracle/rank measured against the MODEL'S OWN generated text (final_ids), "
        "not real ground-truth OWT continuations -- matches EXP-01v3's protocol exactly.",
    ]
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PT7] Saved {json_path}")
    print(f"[PT7] Mean gap (G_oracle - G_reverse) across checkpoints: "
          f"{np.mean(np.array(results['G_oracle']) - np.array(results['G_reverse'])):.4f}")


if __name__ == "__main__":
    main()
