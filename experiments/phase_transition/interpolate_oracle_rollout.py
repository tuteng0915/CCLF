"""EXP-PT7 (part 2): Causal interpolation between free-running rollout and
paired oracle states.

The one genuinely CAUSAL test in EXP-PT7 (part 1, compare_oracle_rollout.py,
is purely descriptive/correlational -- it just measures the gap). At a chosen
intervention time t_intervene, constructs
    z(lambda) = (1-lambda)*z_roll + lambda*z_oracle,  lambda in {0,0.25,0.5,0.75,1}
using the SAME free-running trajectory and SAME paired-oracle construction as
compare_oracle_rollout.py, then CONTINUES the solver from t_intervene to
t_max with the interpolated state as the new starting point. Because both
adapters' solver_step is a deterministic ODE (Euler) integrator (no
stochastic noise injected mid-rollout), "matched future randomness" (doc's
requirement) is automatically satisfied -- there is no future randomness to
match.

Quality metric: since PT7's target (y_final, from compare_oracle_rollout.py)
is the free-running trajectory's OWN final decode (there's no external
ground truth -- matches EXP-01v3's protocol), "does moving toward oracle
improve quality" is operationalized as: does the continued-from-lambda
trajectory's final decode agree MORE with y_final as lambda increases. This
is somewhat definitionally true as lambda->1 (that IS y_final's own oracle
encoding), so the informative part is whether intermediate lambda (0.25, 0.5,
0.75) already shows a meaningful, roughly monotonic improvement -- if so,
that supports "the oracle-rollout gap is causally relevant, not just
descriptive" (doc's decision rule); if quality only jumps at lambda=1, the
gap is more consistent with lambda<1 states not actually being "on the path"
to a better outcome.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/interpolate_oracle_rollout.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 32 --n_gen_steps 32 --t_intervene 0.4 --n_continue_steps 12 --label full
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

from compare_oracle_rollout import free_running_rollout  # noqa: E402

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_gen_steps", type=int, default=32, help="ODE steps for the initial free-running rollout")
    p.add_argument("--t_intervene", type=float, default=0.4,
                    help="t at which to interpolate and branch off (nearest grid point used)")
    p.add_argument("--n_continue_steps", type=int, default=12, help="ODE steps for the post-interpolation continuation")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def continue_rollout(adapter, z0, t_start, t_end, n_steps, batch_size, device):
    """Chunked continuation from an arbitrary (possibly interpolated) state.
    sc_state is reset to None at the branch point -- we don't have a
    meaningful self-cond state for an interpolated/oracle state, matching
    every other oracle-probe script's zero-SC convention in this suite."""
    t_steps = np.linspace(t_start, t_end, n_steps + 1)
    N = z0.shape[0]
    final_chunks = []
    for i in range(0, N, batch_size):
        z = z0[i:i + batch_size].to(device)
        sc = None
        for k in range(n_steps):
            z, sc = adapter.solver_step(z, sc, float(t_steps[k]), float(t_steps[k + 1]))
        out = adapter.forward_state(z.cpu(), sc.cpu() if sc is not None else None,
                                     float(t_steps[-1]), batch_size=z.shape[0])
        final_chunks.append(out["logits"].argmax(-1))
        del z, sc
        gc.collect()
    return torch.cat(final_chunks, dim=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[PT7-interp] Loading {args.model} model...")
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
    print(f"[PT7-interp] Generating {N} free-running samples, L={L}, {args.n_gen_steps} ODE steps...")
    trajectory, final_ids, t_steps = free_running_rollout(
        adapter, N, L, d, t_eps, 0.999, args.n_gen_steps, args.batch_size, device, init_noise_scale)

    if args.model == "elf":
        attn_mask = torch.ones(N, L, dtype=torch.long)
        x_final = adapter.encode_clean(final_ids, attn_mask).cpu()
    else:
        x_final = adapter.encode_clean(final_ids).cpu()

    # Nearest grid point to t_intervene
    k_intervene = int(np.argmin(np.abs(t_steps - args.t_intervene)))
    t_int = float(t_steps[k_intervene])
    z_roll = trajectory[k_intervene]["z"]
    print(f"[PT7-interp] Branching at t={t_int:.3f} (step {k_intervene}/{args.n_gen_steps})")

    eps = torch.randn_like(x_final)
    z_oracle = t_int * x_final + (1.0 - t_int) * eps

    # Reference: lambda=0 continuation IS just "keep free-running" -- sanity
    # check that it agrees with the original trajectory's own final_ids
    # (should be ~100% by construction, modulo any stochastic elements).
    results = {"lambdas": LAMBDAS, "t_intervene": t_int, "agreement_with_final_ids": [],
               "mean_dist_to_oracle_target_at_branch": []}

    for lam in LAMBDAS:
        z_lambda = (1.0 - lam) * z_roll + lam * z_oracle
        final_continued = continue_rollout(adapter, z_lambda, t_int, 0.999, args.n_continue_steps,
                                            args.batch_size, device)
        agreement = (final_continued == final_ids).float().mean().item()
        dist = (z_lambda - x_final).norm(dim=-1).mean().item()
        results["agreement_with_final_ids"].append(agreement)
        results["mean_dist_to_oracle_target_at_branch"].append(dist)
        print(f"  lambda={lam:.2f}  agreement_with_original_final={agreement:.4f}  "
              f"dist(z_lambda, x_final)={dist:.3f}")
        del z_lambda, final_continued
        gc.collect()

    # Monotonicity check: is agreement non-decreasing in lambda? (the core
    # signal for "moving toward oracle causally helps", per doc's decision rule)
    agr = np.array(results["agreement_with_final_ids"])
    n_monotone_steps = int(np.sum(np.diff(agr) >= -1e-6))
    results["is_monotone_nondecreasing"] = bool(n_monotone_steps == len(agr) - 1)
    results["agreement_gain_lambda_0_to_0.5"] = float(agr[2] - agr[0]) if len(agr) > 2 else None
    results["agreement_gain_lambda_0.5_to_1"] = float(agr[-1] - agr[2]) if len(agr) > 2 else None

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "n_gen_steps": args.n_gen_steps, "n_continue_steps": args.n_continue_steps,
        **results,
        "notes": [
            "Deterministic ODE solver -> 'matched future randomness' (doc's requirement) is "
            "automatically satisfied; there is no stochastic noise to match across lambda.",
            "Quality = agreement with the ORIGINAL free-running trajectory's own final decode "
            "(y_final) -- there is no external ground truth (matches EXP-01v3/PT7-part-1's "
            "protocol). This makes lambda=1 near-tautologically high agreement (z_oracle is "
            "built FROM y_final); the informative signal is whether lambda=0.25/0.5/0.75 "
            "already show a meaningful, roughly monotonic gain over lambda=0.",
            "sc_state is reset to None (zero) at the branch point for all lambda, including "
            "lambda=0 -- so even the lambda=0 'pure continuation' arm is not a perfect replay "
            "of the original single-pass rollout (which carried a real self-cond state through "
            "the branch point); this is a source of noise in the lambda=0 agreement number, "
            "not a bug, but worth flagging when interpreting small effects near lambda=0.",
        ],
    }
    json_path = out_dir / f"oracle_rollout_interpolation_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT7-interp] monotone_nondecreasing={results['is_monotone_nondecreasing']}  "
          f"gain[0->0.5]={results['agreement_gain_lambda_0_to_0.5']:.4f}  "
          f"gain[0.5->1]={results['agreement_gain_lambda_0.5_to_1']:.4f}")
    print(f"[PT7-interp] Saved {json_path}")


if __name__ == "__main__":
    main()
