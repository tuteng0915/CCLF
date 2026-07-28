"""EXP-PT6: Local Stability Around the Crossing (scoped implementation).

Determines whether a top-1 crossing is a robust phase transition or a
fragile ranking fluctuation, by branching the full state (z_t, sc_t) at a
few checkpoints around the crossing and perturbing it, then CONTINUING the
ODE rollout to t=1 to see whether the final decoded token changes.

SCOPE NOTE (full rationale in docs/specs/EXP-PT6-spec.md): the suite doc asks
for PER-POSITION checkpoints (each position branched around its own tau_b),
5 perturbation directions, 5 eta magnitudes, and >=5 seeds. At ELF's L=1024,
N=128 scale that is thousands of full ODE rollouts -- not tractable in one
pass. This script uses POPULATION-level checkpoints (single tau_b/tau_s
averaged across positions, taken from EXP-PT2's output) applied to the whole
batch at once, 2 of the 5 perturbation directions (isotropic random +
token-discriminative, reusing EXP-PT3's saved u_yf/random_dir where
available), 3 of the 5 eta magnitudes, and 2 branches per condition (not the
recommended >=5) -- all explicit, documented departures from the shared
protocol's minimums (docs/phase_transition_experiment_suite.md section 2).

NOT implemented: orthogonalized-random control, empirical rollout-drift
direction (needs real free-running Protocol-B trajectories -- EXP-PT7
territory), context-only / target-position-only perturbations (need
position-selective perturbation, an extra dimension of complexity).

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/branch_around_transition.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --transition_json results/phase_transition/elf/baseline/transition_failure_analysis_full.json \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 64 --n_rollout_steps 8 --label full
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

ETAS = [1e-3, 3e-3, 1e-2]
N_BRANCHES = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--transition_json", required=True,
                    help="EXP-PT2's transition_failure_analysis_<label>.json, for population tau_b/tau_s")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--delta_t", type=float, default=0.1, help="offset for tau_b-delta / tau_b+delta checkpoints")
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.98)
    p.add_argument("--n_rollout_steps", type=int, default=8)
    p.add_argument("--n_centroid_samples", type=int, default=128,
                    help="independent split for the token-discriminative direction's centroids")
    p.add_argument("--min_centroid_count", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def decode_top1(adapter, z, t, batch_size):
    out = adapter.forward_state(z.cpu(), None, t, batch_size=batch_size)
    return out["logits"].argmax(-1), out["predicted_clean"]


def build_centroid_table(token_ids, clean_emb, min_count):
    """Fully vectorized (no Python per-position loop -- see EXP-PT3-spec.md's
    performance note; fixed here). Returns (max_id+1, d) centroid table and
    a (max_id+1,) bool 'has_centroid' mask."""
    d = clean_emb.shape[-1]
    flat_ids = token_ids.reshape(-1)
    flat_emb = clean_emb.reshape(-1, d)
    max_id = int(flat_ids.max().item())
    sums = torch.zeros(max_id + 1, d)
    cnt = torch.zeros(max_id + 1)
    sums.index_add_(0, flat_ids, flat_emb)
    cnt.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))
    has_centroid = cnt >= min_count
    centroid_table = sums / cnt.clamp_min(1).unsqueeze(-1)
    return centroid_table, has_centroid


@torch.no_grad()
def rollout_to_end(adapter, z0, t_start, t_end, n_steps, batch_size, device):
    """Chunked ODE rollout from t_start to t_end. Returns final decoded token ids (N,L)."""
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
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    with open(args.transition_json) as f:
        trans = json.load(f)
    tau_b_pop = trans["tau_b_mean_finite"]
    tau_s_pop = trans["tau_s_mean_finite"]
    print(f"[PT6] Using population tau_b={tau_b_pop:.3f}, tau_s={tau_s_pop:.3f} from {args.transition_json}")

    print(f"[PT6] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        seq_len = adapter.seq_len
        ids, mask = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        x_clean = adapter.encode_clean(ids, mask).cpu()
        gt_ids = ids
        cen_ids, cen_mask = adapter.load_owt_sequences(args.n_centroid_samples, seq_len=seq_len)
        cen_emb = adapter.encode_clean(cen_ids, cen_mask).cpu()
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        seq_len = args.seq_len
        ids, mask, x_clean = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        gt_ids = ids
        cen_ids, cen_mask, cen_emb = adapter.load_owt_sequences(args.n_centroid_samples, seq_len=seq_len)

    N, L, d = x_clean.shape
    print(f"[PT6] {N} sequences, L={L}")

    # Default competitor f_i, fixed at the earliest t (matches PT1/PT2/PT3's
    # own convention), and the token-discriminative direction u_yf, built
    # via vectorized index_add/gather (no Python per-position loop -- see
    # EXP-PT3-spec.md's performance note; fixed here).
    eps_f = adapter.sample_epsilon((N, L, d))
    z_f = adapter.make_oracle_state(x_clean.to(device), eps_f, args.t_min).cpu()
    f_ids, _ = decode_top1(adapter, z_f, args.t_min, args.batch_size)
    del eps_f, z_f
    gc.collect()

    centroid_table, has_centroid = build_centroid_table(cen_ids, cen_emb, args.min_centroid_count)
    max_id = centroid_table.shape[0] - 1
    gt_clamped = gt_ids.clamp(max=max_id)
    f_clamped = f_ids.clamp(max=max_id)
    u_yf = F.normalize(centroid_table[gt_clamped] - centroid_table[f_clamped], dim=-1)
    valid_token_dir = (has_centroid[gt_clamped] & has_centroid[f_clamped] & (gt_ids != f_ids))
    frac_valid = float(valid_token_dir.float().mean())
    print(f"[PT6] token-direction valid for {frac_valid*100:.1f}% of positions "
          f"(invalid positions fall back to a random direction for that slot)")
    del cen_emb
    gc.collect()

    checkpoints = {
        "tau_b_minus": max(args.t_min, tau_b_pop - args.delta_t),
        "tau_b": tau_b_pop,
        "tau_b_plus": min(args.t_max, tau_b_pop + args.delta_t),
        "tau_s": min(args.t_max, tau_s_pop),
    }
    print(f"[PT6] Checkpoints: {checkpoints}")

    results = {}
    for ckpt_name, t_c in checkpoints.items():
        t_c = float(t_c)
        eps = adapter.sample_epsilon((N, L, d))
        z_tc = adapter.make_oracle_state(x_clean.to(device), eps, t_c).cpu()

        top1_unpert, _ = decode_top1(adapter, z_tc, t_c, args.batch_size)
        final_unpert = rollout_to_end(adapter, z_tc, t_c, args.t_max, args.n_rollout_steps,
                                       args.batch_size, device)

        z_norm = z_tc.norm(dim=-1, keepdim=True)  # (N,L,1)

        def random_direction():
            return F.normalize(torch.randn(N, L, d, generator=gen), dim=-1)

        def token_direction():
            # falls back to a fresh random direction at positions without a
            # valid u_yf (see frac_valid printed above)
            rand = random_direction()
            return torch.where(valid_token_dir.unsqueeze(-1), u_yf, rand)

        directions = {"random": random_direction, "token_direction": token_direction}

        ckpt_result = {"t": t_c, "conditions": {}}
        for dir_name, dir_fn in directions.items():
            for eta in ETAS:
                immediate_flips, final_flips = [], []
                branch_finals = []
                for b in range(N_BRANCHES):
                    u = dir_fn()
                    delta = eta * z_norm * u
                    z_pert = z_tc + delta

                    top1_pert, pred_pert = decode_top1(adapter, z_pert, t_c, args.batch_size)
                    imm_flip = (top1_pert != top1_unpert).float().mean().item()
                    immediate_flips.append(imm_flip)

                    final_pert = rollout_to_end(adapter, z_pert, t_c, args.t_max,
                                                 args.n_rollout_steps, args.batch_size, device)
                    fin_flip = (final_pert != final_unpert).float().mean().item()
                    final_flips.append(fin_flip)
                    branch_finals.append(final_pert)

                    del delta, z_pert, top1_pert, pred_pert, final_pert
                    gc.collect()

                # Modal outcome probability + branch entropy (per position,
                # over the N_BRANCHES perturbed rollouts).
                stacked = torch.stack(branch_finals, dim=0)  # (n_branches, N, L)
                modal_frac = []
                for pos_flat in stacked.permute(1, 2, 0).reshape(-1, N_BRANCHES).numpy():
                    vals, counts = np.unique(pos_flat, return_counts=True)
                    modal_frac.append(counts.max() / N_BRANCHES)
                modal_outcome_prob = float(np.mean(modal_frac))

                key = f"{dir_name}_eta{eta}"
                ckpt_result["conditions"][key] = {
                    "immediate_flip_rate": float(np.mean(immediate_flips)),
                    "final_flip_rate": float(np.mean(final_flips)),
                    "modal_outcome_probability": modal_outcome_prob,
                }
                print(f"  [{ckpt_name} t={t_c:.3f}] {key}: "
                      f"imm_flip={ckpt_result['conditions'][key]['immediate_flip_rate']:.4f}  "
                      f"final_flip={ckpt_result['conditions'][key]['final_flip_rate']:.4f}  "
                      f"modal_prob={modal_outcome_prob:.4f}")

        results[ckpt_name] = ckpt_result
        del eps, z_tc, top1_unpert, final_unpert
        gc.collect()

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "seq_len": L, "tau_b_pop": tau_b_pop, "tau_s_pop": tau_s_pop,
        "checkpoints": checkpoints, "etas": ETAS, "n_branches": N_BRANCHES,
        "n_rollout_steps": args.n_rollout_steps,
        "results": results,
        "notes": [
            "Population-level (not per-position) checkpoints; single global tau_b/tau_s applied "
            "to the whole batch at once -- see EXP-PT6-spec.md.",
            "Only 2/5 perturbation directions implemented (isotropic random, and a real "
            "token-discriminative direction u_yf built via a fresh, independently-split centroid "
            "pass -- vectorized, not the slow Python loop EXP-PT3 used).",
            "Only 3/5 eta magnitudes, 2 branches (not the shared protocol's recommended >=5 seeds).",
            "NOT implemented: orthogonalized-random control, empirical rollout-drift direction "
            "(needs EXP-PT7), context-only/target-position-only perturbations, pairwise branch "
            "agreement metric, local-gain metric.",
        ],
    }
    json_path = out_dir / f"branch_stability_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[PT6] Saved {json_path}")


if __name__ == "__main__":
    main()
