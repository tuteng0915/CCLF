"""EXP-PT4: Causal Context-Source Ablation (scoped implementation).

Determines where early sample-specific evidence comes from: the target
position's own noisy signal, local context, or global/distant context.

IMPORTANT SCOPE NOTE (see docs/specs/EXP-PT4-spec.md for full rationale):
the suite doc's protocol implies per-POSITION interventions ("keep the
target position state unchanged and intervene only on other positions").
Doing that exactly needs either (a) one forward pass per target position
(computationally infeasible: L=1024 for ELF), or (b) a per-query attention
mask, which would require patching ELF's model.py (no mask support exists
for LangFlow's attention at all) -- both were judged too invasive/expensive
for this pass. Instead, this script uses a VALUE-LEVEL, single-forward-pass
proxy:

  - Pick sparse, well-separated "probe" positions (spacing > 2*max_radius so
    windows never overlap).
  - For "local_window" (radius r): positions within r of ANY probe keep
    their true noised value; everything else is replaced with the
    corresponding position's value from a DIFFERENT (randomly permuted)
    sequence -- i.e. cross-sequence content swap, same mechanism as
    EXP-PT1's Reference B. Metrics are read out ONLY at the probe positions.
  - "global_only" (radius r): the complement -- positions within r of a
    probe are swapped OUT, everything else (including distant real context)
    stays true.
  - "no_context" = local_window at r=0 (probe position itself is the only
    thing kept true).
  - "full_context" = baseline, nothing swapped, read out at the same probe
    positions for a fair comparison.
  - "within_sequence_shuffle" / "cross_sequence_swap": global versions
    (matches EXP-PT1 Reference C / Reference B exactly), read out at ALL
    positions since these don't have a "local" structure to probe.

NOT implemented: oracle-clean-context substitution, wrong-but-grammatically-
matched-context substitution (both need curated substitute text / an
external grammaticality check -- flagged as a gap, not attempted here).

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/intervene_context.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 128 --n_t_steps 8 --label full
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

from estimate_reference_prior import rank_of_gt, derangement  # noqa: E402

RADII = [0, 1, 2, 4, 8, 16]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_steps", type=int, default=8)
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--probe_spacing", type=int, default=40,
                    help="must be > 2*max(RADII) so probe windows never overlap")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_keep_mask(L, probes, radius, invert=False):
    """bool (L,): True = keep the position's true value. invert=True gives
    the 'global_only' complement (corrupt near probes, keep distant)."""
    near = np.zeros(L, dtype=bool)
    for p in probes:
        lo, hi = max(0, p - radius), min(L, p + radius + 1)
        near[lo:hi] = True
    return (~near) if invert else near


@torch.no_grad()
def forward_logits(adapter, z, t, batch_size):
    out = adapter.forward_state(z.cpu(), None, t, batch_size=batch_size)
    return out["logits"]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    max_r = max(RADII)
    assert args.probe_spacing > 2 * max_r, "probe_spacing must exceed 2*max(RADII) to avoid window overlap"

    print(f"[PT4] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        seq_len = adapter.seq_len
        if args.seq_len != seq_len:
            print(f"[PT4] NOTE: --seq_len ignored for ELF; using config.max_length={seq_len}")
        ids, mask = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        x_clean = adapter.encode_clean(ids, mask).cpu()
        gt_ids = ids
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        seq_len = args.seq_len
        ids, mask, x_clean = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        gt_ids = ids

    N, L, d = x_clean.shape
    t_grid = np.linspace(args.t_min, args.t_max, args.n_t_steps)
    probes = list(range(args.probe_spacing // 2, L, args.probe_spacing))
    print(f"[PT4] {N} sequences, L={L}, T={len(t_grid)} t-points, {len(probes)} probe positions")

    results = {"t": [], "conditions": {}}
    cond_names = (["full_context"]
                  + [f"local_window_r{r}" for r in RADII]
                  + [f"global_only_r{r}" for r in RADII]
                  + ["within_sequence_shuffle", "cross_sequence_swap"])
    for c in cond_names:
        results["conditions"][c] = {"G_probe": [], "rank_probe_mean": [], "acc_per_seq": []}

    probe_idx_t = torch.tensor(probes, dtype=torch.long)

    for t in t_grid:
        t = float(t)
        eps = adapter.sample_epsilon((N, L, d))
        z_real = adapter.make_oracle_state(x_clean.to(device), eps, t).cpu()
        perm = derangement(N, gen)
        z_swapped_full = z_real[perm]  # cross-sequence swap source

        def eval_condition(z_input, idx_subset):
            logits = forward_logits(adapter, z_input, t, args.batch_size)
            logits_sub = logits[:, idx_subset, :]
            gt_sub = gt_ids[:, idx_subset]
            probs = F.softmax(logits_sub.float(), dim=-1)
            rank = rank_of_gt(probs, gt_sub)
            correct = (probs.argmax(-1) == gt_sub).float()
            acc_per_seq = correct.mean(dim=1)  # (N,) -- per-SEQUENCE accuracy over its probe positions
            return acc_per_seq.mean().item(), rank.float().mean().item(), acc_per_seq.tolist()

        def record(name, acc, rk, acc_per_seq):
            results["conditions"][name]["G_probe"].append(acc)
            results["conditions"][name]["rank_probe_mean"].append(rk)
            results["conditions"][name]["acc_per_seq"].append(acc_per_seq)

        # full_context: no intervention, read out at probe positions only
        record("full_context", *eval_condition(z_real, probe_idx_t))

        for r in RADII:
            keep_local = build_keep_mask(L, probes, r, invert=False)
            keep_local_t = torch.from_numpy(keep_local)
            z_local = torch.where(keep_local_t[None, :, None], z_real, z_swapped_full)
            record(f"local_window_r{r}", *eval_condition(z_local, probe_idx_t))

            keep_distant = build_keep_mask(L, probes, r, invert=True)
            keep_distant_t = torch.from_numpy(keep_distant)
            z_global_only = torch.where(keep_distant_t[None, :, None], z_real, z_swapped_full)
            record(f"global_only_r{r}", *eval_condition(z_global_only, probe_idx_t))

        # Global conditions -- read out at ALL positions, matches EXP-PT1's
        # Reference C / Reference B exactly (same derangement/shuffle recipe).
        perm_l = derangement(L, gen)
        z_shuffled = z_real[:, perm_l, :]
        record("within_sequence_shuffle", *eval_condition(z_shuffled, torch.arange(L)))
        record("cross_sequence_swap", *eval_condition(z_swapped_full, torch.arange(L)))

        results["t"].append(t)
        print(f"  t={t:.3f}  full={results['conditions']['full_context']['G_probe'][-1]:.4f}  "
              + "  ".join(f"local_r{r}={results['conditions'][f'local_window_r{r}']['G_probe'][-1]:.4f}"
                          for r in [0, 4, 16]))

        del eps, z_real, z_swapped_full, z_shuffled
        gc.collect()

    json_path = out_dir / f"context_ablation_{args.label}.json"
    results["model"] = args.model
    results["checkpoint"] = args.checkpoint
    results["n_samples"] = N
    results["n_probes"] = len(probes)
    results["radii"] = RADII
    results["notes"] = [
        "Value-level proxy, not true per-position attention masking -- see script docstring.",
        "local_window_r / global_only_r metrics are read out ONLY at sparse, non-overlapping "
        "probe positions (spacing > 2*max_radius); within_sequence_shuffle / cross_sequence_swap "
        "are read out at ALL positions (global conditions, no probe structure needed).",
        "Oracle-clean-context and wrong-grammatically-matched-context substitutions "
        "(doc conditions 7-8) are NOT implemented.",
        "acc_per_seq (added in the rigor-audit pass) holds one accuracy value per SEQUENCE per "
        "t per condition -- resample this with bootstrap_utils.bootstrap_ci() for sequence-level "
        "CIs on any G_probe number; see EXP-PT-rigor-audit.md.",
    ]
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PT4] Saved {json_path}")
    print("[PT4] Summary at final t: G_probe by condition:")
    for c in cond_names:
        print(f"  {c:>24}: {results['conditions'][c]['G_probe'][-1]:.4f}")


if __name__ == "__main__":
    main()
