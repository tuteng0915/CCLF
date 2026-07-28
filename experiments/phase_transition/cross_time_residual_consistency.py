"""EXP-PT9 supplement (2): temporal consistency of the "prior-subtracted
logits" representation -- the suite doc's 4th state representation, which
probe_cross_time_transfer_extra_reps.py explicitly declined to implement as
a trained probe (see that script's docstring: a residual logit already lives
in vocab-size space and IS already a full per-class score, so training a new
Linear(vocab_size, vocab_size) probe on top of it is both intractable
(~1B+ params at this sample size) and conceptually redundant).

This script does NOT retract that decision -- training such a probe remains
a bad idea. Instead it answers a related but DIFFERENT, tractable question
using the same residual/debiased signal: is the SET OF POSITIONS where the
debiased prediction e_t(v) = log p_oracle(v|z_t) - log q_null(v|t) is
correct stable across time, or does correctness come and go independently
at each t? This reuses PT2/PT5's exact null-reference machinery
(oracle_probs/null_probs from analyze_margin_trajectory.py, same
EXP-05v3-style global null used throughout this suite for consistency and
compute cost) -- no probe training, no new GPU-heavy machinery.

IMPORTANT CAVEAT (read before comparing to probe_cross_time_transfer.py's
diag/upper/lower numbers): this is NOT the same operationalization as the
trained-probe transfer matrix for predicted_clean/raw_z/hidden. Those matrices
measure whether a LEARNED DIRECTION trained at t_a still predicts well at
t_b. Here there is no learned direction -- e_t is a fixed, independently
recomputed formula at every t, so "transfer" is redefined as a conditional
probability of correctness: M[a,b] = P(correct at t_b | correct at t_a).
This is asymmetric (unlike a raw correlation coefficient) and gives a
genuinely analogous "does an early/late signal say something about a
later/earlier one" reading, but the numbers are NOT directly comparable in
magnitude to the trained-probe diag_mean/upper_tri_mean/lower_tri_mean
elsewhere in EXP-PT9 -- only the qualitative upper-vs-lower asymmetry
direction should be compared.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \
        experiments/phase_transition/cross_time_residual_consistency.py \
        --model elf --checkpoint baseline \
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \
        --out_dir results/phase_transition/elf/baseline \
        --n_samples 128 --n_t_steps 7 --label full
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from analyze_margin_trajectory import null_probs, oracle_probs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_steps", type=int, default=7)
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--n_oracle", type=int, default=4)
    p.add_argument("--n_null", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[PT9-residual] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        seq_len = adapter.seq_len
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
    T = len(t_grid)
    print(f"[PT9-residual] {N} sequences, L={L}, T={T} t-points")

    correct_over_t = np.zeros((T, N * L), dtype=bool)
    diag_acc = np.zeros(T)
    for ti, t in enumerate(t_grid):
        t = float(t)
        p_probs = oracle_probs(adapter, x_clean, t, args.n_oracle, args.batch_size, N, L, d, device)
        q_probs = null_probs(adapter, x_clean, t, args.n_null, args.batch_size, N, L, d, device)
        e = torch.log(p_probs + 1e-12) - torch.log(q_probs + 1e-12)
        argmax_e = e.argmax(-1)
        correct = (argmax_e == gt_ids).numpy().reshape(-1)
        correct_over_t[ti] = correct
        diag_acc[ti] = correct.mean()
        print(f"  t={t:.3f}  residual_argmax_acc={diag_acc[ti]:.4f}")
        del p_probs, q_probs, e, argmax_e, correct

    # M[a,b] = P(correct at t_b | correct at t_a) -- see module docstring for
    # why this replaces the trained-probe transfer matrix for this
    # representation.
    M = np.zeros((T, T))
    for ai in range(T):
        denom = max(1, correct_over_t[ai].sum())
        for bi in range(T):
            M[ai, bi] = (correct_over_t[ai] & correct_over_t[bi]).sum() / denom

    triu_i, triu_j = np.triu_indices(T, k=1)
    tril_i, tril_j = np.tril_indices(T, k=-1)
    upper_mean = float(M[triu_i, triu_j].mean())
    lower_mean = float(M[tril_i, tril_j].mean())

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "seq_len": L, "t_grid": t_grid.tolist(),
        "diag_native_accuracy": diag_acc.tolist(),
        "conditional_transfer_matrix": M.tolist(),
        "upper_tri_mean_P(later_correct|earlier_correct)": upper_mean,
        "lower_tri_mean_P(earlier_correct|later_correct)": lower_mean,
        "notes": [
            "This is EXP-PT9's 4th ('prior-subtracted logits') representation, "
            "operationalized as cross-time CONDITIONAL correctness (P(correct at "
            "t_b | correct at t_a)), NOT a trained-probe transfer matrix -- see "
            "module docstring for why a literal Linear(vocab_size,vocab_size) "
            "probe was rejected as intractable/redundant. Not directly comparable "
            "in magnitude to the predicted_clean/raw_z/hidden diag/upper/lower "
            "numbers elsewhere in this experiment; only the qualitative "
            "upper-vs-lower asymmetry direction is analogous.",
            "Reuses PT2/PT5's oracle_probs/null_probs (EXP-05v3-style global null "
            "reference), same choice as PT2/PT5/PT10 for consistency and cost.",
            "diag_native_accuracy is literally G_debias(t) -- already reported by "
            "EXP-PT1/PT5 under different names; included here only for reference.",
        ],
    }
    json_path = out_dir / f"cross_time_residual_consistency_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT9-residual] P(later correct | earlier correct) mean = {upper_mean:.4f}")
    print(f"[PT9-residual] P(earlier correct | later correct) mean = {lower_mean:.4f}")
    print(f"[PT9-residual] Saved {json_path}")


if __name__ == "__main__":
    main()
