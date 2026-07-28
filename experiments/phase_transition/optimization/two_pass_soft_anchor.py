"""Optimization direction B: two-pass generation with a soft anchor toward
the model's own first-pass draft (inference-time only, no retraining).

Motivation (see docs/specs/EXP-PT7-spec.md's causal-interpolation section):
blending a free-running trajectory toward a paired "oracle" state at an
intermediate lambda (around 0.25) causally improves agreement with the
eventual answer, but going all the way to lambda=1 is often WORSE, sometimes
catastrophically so for KD checkpoints. That experiment used the trajectory's
OWN eventual final decode as the oracle target (a diagnostic ablation, not
something available at real generation time, since it requires knowing the
future). This script operationalizes the same blend+continue mechanism with
a target that IS available at real inference time: generate once (a "draft"),
encode that draft back into latent space, then run a SECOND, independent
generation and blend its state toward the draft-derived pseudo-oracle at some
intermediate t, continuing to completion.

Reuses, verbatim where possible:
  - free_running_rollout (compare_oracle_rollout.py) for both passes.
  - the z_lambda blend + continue_rollout pattern (interpolate_oracle_rollout.py),
    swapping the interpolation target from "this trajectory's own final ids"
    to "an independent first-pass draft's final ids".

This is explicitly a SMALL-SCALE VALIDATION of the technique, not a finished
method: no claim is made here about the technique being ready for general use.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/optimization/two_pass_soft_anchor.py \\
        --model elf --checkpoint kd_cr \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/optimization/elf/kd_cr \\
        --n_samples 32 --n_gen_steps 32 --t_intervene 0.4 --n_continue_steps 12 --label full
"""

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
_PT_DIR = _THIS_DIR.parent
for p in (_THIS_DIR, _PT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from compare_oracle_rollout import free_running_rollout  # noqa: E402
from interpolate_oracle_rollout import continue_rollout  # noqa: E402

LAMBDAS = [0.0, 0.25, 0.5, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_gen_steps", type=int, default=32, help="ODE steps for each free-running pass")
    p.add_argument("--t_intervene", type=float, default=0.4)
    p.add_argument("--n_continue_steps", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def text_quality_metrics(texts):
    """Same repetitive-word / non-ASCII heuristics used in this project's
    EXP-36v2 text-quality analysis (>35% single-word share, >2% non-ASCII
    chars flag a sample as degenerate/multilingual respectively)."""
    rep_fracs, ml_fracs = [], []
    for t in texts:
        words = t.split()
        if words:
            most_common = Counter(words).most_common(1)[0][1]
            rep_fracs.append(most_common / len(words))
        else:
            rep_fracs.append(0.0)
        non_ascii = sum(1 for c in t if ord(c) > 127)
        ml_fracs.append(non_ascii / max(1, len(t)))
    rep_fracs, ml_fracs = np.array(rep_fracs), np.array(ml_fracs)
    return {
        "repetitive_rate": float((rep_fracs > 0.35).mean()),
        "multilingual_rate": float((ml_fracs > 0.02).mean()),
        "mean_repetitive_frac": float(rep_fracs.mean()),
        "mean_multilingual_frac": float(ml_fracs.mean()),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[two-pass] Loading {args.model} model...")
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

    print(f"[two-pass] Pass 1 (draft): {N} samples, L={L}, {args.n_gen_steps} ODE steps...")
    _, draft_ids, _ = free_running_rollout(
        adapter, N, L, d, t_eps, 0.999, args.n_gen_steps, args.batch_size, device, init_noise_scale)

    if args.model == "elf":
        attn_mask = torch.ones(N, L, dtype=torch.long)
        x_draft = adapter.encode_clean(draft_ids, attn_mask).cpu()
    else:
        x_draft = adapter.encode_clean(draft_ids).cpu()

    print(f"[two-pass] Pass 2 (independent trajectory to anchor): {N} samples...")
    trajectory2, orig_final_ids2, t_steps = free_running_rollout(
        adapter, N, L, d, t_eps, 0.999, args.n_gen_steps, args.batch_size, device, init_noise_scale)

    k_intervene = int(np.argmin(np.abs(t_steps - args.t_intervene)))
    t_int = float(t_steps[k_intervene])
    z_roll = trajectory2[k_intervene]["z"]
    print(f"[two-pass] Branching pass 2 at t={t_int:.3f} (step {k_intervene}/{args.n_gen_steps})")

    eps = torch.randn_like(x_draft)
    z_draft_noised = t_int * x_draft + (1.0 - t_int) * eps

    def decode_texts(ids):
        return adapter.tokenizer.batch_decode(ids, skip_special_tokens=True)

    draft_texts = decode_texts(draft_ids)
    draft_quality = text_quality_metrics(draft_texts)
    print(f"[two-pass] Draft (pass 1) quality: {draft_quality}")

    results = {"lambdas": LAMBDAS, "t_intervene": t_int, "per_lambda": []}
    for lam in LAMBDAS:
        z_lambda = (1.0 - lam) * z_roll + lam * z_draft_noised
        final_continued = continue_rollout(adapter, z_lambda, t_int, 0.999, args.n_continue_steps,
                                            args.batch_size, device)
        texts = decode_texts(final_continued)
        quality = text_quality_metrics(texts)
        agreement_with_own_unanchored = (final_continued == orig_final_ids2).float().mean().item()
        agreement_with_draft = (final_continued == draft_ids).float().mean().item()
        entry = {
            "lambda": lam, **quality,
            "agreement_with_pass2_unanchored_final": agreement_with_own_unanchored,
            "agreement_with_draft": agreement_with_draft,
        }
        results["per_lambda"].append(entry)
        print(f"  lambda={lam:.2f}  repetitive_rate={quality['repetitive_rate']:.4f}  "
              f"multilingual_rate={quality['multilingual_rate']:.4f}  "
              f"agree_w_unanchored={agreement_with_own_unanchored:.4f}  "
              f"agree_w_draft={agreement_with_draft:.4f}")
        del z_lambda, final_continued
        gc.collect()

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "n_gen_steps": args.n_gen_steps, "n_continue_steps": args.n_continue_steps,
        "draft_quality": draft_quality,
        **results,
        "notes": [
            "SMALL-SCALE VALIDATION of a proposed inference-time technique, not a finished method.",
            "lambda=0 is NOT identical to a plain single-pass generation -- like "
            "interpolate_oracle_rollout.py, sc_state is reset to None at the branch point for "
            "every lambda including 0, so even 'no anchor' has this small perturbation relative "
            "to a true single continuous rollout. Use lambda=0 here as the paired control, not "
            "as a stand-in for 'baseline single-pass generation quality'.",
            "Quality metrics (repetitive_rate, multilingual_rate) use the same >35%/>2% "
            "thresholds as this project's EXP-36v2 text-quality analysis, computed inline here "
            "since no shared quality-metric utility exists elsewhere in the codebase (checked "
            "eval.py -- confirmed absent).",
            "agreement_with_draft is expected to rise toward 1.0 as lambda->1 near-tautologically "
            "(z_draft_noised is built FROM the draft); the informative comparison is the quality "
            "metrics across lambda, and whether lambda=0.25 beats both lambda=0 and lambda=1.",
        ],
    }
    json_path = out_dir / f"two_pass_soft_anchor_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[two-pass] Saved {json_path}")


if __name__ == "__main__":
    main()
