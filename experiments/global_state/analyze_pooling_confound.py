"""EXP-GS11 (P0-1): Pooling/Averaging Confound Check.

Tests whether GS1-GS3's "early global signal" is mostly explained by the
sqrt(L) noise-reduction that mean-pooling over ~1024 positions provides on
the RAW oracle state, independent of any model processing. Compares
self-retrieval accuracy of (a) raw z_t mean-pooled vs (b) the model's own
predicted_clean mean-pooled, across a genuine document-length sweep
(documents are truncated to L_eff REAL tokens before T5 encoding -- not just
masked post-hoc, so the model/encoder genuinely only sees L_eff tokens).

See docs/specs/EXP-GS11-spec.md for the full design rationale (this addresses
a confound raised in review of EXP-GS1/GS2/GS3).

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/analyze_pooling_confound.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_samples 48 --label pilot
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

from common import cosine_rows, load_adapter, masked_mean_pool  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=48)
    p.add_argument("--n_candidates", type=int, default=200,
                    help="how many OWT docs to scan to find n_samples with full real length")
    p.add_argument("--l_effs", type=int, nargs="+", default=[32, 128, 512, 1000],
                    help="max should be <= actual max real-token length in the OWT sample "
                         "(observed max ~1020 for T5-tokenized OWT at seq_len=1024, not "
                         "exactly 1024 due to EOS/tokenizer boundary effects)")
    p.add_argument("--t_grid", type=float, nargs="+", default=[0.05, 0.28])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def self_retrieval_top1(summary, clean_ref):
    """summary, clean_ref: (N,d) numpy. Returns top-1 self-retrieval accuracy:
    for each i, does the nearest clean_ref (by cosine sim) to summary[i] equal
    clean_ref[i]?"""
    N = summary.shape[0]
    s_n = summary / (np.linalg.norm(summary, axis=1, keepdims=True) + 1e-12)
    r_n = clean_ref / (np.linalg.norm(clean_ref, axis=1, keepdims=True) + 1e-12)
    sims = s_n @ r_n.T  # (N,N)
    nearest = sims.argmax(axis=1)
    return float((nearest == np.arange(N)).mean())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS11] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    L_full = adapter.seq_len
    max_l_eff = max(args.l_effs)
    assert max_l_eff <= L_full

    # GS11 needs the raw (ids, mask) pool to filter by real length BEFORE
    # truncating + re-encoding -- doesn't use load_owt_docs's x_clean.
    if args.model == "elf":
        ids_pool, mask_pool = adapter.load_owt_sequences(args.n_candidates, seq_len=L_full)
    else:
        ids_pool, mask_pool, _ = adapter.load_owt_sequences(args.n_candidates, seq_len=L_full)
    full_len = mask_pool.sum(dim=1)
    keep = (full_len >= max_l_eff).nonzero(as_tuple=True)[0][:args.n_samples]
    ids = ids_pool[keep]
    mask_full = mask_pool[keep]
    N = ids.shape[0]
    print(f"[GS11] {N}/{args.n_samples} requested docs have >= {max_l_eff} real tokens "
          f"(scanned {args.n_candidates} candidates)")
    assert N >= 8, "too few long-enough documents found -- increase --n_candidates"

    pad_id = adapter.tokenizer.eos_token_id or 1
    eps = adapter.sample_epsilon((N, L_full, adapter.d_model))

    records = []
    for l_eff in args.l_effs:
        ids_trunc = ids.clone()
        ids_trunc[:, l_eff:] = pad_id
        mask_trunc = torch.zeros_like(mask_full)
        mask_trunc[:, :l_eff] = 1

        x_clean_trunc = adapter.encode_clean(ids_trunc, mask_trunc).cpu()
        clean_ref = masked_mean_pool(x_clean_trunc, mask_trunc).numpy()

        for t in args.t_grid:
            t = float(t)
            z_t = adapter.make_oracle_state(x_clean_trunc.to(device), eps, t).cpu()
            g_raw = masked_mean_pool(z_t, mask_trunc).numpy()

            out = adapter.forward_state(z_t, None, t, batch_size=args.batch_size)
            predicted_clean = out["predicted_clean"]
            g_model = masked_mean_pool(predicted_clean, mask_trunc).numpy()

            acc_raw = self_retrieval_top1(g_raw, clean_ref)
            acc_model = self_retrieval_top1(g_model, clean_ref)
            cos_raw = float(np.mean(cosine_rows(g_raw, clean_ref)))
            cos_model = float(np.mean(cosine_rows(g_model, clean_ref)))

            records.append({"l_eff": l_eff, "t": t, "n": N,
                             "retrieval_acc_raw": acc_raw, "retrieval_acc_model": acc_model,
                             "cos_raw": cos_raw, "cos_model": cos_model})
            print(f"  [GS11] L_eff={l_eff:4d} t={t:.2f}  "
                  f"retrieval_acc: raw={acc_raw:.3f} model={acc_model:.3f}  "
                  f"(chance={1.0/N:.3f})  cos: raw={cos_raw:.3f} model={cos_model:.3f}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "l_effs": args.l_effs, "t_grid": args.t_grid,
        "chance_level": 1.0 / N,
        "records": records,
        "notes": [
            "Documents are genuinely truncated to L_eff real tokens BEFORE T5 encoding "
            "(re-encoded per L_eff), not just masked post-hoc -- the encoder/backbone "
            "only ever sees L_eff real tokens for a given L_eff condition.",
            "retrieval_acc has a well-defined chance level (1/N), unlike raw cosine "
            "similarity which EXP-GS1 found saturates near ceiling in this space.",
            "Addresses a confound raised in review of EXP-GS1/GS2/GS3: does 'early global "
            "signal' scale the way pure 1/sqrt(L_eff) noise-averaging on the RAW oracle "
            "state would predict, or does the model add something beyond that?",
            "Pilot scale (n_samples=%d) -- see EXP-GS11-spec.md before citing numbers." % N,
        ],
    }
    json_path = out_dir / f"pooling_confound_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS11] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
