"""EXP-GS9: Minimal Global Contrast Sets.

For each hand-constructed (context_A, context_B, target_A, target_B) pair
(experiments/global_state/build_global_minimal_pairs.py), compares the
log-probability of each target token when embedded in its OWN frame vs the
OTHER frame, at several oracle t. See docs/specs/EXP-GS9-spec.md for the
metric definition and the experimenter-bias caveat.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/global_state/probe_global_minimal_pairs.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --label pilot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
_PT_DIR = _THIS_DIR.parent / "phase_transition"
for p in (_THIS_DIR, _PT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_global_minimal_pairs import PAIRS  # noqa: E402
from common import load_adapter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--t_grid", type=float, nargs="+", default=[0.05, 0.28, 0.65, 0.99])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def encode_pair_variant(tokenizer, context, target, seq_len, device):
    """Returns (ids, mask, target_token_id, target_pos).

    target_pos is found by encoding `context` alone (no trailing space) and
    stripping a trailing EOS if the tokenizer auto-adds one. T5 auto-appends
    </s> to every encoded string (stripped here); GPT-2/LangFlow's tokenizer
    does not, and " "+target merges into a single leading-space BPE token
    (e.g. " destruction") rather than staying separate -- so the T5-specific
    "encode `context + ' '`, then drop one trailing token" trick used in an
    earlier version of this function is wrong for GPT-2 and silently landed
    on the wrong position (off by one). Verified against both tokenizers
    directly before this fix, see docs/specs/EXP-GS9-spec.md LangFlow notes.
    """
    ctx_ids = tokenizer(context)["input_ids"]
    if tokenizer.eos_token_id is not None and ctx_ids and ctx_ids[-1] == tokenizer.eos_token_id:
        ctx_ids = ctx_ids[:-1]
    target_pos = len(ctx_ids)
    full_text = context + " " + target
    enc = tokenizer(full_text, return_tensors="pt", truncation=True,
                     max_length=seq_len, padding="max_length")
    ids = enc["input_ids"][0]
    mask = enc["attention_mask"][0]
    target_token_id = ids[target_pos].item()
    return ids, mask, target_token_id, target_pos


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS9] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    tokenizer = adapter.tokenizer
    seq_len = adapter.seq_len
    print(f"[GS9] {len(PAIRS)} pairs, seq_len={seq_len}")

    # Build the 4 variants (A+tA, A+tB, B+tA, B+tB) per pair.
    variants = []  # list of dicts per pair
    for pair in PAIRS:
        v = {}
        for ctx_name, ctx in [("A", pair["context_A"]), ("B", pair["context_B"])]:
            for tgt_name, tgt in [("A", pair["target_A"]), ("B", pair["target_B"])]:
                ids, mask, tok_id, pos = encode_pair_variant(
                    tokenizer, ctx, tgt, seq_len, device)
                v[f"ctx{ctx_name}_tgt{tgt_name}"] = {
                    "ids": ids, "mask": mask, "token_id": tok_id, "pos": pos,
                }
        variants.append(v)

    N = len(PAIRS)
    L = seq_len

    # One shared noise per pair-variant-combo... actually one shared epsilon per PAIR
    # (paired-noise design, all 4 variants of a pair use the same eps -- reduces variance,
    # matches this repo's paired-oracle convention).
    eps_per_pair = [adapter.sample_epsilon((1, L, adapter.d_model)) for _ in range(N)]

    records = []
    for t in args.t_grid:
        t = float(t)
        deltas_A, deltas_B = [], []
        for i, pair in enumerate(PAIRS):
            v = variants[i]
            row = {}
            for key in ["ctxA_tgtA", "ctxA_tgtB", "ctxB_tgtA", "ctxB_tgtB"]:
                ids = v[key]["ids"].unsqueeze(0)
                mask = v[key]["mask"].unsqueeze(0)
                x_clean = adapter.encode_clean(ids, mask).cpu()
                z_t = adapter.make_oracle_state(x_clean.to(device), eps_per_pair[i], t).cpu()
                out = adapter.forward_state(z_t, None, t, batch_size=1)
                log_p = F.log_softmax(out["logits"][0].float(), dim=-1)
                pos = v[key]["pos"]
                tok_id = v[key]["token_id"]
                row[key] = float(log_p[pos, tok_id])

            delta_A = row["ctxA_tgtA"] - row["ctxB_tgtA"]
            delta_B = row["ctxB_tgtB"] - row["ctxA_tgtB"]
            deltas_A.append(delta_A)
            deltas_B.append(delta_B)
            records.append({"t": t, "domain": pair["domain"],
                             "ell_ctxA_tgtA": row["ctxA_tgtA"], "ell_ctxB_tgtA": row["ctxB_tgtA"],
                             "ell_ctxB_tgtB": row["ctxB_tgtB"], "ell_ctxA_tgtB": row["ctxA_tgtB"],
                             "delta_A": delta_A, "delta_B": delta_B})

        frac_pos_A = float(np.mean([d > 0 for d in deltas_A]))
        frac_pos_B = float(np.mean([d > 0 for d in deltas_B]))
        print(f"  [GS9] t={t:.3f}  mean(Delta_A)={np.mean(deltas_A):+.3f} "
              f"(frac>0={frac_pos_A:.2f})  mean(Delta_B)={np.mean(deltas_B):+.3f} "
              f"(frac>0={frac_pos_B:.2f})")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_pairs": N, "t_grid": args.t_grid,
        "records": records,
        "notes": [
            "12 hand-constructed pairs -- experimenter selection bias risk, see "
            "EXP-GS9-spec.md Section 3 point 1.",
            "No default-competitor margin (ELF is bidirectional, not autoregressive) -- "
            "uses direct log-prob comparison of the same target token across contexts.",
            "One shared epsilon per pair across all 4 (context,target) variants "
            "(paired-noise design).",
        ],
    }
    json_path = out_dir / f"minimal_pairs_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS9] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
