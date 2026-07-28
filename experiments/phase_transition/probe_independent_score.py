"""EXP-PT2 supplement: Independent-probe score.

EXP-PT2's suite-doc measurement list includes "Independent-probe score"
alongside true-token rank, margin, entropy, and native top-1 identity --
this was the one measurement left unimplemented in the initial PT2 pass
(see docs/specs/EXP-PT2-spec.md). This script fills that gap: at each t,
train a linear probe (same architecture as EXP-PT9's
probe_cross_time_transfer.py / this project's EXP-07c) on predicted_clean
states from TRAIN sequences, evaluate it on held-out VAL sequences, and
compare against the native decode accuracy on those SAME val sequences --
directly reproducing this project's "Story A: Probe Gap" framing
(EXP-07/EXP-07v2 for ELF, EXP-21/21v2 for LangFlow) but now inside the
unified phase-transition adapter framework, at the SAME t-grid used by the
rest of the PT1-10 suite (so probe-gap numbers are directly comparable to
this suite's other margin/rank curves, not just to the older EXP-07-family
numbers which used different sampling/splits).

Sequence-level train/val split (the EXP-07->EXP-07v2 lesson), reused here.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/probe_independent_score.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 128 --n_t_steps 11 --label full
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

from probe_cross_time_transfer import LinearProbe, train_probe, eval_probe  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_steps", type=int, default=11)
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--val_frac", type=float, default=0.3)
    p.add_argument("--n_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[probe_gap] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        seq_len = adapter.seq_len
        ids, mask = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        x_clean = adapter.encode_clean(ids, mask).cpu()
        gt_ids = ids
        vocab_size = adapter.vocab_size
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        seq_len = args.seq_len
        ids, mask, x_clean = adapter.load_owt_sequences(args.n_samples, seq_len=seq_len)
        gt_ids = ids
        vocab_size = adapter.vocab_size

    N, L, d = x_clean.shape
    t_grid = np.linspace(args.t_min, args.t_max, args.n_t_steps)
    print(f"[probe_gap] {N} sequences, L={L}, T={len(t_grid)} t-points, vocab={vocab_size}")

    rng = np.random.default_rng(args.seed)
    seq_order = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_seqs, train_seqs = seq_order[:n_val], seq_order[n_val:]
    print(f"[probe_gap] {len(train_seqs)} train sequences, {len(val_seqs)} val sequences")

    @torch.no_grad()
    def get_state_and_logits(t):
        eps = adapter.sample_epsilon((N, L, d))
        z = adapter.make_oracle_state(x_clean.to(device), eps, float(t))
        out = adapter.forward_state(z.cpu(), None, float(t), batch_size=args.batch_size)
        return out["predicted_clean"], out["logits"]

    results = {"t": [], "native_acc_val": [], "probe_acc_val": [], "gap_probe_minus_native": []}
    for t in t_grid:
        t = float(t)
        h, logits = get_state_and_logits(t)

        h_train = h[train_seqs].reshape(-1, d)
        y_train = gt_ids[train_seqs].reshape(-1)
        h_val = h[val_seqs].reshape(-1, d)
        y_val = gt_ids[val_seqs].reshape(-1)

        probe = train_probe(h_train, y_train, vocab_size, device, n_epochs=args.n_epochs)
        probe_acc = eval_probe(probe, h_val, y_val, device)

        native_preds = logits[val_seqs].argmax(-1).reshape(-1)
        native_acc = (native_preds == y_val).float().mean().item()

        gap = probe_acc - native_acc
        results["t"].append(t)
        results["native_acc_val"].append(native_acc)
        results["probe_acc_val"].append(probe_acc)
        results["gap_probe_minus_native"].append(gap)
        print(f"  t={t:.3f}  native_acc={native_acc:.4f}  probe_acc={probe_acc:.4f}  "
              f"gap(probe-native)={gap:+.4f}")

        del probe, h, logits, h_train, h_val
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "n_train_seqs": len(train_seqs), "n_val_seqs": len(val_seqs),
        "seq_len": L, **results,
        "mean_gap": float(np.mean(results["gap_probe_minus_native"])),
        "notes": [
            "Fills EXP-PT2's previously-unimplemented 'independent-probe score' measurement.",
            "Reproduces this project's Story-A 'probe gap' framing (EXP-07v2 for ELF, "
            "EXP-21v2 for LangFlow) inside the unified phase-transition adapter/t-grid, "
            "so numbers are comparable across this suite's other PT1/2/3/5 results -- but "
            "NOT numerically identical to EXP-07v2/EXP-21v2 (different sampling, seeds, "
            "t-grid, and probe training epochs).",
            "Sequence-level train/val split (EXP-07->EXP-07v2 lesson).",
        ],
    }
    json_path = out_dir / f"probe_gap_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[probe_gap] mean gap (probe - native) across t: {summary['mean_gap']:+.4f}")
    print(f"[probe_gap] Saved {json_path}")


if __name__ == "__main__":
    main()
