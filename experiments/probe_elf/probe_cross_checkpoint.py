"""
EXP-07c: Cross-checkpoint probe transfer.

Loads already-collected layer_states (exp07b) and tests whether a linear probe
trained on baseline backbone features transfers to kd-cr/kd2 at each layer.

If transfer_acc[baseline->kd_cr] >= trained_acc[kd_cr] -> backbone convergence
(KD trains decoder, not backbone).

Usage (from ELF-torch root):
  CUDA_VISIBLE_DEVICES=7 python experiments/probe_elf/probe_cross_checkpoint.py \
    --baseline_dir results/exp07b_baseline \
    --kd_cr_dir results/exp07b_kd_cr \
    --kd2_dir results/exp07b_kd2 \
    --output_dir results/exp07c
"""

import argparse, json, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_probe(feats, labels, mask, vocab_size, device, n_epochs=20, lr=1e-2):
    N, L, d = feats.shape
    fflat = feats.reshape(-1, d)
    yflat = labels.reshape(-1)
    mflat = mask.reshape(-1).bool()
    fv, yv = fflat[mflat].float(), yflat[mflat].long()

    n = len(fv); n_tr = int(0.8 * n)
    rng = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=rng)
    tr, va = idx[:n_tr], idx[n_tr:]

    tr_dl = DataLoader(TensorDataset(fv[tr], yv[tr]), batch_size=4096, shuffle=True)
    va_dl = DataLoader(TensorDataset(fv[va], yv[va]), batch_size=8192)

    probe = nn.Linear(d, vocab_size).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for _ in range(n_epochs):
        probe.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); F.cross_entropy(probe(xb), yb).backward(); opt.step()
        sched.step()

    probe.eval(); c = tot = 0
    with torch.no_grad():
        for xb, yb in va_dl:
            xb, yb = xb.to(device), yb.to(device)
            c += (probe(xb).argmax(-1) == yb).sum().item(); tot += yb.size(0)
    return probe, c / tot


def eval_all(probe, feats, labels, mask, device):
    N, L, d = feats.shape
    mflat = mask.reshape(-1).bool()
    fv = feats.reshape(-1, d)[mflat].float()
    yv = labels.reshape(-1)[mflat].long()
    dl = DataLoader(TensorDataset(fv, yv), batch_size=8192)
    probe.eval(); c = tot = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            c += (probe(xb).argmax(-1) == yb).sum().item(); tot += yb.size(0)
    return c / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default="results/exp07b_baseline")
    ap.add_argument("--kd_cr_dir",    default="results/exp07b_kd_cr")
    ap.add_argument("--kd2_dir",      default="results/exp07b_kd2")
    ap.add_argument("--output_dir",   default="results/exp07c")
    ap.add_argument("--t_values",     default="0.20,0.30,0.50,0.70")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    t_vals = [float(x) for x in args.t_values.split(",")]
    dirs   = {"baseline": args.baseline_dir, "kd_cr": args.kd_cr_dir, "kd2": args.kd2_dir}
    names  = list(dirs.keys())

    all_results = {}

    for t_val in t_vals:
        print(f"\n{'='*64}\nt = {t_val:.2f}\n{'='*64}")

        states = {}
        for name, d in dirs.items():
            p = os.path.join(d, f"layer_states_t{t_val:.3f}.pt")
            print(f"  Loading {p} ...", flush=True)
            states[name] = torch.load(p, map_location="cpu", weights_only=False)

        # All checkpoints use same sequences (same y_tokens / attn_mask)
        y_tok   = states["baseline"]["y_tokens"]    # (N, L)
        mask    = states["baseline"]["attn_mask"]   # (N, L)
        vocab   = int(y_tok.max().item()) + 1
        n_lay   = len(states["baseline"]["layer_feats"])
        print(f"  vocab={vocab} layers={n_lay} seqs={y_tok.shape[0]}", flush=True)

        t_res = {"final_layer": {}, "layer_profile": {}}

        # --- Final layer: train on each source, eval on all ---
        Li = n_lay - 1
        print(f"\n  Final layer ({Li}) cross-checkpoint matrix:", flush=True)
        print("  {:12s}  {:>9s}  {:>9s}  {:>9s}".format("src->tgt", "baseline", "kd_cr", "kd2"))
        for src in names:
            probe, src_acc = train_probe(
                states[src]["layer_feats"][Li], y_tok, mask, vocab, device)
            row = {}
            for tgt in names:
                acc = eval_all(probe, states[tgt]["layer_feats"][Li],
                               states[tgt]["y_tokens"], states[tgt]["attn_mask"], device)
                row[tgt] = round(acc, 5)
            print(f"  {src:<12}  {row['baseline']:9.4f}  {row['kd_cr']:9.4f}  {row['kd2']:9.4f}", flush=True)
            t_res["final_layer"][src] = row

        # --- Layer-wise: probe trained on baseline, eval on all ---
        print(f"\n  Layer-wise (probe trained on baseline, eval on all):", flush=True)
        print(f"  {'layer':>5}  {'baseline':>9}  {'kd_cr':>9}  {'kd2':>9}")
        layer_rows = {}
        for li in range(n_lay):
            probe, _ = train_probe(
                states["baseline"]["layer_feats"][li], y_tok, mask, vocab, device, n_epochs=15)
            row = {}
            for tgt in names:
                acc = eval_all(probe, states[tgt]["layer_feats"][li],
                               states[tgt]["y_tokens"], states[tgt]["attn_mask"], device)
                row[tgt] = round(acc, 5)
            print(f"  {li:5d}  {row['baseline']:9.4f}  {row['kd_cr']:9.4f}  {row['kd2']:9.4f}", flush=True)
            layer_rows[str(li)] = row
        t_res["layer_profile"] = layer_rows

        all_results[str(t_val)] = t_res

    out = os.path.join(args.output_dir, "cross_checkpoint_transfer.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[EXP-07c] Saved to {out}")


if __name__ == "__main__":
    main()
