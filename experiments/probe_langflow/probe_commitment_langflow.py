"""
probe_commitment_langflow.py — EXP-22: LangFlow Per-Position Commitment Timing
                               EXP-23: LangFlow Geometry Null Model

EXP-22: Tracks committed_correct, committed_wrong, uncommitted per-position at each t
        using fixed-epsilon oracle protocol (same as EXP-16 for ELF).
        Commitment criterion: H(p_{i,t}) < entropy_thresh AND top1 == gt_id.

EXP-23: Feeds pure Gaussian z to LangFlow; measures mode_fraction and G_null vs true tokens.
        Parallels EXP-04 for ELF. ELF result: G_null ≈ 0.17% (negligible geometry bias).

Usage:
  conda run -n elf python experiments/probe_langflow/probe_commitment_langflow.py \
      --checkpoint Continuous-Rivals-Discrete/langflow-owt \
      --n_samples 64 --seq_len 128 --n_t_steps 40 --entropy_thresh 1.0 \
      --out_dir results/exp22_langflow
"""

import sys, os, argparse, json, math
from pathlib import Path

import numpy as np

_PROBE_DIR = Path(__file__).parent
_LF_SRC = _PROBE_DIR.parents[1] / "models" / "LangFlow"
for _p in [str(_LF_SRC)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(_PROBE_DIR))
from probe_langflow import (
    load_langflow, encode_with_langflow, load_owt_texts,
    gamma_from_t, softmax_np, token_entropy,
)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def probe_commitment_single(model, sample, t_grid, gamma_grid, seed, entropy_thresh, sc):
    """
    Track per-position commitment along fixed-eps oracle path.

    Returns list of dicts, one per t value.
    """
    gt_ids, clean_emb, attn_mask = sample
    L, d = clean_emb.shape

    rng = np.random.default_rng(seed)
    eps_np = rng.standard_normal((L, d)).astype(np.float32)  # fixed!
    eps_t = torch.from_numpy(eps_np).to(device)
    x_t = torch.from_numpy(clean_emb).to(device, dtype=torch.float32)

    states = []
    for t_val, gamma in zip(t_grid, gamma_grid):
        alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma)).item())
        sigma = math.sqrt(torch.sigmoid(torch.tensor( gamma)).item())
        z_t = (alpha * x_t + sigma * eps_t)[None]  # [1, L, d]
        gamma_t = torch.full((1,), gamma, device=device, dtype=torch.float32)
        sc_in = torch.zeros_like(z_t) if sc else None

        with torch.no_grad():
            out = model(
                noisy_embeds=z_t,
                timesteps=gamma_t,
                x_self_cond=sc_in,
                return_dict=False,
            )
        logits_np = (out[0] if isinstance(out, (tuple, list)) else out)[0].cpu().float().numpy()
        p = softmax_np(logits_np, tau=1.0)   # [L, V]
        H = -(p * np.log(p + 1e-9)).sum(-1)  # [L] entropy in nats
        top1 = np.argmax(p, axis=-1)          # [L]

        committed = H < entropy_thresh
        correct = (top1 == gt_ids)

        states.append({
            "t": float(t_val),
            "committed_correct": float((committed & correct).mean()),
            "committed_wrong":   float((committed & ~correct).mean()),
            "uncommitted":       float((~committed).mean()),
            "top1_native":       float(correct.mean()),
            "mean_entropy":      float(H.mean()),
        })
    return states


def run_null_model(model, sample, t_val, gamma_t, sc):
    """EXP-23: feed pure Gaussian noise, measure mode_fraction and G_null."""
    gt_ids, clean_emb, attn_mask = sample
    L, d = clean_emb.shape
    z_null = torch.randn(1, L, d, device=device, dtype=torch.float32)
    gamma_t_t = torch.full((1,), gamma_t, device=device, dtype=torch.float32)
    sc_in = torch.zeros(1, L, d, device=device) if sc else None

    with torch.no_grad():
        out = model(noisy_embeds=z_null, timesteps=gamma_t_t, x_self_cond=sc_in, return_dict=False)
    logits_np = (out[0] if isinstance(out, (tuple, list)) else out)[0].cpu().float().numpy()
    p = softmax_np(logits_np, tau=1.0)
    top1_null = np.argmax(p, axis=-1)

    G_null = float((top1_null == gt_ids).mean())
    mode_id = np.argmax(p.mean(0))
    mode_frac = float((top1_null == mode_id).mean())
    return G_null, mode_frac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_steps", type=int, default=40)
    p.add_argument("--entropy_thresh", type=float, default=1.0)
    p.add_argument("--out_dir", default="results/exp22_langflow")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid = np.linspace(0.0, 1.0, args.n_t_steps + 1)[1:]  # avoid t=0
    print(f"[load] Loading LangFlow from {args.checkpoint}")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    gamma_grid = gamma_from_t(t_grid, gamma_min, gamma_max)
    sc = model.config.self_conditioning

    print(f"[data] Loading {args.n_samples} OWT samples")
    texts = load_owt_texts(args.n_samples)
    samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)

    # EXP-22: per-position commitment
    print(f"\n[EXP-22] Per-position commitment (thresh={args.entropy_thresh} nats)...")
    all_states = []
    for si, sample in enumerate(samples):
        if si % 10 == 0:
            print(f"  sample {si+1}/{len(samples)}")
        s = probe_commitment_single(
            model, sample, t_grid, gamma_grid, seed=si * 1000,
            entropy_thresh=args.entropy_thresh, sc=sc,
        )
        all_states.append(s)

    # Aggregate: mean across samples at each t
    agg = []
    for ti, t_val in enumerate(t_grid):
        row = {"t": float(t_val)}
        for key in ["committed_correct", "committed_wrong", "uncommitted", "top1_native", "mean_entropy"]:
            vals = [all_states[si][ti][key] for si in range(len(all_states))]
            row[key + "_mean"] = float(np.mean(vals))
            row[key + "_std"]  = float(np.std(vals))
        agg.append(row)

    out_22 = out_dir / "commitment_by_t.json"
    with open(out_22, "w") as f:
        json.dump({"agg": agg, "args": vars(args)}, f, indent=2)
    print(f"[saved] {out_22}")

    # EXP-23: null model at key t values
    print(f"\n[EXP-23] Geometry null model...")
    null_t_vals = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
    null_results = []
    for t_val in null_t_vals:
        gamma_t = float(gamma_from_t(np.array([t_val]), gamma_min, gamma_max)[0])
        G_nulls, mode_fracs = [], []
        for sample in samples:
            g, m = run_null_model(model, sample, t_val, gamma_t, sc)
            G_nulls.append(g)
            mode_fracs.append(m)
        print(f"  t={t_val:.2f}: G_null={np.mean(G_nulls):.4f}  mode_frac={np.mean(mode_fracs):.4f}")
        null_results.append({
            "t": t_val,
            "G_null_mean": float(np.mean(G_nulls)),
            "G_null_std":  float(np.std(G_nulls)),
            "mode_frac_mean": float(np.mean(mode_fracs)),
        })

    out_23 = out_dir / "null_model.json"
    with open(out_23, "w") as f:
        json.dump({"null_results": null_results, "args": vars(args)}, f, indent=2)
    print(f"[saved] {out_23}")

    # Print commitment summary
    print("\n── EXP-22 Summary (LangFlow per-position commitment) ─────────────")
    print(f"{'t':>5}  {'correct':>8}  {'wrong':>7}  {'uncommit':>8}  │ top1_nat  entropy")
    for r in agg[::4]:  # every ~10%
        print(f"{r['t']:>5.2f}  {r['committed_correct_mean']:>8.3f}  "
              f"{r['committed_wrong_mean']:>7.3f}  {r['uncommitted_mean']:>8.3f}  "
              f"│ {r['top1_native_mean']:>8.3f}  {r['mean_entropy_mean']:>7.3f}")

    print("\n── EXP-23 Summary (LangFlow null model) ────────────────────────────")
    print(f"{'t':>5}  {'G_null':>8}  {'mode_frac':>9}  │ ELF_G_null (EXP-04)")
    for r in null_results:
        print(f"{r['t']:>5.2f}  {r['G_null_mean']:>8.4f}  {r['mode_frac_mean']:>9.4f}  │ ~0.0017")


if __name__ == "__main__":
    main()
