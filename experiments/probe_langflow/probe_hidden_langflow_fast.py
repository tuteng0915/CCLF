"""
probe_hidden_langflow_fast.py — EXP-21 fast variant (PyTorch linear probe on GPU)

Same as probe_hidden_langflow.py but replaces sklearn SAGA (100+ hrs) with
a GPU-accelerated linear layer trained with Adam for 30 epochs (~minutes total).

Usage:
  CUDA_VISIBLE_DEVICES=2 conda run -n elf python experiments/probe_langflow/probe_hidden_langflow_fast.py \
      --checkpoint Continuous-Rivals-Discrete/langflow-owt \
      --n_samples 64 --seq_len 128 --n_noise 4 \
      --out_dir results/exp21_langflow
"""

import sys, os, argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROBE_DIR = Path(__file__).parent
_LF_SRC = _PROBE_DIR.parents[1] / "models" / "LangFlow"
for _p in [str(_LF_SRC)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(_PROBE_DIR))
from probe_langflow import (
    load_langflow, encode_with_langflow, load_owt_texts,
    gamma_from_t, softmax_np,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_hidden_states(model, samples, t_grid, gamma_grid, n_noise, seed):
    """Collect last-block hidden states and native-head top-1 for all (sample, t, noise)."""
    sc = model.config.self_conditioning
    T = len(t_grid)
    h_list = [[] for _ in range(T)]
    y_list = [[] for _ in range(T)]
    top1_native_list = [[] for _ in range(T)]

    rng = np.random.default_rng(seed)

    for si, (gt_ids, clean_emb, attn_mask) in enumerate(samples):
        if si % 10 == 0:
            print(f"  sample {si+1}/{len(samples)}")
        L, d = clean_emb.shape
        x_torch = torch.from_numpy(clean_emb).to(device, dtype=torch.float32)

        for ti, (t_val, gamma) in enumerate(zip(t_grid, gamma_grid)):
            alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma)).item())
            sigma = math.sqrt(torch.sigmoid(torch.tensor( gamma)).item())

            eps = rng.standard_normal((n_noise, L, d)).astype(np.float32)
            z_batch = alpha * x_torch[None] + sigma * torch.from_numpy(eps).to(device)
            gamma_batch = torch.full((n_noise,), gamma, device=device, dtype=torch.float32)
            sc_in = torch.zeros_like(z_batch) if sc else None

            with torch.no_grad():
                result = model(
                    noisy_embeds=z_batch,
                    timesteps=gamma_batch,
                    x_self_cond=sc_in,
                    output_hidden_states=True,
                    return_dict=False,
                )
            logits, all_hidden = result
            h_t = all_hidden[-1].cpu().float().numpy()   # [N, L, H]
            logits_np = logits.cpu().float().numpy()      # [N, L, V]

            top1 = (np.argmax(logits_np, axis=-1) == gt_ids[None, :])
            top1_native_list[ti].append(float(top1.mean()))

            h_flat = h_t.reshape(-1, h_t.shape[-1])
            y_flat = np.tile(gt_ids, n_noise)
            h_list[ti].append(h_flat)
            y_list[ti].append(y_flat)

    return (
        [np.concatenate(h_list[ti], axis=0) for ti in range(T)],
        [np.concatenate(y_list[ti], axis=0) for ti in range(T)],
        [float(np.mean(top1_native_list[ti])) for ti in range(T)],
    )


def train_linear_probe_gpu(h, y, n_classes, epochs=30, lr=1e-2, batch_size=2048, val_frac=0.2):
    """Train a linear probe with Adam on GPU. Returns test accuracy."""
    n = len(y)
    n_val = int(n * val_frac)
    n_train = n - n_val

    idx = np.random.RandomState(42).permutation(n)
    h_train_np = h[idx[:n_train]]
    y_train_np = y[idx[:n_train]]
    h_val_np   = h[idx[n_train:]]
    y_val_np   = y[idx[n_train:]]

    # Keep only tokens that appear >= 2 in both train and val
    uniq_tr, cnt_tr = np.unique(y_train_np, return_counts=True)
    keep = set(uniq_tr[cnt_tr >= 1])
    uniq_v, cnt_v = np.unique(y_val_np, return_counts=True)
    keep &= set(uniq_v[cnt_v >= 1])
    tr_mask = np.array([yi in keep for yi in y_train_np])
    v_mask  = np.array([yi in keep for yi in y_val_np])
    h_train_np, y_train_np = h_train_np[tr_mask], y_train_np[tr_mask]
    h_val_np,   y_val_np   = h_val_np[v_mask],   y_val_np[v_mask]

    if len(keep) < 2 or len(h_train_np) == 0:
        return float("nan")

    # Re-encode labels to 0..K-1
    keep_sorted = sorted(keep)
    tok_to_idx = {tok: i for i, tok in enumerate(keep_sorted)}
    y_tr = torch.tensor([tok_to_idx[yi] for yi in y_train_np], dtype=torch.long, device=device)
    y_v  = torch.tensor([tok_to_idx[yi] for yi in y_val_np],   dtype=torch.long, device=device)
    h_tr = torch.tensor(h_train_np, dtype=torch.float32, device=device)
    h_v  = torch.tensor(h_val_np,   dtype=torch.float32, device=device)

    K = len(keep_sorted)
    dim = h_tr.shape[1]
    probe = nn.Linear(dim, K).to(device)
    nn.init.zeros_(probe.weight)
    nn.init.zeros_(probe.bias)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    n_tr = len(y_tr)
    for epoch in range(epochs):
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch_size):
            idx_b = perm[i:i+batch_size]
            logits = probe(h_tr[idx_b])
            loss = F.cross_entropy(logits, y_tr[idx_b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        val_acc = (probe(h_v).argmax(dim=1) == y_v).float().mean().item()
    return val_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seq_len",   type=int, default=128)
    p.add_argument("--n_noise",   type=int, default=4)
    p.add_argument("--n_t_steps", type=int, default=12)
    p.add_argument("--out_dir",   default="results/exp21_langflow")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--probe_epochs", type=int, default=30)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00])

    print(f"[load] Loading LangFlow from {args.checkpoint}")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    n_classes = E_all.shape[0]
    gamma_grid = gamma_from_t(t_grid, gamma_min, gamma_max)

    print(f"[data] Loading {args.n_samples} OWT samples")
    texts = load_owt_texts(args.n_samples)
    samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)

    print(f"[probe] Collecting hidden states for {len(t_grid)} t values...")
    h_per_t, y_per_t, top1_native = collect_hidden_states(
        model, samples, t_grid, gamma_grid, args.n_noise, args.seed
    )

    results = []
    for ti, t_val in enumerate(t_grid):
        h, y = h_per_t[ti], y_per_t[ti]
        print(f"\n[t={t_val:.2f}] n={len(y)}, h_shape={h.shape}, native_top1={top1_native[ti]:.4f}")
        print(f"  Training linear probe (PyTorch, {args.probe_epochs} epochs) ...")
        probe_acc = train_linear_probe_gpu(h, y, n_classes, epochs=args.probe_epochs)
        gap = probe_acc - top1_native[ti] if not math.isnan(probe_acc) else float("nan")
        print(f"  probe_acc={probe_acc:.4f}  gap={gap:+.4f}")
        results.append({
            "t": float(t_val),
            "gamma": float(gamma_grid[ti]),
            "native_top1": top1_native[ti],
            "probe_acc": probe_acc,
            "gap": gap,
            "n_samples": len(y),
        })

    out_json = out_dir / "probe_gap_results.json"
    with open(out_json, "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\n[saved] {out_json}")

    print("\n── Summary (LangFlow probe gap, EXP-21 fast) ────────────────────")
    print(f"{'t':>5}  {'native_top1':>11}  {'probe_acc':>9}  {'gap':>8}  │ ELF_baseline_gap (EXP-07)")
    elf_gaps = {0.10: 12.3, 0.20: 45.8, 0.25: 42.7, 0.30: 28.7, 0.70: 11.2, 1.00: -3.4}
    for r in results:
        elf_g = elf_gaps.get(round(r["t"], 2), float("nan"))
        elf_str = f"{elf_g:+.1f}pp" if not math.isnan(elf_g) else "  --  "
        print(f"{r['t']:>5.2f}  {r['native_top1']:>11.4f}  {r['probe_acc']:>9.4f}  {r['gap']:>+8.4f}  │ {elf_str}")


if __name__ == "__main__":
    main()
