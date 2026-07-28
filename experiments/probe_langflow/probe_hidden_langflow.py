"""
probe_hidden_langflow.py — EXP-21: LangFlow Linear Probe Gap

Compares:
  (a) native LM head top-1 accuracy (logits after bias skip correction)
  (b) independent linear probe trained on last-block hidden state h_t

Parallels EXP-07 for ELF. Tells us whether LangFlow has a probe-vs-native-head
gap similar to baseline ELF (+45pp at t=0.20), or not.

Usage:
  conda run -n elf python experiments/probe_langflow/probe_hidden_langflow.py \
      --checkpoint Continuous-Rivals-Discrete/langflow-owt \
      --n_samples 64 --seq_len 128 --n_noise 4 \
      --out_dir results/exp21_langflow
"""

import sys, os, argparse, json, math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# LangFlow import
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

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_hidden_states(model, samples, t_grid, gamma_grid, n_noise, seed):
    """
    Collect last-block hidden states (h_t) and logits for all (sample, t, noise).

    Returns per-t lists of:
      h_list[t_idx] : np.array [N_total, hidden_dim]  (N_total = n_samples * n_noise * L)
      y_list[t_idx] : np.array [N_total]               ground-truth token ids
      top1_native[t_idx] : float                        native head top-1 accuracy
    """
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
            z_batch = alpha * x_torch[None] + sigma * torch.from_numpy(eps).to(device)  # [N, L, d]
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
            logits, all_hidden = result   # logits: [N, L, V], all_hidden: list of [N, L, H]
            h_t = all_hidden[-1].cpu().float().numpy()  # [N, L, hidden_dim] — last block output
            logits_np = logits.cpu().float().numpy()    # [N, L, V]

            # Native head top-1
            top1 = (np.argmax(logits_np, axis=-1) == gt_ids[None, :])  # [N, L]
            top1_native_list[ti].append(float(top1.mean()))

            # Flatten (N, L) -> (N*L,)
            h_flat = h_t.reshape(-1, h_t.shape[-1])      # [N*L, H]
            y_flat = np.tile(gt_ids, n_noise)             # [N*L]
            h_list[ti].append(h_flat)
            y_list[ti].append(y_flat)

    return (
        [np.concatenate(h_list[ti], axis=0) for ti in range(T)],
        [np.concatenate(y_list[ti], axis=0) for ti in range(T)],
        [float(np.mean(top1_native_list[ti])) for ti in range(T)],
    )


def train_probe(h, y, max_iter=300, C=1.0):
    """Train a logistic regression probe on h -> y. Returns test accuracy."""
    # Subsample to at most 20000 instances for speed
    if len(y) > 20000:
        idx = np.random.RandomState(0).choice(len(y), 20000, replace=False)
        h, y = h[idx], y[idx]

    # Only keep tokens that appear >= 2 times (need both train/test)
    uniq, counts = np.unique(y, return_counts=True)
    keep_tokens = set(uniq[counts >= 2])
    mask = np.array([yi in keep_tokens for yi in y])
    h, y = h[mask], y[mask]

    if len(np.unique(y)) < 2:
        return float("nan")

    h_train, h_test, y_train, y_test = train_test_split(h, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(
        solver="saga", max_iter=max_iter, C=C,
        n_jobs=4, tol=1e-3,
    )
    clf.fit(h_train, y_train)
    return float(clf.score(h_test, y_test))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_noise", type=int, default=4)
    p.add_argument("--n_t_steps", type=int, default=12)
    p.add_argument("--out_dir", default="results/exp21_langflow")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # t grid: key points matching EXP-07
    t_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00])

    print(f"[load] Loading LangFlow from {args.checkpoint}")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
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
        print(f"  Training linear probe (LogReg) on {h.shape[1]}-dim hidden state...")
        probe_acc = train_probe(h, y)
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

    print("\n── Summary (LangFlow probe gap, EXP-21) ─────────────────────────")
    print(f"{'t':>5}  {'native_top1':>11}  {'probe_acc':>9}  {'gap':>8}  │ ELF_baseline_gap (EXP-07)")
    elf_gaps = {0.10: 12.3, 0.20: 45.8, 0.25: 42.7, 0.30: 28.7, 0.70: 11.2, 1.00: -3.4}
    for r in results:
        elf_g = elf_gaps.get(round(r["t"], 2), float("nan"))
        elf_str = f"{elf_g:+.1f}pp" if not math.isnan(elf_g) else "  --  "
        print(f"{r['t']:>5.2f}  {r['native_top1']:>11.4f}  {r['probe_acc']:>9.4f}  {r['gap']:>+8.4f}  │ {elf_str}")


if __name__ == "__main__":
    main()
