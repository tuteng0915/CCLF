"""
probe_layerwise_langflow.py — EXP-30: LangFlow Layer-wise Linear Probe

EXP-07b analog for LangFlow: probe each of the 12 DDiT transformer blocks
using output_hidden_states=True (same API as EXP-21 fast).

Usage:
  cd /home/wjzhang/tt_workspace/model/CCLF/CCLF
  CUDA_VISIBLE_DEVICES=6 conda run -n elf python experiments/probe_langflow/probe_layerwise_langflow.py \
      --checkpoint Continuous-Rivals-Discrete/langflow-owt \
      --n_samples 64 --seq_len 128 --n_noise 4 \
      --out_dir results/exp30_langflow_layerwise \
      > /tmp/exp30_langflow_layerwise.log 2>&1 &
"""

import sys, os, argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROBE_DIR = Path(__file__).parent
_LF_SRC    = _PROBE_DIR.parents[1] / "models" / "LangFlow"
LANGFLOW_REPO = os.path.expanduser("~/LangFlow")
for _p in [LANGFLOW_REPO, str(_LF_SRC)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_PROBE_DIR))

from probe_langflow import load_langflow, load_owt_texts, gamma_from_t, encode_with_langflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_gpu_probe(h: np.ndarray, y: np.ndarray, n_epochs=30, lr=1e-2, batch_size=4096) -> float:
    n = len(y)
    idx = np.random.RandomState(42).permutation(n)
    n_tr = int(n * 0.8)
    h_tr = torch.tensor(h[idx[:n_tr]], dtype=torch.float32, device=device)
    y_tr = torch.tensor(y[idx[:n_tr]], dtype=torch.long,    device=device)
    h_va = torch.tensor(h[idx[n_tr:]], dtype=torch.float32, device=device)
    y_va = torch.tensor(y[idx[n_tr:]], dtype=torch.long,    device=device)

    V = int(y.max()) + 1
    d = h.shape[1]
    W = nn.Linear(d, V, bias=True).to(device)
    nn.init.zeros_(W.weight); nn.init.zeros_(W.bias)
    opt = optim.Adam(W.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        perm = torch.randperm(len(y_tr), device=device)
        for i in range(0, len(y_tr), batch_size):
            b = perm[i:i+batch_size]
            opt.zero_grad()
            ce(W(h_tr[b]), y_tr[b]).backward()
            opt.step()

    with torch.no_grad():
        acc = float((W(h_va).argmax(1) == y_va).float().mean())
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--n_samples",    type=int, default=64)
    p.add_argument("--seq_len",      type=int, default=128)
    p.add_argument("--n_noise",      type=int, default=4)
    p.add_argument("--out_dir",      default="results/exp30_langflow_layerwise")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--probe_epochs", type=int, default=30)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[EXP-30] LangFlow layer-wise linear probe (EXP-07b analog)")
    model, tokenizer, gamma_min, gamma_max, E = load_langflow(args.checkpoint)
    sc = model.config.self_conditioning
    print(f"  γ range: [{gamma_min:.2f}, {gamma_max:.2f}]  self_cond={sc}")

    t_grid = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 1.00])
    gamma_grid = gamma_from_t(t_grid, gamma_min, gamma_max)

    texts = load_owt_texts(args.n_samples)
    samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)
    print(f"  Loaded {len(samples)} samples")

    rng = np.random.default_rng(args.seed)
    T   = len(t_grid)

    # Accumulators: per t, per layer
    h_layers = [[[] for _ in range(13)] for _ in range(T)]  # 12 blocks + output
    y_per_t  = [[] for _ in range(T)]
    nat_per_t = [[] for _ in range(T)]

    for si, (gt_ids, clean_emb, attn_mask) in enumerate(samples):
        if si % 10 == 0:
            print(f"  sample {si+1}/{len(samples)}")
        L, d = clean_emb.shape
        x_t = torch.from_numpy(clean_emb).to(device, dtype=torch.float32)

        for ti, (t_val, gamma) in enumerate(zip(t_grid, gamma_grid)):
            alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma)).item())
            sigma = math.sqrt(torch.sigmoid(torch.tensor( gamma)).item())
            eps   = rng.standard_normal((args.n_noise, L, d)).astype(np.float32)
            z_b   = alpha * x_t[None] + sigma * torch.from_numpy(eps).to(device)
            g_b   = torch.full((args.n_noise,), gamma, device=device, dtype=torch.float32)
            sc_in = torch.zeros_like(z_b) if sc else None

            with torch.no_grad():
                result = model(
                    noisy_embeds=z_b,
                    timesteps=g_b,
                    x_self_cond=sc_in,
                    output_hidden_states=True,
                    return_dict=False,
                )
            logits, all_hidden = result
            # all_hidden: tuple of (n_noise, L, dim) tensors, one per layer
            n_layers = len(all_hidden)

            logits_np = logits.cpu().float().numpy()  # [N, L, V]
            nat_top1  = float((np.argmax(logits_np, -1) == gt_ids[None]).mean())
            nat_per_t[ti].append(nat_top1)
            y_per_t[ti].append(np.tile(gt_ids, args.n_noise))

            for li in range(min(n_layers, 13)):
                h_np = all_hidden[li].cpu().float().numpy()  # [N, L, dim]
                h_layers[ti][li].append(h_np.reshape(-1, h_np.shape[-1]))

    # Train probes
    # First check how many layers
    n_hidden_layers = min(sum(1 for ll in h_layers[0] if ll), 13)
    print(f"\n  n_hidden_states returned = {n_hidden_layers}")

    results = []
    for ti, t_val in enumerate(t_grid):
        native_top1 = float(np.mean(nat_per_t[ti]))
        y = np.concatenate(y_per_t[ti], axis=0)
        print(f"\n[t={t_val:.2f}]  native_top1={native_top1:.4f}  n_instances={len(y)}")

        layer_results = {}
        for li in range(n_hidden_layers):
            if not h_layers[ti][li]:
                continue
            h = np.concatenate(h_layers[ti][li], axis=0)
            label = f"block_{li}" if li < 12 else "output_layer"
            print(f"    [{label}] h={h.shape}  training probe ...")
            acc = train_gpu_probe(h, y, n_epochs=args.probe_epochs)
            gap = acc - native_top1
            layer_results[label] = {"probe_acc": round(acc, 5), "gap": round(gap, 5)}
            print(f"    [{label}] probe={acc*100:.2f}%  gap={gap*100:+.2f}pp")

        results.append({
            "t": float(t_val),
            "native_top1": round(native_top1, 5),
            "layer_probe": layer_results,
        })

    out_file = out_dir / "layerwise_results.json"
    with open(out_file, "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\n[saved] {out_file}")

    # Summary table
    n_lyr = n_hidden_layers
    print("\n── probe_acc (%) by layer ──────────────────────────────────────────────")
    keys = [f"block_{i}" for i in range(min(n_lyr, 12))] + (["output_layer"] if n_lyr >= 13 else [])
    hdr  = "  t   |  " + "  ".join(f"B{i:02d}" for i in range(min(n_lyr, 12)))
    if n_lyr >= 13:
        hdr += "  out"
    hdr += "  | native"
    print(hdr)
    for row in results:
        vals = [row["layer_probe"].get(k, {}).get("probe_acc", 0) * 100 for k in keys]
        nat  = row["native_top1"] * 100
        print(f"{row['t']:5.2f} | " + "  ".join(f"{v:5.1f}" for v in vals) + f"  | {nat:6.2f}")


if __name__ == "__main__":
    main()
