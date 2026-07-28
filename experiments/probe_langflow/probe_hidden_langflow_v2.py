"""
probe_hidden_langflow_v2.py — EXP-21v2: Skip-connection separation

Decomposes LangFlow's native logits into backbone and skip contributions:
  native_logits = backbone_logits + c_skip(γ) * z_t @ E.T

Runs five evaluations:
  1. native_top1          — argmax(backbone + skip)       [ground truth]
  2. backbone_top1        — argmax(backbone)               [backbone-only]
  3. skip_top1            — argmax(c_skip * z @ E.T)       [skip-only]
  4. probe_h (h_last)     — linear probe on last hidden     [current EXP-21]
  5. probe_hz ([h,z])     — linear probe on [h_last, z_t]  [full-info probe]
  6. native_init_probe    — probe init'd with native weights, no training  [sanity check]

Usage:
  CUDA_VISIBLE_DEVICES=4 conda run -n elf python \
    experiments/probe_langflow/probe_hidden_langflow_v2.py \
    --checkpoint Continuous-Rivals-Discrete/langflow-owt \
    --n_samples 64 --seq_len 128 --n_noise 4 \
    --out_dir results/exp21v2_langflow
"""

import sys, os, argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROBE_DIR = Path(__file__).parent
_LF_SRC = _PROBE_DIR.parents[1] / "models" / "LangFlow"
for _p in [str(_LF_SRC), str(_PROBE_DIR)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from probe_langflow import (
    load_langflow, encode_with_langflow, load_owt_texts,
    gamma_from_t, softmax_np,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_c_skip(model, gamma_scalar):
    """Compute c_skip scalar for a given gamma value (matches langflow/model.py line 474)."""
    sigma = torch.sigmoid(torch.tensor(gamma_scalar)).sqrt()
    c_skip = ((F.softplus(-sigma) - sigma) / 2).exp()
    return c_skip.item()


def collect_states(model, samples, t_grid, gamma_grid, n_noise, seed, E_all):
    """
    For each (sample, t, noise), collect:
      - h_last: last transformer block hidden state [N*L, H]
      - z_t:    noisy input (in embedding space)    [N*L, d]
      - native_top1: P(argmax(full logits) == y)
      - backbone_top1: P(argmax(full - skip) == y)
      - skip_top1: P(argmax(skip only) == y)
    """
    sc_flag = model.config.self_conditioning
    T = len(t_grid)

    # Output accumulators per t
    h_per_t      = [[] for _ in range(T)]
    z_per_t      = [[] for _ in range(T)]
    y_per_t      = [[] for _ in range(T)]
    nat_per_t    = [[] for _ in range(T)]
    back_per_t   = [[] for _ in range(T)]
    skip_per_t   = [[] for _ in range(T)]

    rng = np.random.default_rng(seed)

    # Embedding matrix E: [V, d] — passed in from load_langflow
    E = E_all.to(device) if isinstance(E_all, torch.Tensor) else torch.from_numpy(E_all).to(device)
    E = E.float()

    for si, (gt_ids, clean_emb, attn_mask) in enumerate(samples):
        if si % 10 == 0:
            print(f"  sample {si+1}/{len(samples)}")
        L, d = clean_emb.shape
        x_torch = torch.from_numpy(clean_emb).to(device, dtype=torch.float32)

        for ti, (t_val, gamma) in enumerate(zip(t_grid, gamma_grid)):
            alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma)).item())
            sigma = math.sqrt(torch.sigmoid(torch.tensor( gamma)).item())
            c_skip = get_c_skip(model, gamma)

            eps = rng.standard_normal((n_noise, L, d)).astype(np.float32)
            z_batch = alpha * x_torch[None] + sigma * torch.from_numpy(eps).to(device)
            # z_batch: [N, L, d]
            gamma_batch = torch.full((n_noise,), gamma, device=device, dtype=torch.float32)
            sc_in = torch.zeros_like(z_batch) if sc_flag else None

            with torch.no_grad():
                result = model(
                    noisy_embeds=z_batch,
                    timesteps=gamma_batch,
                    x_self_cond=sc_in,
                    output_hidden_states=True,
                    return_dict=False,
                )
            native_logits, all_hidden = result
            # native_logits: [N, L, V]
            # all_hidden[-1]: [N, L, H]  (last transformer block, before output_layer)
            h_last = all_hidden[-1]  # [N, L, H]

            # Decompose: skip_contribution = c_skip * z_t @ E.T
            skip_contribution = c_skip * torch.matmul(
                z_batch.float(), E.t().float()
            )  # [N, L, V]

            # backbone = native - skip
            backbone_logits = native_logits.float() - skip_contribution

            # Accuracy for each component
            gt = torch.from_numpy(gt_ids).to(device)  # [L]

            nat_top1  = (native_logits.argmax(-1)   == gt[None]).float().mean().item()
            back_top1 = (backbone_logits.argmax(-1)  == gt[None]).float().mean().item()
            skip_top1 = (skip_contribution.argmax(-1) == gt[None]).float().mean().item()

            nat_per_t[ti].append(nat_top1)
            back_per_t[ti].append(back_top1)
            skip_per_t[ti].append(skip_top1)

            # Collect hidden states for probes
            h_np = h_last.cpu().float().numpy().reshape(-1, h_last.shape[-1])  # [N*L, H]
            z_np = z_batch.cpu().float().numpy().reshape(-1, d)                 # [N*L, d]
            y_flat = np.tile(gt_ids, n_noise)

            h_per_t[ti].append(h_np)
            z_per_t[ti].append(z_np)
            y_per_t[ti].append(y_flat)

    return (
        [np.concatenate(h_per_t[ti]) for ti in range(T)],
        [np.concatenate(z_per_t[ti]) for ti in range(T)],
        [np.concatenate(y_per_t[ti]) for ti in range(T)],
        [float(np.mean(nat_per_t[ti]))  for ti in range(T)],
        [float(np.mean(back_per_t[ti])) for ti in range(T)],
        [float(np.mean(skip_per_t[ti])) for ti in range(T)],
    )


def train_probe(feat, y, n_classes, epochs=50, lr=1e-2, batch_size=2048, val_frac=0.2,
                init_weight=None, init_bias=None, no_train=False):
    """Train a linear probe. If init_weight given, initialize from it (optional no-train eval)."""
    n = len(y)
    n_val = int(n * val_frac)
    idx = np.random.RandomState(42).permutation(n)
    h_tr, y_tr_np = feat[idx[:-n_val]], y[idx[:-n_val]]
    h_v,  y_v_np  = feat[idx[-n_val:]], y[idx[-n_val:]]

    uniq_tr, cnt_tr = np.unique(y_tr_np, return_counts=True)
    keep = set(uniq_tr[cnt_tr >= 1]) & set(np.unique(y_v_np))
    if len(keep) < 2:
        return float("nan"), float("nan")

    keep_sorted = sorted(keep)
    tok2i = {t: i for i, t in enumerate(keep_sorted)}
    mask_tr = np.array([yi in tok2i for yi in y_tr_np])
    mask_v  = np.array([yi in tok2i for yi in y_v_np])
    h_tr, y_tr_np = h_tr[mask_tr], y_tr_np[mask_tr]
    h_v,  y_v_np  = h_v[mask_v],   y_v_np[mask_v]

    K   = len(keep_sorted)
    dim = feat.shape[1]
    probe = nn.Linear(dim, K).to(device)

    if init_weight is not None:
        with torch.no_grad():
            # init_weight is [V, H]; slice to the K classes that appear in data
            keep_idx = torch.tensor(keep_sorted, dtype=torch.long)
            probe.weight.copy_(init_weight[keep_idx].to(device))
            if init_bias is not None:
                probe.bias.copy_(init_bias[keep_idx].to(device))
    else:
        nn.init.zeros_(probe.weight)
        nn.init.zeros_(probe.bias)

    h_v_t  = torch.tensor(h_v,  dtype=torch.float32, device=device)
    y_v_t  = torch.tensor([tok2i[yi] for yi in y_v_np],  dtype=torch.long, device=device)

    if no_train:
        with torch.no_grad():
            acc_no_train = (probe(h_v_t).argmax(1) == y_v_t).float().mean().item()
        return acc_no_train, acc_no_train  # train_acc = val_acc = same thing

    h_tr_t = torch.tensor(h_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor([tok2i[yi] for yi in y_tr_np], dtype=torch.long, device=device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    n_tr = len(y_tr_t)
    for ep in range(epochs):
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch_size):
            ib = perm[i:i+batch_size]
            loss = F.cross_entropy(probe(h_tr_t[ib]), y_tr_t[ib])
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        train_acc = (probe(h_tr_t).argmax(1) == y_tr_t).float().mean().item()
        val_acc   = (probe(h_v_t).argmax(1)  == y_v_t).float().mean().item()
    return train_acc, val_acc


def get_native_head_weights(model):
    """
    Extract native head weights that project to logit space.
    In LangFlow: h_last -> output_layer(h_last, t_cond) -> backbone_logits
    output_layer is a DDiTFinalLayer. For native_init probe we extract its effective
    weight at a representative t. Returns None if cannot extract (complex layer).
    """
    try:
        # DDiTFinalLayer has: linear_1 (in*2 -> in), linear_2 (in -> out)
        # For a rough extraction, try linear_2 weight (output projection)
        ol = model.backbone.output_layer
        # Try to get final linear weight
        if hasattr(ol, 'linear'):
            W = ol.linear.weight.data.clone()  # [out_dim, in_dim]
            b = ol.linear.bias.data.clone() if ol.linear.bias is not None else None
        elif hasattr(ol, 'linear_2'):
            W = ol.linear_2.weight.data.clone()
            b = ol.linear_2.bias.data.clone() if ol.linear_2.bias is not None else None
        else:
            return None, None
        return W, b
    except Exception as e:
        print(f"  [warn] Could not extract native head weights: {e}")
        return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    p.add_argument("--n_samples",   type=int, default=64)
    p.add_argument("--seq_len",     type=int, default=128)
    p.add_argument("--n_noise",     type=int, default=4)
    p.add_argument("--out_dir",     default="results/exp21v2_langflow")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--probe_epochs", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 1.00])

    print(f"[load] LangFlow from {args.checkpoint}")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    gamma_grid = gamma_from_t(t_grid, gamma_min, gamma_max)

    # Verify self_conditioning
    print(f"  self_conditioning = {model.config.self_conditioning}")

    print(f"[data] Loading {args.n_samples} OWT samples")
    texts   = load_owt_texts(args.n_samples)
    samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)

    print("[collect] Running forward passes to collect hidden states + skip decomposition...")
    h_per_t, z_per_t, y_per_t, nat_top1, back_top1, skip_top1 = collect_states(
        model, samples, t_grid, gamma_grid, args.n_noise, args.seed, E_all
    )
    E = E_all.to(device) if isinstance(E_all, torch.Tensor) else torch.from_numpy(E_all).to(device)

    # Try to extract native head weights for native_init probe
    W_native, b_native = get_native_head_weights(model)
    if W_native is not None:
        print(f"  [native init] extracted head weights: W={W_native.shape}, b={b_native.shape if b_native is not None else None}")
    else:
        print("  [native init] could not extract native head weights — skipping init check")

    results = []
    for ti, t_val in enumerate(t_grid):
        h = h_per_t[ti]     # [N, H]
        z = z_per_t[ti]     # [N, d]
        y = y_per_t[ti]     # [N]
        hz = np.concatenate([h, z], axis=1)  # [N, H+d]

        n_cls = int(E.shape[0])
        print(f"\n[t={t_val:.2f}] n={len(y)}, h_dim={h.shape[1]}, z_dim={z.shape[1]}")
        print(f"  native={nat_top1[ti]:.4f}  backbone={back_top1[ti]:.4f}  skip={skip_top1[ti]:.4f}")

        # Probe on h_last only (replication of EXP-21)
        print("  training probe(h_last)...")
        tr_h, val_h = train_probe(h, y, n_cls, epochs=args.probe_epochs)
        print(f"    train={tr_h:.4f}  val={val_h:.4f}")

        # Probe on [h_last, z_t]
        print("  training probe([h_last, z_t])...")
        tr_hz, val_hz = train_probe(hz, y, n_cls, epochs=args.probe_epochs)
        print(f"    train={tr_hz:.4f}  val={val_hz:.4f}")

        # Probe on z_t only (skip-only baseline)
        print("  training probe(z_t only)...")
        tr_z, val_z = train_probe(z, y, n_cls, epochs=args.probe_epochs)
        print(f"    train={tr_z:.4f}  val={val_z:.4f}")

        # Native-init probe: initialize with native weights, NO training
        val_init = float("nan")
        if W_native is not None:
            print("  evaluating native_init probe (no training)...")
            # h_last -> output_layer -> native_logits_backbone
            # W_native maps h_dim -> out_dim (which then maps to vocab)
            # This init may not directly work but tests hook correctness
            _, val_init = train_probe(h, y, n_cls, init_weight=W_native, init_bias=b_native,
                                      no_train=True)
            print(f"    native_init val={val_init:.4f}  (native_top1={nat_top1[ti]:.4f})")

        entry = {
            "t": float(t_val),
            "native_top1":      nat_top1[ti],
            "backbone_top1":    back_top1[ti],  # native minus skip
            "skip_top1":        skip_top1[ti],  # skip only
            "probe_h_train":    tr_h,
            "probe_h_val":      val_h,          # current EXP-21 probe
            "probe_hz_train":   tr_hz,
            "probe_hz_val":     val_hz,         # full-info probe
            "probe_z_train":    tr_z,
            "probe_z_val":      val_z,          # skip-only probe
            "native_init_val":  val_init,       # sanity: init with native weights
            "gap_h":    val_h   - nat_top1[ti],
            "gap_hz":   val_hz  - nat_top1[ti],
            "gap_back": back_top1[ti] - nat_top1[ti],
        }
        results.append(entry)

    out_json = out_dir / "skip_separation.json"
    with open(out_json, "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\n[saved] {out_json}")

    print("\n── EXP-21v2 Summary: Skip-contribution decomposition ───────────────────")
    print(f"{'t':>5}  {'native':>8}  {'backbone':>9}  {'skip':>7}  {'probe_h':>9}  {'probe_hz':>9}  {'init':>8}")
    for r in results:
        print(f"{r['t']:>5.2f}  {r['native_top1']:>8.4f}  {r['backbone_top1']:>9.4f}  "
              f"{r['skip_top1']:>7.4f}  {r['probe_h_val']:>9.4f}  {r['probe_hz_val']:>9.4f}  "
              f"{r['native_init_val']:>8.4f}")

    print("\nKey question: is probe_hz ≈ native?  (if yes, skip separation explains the gap)")
    print("Key question: does backbone > probe_h?  (if yes, output_layer is doing work beyond h_last)")


if __name__ == "__main__":
    main()
