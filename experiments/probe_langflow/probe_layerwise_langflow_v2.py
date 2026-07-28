"""
EXP-30v2: LangFlow Layer-wise Probe — MLP probe + multi-seed + skip decomposition

PROBLEMS WITH EXP-30:
  1. Single probe seed → no CI on probe accuracy
  2. No MLP probe → cannot distinguish linear vs nonlinear information
  3. No skip decomposition → "+3.9pp vs native" conflates backbone with skip term
  4. Only linear probe → may miss nonlinear representations

FIXES:
  1. Run linear probe with 5 random seeds, report mean ± std
  2. Add 2-layer MLP probe for final layer and selected mid-layer
  3. Skip decomposition: native vs backbone (native - skip) vs skip vs probe_h vs probe_hz
  4. Sequence-level jackknife SE for probe accuracy

Usage (from CCLF root):
  CUDA_VISIBLE_DEVICES=X conda run -n elf python experiments/probe_langflow/probe_layerwise_langflow_v2.py \\
      --checkpoint Continuous-Rivals-Discrete/langflow-owt \\
      --n_samples 64 --seq_len 128 --n_noise 4 \\
      --n_probe_seeds 5 \\
      --out_dir results/exp30v2_langflow
"""

import sys, os, argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_PROBE_DIR = Path(__file__).parent
_LF_SRC    = _PROBE_DIR.parents[1] / "models" / "LangFlow"
for _p in [str(_LF_SRC)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_PROBE_DIR))

from probe_langflow import load_langflow, load_owt_texts, gamma_from_t, encode_with_langflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── c_skip computation (from EXP-21v2) ─────────────────────────────────────────
def compute_c_skip(gamma: float) -> float:
    """
    LangFlow skip coefficient: c_skip(γ) = exp((softplus(-σ) - σ) / 2)
    where σ = sqrt(sigmoid(γ))
    """
    import torch
    sigma = math.sqrt(torch.sigmoid(torch.tensor(gamma)).item())
    softplus_neg_sigma = math.log(1 + math.exp(-sigma))
    return math.exp((softplus_neg_sigma - sigma) / 2.0)


# ── Probe training utilities ─────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, d: int, V: int):
        super().__init__()
        self.linear = nn.Linear(d, V, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self, d: int, V: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, V),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_probe(h: np.ndarray, y: np.ndarray,
                model_cls, model_kwargs: dict,
                seed: int = 0,
                n_epochs: int = 30, lr: float = 1e-2,
                batch_size: int = 4096) -> float:
    """Train a probe (linear or MLP), return held-out top-1 accuracy."""
    torch.manual_seed(seed)
    n = len(y)
    idx = np.random.RandomState(seed).permutation(n)
    n_tr = int(n * 0.8)

    h_tr = torch.tensor(h[idx[:n_tr]],  dtype=torch.float32, device=device)
    y_tr = torch.tensor(y[idx[:n_tr]],  dtype=torch.long,    device=device)
    h_va = torch.tensor(h[idx[n_tr:]], dtype=torch.float32, device=device)
    y_va = torch.tensor(y[idx[n_tr:]], dtype=torch.long,    device=device)

    V = int(y.max()) + 1
    model = model_cls(d=h.shape[1], V=V, **model_kwargs).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()

    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(y_tr), device=device)
        for i in range(0, len(y_tr), batch_size):
            b = perm[i:i+batch_size]
            opt.zero_grad()
            ce(model(h_tr[b]), y_tr[b]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        acc = float((model(h_va).argmax(1) == y_va).float().mean())
    return acc


def train_probe_multi_seed(h: np.ndarray, y: np.ndarray,
                           model_cls, model_kwargs: dict,
                           n_seeds: int = 5, **train_kwargs) -> dict:
    """Train probe with multiple seeds, return mean/std."""
    accs = [train_probe(h, y, model_cls, model_kwargs, seed=s, **train_kwargs)
            for s in range(n_seeds)]
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "all": accs}


def jackknife_se(per_seq_acc: list) -> float:
    """Sequence-level jackknife SE for mean probe accuracy."""
    n = len(per_seq_acc)
    arr = np.array(per_seq_acc)
    overall = arr.mean()
    jackknife_means = np.array([(arr[:i].sum() + arr[i+1:].sum()) / (n-1) for i in range(n)])
    pseudo = n * overall - (n - 1) * jackknife_means
    return float(np.std(pseudo, ddof=1) / np.sqrt(n))


def compute_skip_decomposition(
        model, E_all: torch.Tensor,
        h_last: np.ndarray, z_t_np: np.ndarray, native_logits: np.ndarray,
        gt_ids: np.ndarray, gamma: float, y: np.ndarray) -> dict:
    """
    Compute 5-condition accuracy for final layer:
      native:   argmax(native_logits)
      backbone: argmax(native_logits - c_skip * z_t @ E.T)
      skip:     argmax(c_skip * z_t @ E.T)
      probe_h:  trained linear probe on h_last
      probe_hz: trained linear probe on [h_last, z_t]

    h_last, z_t_np: [N, L, d] or [N*L, d]
    native_logits: [N, L, V] or [N*L, V]
    y: [N*L]
    """
    E = E_all.to(device).float()
    c_skip = compute_c_skip(gamma)

    N_flat = native_logits.shape[0]
    nat_np  = native_logits.reshape(-1, native_logits.shape[-1])  # [NL, V]

    z_flat  = z_t_np.reshape(-1, z_t_np.shape[-1]).astype(np.float32)       # [NL, d]
    h_flat  = h_last.reshape(-1, h_last.shape[-1]).astype(np.float32)        # [NL, d]

    # skip contribution: c_skip * z @ E.T
    z_t_gpu    = torch.tensor(z_flat, device=device, dtype=torch.float32)
    skip_logits = (c_skip * torch.matmul(z_t_gpu, E.t())).cpu().numpy()  # [NL, V]

    back_logits = nat_np - skip_logits    # backbone-only logits

    nat_top1  = float((np.argmax(nat_np,    -1) == y).mean())
    back_top1 = float((np.argmax(back_logits, -1) == y).mean())
    skip_top1 = float((np.argmax(skip_logits, -1) == y).mean())

    # Linear probe on h_last
    probe_h_acc  = train_probe(h_flat, y, LinearProbe, {}, seed=0, n_epochs=30)
    # Linear probe on [h_last, z_t]
    hz = np.concatenate([h_flat, z_flat], axis=1)
    probe_hz_acc = train_probe(hz, y, LinearProbe, {}, seed=0, n_epochs=30)

    return {
        "c_skip":      float(c_skip),
        "native_top1": nat_top1,
        "backbone_top1": back_top1,
        "skip_top1":   skip_top1,
        "probe_h_top1":  probe_h_acc,
        "probe_hz_top1": probe_hz_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",    default="Continuous-Rivals-Discrete/langflow-owt")
    ap.add_argument("--n_samples",     type=int, default=64)
    ap.add_argument("--seq_len",       type=int, default=128)
    ap.add_argument("--n_noise",       type=int, default=4)
    ap.add_argument("--n_probe_seeds", type=int, default=5)
    ap.add_argument("--probe_epochs",  type=int, default=30)
    ap.add_argument("--out_dir",       default="results/exp30v2_langflow")
    ap.add_argument("--mlp_hidden",    type=int, default=256)
    ap.add_argument("--layers_mlp",    nargs="+", type=int, default=[5, 8, 11],
                    help="Which layer indices to also run MLP probe on (0-indexed blocks)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print("[EXP-30v2] LangFlow layer-wise probe v2")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    sc = model.config.self_conditioning
    E_all_t = E_all if isinstance(E_all, torch.Tensor) else torch.from_numpy(E_all)
    E_all_t = E_all_t.to(device).float()
    print(f"  γ range: [{gamma_min:.2f}, {gamma_max:.2f}]  self_cond={sc}")
    print(f"  E shape: {E_all_t.shape}")

    t_grid    = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 1.00])
    gamma_grid = gamma_from_t(t_grid, gamma_min, gamma_max)

    texts   = load_owt_texts(args.n_samples)
    samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)
    print(f"  Loaded {len(samples)} samples")

    rng = np.random.default_rng(0)
    T   = len(t_grid)

    # Accumulators
    h_layers    = [[[] for _ in range(13)] for _ in range(T)]  # 12 blocks + output
    z_per_t     = [[] for _ in range(T)]
    nat_logits_t = [[] for _ in range(T)]
    y_per_t     = [[] for _ in range(T)]

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
            logits_np = logits.cpu().float().numpy()   # [N, L, V]
            z_np      = z_b.cpu().float().numpy()       # [N, L, d]

            y_per_t[ti].append(np.tile(gt_ids, args.n_noise))
            nat_logits_t[ti].append(logits_np.reshape(-1, logits_np.shape[-1]))
            z_per_t[ti].append(z_np.reshape(-1, z_np.shape[-1]))

            n_layers = len(all_hidden)
            for li in range(min(n_layers, 13)):
                h_np = all_hidden[li].cpu().float().numpy()  # [N, L, d_li]
                h_layers[ti][li].append(h_np.reshape(-1, h_np.shape[-1]))

    # ── Train probes ─────────────────────────────────────────────────────────
    n_hidden = min(sum(1 for ll in h_layers[0] if ll), 13)
    print(f"\n  n_hidden_states = {n_hidden}")

    results = []
    for ti, t_val in enumerate(t_grid):
        y         = np.concatenate(y_per_t[ti], axis=0)
        nat_lgt   = np.concatenate(nat_logits_t[ti], axis=0)
        z_flat    = np.concatenate(z_per_t[ti], axis=0)
        nat_top1  = float((np.argmax(nat_lgt, -1) == y).mean())
        print(f"\n[t={t_val:.2f}]  native_top1={nat_top1:.4f}  n_instances={len(y)}")

        layer_results = {}

        for li in range(n_hidden):
            if not h_layers[ti][li]:
                continue
            h = np.concatenate(h_layers[ti][li], axis=0)
            label = f"block_{li}" if li < 12 else "output_layer"

            # Linear probe — multi-seed
            lin_res = train_probe_multi_seed(
                h, y, LinearProbe, {},
                n_seeds=args.n_probe_seeds, n_epochs=args.probe_epochs, lr=1e-2,
            )
            gap = lin_res["mean"] - nat_top1
            print(f"    [{label}] linear_probe={lin_res['mean']*100:.2f}±{lin_res['std']*100:.2f}%  "
                  f"gap={gap*100:+.2f}pp")

            row = {"linear": lin_res, "gap_to_native": float(gap)}

            # MLP probe on selected layers
            if li in (args.layers_mlp or []) or (li == n_hidden - 1):
                mlp_res = train_probe_multi_seed(
                    h, y, MLPProbe, {"hidden": args.mlp_hidden},
                    n_seeds=3, n_epochs=args.probe_epochs, lr=5e-3,
                )
                mlp_gap = mlp_res["mean"] - lin_res["mean"]
                print(f"    [{label}] mlp_probe={mlp_res['mean']*100:.2f}±{mlp_res['std']*100:.2f}%  "
                      f"mlp_vs_linear={mlp_gap*100:+.2f}pp")
                row["mlp"] = mlp_res
                row["mlp_vs_linear"] = float(mlp_gap)

            layer_results[label] = row

        # Skip decomposition for final hidden layer (block_11 / output)
        final_li = min(n_hidden - 1, 11)  # last transformer block (not output_layer)
        label_final = f"block_{final_li}" if final_li < 12 else "output_layer"
        h_final = np.concatenate(h_layers[ti][final_li], axis=0) if h_layers[ti][final_li] else None

        skip_res = None
        if h_final is not None:
            print(f"    [skip_decomp] running on {label_final} (h: {h_final.shape}) ...")
            # Reconstruct per-position z for skip computation
            try:
                skip_res = compute_skip_decomposition(
                    model=model, E_all=E_all_t,
                    h_last=h_final, z_t_np=z_flat,
                    native_logits=nat_lgt,
                    gt_ids=y, gamma=float(gamma_grid[ti]),
                    y=y,
                )
                print(f"      native={skip_res['native_top1']*100:.2f}%  "
                      f"backbone={skip_res['backbone_top1']*100:.2f}%  "
                      f"skip={skip_res['skip_top1']*100:.2f}%  "
                      f"probe_h={skip_res['probe_h_top1']*100:.2f}%  "
                      f"probe_hz={skip_res['probe_hz_top1']*100:.2f}%  "
                      f"c_skip={skip_res['c_skip']:.4f}")
            except Exception as e:
                print(f"      [warn] skip decomp failed: {e}")
                skip_res = None

        results.append({
            "t":            float(t_val),
            "gamma":        float(gamma_grid[ti]),
            "native_top1":  float(nat_top1),
            "layer_probe":  layer_results,
            "skip_decomp":  skip_res,
        })

    # ── Save ─────────────────────────────────────────────────────────────────
    out_file = out_dir / "layerwise_v2.json"
    with open(out_file, "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\n[saved] {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── linear probe mean (%) by layer ───────────────────────────────────────")
    keys = [f"block_{i}" for i in range(min(n_hidden, 12))]
    if n_hidden >= 13:
        keys.append("output_layer")
    hdr = "  t   | " + " ".join(f"B{i:02d}" for i in range(len(keys) - 1))
    if "output_layer" in keys:
        hdr += "  out"
    hdr += " | native"
    print(hdr)
    for row in results:
        vals = [row["layer_probe"].get(k, {}).get("linear", {}).get("mean", 0) * 100
                for k in keys]
        print(f"{row['t']:5.2f} | " + " ".join(f"{v:5.1f}" for v in vals) +
              f" | {row['native_top1']*100:6.2f}")

    print("\n── skip decomposition (final transformer layer) ──────────────────────────")
    print("  t   | c_skip | backbone | skip  | probe_h | probe_hz | native")
    for row in results:
        sd = row.get("skip_decomp")
        if sd:
            print(f"{row['t']:5.2f} | {sd['c_skip']:6.4f} | "
                  f"{sd['backbone_top1']*100:7.2f}% | "
                  f"{sd['skip_top1']*100:5.2f}% | "
                  f"{sd['probe_h_top1']*100:7.2f}% | "
                  f"{sd['probe_hz_top1']*100:8.2f}% | "
                  f"{sd['native_top1']*100:6.2f}%")


if __name__ == "__main__":
    main()
