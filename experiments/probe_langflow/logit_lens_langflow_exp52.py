#!/usr/bin/env python3
"""
EXP-52: LangFlow Logit Lens (EXP-38 analogue)

Apply LangFlow's output_layer to each block's intermediate hidden state,
and measure top-1 oracle accuracy at each depth.

Two conditions:
  1. backbone-only: output_layer(h_i, c) — backbone contribution only
  2. full (+ skip): backbone logits + c_skip * (z @ E.T) — what the model actually uses

EXP-21v2 found skip dominates (92.4%). Expected result:
  - backbone-only: near-zero accuracy at all layers (confirming skip story)
  - full: roughly constant, slight increase at final block
  - Contrast with ELF (kd_cr): monotone 40% → 99.5% across blocks — very different!

Usage (from CCLF root):
    CUDA_VISIBLE_DEVICES=X conda run -n elf python \
        experiments/probe_langflow/logit_lens_langflow_exp52.py \
        --device cuda:0 \
        --out_dir results/exp52_langflow_logit_lens
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROBE_DIR = Path(__file__).parent
_LF_SRC    = _PROBE_DIR.parents[1] / "models" / "LangFlow"
for _p in [str(_LF_SRC)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_PROBE_DIR))

from probe_langflow import (
    load_langflow, encode_with_langflow, load_owt_texts,
    gamma_from_t, softmax_np,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_c_skip(gamma_val: float) -> float:
    """c_skip(γ) = exp((softplus(-σ) - σ) / 2), σ = sqrt(sigmoid(γ))."""
    import torch as _t
    sigma = math.sqrt(_t.sigmoid(_t.tensor(gamma_val)).item())
    return math.exp((math.log(1 + math.exp(-sigma)) - sigma) / 2.0)


@torch.no_grad()
def logit_lens_at_t(model, z_t: torch.Tensor, gamma_val: float,
                    sc_in: torch.Tensor, gt_ids: torch.Tensor):
    """
    Apply output_layer to every block's hidden state.
    Returns dict: {depth_i: {"backbone_top1": float, "full_top1": float}}

    depth_i: 0 = after embedding (before block 0),
             1..12 = after block 0..11
    """
    B, L, D = z_t.shape
    sigma = torch.full((B,), gamma_val, device=z_t.device)

    # Forward with hidden states
    model_out = model(
        noisy_embeds=z_t,
        timesteps=sigma,
        x_self_cond=sc_in,
        output_hidden_states=True,
        return_dict=False,
    )
    # model_out = (final_logits, all_hidden_states)
    final_logits = model_out[0]            # [B, L, V] — includes skip
    all_hs = model_out[1]                  # list of 13 tensors [B, L, D]

    # Precompute skip logits and c_skip (constant across depths)
    c_skip_val = compute_c_skip(gamma_val)
    embedding = model._get_embedding_matrix()  # [V, D]
    skip_logits = torch.matmul(z_t.float(), embedding.t().float()).to(z_t.dtype)
    # [B, L, V]

    # backbone t_cond  (recompute; same as inside backbone forward)
    backbone = model.backbone
    t_cond = F.silu(backbone.sigma_map(sigma))  # [B, cond_dim]

    results = {}
    for depth, h in enumerate(all_hs):
        # Apply output_layer to h_depth
        backbone_logits = backbone.output_layer(h.float(), c=t_cond)  # [B, L, V]

        # Backbone-only top-1
        backbone_pred = backbone_logits.argmax(dim=-1)   # [B, L]
        backbone_top1 = (backbone_pred == gt_ids).float().mean().item()

        # Full logits (+ skip)
        full_logits = backbone_logits + c_skip_val * skip_logits.float()
        full_pred   = full_logits.argmax(dim=-1)
        full_top1   = (full_pred == gt_ids).float().mean().item()

        results[depth] = {
            "backbone_top1": backbone_top1,
            "full_top1":     full_top1,
        }

    # Also record skip-only top-1 for reference
    skip_pred = skip_logits.argmax(dim=-1)
    skip_top1 = (skip_pred == gt_ids).float().mean().item()
    results["skip_only"] = {"backbone_top1": 0.0, "full_top1": skip_top1}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Continuous-Rivals-Discrete/langflow-owt")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_noise", type=int, default=1,
                        help="Fixed noise draws per sample (keep 1 for consistency with EXP-22)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="results/exp52_langflow_logit_lens")
    args = parser.parse_args()

    global device
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # t grid matching EXP-38 for ELF (6 key t values)
    T_VALUES = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
    SEED = 42

    print(f"Loading LangFlow from {args.checkpoint}")
    model, tokenizer, gamma_min, gamma_max, E_all = load_langflow(args.checkpoint)
    model = model.to(device).eval()

    sc = model.config.self_conditioning
    print(f"  self_conditioning={sc}")

    print(f"Loading {args.n_samples} OWT samples")
    texts = load_owt_texts(args.n_samples)
    samples = encode_with_langflow(texts, tokenizer, model, args.seq_len)

    n_blocks = len(model.backbone.blocks)  # 12
    depth_names = [f"h{i}" for i in range(n_blocks + 1)] + ["skip_only"]

    # Aggregate: sum across samples/positions then divide
    # {t_str: {depth: {"backbone_top1": float, "full_top1": float}}}
    agg = {str(t): {} for t in T_VALUES}

    for ti, t_val in enumerate(T_VALUES):
        gamma_val = float(gamma_from_t(np.array([t_val]), gamma_min, gamma_max)[0])
        alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma_val)).item())
        sigma_val = math.sqrt(torch.sigmoid(torch.tensor(gamma_val)).item())

        depth_sums = {d: {"backbone_top1": 0.0, "full_top1": 0.0}
                      for d in list(range(n_blocks + 1)) + ["skip_only"]}
        n_total = 0

        rng = np.random.default_rng(SEED)

        for si, sample in enumerate(samples):
            gt_ids, clean_emb, attn_mask = sample
            L, D = clean_emb.shape

            # Fixed noise draw
            eps = rng.standard_normal((L, D)).astype(np.float32)
            z_t = torch.from_numpy(
                (alpha * clean_emb + sigma_val * eps)[None]).to(device)  # [1, L, D]

            gt_tensor = torch.from_numpy(gt_ids).unsqueeze(0).to(device)  # [1, L]
            sigma_t   = torch.full((1,), gamma_val, device=device, dtype=torch.float32)
            sc_in     = torch.zeros_like(z_t) if sc else None

            depth_res = logit_lens_at_t(model, z_t, gamma_val, sc_in, gt_tensor)
            for depth_key, vals in depth_res.items():
                depth_sums[depth_key]["backbone_top1"] += vals["backbone_top1"] * L
                depth_sums[depth_key]["full_top1"]     += vals["full_top1"] * L
            n_total += L

        # Average
        for depth_key in depth_sums:
            for k in depth_sums[depth_key]:
                depth_sums[depth_key][k] /= n_total

        agg[str(t_val)] = depth_sums

        # Print summary table
        print(f"\nt={t_val:.2f}  (γ={gamma_val:.2f})  skip_only: {depth_sums['skip_only']['full_top1']:.3f}")
        print(f"  {'depth':<8} {'backbone':>10} {'full(+skip)':>12}")
        for depth in range(n_blocks + 1):
            b = depth_sums[depth]["backbone_top1"]
            f = depth_sums[depth]["full_top1"]
            print(f"  h{depth:<7} {b:>10.4f} {f:>12.4f}")

    out_path = out_dir / "logit_lens.json"
    with open(out_path, "w") as fp:
        json.dump({"results": agg, "args": vars(args), "n_blocks": n_blocks}, fp, indent=2)
    print(f"\nSaved → {out_path}")

    # Print cross-t summary
    print("\n=== EXP-52 Summary: full_top1 by depth and t ===")
    header = f"{'depth':<8}" + "".join(f"  t={t:.2f}" for t in T_VALUES)
    print(header)
    print("-" * len(header))
    for depth in list(range(n_blocks + 1)) + ["skip_only"]:
        row = f"{'h' + str(depth) if depth != 'skip_only' else 'skip':<8}"
        for t in T_VALUES:
            row += f"  {agg[str(t)][depth]['full_top1']:>7.3f}"
        print(row)


if __name__ == "__main__":
    main()
