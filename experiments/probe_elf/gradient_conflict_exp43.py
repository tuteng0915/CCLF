#!/usr/bin/env python3
"""
EXP-43: Dual-Path Gradient Conflict Analysis

Tests whether baseline's B11 residual update creates a geometric conflict
between the two downstream paths of h_11:
  - Decode path:        h_11 → proj_kernel → GELU → unembed → token logits
  - Reconstruction path: h_11 → RMSNorm → final_layer.linear → x̂_t

Method:
  1. Interpolation curve: h(α) = h_10 + α·(h_11 - h_10), α ∈ [-0.5, 1.5]
     Compute L_dec(α) and L_rec(α) simultaneously.
     Tradeoff predicts: baseline shows L_dec↑ as L_rec↓ for α: 0→1.
     KD predicts: both improve (or at least L_dec does not worsen).

  2. Gradient conflict: c(h) = cos(∇_h L_dec, ∇_h L_rec) at h = h_11.
     c < 0 → improving decode hurts reconstruction (conflict)
     c > 0 → improving both is geometrically compatible

x* (reconstruction target) = x_hat at t=1.0 (clean limit: z_1 = x_0).

Usage:
    conda run -n elf python3 experiments/probe_elf/gradient_conflict_exp43.py
    conda run -n elf python3 experiments/probe_elf/gradient_conflict_exp43.py --device cuda:0
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

CHECKPOINTS = {
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
    "kd_cr":    "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":      "converted/elf_b-owt-kd2_torch.pt",
}
DATA_ROOT = Path("results")
OUT_DIR = Path("results/exp43_gradient_conflict")

ALPHAS = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
T_VALUES = ["0.200", "0.500", "1.000"]   # analyze at multiple t
T_XSTAR = "1.000"
BATCH_SIZE = 16
N_GRAD_SAMPLE = 4096   # positions sampled for per-position gradient conflict


# ── Primitives ────────────────────────────────────────────────────────────────

def rms_norm(x, weight, eps=1e-6):
    """RMSNorm. x: [..., D], weight: [D] -> [..., D]"""
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return x.float() * rms * weight.float()


def decode_head_forward(h, head):
    """h: [..., 768] -> logits [..., V=32100]"""
    hidden = F.gelu(h.float() @ head["proj_kernel"] + head["proj_bias"],
                    approximate="tanh")
    return hidden @ head["unembed_kernel"] + head["unembed_bias"]


def recon_path_forward(h, norm_w, lin_w, lin_b):
    """h: [..., 768] -> x_hat [..., 512]"""
    return F.linear(rms_norm(h, norm_w), lin_w, lin_b)


def load_weights(ckpt_path, device):
    p = torch.load(ckpt_path, map_location="cpu", weights_only=False)["params"]
    head = {
        "proj_kernel":    p["proj_kernel"].float().to(device),
        "proj_bias":      p["proj_bias"].float().to(device),
        "unembed_kernel": p["unembed_kernel"].float().to(device),
        "unembed_bias":   p["unembed_bias"].float().to(device),
    }
    recon = {
        "norm_w": p["final_layer.norm_final.weight"].float().to(device),
        "lin_w":  p["final_layer.linear.weight"].float().to(device),  # [512, 768]
        "lin_b":  p["final_layer.linear.bias"].float().to(device),    # [512]
    }
    return head, recon


def load_states(ckpt_name, t_str):
    path = DATA_ROOT / f"exp07b_v2_{ckpt_name}" / f"layer_states_t{t_str}.pt"
    d = torch.load(path, map_location="cpu")
    h10 = d["layer_feats"][10].float()    # [256, 1024, 768]
    h11 = d["layer_feats"][11].float()    # [256, 1024, 768]
    y   = d["y_tokens"].long()             # [256, 1024]
    mask = d["attn_mask"].bool()           # [256, 1024]
    x_hat = d["x_hat"].float()            # [256, 1024, 512]
    return h10, h11, y, mask, x_hat


# ── Core computations ─────────────────────────────────────────────────────────

def compute_interp_curve(h10, h11, y, mask, x_star, head, recon, device):
    """
    For each α in ALPHAS, compute (L_dec, L_rec, top1_acc) averaged over
    all valid positions.  Returns dict keyed by str(α).
    """
    delta = h11 - h10   # [N, L, 768]
    N = h10.shape[0]
    results = {}

    for alpha in ALPHAS:
        sum_dec = 0.0
        sum_rec = 0.0
        n_correct = 0
        n_valid = 0

        for i in range(0, N, BATCH_SIZE):
            bh10  = h10[i:i+BATCH_SIZE].to(device)
            bdelta = delta[i:i+BATCH_SIZE].to(device)
            by    = y[i:i+BATCH_SIZE].to(device)
            bm    = mask[i:i+BATCH_SIZE].to(device)
            bxs   = x_star[i:i+BATCH_SIZE].to(device)

            ha = bh10 + alpha * bdelta   # [B, L, 768]

            with torch.no_grad():
                logits = decode_head_forward(ha, head)   # [B, L, V]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    by.reshape(-1), reduction="none"
                ).reshape(by.shape)                       # [B, L]
                top1 = (logits.argmax(-1) == by)

                x_hat_pred = recon_path_forward(ha, recon["norm_w"],
                                                recon["lin_w"], recon["lin_b"])
                rec = ((x_hat_pred - bxs) ** 2).sum(-1)  # [B, L] sum over dim

            nv = bm.sum().item()
            sum_dec    += (ce * bm.float()).sum().item()
            sum_rec    += (rec * bm.float()).sum().item()
            n_correct  += (top1 * bm).sum().item()
            n_valid    += nv

        results[str(round(alpha, 3))] = {
            "l_dec":    sum_dec / n_valid,
            "l_rec":    sum_rec / n_valid,
            "top1_acc": n_correct / n_valid,
        }

    return results


def compute_gradient_conflict(h11, y, mask, x_star, head, recon, device):
    """
    Compute gradient conflict at h_11.

    Two metrics:
      1. per-position: for each sampled position, compute cos(g_dec_i, g_rec_i)
         then average.  Captures local geometric tension.
      2. aggregate: cos(mean(g_dec), mean(g_rec)).  Captures population-level
         gradient alignment.
    """
    # --- Sample positions ---
    rng = torch.Generator()
    rng.manual_seed(42)
    valid_flat = mask.reshape(-1).nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(valid_flat), generator=rng)[:N_GRAD_SAMPLE]
    idx = valid_flat[perm]

    N, L, D = h11.shape
    h_sample  = h11.reshape(-1, D)[idx].to(device)         # [S, 768]
    y_sample  = y.reshape(-1)[idx].to(device)               # [S]
    xs_sample = x_star.reshape(-1, 512)[idx].to(device)    # [S, 512]

    h_var = h_sample.detach().clone().requires_grad_(True)

    # --- Decode gradient ---
    logits = decode_head_forward(h_var, head)               # [S, V]
    l_dec = F.cross_entropy(logits, y_sample)
    l_dec.backward()
    g_dec = h_var.grad.detach().clone()                     # [S, 768]

    # --- Reconstruction gradient ---
    h_var.grad = None
    x_hat_pred = recon_path_forward(h_var, recon["norm_w"],
                                    recon["lin_w"], recon["lin_b"])  # [S, 512]
    l_rec = ((x_hat_pred - xs_sample) ** 2).mean()
    l_rec.backward()
    g_rec = h_var.grad.detach().clone()                     # [S, 768]

    # --- Per-position cosine ---
    cos_pos = F.cosine_similarity(g_dec, g_rec, dim=-1)     # [S]

    # --- Aggregate cosine ---
    cos_agg = F.cosine_similarity(
        g_dec.mean(0, keepdim=True),
        g_rec.mean(0, keepdim=True)
    ).item()

    # --- Gradient norms (for reference) ---
    gnorm_dec = g_dec.norm(dim=-1).mean().item()
    gnorm_rec = g_rec.norm(dim=-1).mean().item()

    return {
        "cos_per_pos_mean":  cos_pos.mean().item(),
        "cos_per_pos_std":   cos_pos.std().item(),
        "cos_per_pos_q10":   cos_pos.quantile(0.10).item(),
        "cos_per_pos_q50":   cos_pos.quantile(0.50).item(),
        "cos_aggregate":     cos_agg,
        "l_dec_at_h11":      l_dec.item(),
        "l_rec_at_h11":      l_rec.item(),
        "gnorm_dec":         gnorm_dec,
        "gnorm_rec":         gnorm_rec,
        "n_sample":          len(idx),
    }


# ── Delta h analysis: direction relative to decode gradient ───────────────────

def compute_delta_alignment(h10, h11, y, mask, x_star, head, recon, device):
    """
    Compute cos(Δh_11, g_dec) and cos(Δh_11, g_rec) where Δh_11 = h_11 - h_10.
    This measures whether the block 11 residual update aligns with the decode
    gradient (positive = helpful for decode) or conflicts (negative = anti-decode).
    """
    rng = torch.Generator()
    rng.manual_seed(42)
    valid_flat = mask.reshape(-1).nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(valid_flat), generator=rng)[:N_GRAD_SAMPLE]
    idx = valid_flat[perm]

    N, L, D = h11.shape
    dh = (h11 - h10).reshape(-1, D)[idx].to(device)        # [S, 768] delta
    h_sample  = h11.reshape(-1, D)[idx].to(device)         # [S, 768]
    y_sample  = y.reshape(-1)[idx].to(device)
    xs_sample = x_star.reshape(-1, 512)[idx].to(device)

    h_var = h_sample.detach().clone().requires_grad_(True)

    logits = decode_head_forward(h_var, head)
    l_dec = F.cross_entropy(logits, y_sample)
    l_dec.backward()
    g_dec = h_var.grad.detach().clone()

    h_var.grad = None
    x_hat_pred = recon_path_forward(h_var, recon["norm_w"],
                                    recon["lin_w"], recon["lin_b"])
    l_rec = ((x_hat_pred - xs_sample) ** 2).mean()
    l_rec.backward()
    g_rec = h_var.grad.detach().clone()

    # cos(Δh, -g_dec): positive means Δh moves in the decode-improving direction
    # (gradient descent direction is -g_dec)
    neg_g_dec = -g_dec
    neg_g_rec = -g_rec

    cos_dh_dec = F.cosine_similarity(dh, neg_g_dec, dim=-1)   # [S]
    cos_dh_rec = F.cosine_similarity(dh, neg_g_rec, dim=-1)   # [S]

    return {
        "cos_delta_h_vs_neg_g_dec_mean": cos_dh_dec.mean().item(),
        "cos_delta_h_vs_neg_g_dec_std":  cos_dh_dec.std().item(),
        "cos_delta_h_vs_neg_g_rec_mean": cos_dh_rec.mean().item(),
        "cos_delta_h_vs_neg_g_rec_std":  cos_dh_rec.std().item(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*50}")
        print(f"Checkpoint: {ckpt_name}")
        head, recon = load_weights(ckpt_path, device)

        # Load x* from t=1.0 (clean limit ≈ true x_0)
        _, _, _, _, x_star = load_states(ckpt_name, T_XSTAR)
        print(f"  x_star loaded from t={T_XSTAR}: shape={x_star.shape}")

        ckpt_results = {}

        for t_str in T_VALUES:
            print(f"\n  t={t_str}")
            h10, h11, y, mask, _ = load_states(ckpt_name, t_str)

            print(f"    Interpolation curve ({len(ALPHAS)} points)...")
            interp = compute_interp_curve(h10, h11, y, mask, x_star, head, recon, device)

            print(f"    Gradient conflict at h11...")
            conflict = compute_gradient_conflict(h11, y, mask, x_star, head, recon, device)

            print(f"    Delta-h alignment...")
            delta_align = compute_delta_alignment(h10, h11, y, mask, x_star, head, recon, device)

            # Key summary
            ic0 = interp["0.0"]
            ic1 = interp["1.0"]
            print(f"    L_dec: {ic0['l_dec']:.4f}(h10) -> {ic1['l_dec']:.4f}(h11)  "
                  f"[{'↑WORSE' if ic1['l_dec'] > ic0['l_dec'] else '↓better'}]")
            print(f"    L_rec: {ic0['l_rec']:.4f}(h10) -> {ic1['l_rec']:.4f}(h11)  "
                  f"[{'↑WORSE' if ic1['l_rec'] > ic0['l_rec'] else '↓better'}]")
            print(f"    Conflict cos: {conflict['cos_per_pos_mean']:.4f} (per-pos mean)  "
                  f"{conflict['cos_aggregate']:.4f} (aggregate)")
            print(f"    Δh vs -g_dec: {delta_align['cos_delta_h_vs_neg_g_dec_mean']:.4f}  "
                  f"Δh vs -g_rec: {delta_align['cos_delta_h_vs_neg_g_rec_mean']:.4f}")

            ckpt_results[t_str] = {
                "interpolation": interp,
                "gradient_conflict": conflict,
                "delta_alignment": delta_align,
            }

        all_results[ckpt_name] = ckpt_results

    out_path = OUT_DIR / "gradient_conflict.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved to {out_path}")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n\n=== FINAL SUMMARY (t=0.500) ===")
    print(f"{'ckpt':<12} {'ΔL_dec(0→1)':>14} {'ΔL_rec(0→1)':>14} "
          f"{'conflict cos':>14} {'Δh·(-g_dec)':>14}")
    print("-" * 68)
    for ckpt in ["baseline", "kd_cr", "kd2"]:
        r = all_results[ckpt]["0.500"]
        ic = r["interpolation"]
        dl_dec = ic["1.0"]["l_dec"] - ic["0.0"]["l_dec"]
        dl_rec = ic["1.0"]["l_rec"] - ic["0.0"]["l_rec"]
        cos    = r["gradient_conflict"]["cos_per_pos_mean"]
        da     = r["delta_alignment"]["cos_delta_h_vs_neg_g_dec_mean"]
        print(f"{ckpt:<12} {dl_dec:>+14.4f} {dl_rec:>+14.4f} {cos:>14.4f} {da:>+14.4f}")


if __name__ == "__main__":
    main()
