#!/usr/bin/env python3
"""
EXP-44: Module Factorial Patching

Tests which module (B11 weights, decode head, recon head) drives the kd_cr vs kd2
difference in oracle accuracy and SC interaction.

Phase 1 (fast): chimeric oracle accuracy + L_rec using stored exp07b_v2 h_10 states.
Phase 2 (slow): SC interaction for key arms via full generation.

Chimeric arms:
  native_kd_cr:        h10=kd_cr, B11=kd_cr, decode=kd_cr, recon=kd_cr
  native_kd2:          h10=kd2,   B11=kd2,   decode=kd2,   recon=kd2
  B11_cr_on_kd2:       h10=kd2,   B11=kd_cr, decode=kd2,   recon=kd2
  B11_kd2_on_cr:       h10=kd_cr, B11=kd2,   decode=kd_cr, recon=kd_cr
  decode_cr_on_kd2:    h10=kd2,   B11=kd2,   decode=kd_cr, recon=kd2
  decode_kd2_on_cr:    h10=kd_cr, B11=kd_cr, decode=kd2,   recon=kd_cr
  recon_cr_on_kd2:     h10=kd2,   B11=kd2,   decode=kd2,   recon=kd_cr
  recon_kd2_on_cr:     h10=kd_cr, B11=kd_cr, decode=kd_cr, recon=kd2

Note on RoPE: exp07b_v2 stores content-only h states (prefix stripped).
The model's feat_rope uses num_empty_token=12, so content tokens at absolute positions
12..12+L-1 get RoPE positions 0..L-1. Our content-only chimeric pass uses a RoPE with
num_empty_token=0, giving identical rotations to the original model.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CHECKPOINTS = {
    "kd_cr": "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":   "converted/elf_b-owt-kd2_torch.pt",
}
DATA_DIRS = {
    "kd_cr": "results/exp07b_v2_kd_cr",
    "kd2":   "results/exp07b_v2_kd2",
}

T_VALS      = [0.3, 0.5]    # t values for Phase 1
N_HEADS     = 12
HEAD_DIM    = 64             # 768 / 12
BATCH_SIZE  = 32             # sequences per forward-pass chunk
OUT_DIR     = Path("results/exp44_module_patch")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_params(ckpt_name: str) -> dict:
    ckpt = torch.load(CHECKPOINTS[ckpt_name], map_location="cpu", weights_only=False)
    return ckpt["params"]


def load_layer_states(ckpt_name: str, t: float) -> dict:
    fname = f"{DATA_DIRS[ckpt_name]}/layer_states_t{t:.3f}.pt"
    return torch.load(fname, map_location="cpu", weights_only=False)


def rms_norm_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def make_rope_buffers(seq_len: int, head_dim: int, device: torch.device):
    """Build RoPE cos/sin for content-only sequence of length seq_len."""
    theta = 10000.0
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32)[:half] / head_dim))
    # pos = 0..seq_len-1 (matches the model's content positions after prefix stripping)
    pos = torch.arange(seq_len, dtype=torch.float32)
    outer = torch.einsum("p, f -> p f", pos, freqs)          # [seq_len, half]
    outer = torch.cat([outer, outer], dim=-1)                  # [seq_len, head_dim] (repeat r=2)
    freqs_cos = torch.cos(outer).to(device)
    freqs_sin = torch.sin(outer).to(device)
    return freqs_cos, freqs_sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """q: [B, n_heads, L, head_dim]; freqs_*: [L, head_dim]"""
    return q * freqs_cos[None, None] + rotate_half(q) * freqs_sin[None, None]


# ── chimeric B11 forward ──────────────────────────────────────────────────────

@torch.no_grad()
def chimeric_b11_forward(
    h: torch.Tensor,          # [B, L, 768] content-only hidden states at block 10 output
    b11_params: dict,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    attn_mask: torch.Tensor,  # [B, L] bool: True=valid
) -> torch.Tensor:
    """Returns chimeric h_11: [B, L, 768]."""
    B, L, D = h.shape
    dev = h.device
    dtype = torch.bfloat16

    h = h.to(dtype)

    def p(key):
        return b11_params[key].to(dev, dtype=dtype)

    # Pre-norm 1 + multi-head attention
    h_norm = rms_norm_fn(h, p("blocks.11.norm1.weight"))
    qkv = F.linear(h_norm, p("blocks.11.attn.qkv.weight"), p("blocks.11.attn.qkv.bias"))
    # [B, L, 3, n_heads, head_dim] → [3, B, n_heads, L, head_dim]
    qkv = qkv.reshape(B, L, 3, N_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # QK-norm
    q = rms_norm_fn(q, p("blocks.11.attn.q_norm.weight"))
    k = rms_norm_fn(k, p("blocks.11.attn.k_norm.weight"))

    # RoPE (content-only positions)
    fc = freqs_cos.to(dtype)
    fs = freqs_sin.to(dtype)
    q = apply_rope(q, fc, fs)
    k = apply_rope(k, fc, fs)

    # Scaled dot-product attention with padding mask
    bool_mask = attn_mask[:, None, None, :].to(dev)          # [B, 1, 1, L]
    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, D)
    attn_out = F.linear(attn_out, p("blocks.11.attn.proj.weight"), p("blocks.11.attn.proj.bias"))
    h = h + attn_out

    # Pre-norm 2 + SwiGLU MLP
    h_norm = rms_norm_fn(h, p("blocks.11.norm2.weight"))
    gate_up = F.linear(h_norm, p("blocks.11.mlp.w12.weight"), p("blocks.11.mlp.w12.bias"))
    gate, up = gate_up.chunk(2, dim=-1)
    mlp_out = F.linear(F.silu(gate) * up, p("blocks.11.mlp.w3.weight"), p("blocks.11.mlp.w3.bias"))
    h = h + mlp_out

    return h


# ── oracle accuracy + L_rec evaluation ───────────────────────────────────────

@torch.no_grad()
def evaluate_h11(
    h11: torch.Tensor,        # [N, L, 768]
    dec_params: dict,
    recon_params: dict,
    y_tokens: torch.Tensor,   # [N, L]
    attn_mask: torch.Tensor,  # [N, L] bool
    x_hat_clean: torch.Tensor, # [N, L, 512] oracle x̂ at t=1.0
    device: torch.device,
) -> tuple[float, float]:
    """Returns (oracle_acc, L_rec)."""
    h11 = h11.float().to(device)

    def p(d, key):
        return d[key].float().to(device)

    # Decode path: h11 [N,L,768] @ proj_kernel [768,512] → GELU → @ unembed_kernel [512,32100]
    hidden = F.gelu(h11 @ p(dec_params, "proj_kernel") + p(dec_params, "proj_bias"),
                    approximate="tanh")
    logits = hidden @ p(dec_params, "unembed_kernel") + p(dec_params, "unembed_bias")
    pred = logits.argmax(dim=-1)

    valid = attn_mask.bool().to(device)
    correct = (pred == y_tokens.to(device)) & valid
    acc = correct.sum().float() / valid.sum().float()

    # Reconstruction path: h11 → RMSNorm(norm_final) → Linear(768→512) → x̂
    h_normed = rms_norm_fn(h11, p(recon_params, "final_layer.norm_final.weight"))
    x_hat = F.linear(h_normed, p(recon_params, "final_layer.linear.weight"),
                     p(recon_params, "final_layer.linear.bias"))
    x_oracle = x_hat_clean.float().to(device)
    diff_sq = (x_hat - x_oracle).pow(2).sum(-1)          # [N, L]
    l_rec = (diff_sq * valid.float()).sum() / valid.sum()

    return acc.item(), l_rec.item()


# ── Phase 1 main ─────────────────────────────────────────────────────────────

def run_phase1(device: torch.device) -> dict:
    print("=== Phase 1: Chimeric Oracle Accuracy + L_rec ===")

    params_cr  = load_params("kd_cr")
    params_kd2 = load_params("kd2")

    # x_hat at t=1.0 (oracle clean embedding) for both checkpoints
    xhat_clean = {
        ck: load_layer_states(ck, 1.0)["x_hat"]   # [256, 1024, 512]
        for ck in ("kd_cr", "kd2")
    }

    all_results = {}

    for t in T_VALS:
        print(f"\n--- t = {t:.1f} ---")

        data_cr  = load_layer_states("kd_cr",  t)
        data_kd2 = load_layer_states("kd2",    t)

        h10_cr  = data_cr["layer_feats"][10].bfloat16()   # [256, 1024, 768]
        h10_kd2 = data_kd2["layer_feats"][10].bfloat16()

        y_cr    = data_cr["y_tokens"]
        y_kd2   = data_kd2["y_tokens"]
        mask_cr  = data_cr["attn_mask"]
        mask_kd2 = data_kd2["attn_mask"]

        L = h10_cr.shape[1]
        freqs_cos, freqs_sin = make_rope_buffers(L, HEAD_DIM, device)

        arms = [
            # (arm_name, h10_ck,  b11_params,  dec_params,  recon_params, y, mask, xhat_oracle)
            ("native_kd_cr",      h10_cr,  params_cr,  params_cr,  params_cr,  y_cr,  mask_cr,  xhat_clean["kd_cr"]),
            ("native_kd2",        h10_kd2, params_kd2, params_kd2, params_kd2, y_kd2, mask_kd2, xhat_clean["kd2"]),
            ("B11_cr_on_kd2",     h10_kd2, params_cr,  params_kd2, params_kd2, y_kd2, mask_kd2, xhat_clean["kd2"]),
            ("B11_kd2_on_cr",     h10_cr,  params_kd2, params_cr,  params_cr,  y_cr,  mask_cr,  xhat_clean["kd_cr"]),
            ("decode_cr_on_kd2",  h10_kd2, params_kd2, params_cr,  params_kd2, y_kd2, mask_kd2, xhat_clean["kd2"]),
            ("decode_kd2_on_cr",  h10_cr,  params_cr,  params_kd2, params_cr,  y_cr,  mask_cr,  xhat_clean["kd_cr"]),
            ("recon_cr_on_kd2",   h10_kd2, params_kd2, params_kd2, params_cr,  y_kd2, mask_kd2, xhat_clean["kd2"]),
            ("recon_kd2_on_cr",   h10_cr,  params_cr,  params_cr,  params_kd2, y_cr,  mask_cr,  xhat_clean["kd_cr"]),
        ]

        t_results = {}
        for arm_name, h10, b11_p, dec_p, recon_p, y, mask, x_oracle in arms:
            N = h10.shape[0]
            acc_list, lrec_list = [], []

            for start in range(0, N, BATCH_SIZE):
                h10_b  = h10[start:start+BATCH_SIZE].to(device)
                mask_b = mask[start:start+BATCH_SIZE].bool()
                y_b    = y[start:start+BATCH_SIZE]
                xo_b   = x_oracle[start:start+BATCH_SIZE]

                h11_b = chimeric_b11_forward(h10_b, b11_p, freqs_cos, freqs_sin, mask_b)
                acc_b, lrec_b = evaluate_h11(h11_b, dec_p, recon_p, y_b, mask_b, xo_b, device)

                # Weight by number of valid tokens for correct aggregate
                n_valid = mask_b.sum().item()
                acc_list.append((acc_b, n_valid))
                lrec_list.append((lrec_b, n_valid))

            total_valid = sum(n for _, n in acc_list)
            acc = sum(a * n for a, n in acc_list) / total_valid
            lrec = sum(l * n for l, n in lrec_list) / total_valid

            t_results[arm_name] = {"oracle_acc": round(acc, 5), "L_rec": round(lrec, 3)}
            print(f"  {arm_name:<25}  acc={acc:.3%}  L_rec={lrec:.1f}")

        all_results[f"t={t}"] = t_results

    return all_results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase1-only", action="store_true", default=True,
                        help="Run Phase 1 only (oracle accuracy; default=True)")
    args = parser.parse_args()

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    results["phase1"] = run_phase1(device)

    out_path = OUT_DIR / "patch_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n=== SUMMARY (t=0.5) ===")
    print(f"{'arm':<25}  {'oracle_acc':>10}  {'L_rec':>8}")
    print("-" * 50)
    for arm, vals in results["phase1"].get("t=0.5", {}).items():
        print(f"{arm:<25}  {vals['oracle_acc']:>10.3%}  {vals['L_rec']:>8.1f}")


if __name__ == "__main__":
    main()
