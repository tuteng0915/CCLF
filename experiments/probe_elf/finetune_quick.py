#!/usr/bin/env python3
"""
Quick fine-tuning driver for D1 and D3 experiments.

Loads a checkpoint, runs N_STEPS of training with additional loss terms
(intermediate_rec or gram_align), saves the fine-tuned checkpoint, and
reports loss curves.

Usage:
    # D1: intermediate L_rec at B10
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/finetune_quick.py \
        --mode d1 --device cuda:6 --steps 500

    # D3: Gram alignment
    conda run -n elf python3 experiments/probe_elf/finetune_quick.py \
        --mode d3 --device cuda:7 --steps 500
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import add_noise, sample_timesteps

# ── config ─────────────────────────────────────────────────────────────────────
CK_KD_CR = REPO_ROOT / "converted/elf_b-owt-kd-cr_torch.pt"
OUT_DIR   = Path("results/finetune_quick")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=128,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)
PREFIX_SZ = 12   # 4 mode + 4 time + 4 SC-cfg tokens

LR         = 3e-5
N_STEPS    = 500
BATCH_SIZE = 8
LATENT_STD = 0.2
T_EPS      = 0.05


# ── synthetic batch generator (avoids HuggingFace dataloader complexity) ───────

def synthetic_batch(batch_size, seq_len, d_model, device):
    """Generate a random "clean embedding" batch (x0) and token IDs."""
    x0 = torch.randn(batch_size, seq_len, d_model, device=device) * LATENT_STD
    return x0


# ── D1 loss: intermediate reconstruction at B10 ────────────────────────────────

def compute_d1_loss(model, x0, t, noise, denoiser_z, device):
    """Captures h_10 during forward pass and computes MSE(final_layer(h_10), x0)."""
    _h_captured = {}

    def _hook(module, inp, out):
        _h_captured[10] = out

    handle = model.blocks[10].register_forward_hook(_hook)

    # Minimal forward: [denoiser_z, zeros_sc] input (no SC for simplicity)
    B, S, D = denoiser_z.shape
    zeros_sc = torch.zeros_like(denoiser_z)
    z_input = torch.cat([denoiser_z, zeros_sc], dim=-1)

    sc_scale = torch.ones(B, dtype=denoiser_z.dtype, device=device)
    use_bf16 = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        net_out, _, _ = model(z_input, t, deterministic=True,
                              self_cond_cfg_scale=sc_scale)

    handle.remove()

    h_N_full = _h_captured[10]               # [B, PREFIX_SZ + S, 768]
    h_N      = h_N_full[:, PREFIX_SZ:, :]    # [B, S, 768]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        x_hat_N = model.final_layer(h_N)     # [B, S, 512]

    rec_per_tok = ((x_hat_N.float() - x0.float()) ** 2).mean(-1)  # [B, S]
    return rec_per_tok.mean(), net_out.detach()


# ── D3 loss: Gram-matrix alignment ─────────────────────────────────────────────

def compute_d3_loss(model):
    """||G_F/tr_F - G_P/tr_P||_F^2 where G=Gram matrix."""
    F_w = model.final_layer.linear.weight.float()     # [512, 768]
    P_w = model.self_cond_proj.weight[:, 512:].float()  # [512, 512]
    gram_F = (F_w @ F_w.T) / F_w.shape[1]
    gram_P = (P_w @ P_w.T) / P_w.shape[1]
    tr_F = gram_F.diagonal().mean().clamp(min=1e-6)
    tr_P = gram_P.diagonal().mean().clamp(min=1e-6)
    return ((gram_F / tr_F - gram_P / tr_P) ** 2).mean()


# ── denoising L2 loss ──────────────────────────────────────────────────────────

def compute_denoiser_loss(model, x0, t, device):
    """Standard x0-prediction L2 loss (ELF predicts x0 directly, not velocity)."""
    noise = torch.randn_like(x0)
    t_exp = t.reshape(-1, 1, 1)
    denoiser_z = t_exp * x0 + (1 - t_exp) * noise * 2.0   # denoiser_noise_scale=2.0

    B, S, D = denoiser_z.shape
    zeros_sc = torch.zeros_like(denoiser_z)
    z_input  = torch.cat([denoiser_z, zeros_sc], dim=-1)
    sc_scale = torch.ones(B, dtype=denoiser_z.dtype, device=device)

    use_bf16 = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        net_out, _, _ = model(z_input, t, deterministic=True, self_cond_cfg_scale=sc_scale)

    # net_out = predicted x0; minimize reconstruction MSE
    return ((net_out.float() - x0.float()) ** 2).mean(), denoiser_z


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["d1", "d3"], required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lambda_d1", type=float, default=0.5)
    parser.add_argument("--lambda_d3", type=float, default=0.1)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_prefix = OUT_DIR / args.mode

    print(f"Loading kd_cr checkpoint...")
    ckpt = torch.load(CK_KD_CR, map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.train().to(device)

    # For D3: measure initial Gram alignment before training
    if args.mode == "d3":
        with torch.no_grad():
            init_align = compute_d3_loss(model).item()
        print(f"Initial Gram alignment loss: {init_align:.6f}")

    # Optimizer: only train final_layer + self_cond_proj for D3;
    # train all for D1 (need gradient flow through blocks)
    if args.mode == "d3":
        params = list(model.final_layer.parameters()) + list(model.self_cond_proj.parameters())
        print(f"D3: optimizing final_layer + self_cond_proj only ({sum(p.numel() for p in params):,} params)")
    else:
        params = list(model.parameters())
        print(f"D1: optimizing all params ({sum(p.numel() for p in params):,} params)")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    log = {"steps": [], "total_loss": [], "aux_loss": [], "l2_loss": []}

    print(f"\nFine-tuning [{args.mode.upper()}] for {args.steps} steps...")
    for step in tqdm(range(args.steps)):
        opt.zero_grad()

        x0 = synthetic_batch(BATCH_SIZE, MODEL_CFG["max_length"], 512, device)

        # Sample timesteps (uniform — faster convergence for diagnostics)
        t = torch.rand(BATCH_SIZE, device=device) * (1 - T_EPS) + T_EPS

        # Standard denoising L2 loss (regularizer)
        l2_loss, denoiser_z = compute_denoiser_loss(model, x0, t, device)

        # Mode-specific auxiliary loss
        if args.mode == "d1":
            aux_loss, _ = compute_d1_loss(model, x0, t, None, denoiser_z, device)
            total_loss = args.lambda_l2 * l2_loss + args.lambda_d1 * aux_loss
        else:  # d3
            aux_loss = compute_d3_loss(model)
            total_loss = args.lambda_l2 * l2_loss + args.lambda_d3 * aux_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        if step % 50 == 0:
            log["steps"].append(step)
            log["total_loss"].append(total_loss.item())
            log["aux_loss"].append(aux_loss.item())
            log["l2_loss"].append(l2_loss.item())
            tqdm.write(f"  step {step:4d}  total={total_loss.item():.4f}  "
                       f"aux={aux_loss.item():.4f}  l2={l2_loss.item():.4f}")

    # Final alignment check for D3
    if args.mode == "d3":
        with torch.no_grad():
            final_align = compute_d3_loss(model).item()
        print(f"\nGram alignment: {init_align:.6f} → {final_align:.6f}  "
              f"(Δ={final_align - init_align:+.6f})")

    # Save fine-tuned checkpoint
    ck_out = out_prefix.with_suffix(".pt")
    torch.save({"params": {k: v.cpu() for k, v in model.state_dict().items()}}, ck_out)
    print(f"Saved fine-tuned checkpoint → {ck_out}")

    # Save log
    log_out = out_prefix.with_name(f"{out_prefix.name}_log.json")
    with open(log_out, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Loss log → {log_out}")


if __name__ == "__main__":
    main()
