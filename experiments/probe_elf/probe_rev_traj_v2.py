"""
EXP-01v2: Protocol B with correct decode-path G(t) on L11 hidden states.

Uses saved z_t from EXP-01 trajectories. At each step (t, z_t):
  1. Run backbone with zero self-cond on z_t to get L11 hidden state h_11
  2. Apply decode path: GELU(h_11 @ proj_kernel + proj_bias) @ unembed_kernel → logits
  3. G_B(t) = frac_correct where argmax(logits) == y_proxy
     where y_proxy = final-step decode-path prediction

Usage (from ELF-torch root):
  CUDA_VISIBLE_DEVICES=4 python experiments/probe_elf/probe_rev_traj_v2.py \
    --traj_dir results/exp01/trajectories \
    --checkpoint converted/elf_b-owt-kd-cr_torch.pt \
    --output_dir results/exp01v2
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_model_and_weights(checkpoint_path, device):
    from configs.config import load_config_from_yaml
    from modules.model import ELF_models
    from utils.checkpoint_utils import load_checkpoint
    from utils.train_utils import TrainState, get_optimizer
    from modules.t5_encoder import T5EncoderConfig

    cfg_path = os.path.join(os.path.dirname(__file__),
                            "../../src/configs/training_configs/eval_exp13.yml")
    config = load_config_from_yaml(cfg_path)
    encoder_config = T5EncoderConfig.from_pretrained(config.encoder_model_name)

    _ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _params = _ckpt.get("params", _ckpt)
    vocab_size = _params["unembed_kernel"].shape[1]

    # Load decode weights
    decode_weights = {
        "proj_kernel":    _params["proj_kernel"].float(),
        "proj_bias":      _params["proj_bias"].float(),
        "unembed_kernel": _params["unembed_kernel"].float(),
        "unembed_bias":   _params["unembed_bias"].float(),
    }
    del _ckpt, _params

    model = ELF_models[getattr(config, "model", "ELF-B")](
        text_encoder_dim=encoder_config.d_model,
        max_length=getattr(config, "max_length", 1024),
        attn_drop=0.0, proj_drop=0.0,
        num_time_tokens=getattr(config, "num_time_tokens", 4),
        num_self_cond_cfg_tokens=getattr(config, "num_self_cond_cfg_tokens", 4),
        vocab_size=vocab_size,
        num_model_mode_tokens=getattr(config, "num_model_mode_tokens", 4),
        bottleneck_dim=getattr(config, "bottleneck_dim", 128),
    ).to(device).eval()

    optimizer = get_optimizer(model, config, lr=1e-4)
    g = torch.Generator(device="cpu").manual_seed(42)
    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=None,
        ema_params1=TrainState.init_ema(model), step=0, epoch=0,
        dropout_generator=g,
    )
    state, _ = load_checkpoint(checkpoint_path, state)
    return state.model, decode_weights, config


@torch.no_grad()
def get_l11_hidden(model, z_t, t_val, device, batch_size=8):
    """
    Run backbone on z_t with zero self-cond, capture L11 hidden state via hook.
    Returns: (B, L, 768) float32 tensor.
    """
    captured = {}

    def hook(mod, inp, out):
        # out: (B, total_tokens, 768)
        captured["h11"] = out.float().cpu()

    handle = model.blocks[11].register_forward_hook(hook)

    B, L, d = z_t.shape
    all_h11 = []
    sc_scale = torch.ones(batch_size, dtype=torch.float32, device=device)

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        zb = z_t[start:end].to(device)
        Bb = zb.shape[0]
        zeros_sc = torch.zeros_like(zb)
        z_in = torch.cat([zb, zeros_sc], dim=-1)
        t_batch = torch.full((Bb,), t_val, dtype=torch.float32, device=device)
        sc_s = sc_scale[:Bb]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(z_in, t_batch, decoder_step_active=None,
                  self_cond_cfg_scale=sc_s, attention_mask=None)

        h = captured["h11"]  # (Bb, total_tokens, 768)
        # Figure out prefix
        total_tok = h.shape[1]
        prefix = total_tok - L
        h_content = h[:, prefix:prefix + L, :]  # (Bb, L, 768)
        all_h11.append(h_content)

    handle.remove()
    return torch.cat(all_h11, dim=0)  # (B, L, 768)


def decode_path(h11, decode_weights, device):
    """Apply proj_kernel + GELU + unembed to get token logits."""
    h = h11.float().to(device)
    proj_k = decode_weights["proj_kernel"].to(device)
    proj_b = decode_weights["proj_bias"].to(device)
    unemb_k = decode_weights["unembed_kernel"].to(device)
    unemb_b = decode_weights["unembed_bias"].to(device)

    hidden = F.gelu(h @ proj_k + proj_b, approximate="tanh")  # (B, L, 512)
    logits = hidden @ unemb_k + unemb_b                        # (B, L, V)
    return logits.argmax(dim=-1).cpu()  # (B, L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", default="results/exp01/trajectories")
    parser.add_argument("--checkpoint", default="converted/elf_b-owt-kd-cr_torch.pt")
    parser.add_argument("--output_dir", default="results/exp01v2")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-01v2] Device: {device}")

    print(f"[EXP-01v2] Loading model from {args.checkpoint} ...")
    model, decode_weights, config = load_model_and_weights(args.checkpoint, device)

    files = sorted(f for f in os.listdir(args.traj_dir) if f.endswith(".pt"))
    print(f"[EXP-01v2] Found {len(files)} trajectory files")

    # Collect per-step stats
    step_stats = {}  # t_val -> {"correct": int, "total": int}

    for fname in files:
        print(f"\n[EXP-01v2] Processing {fname} ...")
        traj = torch.load(os.path.join(args.traj_dir, fname), map_location="cpu", weights_only=False)

        # Step 1: get final-step proxy GT via decode path on L11
        last_step = traj[-1]
        t_last = last_step["t_next"]  # t=1.0 (or very close)
        z_last = last_step["z_t"]    # the z_t AFTER last step update — use x_pred instead
        # Actually, use x_pred at final step directly? No, we need to rerun with hook.
        # The "final" output would be the z at t=t_next=1.0. Let's use x_pred at last step
        # as a proxy for z_T, and run backbone on it with t=1.0.
        z_final_approx = last_step["x_pred"]  # (B, L, 512) ≈ x_clean at final step

        # Re-run backbone on final z to get L11 and decode-path prediction
        print(f"  Computing proxy GT (final step, t≈{t_last:.3f}) ...")
        h11_final = get_l11_hidden(model, z_final_approx, t_last, device, args.batch_size)
        y_proxy = decode_path(h11_final, decode_weights, device)  # (B, L)
        del h11_final

        B, L = y_proxy.shape
        print(f"  B={B}, L={L}")

        # Step 2: for each ODE step, rerun backbone on z_t and compute G_B(t)
        for step_idx, step in enumerate(traj):
            t_val = step["t"]
            z_t = step["z_t"]  # (B, L, d) — actual ODE z at this step

            h11_t = get_l11_hidden(model, z_t, t_val, device, args.batch_size)
            pred_t = decode_path(h11_t, decode_weights, device)  # (B, L)
            del h11_t

            correct = int((pred_t == y_proxy).sum().item())
            total = B * L

            t_key = f"{t_val:.4f}"
            if t_key not in step_stats:
                step_stats[t_key] = {"t": t_val, "correct": 0, "total": 0}
            step_stats[t_key]["correct"] += correct
            step_stats[t_key]["total"] += total

            if step_idx % 8 == 0:
                G = correct / total
                print(f"  step {step_idx+1}/{len(traj)} (t={t_val:.3f}): G_B = {G:.4f}")

    # Aggregate
    results = []
    for t_key in sorted(step_stats.keys(), key=lambda k: float(k)):
        s = step_stats[t_key]
        G = s["correct"] / s["total"]
        results.append({"t": s["t"], "G_B": G, "n": s["total"]})
        print(f"  t={s['t']:.4f}  G_B={G:.4f}  n={s['total']}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "proto_B_decode_G.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXP-01v2] Saved to {out_path}")

    # Print summary table comparing with Protocol A (EXP-16 kd-cr)
    print("\n=== Protocol A (EXP-16, kd-cr) vs Protocol B (EXP-01v2) ===")
    print(f"{'t':>6}  {'G_B (decode)':>14}  {'G_A (EXP-16)':>14}")
    proto_a = {0.10: 0.1255, 0.20: 0.5849, 0.30: 0.8926, 0.50: 0.9949, 0.70: 0.9984}
    for r in results:
        t = r["t"]
        G_A = proto_a.get(round(t, 2), float("nan"))
        print(f"  {t:.4f}  {r['G_B']:>14.4f}  {G_A:>14.4f}")


if __name__ == "__main__":
    main()
