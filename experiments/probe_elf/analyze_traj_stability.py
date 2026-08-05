"""
EXP-14v2: Commit-release-recommit analysis on actual ODE reverse trajectories.

Uses correct decode path (GELU(h_L11 @ proj_kernel + proj_bias) @ unembed_kernel)
by re-running backbone on each trajectory z_t with block-11 hook, matching EXP-01v2.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_model_and_decode_weights(checkpoint_path, device):
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
def get_decode_path_preds(model, z_t, t_val, decode_weights, device, batch_size=8):
    """
    Re-run backbone on z_t with block-11 hook to get correct decode-path predictions.
    Returns (B, L) token argmax.
    """
    captured = {}

    def hook(mod, inp, out):
        captured["h11"] = out.float().cpu()

    handle = model.blocks[11].register_forward_hook(hook)
    B, L, d = z_t.shape
    all_preds = []
    sc_scale = torch.ones(batch_size, dtype=torch.float32, device=device)

    proj_k = decode_weights["proj_kernel"].to(device)
    proj_b = decode_weights["proj_bias"].to(device)
    unemb_k = decode_weights["unembed_kernel"].to(device)
    unemb_b = decode_weights["unembed_bias"].to(device)

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        zb = z_t[start:end].to(device)
        Bb = zb.shape[0]
        zeros_sc = torch.zeros_like(zb)
        z_in = torch.cat([zb, zeros_sc], dim=-1)
        t_batch = torch.full((Bb,), t_val, dtype=torch.float32, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(z_in, t_batch, decoder_step_active=None,
                  self_cond_cfg_scale=sc_scale[:Bb], attention_mask=None)

        h = captured["h11"]  # (Bb, total_tokens, 768)
        prefix = h.shape[1] - L
        h_content = h[:, prefix:prefix + L, :].to(device)  # (Bb, L, 768)
        hidden = F.gelu(h_content @ proj_k + proj_b, approximate="tanh")
        logits = hidden @ unemb_k + unemb_b
        all_preds.append(logits.argmax(dim=-1).cpu())

    handle.remove()
    return torch.cat(all_preds, dim=0)  # (B, L)


def analyze_trajectory_file(traj_path, model, decode_weights, device, batch_size=8):
    data = torch.load(traj_path, map_location="cpu")
    n_steps = len(data)
    B, L, d = data[0]['z_t'].shape
    t_vals = [step['t'] for step in data]
    step_argmax = torch.zeros(n_steps, B, L, dtype=torch.long)

    # Proxy GT: decode-path prediction at the final step (t≈1.0 after ODE ends)
    # Use x_pred from last step as z approximation at t_next≈1.0, re-run backbone
    last_step = data[-1]
    t_last = last_step.get("t_next", last_step["t"])
    z_final = last_step["x_pred"]  # (B, L, 512) — final x̂ as proxy for z_{T}
    print(f"  Computing proxy GT from final step (t≈{t_last:.3f}) via decode path ...")
    proxy_gt = get_decode_path_preds(model, z_final, t_last, decode_weights, device, batch_size)

    for step_i, step in enumerate(data):
        z_t = step['z_t']  # (B, L, 512)
        t_val = step['t']
        preds = get_decode_path_preds(model, z_t, t_val, decode_weights, device, batch_size)
        step_argmax[step_i] = preds
        if step_i % 8 == 0:
            frac_match = (preds == proxy_gt).float().mean().item()
            print(f"  Step {step_i+1}/{n_steps} (t={t_val:.3f}): match_proxy_GT={frac_match:.4f}")

    return step_argmax, t_vals, proxy_gt


def compute_stability_metrics(step_argmax, proxy_gt, t_vals, stable_m=3):
    """
    step_argmax : (T, B, L) long — decode-path token prediction at each ODE step
    proxy_gt    : (B, L) long — final decode-path prediction (proxy GT)
    t_vals      : list of T floats
    stable_m    : min consecutive steps with same prediction to count as committed
    """
    n_steps, B, L = step_argmax.shape
    changes = (step_argmax[1:] != step_argmax[:-1]).long()  # (T-1, B, L)
    n_flips = changes.sum(dim=0).float()   # (B, L)
    n_flat  = n_flips.reshape(-1)

    # Match against proxy GT (decode path at final step)
    matches_proxy = (step_argmax == proxy_gt.unsqueeze(0))  # (T, B, L)
    frac_match_proxy_per_step = matches_proxy.reshape(n_steps, -1).float().mean(dim=-1)

    # Flip distribution
    flip_counts = {
        "zero_flips":   (n_flat == 0).float().mean().item(),
        "one_flip":     (n_flat == 1).float().mean().item(),
        "two_to_four":  ((n_flat >= 2) & (n_flat <= 4)).float().mean().item(),
        "five_plus":    (n_flat >= 5).float().mean().item(),
        "mean_flips":   n_flat.mean().item(),
        "median_flips": float(n_flat.median().item()),
    }

    # Flip rate per unit t (normalize by total t-span)
    t_span = t_vals[-1] - t_vals[0] if len(t_vals) > 1 else 1.0
    flip_rate_per_t = n_flat.mean().item() / (t_span + 1e-8)

    # Last flip step for positions that flipped
    last_flip = torch.full((B, L), -1, dtype=torch.long)
    for step_i in range(n_steps - 1):
        mask = changes[step_i].bool()
        last_flip[mask] = step_i + 1
    flipped_mask = n_flat.reshape(B, L) > 0
    if flipped_mask.any():
        lf = last_flip[flipped_mask].float()
        mean_last_flip = lf.mean().item()
        frac_early = (lf < n_steps / 2).float().mean().item()
    else:
        mean_last_flip = float('nan')
        frac_early = float('nan')

    # Stable commitment time: first step where token is same for m consecutive steps
    stable_commit_step = torch.full((B, L), n_steps, dtype=torch.long)
    for step_i in range(n_steps - stable_m + 1):
        consecutive_same = torch.ones(B, L, dtype=torch.bool)
        base_token = step_argmax[step_i]
        for k in range(1, stable_m):
            consecutive_same &= (step_argmax[step_i + k] == base_token)
        new_commit = consecutive_same & (stable_commit_step == n_steps)
        stable_commit_step[new_commit] = step_i

    frac_stably_committed = (stable_commit_step < n_steps).float().mean().item()
    committed_mask = stable_commit_step < n_steps
    if committed_mask.any():
        mean_stable_commit = stable_commit_step[committed_mask].float().mean().item()
    else:
        mean_stable_commit = float('nan')

    return {
        "n_steps": n_steps,
        "n_positions": int(B * L),
        "flip_distribution": flip_counts,
        "flip_rate_per_unit_t": flip_rate_per_t,
        "frac_match_proxy_gt_per_step": frac_match_proxy_per_step.tolist(),
        "mean_last_flip_step": mean_last_flip,
        "frac_last_flip_early_half": frac_early,
        "stable_m": stable_m,
        "frac_stably_committed": frac_stably_committed,
        "mean_stable_commit_step": mean_stable_commit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--traj_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stable_m", type=int, default=3,
                        help="Min consecutive steps with same prediction for stable commit")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-14v2] Device={device}, stable_m={args.stable_m}")
    print(f"[EXP-14v2] Loading model + decode weights from {args.checkpoint} ...")
    model, decode_weights, config = load_model_and_decode_weights(args.checkpoint, device)

    traj_files = sorted([
        os.path.join(args.traj_dir, f)
        for f in os.listdir(args.traj_dir) if f.startswith("traj_") and f.endswith(".pt")
    ])
    print(f"[EXP-14v2] Found {len(traj_files)} trajectory files")

    all_argmax = []
    all_proxy_gt = []
    t_vals = None
    for tf in traj_files:
        print(f"\n[EXP-14v2] Processing {os.path.basename(tf)} ...")
        sa, tv, proxy_gt = analyze_trajectory_file(tf, model, decode_weights, device, args.batch_size)
        all_argmax.append(sa)
        all_proxy_gt.append(proxy_gt)
        t_vals = tv

    # Concatenate along batch dimension
    combined = torch.cat(all_argmax, dim=1)          # (T, B_total, L)
    combined_proxy = torch.cat(all_proxy_gt, dim=0)  # (B_total, L)
    print(f"[EXP-14v2] Combined: {combined.shape}")
    metrics = compute_stability_metrics(combined, combined_proxy, t_vals, stable_m=args.stable_m)
    metrics["t_values"] = t_vals

    fd = metrics["flip_distribution"]
    print("\n=== EXP-14v2 Trajectory Stability (correct decode path) ===")
    print(f"  n_positions={metrics['n_positions']:,}, n_steps={metrics['n_steps']}")
    print(f"  0 flips: {fd['zero_flips']*100:.1f}%  |  1 flip: {fd['one_flip']*100:.1f}%  |  2-4: {fd['two_to_four']*100:.1f}%  |  5+: {fd['five_plus']*100:.1f}%")
    print(f"  mean flips: {fd['mean_flips']:.2f}, median: {fd['median_flips']:.1f}")
    print(f"  flip rate per unit t: {metrics['flip_rate_per_unit_t']:.4f}")
    print(f"  mean last-flip step (of flipped): {metrics['mean_last_flip_step']:.1f}/{metrics['n_steps']}")
    print(f"  Stably committed (m={args.stable_m}): {metrics['frac_stably_committed']:.4f}, mean_step={metrics['mean_stable_commit_step']:.1f}")
    print(f"  Frac matching proxy GT at each step (decode path):")
    for i, (t, frac) in enumerate(zip(t_vals, metrics['frac_match_proxy_gt_per_step'])):
        if i % 4 == 0 or i == len(t_vals)-1:
            print(f"    step {i+1:2d} (t={t:.3f}): {frac:.4f}")

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[EXP-14v2] Saved to {args.output}")


if __name__ == "__main__":
    main()
