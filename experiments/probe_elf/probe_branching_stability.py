"""
EXP-11: Branching stability — given z_{t*} + small perturbation, does the ODE
converge to the same final token?

For a deterministic ODE (ELF default), from the same z_{t*} the result is identical.
We add K independent small perturbations δ ~ N(0, σ²I) with σ = noise_frac * ||z_{t*}||_rms
and run the remainder of the ODE to completion.

Measures per-position "stability" = fraction of K runs that agree with the unperturbed run.

Usage:
  CUDA_VISIBLE_DEVICES=5 python experiments/probe_elf/probe_branching_stability.py \
    --traj_dir results/exp01/trajectories \
    --checkpoint converted/elf_b-owt-kd-cr_torch.pt \
    --output_dir results/exp11 \
    --t_splits "0.10,0.20,0.30,0.50,0.70" \
    --K 8 --noise_frac 0.01 --n_seqs 64
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_model(checkpoint_path, device):
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
def run_ode_from_z(model, z_start, t_start, all_t_steps, decode_weights, device, batch_size=8):
    """
    Continue ODE from z_start at t_start through all_t_steps (list of t values ascending).
    Uses zero self-conditioning throughout.
    Returns final token predictions (B, L) using decode path on L11 hidden.
    """
    proj_k = decode_weights["proj_kernel"].to(device)
    proj_b = decode_weights["proj_bias"].to(device)
    unemb_k = decode_weights["unembed_kernel"].to(device)
    unemb_b = decode_weights["unembed_bias"].to(device)

    # Get the steps that come after t_start
    remaining = [(i, t, t_next) for i, (t, t_next) in enumerate(zip(all_t_steps[:-1], all_t_steps[1:]))
                 if t >= t_start - 1e-5]

    B, L, d = z_start.shape
    z = z_start.clone()
    captured = {}

    def hook_fn(mod, inp, out):
        captured["h11"] = out.float().cpu()

    handle = model.blocks[11].register_forward_hook(hook_fn)
    sc_ones = torch.ones(batch_size, dtype=torch.float32, device=device)

    all_final_tokens = []
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        Bb = end - start
        z_b = z[start:end].to(device)

        for step_idx, (_, t_val, t_next) in enumerate(remaining):
            zeros_sc = torch.zeros_like(z_b)
            z_in = torch.cat([z_b, zeros_sc], dim=-1)
            t_batch = torch.full((Bb,), t_val, dtype=torch.float32, device=device)
            sc_s = sc_ones[:Bb]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                x_hat, _, _ = model(z_in, t_batch, decoder_step_active=None,
                                    self_cond_cfg_scale=sc_s, attention_mask=None)

            # Euler step: z_{t+dt} = z_t + (t_next - t) * (z_t - x_hat) / (1 - t)
            x_hat = x_hat.float()
            if abs(1.0 - t_val) > 1e-6:
                v = (z_b - x_hat) / (1.0 - t_val)
                z_b = z_b + (t_next - t_val) * v
            # At t=1.0 we stop

        # Get L11 final prediction
        h11 = captured["h11"][:Bb]  # (Bb, total, 768)
        full_L = h11.shape[1]
        prefix = full_L - L
        h11_content = h11[:, prefix:prefix + L, :].to(device)

        hidden = F.gelu(h11_content @ proj_k + proj_b, approximate="tanh")
        logits = hidden @ unemb_k + unemb_b
        final_tok = logits.argmax(dim=-1).cpu()  # (Bb, L)
        all_final_tokens.append(final_tok)

    handle.remove()
    return torch.cat(all_final_tokens, dim=0)  # (B, L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", default="results/exp01/trajectories")
    parser.add_argument("--checkpoint", default="converted/elf_b-owt-kd-cr_torch.pt")
    parser.add_argument("--output_dir", default="results/exp11")
    parser.add_argument("--t_splits", default="0.10,0.20,0.30,0.50,0.70,0.80,0.90,0.95")
    parser.add_argument("--K", type=int, default=8, help="Number of perturbation runs")
    parser.add_argument("--eta_values", type=str, default="1e-4,3e-4,1e-3,3e-3,1e-2",
                        help="Comma-separated η values (relative L2 norm perturbation per position)")
    parser.add_argument("--noise_frac", type=float, default=None,
                        help="[deprecated] Use --eta_values instead")
    parser.add_argument("--n_seqs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    eta_list = [float(x) for x in args.eta_values.split(",")]
    if args.noise_frac is not None:
        eta_list = [args.noise_frac]

    t_splits = [float(t) for t in args.t_splits.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-11] Device: {device}, K={args.K}, eta_values={eta_list}")

    model, decode_weights, config = load_model(args.checkpoint, device)

    traj_files = sorted(f for f in os.listdir(args.traj_dir) if f.endswith(".pt"))[:args.n_seqs // 64 + 1]
    print(f"[EXP-11] Processing {len(traj_files)} trajectory files")

    results = []

    for fname in traj_files:
        print(f"\n[EXP-11] File: {fname}")
        traj = torch.load(os.path.join(args.traj_dir, fname), map_location="cpu", weights_only=False)

        # Build time grid from trajectory
        all_t = [s["t"] for s in traj] + [traj[-1]["t_next"]]  # includes t=0 and t=1
        all_t_steps = sorted(set(round(t, 5) for t in all_t))

        # Get unperturbed final prediction (from last trajectory step)
        # Use decode path on L11 at the final ODE state
        last_z = traj[-1]["z_t"]  # (B, L, 512) — ODE state at t≈0.97
        t_final = traj[-1]["t"]
        print(f"  Getting unperturbed final predictions (t={t_final:.3f})...")
        y_unperturbed = run_ode_from_z(model, last_z, t_final, all_t_steps,
                                       decode_weights, device, args.batch_size)
        B, L = y_unperturbed.shape

        for t_split in t_splits:
            # Find the trajectory step closest to t_split
            step_idx = min(range(len(traj)), key=lambda i: abs(traj[i]["t"] - t_split))
            t_actual = traj[step_idx]["t"]
            z_split = traj[step_idx]["z_t"]  # (B, L, 512)

            for eta in eta_list:
                print(f"  t_split={t_split:.2f} (actual={t_actual:.4f}), η={eta:.0e}: running {args.K} perturbed ODEs...")

                # Correct per-position unit-sphere perturbation:
                # δ = η * |z|_2 * (u / |u|_2), so |δ|_2 = η * |z|_2 exactly
                perturbed_preds = []
                for k in range(args.K):
                    torch.manual_seed(k * 1000 + step_idx)
                    u = torch.randn_like(z_split)
                    u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
                    delta = eta * z_split.norm(dim=-1, keepdim=True) * u
                    z_perturbed = z_split + delta

                    y_k = run_ode_from_z(model, z_perturbed, t_actual, all_t_steps,
                                         decode_weights, device, args.batch_size)
                    perturbed_preds.append(y_k)  # (B, L)

                preds_stack = torch.stack(perturbed_preds, dim=0)  # (K, B, L)

                # S_orig: fraction of branches agreeing with unperturbed
                agree_with_orig = (preds_stack == y_unperturbed.unsqueeze(0)).float()  # (K, B, L)
                s_orig = agree_with_orig.mean(dim=0)  # (B, L)

                # S_pair: pairwise agreement between branches
                pair_agree = 0.0
                n_pairs = 0
                for i in range(args.K):
                    for j in range(i + 1, args.K):
                        pair_agree += (preds_stack[i] == preds_stack[j]).float().mean().item()
                        n_pairs += 1
                s_pair = pair_agree / max(n_pairs, 1)

                # Modal probability: fraction of positions where all-same
                all_same = (preds_stack == preds_stack[0].unsqueeze(0)).all(dim=0)  # (B, L)
                modal_prob_all_same = all_same.float().mean().item()

                mean_s_orig = s_orig.mean().item()
                p10_s_orig = s_orig.quantile(0.10).item()
                print(f"    S_orig={mean_s_orig:.4f} (p10={p10_s_orig:.4f}), S_pair={s_pair:.4f}, all_same={modal_prob_all_same:.4f}")

                results.append({
                    "file": fname,
                    "t_split": t_actual,
                    "K": args.K,
                    "eta": eta,
                    "mean_s_orig": mean_s_orig,
                    "p10_s_orig": p10_s_orig,
                    "s_pair": s_pair,
                    "modal_prob_all_same": modal_prob_all_same,
                    "n_positions": B * L,
                })

    # Aggregate across files
    print("\n=== EXP-11v2 Aggregated Results (corrected perturbation) ===")
    from collections import defaultdict
    by_t_eta = defaultdict(list)
    for r in results:
        by_t_eta[(round(r["t_split"], 3), r["eta"])].append(r)
    print(f"{'t_split':>8}  {'η':>8}  {'S_orig':>8}  {'S_pair':>8}  {'all_same':>9}")
    for (t_key, eta) in sorted(by_t_eta.keys()):
        recs = by_t_eta[(t_key, eta)]
        n_tot = sum(r["n_positions"] for r in recs)
        ms = sum(r["mean_s_orig"] * r["n_positions"] for r in recs) / n_tot
        sp = sum(r["s_pair"] for r in recs) / len(recs)
        ma = sum(r["modal_prob_all_same"] for r in recs) / len(recs)
        print(f"  {t_key:.4f}  {eta:.0e}  {ms:.4f}  {sp:.4f}  {ma:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "branching_stability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXP-11] Saved to {out_path}")


if __name__ == "__main__":
    main()
