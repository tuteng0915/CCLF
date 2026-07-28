"""
EXP-01v3: Reverse ODE Trajectory vs Oracle Forward-Noise Probe

Paired comparison of:
  Protocol B (reverse ODE): G(z_t^reverse) — backbone probe on actual generation states
  Protocol A (oracle):      G(z_t^oracle)  — backbone probe on t*x_final + (1-t)*eps

The key: both share the SAME y_final (generated text), so x_final = T5(y_final) is the
"ground truth" for the oracle path. This directly measures the oracle-trajectory gap:
"How much earlier does ELF show commitment on oracle forward-noise states vs actual trajectory?"

Also reports:
  dist(t) = ||z_t^reverse - z_t^oracle||_2  (geometric distance between paths)
  x_pred quality: G(x_pred^reverse[t]) = backbone oracle accuracy of the actual x_pred

Usage:
    cd /home/wjzhang/tt_workspace/model/CCLF/CCLF/models/ELF-torch
    CUDA_VISIBLE_DEVICES=4 python ../../experiments/probe_elf/probe_reverse_trajectory.py \\
        --checkpoint converted/elf_b-owt-baseline_torch.pt \\
        --config src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir ../../experiments/probe_elf/results/exp01v3_baseline \\
        --n_samples 64 --n_steps 32 --label baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── ELF-torch source path ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_ELF_TORCH_DIR = _SCRIPT_DIR.parents[1] / "models" / "ELF-torch"
_ELF_SRC = _ELF_TORCH_DIR / "src"
ELF_SRC = str(_ELF_SRC)
if ELF_SRC not in sys.path:
    sys.path.insert(0, ELF_SRC)

from configs.config import load_config_from_yaml, SamplingConfig
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.sampling_utils import _ode_step


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", default="results/exp01v3")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=32, help="ODE steps for generation")
    p.add_argument("--seq_len", type=int, default=None,
                   help="Override seq_len (default: use config.max_length)")
    p.add_argument("--n_oracle_noise", type=int, default=4, help="Noise seeds for oracle path")
    p.add_argument("--label", default="baseline")
    return p.parse_args()


def load_model(config_path, checkpoint_path, device):
    orig_dir = os.getcwd()
    elf_dir = str(_ELF_TORCH_DIR)
    os.chdir(elf_dir)
    abs_cfg = config_path if os.path.isabs(config_path) else os.path.join(orig_dir, config_path)
    config = load_config_from_yaml(abs_cfg)
    os.chdir(orig_dir)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)
    enc_config, t5_model = get_encoder(config.encoder_model_name, dtype=torch.float32)
    t5_model = t5_model.to(device).eval()

    model_cls = ELF_models[config.model]
    model = model_cls(
        text_encoder_dim=enc_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=tokenizer.vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    ).to(device).eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "params" in ckpt:
        params = ckpt.get("ema_params1", ckpt["params"])
        missing, _ = model.load_state_dict(params, strict=False)
        if missing:
            print(f"[model] missing keys: {len(missing)}")
        print(f"[model] loaded (EMA: {'ema_params1' in ckpt})")

    return model, tokenizer, t5_model, config, enc_config


@torch.no_grad()
def run_ode_trajectory(model, config, n_steps, seq_len, device, batch_size=4):
    """Run ODE generation with trajectory recording.
    Returns: list of dicts with keys t, z_t (B,L,512), x_pred (B,L,512)
    Also returns z_final (B,L,512) at t=1.
    """
    t_eps = getattr(config, "t_eps", 0.05)
    t_steps = torch.linspace(t_eps, 1.0, n_steps + 1, device=device)

    B = batch_size
    L = seq_len
    d = 512  # T5-small hidden dim

    noise_scale = float(getattr(config, "denoiser_noise_scale", 2.0))
    z = torch.randn(B, L, d, device=device) * noise_scale

    x_pred = torch.zeros_like(z)
    trajectory = []

    use_bf16 = getattr(config, "use_bf16", True) and z.is_cuda

    # Dummy cond_seq (unconditional)
    cond_seq = torch.zeros_like(z)
    cond_seq_mask = torch.zeros(B, L, device=device)

    step_kwargs = dict(
        model=model,
        config=config,
        cfg_scale=1.0,
        self_cond_cfg_scale=1.0,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
    )

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(len(t_steps) - 1):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            z_before = z.detach().clone()
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
            trajectory.append({
                "t": t,
                "t_next": t_next,
                "z_t": z_before.cpu(),
                "x_pred": x_pred.detach().cpu().clone(),
            })

    z_final = z.detach().cpu()  # (B, L, 512) at t=1
    return trajectory, z_final


@torch.no_grad()
def decode_to_tokens(model, z_final, config, device, batch_size=4):
    """Run decode head on z_final to get token IDs."""
    N = z_final.shape[0]
    all_token_ids = []
    for i in range(0, N, batch_size):
        z_b = z_final[i:i+batch_size].to(device)
        B_b = z_b.shape[0]
        L = z_b.shape[1]
        sc = torch.zeros_like(z_b)
        z_in = torch.cat([z_b, sc], dim=-1) if config.self_cond_prob > 0 else z_b
        t_b = torch.ones(B_b, device=device)
        sc_scale = torch.zeros(B_b, device=device) if config.num_self_cond_cfg_tokens > 0 else None
        _, logits, _ = model(z_in, t_b, self_cond_cfg_scale=sc_scale, decoder_step_active=True)
        token_ids = logits.argmax(dim=-1)  # (B, L)
        all_token_ids.append(token_ids.cpu())
    return torch.cat(all_token_ids, dim=0)  # (N, L)


@torch.no_grad()
def encode_tokens(t5_model, tokenizer, token_ids, latent_mean, latent_std, device, batch_size=8):
    """Encode token IDs with T5 to get clean embeddings."""
    N = token_ids.shape[0]
    x_clean_list = []
    for i in range(0, N, batch_size):
        batch = token_ids[i:i+batch_size].to(device)
        mask = (batch != (tokenizer.eos_token_id or 1)).long()
        out = t5_model.model(input_ids=batch, attention_mask=mask)
        emb = (out.last_hidden_state - latent_mean) / latent_std
        x_clean_list.append(emb.cpu())
    return torch.cat(x_clean_list, dim=0)  # (N, L, 512)


@torch.no_grad()
def probe_accuracy(model, z_input, t_val, config, device, gt_ids, batch_size=4):
    """Compute decoder head accuracy: % positions where argmax == gt_ids."""
    N = z_input.shape[0]
    all_logits = []
    for i in range(0, N, batch_size):
        z_b = z_input[i:i+batch_size].to(device)
        B_b = z_b.shape[0]
        sc = torch.zeros_like(z_b)
        z_in = torch.cat([z_b, sc], dim=-1) if config.self_cond_prob > 0 else z_b
        t_b = torch.full((B_b,), t_val, device=device, dtype=torch.float32)
        sc_scale = torch.zeros(B_b, device=device) if config.num_self_cond_cfg_tokens > 0 else None
        _, logits, _ = model(z_in, t_b, self_cond_cfg_scale=sc_scale, decoder_step_active=True)
        all_logits.append(logits.cpu())
    logits = torch.cat(all_logits, dim=0)
    preds = logits.argmax(dim=-1)
    return (preds == gt_ids).float().mean().item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp01v3] Loading model from {args.checkpoint}")
    model, tokenizer, t5_model, config, enc_config = load_model(
        args.config, args.checkpoint, device)

    latent_mean = getattr(config, "latent_mean", 0.0)
    latent_std = getattr(config, "latent_std", 0.2)

    # Use model's max_length unless overridden
    seq_len = args.seq_len if args.seq_len is not None else config.max_length
    print(f"[exp01v3] seq_len={seq_len} (config.max_length={config.max_length})")

    # Run generation in batches to collect trajectories
    print(f"[exp01v3] Generating {args.n_samples} samples with {args.n_steps} ODE steps...")
    all_trajectories = []
    all_z_final = []
    batch_size = 4

    for batch_start in range(0, args.n_samples, batch_size):
        B = min(batch_size, args.n_samples - batch_start)
        print(f"  Batch {batch_start//batch_size + 1}/{(args.n_samples+batch_size-1)//batch_size} (B={B})")
        traj, z_final = run_ode_trajectory(model, config, args.n_steps, seq_len, device, B)
        all_trajectories.append(traj)   # list of steps, each with shape (B, L, 512)
        all_z_final.append(z_final)

    # Merge trajectories: each element is list of step dicts, each with (B,L,d) tensors
    z_final_all = torch.cat(all_z_final, dim=0)  # (N, L, 512)

    # Decode final z to tokens
    print("[exp01v3] Decoding final states to tokens...")
    y_final_ids = decode_to_tokens(model, z_final_all, config, device)  # (N, seq_len)
    print(f"[exp01v3] y_final_ids shape: {y_final_ids.shape}")

    # Re-encode generated tokens with T5 → x_final
    print("[exp01v3] Re-encoding generated tokens with T5...")
    x_final = encode_tokens(t5_model, tokenizer, y_final_ids, latent_mean, latent_std, device)
    print(f"[exp01v3] x_final shape: {x_final.shape}")  # (N, L, 512)

    # Merge all trajectory steps across batches
    n_traj_steps = len(all_trajectories[0])
    # For each step index, cat across batches
    merged_traj = []
    for step_i in range(n_traj_steps):
        merged_traj.append({
            "t": all_trajectories[0][step_i]["t"],
            "t_next": all_trajectories[0][step_i]["t_next"],
            "z_t": torch.cat([b[step_i]["z_t"] for b in all_trajectories], dim=0),  # (N,L,d)
            "x_pred": torch.cat([b[step_i]["x_pred"] for b in all_trajectories], dim=0),
        })

    print(f"[exp01v3] Trajectory has {len(merged_traj)} steps")

    # Compute metrics at each trajectory step
    results = {
        "t": [],
        "G_reverse": [],     # probe accuracy on actual reverse ODE states
        "G_oracle_mean": [], # probe accuracy on oracle forward-noise states (mean over noise seeds)
        "G_oracle_std": [],
        "G_xpred": [],       # probe accuracy of x_pred itself (the model's denoised output)
        "dist_z_mean": [],   # ||z_reverse - z_oracle||_2 averaged over noise seeds
        "dist_z_std": [],
    }

    print(f"\n{'t':>5} | {'G_reverse':>9} | {'G_oracle':>9} | {'G_xpred':>9} | {'dist_z':>7}")
    print("-" * 55)

    for step in merged_traj:
        t = step["t"]
        z_reverse = step["z_t"]  # (N, L, 512) — actual reverse ODE state
        x_pred = step["x_pred"]  # (N, L, 512) — model's prediction at this step

        # G_reverse: probe accuracy on actual reverse state
        g_rev = probe_accuracy(model, z_reverse, t, config, device, y_final_ids)

        # G_oracle: probe accuracy on oracle forward-noise states (multiple seeds)
        g_oracle_list = []
        dist_list = []
        for _ in range(args.n_oracle_noise):
            eps = torch.randn_like(x_final)
            z_oracle = t * x_final + (1.0 - t) * eps
            g_oracle_list.append(probe_accuracy(model, z_oracle, t, config, device, y_final_ids))
            # Distance between reverse and oracle states
            dist = (z_reverse - z_oracle).norm(dim=-1).mean().item()
            dist_list.append(dist)

        # G_xpred: probe x_pred directly (this is the model's actual denoised prediction)
        g_xpred = probe_accuracy(model, x_pred, 1.0, config, device, y_final_ids)

        results["t"].append(t)
        results["G_reverse"].append(g_rev)
        results["G_oracle_mean"].append(float(np.mean(g_oracle_list)))
        results["G_oracle_std"].append(float(np.std(g_oracle_list)))
        results["G_xpred"].append(g_xpred)
        results["dist_z_mean"].append(float(np.mean(dist_list)))
        results["dist_z_std"].append(float(np.std(dist_list)))

        print(f"{t:5.3f} | {g_rev:9.4f} | {results['G_oracle_mean'][-1]:9.4f} | "
              f"{g_xpred:9.4f} | {results['dist_z_mean'][-1]:7.3f}")

    # Save
    out_path = out_dir / f"probe_reverse_traj_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[exp01v3] Saved to {out_path}")

    # Key summary
    t_grid = np.array(results["t"])
    t03_idx = int(np.argmin(np.abs(t_grid - 0.30)))
    t05_idx = int(np.argmin(np.abs(t_grid - 0.50)))
    t07_idx = int(np.argmin(np.abs(t_grid - 0.70)))

    print("\n── Oracle-Trajectory Gap at Key t Values ──")
    print(f"{'t':>5} | {'G_reverse':>9} | {'G_oracle':>9} | {'gap (oracle-rev)':>16}")
    for idx, label in [(t03_idx, "~0.30"), (t05_idx, "~0.50"), (t07_idx, "~0.70")]:
        g_r = results["G_reverse"][idx]
        g_o = results["G_oracle_mean"][idx]
        print(f"{results['t'][idx]:5.3f} | {g_r:9.4f} | {g_o:9.4f} | {g_o - g_r:16.4f}")

    print("\n[exp01v3] Done.")


if __name__ == "__main__":
    main()
