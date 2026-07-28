"""
EXP-04v2: Head-Only Null Model Probe

Tests whether the decode head (proj_kernel → GELU → unembed_kernel) alone,
applied to pure Gaussian noise bypassing the backbone, can predict true tokens.

Compares:
  G_head_null(t): randn(B,L,768) → proj_kernel → GELU → unembed_kernel → argmax vs y_true
  G_oracle(t):    z_t → backbone → decoder_logits → argmax vs y_true
  G_backbone_null(t): randn(B,L,512) → backbone(t) → decoder_logits → argmax vs y_true

Usage:
    cd /home/wjzhang/tt_workspace/model/CCLF/CCLF/models/ELF-torch
    CUDA_VISIBLE_DEVICES=3 python ../../experiments/probe_elf/probe_head_null.py \\
        --checkpoint converted/elf_b-owt-baseline_torch.pt \\
        --config src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir ../../experiments/probe_elf/results/exp04v2_baseline \\
        --n_samples 128 --label baseline
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

from configs.config import load_config_from_yaml
from modules.model import ELF_models
from modules.t5_encoder import get_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--config", required=True, help="Path to config YAML")
    p.add_argument("--out_dir", default="results/exp04v2", help="Output directory")
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--n_noise", type=int, default=4, help="Noise seeds per t")
    p.add_argument("--n_t_steps", type=int, default=20)
    p.add_argument("--seq_len", type=int, default=None,
                   help="Override seq_len (default: use config.max_length)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--label", default="baseline")
    return p.parse_args()


def load_model(config_path, checkpoint_path, device):
    """Load ELF-torch model."""
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
    # Converted checkpoints use "params" key; eval uses EMA params if available
    if isinstance(ckpt, dict) and "params" in ckpt:
        # Use EMA params for eval if available
        params = ckpt.get("ema_params1", ckpt["params"])
        missing, unexpected = model.load_state_dict(params, strict=False)
        if missing:
            print(f"[model] missing keys (using init): {len(missing)}")
        print(f"[model] loaded from ckpt['params'] (EMA: {'ema_params1' in ckpt})")
    else:
        print(f"[model] unexpected ckpt type: {type(ckpt)}")

    return model, tokenizer, t5_model, config, enc_config


@torch.no_grad()
def encode_batch(t5_model, input_ids, attn_mask, latent_mean, latent_std, device):
    out = t5_model.model(input_ids=input_ids.to(device),
                          attention_mask=attn_mask.to(device))
    emb = out.last_hidden_state
    return (emb - latent_mean) / latent_std


@torch.no_grad()
def elf_forward(model, z_clean, t_val, config, device, batch_size=8):
    """Run ELF backbone with SC=0. Returns decoder_logits (B, L, V)."""
    all_logits = []
    N = z_clean.shape[0]
    for i in range(0, N, batch_size):
        z_b = z_clean[i:i+batch_size].to(device)
        B_b = z_b.shape[0]
        # Concatenate SC (zeros) for self-conditioning
        if config.self_cond_prob > 0:
            z_in = torch.cat([z_b, torch.zeros_like(z_b)], dim=-1)
        else:
            z_in = z_b
        t_b = torch.full((B_b,), t_val, device=device, dtype=torch.float32)
        sc_scale = torch.zeros(B_b, device=device) if config.num_self_cond_cfg_tokens > 0 else None
        _, logits, _ = model(z_in, t_b,
                              self_cond_cfg_scale=sc_scale,
                              decoder_step_active=True)
        all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


def top1_acc(logits, gt_ids):
    """Compute fraction of positions where argmax matches gt_ids."""
    preds = logits.argmax(dim=-1)
    return (preds == gt_ids).float().mean().item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp04v2] Loading model from {args.checkpoint}")
    model, tokenizer, t5_model, config, enc_config = load_model(
        args.config, args.checkpoint, device)

    # Extract decode head params
    proj_kernel = model.proj_kernel.detach().to(device)    # (hidden_size=768, bn=512)
    proj_bias = model.proj_bias.detach().to(device)        # (512,)
    unembed_kernel = model.unembed_kernel.detach().to(device)  # (512, vocab_size)
    unembed_bias = model.unembed_bias.detach().to(device)      # (vocab_size,)
    hidden_size = proj_kernel.shape[0]
    print(f"[exp04v2] hidden_size={hidden_size}, proj: {proj_kernel.shape}, unembed: {unembed_kernel.shape}")

    latent_mean = getattr(config, "latent_mean", 0.0)
    latent_std = getattr(config, "latent_std", 0.2)

    # Use model's max_length unless overridden (must be before dataset loading)
    seq_len = args.seq_len if args.seq_len is not None else config.max_length
    print(f"[exp04v2] seq_len={seq_len} (config.max_length={config.max_length})")

    # Load validation data
    print(f"[exp04v2] Loading {args.n_samples} sequences from dataset")
    from datasets import load_dataset
    ds = load_dataset("embedded-language-flows/openwebtext-t5",
                      split="train", streaming=True)

    samples = []
    for ex in ds:
        ids = torch.tensor(ex["input_ids"][:seq_len], dtype=torch.long)
        if len(ids) < seq_len:
            ids = F.pad(ids, (0, seq_len - len(ids)),
                        value=tokenizer.eos_token_id or 1)
        mask = (ids != (tokenizer.eos_token_id or 1)).long()
        samples.append((ids, mask))
        if len(samples) >= args.n_samples:
            break
    print(f"[exp04v2] Loaded {len(samples)} sequences")

    # Encode all with T5
    all_ids = torch.stack([s[0] for s in samples])      # (N, L)
    all_masks = torch.stack([s[1] for s in samples])    # (N, L)

    x_clean_list = []
    for i in range(0, len(samples), args.batch_size):
        emb = encode_batch(t5_model, all_ids[i:i+args.batch_size],
                           all_masks[i:i+args.batch_size],
                           latent_mean, latent_std, device)
        x_clean_list.append(emb.cpu())
    x_clean = torch.cat(x_clean_list, dim=0)  # (N, L, 512)
    gt_ids = all_ids  # (N, L) — T5 token IDs
    print(f"[exp04v2] x_clean: {x_clean.shape}, gt_ids: {gt_ids.shape}")

    t_grid = np.linspace(0.05, 1.0, args.n_t_steps)
    torch.manual_seed(42)

    results = {
        "t": t_grid.tolist(),
        "G_head_null": [],       # Gaussian(768) → decode head directly
        "G_backbone_null": [],   # Gaussian(512) → full backbone → decode head
        "G_oracle": [],          # z_t → backbone → decode head
        "label": args.label,
        "n_samples": len(samples),
        "seq_len": seq_len,
    }

    print(f"\n{'t':>5} | {'G_head_null':>11} | {'G_back_null':>11} | {'G_oracle':>9}")
    print("-" * 48)

    for ti, t in enumerate(t_grid):
        g_head_list, g_back_list, g_oracle_list = [], [], []

        for si in range(args.n_noise):
            # ── Head-only null (skip backbone entirely) — process in mini-batches ──
            head_null_correct = 0
            head_null_total = 0
            for bi in range(0, len(samples), args.batch_size):
                h_b = torch.randn(min(args.batch_size, len(samples) - bi),
                                   seq_len, hidden_size, device=device)
                with torch.no_grad():
                    h_f32 = h_b.float()
                    hidden = F.gelu(h_f32 @ proj_kernel.float() + proj_bias.float(),
                                    approximate="tanh")
                    logits_b = hidden @ unembed_kernel.float() + unembed_bias.float()
                preds = logits_b.argmax(dim=-1).cpu()
                head_null_correct += (preds == gt_ids[bi:bi+args.batch_size]).float().sum().item()
                head_null_total += preds.numel()
                del h_b, h_f32, hidden, logits_b
                torch.cuda.empty_cache()
            g_head_list.append(head_null_correct / head_null_total)

            # ── Backbone null (Gaussian through full backbone) ───────────────
            z_null = torch.randn(len(samples), seq_len, 512)  # input space
            logits_bnull = elf_forward(model, z_null, t, config, device, args.batch_size)
            g_back_list.append(top1_acc(logits_bnull, gt_ids))

            # ── Oracle (z_t = t*x_clean + (1-t)*eps) ────────────────────────
            eps = torch.randn_like(x_clean)
            z_t = t * x_clean + (1.0 - t) * eps
            logits_oracle = elf_forward(model, z_t, t, config, device, args.batch_size)
            g_oracle_list.append(top1_acc(logits_oracle, gt_ids))

        results["G_head_null"].append(float(np.mean(g_head_list)))
        results["G_backbone_null"].append(float(np.mean(g_back_list)))
        results["G_oracle"].append(float(np.mean(g_oracle_list)))

        print(f"{t:5.2f} | {results['G_head_null'][-1]:11.4f} | "
              f"{results['G_backbone_null'][-1]:11.4f} | {results['G_oracle'][-1]:9.4f}")

    # Save
    out_path = out_dir / f"probe_head_null_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[exp04v2] Saved to {out_path}")

    # Key summary
    t50_idx = np.argmin(np.abs(t_grid - 0.30))
    print(f"\n── Key results at t=0.30 ──")
    print(f"  G_head_null:     {results['G_head_null'][t50_idx]:.4f}  (pure head geometry on random input)")
    print(f"  G_backbone_null: {results['G_backbone_null'][t50_idx]:.4f}  (backbone learned prior + head)")
    print(f"  G_oracle:        {results['G_oracle'][t50_idx]:.4f}  (signal from z_t)")
    print("\n[exp04v2] Done.")


if __name__ == "__main__":
    main()
