"""
EXP-07b: Layer-wise linear probe on ELF transformer blocks.

For each transformer block i (0..depth-1) and each t in t_grid:
  1. Run ELF backbone with hooks to capture h_i (768-dim hidden state after block i)
  2. Train linear probe: h_i -> vocab token
  3. Compare vs final x̂_t probe and native decoder

Output: results/exp07b_{name}/layerwise_probe_accuracies.json

Usage:
  CUDA_VISIBLE_DEVICES=2 python experiments/probe_elf/probe_layerwise.py \
    --checkpoint converted/elf_b-owt-baseline_torch.pt \
    --output_dir results/exp07b_baseline \
    --n_seqs 256 --t_values 0.10,0.20,0.30,0.50,0.70
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_model(checkpoint_path, device):
    from configs.config import load_config_from_yaml
    from modules.model import ELF_models
    from utils.checkpoint_utils import load_checkpoint
    from utils.train_utils import TrainState, get_optimizer

    cfg_path = os.path.join(os.path.dirname(__file__),
                            "../../src/configs/training_configs/eval_exp13.yml")
    config = load_config_from_yaml(cfg_path)
    from modules.t5_encoder import T5EncoderConfig
    encoder_config = T5EncoderConfig.from_pretrained(config.encoder_model_name)
    _ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _params = _ckpt.get("params", _ckpt)
    vocab_size = _params["unembed_kernel"].shape[1]
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
    return state.model, encoder_config, config


def load_encoder(encoder_model_name, device):
    from modules.t5_encoder import get_encoder
    _, encoder = get_encoder(encoder_model_name, dtype=torch.float32)
    return encoder.to(device).eval()


def load_data(n_seqs, max_length):
    from datasets import load_dataset as hf_load
    ds = hf_load("embedded-language-flows/openwebtext-t5", split="train", streaming=True)
    ids_list, mask_list = [], []
    for i, ex in enumerate(ds):
        if i >= n_seqs:
            break
        ids = ex["input_ids"][:max_length]
        L = len(ids)
        pad = max_length - L
        ids_list.append(ids + [0] * pad)
        mask_list.append([1] * L + [0] * pad)
    return np.array(ids_list, dtype=np.int64), np.array(mask_list, dtype=np.float32)


@torch.no_grad()
def collect_layer_states(model, encoder, config, input_ids_np, attn_mask_np,
                         t_values, batch_size, latent_std, device, output_dir,
                         fixed_noise_seed=42):
    os.makedirs(output_dir, exist_ok=True)
    n_seqs = input_ids_np.shape[0]
    depth = model.depth
    n_batches = (n_seqs + batch_size - 1) // batch_size

    # Encode sequences
    print(f"[EXP-07b] Encoding {n_seqs} sequences ...")
    x_clean_list, y_tokens_list = [], []
    for b in range(n_batches):
        sl = slice(b * batch_size, min((b + 1) * batch_size, n_seqs))
        ids = torch.tensor(input_ids_np[sl], dtype=torch.long, device=device)
        mask = torch.tensor(attn_mask_np[sl], dtype=torch.float32, device=device)
        latents = encoder(input_ids=ids, attention_mask=mask, deterministic=True)
        x_clean_list.append((latents / latent_std).cpu().float())
        y_tokens_list.append(ids.cpu())

    x_clean_all = torch.cat(x_clean_list, dim=0)   # (N, L, 512)
    y_tokens_all = torch.cat(y_tokens_list, dim=0)  # (N, L)
    attn_mask_all = torch.tensor(attn_mask_np, dtype=torch.float32)
    print(f"  x_clean shape: {x_clean_all.shape}")

    # Generate fixed noise once and reuse across all t values so that
    # commit_times comparisons across t use the same ε draw per position.
    g_noise = torch.Generator(device='cpu').manual_seed(fixed_noise_seed)
    eps_all = torch.randn(x_clean_all.shape, generator=g_noise)  # (N, L, d_latent)
    print(f"  Fixed noise generated (seed={fixed_noise_seed}, shape={tuple(eps_all.shape)})")

    for t_val in t_values:
        print(f"\n[EXP-07b] t={t_val:.3f} — collecting {depth} layer activations ...")
        # layer_feats[i] = list of (B, L, hidden_size) tensors
        layer_feats = [[] for _ in range(depth)]
        x_hat_list = []

        for b in range(n_batches):
            sl = slice(b * batch_size, min((b + 1) * batch_size, n_seqs))
            x_c = x_clean_all[sl].to(device)
            attn = attn_mask_all[sl].to(device)
            B, L, d = x_c.shape

            eps = eps_all[sl].to(device)  # same ε for all t values
            z_t = t_val * x_c + (1.0 - t_val) * eps
            zeros_sc = torch.zeros_like(z_t)
            z_in = torch.cat([z_t, zeros_sc], dim=-1)
            t_batch = torch.full((B,), t_val, dtype=torch.float32, device=device)
            sc_scale = torch.ones(B, dtype=torch.float32, device=device)

            # Register hooks to capture block outputs (token positions only)
            captured = {}
            hooks = []
            for i, block in enumerate(model.blocks):
                def make_hook(idx):
                    def hook(mod, inp, out):
                        # out shape: (B, n_tokens+L, hidden_size) — grab token positions
                        captured[idx] = out.float().cpu()
                    return hook
                hooks.append(block.register_forward_hook(make_hook(i)))

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                x_hat, _, _ = model(z_in, t_batch, decoder_step_active=None,
                                    self_cond_cfg_scale=sc_scale,
                                    attention_mask=attn)
            for h in hooks:
                h.remove()

            x_hat_list.append(x_hat.float().cpu())
            # captured[i] has shape (B, n_prefix + L, hidden_size)
            # We need to figure out how many prefix tokens there are.
            # n_prefix = num_time_tokens + num_self_cond_cfg_tokens + num_model_mode_tokens + bottleneck_tokens
            # Actually let's just figure it out from the shape
            full_L = captured[0].shape[1]
            prefix = full_L - L  # prefix tokens to skip
            for i in range(depth):
                # Take only the L content token positions
                layer_feats[i].append(captured[i][:, prefix:prefix+L, :])

        x_hat_all = torch.cat(x_hat_list, dim=0)  # (N, L, 512)
        layer_feats_all = [torch.cat(layer_feats[i], dim=0) for i in range(depth)]  # each (N, L, 768)

        out_path = os.path.join(output_dir, f"layer_states_t{t_val:.3f}.pt")
        torch.save({
            "t": t_val,
            "x_hat": x_hat_all,
            "y_tokens": y_tokens_all,
            "attn_mask": attn_mask_all,
            "layer_feats": layer_feats_all,  # list of depth tensors, each (N, L, hidden_size)
        }, out_path)
        print(f"  Saved {out_path}")


def train_linear_probe(feats, labels, mask, d_in, vocab_size, device,
                       n_epochs=20, batch_size=4096, lr=1e-2):
    """Train a linear probe on (feats, labels) and return validation accuracy."""
    N, L, d = feats.shape
    feats_flat = feats.reshape(-1, d)
    labels_flat = labels.reshape(-1)
    mask_flat = mask.reshape(-1).bool()

    feats_valid = feats_flat[mask_flat].float()
    labels_valid = labels_flat[mask_flat].long()

    # 80/20 split
    n_total = len(feats_valid)
    n_train = int(0.8 * n_total)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_ds = TensorDataset(feats_valid[train_idx], labels_valid[train_idx])
    val_ds   = TensorDataset(feats_valid[val_idx],   labels_valid[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size*2)

    probe = nn.Linear(d, vocab_size).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(n_epochs):
        probe.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(probe(xb), yb).backward()
            opt.step()

    probe.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            correct += (probe(xb).argmax(-1) == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/exp07b_baseline")
    parser.add_argument("--n_seqs", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--latent_std", type=float, default=0.2)
    parser.add_argument("--t_values", type=str, default="0.10,0.20,0.30,0.50,0.70,1.00")
    parser.add_argument("--skip_collect", action="store_true",
                        help="Skip collection phase; load existing layer_states_*.pt files")
    parser.add_argument("--fixed_noise_seed", type=int, default=42,
                        help="Seed for fixed noise; same ε reused across all t values")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_values = [float(t) for t in args.t_values.split(",")]
    print(f"[EXP-07b] Device={device}, t_values={t_values}")

    if not args.skip_collect:
        model, encoder_config, config = load_model(args.checkpoint, device)
        encoder = load_encoder(getattr(config, "encoder_model_name", "t5-small"), device)
        input_ids_np, attn_mask_np = load_data(args.n_seqs, args.max_length)
        collect_layer_states(model, encoder, config, input_ids_np, attn_mask_np,
                             t_values, args.batch_size, args.latent_std, device,
                             args.output_dir, fixed_noise_seed=args.fixed_noise_seed)
        # Free model memory
        del model, encoder
        torch.cuda.empty_cache()

    # Train probes on collected states
    results = []
    import re
    state_files = sorted([
        f for f in os.listdir(args.output_dir) if f.startswith("layer_states_t")
    ])
    print(f"\n[EXP-07b] Training probes on {len(state_files)} t-values ...")
    for fname in state_files:
        data = torch.load(os.path.join(args.output_dir, fname), map_location="cpu")
        t_val = data["t"]
        y = data["y_tokens"]
        mask = data["attn_mask"]
        x_hat = data["x_hat"]
        layer_feats = data["layer_feats"]
        depth = len(layer_feats)
        vocab_size = y.max().item() + 1

        print(f"\n  t={t_val:.3f}")
        row = {"t": t_val, "layers": {}}

        # Probe each layer
        for i in range(depth):
            acc = train_linear_probe(layer_feats[i], y, mask, layer_feats[i].shape[-1],
                                     vocab_size, device)
            row["layers"][f"L{i}"] = acc
            print(f"    Layer {i:2d}: {acc:.4f}")

        # Also probe final x̂_t (512-dim)
        acc_final = train_linear_probe(x_hat, y, mask, x_hat.shape[-1], vocab_size, device)
        row["x_hat"] = acc_final
        print(f"    x̂_t (final): {acc_final:.4f}")

        results.append(row)

    out_path = os.path.join(args.output_dir, "layerwise_probe_accuracies.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXP-07b] Saved to {out_path}")

    # Print summary table
    if results:
        depth = len(results[0]["layers"])
        header = "  t    " + "".join(f"  L{i:2d}" for i in range(depth)) + "  x̂_t"
        print("\n" + header)
        for row in results:
            vals = "".join(f"  {row['layers'][f'L{i}']:.3f}" for i in range(depth))
            vals += f"  {row['x_hat']:.3f}"
            print(f"  {row['t']:.2f} {vals}")


if __name__ == "__main__":
    main()
