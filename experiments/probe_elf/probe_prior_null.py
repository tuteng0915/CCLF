"""
EXP-05v3: Prior Estimation via Global Null (Zero-Signal Input)

Fixes two methodological problems from EXP-05:
  1. Prior is now estimated from z_t_null = (1-t)*eps (no x_clean signal at all)
     instead of batch-shuffle (which gave wrong-instance posterior).
  2. Prior probability is computed by averaging softmax outputs over N_NULL noise draws
     instead of softmax(mean(logits)) — correct Jensen inequality treatment.

Metrics at each t:
  G_oracle(t)  — argmax(logits_oracle) vs y_true (standard oracle protocol)
  G_null(t)    — argmax(logits_null, averaged probs) vs y_true (null model accuracy)
  G_debias(t)  — argmax(log p_t - log q_t) vs y_true (debiased)
  rank_oracle  — rank of y_true in oracle probability distribution
  rank_null    — rank of y_true in null probability distribution
  rank_debias  — rank of y_true in debiased log-ratio distribution

Usage:
    cd /home/wjzhang/tt_workspace/model/CCLF/CCLF/models/ELF-torch
    CUDA_VISIBLE_DEVICES=0 python ../../experiments/probe_elf/probe_prior_null.py \\
        --checkpoint converted/elf_b-owt-baseline_torch.pt \\
        --config src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir ../../experiments/probe_elf/results/exp05v3_baseline \\
        --n_samples 128 --n_null 8 --label baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).parent
_ELF_TORCH_DIR = _SCRIPT_DIR.parents[1] / "models" / "ELF-torch"
_ELF_SRC = _ELF_TORCH_DIR / "src"
if str(_ELF_SRC) not in sys.path:
    sys.path.insert(0, str(_ELF_SRC))

from configs.config import load_config_from_yaml
from modules.model import ELF_models
from modules.t5_encoder import get_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", default="results/exp05v3")
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--n_oracle", type=int, default=4, help="Oracle noise seeds per (t, sample)")
    p.add_argument("--n_null", type=int, default=8, help="Null prior noise seeds per (t, sample)")
    p.add_argument("--n_t_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--label", default="baseline")
    return p.parse_args()


def load_model(config_path, checkpoint_path, device):
    orig_dir = os.getcwd()
    os.chdir(str(_ELF_TORCH_DIR))
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
            print(f"[model] missing {len(missing)} keys")
        print(f"[model] loaded (EMA: {'ema_params1' in ckpt})")
    return model, tokenizer, t5_model, config


@torch.no_grad()
def backbone_forward(model, z_input, t_val, config, device, batch_size):
    """Run backbone and return decoder logits (B, L, V)."""
    all_logits = []
    N = z_input.shape[0]
    for i in range(0, N, batch_size):
        z_b = z_input[i:i+batch_size].to(device)
        B_b = z_b.shape[0]
        if config.self_cond_prob > 0:
            z_in = torch.cat([z_b, torch.zeros_like(z_b)], dim=-1)
        else:
            z_in = z_b
        t_b = torch.full((B_b,), t_val, device=device, dtype=torch.float32)
        sc_scale = torch.zeros(B_b, device=device) if config.num_self_cond_cfg_tokens > 0 else None
        _, logits, _ = model(z_in, t_b, self_cond_cfg_scale=sc_scale, decoder_step_active=True)
        all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)  # (N, L, V)


def compute_rank(probs_or_logits, gt_ids):
    """Compute mean rank (0-indexed) of y_true token in distribution."""
    N, L, V = probs_or_logits.shape
    sorted_idx = probs_or_logits.argsort(dim=-1, descending=True)  # (N, L, V)
    gt_ids_exp = gt_ids.unsqueeze(-1)  # (N, L, 1)
    rank = (sorted_idx == gt_ids_exp).nonzero(as_tuple=False)  # (..., 3)
    rank_matrix = torch.zeros(N, L, dtype=torch.long)
    rank_matrix[rank[:,0], rank[:,1]] = rank[:,2]
    return rank_matrix.float().mean().item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp05v3] Loading model from {args.checkpoint}")
    model, tokenizer, t5_model, config = load_model(args.config, args.checkpoint, device)

    latent_mean = getattr(config, "latent_mean", 0.0)
    latent_std = getattr(config, "latent_std", 0.2)
    seq_len = config.max_length

    # Load dataset sequences
    print(f"[exp05v3] Loading {args.n_samples} sequences...")
    from datasets import load_dataset
    ds = load_dataset("embedded-language-flows/openwebtext-t5", split="train", streaming=True)
    samples = []
    for ex in ds:
        ids = torch.tensor(ex["input_ids"][:seq_len], dtype=torch.long)
        if len(ids) < seq_len:
            ids = F.pad(ids, (0, seq_len - len(ids)), value=tokenizer.eos_token_id or 1)
        mask = (ids != (tokenizer.eos_token_id or 1)).long()
        samples.append((ids, mask))
        if len(samples) >= args.n_samples:
            break
    print(f"[exp05v3] Loaded {len(samples)} sequences")

    all_ids = torch.stack([s[0] for s in samples])
    all_masks = torch.stack([s[1] for s in samples])
    gt_ids = all_ids  # (N, L) — T5 token IDs

    # Encode with T5
    x_clean_list = []
    for i in range(0, len(samples), args.batch_size):
        batch_ids = all_ids[i:i+args.batch_size].to(device)
        batch_masks = all_masks[i:i+args.batch_size].to(device)
        with torch.no_grad():
            out = t5_model.model(input_ids=batch_ids, attention_mask=batch_masks)
            emb = (out.last_hidden_state - latent_mean) / latent_std
        x_clean_list.append(emb.cpu())
    x_clean = torch.cat(x_clean_list, dim=0)  # (N, L, d)
    d = x_clean.shape[-1]
    print(f"[exp05v3] x_clean: {x_clean.shape}, gt_ids: {gt_ids.shape}")

    t_grid = np.linspace(0.05, 1.0, args.n_t_steps)
    torch.manual_seed(42)

    results = {
        "t": t_grid.tolist(),
        "G_oracle": [], "G_null": [], "G_debias": [],
        "rank_oracle": [], "rank_null": [], "rank_debias": [],
        "label": args.label,
        "n_samples": len(samples),
        "seq_len": seq_len,
        "n_oracle": args.n_oracle,
        "n_null": args.n_null,
    }

    print(f"\n{'t':>5} | {'G_oracle':>9} | {'G_null':>8} | {'G_debias':>9} | "
          f"{'rank_or':>8} | {'rank_nu':>8} | {'rank_db':>8}")
    print("-" * 75)

    for t in t_grid:
        # ── Oracle: z_t = t*x_clean + (1-t)*eps (mean over n_oracle seeds) ──
        oracle_correct = 0
        oracle_total = 0
        oracle_logits_sum = None  # for rank calculation

        for _ in range(args.n_oracle):
            eps = torch.randn_like(x_clean)
            z_t = t * x_clean + (1.0 - t) * eps
            logits = backbone_forward(model, z_t, t, config, device, args.batch_size)
            preds = logits.argmax(dim=-1)
            oracle_correct += (preds == gt_ids).float().sum().item()
            oracle_total += preds.numel()
            if oracle_logits_sum is None:
                oracle_logits_sum = logits.float()
            else:
                oracle_logits_sum += logits.float()

        G_oracle = oracle_correct / oracle_total
        oracle_logits_avg = oracle_logits_sum / args.n_oracle  # (N, L, V)

        # ── Null prior: z_t_null = (1-t)*eps (NO x_clean signal) ──
        # Average over softmax(logits), not softmax(mean(logits)) — correct Jensen treatment
        null_probs_sum = None

        for _ in range(args.n_null):
            eps_null = torch.randn(len(samples), seq_len, d)
            z_t_null = (1.0 - t) * eps_null
            logits_null = backbone_forward(model, z_t_null, t, config, device, args.batch_size)
            probs_null = F.softmax(logits_null.float(), dim=-1)
            if null_probs_sum is None:
                null_probs_sum = probs_null
            else:
                null_probs_sum += probs_null

        q_probs = null_probs_sum / args.n_null  # (N, L, V) — averaged softmax

        # G_null: accuracy from null prior
        null_preds = q_probs.argmax(dim=-1)
        G_null = (null_preds == gt_ids).float().mean().item()

        # ── Debiased: log p_t - log q_t ──
        p_probs = F.softmax(oracle_logits_avg, dim=-1)
        log_p = torch.log(p_probs + 1e-12)
        log_q = torch.log(q_probs + 1e-12)
        r_t = log_p - log_q  # (N, L, V)
        debias_preds = r_t.argmax(dim=-1)
        G_debias = (debias_preds == gt_ids).float().mean().item()

        # ── Rank metrics ──
        rank_oracle = compute_rank(oracle_logits_avg, gt_ids)
        rank_null = compute_rank(q_probs, gt_ids)
        rank_debias = compute_rank(r_t, gt_ids)

        results["G_oracle"].append(float(G_oracle))
        results["G_null"].append(float(G_null))
        results["G_debias"].append(float(G_debias))
        results["rank_oracle"].append(float(rank_oracle))
        results["rank_null"].append(float(rank_null))
        results["rank_debias"].append(float(rank_debias))

        print(f"{t:5.2f} | {G_oracle:9.4f} | {G_null:8.4f} | {G_debias:9.4f} | "
              f"{rank_oracle:8.0f} | {rank_null:8.0f} | {rank_debias:8.0f}")

        del oracle_logits_sum, oracle_logits_avg, null_probs_sum, q_probs, p_probs, log_p, log_q, r_t
        torch.cuda.empty_cache()

    out_path = out_dir / f"prior_null_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[exp05v3] Saved to {out_path}")

    # Summary
    t_grid_arr = np.array(results["t"])
    for target_t, tname in [(0.20, "0.20"), (0.30, "0.30"), (0.50, "0.50")]:
        idx = int(np.argmin(np.abs(t_grid_arr - target_t)))
        print(f"t≈{tname}: G_oracle={results['G_oracle'][idx]:.3f} "
              f"G_null={results['G_null'][idx]:.4f} "
              f"G_debias={results['G_debias'][idx]:.3f} "
              f"rank_oracle={results['rank_oracle'][idx]:.0f}")

    print("\n[exp05v3] Done.")


if __name__ == "__main__":
    main()
