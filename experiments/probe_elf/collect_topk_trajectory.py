"""
Collect per-position, per-step top-K token predictions along ODE trajectory.

Uses correct decode path: L11 hidden (768-dim) → proj_kernel+GELU → unembed_kernel → logits.
Proxy GT = final-step decode-path prediction (same as EXP-01v2).

Output: results/topk_traj/topk_data.json
  {
    "seqs": [
      {
        "seq_id": int,
        "steps": [
          {
            "t": float,
            "positions": [
              {"top_tokens": [int, ...], "top_logits": [float, ...]}  # len=K, per position
            ]
          }
        ],
        "proxy_gt": [int, ...]  # (L,) final-step prediction
      }
    ],
    "vocab": dict mapping token_id -> token_string,  # subset of used tokens
    "K": int, "n_steps": int, "n_seqs": int, "seq_length": int
  }

Usage:
  # Run after GPU frees up, uses CPU-only re-analysis of existing trajectories
  python experiments/probe_elf/collect_topk_trajectory.py \
    --traj_dir results/exp01/trajectories \
    --checkpoint converted/elf_b-owt-kd-cr_torch.pt \
    --n_seqs 20 --K 5 \
    --output_dir results/topk_traj
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


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
    return state.model, decode_weights


@torch.no_grad()
def get_topk_at_step(model, z_t, t_val, K, decode_weights, device, batch_size=8):
    """Returns top_tokens (B, L, K) and top_logits (B, L, K) for given z_t."""
    proj_k = decode_weights["proj_kernel"].to(device)
    proj_b = decode_weights["proj_bias"].to(device)
    unemb_k = decode_weights["unembed_kernel"].to(device)
    unemb_b = decode_weights["unembed_bias"].to(device)

    captured = {}
    def hook_fn(mod, inp, out):
        captured["h11"] = out.float().cpu()

    handle = model.blocks[11].register_forward_hook(hook_fn)

    B, L, d = z_t.shape
    all_tokens, all_logits = [], []
    sc_ones = torch.ones(batch_size, device=device)

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        Bb = end - start
        zb = z_t[start:end].to(device)
        zeros_sc = torch.zeros_like(zb)
        z_in = torch.cat([zb, zeros_sc], dim=-1)
        t_batch = torch.full((Bb,), t_val, dtype=torch.float32, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(z_in, t_batch, decoder_step_active=None,
                  self_cond_cfg_scale=sc_ones[:Bb], attention_mask=None)

        h = captured["h11"][:Bb]  # (Bb, total, 768)
        prefix = h.shape[1] - L
        h_c = h[:, prefix:prefix + L, :].to(device)

        hidden = F.gelu(h_c @ proj_k + proj_b, approximate="tanh")
        logits = hidden @ unemb_k + unemb_b  # (Bb, L, V)

        top_log, top_tok = logits.topk(K, dim=-1)  # (Bb, L, K)
        all_tokens.append(top_tok.cpu())
        all_logits.append(top_log.cpu())

    handle.remove()
    return torch.cat(all_tokens, dim=0), torch.cat(all_logits, dim=0)  # (B, L, K)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", default="results/exp01/trajectories")
    parser.add_argument("--checkpoint", default="converted/elf_b-owt-kd-cr_torch.pt")
    parser.add_argument("--n_seqs", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--output_dir", default="results/topk_traj")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_pos", type=int, default=256,
                        help="Only keep first max_pos positions to limit output size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[topk] Device: {device}, K={args.K}, n_seqs={args.n_seqs}")

    model, decode_weights = load_model(args.checkpoint, device)

    # Load T5 tokenizer for token string lookup
    tok_vocab = {}
    if _HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            print(f"[topk] Loaded T5 tokenizer (vocab size = {tokenizer.vocab_size})")
        except Exception:
            tokenizer = None
    else:
        tokenizer = None

    traj_files = sorted(f for f in os.listdir(args.traj_dir) if f.endswith(".pt"))
    print(f"[topk] Found {len(traj_files)} traj files")

    seqs_collected = 0
    all_seq_data = []

    for fname in traj_files:
        if seqs_collected >= args.n_seqs:
            break
        print(f"\n[topk] {fname} ...")
        traj = torch.load(os.path.join(args.traj_dir, fname), map_location="cpu", weights_only=False)

        B_full = traj[0]["z_t"].shape[0]
        L_full = traj[0]["z_t"].shape[1]
        L = L_full  # pass full sequence; clip to max_pos after forward

        # Get proxy GT from final step (pass full sequence; clip to max_pos after)
        last_z = traj[-1]["z_t"]  # FULL sequence
        t_last = traj[-1]["t_next"]
        print(f"  Proxy GT at t={t_last:.3f} ...")
        gt_tokens, _ = get_topk_at_step(model, last_z, t_last, 1, decode_weights, device, args.batch_size)
        proxy_gt = gt_tokens[:, :args.max_pos, 0]  # (B, max_pos)

        n_take = min(args.n_seqs - seqs_collected, B_full)

        for seq_i in range(n_take):
            print(f"  Seq {seqs_collected+1}/{args.n_seqs} ...")
            seq_steps = []

            for step_idx, step in enumerate(traj):
                t_val = step["t"]
                z_t = step["z_t"][seq_i:seq_i+1]  # (1, L_full, d) — FULL sequence

                top_tok, top_log = get_topk_at_step(
                    model, z_t, t_val, args.K, decode_weights, device, batch_size=1)
                # top_tok: (1, L_full, K), top_log: (1, L_full, K) — clip to max_pos
                top_tok_np = top_tok[0, :args.max_pos].numpy().tolist()   # (max_pos, K)
                top_log_np = top_log[0, :args.max_pos].numpy().tolist()   # (max_pos, K)

                # Collect vocab strings for new tokens
                if tokenizer is not None:
                    for pos_tokens in top_tok_np:
                        for tid in pos_tokens:
                            if tid not in tok_vocab:
                                try:
                                    tok_vocab[tid] = tokenizer.convert_ids_to_tokens([tid])[0]
                                except Exception:
                                    tok_vocab[tid] = f"<{tid}>"

                step_data = {
                    "t": t_val,
                    "top_tokens": top_tok_np,  # (L, K) lists
                    "top_logits": [[round(x, 4) for x in row] for row in top_log_np],
                }
                if step_idx % 8 == 0:
                    print(f"    step {step_idx+1}/{len(traj)} (t={t_val:.3f})")
                seq_steps.append(step_data)

            all_seq_data.append({
                "seq_id": seqs_collected,
                "steps": seq_steps,
                "proxy_gt": proxy_gt[seq_i].numpy().tolist(),
            })
            seqs_collected += 1

    # Build output
    out = {
        "K": args.K,
        "n_seqs": seqs_collected,
        "n_steps": len(traj),
        "seq_length": L,
        "vocab": {str(k): v for k, v in tok_vocab.items()},
        "seqs": all_seq_data,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "topk_data.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[topk] Saved to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
