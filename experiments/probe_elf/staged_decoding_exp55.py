#!/usr/bin/env python3
"""
EXP-55: Two-pass staged decoding (Strategy B)

Hypothesis: high-confidence positions from ODE-32 pass 1 can be committed as
conditioning for pass 2, letting the model fill remaining positions with
better context about the committed tokens.

This tests the "B" direction of progressive commitment: positions decode EARLY
(ahead of the rest), rather than EXP-31v2's DF direction of positions being
left noisy for longer.

Key mechanism: cond_seq/cond_mask pins committed positions at x_pred from pass 1.
At every ODE step: v=0 for committed positions (z stays fixed), x_pred at
committed positions → cond_seq. This bypasses self_cond_proj entirely for
committed positions, potentially benefiting kd2 (which has B11 anti-correlation).

Arms:
  standard      : ODE-32, no staging (reference)
  staged_left50 : pass1=ODE-32, commit left 50% positions, pass2=ODE-32
  staged_conf90 : pass1=ODE-32, commit confidence>0.90, pass2=ODE-32
  staged_conf80 : pass1=ODE-32, commit confidence>0.80, pass2=ODE-32
  staged_conf70 : pass1=ODE-32, commit confidence>0.70, pass2=ODE-32

Models: kd_cr, kd2 (both; kd2 particularly interesting — cond_seq bypasses
self_cond_proj anti-correlation from EXP-43)

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/staged_decoding_exp55.py --device cuda:1
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import restore_cond, _ode_step, get_sampling_steps

CHECKPOINTS = {
    "kd_cr": "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":   "converted/elf_b-owt-kd2_torch.pt",
}
N_SEQ      = 256
N_STEPS    = 32
SEED       = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
SCCFG      = 1.0   # sccfg=1 for consistency with h10 SC baseline (EXP-54b)
OUT_DIR    = Path("results/exp55_staged")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)

CONF_THRESHOLDS = {
    "staged_conf90": 0.90,
    "staged_conf80": 0.80,
    "staged_conf70": 0.70,
}


class _Cfg:
    t_eps = 0.05
    self_cond_prob = 1.0
    num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 1.0
    use_bf16 = True


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_ode(model, z0, t_steps, sccfg, device):
    """Standard ODE-32. Returns (z_final, x_pred_final)."""
    cfg = _Cfg()
    B, L, D = z0.shape
    cond_seq  = torch.zeros_like(z0)
    cond_mask = torch.zeros(B, L, dtype=z0.dtype, device=device)
    step_kw = dict(
        model=model, config=cfg, cfg_scale=1.0, self_cond_cfg_scale=sccfg,
        cond_seq=cond_seq, cond_seq_mask=cond_mask,
    )
    z      = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)

    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(t_steps.shape[0] - 2):
            t      = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kw)
        t      = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kw)
    return z, x_pred


@torch.no_grad()
def get_top1_prob(x_pred, model, device):
    """Decode-branch top-1 softmax probability per position. Returns (B, L)."""
    B = x_pred.shape[0]
    z_in = torch.cat([x_pred.detach(), torch.zeros_like(x_pred)], dim=-1)
    t_dec = torch.ones(B, dtype=x_pred.dtype, device=device)
    sc    = torch.ones(B, dtype=x_pred.dtype, device=device)
    use_bf16 = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, logits, _ = model(z_in, t_dec, deterministic=True,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
    return F.softmax(logits.float(), dim=-1).max(dim=-1).values  # (B, L)


@torch.no_grad()
def generate_staged(model, z0, t_steps, cond_seq, cond_mask_bl, sccfg, device):
    """ODE-32 pass 2 with committed positions pinned via cond_seq/cond_mask.

    cond_seq: (B, L, D) — x_pred from pass 1
    cond_mask_bl: (B, L) float — 1.0 for committed positions
    """
    cfg = _Cfg()
    B, L, D = z0.shape
    step_kw = dict(
        model=model, config=cfg, cfg_scale=1.0, self_cond_cfg_scale=sccfg,
        cond_seq=cond_seq, cond_seq_mask=cond_mask_bl,
    )
    # Pin committed positions in initial z to cond_seq (x_pred from pass 1)
    z      = restore_cond(z0.clone(), cond_seq, cond_mask_bl)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask_bl)

    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(t_steps.shape[0] - 2):
            t      = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kw)
        t      = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z, _ = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kw)
    return z


@torch.no_grad()
def decode_z(z, model, device):
    B = z.shape[0]
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
    t1 = torch.ones(B, dtype=z.dtype, device=device)
    sc = torch.ones(B, dtype=z.dtype, device=device)
    use_bf16 = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, logits, _ = model(z_in, t1, deterministic=True, self_cond_cfg_scale=sc,
                             decoder_step_active=True)
    return logits.argmax(dim=-1)


def ids_to_text(ids, tokenizer):
    eos = tokenizer.eos_token_id
    texts = []
    for row in ids.tolist():
        try:
            row = row[:row.index(eos)]
        except ValueError:
            pass
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def compute_ppl(texts, ppl_model, ppl_tok, device, max_length=256):
    if not texts:
        return float("nan")
    enc = ppl_tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, return_attention_mask=True)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    total_nll = total_tok = 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], 8):
            ids_b, msk_b = ids[i:i+8], mask[i:i+8]
            out = ppl_model(input_ids=ids_b, attention_mask=msk_b)
            logits = out.logits[:, :-1, :].float()
            tgts   = ids_b[:, 1:]
            msk_s  = msk_b[:, 1:].bool()
            nll = -(F.log_softmax(logits, dim=-1)
                    .gather(-1, tgts.unsqueeze(-1)).squeeze(-1)
                    * msk_s.float()).sum().item()
            total_nll += nll
            total_tok += msk_s.sum().item()
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ckpt(ckpt_name, device, ppl_model, ppl_tok, elf_tok):
    print(f"\n{'='*60}\nCheckpoint: {ckpt_name}\n{'='*60}")

    ckpt_path = REPO_ROOT / CHECKPOINTS[ckpt_name]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    # Fresh noise for pass 2 (independent seed)
    gen2 = torch.Generator(device=device)
    gen2.manual_seed(SEED + 1000)
    all_z0_pass2 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen2, device=device)

    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)
    results = {}

    # ── Pass 1 for all sequences (standard ODE-32) ──
    print("  Running pass 1 (ODE-32, all seqs)...")
    all_z_p1   = []
    all_xp_p1  = []
    all_texts_standard = []
    for b in range(0, N_SEQ, BATCH_SIZE):
        z0_b = all_z0[b:b+BATCH_SIZE]
        z_p1_b, xp_p1_b = generate_ode(model, z0_b, t_steps, SCCFG, device)
        ids = decode_z(z_p1_b, model, device)
        all_texts_standard.extend(ids_to_text(ids.cpu(), elf_tok))
        all_z_p1.append(z_p1_b.detach())
        all_xp_p1.append(xp_p1_b.detach())
    ppl_standard = compute_ppl(all_texts_standard, ppl_model, ppl_tok, device)
    print(f"  {'standard':<30} PPL = {ppl_standard:.2f}")
    results["standard"] = {"ppl": ppl_standard, "I": 0.0}

    # ── staged arms (pass 2) ──
    arm_specs = [("staged_left50", None)] + [(k, v) for k, v in CONF_THRESHOLDS.items()]

    for arm_name, conf_thresh in arm_specs:
        texts = []
        commit_fracs = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            b_end = min(b + BATCH_SIZE, N_SEQ)
            xp_p1_b = all_xp_p1[b // BATCH_SIZE]   # (<=BATCH_SIZE, L, D)
            z0_p2_b = all_z0_pass2[b:b_end]

            B_b, L, D = xp_p1_b.shape

            if arm_name == "staged_left50":
                cond_mask = torch.zeros(B_b, L, dtype=xp_p1_b.dtype, device=device)
                cond_mask[:, :L // 2] = 1.0
            else:
                # Decode branch confidence from pass-1 x_pred
                conf = get_top1_prob(xp_p1_b, model, device)  # (B, L)
                cond_mask = (conf > conf_thresh).float()       # (B, L)

            commit_frac = cond_mask.mean().item()
            commit_fracs.append(commit_frac)

            z_out = generate_staged(
                model, z0_p2_b, t_steps,
                cond_seq=xp_p1_b, cond_mask_bl=cond_mask,
                sccfg=SCCFG, device=device,
            )
            ids = decode_z(z_out, model, device)
            texts.extend(ids_to_text(ids.cpu(), elf_tok))

        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        avg_commit = sum(commit_fracs) / len(commit_fracs) if commit_fracs else 0.0
        I = ppl - ppl_standard
        print(f"  {arm_name:<30} PPL = {ppl:.2f}  I={I:+.2f}  "
              f"commit_frac={avg_commit*100:.1f}%")
        results[arm_name] = {"ppl": ppl, "I": I, "commit_frac": avg_commit}

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--ckpt", default=None, help="single checkpoint name to run")
    args = parser.parse_args()
    device = torch.device(args.device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 Large PPL model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    ppl_tok = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tok.pad_token is None:
        ppl_tok.pad_token = ppl_tok.eos_token
        ppl_tok.pad_token_id = ppl_tok.eos_token_id
    ppl_model = (AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.bfloat16)
                 .to(device).eval())
    elf_tok = T5Tokenizer.from_pretrained("t5-small")

    ckpts_to_run = [args.ckpt] if args.ckpt else list(CHECKPOINTS.keys())
    all_results = {}
    for ck in ckpts_to_run:
        all_results[ck] = run_ckpt(ck, device, ppl_model, ppl_tok, elf_tok)

    out = OUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")

    print("\n=== SUMMARY ===")
    for ck, r in all_results.items():
        print(f"\n{ck}:")
        std_ppl = r["standard"]["ppl"]
        for arm, v in r.items():
            cfrac = f"  commit={v.get('commit_frac',0)*100:.0f}%" if arm != "standard" else ""
            print(f"  {arm:<30} PPL={v['ppl']:.1f}  I={v['I']:+.1f}{cfrac}")


if __name__ == "__main__":
    main()
