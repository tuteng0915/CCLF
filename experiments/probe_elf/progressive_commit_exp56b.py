#!/usr/bin/env python3
"""
EXP-56b: Within-ODE progressive commitment — commit timing sweep

EXP-56 showed prog_t05_c70 gives I=−164 (kd_cr) and I=−72 (kd2).
This experiment sweeps the commit time t ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
with threshold fixed at 0.7 (best from EXP-56) to find the optimal
commitment point in the ODE trajectory.

Motivation:
- Too early (t=0.3): x_pred is noisy → low-quality anchors → may hurt
- Too late (t=0.7): few steps left to refine → anchors help less
- t=0.5 (EXP-56): motivated by EXP-42 CKA bifurcation and EXP-54c gate

Arms: standard + prog_tXX_c70 for XX ∈ {30, 40, 50, 60, 70}
Models: kd_cr, kd2
N=256, sccfg=1, seed=42

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/progressive_commit_exp56b.py --device cuda:0
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
    "kd_cr":    "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":      "converted/elf_b-owt-kd2_torch.pt",
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
}
N_SEQ      = 256
N_STEPS    = 32
SEED       = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
SCCFG      = 1.0
CONF_THRESH = 0.70   # fixed at best EXP-56 threshold
COMMIT_TIMES = [0.30, 0.40, 0.50, 0.60, 0.70]
OUT_DIR    = Path("results/exp56b_timing_sweep")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)


class _Cfg:
    t_eps = 0.05
    self_cond_prob = 1.0
    num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 1.0
    use_bf16 = True


@torch.no_grad()
def get_top1_prob(x_pred, model, device):
    B = x_pred.shape[0]
    z_in = torch.cat([x_pred.detach(), torch.zeros_like(x_pred)], dim=-1)
    t_dec = torch.ones(B, dtype=x_pred.dtype, device=device)
    sc    = torch.ones(B, dtype=x_pred.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(z_in, t_dec, deterministic=True,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
    return F.softmax(logits.float(), dim=-1).max(dim=-1).values  # (B, L)


@torch.no_grad()
def generate_progressive(model, z0, t_steps, commit_t, conf_thresh, device):
    """ODE-32 with commitment at first t_next >= commit_t. Returns (z, commit_info)."""
    cfg = _Cfg()
    B, L, D = z0.shape
    cond_seq  = torch.zeros_like(z0)
    cond_mask = torch.zeros(B, L, dtype=z0.dtype, device=device)

    z      = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    committed = False
    commit_info = None

    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(t_steps.shape[0] - 2):
            t      = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                                  model=model, config=cfg, cfg_scale=1.0,
                                  self_cond_cfg_scale=SCCFG,
                                  cond_seq=cond_seq, cond_seq_mask=cond_mask)

            if conf_thresh is not None and not committed and t_next >= commit_t:
                conf = get_top1_prob(x_pred, model, device)
                new_committed = (conf > conf_thresh)
                commit_frac = new_committed.float().mean().item()
                cond_seq  = x_pred.detach().clone()
                cond_mask = new_committed.float()
                z = z.clone()
                z[new_committed] = x_pred.detach()[new_committed]
                committed = True
                commit_info = {"t_commit": t_next, "commit_frac": commit_frac}

        # last step
        z, _ = _ode_step(z=z, t=t_steps[-2].item(), t_next=t_steps[-1].item(),
                         x_pred_prev=x_pred, model=model, config=cfg,
                         cfg_scale=1.0, self_cond_cfg_scale=SCCFG,
                         cond_seq=cond_seq, cond_seq_mask=cond_mask)
    return z, commit_info


@torch.no_grad()
def decode_z(z, model, device):
    B = z.shape[0]
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
    t1 = torch.ones(B, dtype=z.dtype, device=device)
    sc = torch.ones(B, dtype=z.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(z_in, t1, deterministic=True, self_cond_cfg_scale=sc,
                             decoder_step_active=True)
    return logits.argmax(dim=-1)


def ids_to_text(ids, tokenizer):
    eos = tokenizer.eos_token_id
    texts = []
    for row in ids.tolist():
        try: row = row[:row.index(eos)]
        except ValueError: pass
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def compute_ppl(texts, ppl_model, ppl_tok, device, max_length=256):
    if not texts: return float("nan")
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


def run_ckpt(ckpt_name, device, ppl_model, ppl_tok, elf_tok):
    print(f"\n{'='*60}\nCheckpoint: {ckpt_name}\n{'='*60}")
    ckpt = torch.load(REPO_ROOT / CHECKPOINTS[ckpt_name], map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    results = {}

    arm_specs = [("standard", None)] + \
                [(f"prog_t{int(ct*100):02d}_c70", ct) for ct in COMMIT_TIMES]

    ppl_standard = None
    for arm_name, commit_t in arm_specs:
        texts = []
        commit_fracs = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            z0_b = all_z0[b:b+BATCH_SIZE]
            z_out, cinfo = generate_progressive(model, z0_b, t_steps, commit_t,
                                                CONF_THRESH if commit_t else None, device)
            ids = decode_z(z_out, model, device)
            texts.extend(ids_to_text(ids.cpu(), elf_tok))
            if cinfo: commit_fracs.append(cinfo["commit_frac"])

        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        if arm_name == "standard":
            ppl_standard = ppl
        avg_commit = sum(commit_fracs) / len(commit_fracs) if commit_fracs else 0.0
        I = 0.0 if arm_name == "standard" else ppl - ppl_standard
        cfrac_str = f"  commit={avg_commit*100:.1f}%" if commit_t else ""
        print(f"  {arm_name:<30} PPL = {ppl:.2f}  I={I:+.2f}{cfrac_str}")
        results[arm_name] = {"ppl": ppl, "I": I, "commit_frac": avg_commit,
                              "commit_t": commit_t}

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt", default=None)
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
        for arm, v in r.items():
            ct = v.get("commit_t")
            cfrac = f"  commit={v.get('commit_frac',0)*100:.0f}%" if ct else ""
            print(f"  {arm:<30} PPL={v['ppl']:.1f}  I={v['I']:+.1f}{cfrac}")


if __name__ == "__main__":
    main()
