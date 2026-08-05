#!/usr/bin/env python3
"""
EXP-57: Stacked h10 SC + within-ODE progressive commitment

Tests whether the two best interventions are complementary or redundant:
  - h10 SC (EXP-54b): replaces x_pred with final_layer(h10) at t_next >= 0.5
    → targets B11 anti-correlation in kd2's SC pathway (I=−130 for kd2)
  - Progressive commitment (EXP-56): commits high-confidence positions at t=0.5
    via cond_mask, providing stable anchors for second half
    → I=−72 for kd2, I=−164 for kd_cr

If complementary (different mechanisms):
  - h10 SC improves x_pred quality → better confidence estimates → better commitment
  - Stacking might approach I=−130 + some fraction of −72 for kd2

Arms (all models):
  standard       : natural SC (reference)
  h10_only       : h10 SC gate t>=0.5, no commitment (reproduce EXP-54b/EXP-56 standard)
  prog_only      : commitment c70 t=0.5, no h10 SC (reproduce EXP-56)
  h10_prog       : BOTH — h10 SC for uncommitted positions + commit c70 at t=0.5

Key design for h10_prog:
  1. At t_next >= 0.5: apply h10 SC → x_pred_h10
  2. At the FIRST step where t_next >= 0.5: compute confidence from x_pred_h10
     (better quality from h10), commit positions, update cond_mask
  3. After commitment: h10 SC only updates UNCOMMITTED positions (re-apply restore_cond)

Models: kd_cr, kd2
N=256, sccfg=1, seed=42

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/stacked_h10_prog_exp57.py --device cuda:2
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
SCCFG      = 1.0
SC_T_MIN   = 0.5   # h10 SC gate (from EXP-54c)
COMMIT_T   = 0.5   # progressive commitment timing
CONF_THRESH = 0.70  # best from EXP-56
OUT_DIR    = Path("results/exp57_stacked")

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


class IntermediateSCHook:
    """Captures h_10 during model forward passes (same as EXP-54)."""
    def __init__(self, model):
        self.model = model
        self._h10 = None
        self._h11_seq_len = None
        self._h10_handle = model.blocks[10].register_forward_hook(self._cap_h10)
        self._h11_handle = model.final_layer.register_forward_pre_hook(self._cap_h11)

    def _cap_h10(self, module, inp, out):
        self._h10 = out.detach().float()

    def _cap_h11(self, module, inp):
        self._h11_seq_len = inp[0].shape[1]

    def get_xhat_h10(self):
        h10 = self._h10
        seq_len = self._h11_seq_len
        h10_seq = h10[:, -seq_len:, :]
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            return self.model.final_layer(h10_seq.to(dev))

    def remove(self):
        self._h10_handle.remove()
        self._h11_handle.remove()


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
def generate(model, hook, z0, t_steps, arm, device):
    """
    arm='standard'  : natural SC, no commitment
    arm='h10_only'  : h10 SC at t_next>=0.5, no commitment
    arm='prog_only' : commitment c70 at t=0.5, no h10 SC
    arm='h10_prog'  : h10 SC + commitment (stacked)
    """
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

            # Apply h10 SC (for arms that use it)
            if arm in ("h10_only", "h10_prog") and t_next >= SC_T_MIN:
                x_pred_h10 = hook.get_xhat_h10().to(device)
                x_pred = x_pred_h10
                # Re-pin committed positions to cond_seq (h10 output overrides; we fix it)
                if committed:
                    x_pred = restore_cond(x_pred, cond_seq, cond_mask)

            # Commit positions at first t_next >= COMMIT_T
            if arm in ("prog_only", "h10_prog") and not committed and t_next >= COMMIT_T:
                # For h10_prog: use h10-improved x_pred for confidence
                # For prog_only: use standard x_pred
                conf = get_top1_prob(x_pred, model, device)
                new_committed = (conf > CONF_THRESH)
                commit_frac = new_committed.float().mean().item()
                cond_seq  = x_pred.detach().clone()
                cond_mask = new_committed.float()
                z = z.clone()
                z[new_committed] = x_pred.detach().to(z.dtype)[new_committed]
                committed = True
                commit_info = {"t_commit": t_next, "commit_frac": commit_frac}

        # Last step
        z, x_pred = _ode_step(z=z, t=t_steps[-2].item(), t_next=t_steps[-1].item(),
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
    hook = IntermediateSCHook(model)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    results = {}
    arm_names = ["standard", "h10_only", "prog_only", "h10_prog"]
    ppl_standard = None

    for arm_name in arm_names:
        texts = []
        commit_fracs = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            z_out, cinfo = generate(model, hook, all_z0[b:b+BATCH_SIZE],
                                    t_steps, arm_name, device)
            ids = decode_z(z_out, model, device)
            texts.extend(ids_to_text(ids.cpu(), elf_tok))
            if cinfo: commit_fracs.append(cinfo["commit_frac"])

        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        if arm_name == "standard":
            ppl_standard = ppl
        avg_commit = sum(commit_fracs) / len(commit_fracs) if commit_fracs else 0.0
        I = 0.0 if arm_name == "standard" else ppl - ppl_standard
        cfrac_str = f"  commit={avg_commit*100:.1f}%" if commit_fracs else ""
        print(f"  {arm_name:<20} PPL = {ppl:.2f}  I={I:+.2f}{cfrac_str}")
        results[arm_name] = {"ppl": ppl, "I": I, "commit_frac": avg_commit}

    hook.remove()
    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
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
        std_ppl = r["standard"]["ppl"]
        print(f"\n{ck}: (standard={std_ppl:.1f})")
        for arm, v in r.items():
            cfrac = f"  commit={v.get('commit_frac',0)*100:.0f}%" if arm not in ("standard","h10_only") else ""
            print(f"  {arm:<20} PPL={v['ppl']:.1f}  I={v['I']:+.1f}{cfrac}")
    print("\nReference:")
    print("  EXP-54b h10_only kd2 : I=−130 (256 seq, sccfg=1)")
    print("  EXP-56  prog_only kd2 : I=−72  (256 seq, sccfg=1, c70)")
    print("  EXP-56  prog_only kd_cr: I=−164 (256 seq, sccfg=1, c70)")


if __name__ == "__main__":
    main()
