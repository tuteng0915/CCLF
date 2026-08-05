# EXP-55 Spec: Two-Pass Staged Decoding (Strategy B)

**Date:** 2026-07-31
**Status:** Running
**Script:** `models/ELF-torch/experiments/probe_elf/staged_decoding_exp55.py`

---

## Motivation

EXP-31v2 (Strategy A / Diffusion Forcing) showed that asynchronous noise levels
can help kd2 (I=−106 for freeze_1.0), validating that position-asymmetric denoising
has potential. Strategy A makes "lagging" positions noisier. Strategy B is the
**complementary direction**: make high-confidence positions "ahead" of the rest by
committing them as hard conditioning for a second ODE pass.

The cond_seq/cond_mask mechanism (already in the generation pipeline) pins committed
positions to their estimated x0 at **every** ODE step: v=0 (no movement in z) and
x_pred=cond_seq at those positions. Critically, this **bypasses self_cond_proj entirely**
for committed positions — motivated by EXP-43 (B11 dual-path conflict) and relevant
for kd2 which has anti-correlation in the SC pathway.

---

## Design

**Pass 1:** Standard ODE-32 → z_final, x_pred_final  
**Commitment:** Determine committed positions by:
  - `staged_left50`: first 50% positions by index (no model call needed)
  - `staged_conf{70,80,90}`: positions with decode-branch top-1 probability > threshold  
**Pass 2:** Fresh noise z0', run ODE-32 with `cond_seq=x_pred_pass1`, `cond_mask=committed`

**Why fresh noise for pass 2?** Starts from a different trajectory; committed positions
provide structure, uncommitted positions can explore a new path.

**Confidence metric:** decode-branch top-1 softmax probability (from `x_pred_final`).
Same decode branch used in EXP-13 (dec_sc), called once at end of pass 1.

---

## Parameters

| Parameter | Value |
|-----------|-------|
| Models | kd_cr, kd2 |
| N sequences | 256 |
| ODE steps | 32 (both passes) |
| sccfg | 1.0 (consistent with EXP-54b h10 baseline) |
| seed | 42 (pass 1), 1042 (pass 2 noise) |
| MAX_LENGTH | 128 |

## Arms

| Arm | Committed positions | Commit fraction (expected) |
|-----|---------------------|---------------------------|
| standard | none (reference) | 0% |
| staged_left50 | first 50% by index | 50% |
| staged_conf90 | top-1 prob > 0.90 | ~20–40% |
| staged_conf80 | top-1 prob > 0.80 | ~40–60% |
| staged_conf70 | top-1 prob > 0.70 | ~50–70% |

---

## Expected baselines

- kd2 standard sccfg=1: ~299.8 PPL (from EXP-54b seed=42)
- kd_cr standard sccfg=1: TBD (from this experiment's standard arm)

---

## Results

### kd_cr

| Arm | PPL | I | Commit% |
|-----|-----|---|---------|
| standard | 332.06 | 0 | — |
| staged_left50 | 441.13 | **+109.1** | 50% |
| staged_conf90 | 391.85 | +59.8 | 51% |
| staged_conf80 | 393.25 | +61.2 | 59% |
| staged_conf70 | 383.53 | +51.5 | 66% |

### kd2

| Arm | PPL | I | Commit% |
|-----|-----|---|---------|
| standard | 284.72 | 0 | — |
| staged_left50 | 163.21 | **-121.5** | 50% |
| staged_conf90 | 365.79 | +81.1 | 48% |
| staged_conf80 | 381.11 | +96.4 | 56% |
| staged_conf70 | 349.61 | +64.9 | 63% |

---

## Key Findings

1. **kd2 + staged_left50: I=-121.5** — comparable to h10 SC (I=-130, EXP-54b). Large improvement from committing first 50% by position.

2. **kd_cr + staged_left50: I=+109** — strongly degrades. kd_cr's self_cond_proj is working; bypassing it via cond_seq removes useful information. Confirms EXP-44 finding that kd_cr benefits from its SC pathway (opposite of kd2).

3. **Confidence-based arms (conf70/80/90) uniformly degrade both models** — scattered committed positions + fresh noise creates trajectory inconsistency. The model receives x_pred constraints from a different random trajectory than the one it's currently computing.

4. **Root cause of failure**: `all_z0_pass2` uses a different random seed (SEED+1000) from pass 1. With fresh noise, pass 2 generates a completely different trajectory for uncommitted positions, creating conflicts with the committed positions from pass 1's trajectory.

5. **Why left50 works for kd2 but not confidence**: left50 is equivalent to **prefix completion** — fix the first half, generate the second half conditioned on it. This is precisely the task the model sees during training (cond_seq prefix conditioning). Scattered commitment is an out-of-distribution constraint pattern.

6. **Why left50 hurts kd_cr**: kd_cr has well-functioning SC (EXP-44); cond_seq commits positions and sets v=0, bypassing the SC loop for those positions. For kd_cr, this removes useful self-correction. For kd2, cond_seq bypasses the anti-correlated self_cond_proj (EXP-43 B11 conflict), which helps.

---

## Follow-up: EXP-55v2 (same-noise staged)

Run staged_conf arms with SAME z0 for pass 2 (same seed as pass 1). This would test whether the trajectory inconsistency (fresh noise) is the cause of confidence-arm failure, or whether scattered commitment is inherently problematic regardless of noise seed.
