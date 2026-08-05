# EXP-56 Spec: Within-ODE Progressive Commitment (Strategy C)

**Date:** 2026-07-31
**Status:** Running
**Script:** `models/ELF-torch/experiments/probe_elf/progressive_commit_exp56.py`

---

## Motivation

Strategy B (EXP-55) uses two separate ODE passes. Strategy C achieves progressive
commitment **within a single ODE pass**, at the same FLOPs as standard ODE-32
(plus one decode-branch forward at the commit step, ≈3% overhead).

The commit timing is motivated by:
- **EXP-42 CKA**: B08–B11 representations diverge sharply at t=0.5 (CKA 0.896→0.427)
- **EXP-54c**: SC_T_MIN=0.5 gate is essential; the model can be reliably corrected
  only at t≥0.5; below that, x_pred is too noisy to use as SC

At t_next=0.5 (midpoint of 32-step trajectory = step 16), x_pred represents the
model's best estimate of x0 based on the first half of the trajectory.
Committing high-confidence positions at this point lets the second half refine
the remaining positions with stable anchors.

---

## Design

**Standard ODE** (t: 0→1, 32 steps) with ONE modification:

At the first ODE step where t_next ≥ 0.5:
1. Compute decode-branch top-1 confidence from current `x_pred`
2. Mark positions with confidence > threshold as committed
3. `cond_seq ← x_pred.detach()` (clean estimate at commit time)
4. `cond_mask ← committed.float()`
5. `z[committed] ← x_pred[committed]` (jump z to clean estimate)

For all subsequent steps: `v=0`, `x_pred=cond_seq` at committed positions
(standard restore_vx behavior in `_ode_step`).

**Key difference from EXP-31v2 DF**: DF only freezes z (still computes v normally);
here cond_mask also pins x_pred and sets v=0, creating a stronger/cleaner commitment.

---

## Parameters

| Parameter | Value |
|-----------|-------|
| Models | kd_cr, kd2 |
| N sequences | 256 |
| ODE steps | 32 |
| sccfg | 1.0 |
| seed | 42 |
| Commit threshold t | 0.5 (first t_next ≥ 0.5 = step 16/32) |
| MAX_LENGTH | 128 |

## Arms

| Arm | Threshold | Expected commit% |
|-----|-----------|-----------------|
| standard | none | 0% |
| prog_t05_c90 | 0.90 | ~20–40% |
| prog_t05_c80 | 0.80 | ~40–60% |
| prog_t05_c70 | 0.70 | ~50–70% |

---

## Results

### kd_cr

| Arm | PPL | I | Commit% |
|-----|-----|---|---------|
| standard | 332.06 | 0 | — |
| prog_t05_c90 | 187.14 | **-144.9** | 57% |
| prog_t05_c80 | 175.19 | **-156.9** | 64% |
| prog_t05_c70 | 168.02 | **-164.0** | 70% |

### kd2

| Arm | PPL | I | Commit% |
|-----|-----|---|---------|
| standard | 284.72 | 0 | — |
| prog_t05_c90 | 226.92 | **-57.8** | 50% |
| prog_t05_c80 | 218.90 | **-65.8** | 56% |
| prog_t05_c70 | 212.99 | **-71.7** | 61% |

---

## Key Findings

1. **Both models benefit significantly** — EXP-56 is the first intervention that helps kd_cr AND kd2 simultaneously.

2. **kd_cr dominates**: prog_t05_c70 → PPL 332→168, **I=-164**. This EXCEEDS h10 SC on kd2 (I=-130, EXP-54b). kd_cr + EXP-56 is currently the best single-step inference improvement.

3. **kd2 also improves**: prog_t05_c70 → PPL 285→213, **I=-72**. Smaller than h10 SC (-130) because h10 SC directly addresses the B11 anti-correlation issue in kd2's SC pathway, which is a more targeted fix.

4. **Monotonic improvement with lower threshold**: more positions committed → better PPL for both models. This suggests even more aggressive commitment (threshold 0.5-0.6) may further help.

5. **Why kd_cr benefits MORE than kd2**: At t=0.5, kd_cr's backbone (improved by KD, EXP-39/42) produces high-quality x_pred. Committing these positions pins x_pred to clean values, providing strong anchors for the second half. kd_cr's good backbone makes the committed positions very reliable. For kd2, x_pred at t=0.5 is less reliable (B11 anti-correlation degrades SC quality), so fewer committed positions actually help in the second half.

6. **Mechanism vs EXP-55**: No fresh noise = committed positions are consistent with the trajectory. The uncommitted positions in the second half [t=0.5, t=1.0] evolve with committed positions as stable cond_seq anchors. This is the "within-trajectory progressive commitment" that EXP-31v2 (diffusion forcing) approximated but couldn't cleanly implement.

7. **Commit fraction ~50-70%**: At t=0.5 with ODE-32, roughly half or more of positions are already committed at >70% confidence. This is consistent with EXP-16v2 (kd_cr: 99.5% stable by t=0.5) and EXP-14v2 (stable commitment is high by midpoint).

---

## Comparison with EXP-55 (Strategy B)

| | EXP-55 best | EXP-56 best |
|--|-----------|-----------|
| kd_cr | staged_left50: **+109** (hurts) | prog_t05_c70: **-164** |
| kd2 | staged_left50: **-121** | prog_t05_c70: **-72** |
| mechanism | fresh noise = trajectory mismatch | single trajectory = coherent |
| cost | 2× ODE-32 + confidence call | 1× ODE-32 + 1 forward |

**Verdict: EXP-56 (C) is strictly better** for kd_cr; EXP-55 left50 has an edge for kd2, but may be explained by prefix-completion training data (not genuine diffusion forcing).

---

## Follow-up Directions

- **EXP-57**: h10 SC + progressive commitment stacked (kd2 + both interventions; does -130 + -72 = more?)
- **EXP-56b**: sweep commit threshold lower (0.5, 0.6) and commit time later (t=0.7)
- **EXP-55v2**: same noise for pass 2 (test whether trajectory inconsistency explains confidence-arm failure)
