# EXP-56b Spec: Within-ODE Progressive Commitment — Commit Timing Sweep

**Date:** 2026-07-31 (baseline arm added 2026-08-03)
**Status:** DONE
**Script:** `models/ELF-torch/experiments/probe_elf/progressive_commit_exp56b.py`

---

## Motivation

EXP-56 fixed commit time at t=0.5 (motivated by EXP-42 CKA bifurcation and EXP-54c gate).
This experiment sweeps commit time t ∈ {0.3, 0.4, 0.5, 0.6, 0.7} with threshold=0.7 fixed.

Key question: is t=0.5 optimal, or can early commitment help more?

---

## Results

### kd_cr

| Arm | PPL | I | Commit% |
|-----|-----|---|---------|
| standard | 332.06 | 0 | — |
| prog_t30_c70 | 158.61 | **-173.5** | 37% |
| prog_t40_c70 | 156.94 | **-175.1** ← best | 56% |
| prog_t50_c70 | 168.02 | -164.0 | 70% |
| prog_t60_c70 | 227.14 | -104.9 | 74% |
| prog_t70_c70 | 248.95 | -83.1 | 73% |

### kd2

| Arm | PPL | I | Commit% |
|-----|-----|---|---------|
| standard | 284.72 | 0 | — |
| prog_t30_c70 | **98.34** | **-186.4** ← BEST EVER | 46% |
| prog_t40_c70 | 177.39 | -107.3 | 53% |
| prog_t50_c70 | 212.99 | -71.7 | 61% |
| prog_t60_c70 | 239.42 | -45.3 | 64% |
| prog_t70_c70 | 246.12 | -38.6 | 63% |

### baseline (2026-08-03)

N=256, seed=42, sccfg=1, GPU 0. Script run with `--ckpt baseline`.

| Arm | PPL | I | Commit% |
|-----|-----|---|---------|
| standard | 122.11 | 0 | — |
| prog_t30_c70 | 180.80 | **+58.69** ⛔ | 72.1% |
| prog_t40_c70 | 117.34 | −4.77 | 89.0% |
| **prog_t50_c70** | **110.84** | **−11.26** ← best | **95.3%** |
| prog_t60_c70 | 116.21 | −5.90 | 97.9% |
| prog_t70_c70 | 119.68 | −2.43 | 98.3% |

**progressive commitment works on baseline** (I=−11.26, 9.2% PPL reduction), but with key differences:
- **Optimal t=0.50** (vs kd2 t=0.30, kd_cr t=0.40) — baseline needs more denoising before commit
- **t=0.30 is catastrophic** (+58.69): baseline G_oracle(t=0.3)≈63%, but decode branch commits 72.1%
  of positions — many wrong high-confidence predictions locked in
- Benefit is modest compared to KD models because baseline decode branch is less reliable early

---

## Key Findings

### 1. Earlier commitment is dramatically better (KD models)

Both models peak at t=0.3–0.4, NOT t=0.5. Monotonic DECREASE in benefit as commit time gets later:
- kd2: t=0.3 → I=−186, t=0.4 → I=−107, t=0.5 → I=−72, t=0.6 → I=−45, t=0.7 → I=−39
- kd_cr: t=0.4 → I=−175, t=0.5 → I=−164, t=0.6 → I=−105, t=0.7 → I=−83

### 2. kd2 + prog_t30_c70: **I=−186, PPL=98** — BEST result in the entire experiment series

This is by far the best single-metric improvement achieved:
- vs standard ODE-32 kd2 (sccfg=1): PPL 285→98 (65% reduction)
- vs h10 SC (I=−130, EXP-54b): this is I=−186, 43% more improvement
- Comparable to or better than 16-step SDE (EXP-37a kd2 ~128 PPL)

### 3. Decode-branch confidence enables early reliable commitment

EXP-54c showed that applying h10 SC at t<0.5 is catastrophic (+1084 PPL). But progressive
commitment at t=0.3 is excellent. The difference:
- h10 SC: uses h10 (backbone intermediate) which is unreliable before B11 reorganization (t=0.5)
- Commitment: uses DECODE-BRANCH top-1 probability — the decode branch is reliable much earlier
  (consistent with EXP-01v3: kd2 G_reverse crosses at t≈0.184; kd_cr at t≈0.213)

At t=0.3, 46% of kd2 positions already have decode-branch confidence >70%. These are the
positions that EXP-16v2 identified as having early stable commit timing (T_stable(K=3) < 0.3).

### 4. Commit fraction decreases with earlier commit time

| t | kd2 commit% | kd_cr commit% |
|---|-------------|---------------|
| 0.3 | 46% | 37% |
| 0.4 | 53% | 56% |
| 0.5 | 61% | 70% |
| 0.6 | 64% | 74% |
| 0.7 | 63% | 73% |

Fewer positions are committed at t=0.3 but they're MORE reliable → better quality anchors.
More positions committed later → more positions frozen but quality degrades at late steps.

### 5. Optimal commit time tracks model reliability (new finding from baseline arm)

The three models form a clear pattern:

| Model | Optimal t | I (best) | Mechanism |
|-------|-----------|----------|-----------|
| kd2 | 0.30 | −186.4 | Decode branch reliable at t=0.18 (EXP-01v3) |
| kd_cr | 0.40 | −175.1 | Decode branch reliable at t=0.21 (EXP-01v3) |
| baseline | 0.50 | −11.26 | Decode branch reliable at t=0.24 (EXP-01v3); G_oracle(0.3)=63% |

Committing too early for the model's reliability → locking wrong positions (t=0.30 on baseline: +58.69).
The benefit is proportional to decode-branch early reliability (KD dramatically improves this).

### 6. kd2 benefits more from early commitment than kd_cr

kd2 t=0.3: I=−186 vs kd_cr t=0.4: I=−175 vs baseline t=0.5: I=−11.26. kd2's decode branch is earlier-reliable (EXP-01v3:
crosses at t=0.184 vs kd_cr at t=0.213). The B11 reorganization in kd2 from KD makes positions
commit earlier via the decode path, enabling earlier commitment.

---

## Mechanism Interpretation

The decode-branch confidence gate at t=0.3 identifies positions that have ALREADY committed in
the EXP-01v3/EXP-16v2 sense (oracle top-1 stable). By locking these into cond_seq, the second
half of the ODE trajectory (t=0.3 to t=1.0, 70% of steps!) refines the remaining positions
with stable anchors.

This is effectively **causal bootstrapping within the reverse ODE** — early-committing positions
(function words, high-frequency tokens, EXP-08v2) anchor the trajectory for late-committing
content positions.

---

## Follow-up

- **EXP-56c**: Lower threshold sweep (0.5, 0.6) at t=0.3 — can we push kd2 below 98 PPL?
- **EXP-57**: Stack h10 SC + prog_t30_c70 (DONE: anti-synergistic, h10 SC reduces commit fraction)
- **EXP-56d**: Adaptive threshold (commit more aggressively as t increases, e.g., lower threshold
  at later t)
- **Baseline multi-seed**: current baseline arm is single seed=42 (N=256); modest −11.26 may vary
  across seeds, but direction is established
