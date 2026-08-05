# EXP-55v2 Spec: Staged Decoding — Same-Noise Pass 2

**Date:** 2026-07-31
**Status:** DONE
**Script:** `models/ELF-torch/experiments/probe_elf/staged_decoding_exp55v2.py`

---

## Motivation

EXP-55 tested two-pass staged decoding with FRESH noise (z0_pass2 from seed+1000). All confidence-based
arms failed for both kd_cr and kd2; only staged_left50 worked for kd2 (I=-121). The hypothesis was
that fresh z0 in pass 2 creates a trajectory mismatch: the model commits positions from pass-1 x_pred,
then runs a completely different trajectory in pass 2.

EXP-55v2 tests the same setup but uses the SAME z0 for both passes — directly testing whether
trajectory inconsistency caused the confidence-arm failures.

---

## Results

### kd_cr (same noise)

| Arm | PPL | I | Commit% | EXP-55 fresh-noise I |
|-----|-----|---|---------|---------------------|
| standard | 332.06 | 0 | — | 0 |
| staged_left50 | 410.63 | **+78.6** | 50% | +109.5 |
| staged_conf90 | 390.76 | +58.7 | 51% | +44.3 |
| staged_conf80 | 403.81 | +71.7 | 59% | +73.8 |
| staged_conf70 | 391.68 | +59.6 | 66% | +67.1 |

### kd2 (same noise)

| Arm | PPL | I | Commit% | EXP-55 fresh-noise I |
|-----|-----|---|---------|---------------------|
| standard | 284.72 | 0 | — | 0 |
| staged_left50 | 166.80 | **-117.9** | 50% | **-121.0** |
| staged_conf90 | 291.13 | +6.4 | 48% | +13.2 |
| staged_conf80 | 322.73 | +38.0 | 56% | +55.3 |
| staged_conf70 | 325.71 | +41.0 | 63% | +74.4 |

---

## Key Findings

### 1. Same-noise barely changes kd2 left50 (I=-118 vs -121)

The left50 result is essentially identical between fresh and same noise. For prefix completion, the
constraint (left 50 tokens fixed) is strong enough that the second ODE trajectory converges similarly
regardless of z0. This confirms left50 is an in-distribution task for kd2.

### 2. Confidence arms STILL fail with same noise

The trajectory-mismatch hypothesis is **refuted**. Same z0 does NOT fix confidence-based staged decoding:
- kd2 conf70 same-noise: I=+41 (vs fresh-noise: +74) — slightly less bad, but still catastrophically bad
- kd_cr conf70 same-noise: I=+60 (vs fresh-noise: +67) — marginally different

The failure is not due to different z0. The fundamental problem is something else.

### 3. What actually causes confidence-arm failure in staged decoding?

Most likely: **information bottleneck**. The decoder selects positions at t=1 (end of pass 1), where
x_pred is a mix of committed and uncommitted positions. Committed positions from pass 1 create
constraints that force the remaining positions to fit around them. But:
- The committed positions were selected based on pass-1 trajectory context
- In pass 2, uncommitted positions must be generated conditioned on those anchors
- The model was never trained to do this conditional generation task
- Confidence ≠ correctness in the OOD setting of staged decoding

Compare to EXP-56 (progressive commitment within a SINGLE ODE): commitment happens mid-trajectory
at t=0.3-0.5. The remaining ODE steps (which account for the majority of the trajectory) are run
with the committed positions as anchors from within the same trajectory. The model has never
been trained for "two-pass ODE with anchor injection" but HAS essentially been trained with
cond_seq conditioning (via the restore_vx mechanism in flow matching with cond_mask).

### 4. kd_cr left50 hurts (+79) in both EXP-55 and EXP-55v2

kd_cr's pass-2 PPL jumps to 410+ even with left-prefix constraint. kd_cr's left-half tokens are
evidently not as constraining for the right half as kd2's are. kd2's tighter semantic embedding
space means left-prefix anchors propagate effectively; kd_cr's looser space does not.

---

## Conclusion

Staged decoding is fundamentally inferior to within-ODE progressive commitment (EXP-56):
- EXP-56 kd_cr optimal: I=-175 (vs staged best: I=-121 kd2 only)
- EXP-56b kd2 t=0.3: I=-186 (vs staged kd2 left50: I=-118)

The within-ODE mechanism has access to the mid-trajectory state and does not require the model
to complete a novel conditional generation task. Staged decoding should be abandoned in favor
of progressive commitment.
