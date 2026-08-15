# EXP-112 Spec — Frozen Two-Wave Unlock Confirmation

**Status:** ACTIVE / CONDITIONAL N=128 RUNNING
**Depends on:** EXP-111's same-direction two-bank result and significant
pooled paired-NLL effect

## Purpose

Confirm or reject the frozen Two-Wave-New intervention at adequate sample
size. This is not another search. The method remains exactly:

- deterministic ELF baseline ODE-32;
- first confidence-`.90` Unlock wave at requested `.40`;
- second wave at requested `.45`, restricted to positions never selected by
  wave one;
- each selected position expires after four native solver intervals;
- 64-token real OWT prefix plus 64-token continuation.

No time, confidence, duration, or recruitment rule may change after opening
this bank.

## Confirmation bank

- seed 2026 / OWT offset 45000;
- `n=128` paired trajectories;
- primary comparison: Two-Wave-New versus fixed Unlock-4 at `.40`;
- controls: Standard ODE-32, fixed `.45`, ignored-readout sham, and
  Two-Wave-Refresh.

## Gates

Promote to an unconditional confirmation only if:

1. C-PPL improves by at least 2%;
2. paired mean-NLL bootstrap CI upper bound is below zero;
3. D1 delta is at least `-.005`;
4. Rep-4 delta is at most `+.005`;
5. prompt-gain delta is at least `-.01`;
6. degeneration increases by at most `2/128=.015625` and its paired change is
   not statistically significant;
7. second-wave density is at least 5%;
8. native fixed `.40` and the readout sham retain exact agreement `1.0`.

The degeneration rule is fixed before this bank. It makes the intended
roughly-1.5-percentage-point tolerance compatible with discrete `n=128`
counts; it is not used to relabel EXP-111.

## Conditional next stage

Only after every gate passes, run a paired `n=128` unconditional panel with
Standard ODE-32, fixed `.40`, and frozen Two-Wave-New using the same seed.
Report the complete quality panel and compute/readout counts. If either the
conditional confirmation or unconditional direction fails, retain the result
as an ODE conditional method boundary rather than broadening to SDE/Plaid.
