# EXP-112 Spec — Frozen Two-Wave Unlock Confirmation

**Status:** DONE / CONDITIONAL POSITIVE; UNCONDITIONAL INCONCLUSIVE
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

## Conditional confirmation result

All gates pass on seed 2026 / offset 45000:

| arm | C-PPL | D1 | D2 | Rep-4 | degeneration | prompt gain | wave-2 density |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed `.40` | 412.73 | .49763 | .89682 | .02271 | .015625 | .24928 | .0000 |
| Two-Wave-New | **401.58** | .49744 | .89588 | .02361 | .015625 | .25445 | .0780 |

C-PPL improves `2.70%`; paired mean NLL is `-.0264
[-.0521,-.00078]`. D1 changes `-.00019`, Rep-4 `+.00089`, degeneration
`.00000`, and prompt gain `+.00517`. Native fixed `.40` and readout sham both
retain agreement `1.0`. Two-Wave-Refresh improves only `.93%` and its CI
crosses zero. The preregistered unconditional confirmation is therefore
authorized with no method changes.

Conditional runner:
`models/ELF-torch/experiments/probe_elf/two_wave_unlock_exp111.py`.

Unconditional runner:
`models/ELF-torch/experiments/probe_elf/two_wave_unlock_unconditional_exp112.py`.

## Unconditional result and final decision

| arm | U-PPL | D1 | D2 | Rep-4 | degeneration | wave-2 density | revision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard ODE-32 | 276.30 | .46728 | .88547 | .00920 | .03906 | .0000 | .0000 |
| fixed `.40` | 208.71 | .44801 | .87687 | .00993 | .03906 | .0000 | .0973 |
| Two-Wave-New | **204.75** | .44633 | .87627 | .01072 | .05469 | .0682 | .1100 |

Two-Wave-New retains a favorable U-PPL direction (`1.90%`), but misses the
frozen `2%` threshold and its paired mean-NLL CI crosses zero: `-.0175
[-.0371,.00134]`. D1 changes `-.00169`, D2 `-.00060`, Rep-4 `+.00079`, and
degeneration `+2/128=.015625`. The readout sham remains exactly identical.
The two additional harmful discordances and zero improving discordances give
an exact paired McNemar/binomial `p=.50`, so the preregistered degeneration
non-significance condition is satisfied even though the overall method gate is
not.

The final claim is therefore conditional, not universal: token-level
overlapping Unlock is a confirmed improvement for real-prefix ELF ODE
continuation, while unconditional evidence is favorable but inconclusive. No
SDE or Plaid run is required to define this ELF boundary.
