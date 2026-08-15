# EXP-108 Spec — ELF ODE Unlock-4 Trigger-Time Headroom

**Status:** IMPLEMENTED / BANKS PENDING
**Purpose:** determine whether deterministic ELF trajectories have enough
per-sequence trigger-time variation to justify an adaptive Unlock-4 controller.

## Scope correction

ELF ODE is the primary flow-matching sampler, not a provisional result that
must first survive SDE. EXP-78/80 already establish Unlock-4 as a replicated
ODE method. Its SDE failure is a portability boundary, not a reason to move
method discovery away from ELF.

## Frozen protocol

- model: ELF baseline, deterministic uniform ODE-32;
- task: real 64-token OWT prefix plus 64-token continuation;
- intervention: confidence at least `.90`, temporary hard predicted-clean
  conditioning, release after four solver intervals;
- requested trigger times: `.25,.30,.35,.40,.45,.50,.55,.60`;
- fixed reference: `.40`;
- discovery: seed 42 / OWT offset 40000;
- validation: seed 123 / OWT offset 41000;
- `n=64` paired trajectories per bank.

Every trigger arm uses the same prefix and initial latent. ODE dynamics are
deterministic, so no stochastic future-noise routing is involved. Record each
requested time's actual first grid checkpoint, anchor fraction, complete text
quality panel, true/shuffled-prefix NLL, prompt gain, and per-sequence NLL.

The per-trajectory oracle is

```text
k_star(i) = argmin_k NLL_i(Unlock4 at trigger k).
```

It is a diagnostic upper bound and must never be called a deployable sampler.

## Gates

Proceed to an online response experiment only if oracle best-of-eight:

1. improves fixed `.40` C-PPL by at least 5% on both banks;
2. has paired NLL bootstrap CI upper bound below zero on both banks;
3. does not materially regress D1 (`>=-.005`), Rep-4 (`<=+.005`),
   degeneration (`<=+.015`), or prompt gain (`>=-.01`) relative to fixed
   Unlock-4.

Also report the best aggregate fixed time and winner histogram. If headroom is
large but oracle quality fails, the next selector must explicitly constrain
quality. If either bank lacks likelihood headroom, retain fixed Unlock-4 and
close adaptive timing on ELF.

Implementation:
`models/ELF-torch/experiments/probe_elf/unlock_trigger_headroom_exp108.py`.
