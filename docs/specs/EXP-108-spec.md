# EXP-108 Spec — ELF ODE Unlock-4 Trigger-Time Headroom

**Status:** DONE / REPLICATED HEADROOM; UNRESTRICTED QUALITY GATE FAILED
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

## Results

The unrestricted oracle has large replicated likelihood headroom, but it
mostly chooses the premature `.25/.30` arms and fails the complete quality
gate.

| bank | fixed `.40` C-PPL | best-of-8 C-PPL | gain | paired mean NLL delta [95% CI] | winners `.25/.30/.35/.40/.45/.50/.55/.60` |
|---|---:|---:|---:|---:|---|
| seed 42 / offset 40000 | 391.88 | 255.99 | 34.68% | `-.4375 [-.5311,-.3453]` | `23/21/14/2/4/0/0/0` |
| seed 123 / offset 41000 | 389.88 | 247.58 | 36.50% | `-.4562 [-.5544,-.3683]` | `24/19/6/7/4/1/0/3` |

Relative to fixed `.40`, the unrestricted oracle changes D1 by `-.0195` and
`-.0176`, and degeneration by `+.0156` and `+.0313`. It therefore cannot be
used as evidence for a deployable adaptive sampler.

The preregistered follow-up restricts the action space to triggers at or after
`.40`. This preserves `7.90%` and `6.59%` C-PPL headroom with paired CIs
`[-.1396,-.0468]` and `[-.1133,-.0383]`. The validation bank passes every
quality gate. The discovery bank is also safe on Rep-4, degeneration, and
prompt gain, but its D1 delta is `-.005176`, missing the `-.005` cutoff by
`.000176`; this is deliberately recorded as a gate failure rather than rounded
into a pass.

| bank | late oracle C-PPL | D1 delta | D2 delta | Rep-4 delta | degeneration delta | prompt-gain delta | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| seed 42 | 360.94 | -.00518 | -.00120 | -.00016 | .00000 | +.01498 | fail: marginal D1 |
| seed 123 | 364.17 | -.00338 | -.00049 | -.00035 | .00000 | -.00326 | pass |

Finally, restricting the oracle further to only `.40` versus `.45` passes the
complete gate on both banks:

| bank | binary-oracle C-PPL | gain | paired NLL delta [95% CI] | D1 | D2 | Rep-4 | degeneration | prompt gain | winners `.40/.45` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| seed 42 | 369.70 | 5.66% | `-.0611 [-.0967,-.0311]` | -.00426 | -.00124 | -.00013 | .00000 | +.01426 | `46/18` |
| seed 123 | 368.02 | 5.61% | `-.0607 [-.1001,-.0296]` | -.00218 | +.00045 | -.00039 | .00000 | -.00964 | `44/20` |

Thus the scientific result is not “earlier is better.” Trigger timing is
trajectory-dependent even in deterministic ELF ODE, and a quality-safe target
is the narrow decision between `.40` and `.45`. A learned or causal controller
must be developed on new banks and compared with fixed `.40`; the offline
oracle remains diagnostic only.

Supplementary evaluator:
`models/ELF-torch/experiments/probe_elf/analyze_unlock_late_oracle_exp108.py`.
