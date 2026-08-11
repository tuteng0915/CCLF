# EXP-80 Spec — Paired Conditional Revalidation

**Status:** P0 DONE / P1 ROBUSTNESS REPLICATION RUNNING

## Question

Do the retained ELF interventions behave differently when the model receives a
real observed prefix? Unconditional and conditional generation are separate
estimands; neither is a launch gate for the other.

For a prompt `c` and continuation `y`, report both

```text
q_M(c,y) = q_M(c) q_M(y | c)
```

through paired unconditional generation and fixed-prefix continuation.

## P0 scope

```text
checkpoint = ELF base
length = 128, prefix length = 64
uniform ODE, native noise scale 2, SC-CFG 3
n_unconditional = 64, n_conditional = 64, seed = 42
```

The conditional panel initially uses a deterministic offset into
`embedded-language-flows/openwebtext-t5`. The released dataset exposes only a
train split, so this is labeled **in-domain OWT**, not train-disjoint or
held-out. Gutenberg remains an optional out-of-domain replication.

Every arm reuses paired suffix noise. Observed prefix states remain in ELF's
native `cond_seq` pathway. LTR/RTL/random groups and soft-leader fractions are
defined only over the 64 free suffix positions; fixed prompt positions never
consume a wave group.

## Arms and compute controls

| Arm | Purpose | Denoiser calls |
|---|---|---:|
| Standard-32 | native reference / Unlock-4 control | 32 |
| Standard-64 | compute control for two-forward soft anchors | 64 |
| Standard-136 | compute control for refine-8 block methods | 136 |
| Local-clock + refine 8 | best retained true-local-clock diagnostic | 136 |
| Soft LTR | directional synchronized soft leader | 64 |
| Soft random | symmetry-breaking control | 64 |
| Canonical LTR + refine 8 | predicted-clean heterogeneous context | 136 |
| Unlock-4 | ODE-positive conditional control | 32 + 1 readout |

The smoke may omit Standard-136 to reduce time, but the decisive P0 cannot.

## Required paired metrics

Unconditional:

```text
PPL, D1, D2, Rep-4, degeneration, words,
MaxShare, UniqueRatio, unigram collapse.
```

Conditional suffix:

```text
PPL(y), PPL(y | c), ROUGE-L(y, y*),
PPL(y | shuffled(c)),
prompt_gain = log PPL(y | shuffled(c)) - log PPL(y | c),
D1, D2, Rep-4, degeneration,
decoded-prefix agreement, latent prompt-clamp error.
```

Always report denoiser calls, readout calls, processed-token calls, and wall
time. Prompt-conditioned PPL and prompt gain are primary; ROUGE-L is
supplementary because natural continuation is open-ended.

## Gates and decision

Smoke requires:

1. latent prompt-clamp error exactly zero for every arm;
2. identical denoiser-call counts between paired unconditional/conditional
   scopes;
3. conditional schedules cover only free suffix positions;
4. no NaNs outside deliberately skipped PPL fields.

Promotion requires a method to improve its compute-matched Standard control on
prompt-conditioned PPL or prompt gain without a material ROUGE-L,
repetition, degeneration, or diversity regression. A conditional-only gain is
valid but must be scoped as a continuation method. An unconditional-only gain
cannot support a conditional claim.

Runner:
`models/ELF-torch/experiments/probe_elf/paired_conditional_revalidation_exp80.py`.

## P0 result (2026-08-11)

All smoke and formal gates passed. Conditional schedules operated only on the
64 free suffix positions; every latent prompt-clamp error was zero. The table
reports all primary quality fields from paired `n=64` scopes:

| Arm | U-PPL | U-D1 | U-D2 | U-Rep4 | U-Deg | C-PPL\|prompt | C-PPL\|shuffle | Gain | C-RL | C-D1 | C-D2 | C-Rep4 | C-Deg | Calls+R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard-32 | 278.7 | .4970 | .8879 | .0137 | .0000 | 587.1 | 770.9 | .2724 | .0851 | .5619 | .9224 | .0162 | .0000 | 32+0 |
| Standard-64 | 101.5 | .4438 | .8248 | .0346 | .0469 | 267.4 | 358.9 | .2944 | **.0886** | .5165 | .8779 | .0349 | .0469 | 64+0 |
| Standard-136 | **62.7** | .4160 | .7588 | .0853 | .1406 | **183.5** | **243.6** | .2833 | .0816 | .4965 | .8434 | .0492 | .0938 | 136+0 |
| Unlock-4 | 201.7 | .4857 | .8801 | .0123 | .0000 | 393.0 | 541.7 | **.3211** | .0877 | .5398 | .9077 | .0202 | .0156 | 32+1 |
| Soft LTR | 232.9 | .4840 | .8762 | .0165 | .0156 | 512.3 | 657.3 | .2492 | .0846 | .5448 | .9076 | .0200 | .0000 | 64+0 |
| Soft random | 213.3 | .4837 | .8698 | .0198 | .0000 | 506.0 | 662.0 | .2689 | .0860 | .5524 | .9098 | .0217 | .0000 | 64+0 |
| Local + refine 8 | 1583.6 | .6020 | .9569 | .0014 | .0000 | 1915.5 | 2346.0 | .2027 | .0643 | .6362 | .9609 | .0018 | .0000 | 136+0 |
| Canonical + refine 8 | 1395.6 | .6924 | .9787 | .0000 | .0469 | 764.9 | 849.4 | .1048 | .0520 | .6104 | .9245 | .0033 | .2188 | 136+0 |

Every arm has positive prompt gain, so the generated suffix is more compatible
with its true prompt than with a shuffled prompt. This does not rescue the
asynchronous methods:

- Soft LTR/random lose to Standard-64 by `245/239` conditional PPL and have
  lower ROUGE-L; random again beats LTR, so there is no directional advantage.
- Local-clock and canonical-context lose to Standard-136 by `1732/581`
  conditional PPL. Canonical also degenerates on `21.9%` of samples.
- Standard-136 itself shows that lower PPL from extra ODE calls can accompany
  more repetition/degeneration and lower ROUGE-L, so PPL is not used alone.
- Unlock-4 remains the valid positive control: versus Standard-32 it improves
  U-PPL by `77.1`, conditional PPL by `194.1`, ROUGE-L by `.0026`, and prompt
  gain by `.0486`, for one additional lexical readout.

**Decision:** fixed-prefix conditioning does not change the asynchronous method
ranking. Close the current local-clock, repeated-soft-anchor, and canonical
context inference implementations. Retain Unlock-4 as an ODE-specific,
revisable conditional intervention. Do not launch asynchronous retraining from
this P0; first establish a positive conditional mechanism intervention.

## P1 robustness replication (running, 2026-08-11)

P1 tests only the retained positive signal rather than reopening the failed
pipeline variants. Every cell evaluates paired unconditional and native-prefix
conditional generation with `n_uncond=n_cond=128` and reports the complete P0
metric panel.

| Run | Conditional corpus | Seed / data block | Arms | Server session |
|---|---|---|---|---|
| OWT replication A | in-domain OWT | seed 43 / offset 11000 | Standard-32/64/136, Unlock-4 | `exp80_owt43` |
| OWT replication B | in-domain OWT | seed 44 / offset 12000 | Standard-32/64/136, Unlock-4 | `exp80_owt44` |
| Domain-shift replication | Gutenberg | seed 42 | Standard-32/64/136, Unlock-4 | `exp80_gut42` |

The Gutenberg smoke passed before launch: prompt latent clamp error was zero,
the decoded prefix was exact for all four smoke samples, and paired scopes used
identical denoiser-call counts. P1 supports the Unlock-4 claim only if its sign
against Standard-32 persists in prompt-conditioned PPL and prompt gain without
a material diversity, repetition, degeneration, or ROUGE-L regression.
