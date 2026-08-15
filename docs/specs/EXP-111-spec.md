# EXP-111 Spec — ELF ODE Overlapping Two-Wave Unlock

**Status:** ACTIVE / SMOKE PASSED; TWO BANKS RUNNING
**Motivation:** EXP-108 finds replicated `.40/.45` timing headroom, while
EXP-110 shows that one sequence-level trigger choice is not predictably
transferable.

## Question

Can token-level asynchronous anchoring remove the brittle sequence-level
choice by letting already-mature positions lead and newly mature positions
join one checkpoint later?

This experiment keeps one global deterministic ODE clock. It does not feed
mixed-time continuous states to the network. Only temporary predicted-clean
lexical conditions have position-specific activation and expiry times.

## Primary intervention

At the first native checkpoint at or after `.40`, select

\[
A_1=\{j:\max_v p_\theta(v\mid \hat x_j)\ge .90\}.
\]

At the first checkpoint at or after `.45`, select only newly eligible free
positions

\[
A_2=\{j\notin A_1:\max_v p_\theta(v\mid \hat x_j)\ge .90\}.
\]

Each wave remains active for exactly four native solver intervals. Thus
\(A_1\) is released earlier while \(A_2\) remains as temporary context, after
which every position returns to global joint refinement. Prompt positions are
never modified.

The method is called **Two-Wave Unlock**. It is asynchronous at the lexical
conditioning layer, not at the flow-state clock.

## Frozen arms

1. Standard ODE-32;
2. fixed Unlock-4 at `.40`;
3. fixed Unlock-4 at `.45`;
4. fixed `.40` plus a `.45` readout whose result is ignored (readout sham);
5. **Two-Wave-New:** `.40` anchors plus only newly eligible `.45` anchors;
6. **Two-Wave-Refresh:** at `.45`, refresh every currently confident token and
   restart its four-interval expiry.

Arm 5 is primary. Arm 6 separates genuine staggered recruitment from merely
holding/refreshing the first anchor set longer. The sham must reproduce fixed
`.40` exactly before any method result is accepted.

All arms use the same prefix and initial latent. Report actual grid event and
release times, first/second-wave density, overlap, final revision, 32 denoiser
calls, and one versus two decoder readouts.

## Banks

- discovery: seed 789 / OWT offset 43000, `n=64`;
- validation: seed 1011 / OWT offset 44000, `n=64`;
- only if both pass: final U/C panel, seed 2026 / OWT offset 45000.

The first two stages use real 64-token OWT prefixes and 64-token continuations.
The former unopened EXP-110 final bank becomes EXP-111 discovery and is not
called held out after method design.

## Gates

Two-Wave-New must beat fixed `.40` on both discovery and validation:

1. C-PPL improves by at least 2%;
2. paired mean-NLL bootstrap CI upper bound is below zero;
3. D1 delta is at least `-.005`;
4. Rep-4 delta is at most `+.005`;
5. degeneration delta is at most `+.015`;
6. prompt-gain delta is at least `-.01`;
7. second-wave density is at least 5%;
8. fixed `.40` and its readout sham have exact text agreement `1.0`.

If only Two-Wave-Refresh passes, report that longer/updated anchoring helps but
do not claim token-level asynchronous recruitment. If neither passes, close
the two-wave inference design and retain fixed Unlock-4.

## Final panel

After a two-bank pass, freeze the arm and evaluate unconditional plus real
prefix conditional generation on seed 2026. Include Standard ODE-32, fixed
`.40`, fixed `.45`, the selected two-wave arm, the full quality panel, paired
NLL CIs, and exact compute/readout accounting. No SDE or Plaid prerequisite is
part of this ELF ODE discovery protocol.

Implementation:
`models/ELF-torch/experiments/probe_elf/two_wave_unlock_exp111.py`.
