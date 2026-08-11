# EXP-82 Spec — Transition-Calibrated Unlock Pareto

**Status:** READY / P0 SCREEN

## Question

Does Unlock-4 work because a reliable minority becomes useful context for the
unresolved set, or only after almost the whole sequence is projected into a
predicted-clean state?

## Design

Use ELF base, deterministic ODE-32, length 128, paired native noise, and paired
unconditional/native-prefix conditional panels. Select anchors by an exact
eligible-position budget rather than a fixed confidence threshold:

```text
A_k = TopConf_q(x_hat_k),  q in {.25, .50, .75, .875}.
```

Cross a bounded set of trigger checkpoints around the measured transition
with temporary lock durations `H in {1, 4, 8}`. The first screen uses the
following non-redundant cells:

```text
t = {.30, .40, .50}, H = 4, q = {.25, .50, .75, .875};
t = .40, q = {.50, .75}, H = {1, 8}.
```

Controls are Standard-32, a lexical-readout-only null, same-density random
positions with position-correct content, and same-position shuffled anchor
content. Conditional selection excludes the observed prefix.

## Metrics and gate

Report the complete EXP-80 U/C panel, exact latent clamp, anchor density,
calls, and latency. On the unselected set additionally report paired changes
in `tau_first`, `tau_stable`, revisions, and branch-own endpoint margin.

Promote only cells that improve C-PPL or boundary prompt gain over Standard-32
while keeping `D1`, Rep-4, and degeneration within a pre-registered tolerance.
A gain requiring `q=.875` supports mass projection, not sparse coordination.

Planned runner:
`models/ELF-torch/experiments/probe_elf/transition_unlock_pareto_exp82.py`.
