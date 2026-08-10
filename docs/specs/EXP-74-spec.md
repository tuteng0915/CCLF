# EXP-74 Spec — Event-Triggered Reversible Anchoring

**Status:** ACTIVE / P0
**Model:** ELF base, continued-training Control, corrected Early-KD
**Solver:** deterministic ODE-32, native noise 2, SC-CFG 3
**Purpose:** test whether the positive one-shot causal anchor effect can improve
sampling when applied sparsely, without permanent freezing or a second forward
at every solver step.

## Hypothesis

EXP-67 shows that position-correct anchors stabilize unresolved positions,
whereas shuffled content has the opposite effect. EXP-71 shows that refreshing
a large leader set at every step is not compute-competitive. The missing
ingredient may be sparse intervention at the collective transition rather than
continuous leadership.

At trigger step `k`, reuse the current predicted-clean state and update only
the self-conditioning memory:

```text
x_sc[k,i] = (1-alpha_i) x_prev[k,i] + alpha_i stopgrad(x_hat[k,i]).
```

No latent is replaced, no token is discretized, and the intervention expires
after `H` steps unless the position passes a second stability check.

## Screen arms

Use paired initial noise, length 128, `n=64`, seed 42:

1. Standard ODE-32.
2. Standard ODE-64 compute/quality reference.
3. One trigger at `t=.30`, confidence-selected anchors.
4. One trigger at `t=.40`.
5. Two triggers at `t={.30,.40}`; the second can revise the first.
6. Stability trigger: token identity must agree at two adjacent readouts and
   confidence exceed `.60`.
7. Matched shuffled-content control for the best trigger arm.
8. Persistent hard-anchor reference using the same selected positions.

Primary soft arms keep the intervention for `H=4` solver intervals with
`alpha=.5`; include `H=1` and `alpha=1` only as narrow ablations after the
screen. Count every lexical readout separately from denoiser calls.

## Statistics and metrics

At each trigger report anchor fraction, confidence, same-token stability, and
the immediate change on unselected positions in entropy and endpoint margin.
Final metrics are PPL, D1/D2, Rep-4, degeneration, words, MaxShare,
UniqueRatio, unigram collapse, `tau_first`, `tau_stable`, revisions, denoiser
calls, readout calls, and latency.

## Decision rule

- **Positive:** a sparse reversible arm improves Standard-32 on coherence and
  stable timing, beats its shuffled control, and is competitive with ODE-64
  after accounting for calls.
- **Mechanism-only:** correct content beats shuffled content and improves
  unresolved-position timing, but final quality does not improve.
- **Hard-only:** persistent anchoring wins while soft expiry fails; develop a
  revisable discrete/continuous hybrid rather than another soft schedule.
- **Negative:** no sparse arm improves Standard-32. The EXP-67 intervention is
  explanatory but not a sampler component.

