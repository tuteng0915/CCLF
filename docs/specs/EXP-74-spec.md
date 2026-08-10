# EXP-74 Spec — Event-Triggered Reversible Anchoring

**Status:** DONE / HARD-ONLY POSITIVE (screen, seed 42)
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
8. Persistent hard-anchor references at `t=.30` and `t=.40`.
9. Persistent high-confidence (`>.90`) and two-readout stable-token anchors,
   separating anchor reliability/density from trigger time.

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

## Result (2026-08-10)

The short-lived soft-memory arms fail to improve Standard-32, while every
persistent hard-anchor arm has a favorable sign across all three checkpoints.

| Checkpoint | Standard | Soft stable | Hard `.60` at `.40` | Hard `.90` at `.40` | Hard stable |
|---|---:|---:|---:|---:|---:|
| ELF base | 278.7 | 281.7 | 206.8 | **205.3** | 232.8 |
| Control | 276.4 | 279.9 | 221.3 | **215.8** | 243.4 |
| Early-KD | 199.8 | 203.7 | **168.0** | 169.3 | 181.3 |

Entries are PPL for paired `n=64` generations. The `.90` arm anchors
`87--88%` of positions; the two-readout stable arm anchors `60--64%` and still
improves every checkpoint. Shuffled soft content is harmful (`435.4/374.5/
275.8`), retaining the correct-content mechanism control. Degeneration does
not systematically increase in the best hard arms.

Decision: this is a hard-only positive. The promising method is a single
post-transition conversion of reliable positions into persistent conditions,
not repeated soft self-conditioning. It remains below ODE-64 quality and needs
multi-seed/conditioned/native-SDE fidelity before promotion.
