# EXP-70 Spec — Pipeline Clock and Mixed-State Factorization

**Status:** DONE / NEGATIVE (screen, seed 42)
**Model:** ELF base, continued-training Control, corrected Early-KD  
**Solver:** deterministic ODE, native noise scale 2, SC-CFG 3  
**Planned script:** `models/ELF-torch/experiments/probe_elf/pipeline_factorization_exp70.py`  
**Purpose:** determine whether the current Pipeline fails because of its shared
average clock, the heterogeneous sequence state itself, or missing final joint
coordination.

## Question

The current Pipeline assigns each contiguous block its own update counter but
queries a scalar-time model at the average time of all active blocks. For block
`j` on its local update `m`, with `T` blocks,

```text
intended local time:  tau[j,m] = m / T
queried shared time:  t_bar[j,m] = (j + m) / (2T)
clock error:          t_bar - tau = (j - m) / (2T).
```

At `T=16`, the first block receives sixteen updates using model times only in
`[0,.469]`, while the last block uses times in `[.469,.938]`. EXP-70 separates
this target-clock error from the effect of attending to positions at different
denoising stages.

## Fixed protocol

```text
length              = 128
standard solver     = ODE-32, uniform grid
pipeline groups     = 16 contiguous balanced blocks
initial noise       = 2 * epsilon, paired across arms
self-cond CFG       = 3
screen              = 32 trajectories, seed 42
formal promotion    = 128 trajectories x 3 seeds
```

Use `group(i) = floor(i*T/L)` rather than `floor(i/(L//T))`, so non-divisible
sequence lengths do not overload the final group.

## Arms

1. **Synchronous:** ordinary ODE-32.
2. **Pipeline-shared:** the current `pipeline_avg` implementation.
3. **Pipeline-local oracle:** keep exactly the same heterogeneous state. At
   each stage, run one scalar-time forward per active local-time bucket and
   retain only the output for the corresponding block:

   ```text
   z[j] <- z[j] + (1/T) v_theta(z_heterogeneous, tau[j]; positions=j).
   ```

   This arm is deliberately expensive and is a diagnostic oracle, not a method.
4. **Pipeline-local + refine:** reserve the final 4 or 8 intervals for ordinary
   synchronous refinement. With `R` refinement steps, every block first takes
   `T-R` asynchronous updates of width `1/T`, so all blocks meet at
   `tau=(T-R)/T`; the whole sequence then advances synchronously to one in `R`
   steps. This keeps the endpoint meaningful instead of attempting zero-width
   Euler updates after `tau=1`.
5. **Order controls:** run arm 3 with LTR, RTL, and one fixed random block order
   on the screen only.

All pipeline positions must take exactly `T` local updates and finish at
intended local time one. Assert these invariants before generation.

## Vector-field decomposition

At four fill/drain stages and for every active block, evaluate:

```text
v_shared = v_theta(z_heterogeneous, t_bar)[j]
v_local  = v_theta(z_heterogeneous, tau_j)[j]
v_sync   = v_theta(z_standard(tau_j), tau_j)[j].
```

Report:

```text
E_clock = 1 - cos(v_shared, v_local)
E_state = 1 - cos(v_local,  v_sync)
MSE_clock, MSE_state
KL_clock = KL(p_local || p_shared)
```

`E_clock` isolates scalar-clock aliasing. `E_state` estimates the additional
cost of heterogeneous context after the target block is queried at its proper
time. Also record state norm, predicted-clean norm, confidence, entropy, and
self-conditioning norm by block and local step.

## Generation metrics

- Gen.PPL, D1/D2, Rep-4, degeneration, word count, MaxShare, UniqueRatio;
- conditioned suffix PPL, ROUGE-L, and exact prefix preservation on the screen
  if an unconditional arm is not catastrophic;
- `tau_first`, `tau_stable`, and revisions under each branch's own trajectory;
- metrics stratified by block index to detect left/right and boundary artifacts;
- denoiser calls, wall-clock latency, time to first completed block, and total
  sequence latency.

PPL alone cannot promote an arm.

## Decision rule

- **Clock aliasing dominates:** arm 3 substantially repairs arm 2 and
  `E_clock >> E_state`. Proceed to native tokenwise-time modeling (EXP-72).
- **Mixed context dominates:** arm 3 still fails and `E_state` remains large.
  Do not train the current discrete Pipeline; prioritize synchronized soft
  anchors (EXP-71).
- **Final coordination dominates:** arm 3 fails but arm 4 repairs quality.
  Retain a revisable wave plus mandatory global refinement.
- **No direction-specific signal:** LTR, RTL, and random are similar. Frame any
  surviving effect as symmetry breaking, not linguistic directionality.

EXP-70 supersedes parameter sweeps of the existing `pipeline_avg`. Do not sweep
`T`, length, or schedules until this factorization identifies a recoverable
operator.

## Result (2026-08-10)

The screen closed the discrete heterogeneous-state Pipeline. Correcting the
queried clock did not repair generation: for ELF base, PPL changed from
`1570.4` (shared clock) to `1778.1` (true local clock), versus `296.5` for
synchronous ODE-32. Eight final synchronous refinements helped only to
`1506.1`. The same failure replicated for Control (`272.9 -> 1469.1`) and
Early-KD (`209.7 -> 1153.2`), where each pair is synchronous versus the best
refined local Pipeline arm.

The decomposition identifies heterogeneous context as the larger error:

| Checkpoint | `E_clock` | `E_state` | `E_x_clock` | `E_x_state` | `KL_clock` |
|---|---:|---:|---:|---:|---:|
| ELF base | .0497 | .1998 | .1584 | .4526 | 8.237 |
| Control | .0588 | .1968 | .1737 | .4632 | 8.568 |
| Early-KD | .0585 | .1812 | .1717 | .4145 | 7.249 |

LTR is structured relative to RTL/random, but remains catastrophic; this is
not a positive linguistic-direction result. Do not promote or sweep this
operator. Raw results are under `results/exp70_pipeline_factorization/`.
