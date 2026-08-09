# EXP-72 Spec — Native Multi-Time ELF v2

**Status:** DONE / STOPPED AT STEP 500 (clock-learning gate failed)
**Launch condition:** EXP-70 shows that correct local clocks materially repair
the current Pipeline, or EXP-71 supports a directional leader effect that is
worth expressing through a learned clock.  
**Start checkpoint:** healthy ELF base; corrected Early-KD is a secondary
replication only after the base interaction passes.  
**Planned code:** `models/ELF-torch/src/modules/model.py`, `src/train_step.py`,
and `experiments/probe_elf/eval_multitime_exp72.py`

## Why EXP-60 was not a definitive architecture test

EXP-60 inserted one local-time residual before the transformer and multiplied
it by a zero-initialized scalar gate:

```text
h_i <- h_i + tanh(g) [e(tau_i) - e(mean tau)].
```

After 500 steps the EMA gate remained approximately zero. Half of examples
were synchronous and heterogeneous examples mixed LTR, RTL, and random
orders, so opposing local-time signals could cancel through a single gate.
EXP-72 first requires proof that the model actually uses local time.

## Architecture

Retain the global mean-time prefix for sequence-level progress. Inject local
time into every transformer block using independent learned projections:

```text
r_i = e_local(tau_i) - e_local(mean_j tau_j)
h_i^(ell) <- h_i^(ell) + alpha_ell P_ell r_i.
```

Requirements:

- `P_ell` is initialized with small nonzero variance;
- `alpha_ell` is initialized to a small nonzero value, not exactly zero;
- log one `alpha_ell` and local-time gradient norm per layer;
- optional attention bias `b_ij = MLP(tau_i - tau_j)` is an ablation, not part
  of the first run;
- scalar-time inputs and `Delta=0` must reproduce the architecture-matched
  synchronous control within numerical tolerance.

## Training arms

All arms start from identical weights and use identical data order, optimizer,
and 2,000-step budget.

1. **Architecture control:** per-token architecture, synchronous times only.
2. **LTR curriculum:** LTR heterogeneous examples only; probability rises
   `.10 -> .50`, and wave width rises `0 -> .10` during the first 1,000 steps.
3. **Order-mixture follow-up:** only after arm 2 learns a nontrivial local-time
   response; add 25% RTL and 25% fixed random heterogeneous examples.

Do not begin with a direction mixture. Decoder rows remain at time one and do
not count as heterogeneous denoiser training examples.

Use the smooth endpoint-preserving clock with `s_r=.875` for ODE-32:

```text
e(s) = sin(pi * min(s/s_r, 1))
tau_i(s) = clip(s + Delta e(s) (1 - 2q_i), 0, 1).
```

The offset is exactly zero from `s_r` onward, giving a genuine final four-step
synchronous refinement region. Require `Delta <= s_r/pi` for monotonic clocks.

## Mandatory learning gates

Evaluate at steps 0, 100, 500, 1,000, and 2,000:

1. local-time parameter and gradient norms by layer;
2. output sensitivity after perturbing one position's local time;
3. shuffled-clock sensitivity at fixed heterogeneous latent;
4. `Delta=0` identity against the matched control;
5. denoising L2 stratified by leading and trailing quartile.

Define

```text
S_tau = mean_i ||v_theta(z,tau+delta_i)-v_theta(z,tau)|| / ||delta_i||.
```

Stop early if `S_tau` and all local-time parameter updates remain numerically
negligible through step 500. A completed training run with an unused clock is
an implementation failure, not a negative result about wavefront denoising.

## Evaluation

Paired samplers on each training arm:

- standard ODE-32;
- LTR `Delta in {.05,.10,.15}` plus four synchronous refinements;
- RTL `.10` as the direction control;
- fixed random `.10` as the symmetry-breaking control.

Screen at length 128, `n=64`. Promote only if the causal interaction

```text
[LTR-trained: LTR sampler - standard]
 - [sync-control: LTR sampler - standard]
```

is favorable while the trained model's standard sampler remains healthy.
Promotion requires three seeds, length 1024, conditioned generation, and the
complete quality/timing panel. Native SDE is a separate fidelity stage and is
never inferred from ODE performance.

## Decision rule

- **Architecture success:** local-time sensitivity is nonzero and the LTR
  interaction improves quality and stable commitment without earlier noisy
  first guesses.
- **Clock learned but dynamics fail:** proceed once to on-policy trajectory
  training (EXP-73).
- **Clock not learned:** redesign the injection; do not increase training time
  blindly.
- **Synchronous quality loss:** reject the architecture regardless of its LTR
  numbers.

## Result (2026-08-10)

Both matched arms completed 500 steps. The architecture preserved synchronous
quality (`277.1` Control, `279.9` LTR-trained), and the live LTR-training scale
received nonzero gradients. The EMA checkpoint nevertheless remained almost
at initialization: mean local scale `0.01000` for Control and `0.01003` for
LTR-trained. More importantly, the functional clock diagnostics were
indistinguishable:

| Training | `S_tau` LTR | `S_tau` RTL | LTR/RTL velocity cosine |
|---|---:|---:|---:|
| Sync Control | 101.887 | 101.981 | 1.0000 |
| LTR curriculum | 101.881 | 101.977 | 1.0000 |

The sampler interaction is unfavorable. At `Delta=.10`, LTR costs `+33.3`
PPL over standard for Control and `+53.8` for LTR-trained, an interaction of
`+20.5` PPL. Increasing `Delta` worsens the LTR-trained arm to `343.7`; its RTL
control is instead `274.3`. This is an unused/local-clock architecture failure,
not evidence that a learned wave has exposure bias. Stop at 500 steps and do
not promote to 2,000 steps. Raw results are under
`results/exp72_multitime_v2/`.
