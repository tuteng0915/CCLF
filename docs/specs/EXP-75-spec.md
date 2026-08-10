# EXP-75 Spec — Canonical-Context Wave

**Status:** DONE / NEGATIVE DIAGNOSTIC (screen, seed 42)
**Model:** ELF base, continued-training Control, corrected Early-KD
**Purpose:** test whether heterogeneous Pipeline fails because attention mixes
raw states expressed at incompatible noise levels.

## Hypothesis

EXP-70 finds `E_state` roughly three to four times `E_clock`: even a target
block queried at its correct local time fails when its neighbors occupy other
stages. Before building a new attention architecture, canonicalize only the
context visible to the target block.

For target block `g` at local time `tau_g`, construct the diagnostic input

```text
z_input[i] = z[i]       if group(i) == g
             x_hat[i]   otherwise

model_input = concat(z_input, x_hat).
```

The target retains its raw local latent; non-target positions communicate in
predicted-clean coordinates. Retain only the target block's model output.
This is an expensive diagnostic oracle, not the final architecture.

## Screen arms

Use the EXP-70 paired protocol (`L=128`, 16 blocks, native ODE, `n=32`):

1. Standard ODE-32.
2. True-local Pipeline from EXP-70.
3. Canonical-context LTR.
4. Canonical-context LTR plus eight final synchronous refinements.
5. Canonical-context RTL.
6. Canonical context with predicted-clean vectors shuffled across sequences,
   matched by block and local time.

Record block-stratified state norms and the target velocity discrepancy from a
synchronous reference:

```text
E_raw   = 1 - cos(v_raw_hetero, v_sync)
E_canon = 1 - cos(v_canonical, v_sync).
```

## Promotion and architecture

Promote only if canonicalization substantially reduces both `E_state` and PPL
without degeneration. The trainable version then uses target-time queries and
canonical predicted-clean K/V:

```text
Q_i = W_Q h(z_i, tau_i)
K_i,V_i = W_K,V h(x_hat_i, tau_bar).
```

The first implementation may use alternating latent and clean-context
attention blocks; do not claim efficiency from the per-block oracle.

## Decision rule

- **Canonical-context support:** PPL and `E_state` move materially toward the
  synchronous baseline, and correct context beats shuffled context.
- **Partial support:** vector discrepancy improves but generation remains poor;
  train only a small adapter before changing the full backbone.
- **Negative:** canonical context does not reduce state error or quality gap.
  Mixed-state failure is not fixed by coordinate normalization alone.

## Result (2026-08-10)

Canonical replacement lowers PPL relative to the raw local Pipeline but does
not approach synchronous quality and does not repair the vector field.

| Checkpoint | Standard | Raw local | Canonical LTR | Canonical + refine 8 | Canonical RTL | Shuffled |
|---|---:|---:|---:|---:|---:|---:|
| ELF base | 296.5 | 1778.1 | 1418.8 | 1350.9 | 1098.9 | 2348.8 |
| Control | 272.9 | 1641.0 | 1230.5 | 1316.9 | 1127.0 | 2341.1 |
| Early-KD | 209.7 | 1204.4 | 950.9 | 912.9 | 627.9 | 1441.5 |

`E_canonical` is `.2051/.1988/.1906` versus raw `.1993/.1922/.1794`; MSE is
also unchanged or worse. Canonical LTR degeneration is `.062/.031/.125`.
Correct predicted-clean context contains useful information because shuffling
it is worse, but simple input replacement is not a shared dynamical coordinate
system. Do not build the full Q/K/V architecture from this diagnostic.
