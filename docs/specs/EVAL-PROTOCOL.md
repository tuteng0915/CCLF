# CCLF Evaluation Protocol

**Status**: ACTIVE shared protocol  
**Purpose**: prevent length, solver, and evaluator settings from silently
changing the conclusion across experiments

## 1. There are three different historical settings

| role | generated length | trajectory / solver | sample scale | interpretation |
|---|---:|---|---:|---|
| native ELF checkpoint evaluation | 1024 | SDE-32/64, logit-normal; native noise scale 2 and EMA | config-dependent | official generation-quality reference |
| current mechanism evidence | 1024 | GS16: ODE-32; GS17: dense ODE-128 | GS16: 48 trajectories x 3 seeds; GS17: 48 x 2 completed seeds in the current ledger | paper's rollout-dynamics setting |
| current Pipeline result | 128 | standard ODE-32 vs Pipeline `T=16` / 31 model calls | 256 sequences x 3 seeds | cheap method result requiring native-path and native-length revalidation |

Consequently, “the default” must always identify its purpose. Length 128 is a
screening and historical-comparability setting. Length 1024 is ELF-B's native
OpenWebText context and is required before making a general method claim.

## 2. Shared constants

Unless an experiment is explicitly reproducing a legacy cell:

- load EMA weights;
- initialize `z0 = 2 * epsilon` for the relevant ELF checkpoints;
- use paired initial noise for paired sampler comparisons;
- use uniform-time ODE for mechanism and controlled sampler comparisons;
- match evaluator truncation to generated length (1024 for length-1024 runs);
- report generated length, evaluator length, solver, step count, model calls,
  self-conditioning scale, checkpoint state, and noise scale in every result;
- use one fixed sampling seed (`42`) for reproducibility and paired initial
  noise; do not treat random-seed reruns as a robustness axis;
- spend replication budget on more samples and controlled changes in length,
  solver budget, sampler, and checkpoint.

## 3. Staged test matrix

Do not run the full Cartesian product on every idea. Promote it in stages.

| tier | length | solver budget | samples | sampling | purpose |
|---|---:|---:|---:|---|---|
| smoke | 128 | ODE-32 | 64 | fixed seed 42 | catch implementation failures |
| primary paired result | 128 | ODE-32 | 256 | fixed seed 42, paired noise | compare with the existing method archive |
| native-length check | 1024 | ODE-32 | 256 | fixed seed 42, paired noise | rule out length-specific conclusions |
| solver check | 128 | ODE-16 and ODE-64 | 256 | reuse the same fixed noise bank | rule out one-budget tuning |
| official-fidelity check | 1024 | native SDE-32 (logit-normal, gamma 1.5, SC-CFG 3) | 256 | fixed seed 42 | required only for claims about normal ELF generation quality |

For Pipeline, pair ODE-16/32/64 with `T=8/16/32`, giving 15/31/63 model calls.
Compare quality at nearly matched model-call budgets and report the one-call
difference rather than calling the budgets identical.

If compute is constrained, the irreducible formal set is:

```text
(length=128, ODE=32) + (length=1024, ODE=32)
```

Length 512 is an optional localization point, not a mandatory default: use it
when a 128-versus-1024 discrepancy appears or when reproducing the historical
length-sensitivity experiments.

If uncertainty bars are needed, use a paired bootstrap over the fixed bank of
generated examples. Generality is established by controlled settings and
checkpoints, not by changing the pseudorandom seed.

## 4. Decision rules

- A result seen only at length 128 is a **short-context screening result**.
- A result that survives 128 and 1024 at ODE-32 is **length-robust at the
  primary budget**.
- A result whose sign changes at ODE-16/32/64 is **solver-budget-sensitive**;
  do not average the settings into one headline number.
- A method improvement on ODE but not native SDE is an **ODE-specific sampler
  result**, not a general checkpoint-quality improvement.
- If quality and diversity disagree, retain both plus repetition,
  degeneration, and fixed qualitative samples; Gen.PPL alone is not a pass.

## 5. Why multiple settings are necessary here

The existing archive already contains a qualitative warning: changing ELF
evaluation length from 512 to 1024 substantially changed Gen.PPL and even
reversed the apparent effect of Diffusion Forcing on `kd2`. Solver-step sweeps
also changed relative effects. Length and solver budget are therefore
mechanistic axes in this project, not cosmetic robustness checks.
