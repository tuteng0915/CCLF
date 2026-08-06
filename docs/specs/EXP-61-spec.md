# EXP-61 Spec — Native-Path Revalidation of Pipeline ODE

**Status**: READY  
**Priority**: P0 — must be resolved before treating Pipeline ODE as a method result  
**Models**: ELF-B baseline, `kd_cr`, `kd2`  
**Primary script**: `experiments/probe_elf/pipeline_native_revalidation_exp61.py`  
**Historical references**: EXP-58 and EXP-59

## 1. Problem exposed during presentation audit

The positive Pipeline ODE result was evaluated through a custom script whose
sampling initialization differs from ELF's native evaluation path:

| component | EXP-58/59 custom path | native ELF path |
|---|---|---|
| initial latent | `z0 = randn(...)` | `z0 = 2.0 * randn(...)` |
| checkpoint weights | `params` | EMA weights when available |
| reported checkpoints | `kd_cr`, `kd2` | baseline not tested |

The training and evaluation configs set `denoiser_noise_scale: 2.0`, and
`src/generation.py` multiplies the initial Gaussian by this value. Therefore
EXP-58/59 established a paired result under a non-native initial-noise
distribution, but did not yet establish that Pipeline ODE improves normal ELF
generation.

This is a protocol revalidation, not a robustness embellishment. Until it is
resolved, the EXP-59 method claim is provisional.

## 2. Questions

1. Does the published EXP-59 result reproduce under its exact legacy path?
2. Does Pipeline ODE still help `kd_cr` under native noise and EMA weights?
3. Does it work on the ELF baseline checkpoint?
4. Which factor explains any change: noise scale, EMA weights, or their
   interaction?
5. If unconditional quality survives, does conditioned semantic continuation
   survive as well?

## 3. Stage 0 — implementation gate and legacy reproduction

Run `kd_cr`, seed 42, `n=64` first, then `n=256` if needed:

```text
weights=params, noise_scale=1.0
arms={standard ODE-32, pipeline_avg T=16 / 31 calls}
```

Acceptance gate: the `n=256` run should reproduce EXP-59 seed-42 within normal
numerical tolerance (standard PPL about 338, pipeline PPL about 196, and the
same D1/D2 direction). If it does not, stop and audit code/version drift.

## 4. Stage 1 — factorized path correction

Use identical initial Gaussian draws within every paired comparison.

| cell | weights | noise scale | purpose |
|---|---|---:|---|
| A | params | 1.0 | exact legacy reference |
| B | EMA | 1.0 | isolate weight selection |
| C | params | 2.0 | isolate initial-noise distribution |
| D | EMA | 2.0 | native primary result |

Run the four cells on `kd_cr`, seed 42, `n=64` smoke. Retain all four at
`n=256` only if the native result changes sign or magnitude enough to require
attribution; otherwise run the native cell formally.

## 5. Stage 2 — checkpoint scope

Primary native cell D:

```text
checkpoints = {baseline, kd_cr, kd2}
seeds       = {42, 123, 456}
n           = 256 sequences / seed
length      = 128
arms        = {standard ODE-32, pipeline_avg T=16 / 31 calls}
```

Metrics:

- token-weighted GPT-2-large Gen.PPL;
- corpus Distinct-1 and Distinct-2;
- 4-gram repetition;
- empty/non-ASCII/repetition degeneration rate;
- four fixed qualitative samples per arm;
- paired per-seed difference and 95% CI over seeds.

The independent replication unit is the generation seed, not tokens.

This length-128 cell is the historical-comparability result, not sufficient
evidence of native-length robustness. Any checkpoint on which Pipeline appears
healthy must also pass the standardized promotion checks in
[`EVAL-PROTOCOL.md`](EVAL-PROTOCOL.md): length 1024 at ODE-32, plus the
length-128 ODE-16/64 solver checks. The script exposes `--max_length`,
`--n_steps`, `--pipeline_groups`, and `--ppl_max_length`; keep
`pipeline_groups=n_steps/2` so Pipeline uses one fewer model call than the
standard arm.

## 6. Stage 3 — conditional quality, only if Stage 2 survives

The existing single-seed Gutenberg experiment gives `kd_cr` Pipeline ODE a
lower suffix PPL but lower ROUGE-L than standard (`~0.027` versus `~0.050`).
This is not a clean semantic-quality win.

For any checkpoint with a healthy Stage-2 result, run 3 seeds on fixed
OpenWebText prefix/suffix pairs and report:

- suffix Gen.PPL and D1/D2;
- token ROUGE-L;
- BERTScore or an embedding-based continuation similarity;
- exact prefix preservation;
- repetition/degeneration;
- samples stratified by improvement and failure.

Do not claim semantic continuation improvement from PPL alone. If the method
improves unconditional quality but hurts continuation, frame it as an
unconditional sampler heuristic rather than a general denoising method.

## 7. Decision rule

- **Method survives**: `kd_cr` native-cell Pipeline reduces Gen.PPL on all
  three seeds, D1/D2 and degeneration remain healthy, and conditional quality
  is not materially worse.
- **Checkpoint-scoped sampler**: native unconditional result survives only on
  `kd_cr`, while baseline and `kd2` fail. Keep the result, but explicitly limit
  scope and investigate the KD-window interaction.
- **Legacy-path artifact**: benefit disappears or reverses primarily when
  `noise_scale` changes from 1 to 2. Retract the current Pipeline method claim
  from the main story; retain it as an evaluation-path cautionary result.
- **EMA sensitivity**: sign depends on `params` versus EMA. Report both and
  stop using an unspecified checkpoint state in method comparisons.

## 8. Relation to EXP-60

EXP-61 has priority over native WFF training because it audits an existing
positive claim already shown in the presentation. EXP-60 tests a new model
hypothesis and may run in parallel after both scripts pass smoke tests.
