# EXP-61 Spec — Native-Path Revalidation of Pipeline ODE

**Status**: RUNNING — Stage-0 smoke passed; Stage-1 noise correction launched
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
| checkpoint payload | converted `params` | converted `params` |
| reported checkpoints | `kd_cr`, `kd2` | baseline not tested |

The training and evaluation configs set `denoiser_noise_scale: 2.0`, and
`src/generation.py` multiplies the initial Gaussian by this value. Therefore
EXP-58/59 established a paired result under a non-native initial-noise
distribution, but did not yet establish that Pipeline ODE improves normal ELF
generation.

The checkpoint-key audit corrected an initially suspected EMA mismatch. The
JAX-to-PyTorch converter selects JAX `ema_params1` by default but stores the
converted tensor dictionary under the generic PyTorch key `params`. The
current converted files have no separate `ema_params1` key. Thus loading
`checkpoint["params"]` does **not** by itself imply that EXP-58/59 used raw
training weights. Raw-versus-EMA attribution would require separately
converted source checkpoints and is not part of the primary revalidation.

This is a protocol revalidation, not a robustness embellishment. Until it is
resolved, the EXP-59 method claim is provisional.

## 2. Questions

1. Does the published EXP-59 result reproduce under its exact legacy path?
2. Does Pipeline ODE still help `kd_cr` under native initial noise?
3. Does it work on the ELF baseline checkpoint?
4. Can the converted checkpoint's JAX source (`ema_params1` versus raw
   `params`) be verified from conversion records or a reference comparison?
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

The first server smoke (`2026-08-06`, `n=64`) reproduced the legacy direction:

| arm | Gen.PPL | D1 | D2 | rep-4 | degeneration |
|---|---:|---:|---:|---:|---:|
| standard | 309.59 | 0.317 | 0.851 | 0.014 | 0.250 |
| Pipeline | 188.78 | 0.435 | 0.896 | 0.000 | 0.547 |

This is an implementation check, not method evidence: the degeneration flag
also worsened, and the run deliberately used the legacy noise-scale-1 path.
Stage 1 must determine whether either observation survives native noise.

## 4. Stage 1 — initial-noise correction

Use identical initial Gaussian draws within every paired comparison.

| cell | weights | noise scale | purpose |
|---|---|---:|---|
| A | converted checkpoint (`auto`) | 1.0 | exact legacy reference |
| B | same checkpoint (`auto`) | 2.0 | native primary result |

Run both cells on `kd_cr`, seed 42, `n=64` smoke. If the result changes sign or
magnitude, retain both at `n=256` to attribute the change to initial noise;
otherwise run the native cell formally. For a newly trained PyTorch checkpoint
that contains both keys, `--weights params` and `--weights ema` can be used as
a secondary weight-sensitivity audit.

Reproducible runner:

```bash
CUDA_VISIBLE_DEVICES=5 bash experiments/probe_elf/run_exp61_stage1.sh
```

## 5. Stage 2 — checkpoint scope

Primary native cell B:

```text
checkpoints = {baseline, kd_cr, kd2}
seed        = 42 (fixed for reproducibility, not an experimental axis)
n           = 256 paired initial-noise samples
length      = 128
arms        = {standard ODE-32, pipeline_avg T=16 / 31 calls}
```

Metrics:

- token-weighted GPT-2-large Gen.PPL;
- corpus Distinct-1 and Distinct-2;
- 4-gram repetition;
- empty/non-ASCII/repetition degeneration rate;
- four fixed qualitative samples per arm;
- paired bootstrap interval over the fixed initial-noise bank when needed.

Do not add generation-seed sweeps. Robustness is tested across checkpoint,
length, solver budget, and native versus legacy evaluation path.

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

For any checkpoint with a healthy Stage-2 result, use one fixed set of
OpenWebText prefix/suffix pairs and paired initial noise, and report:

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

- **Method survives**: `kd_cr` native-cell Pipeline reduces Gen.PPL at both
  length 128 and 1024, survives the solver-budget check, D1/D2 and degeneration
  remain healthy, and conditional quality is not materially worse.
- **Checkpoint-scoped sampler**: native unconditional result survives only on
  `kd_cr`, while baseline and `kd2` fail. Keep the result, but explicitly limit
  scope and investigate the KD-window interaction.
- **Legacy-path artifact**: benefit disappears or reverses primarily when
  `noise_scale` changes from 1 to 2. Retract the current Pipeline method claim
  from the main story; retain it as an evaluation-path cautionary result.
- **Checkpoint provenance unresolved**: converted source cannot be verified.
  Preserve the result as applying to the exact converted artifact and record
  its checksum; do not label it raw or EMA without evidence.

## 8. Relation to EXP-60

EXP-61 has priority over native WFF training because it audits an existing
positive claim already shown in the presentation. EXP-60 tests a new model
hypothesis and may run in parallel after both scripts pass smoke tests.
