# EXP-79 Spec — Late-Coupled Block Denoising

**Status:** DONE / NEGATIVE ON DECISIVE CONDITIONAL P1
**Motivation:** The discrete Pipeline fails because it exposes the backbone to
raw positions at incompatible denoising times. ELF's native condition pathway,
however, holds clean `x0` states fixed. Test whether blocks can mature
separately through that pathway and interact only after their clocks align.

## Core question

Can a provisional prefix condition the next block without irreversible
semi-autoregressive commitment, if bidirectional joint refinement is delayed
until both blocks reach the same denoising time?

For two 128-token blocks and an ODE-32 grid, define maturity step `m`:

```text
A:       0 ----------> m
B | A:   0 ----------> m
[A,B]:                  m ----------> 32
```

Equivalently,

```text
z_A^m = Phi_0:m(z_A^0)
z_B^m = Phi_0:m(z_B^0 | c_A^m)
[z_A^32,z_B^32] = Phi_m:32([z_A^m,z_B^m]).
```

The raw latents are concatenated only in the last line, where both blocks have
the same nominal time. Total denoiser calls are `32+m`, versus 64 for ordinary
two-block Semi-AR and 32 for fully parallel generation.

## Condition representations

At step `m`, decode the provisional prefix once and compare:

1. **Continuous:** `c_A^m = xhat_A^m`.
2. **Reencoded:** decode `yhat_A^m`, then use ELF's native normalized frozen-T5
   state `c_A^m = E_T5(yhat_A^m)`.
3. **Hybrid:** use reencoded `x0` where decoder confidence is at least `.90`
   and continuous `xhat` elsewhere.

All three representations occupy the native `cond_seq` pathway while block B
is denoised. They are released before joint refinement. “Hard” therefore means
temporarily fixed by `cond_mask`; it does not imply permanent output tokens.

## P0 arms — smallest decisive screen

Do not launch the full grid first. The initial ELF-base screen contains:

- **Parallel-32:** native fully parallel reference.
- **Parallel-60:** exact denoiser-call control for the main `m=28` arm. Because
  all 60 calls process 256 positions, it is a conservative upper-compute
  control rather than an exact FLOP match.
- **Semi-AR-64:** A finishes and becomes a permanent native `x0` condition.
- **Late-reencoded-24:** eight final joint-refinement intervals.
- **Late-reencoded-28:** the primary `28+4` arm.
- **Late-reencoded-28-freeze-A:** same calls and provisional condition, but A
  cannot change during the final four intervals. This isolates the value of
  backward revision from ordinary prefix conditioning.
- **Late-continuous-28** and **Late-hybrid-28:** representation-boundary
  controls, promoted only after the native reencoded arm passes smoke gates.

Only after P0 gives a positive or diagnostic signal should the runner expand
to `m in {20,24,28,30}` and all three representations.

Use paired `z_A^0,z_B^0` for every arm. A single maturity readout and any T5
re-encoding calls must be reported separately from denoiser calls.

## Stage A0 — unconditional mechanism screen

```text
checkpoint = ELF base
block length = 128, total length = 256
uniform ODE-32, noise scale 2, SC-CFG 3
n = 128, seed = 42
```

Report the complete text-quality panel, prefix revision rate, selected hybrid
fraction, and representative paired samples. The primary comparisons are
against both Semi-AR-64 and the call-matched Parallel-60. An improvement
over Parallel-32 alone is not a method result.

Before P0, run `n=8`, `--skip_ppl` smoke gates and require:

1. Parallel-32 exactly reproduces the native length-256 runner under paired
   noise;
2. every clamped prefix has zero restore error;
3. calls are reported as `32+m`, not evaluation-time amortized calls;
4. the frozen-A arm has exactly zero prefix revision.

This is a mechanism/quality sanity check only. It does not answer whether an
early block provides useful information under a real external prompt.

## Stage A1 — decisive conditional-generation screen

Use the fixed Gutenberg continuation panel shared with EXP-64/66/78. Each
length-256 example contains a real 64-token prefix that remains clamped in all
arms. Positions `64:128` form the first generated region A; positions
`128:256` form B. Thus the experiment directly asks whether partially solving
A improves a prompt-conditioned continuation and the A-to-B transition.

Keep the P0 arms, paired noise, seed, and compute accounting unchanged. Report:

```text
PPL(continuation | prompt),
ROUGE-L(continuation, held-out continuation),
PPL(first 32 tokens of B | prompt, generated A),
PPL_A, PPL_B, D1/D2, Rep-4, degeneration, decoded-prefix agreement.
```

The primary decision is prompt-conditioned continuation PPL jointly with
ROUGE-L and boundary PPL; unconditional PPL cannot stop promotion. Require
latent prompt-clamp error `=0` for every arm. Decoded-prefix agreement is an
autoencoder diagnostic, not a sampler gate: evaluator conditioning always
uses the original observed prompt rather than a re-decoded copy.

## Stage B — checkpoint and length promotion

Only promote the best maturity/representation cell:

1. repeat on Control and Early-KD at the same two-block length;
2. run four or eight blocks at total length 512/1024;
3. compare ODE-16/32/64 using a fixed maturity fraction rather than a fixed
   integer step.

Native SDE is conditional on an ODE result that remains positive at length
1024. Do not infer SDE behavior from this ODE construction.

## Stage C — LangFlow and Plaid boundary test

LangFlow and Plaid do not expose an ELF-equivalent, training-native `cond_seq`
interface in the current adapters. Their test is therefore a portability
boundary rather than an exact replication of Stage A. At splits `{24,28,30}`,
compare four explicitly clamped prefix contexts:

1. **neutral clean:** encoded neutral tokens;
2. **raw:** the prefix's synchronized raw latent and self-conditioning state;
3. **continuous:** the model's predicted-clean prefix;
4. **hard:** decoded tokens passed through the architecture's clean encoder.

Include full-parallel and hard Block-SAR references. Plaid must reuse paired
ancestral step noise across arms. A positive result here would show that late
clock alignment can transfer despite the absence of ELF's native condition
path; a negative result does not invalidate the ELF-specific method.

The cross-architecture runner is
`experiments/interventions/eval_late_coupled_blocks.py`; the ELF-native runner
is `models/ELF-torch/experiments/probe_elf/late_coupled_blocks_exp79.py`.

## Diagnostics and decision rule

Track

```text
prefix_revision = mean_i 1[y_A^m(i) != y_A^32(i)]
```

and split it by the hybrid confidence mask. Also report suffix revision during
the joint phase and four evaluator views:

```text
PPL_full, PPL_A, PPL_B, PPL_B|A(first 32 evaluator tokens).
```

The conditional boundary PPL is the most direct test of whether A supplies a
better condition to the beginning of B. Retain D1/D2, Rep-4, degeneration,
word count, max-word share, unique-word ratio, unigram collapse, denoiser
calls, processed-token calls, readout calls, and T5 encoding calls.

- **Promote:** beats Parallel-60 and matches or beats Semi-AR-64 without a
  diversity/repetition regression; full joint refinement beats freeze-A and
  prefix revision is nonzero.
- **Equivalent but cheaper:** matches Semi-AR quality with fewer than 64
  denoiser calls; retain as a compute result.
- **Extra-compute-only:** beats Parallel-32 but not Parallel-60. Reject the
  scheduling claim.
- **Premature-condition failure:** early `m` corrupts B and quality improves
  monotonically toward Semi-AR as `m -> 32`.
- **No joint-refinement value:** full joint and freeze-A are equivalent, or
  revision is near zero; the best cell is effectively Semi-AR.
- **Representation boundary:** reencoded works but continuous fails, showing
  that native `x0` canonicalization rather than soft latent information is
  essential.

## Unconditional P0 result (2026-08-11)

The `n=8` smoke passed every implementation gate: native Parallel-32 agreement
was `1.0`, all condition-restore errors were exactly zero, freeze-A prefix
revision was zero, and full-joint prefix revision was nonzero (`.080` at
`m=24`, `.051` at `m=28`).

The decisive ELF-base run used `n=128`, seed 42, length 256, native noise 2,
SC-CFG 3, and the complete metric panel:

| Arm | PPL | PPL A | PPL B | boundary PPL B\|A | D1 | D2 | Deg. | prefix rev. | calls | token-calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel-32 | **169.5** | 228.9 | 215.8 | 148.6 | .357 | .832 | .016 | — | 32 | 8192 |
| Parallel-60 | **76.3** | 102.8 | 102.9 | 67.1 | .325 | .785 | .023 | — | 60 | 15360 |
| Semi-AR-64 | 311.9 | 296.7 | 394.2 | 648.8 | .409 | .863 | .000 | — | 64 | 12288 |
| Late reencoded, `m=24` | 300.6 | 285.0 | 382.1 | 602.9 | .401 | .862 | .008 | .057 | 56 | 11264 |
| Late reencoded, `m=28` | 309.1 | 290.8 | 392.8 | 634.8 | .409 | .865 | .000 | .037 | 60 | 11776 |
| `m=28`, freeze A | 312.1 | 297.7 | 395.5 | 630.5 | .410 | .865 | .000 | .000 | 60 | 11776 |

Late coupling is only marginally better than Semi-AR (`-11.4` PPL at `m=24`,
`-2.9` at `m=28`) and is dramatically worse than ordinary parallel decoding.
At `m=28`, full joint refinement improves full PPL by only `3.0` relative to
freeze-A, while boundary conditional PPL is `4.2` worse. Nonzero revision is
therefore not providing a meaningful bidirectional-refinement benefit.

The unconditional result alone was not used to stop the method.

## Conditional P1 result (2026-08-11)

The decisive run used 128 fixed Gutenberg examples with an observed 64-token
prefix and a 192-token continuation. All arms shared prompts, suffix noise,
and seed. Prompt latents were restored exactly (`max error = 0`) in every arm.
`Decoded prefix` is only the frozen T5/ELF autoencoder reconstruction
diagnostic; evaluator conditioning uses the original observed prompt.

| Arm | PPL(cont.) | PPL(cont.\|prompt) | R-L | PPL(B-start\|prompt,A) | D1 | D2 | Rep-4 | Deg. | Decoded prefix | A rev. | B rev. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel-32 | 238.3 | 252.8 | .1020 | 242.9 | .3478 | .8354 | .0102 | .0078 | .9922 | — | — |
| Parallel-60 | **128.3** | **134.5** | **.1027** | **125.5** | .3205 | .7931 | .0261 | .0156 | 1.0000 | — | — |
| Semi-AR-64 | 394.9 | 421.2 | .0968 | 553.1 | .3650 | .8561 | .0104 | .0000 | .9922 | — | — |
| Late reencoded, `m=24` | 393.6 | 419.0 | .0958 | 566.3 | .3653 | .8537 | .0105 | .0000 | .9922 | .0660 | .0817 |
| Late reencoded, `m=28` | 394.2 | 420.2 | .0967 | 568.3 | .3620 | .8530 | .0111 | .0000 | 1.0000 | .0453 | .0502 |
| `m=28`, freeze A | 398.1 | 423.3 | .0967 | 566.8 | .3631 | .8533 | .0100 | .0000 | 1.0000 | .0000 | .0494 |

Late-28 versus Semi-AR changes prompt-conditioned PPL by only `-1.0` and
ROUGE-L by `-0.0001`, while worsening A-to-B boundary PPL by `+15.2`.
Against Parallel-32/60, prompt-conditioned PPL is worse by `+167.4/+285.7`
and ROUGE-L is lower by `.0053/.0059`. Allowing A to revise changes `4.5%` of
its generated positions and improves prompt-conditioned PPL by `3.0` versus
freeze-A, but leaves ROUGE-L unchanged and worsens boundary PPL by `1.5`.

**Decision:** stop after conditional P1. The method is not rejected because of
unconditional PPL: it fails the intended prompted-continuation test as well.
Do not expand representations, checkpoints, lengths, or architectures unless
the algorithm changes materially.
