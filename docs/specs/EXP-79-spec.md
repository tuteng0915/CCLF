# EXP-79 Spec — Late-Coupled Block Denoising

**Status:** ACTIVE / P0 PILOT
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

## Arms and sweep

- **Parallel-32:** all 256 positions denoise jointly for 32 steps.
- **Semi-AR-64:** A completes 32 steps; its decoded tokens are T5-reencoded and
  held as the native condition while B completes 32 steps. No vector-level
  backward revision of A.
- **Late-coupled:** condition representation in `{continuous,reencoded,hybrid}`
  and `m in {20,24,28,30}`.

Use paired `z_A^0,z_B^0` for every arm. A single maturity readout and any T5
re-encoding calls must be reported separately from denoiser calls.

## Stage A — short-context screen

```text
checkpoint = ELF base
block length = 128, total length = 256
uniform ODE-32, noise scale 2, SC-CFG 3
n = 128, seed = 42
```

Report the complete text-quality panel, prefix revision rate, selected hybrid
fraction, and representative paired samples. The primary comparison is against
Semi-AR-64; Parallel-32 is the quality/compute reference rather than an arm the
method is required to beat.

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

and split it by the hybrid confidence mask. Also report full-sequence PPL,
D1/D2, Rep-4, degeneration, word count, max-word share, unique-word ratio,
unigram collapse, denoiser calls, readout calls, and T5 encoding calls.

- **Promote:** beats Semi-AR on PPL without systematic diversity/repetition
  regression, while revision is nonzero and samples remain coherent.
- **Equivalent but cheaper:** matches Semi-AR quality with fewer than 64
  denoiser calls; retain as a compute result.
- **Premature-condition failure:** early `m` corrupts B and quality improves
  monotonically toward Semi-AR as `m -> 32`.
- **No joint-refinement value:** revision is near zero and the best cell is
  effectively Semi-AR.
- **Representation boundary:** reencoded works but continuous fails, showing
  that native `x0` canonicalization rather than soft latent information is
  essential.
