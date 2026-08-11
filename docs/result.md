# CCLF Major Experimental Results

**Last updated:** 2026-08-10
**Purpose:** single source of truth for the major paper-facing experiments and
the complete quality metrics produced by the formal evaluation runners.

This document is a curated result ledger, not a replacement for the full
historical index. Superseded and invalid protocols are included only when they
change the interpretation of a later result. See
[`specs/EXP-INDEX.md`](specs/EXP-INDEX.md) for all experiment IDs,
[`specs/DEAD-ENDS.md`](specs/DEAD-ENDS.md) for invalidated claims, and the
individual specs for implementation details.

## 1. Reading rules

1. **Do not compare PPL across different protocol blocks.** Sequence length,
   ODE versus SDE, initial-noise scale, SC-CFG, and sample bank all change the
   absolute value.
2. Unless stated otherwise, ELF generation uses EMA weights, GPT-2-large for
   evaluation PPL, and T5 tokenization inside the generator.
3. `—` means the runner did not compute that metric. It does not mean zero.
4. Smoke tests are excluded. Formal pilots are marked as pilots and should not
   be promoted to multi-seed claims.
5. The detailed method tables use the exact seed-42 JSON outputs so that every
   recorded metric comes from one internally consistent run. Separate
   multi-seed tables report only metrics actually aggregated across seeds.
6. Raw JSON and generated texts remain on the experiment server under
   `models/ELF-torch/results/`. This file records the compact numeric ledger;
   it does not duplicate generated text.

## 2. Metric dictionary

For generated sequences `x^(m)` and evaluator tokens, the main metrics are:

```text
Gen.PPL = exp[- (1 / N_tok) sum_j log p_GPT2(x_j | x_<j)]

D1 = number of distinct unigrams / number of unigram tokens
D2 = number of distinct bigrams  / number of bigram tokens

Rep-4 = mean_m [1 - unique_4grams(x^(m)) / total_4grams(x^(m))]

MaxShare = mean_m max_w count_m(w) / number_of_words_m
UniqueRatio = mean_m unique_words_m / number_of_words_m
```

- **Deg.**: fraction caught by the shared empty/non-ASCII/repetition
  degeneration detector.
- **U-collapse**: fraction whose most frequent word exceeds the stricter 20%
  unigram-share threshold.
- **Words**: mean decoded word count per sequence.
- **R-L**: token-level ROUGE-L F1 against the held-out suffix in conditioned
  generation.
- **Commit**: selected anchors divided by eligible, non-prefix positions.
- **Calls**: denoiser calls; a lexical confidence readout is counted where the
  runner makes an extra model call.
- **`tau_first`**: first normalized time at which a position equals its branch
  endpoint.
- **`tau_stable`**: first time with endpoint agreement for three consecutive
  checkpoints.
- **`N_rev`**: number of top-1 lexical changes along the recorded trajectory.
- **Own-endpoint margin**: at the first post-intervention step, logit evidence
  for the endpoint eventually reached by that same branch relative to its
  strongest competitor.

## 3. Current result map

| Question | Best current evidence | Decision |
|---|---|---|
| Is the early global signal model-created? | GS11/12 | Mostly no: mean pooling/raw-state statistics create the early retrieval signal. |
| Is exact lexical identity already fixed early? | GS14/16/17 | No: lexical alternatives contract late, with a narrow endpoint-affinity collapse. |
| Do other positions causally affect a target token? | GS13/18-B/EXP-67 | Yes on deterministic ELF rollout; correct position-content anchors stabilize unresolved positions. |
| Is rollout simply a smooth approach to a fixed endpoint? | GS15/16/17/EXP-67 | No. The evidence is more consistent with context-dependent endpoint selection and late collapse. |
| Does post-hoc asynchronous denoising work? | GS19, ELF + Plaid | No; all schedules damage timing and quality. |
| Does native per-position-time fine-tuning rescue it? | EXP-60/72 | No in the tested architectures. Deep injection preserves standard quality but still fails the functional clock-learning gate at step 500; the LTR interaction worsens. |
| Does Pipeline work? | EXP-61/64/70 | No under native noise. Correct local clocks and final joint refinement do not rescue it; heterogeneous context, not average-clock aliasing, is the dominant error. |
| Can synchronized soft leaders provide directional conditioning? | EXP-71 | Correct leader content matters, but repeated soft anchoring loses to ordinary compute-matched ODE-64 and has no LTR advantage. |
| Can sparse triggered anchoring turn the causal anchor effect into a method? | EXP-74 | Soft expiry does not work, but one persistent post-transition hard anchor improves all three checkpoints; high-confidence and stable-density controls retain the sign. |
| Can predicted-clean context repair heterogeneous attention? | EXP-75 | Only partially in PPL; vector error is unchanged/worse and generation remains catastrophic. Simple canonical input replacement is rejected. |
| Can a clock be forced to learn before asynchronous training? | EXP-76 | Partly yes: frozen adapters learn a functional clock without hurting Standard generation, but wave quality remains poor. |
| Does asynchronous block-transition distillation then work? | EXP-77 | No. Standard generation stays healthy, but all fill/drain samplers remain at PPL `3400--3900`; local transitions do not compose. |
| Does corrected temporal KD work? | EXP-63/66 | Early-window KD improves unconditional ODE quality and timing in two training seeds; conditioned gains are not robust. |
| Does hard commitment work? | EXP-64--69/74/78 | Yes as an ODE-specific intervention. Three-seed and conditioned gains replicate, and a four-step lock is sufficient; native-SDE effects remain negligible. |

## 4. Main mechanism evidence

### 4.1 Measurement correction and representation

| Experiment | Protocol/statistic | Main numeric result | Status |
|---|---|---|---|
| GS11 | Raw-state mean-pool self-retrieval, `n=48` | At `t=.28`, retrieval is 1.000 for `L={32,128,512,1000}`; at `t=.05`, it rises `.396 -> .583 -> .792 -> .938` with length. | Early-global headline is a pooling confound. |
| GS12 | Centered mean/SVD decomposition, `n=128`, `k=8` | `MEAN_only` is best or tied in all 18 `(t, repr)` structural-R2 cells. At `t=.65`, model `MEAN+R_c` token accuracy `.814` versus `MEAN+G_c` `.091`. | Mean explains coarse structure; exact token recovery needs the larger residual subspace. |
| GS18-A | Rank/energy-matched subspace controls | Top-k beats middle/bottom/random-k at fixed dimensionality; e.g. ELF `k=128` raw token accuracy `.067` versus `.007/.000/.020`. | Narrows “special high-rank code” to “sufficient dimension/energy is required.” |
| GS7 | Oracle versus free-rollout token recovery | Global geometry largely follows oracle while exact lexical commitment lags. | Oracle-rollout gap is primarily lexical, not an early semantic-state failure. |

### 4.2 Context, branching, and the transition window

| Experiment | Metric | Result | Interpretation |
|---|---|---|---|
| GS13 | Context-only target-margin change | ELF correct-direction effects span roughly `.18` to `.82`, versus orthogonal/random approximately `-.13` to `+.13`; response is non-monotone. | Other positions causally change target lexical evidence, but not through a simple linear topic direction. |
| GS14 | True-trajectory branch consensus | ELF `C_lex=.858 -> .966 -> .992`; `C_topic=.974 -> .982 -> 1.000` at `t={.20,.38,.65}`. | Topic/coarse basin is already stable; lexical alternatives contract later. |
| GS15 | Residual endpoint alignment deficit | `A_rollout-A_linear` is about `-.22` to `-.23` around `t=.38-.50`. | Describes curved/slow residual transport, but the endpoint-aware chord is not a causal null. |
| GS16 | Endpoint specificity and affinity entropy, formal 3-seed | `H_end` peaks `.848 +/- .020` at `t=.301`, then falls to `.524 +/- .001`; `N_eff` plateau `2.48 +/- .05`. | Strong exploration-collapse evidence in a narrow window. |
| GS17 | Unified event timing, formal 3-seed | `tau_50stable=.206 +/- .022`, `tau_aff=.322 +/- .010`, median `tau_v=.170`; `P(tau_v<=tau_50s)=.896`, `P(tau_aff<=tau_50s)=.049`. | Velocity reorientation usually precedes stable tokens; endpoint-affinity collapse is later. |
| GS18-B | Residualized collective-coupling nulls | ELF exceeds all five null 95th percentiles at 13/16 checkpoints. | Collective coupling survives position, sequence, margin, entropy, and variance controls on deterministic rollout. |

### 4.3 Position-correct anchor intervention (EXP-67)

Protocol: ELF ODE-32, length 128, 48 paired trajectories, fork at `t=.40`,
confidence `.60`. Approximately 95% of positions are anchored. Shuffled
anchors preserve positions and match confidence/frequency quartiles but move
the continuous anchor vectors.

| Checkpoint | Commit fraction | Anchors | Shuffled same-token | Mean confidence mismatch | Mean log-frequency mismatch |
|---|---:|---:|---:|---:|---:|
| ELF base | .9526 | 5,853 | .0400 | .0289 | .5009 |
| Control | .9505 | 5,840 | .0348 | .0285 | .4650 |
| Early-KD | .9520 | 5,849 | .0468 | .0291 | .4948 |

#### Unresolved-position causal metrics

| Checkpoint | Arm minus natural | `Delta tau_first` | `Delta tau_stable` | `Delta N_rev` | Own-endpoint margin delta | Natural-endpoint agreement |
|---|---|---:|---:|---:|---:|---:|
| ELF base | true anchors | -.032 | -.014 | -.619 | +4.562 | .265 |
| ELF base | shuffled | +.069 | +.068 | +1.357 | -2.165 | .010 |
| Control | true anchors | -.038 | -.025 | -.572 | +3.886 | .296 |
| Control | shuffled | +.050 | +.047 | +1.214 | -2.589 | .007 |
| Early-KD | true anchors | -.028 | -.031 | -.420 | +3.436 | .353 |
| Early-KD | shuffled | +.080 | +.066 | +1.444 | -3.917 | .020 |

The complete first-post-fork margin contrasts are:

| Checkpoint | Margin target | True-natural | Shuffled-natural | True-shuffled |
|---|---|---:|---:|---:|
| ELF base | natural endpoint | -3.226 | -20.769 | +17.543 |
| ELF base | each branch's own endpoint | +4.562 | -2.165 | +6.727 |
| Control | natural endpoint | -2.911 | -21.544 | +18.633 |
| Control | each branch's own endpoint | +3.886 | -2.589 | +6.475 |
| Early-KD | natural endpoint | -1.894 | -19.398 | +17.504 |
| Early-KD | each branch's own endpoint | +3.436 | -3.917 | +7.353 |

The low natural-endpoint agreement means the intervention often changes the
future token, so “faster movement toward a predetermined endpoint” is too
strong. The supported claim is that position-correct context induces and
stabilizes a coherent lexical future.

<details>
<summary>Complete absolute timing metrics for all, selected, and unresolved positions</summary>

| Checkpoint | Scope | Arm | `tau_first` | `tau_stable` | `N_rev` |
|---|---|---|---:|---:|---:|
| ELF base | all | natural | .3424 | .3571 | 5.8467 |
| ELF base | all | true anchors | .2936 | .3056 | 5.3815 |
| ELF base | all | shuffled | .4322 | .4521 | 6.9712 |
| ELF base | selected | natural | .3306 | .3448 | 5.6169 |
| ELF base | selected | true anchors | .2810 | .2914 | 5.1594 |
| ELF base | selected | shuffled | .4215 | .4411 | 6.7299 |
| ELF base | unresolved | natural | .5784 | .6047 | 10.4674 |
| ELF base | unresolved | true anchors | .5464 | .5907 | 9.8488 |
| ELF base | unresolved | shuffled | .6474 | .6731 | 11.8247 |
| Control | all | natural | .3369 | .3514 | 5.7936 |
| Control | all | true anchors | .2926 | .3038 | 5.3690 |
| Control | all | shuffled | .4298 | .4474 | 6.9227 |
| Control | selected | natural | .3252 | .3391 | 5.5846 |
| Control | selected | true anchors | .2805 | .2904 | 5.1676 |
| Control | selected | shuffled | .4203 | .4377 | 6.7092 |
| Control | unresolved | natural | .5622 | .5874 | 9.8092 |
| Control | unresolved | true anchors | .5245 | .5622 | 9.2368 |
| Control | unresolved | shuffled | .6122 | .6340 | 11.0230 |
| Early-KD | all | natural | .3235 | .3389 | 5.6509 |
| Early-KD | all | true anchors | .2832 | .2962 | 5.2879 |
| Early-KD | all | shuffled | .4231 | .4419 | 6.7876 |
| Early-KD | selected | natural | .3125 | .3265 | 5.4440 |
| Early-KD | selected | true anchors | .2716 | .2832 | 5.0839 |
| Early-KD | selected | shuffled | .4131 | .4314 | 6.5652 |
| Early-KD | unresolved | natural | .5419 | .5845 | 9.7525 |
| Early-KD | unresolved | true anchors | .5136 | .5533 | 9.3322 |
| Early-KD | unresolved | shuffled | .6217 | .6501 | 11.1966 |

</details>

#### Complete generation diagnostics

| Checkpoint | Arm | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | UniqueRatio | U-collapse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | natural | 281.2 | .5170 | .9011 | .0115 | .0000 | 78.3 | .0762 | .6896 | .0000 |
| ELF base | true anchors | 208.7 | .5114 | .8979 | .0119 | .0000 | 77.5 | .0757 | .6899 | .0000 |
| ELF base | shuffled | 4175.9 | .6447 | .9915 | .0000 | .0000 | 86.8 | .0419 | .9090 | .0000 |
| Control | natural | 262.1 | .5154 | .8999 | .0154 | .0208 | 80.7 | .0702 | .6972 | .0000 |
| Control | true anchors | 206.4 | .5101 | .9037 | .0109 | .0208 | 80.2 | .0719 | .7041 | .0000 |
| Control | shuffled | 4139.4 | .6374 | .9940 | .0000 | .0000 | 91.5 | .0346 | .9123 | .0000 |
| Early-KD | natural | 205.8 | .4980 | .9013 | .0152 | .0000 | 78.6 | .0738 | .6718 | .0000 |
| Early-KD | true anchors | 166.2 | .4857 | .8986 | .0148 | .0000 | 77.8 | .0763 | .6682 | .0000 |
| Early-KD | shuffled | 3761.4 | .6160 | .9920 | .0000 | .0000 | 89.5 | .0381 | .9000 | .0000 |

High D1/D2 for shuffled anchors is not quality: PPL reveals incoherent word
salad. Diversity metrics must always be read with coherence metrics.

## 5. Training interventions

### 5.1 Superseded noisy-head KD panel (EXP-62)

This valid negative experiment used the wrong teacher/objective for the
historical KD question. It is retained because it demonstrates metric gaming.

| Training | PPL | D1 | D2 | Rep-4 | Deg. |
|---|---:|---:|---:|---:|---:|
| Continued-training control | 261.8 | .394 | .860 | .008 | .008 |
| Noisy-head KD, full | 220.0 | .441 | .873 | .007 | .004 |
| Noisy-head KD, early | **159.5** | .451 | .866 | .014 | .016 |
| Noisy-head KD, transition | 311.3 | .379 | .855 | .004 | .000 |
| Noisy-head KD, late | 273.0 | .396 | .861 | .007 | .004 |

At ODE-64 the early arm reaches PPL `53.5` but degeneration `.098`, and sample
inspection shows repetitive fragmented pseudo-text. It must not be mixed with
the corrected EXP-63 objective.

### 5.2 Corrected clean-teacher temporal KD (EXP-63)

All rows use length 128, ODE-32. Timing is measured on true rollout.

| Training checkpoint | PPL | D1 | D2 | Rep-4 | Deg. | `tau_first` | `tau_stable` | `N_rev` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Continued-training control | 261.8 | .3937 | .8602 | .0076 | .0078 | .318 | .347 | 5.74 |
| Broad corrected KD | 261.6 | .4225 | .8766 | .0051 | .0039 | .310 | .338 | 5.90 |
| Early window `[.05,.30]` | **211.0** | .3953 | .8551 | .0083 | .0039 | **.301** | **.329** | **5.55** |
| Transition `[.30,.55]` | 278.6 | .3899 | .8651 | .0043 | .0000 | .319 | .345 | 5.76 |
| Late `[.55,.80]` | 254.9 | .3902 | .8596 | .0064 | .0039 | .316 | .344 | 5.78 |
| Seed-7 control | 257.4 | .3951 | .8608 | .0076 | .0039 | .319 | .348 | 5.79 |
| Seed-7 early | **224.2** | .3983 | .8556 | .0065 | .0000 | **.302** | **.330** | **5.47** |

Early-control paired bootstrap 95% intervals are `[-.0219,-.0126]` for
`tau_first`, `[-.0227,-.0127]` for `tau_stable`, and `[-.265,-.113]` for
revisions. Seed-7 intervals are `[-.0225,-.0128]`, `[-.0232,-.0127]`, and
`[-.394,-.253]`. Early KD also improves ODE-16/64 PPL (`634.8/82.6` versus
`775.4/109.0`).

### 5.3 Native Wavefront Flow Fine-Tuning (EXP-60)

Protocol: 500 matched fine-tuning steps, length 128, ODE-32, 256 generations,
noise scale 2, SC-CFG 3. WFF training uses heterogeneous local clocks on 50%
of examples. The historical spec reports the `kd_cr` pair; the baseline
follow-up below is read directly from the formal JSON and remains single-seed.

| Start family | Training | Sampler | PPL | Delta vs own standard | D1 | D2 | Rep-4 |
|---|---|---|---:|---:|---:|---:|---:|
| ELF base | sync control | standard | 279.5 | 0.0 | .4142 | .8659 | .0091 |
| ELF base | sync control | LTR d=.10 | 294.1 | +14.6 | .4153 | .8683 | .0074 |
| ELF base | sync control | LTR d=.20 | 348.5 | +69.0 | .4269 | .8754 | .0062 |
| ELF base | sync control | RTL d=.20 | 286.3 | +6.8 | .4257 | .8684 | .0064 |
| ELF base | WFF-trained | standard | 283.1 | 0.0 | .4149 | .8661 | .0085 |
| ELF base | WFF-trained | LTR d=.10 | 286.4 | +3.2 | .4128 | .8647 | .0080 |
| ELF base | WFF-trained | LTR d=.20 | 341.8 | +58.6 | .4266 | .8730 | .0065 |
| ELF base | WFF-trained | RTL d=.20 | 290.7 | +7.6 | .4251 | .8688 | .0062 |
| `kd_cr` | sync control | standard | 1086.7 | 0.0 | .4084 | .9440 | .0001 |
| `kd_cr` | sync control | LTR d=.10 | 1053.8 | -32.9 | .4040 | .9374 | .0014 |
| `kd_cr` | sync control | LTR d=.20 | 979.8 | -106.9 | .3828 | .9150 | .0051 |
| `kd_cr` | sync control | RTL d=.20 | 1002.3 | -84.4 | .3938 | .9264 | .0014 |
| `kd_cr` | WFF-trained | standard | 1114.0 | 0.0 | .4253 | .9488 | .0001 |
| `kd_cr` | WFF-trained | LTR d=.10 | 1126.6 | +12.6 | .4215 | .9415 | .0007 |
| `kd_cr` | WFF-trained | LTR d=.20 | 1031.7 | -82.3 | .3984 | .9208 | .0054 |
| `kd_cr` | WFF-trained | RTL d=.20 | 1019.2 | -94.8 | .4036 | .9309 | .0015 |

| Start family | LTR `.10` interaction | LTR `.20` interaction | RTL `.20` interaction | EMA local-time gate |
|---|---:|---:|---:|---:|
| ELF base | -11.4 | -10.4 | +0.7 | `-4.85e-5` |
| `kd_cr` | +45.5 | +24.6 | -10.4 | `-1.41e-5` |

The baseline pair contains a small LTR-specific adaptation signal, but no WFF
sampler beats its own standard sampler and the gate remains effectively zero.
Degeneration, word-count, unigram-collapse, and conditioned metrics were not
computed by this runner.

### 5.4 Deep Native Multi-Time ELF v2 (EXP-72)

Protocol: matched 500-step Control and LTR-curriculum fine-tuning from ELF
base; length 128, ODE-32, native noise 2, SC-CFG 3, `n=64`, seed 42. All wave
samplers include a final synchronous refinement region. `MaxShare` is mean
maximum word fraction and `Unique` is mean unique-word ratio.

| Training | Sampler | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | Unique | U-collapse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | Standard | 277.1 | .5000 | .8939 | .0107 | .000 | 77.0 | .0773 | .6870 | .000 |
| Control | LTR d=.05 | 297.3 | .5020 | .8961 | .0118 | .031 | 77.4 | .0743 | .6941 | .000 |
| Control | LTR d=.10 | 310.4 | .5005 | .8953 | .0109 | .016 | 76.8 | .0750 | .6988 | .000 |
| Control | LTR d=.15 | 322.8 | .5119 | .8977 | .0104 | .016 | 76.4 | .0721 | .6976 | .000 |
| Control | RTL d=.10 | 275.3 | .4999 | .8922 | .0129 | .016 | 76.8 | .0752 | .6815 | .000 |
| Control | Random d=.10 | 297.7 | .5028 | .8936 | .0141 | .000 | 77.2 | .0761 | .6939 | .000 |
| LTR-trained | Standard | 279.9 | .5005 | .8957 | .0141 | .000 | 77.3 | .0753 | .6898 | .000 |
| LTR-trained | LTR d=.05 | 308.6 | .5010 | .8961 | .0109 | .016 | 77.2 | .0740 | .6930 | .000 |
| LTR-trained | LTR d=.10 | 333.7 | .5057 | .8942 | .0105 | .016 | 76.8 | .0756 | .6958 | .000 |
| LTR-trained | LTR d=.15 | 343.7 | .5195 | .9029 | .0093 | .016 | 76.5 | .0752 | .7036 | .000 |
| LTR-trained | RTL d=.10 | 274.3 | .5012 | .8908 | .0169 | .016 | 76.7 | .0776 | .6790 | .000 |
| LTR-trained | Random d=.10 | 290.1 | .4990 | .8930 | .0153 | .000 | 77.3 | .0774 | .6840 | .000 |

| Training | Mean EMA local scale | `S_tau` LTR | `S_tau` RTL | LTR/RTL velocity cosine | LTR `.10` minus Standard |
|---|---:|---:|---:|---:|---:|
| Control | .01000 | 101.887 | 101.981 | 1.0000 | +33.3 |
| LTR-trained | .01003 | 101.881 | 101.977 | 1.0000 | +53.8 |

The causal interaction is `+20.5` PPL, opposite the intended direction. The
clock diagnostics remain indistinguishable from Control, so this is a failed
clock-learning gate rather than a clean learned-wave/exposure-bias result.
Training stops at step 500. EXP-73 passed a one-step implementation smoke test
but its formal arms are not launched because this prerequisite failed.

### 5.5 Frozen Clock-Adapter Bootstrapping (EXP-76)

The backbone is frozen and only twelve layerwise local-time projections/scales
are trained against held-out teacher-wave velocity fields.

| Stage | Mean held-out velocity MSE | Fixed-state LTR/RTL cosine | Clock delta | Mean scale | Standard PPL | LTR `.10` | RTL `.10` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen initialization | .05037 | .97365 | .5276 | .0100 | — | — | — |
| 200 steps | .02849 | .96485 | .6025 | .0637 | 265.2 | 319.0 | 330.5 |
| 500 cumulative steps | .02733 | .96349 | .6133 | .0824 | 265.2 | 329.0 | 385.0 |

The separate generation evaluator measures `S_tau=115.0/118.8` and velocity
cosine `.9922/.9911` at 200/500 steps, versus about `101.9` and `1.000` in
EXP-72. Optimization was therefore a real part of the unused-clock problem.
However, stronger clock response does not yield better wave generation.

### 5.6 Asynchronous Block Transition Distillation (EXP-77)

All arms start from the 500-step bootstrapped adapter and train for 200 steps.
Unlike synchronous consistency distillation, one active block is supervised at
its own local time inside a staggered sequence. The inference sampler uses 31
fill/drain calls plus eight of those calls as a final synchronized region.

| Training | Standard-32 | Standard-64 | Block LTR | Block RTL | Block random | LTR/RTL cosine |
|---|---:|---:|---:|---:|---:|---:|
| Sync transition | 288.0 | 109.2 | 3849.9 | 3431.9 | 3515.0 | .9907 |
| Off-policy LTR | 282.9 | 107.4 | 3911.8 | 3528.0 | 3398.5 | .9906 |
| On-policy LTR | 287.2 | 99.8 | 3658.0 | 3477.3 | 3435.6 | .9906 |
| Off-policy RTL | 272.9 | 106.7 | 3886.6 | 3566.9 | 3420.3 | .9905 |

PPL is paired at `n=64`, seed 42. Block outputs have D2 `.991--.996`, Rep-4
near zero, and PPL in the thousands: these are diverse incoherent outputs.
On-policy exposure repair is far too small and there is no LTR advantage.
Because the clock is functional and Standard quality is retained, the result
specifically rejects composition of the tested block-local transition method.

## 6. Inference methods and full quality panels

### 6.1 Pipeline protocol factorization (EXP-61)

Pipeline's historical gain is controlled by initial-noise scale, not SC-CFG.

| Noise scale | SC-CFG | `n` | Standard PPL | Pipeline PPL | Delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 64 | 309.59 | 188.78 | -120.82 |
| 2 | 1 | 64 | 1170.92 | 1185.99 | +15.08 |
| 1 | 3 | 64 | 306.32 | 192.64 | -113.68 |
| 2 | 3 | 64 | 1148.68 | 1197.70 | +49.02 |
| 1 | 1 | 256 | 338.05 | 196.27 | -141.78 |
| 2 | 1 | 256 | 1070.67 | 1244.42 | +173.75 |
| 1 | 3 | 256 | 340.33 | 195.47 | -144.86 |
| 2 | 3 | 256 | 1054.08 | 1251.46 | +197.38 |

The complete `n=64`, noise-1/SC-CFG-1 legacy audit was: standard
`PPL=309.59, D1=.317, D2=.851, Rep-4=.014, Deg=.250`; Pipeline
`PPL=188.78, D1=.435, D2=.896, Rep-4=.000, Deg=.547`. Thus even the favorable
legacy PPL direction has a worse degeneration rate.

### 6.2 Unified length-128 ODE panel (EXP-64)

Protocol: ODE-32, length 128, 256 unconditional + 128 conditioned, noise scale
2, SC-CFG 3. The paper-facing headline is the three-generation-seed mean:

| Checkpoint/method | PPL (delta) | D1 | D2 | Deg. | Cond. R-L | Calls |
|---|---:|---:|---:|---:|---:|---:|
| ELF base / standard | 273 (0) | .419 | .864 | .020 | .087 | 32 |
| ELF base / local-clock LTR | 2253 (+1979) | .537 | .882 | .038 | .076 | 32 |
| ELF base / hard commit | 238 (-35) | .416 | .861 | .020 | .087 | 33 |
| Broad-KD / standard | 1371 (0) | .459 | .950 | .211 | .060 | 32 |
| Broad-KD / two-pass prefix | 1343 (-27) | .430 | .949 | .212 | .055 | 64 |
| Broad-KD / hard commit | 677 (-694) | .363 | .847 | .085 | .096 | 33 |
| Broad-KD / Pipeline | 1557 (+186) | .467 | .959 | .086 | .026 | 31 |
| Commit-KD / standard | 989 (0) | .443 | .963 | .296 | .064 | 32 |
| Commit-KD / hard commit | 727 (-262) | .423 | .911 | .112 | .088 | 33 |
| Commit-KD / Pipeline | 1234 (+246) | .454 | .961 | .112 | .027 | 31 |

PPL deltas across seeds 42/123/456: ELF hard commit `-35.4 +/- 1.8`, Broad-KD
hard commit `-693.9 +/- 20.0`, Commit-KD hard commit `-261.5 +/- 13.2`,
Broad-KD Pipeline `+186.3 +/- 28.4`, and Commit-KD Pipeline
`+245.9 +/- 42.1`.

The following tables contain every quality field in the seed-42 JSON.

#### Unconditional, seed 42

| Checkpoint | Method | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | UniqueRatio | U-collapse | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | standard | 289.7 | .4154 | .8647 | .0089 | .0039 | 73.5 | .0782 | .6991 | .0078 | — |
| ELF base | local-clock LTR | 2347.4 | .5422 | .8910 | .0166 | .0156 | 69.9 | .0820 | .8470 | .0703 | — |
| ELF base | hard commit | 252.7 | .4129 | .8617 | .0079 | .0039 | 73.1 | .0795 | .6946 | .0039 | .9618 |
| Broad-KD | standard | 1360.2 | .4614 | .9440 | .0066 | .2188 | 82.3 | .0554 | .8738 | .0156 | — |
| Broad-KD | two-pass prefix | 1346.0 | .4327 | .9507 | .0065 | .2148 | 86.1 | .0588 | .8747 | .0195 | .5000 |
| Broad-KD | hard commit | 676.1 | .3654 | .8481 | .0028 | .0938 | 77.3 | .0932 | .7350 | .0625 | .5658 |
| Broad-KD | Pipeline | 1514.8 | .4671 | .9567 | .0014 | .0977 | 85.5 | .0412 | .9066 | .0039 | — |
| Commit-KD | standard | 1054.1 | .4452 | .9650 | .0001 | .2891 | 88.0 | .0417 | .8890 | .0000 | — |
| Commit-KD | hard commit | 779.5 | .4268 | .9128 | .0012 | .0938 | 84.3 | .0585 | .8052 | .0000 | .6861 |
| Commit-KD | Pipeline | 1251.5 | .4523 | .9600 | .0001 | .0859 | 90.9 | .0394 | .9053 | .0000 | — |

#### Conditioned suffix, seed 42

| Checkpoint | Method | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | UniqueRatio | U-collapse | R-L | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | standard | 481.4 | .4422 | .8946 | .0108 | .0234 | 44.0 | .0746 | .7882 | .0000 | .0857 | — |
| ELF base | local-clock LTR | 2675.6 | .5739 | .8784 | .0073 | .0000 | 44.7 | .0701 | .8711 | .0000 | .0775 | — |
| ELF base | hard commit | 426.5 | .4378 | .8902 | .0109 | .0234 | 44.0 | .0765 | .7809 | .0000 | .0864 | .9622 |
| Broad-KD | standard | 2075.9 | .5141 | .9717 | .0000 | .0781 | 47.3 | .0510 | .9275 | .0000 | .0576 | — |
| Broad-KD | two-pass prefix | 2090.1 | .5157 | .9747 | .0000 | .0938 | 47.9 | .0510 | .9256 | .0000 | .0522 | .5000 |
| Broad-KD | hard commit | 841.6 | .3788 | .8922 | .0004 | .0234 | 45.7 | .0815 | .8137 | .0078 | .0972 | .6252 |
| Broad-KD | Pipeline | 2187.0 | .5982 | .9937 | .0000 | .1875 | 47.2 | .0429 | .9606 | .0000 | .0284 | — |
| Commit-KD | standard | 1964.8 | .5108 | .9777 | .0000 | .0781 | 48.3 | .0514 | .9250 | .0000 | .0616 | — |
| Commit-KD | hard commit | 888.6 | .4446 | .9294 | .0002 | .0312 | 46.4 | .0716 | .8474 | .0000 | .0870 | .7577 |
| Commit-KD | Pipeline | 1904.2 | .5910 | .9963 | .0000 | .1328 | 51.3 | .0376 | .9657 | .0000 | .0260 | — |

### 6.3 Native-length deterministic ODE panel (EXP-65/66)

Protocol: ODE-32, length 1024, 256 unconditional + 128 conditioned, noise
scale 2, SC-CFG 3. All available JSON metrics are shown. Standard decoding
uses 32 denoiser calls and hard commitment uses 33 in both splits.

#### Unconditional

| Checkpoint | Method | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | UniqueRatio | U-collapse | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | standard | 127.8 | .1743 | .7034 | .0013 | .0000 | 731.1 | .0564 | .5092 | .0000 | — |
| ELF base | hard commit | 118.1 | .1772 | .6994 | .0013 | .0000 | 731.1 | .0563 | .5092 | .0000 | .9682 |
| Broad-KD | standard | 105.9 | .2270 | .7475 | .1117 | .2891 | 560.1 | .2557 | .4657 | .3203 | — |
| Broad-KD | hard commit | 439.7 | .2226 | .7818 | .0053 | .0195 | 718.6 | .0516 | .5946 | .0156 | .7576 |
| Commit-KD | standard | 271.1 | .2322 | .7866 | .0712 | .0977 | 716.2 | .1123 | .6041 | .0820 | — |
| Commit-KD | hard commit | 433.2 | .2364 | .7950 | .0076 | .0273 | 720.1 | .0585 | .5987 | .0234 | .8123 |
| Control | standard | 130.1 | .1697 | .6993 | .0012 | .0000 | 741.3 | .0566 | .5112 | .0000 | — |
| Control | hard commit | 120.8 | .1734 | .6947 | .0011 | .0000 | 741.1 | .0567 | .5104 | .0000 | .9607 |
| Early-KD | standard | 120.8 | .1682 | .6986 | .0016 | .0000 | 742.7 | .0539 | .4932 | .0000 | — |
| Early-KD | hard commit | 111.3 | .1717 | .6928 | .0016 | .0000 | 742.4 | .0540 | .4937 | .0000 | .9621 |
| Control seed 7 | standard | 130.2 | .1699 | .7007 | .0013 | .0000 | 741.3 | .0564 | .5106 | .0000 | — |
| Control seed 7 | hard commit | 120.9 | .1730 | .6962 | .0012 | .0000 | 741.1 | .0564 | .5109 | .0000 | .9612 |
| Early-KD seed 7 | standard | 127.9 | .1682 | .6920 | .0018 | .0000 | 745.2 | .0534 | .4945 | .0000 | — |
| Early-KD seed 7 | hard commit | 117.4 | .1711 | .6869 | .0017 | .0000 | 745.0 | .0533 | .4948 | .0000 | .9636 |

#### Conditioned suffix

| Checkpoint | Method | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | UniqueRatio | U-collapse | R-L | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | standard | 247.1 | .2668 | .8011 | .0048 | .0000 | 366.0 | .0541 | .5752 | .0000 | .1052 | — |
| ELF base | hard commit | 218.4 | .2671 | .7946 | .0045 | .0000 | 366.0 | .0547 | .5775 | .0000 | .1061 | .9447 |
| Broad-KD | standard | 456.8 | .2517 | .8028 | .0239 | .0859 | 385.1 | .0606 | .5924 | .0469 | .0975 | — |
| Broad-KD | hard commit | 540.9 | .2561 | .8268 | .0012 | .0312 | 385.4 | .0446 | .6061 | .0000 | .1042 | .7455 |
| Commit-KD | standard | 15.9 | .0822 | .2304 | .5154 | .7969 | 323.5 | .3921 | .1399 | .8203 | .0531 | — |
| Commit-KD | hard commit | 132.5 | .1660 | .5606 | .1344 | .3125 | 341.4 | .2079 | .3595 | .3438 | .0973 | .5293 |
| Control | standard | 257.4 | .2635 | .8047 | .0031 | .0000 | 369.9 | .0536 | .5855 | .0000 | .1045 | — |
| Control | hard commit | 229.9 | .2647 | .7998 | .0030 | .0000 | 369.7 | .0546 | .5861 | .0000 | .1056 | .9348 |
| Early-KD | standard | 253.4 | .2504 | .7914 | .0064 | .0000 | 372.8 | .0518 | .5599 | .0000 | .1031 | — |
| Early-KD | hard commit | 229.3 | .2524 | .7843 | .0058 | .0000 | 372.5 | .0522 | .5626 | .0000 | .1048 | .9395 |
| Control seed 7 | standard | 257.7 | .2613 | .8069 | .0030 | .0000 | 371.0 | .0532 | .5838 | .0000 | .1039 | — |
| Control seed 7 | hard commit | 229.4 | .2624 | .7993 | .0030 | .0000 | 370.5 | .0541 | .5834 | .0000 | .1059 | .9351 |
| Early-KD seed 7 | standard | 266.7 | .2498 | .7879 | .0063 | .0000 | 373.1 | .0523 | .5618 | .0000 | .1036 | — |
| Early-KD seed 7 | hard commit | 238.1 | .2489 | .7810 | .0059 | .0000 | 373.1 | .0528 | .5623 | .0000 | .1052 | .9400 |

The old Broad-KD/Commit-KD low standard PPL values are metric gaming through
repetition. On the corrected Control/Early-KD checkpoints, hard commitment
gives a clean `9.3--10.5` unconditional PPL improvement without degeneration.
Early-KD is positive for both training seeds but is not robust on conditioned
PPL. The KD-by-commit interaction is weak (`-0.2` and `-1.2` PPL).

### 6.4 Native stochastic SDE fidelity (EXP-68)

Protocol: native SDE-32 with logit-normal grid, length 1024, 256 unconditional
+ 128 conditioned, paired per-step noise, SDE gamma 1.5. The table includes all
JSON fields.

#### Unconditional

| Checkpoint | Method | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | UniqueRatio | U-collapse | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | standard | 30.7 | .0927 | .5065 | .0165 | .0000 | 800.5 | .0611 | .3771 | .0000 | — |
| ELF base | hard commit | 30.4 | .0936 | .5072 | .0163 | .0000 | 800.5 | .0611 | .3792 | .0000 | .9919 |
| Control | standard | 30.0 | .0840 | .4884 | .0147 | .0000 | 815.5 | .0595 | .3689 | .0000 | — |
| Control | hard commit | 29.8 | .0852 | .4887 | .0143 | .0000 | 815.6 | .0595 | .3716 | .0000 | .9903 |
| Early-KD | standard | 27.6 | .0801 | .4758 | .0180 | .0000 | 818.8 | .0563 | .3498 | .0000 | — |
| Early-KD | hard commit | 27.5 | .0813 | .4764 | .0176 | .0000 | 818.9 | .0563 | .3529 | .0000 | .9908 |
| Control seed 7 | standard | 30.1 | .0841 | .4893 | .0145 | .0000 | 814.3 | .0590 | .3698 | .0000 | — |
| Control seed 7 | hard commit | 29.8 | .0857 | .4901 | .0141 | .0000 | 814.4 | .0590 | .3722 | .0000 | .9901 |
| Early-KD seed 7 | standard | 27.6 | .0810 | .4760 | .0182 | .0000 | 818.9 | .0532 | .3530 | .0000 | — |
| Early-KD seed 7 | hard commit | 27.3 | .0819 | .4766 | .0178 | .0000 | 818.8 | .0532 | .3554 | .0000 | .9913 |

#### Conditioned suffix

| Checkpoint | Method | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | UniqueRatio | U-collapse | R-L | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | standard | 48.2 | .1800 | .6328 | .0302 | .0078 | 390.1 | .0814 | .4549 | .0078 | .1123 | — |
| ELF base | hard commit | 48.3 | .1837 | .6374 | .0284 | .0078 | 390.2 | .0809 | .4604 | .0078 | .1124 | .9740 |
| Control | standard | 50.9 | .1813 | .6430 | .0229 | .0078 | 394.7 | .0811 | .4700 | .0078 | .1121 | — |
| Control | hard commit | 51.7 | .1851 | .6459 | .0221 | .0078 | 394.5 | .0807 | .4750 | .0078 | .1124 | .9652 |
| Early-KD | standard | 45.5 | .1667 | .6208 | .0391 | .0078 | 399.5 | .0748 | .4389 | .0078 | .1107 | — |
| Early-KD | hard commit | 46.1 | .1693 | .6224 | .0378 | .0078 | 399.5 | .0749 | .4444 | .0078 | .1108 | .9746 |
| Control seed 7 | standard | 50.9 | .1808 | .6422 | .0235 | .0078 | 395.5 | .0803 | .4676 | .0078 | .1116 | — |
| Control seed 7 | hard commit | 51.7 | .1857 | .6440 | .0226 | .0078 | 395.3 | .0800 | .4727 | .0078 | .1113 | .9658 |
| Early-KD seed 7 | standard | 47.0 | .1646 | .6130 | .0399 | .0078 | 397.6 | .0763 | .4387 | .0078 | .1112 | — |
| Early-KD seed 7 | hard commit | 47.5 | .1676 | .6146 | .0389 | .0078 | 397.6 | .0763 | .4428 | .0078 | .1117 | .9748 |

Hard commitment keeps a tiny favorable unconditional sign (`-.10` to `-.29`
PPL) but slightly worsens conditioned PPL (`+.17` to `+.77`). The frozen
policy selects about 99% of unconditional positions at the first crossing, so
the ODE effect magnitude does not survive native SDE.

### 6.5 Native-SDE anchor-density calibration (EXP-69)

The intervention-free calibration shows how quickly confidence saturates.
All recorded calibration statistics are included below.

| SDE step | Mean time | Time SD | Conf. q10 | q25 | q50 | q75 | q90 | q95 | q99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | .139 | .033 | .065 | .118 | .261 | .613 | .958 | .997 | 1.000 |
| 8 | .197 | .024 | .284 | .557 | .951 | 1.000 | 1.000 | 1.000 | 1.000 |
| 12 | .239 | .024 | .605 | .960 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16 | .303 | .037 | .950 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 20 | .341 | .023 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 24 | .425 | .039 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 28 | .525 | .051 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

| SDE step | Anchor >=.60 | >=.70 | >=.80 | >=.90 | >=.95 | >=.99 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | .256 | .214 | .174 | .131 | .105 | .068 |
| 8 | .726 | .670 | .614 | .549 | .501 | .422 |
| 12 | .902 | .869 | .836 | .792 | .760 | .699 |
| 16 | .966 | .953 | .938 | .917 | .900 | .869 |
| 20 | .987 | .980 | .973 | .964 | .955 | .937 |
| 24 | .994 | .990 | .985 | .980 | .975 | .963 |
| 28 | .996 | .993 | .990 | .986 | .981 | .973 |

Implementation note: the calibration script formed percentile labels with
integer truncation, so the raw JSON keys `q89` and `q94` are the requested
90th and 95th percentiles. The values above use their intended percentile
names; the numeric estimates are unchanged.

Quality screen: 64 unconditional + 32 conditioned, length 1024. Standard
outputs are identical across the three independent paired jobs.

#### Unconditional

| Cell | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | UniqueRatio | U-collapse | Commit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 29.3 | .1558 | .6247 | .0162 | .0000 | 800.8 | .0601 | .3793 | .0000 | — |
| Step 4 / `.60` | 49.1 | .1850 | .6741 | .0108 | .0000 | 774.3 | .0618 | .3971 | .0000 | .2522 |
| Step 8 / `.95` | 43.5 | .1720 | .6524 | .0066 | .0000 | 787.2 | .0626 | .3932 | .0000 | .5241 |
| Step 12 / `.95` | 36.8 | .1659 | .6377 | .0110 | .0000 | 796.5 | .0611 | .3891 | .0000 | .7850 |

#### Conditioned suffix

| Cell | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | UniqueRatio | U-collapse | R-L | Commit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 45.9 | .2725 | .7244 | .0338 | .0000 | 387.0 | .0775 | .4533 | .0000 | .1158 | — |
| Step 4 / `.60` | 68.2 | .2620 | .6885 | .0401 | .0000 | 389.9 | .1007 | .4294 | .0000 | .1151 | .2333 |
| Step 8 / `.95` | 57.0 | .2722 | .7072 | .0341 | .0000 | 380.9 | .0933 | .4449 | .0312 | .1174 | .2996 |
| Step 12 / `.95` | 60.7 | .2782 | .7325 | .0222 | .0000 | 382.8 | .0854 | .4560 | .0000 | .1171 | .5238 |

Every early native-SDE cell worsens unconditional and conditioned PPL. Late
commitment is saturated and inert; early commitment leaves an unresolved set
but damages coherence. The matched shuffled-anchor SDE audit was therefore
stopped at the pre-registered quality gate.

### 6.6 Pipeline clock/state factorization (EXP-70)

Screen protocol: native ODE, length 128, `n=32`, seed 42. `E/C/K` denote ELF
base, continued-training Control, and corrected Early-KD. The table includes
every generated-quality field; `Calls` is denoiser calls per trajectory.

| Model | Sampler | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | Unique | U-collapse | Calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | Standard | 296.5 | .5438 | .9026 | .0160 | .000 | 79.6 | .0748 | .6886 | .000 | 32 |
| E | Shared clock | 1570.4 | .6650 | .9622 | .0045 | .000 | 87.9 | .0500 | .8273 | .000 | 31 |
| E | True local clock | 1778.1 | .6647 | .9741 | .0004 | .000 | 84.3 | .0490 | .8461 | .000 | 256 |
| E | Local + refine 4 | 1671.3 | .6704 | .9725 | .0004 | .000 | 84.1 | .0502 | .8352 | .000 | 196 |
| E | Local + refine 8 | 1506.1 | .6702 | .9685 | .0017 | .000 | 83.5 | .0480 | .8397 | .000 | 136 |
| E | Local RTL | 2035.1 | .6741 | .9730 | .0013 | .000 | 85.4 | .0415 | .8590 | .000 | 256 |
| E | Local random | 3359.3 | .7103 | .9847 | .0000 | .000 | 89.1 | .0357 | .8716 | .000 | 256 |
| C | Standard | 272.9 | .5396 | .9063 | .0187 | .031 | 82.0 | .0667 | .7037 | .000 | 32 |
| C | Shared clock | 1279.3 | .6565 | .9612 | .0031 | .000 | 87.9 | .0512 | .8318 | .000 | 31 |
| C | True local clock | 1641.0 | .6658 | .9697 | .0007 | .000 | 84.4 | .0492 | .8486 | .000 | 256 |
| C | Local + refine 4 | 1591.1 | .6706 | .9717 | .0007 | .000 | 83.9 | .0480 | .8466 | .000 | 196 |
| C | Local + refine 8 | 1469.1 | .6585 | .9714 | .0005 | .000 | 83.0 | .0500 | .8445 | .000 | 136 |
| C | Local RTL | 1756.1 | .6709 | .9756 | .0007 | .000 | 85.7 | .0443 | .8558 | .000 | 256 |
| C | Local random | 3272.8 | .7125 | .9843 | .0000 | .000 | 88.7 | .0341 | .8846 | .000 | 256 |
| K | Standard | 209.7 | .5285 | .9107 | .0162 | .000 | 79.4 | .0762 | .6671 | .000 | 32 |
| K | Shared clock | 1215.9 | .6549 | .9666 | .0023 | .000 | 87.0 | .0513 | .8221 | .000 | 31 |
| K | True local clock | 1204.4 | .6290 | .9640 | .0013 | .000 | 85.2 | .0497 | .8230 | .000 | 256 |
| K | Local + refine 4 | 1175.8 | .6256 | .9617 | .0032 | .000 | 84.1 | .0510 | .8198 | .000 | 196 |
| K | Local + refine 8 | 1153.2 | .6282 | .9600 | .0026 | .000 | 84.6 | .0532 | .8223 | .000 | 136 |
| K | Local RTL | 1464.4 | .6540 | .9740 | .0009 | .000 | 85.3 | .0467 | .8468 | .000 | 256 |
| K | Local random | 2235.9 | .6887 | .9830 | .0000 | .000 | 85.6 | .0380 | .8653 | .000 | 256 |

| Model | `E_clock` | `MSE_clock` | `E_state` | `MSE_state` | `E_x_clock` | `E_x_state` | `KL_clock` |
|---|---:|---:|---:|---:|---:|---:|---:|
| E | .0497 | 1.9719 | .1998 | 2.9636 | .1584 | .4526 | 8.237 |
| C | .0588 | 1.9840 | .1968 | 2.8173 | .1737 | .4632 | 8.568 |
| K | .0585 | 1.7810 | .1812 | 2.7405 | .1717 | .4145 | 7.249 |

True local clocks do not repair Pipeline, even with 4--8 synchronous final
refinements and 4--8 times the standard compute. Mixed-state error is roughly
three to four times clock error in cosine distance. This closes the discrete
heterogeneous-state operator rather than motivating a schedule sweep.

### 6.7 Synchronized soft-anchor Pipeline (EXP-71)

Screen protocol matches EXP-70. Every soft arm uses 64 denoiser calls; the
confidence arm adds 32 lexical readouts. `Leader` is mean selected fraction.

| Model | Sampler | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | Unique | U-collapse | Leader |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | Standard-32 | 296.5 | .5438 | .9026 | .0160 | .000 | 79.6 | .0748 | .6886 | .000 | — |
| E | Standard-64 | 107.5 | .4928 | .8439 | .0341 | .063 | 73.9 | .0896 | .6219 | .031 | — |
| E | Two-forward none | 296.5 | .5438 | .9026 | .0160 | .000 | 79.6 | .0748 | .6886 | .000 | .000 |
| E | Two-forward all | 127.7 | .4967 | .8498 | .0355 | .063 | 75.7 | .0878 | .6129 | .031 | 1.000 |
| E | Soft LTR | 256.3 | .5391 | .8937 | .0177 | .000 | 80.4 | .0789 | .6773 | .000 | .546 |
| E | Soft RTL | 253.9 | .5471 | .9042 | .0179 | .000 | 78.9 | .0796 | .6820 | .000 | .546 |
| E | Soft random | 231.4 | .5356 | .8913 | .0213 | .000 | 79.5 | .0744 | .6813 | .000 | .546 |
| E | Soft confidence | 233.9 | .5280 | .8920 | .0234 | .000 | 80.3 | .0798 | .6697 | .000 | .546 |
| E | Soft shuffled | 1053.0 | .6836 | .9785 | .0003 | .031 | 89.6 | .0371 | .9099 | .000 | .546 |
| C | Standard-32 | 272.9 | .5396 | .9063 | .0187 | .031 | 82.0 | .0667 | .7037 | .000 | — |
| C | Standard-64 | 102.6 | .4845 | .8527 | .0414 | .063 | 80.6 | .0828 | .6158 | .000 | — |
| C | Two-forward none | 272.9 | .5396 | .9063 | .0187 | .031 | 82.0 | .0667 | .7037 | .000 | .000 |
| C | Two-forward all | 110.2 | .4920 | .8542 | .0382 | .063 | 79.7 | .0928 | .6129 | .000 | 1.000 |
| C | Soft LTR | 245.4 | .5312 | .9000 | .0197 | .000 | 82.6 | .0662 | .6788 | .000 | .546 |
| C | Soft RTL | 234.4 | .5404 | .9179 | .0148 | .031 | 82.1 | .0686 | .6912 | .000 | .546 |
| C | Soft random | 251.3 | .5415 | .9028 | .0221 | .031 | 82.0 | .0702 | .6838 | .000 | .546 |
| C | Soft confidence | 247.6 | .5369 | .9005 | .0219 | .000 | 81.7 | .0706 | .6808 | .000 | .546 |
| C | Soft shuffled | 994.6 | .6624 | .9702 | .0007 | .000 | 88.1 | .0418 | .8807 | .000 | .546 |
| K | Standard-32 | 209.7 | .5285 | .9107 | .0162 | .000 | 79.4 | .0762 | .6671 | .000 | — |
| K | Standard-64 | 82.9 | .4699 | .8352 | .0421 | .094 | 77.4 | .0907 | .5813 | .031 | — |
| K | Two-forward none | 209.7 | .5285 | .9107 | .0162 | .000 | 79.4 | .0762 | .6671 | .000 | .000 |
| K | Two-forward all | 97.0 | .4834 | .8535 | .0359 | .063 | 75.4 | .0877 | .5984 | .031 | 1.000 |
| K | Soft LTR | 206.8 | .5295 | .9077 | .0148 | .000 | 79.9 | .0793 | .6597 | .031 | .546 |
| K | Soft RTL | 203.5 | .5187 | .9008 | .0190 | .031 | 79.4 | .0767 | .6658 | .000 | .546 |
| K | Soft random | 187.6 | .5192 | .9022 | .0136 | .031 | 79.6 | .0795 | .6551 | .031 | .546 |
| K | Soft confidence | 199.8 | .5202 | .9042 | .0198 | .000 | 79.0 | .0769 | .6538 | .000 | .546 |
| K | Soft shuffled | 848.3 | .6317 | .9609 | .0003 | .000 | 87.3 | .0427 | .8826 | .000 | .546 |

Shuffling fresh leader content is catastrophic across all checkpoints, which
confirms a causal content effect. Yet no correct soft-leader arm approaches
ordinary ODE-64 at the same denoiser-call budget, and LTR never beats both RTL
and random. Retain this as mechanism evidence, not a decoding method.

### 6.8 Event-triggered persistent anchoring (EXP-74)

Protocol: ODE-32, native noise, length 128, paired `n=64`, seed 42. Every
anchor arm makes 32 denoiser calls and one or two lexical readouts. `Hard .60`
and `Hard .90` trigger once at `t=.40`; `Hard stable` requires agreement at
the `.30` and `.40` readouts. All selected states become persistent conditions.

| Checkpoint | Arm | Anchor fraction | PPL | D1 | D2 | Rep-4 | Deg. |
|---|---|---:|---:|---:|---:|---:|---:|
| ELF base | Standard-32 | — | 278.7 | .497 | .888 | .014 | .000 |
| ELF base | Soft stable | .604 | 281.7 | .499 | .888 | .012 | .000 |
| ELF base | Hard at `.30` | .886 | 227.2 | .487 | .886 | .010 | .000 |
| ELF base | Hard `.60` | .952 | 206.8 | .493 | .889 | .009 | .000 |
| ELF base | Hard `.90` | .880 | **205.3** | .492 | .887 | .012 | .000 |
| ELF base | Hard stable | .604 | 232.8 | .493 | .885 | .011 | .016 |
| Control | Standard-32 | — | 276.4 | .502 | .894 | .013 | .016 |
| Control | Soft stable | .602 | 279.9 | .501 | .894 | .013 | .016 |
| Control | Hard at `.30` | .872 | 243.8 | .483 | .898 | .009 | .016 |
| Control | Hard `.60` | .947 | 221.3 | .494 | .895 | .010 | .016 |
| Control | Hard `.90` | .870 | **215.8** | .495 | .892 | .010 | .016 |
| Control | Hard stable | .602 | 243.4 | .493 | .889 | .014 | .016 |
| Early-KD | Standard-32 | — | 199.8 | .484 | .892 | .012 | .000 |
| Early-KD | Soft stable | .640 | 203.7 | .483 | .891 | .012 | .000 |
| Early-KD | Hard at `.30` | .892 | 191.1 | .470 | .894 | .007 | .016 |
| Early-KD | Hard `.60` | .952 | **168.0** | .475 | .892 | .011 | .000 |
| Early-KD | Hard `.90` | .877 | 169.3 | .476 | .890 | .012 | .000 |
| Early-KD | Hard stable | .640 | 181.3 | .482 | .891 | .013 | .000 |

Short-lived self-conditioning memory is ineffective, while persistent anchors
improve all checkpoints and density controls. The effect strengthens from
`.30` to `.40`, consistent with waiting until after the transition rather than
forcing early confidence. The method remains below Standard-64 and requires
multi-seed, conditioned, and native-SDE validation.

### 6.9 Canonical predicted-clean context (EXP-75)

Protocol matches the EXP-70 screen. Only PPL is repeated here because the full
quality fields do not change the decision: canonical LTR has degeneration
`.062/.031/.125` for base/Control/Early-KD, while the raw local arms have zero.

| Checkpoint | Standard | Raw local Pipeline | Canonical LTR | Canonical + refine 8 | Canonical RTL | Shuffled context |
|---|---:|---:|---:|---:|---:|---:|
| ELF base | 296.5 | 1778.1 | 1418.8 | 1350.9 | 1098.9 | 2348.8 |
| Control | 272.9 | 1641.0 | 1230.5 | 1316.9 | 1127.0 | 2341.1 |
| Early-KD | 209.7 | 1204.4 | 950.9 | 912.9 | 627.9 | 1441.5 |

`E_canonical=.2051/.1988/.1906` versus
`E_raw=.1993/.1922/.1794`; canonical MSE is also unchanged or worse. Replacing
non-target latents with predicted-clean states provides useful content but
does not map heterogeneous positions into a shared vector-field coordinate.

### 6.10 Robust revisable commitment (EXP-78)

ODE protocol: length 128, uniform ODE-32, native noise 2, SC-CFG 3,
`n_uncond=128`, `n_cond=64`, seeds `{42,123,456}`. Values below are means over
the three seeds. `Calls+R` is denoiser calls plus extra lexical readouts.

#### Unconditional ODE generation — complete metric panel

| Checkpoint | Arm | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | Unique | U-collapse | Commit | Calls+R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | Standard | 285.4 | .4719 | .8842 | .0100 | .0208 | 73.0 | .0797 | .6898 | .0052 | .000 | 32+0 |
| ELF base | Hard-highconf | 212.2 | .4620 | .8788 | .0104 | .0234 | 72.5 | .0814 | .6830 | .0052 | .880 | 32+1 |
| ELF base | Hard-stable | 232.2 | .4635 | .8782 | .0107 | .0260 | 72.7 | .0811 | .6823 | .0052 | .613 | 32+2 |
| ELF base | Unlock-4 | **208.5** | .4558 | .8724 | .0124 | .0260 | 72.6 | .0825 | .6710 | .0052 | .880 | 32+1 |
| ELF base | Unlock-8 | 217.5 | .4603 | .8751 | .0120 | .0234 | 72.6 | .0824 | .6768 | .0052 | .880 | 32+1 |
| Control | Standard | 264.6 | .4511 | .8818 | .0108 | .0156 | 77.3 | .0744 | .6988 | .0052 | .000 | 32+0 |
| Control | Hard-highconf | 212.4 | .4471 | .8773 | .0105 | .0260 | 76.9 | .0762 | .6975 | .0104 | .873 | 32+1 |
| Control | Hard-stable | 231.6 | .4487 | .8774 | .0110 | .0234 | 77.1 | .0762 | .6951 | .0130 | .612 | 32+2 |
| Control | Unlock-4 | **203.5** | .4415 | .8713 | .0124 | .0260 | 77.0 | .0773 | .6851 | .0130 | .873 | 32+1 |
| Control | Unlock-8 | 213.0 | .4460 | .8740 | .0116 | .0260 | 77.0 | .0771 | .6899 | .0156 | .873 | 32+1 |
| Early-KD | Standard | 204.2 | .4448 | .8754 | .0115 | .0365 | 73.2 | .0835 | .6651 | .0130 | .000 | 32+0 |
| Early-KD | Hard-highconf | 170.1 | .4383 | .8717 | .0118 | .0312 | 72.7 | .0849 | .6648 | .0156 | .881 | 32+1 |
| Early-KD | Hard-stable | 180.2 | .4404 | .8717 | .0123 | .0365 | 72.9 | .0851 | .6635 | .0156 | .641 | 32+2 |
| Early-KD | Unlock-4 | **165.9** | .4317 | .8651 | .0146 | .0365 | 72.9 | .0863 | .6518 | .0182 | .881 | 32+1 |
| Early-KD | Unlock-8 | 172.3 | .4362 | .8679 | .0134 | .0391 | 72.8 | .0860 | .6555 | .0182 | .881 | 32+1 |

The favorable PPL sign holds for every arm in every seed. Unlock-4 is the best
mean PPL arm at all three checkpoints. Its small D1/D2/Unique reductions and
Rep-4 increases make the complete quality panel essential; degeneration does
not systematically worsen, but PPL is the clearest benefit.

#### Conditioned ODE continuation — complete metric panel

| Checkpoint | Arm | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | Unique | U-collapse | R-L | Prefix | Commit | Calls+R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | Standard | 509.6 | .4965 | .9135 | .0091 | .0104 | 44.8 | .0779 | .7859 | .0052 | .0927 | .9844 | .000 | 32+0 |
| ELF base | Hard-highconf | 380.0 | .4814 | .9056 | .0095 | .0104 | 44.7 | .0802 | .7773 | .0052 | .0933 | .9948 | .877 | 32+1 |
| ELF base | Hard-stable | 411.4 | .4850 | .9057 | .0111 | .0104 | 44.7 | .0796 | .7804 | .0052 | .0928 | .9896 | .621 | 32+2 |
| ELF base | Unlock-4 | **379.7** | .4778 | .9018 | .0122 | .0104 | 44.7 | .0800 | .7717 | .0052 | .0933 | .9688 | .877 | 32+1 |
| ELF base | Unlock-8 | 403.1 | .4836 | .9030 | .0117 | .0104 | 44.7 | .0797 | .7759 | .0052 | .0930 | .9740 | .877 | 32+1 |
| Control | Standard | 477.4 | .5049 | .9130 | .0074 | .0000 | 45.3 | .0772 | .7969 | .0052 | .0919 | .9948 | .000 | 32+0 |
| Control | Hard-highconf | 375.0 | .4878 | .9092 | .0059 | .0000 | 45.3 | .0785 | .7922 | .0052 | .0928 | .9896 | .867 | 32+1 |
| Control | Hard-stable | 403.6 | .4937 | .9085 | .0077 | .0000 | 45.3 | .0787 | .7911 | .0052 | .0926 | .9896 | .605 | 32+2 |
| Control | Unlock-4 | **362.0** | .4864 | .9038 | .0098 | .0000 | 45.3 | .0799 | .7827 | .0052 | .0926 | .9948 | .867 | 32+1 |
| Control | Unlock-8 | 377.3 | .4909 | .9057 | .0083 | .0000 | 45.3 | .0790 | .7867 | .0052 | .0929 | .9948 | .867 | 32+1 |
| Early-KD | Standard | 384.4 | .4813 | .9081 | .0088 | .0052 | 45.2 | .0784 | .7779 | .0000 | .0927 | 1.0000 | .000 | 32+0 |
| Early-KD | Hard-highconf | 309.3 | .4694 | .9035 | .0095 | .0000 | 45.1 | .0807 | .7718 | .0000 | .0936 | .9896 | .885 | 32+1 |
| Early-KD | Hard-stable | 329.7 | .4737 | .9042 | .0090 | .0000 | 45.2 | .0805 | .7746 | .0000 | .0934 | .9948 | .639 | 32+2 |
| Early-KD | Unlock-4 | **305.3** | .4679 | .8987 | .0109 | .0000 | 45.2 | .0810 | .7663 | .0000 | .0946 | 1.0000 | .885 | 32+1 |
| Early-KD | Unlock-8 | 312.5 | .4696 | .9004 | .0111 | .0000 | 45.2 | .0808 | .7697 | .0000 | .0942 | 1.0000 | .885 | 32+1 |

Conditioned PPL improves in all nine paired seed cells for Hard-highconf and
Unlock-4. ROUGE-L is unchanged or slightly higher; mean Unlock-4 deltas are
`+.0006/+.0007/+.0018`. Exact-prefix values are reported rather than assumed:
they are near one but not universally one in the ODE runner.

#### Native SDE fidelity boundary

Protocol: length 1024, native logit-normal SDE-32, gamma 1.5,
`n_uncond=128`, `n_cond=64`, seed 42. The complete text-quality metrics are
essentially unchanged; the compact table records the decision metrics.

| Checkpoint | Scope | Arm | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | Unique | U-collapse | R-L | Prefix | Commit | Calls+R |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF base | U | Standard | 31.253 | .1205 | .5713 | .0155 | .000 | 800.9 | .0602 | .3825 | .000 | — | — | .000 | 32+0 |
| ELF base | U | Hard-highconf | 30.889 | .1214 | .5717 | .0153 | .000 | 801.0 | .0602 | .3845 | .000 | — | — | .976 | 32+1 |
| ELF base | U | Unlock-4 | 31.049 | .1214 | .5715 | .0154 | .000 | 801.2 | .0603 | .3832 | .000 | — | — | .976 | 32+1 |
| ELF base | C | Standard | 47.467 | .2167 | .6753 | .0295 | .000 | 393.9 | .0808 | .4541 | .000 | .1176 | 1.0 | .000 | 32+0 |
| ELF base | C | Hard-highconf | 48.557 | .2202 | .6768 | .0283 | .000 | 393.5 | .0799 | .4577 | .000 | .1178 | 1.0 | .938 | 32+1 |
| ELF base | C | Unlock-4 | 48.017 | .2190 | .6757 | .0294 | .000 | 393.5 | .0810 | .4572 | .000 | .1179 | 1.0 | .938 | 32+1 |
| Control | U | Standard | 30.674 | .1129 | .5543 | .0141 | .000 | 814.3 | .0596 | .3754 | .000 | — | — | .000 | 32+0 |
| Control | U | Hard-highconf | 30.487 | .1143 | .5550 | .0139 | .000 | 814.3 | .0595 | .3779 | .000 | — | — | .971 | 32+1 |
| Control | U | Unlock-4 | 30.465 | .1138 | .5542 | .0141 | .000 | 814.6 | .0596 | .3764 | .000 | — | — | .971 | 32+1 |
| Control | C | Standard | 51.287 | .2194 | .6869 | .0222 | .000 | 398.5 | .0790 | .4662 | .000 | .1162 | 1.0 | .000 | 32+0 |
| Control | C | Hard-highconf | 51.387 | .2211 | .6892 | .0207 | .000 | 398.0 | .0788 | .4688 | .000 | .1171 | 1.0 | .925 | 32+1 |
| Control | C | Unlock-4 | 51.273 | .2198 | .6864 | .0223 | .000 | 397.9 | .0796 | .4674 | .000 | .1173 | 1.0 | .925 | 32+1 |
| Early-KD | U | Standard | 28.018 | .1059 | .5443 | .0187 | .000 | 818.1 | .0568 | .3529 | .000 | — | — | .000 | 32+0 |
| Early-KD | U | Hard-highconf | 27.819 | .1067 | .5450 | .0184 | .000 | 818.3 | .0567 | .3550 | .000 | — | — | .972 | 32+1 |
| Early-KD | U | Unlock-4 | 27.741 | .1062 | .5438 | .0187 | .000 | 818.4 | .0568 | .3535 | .000 | — | — | .972 | 32+1 |
| Early-KD | C | Standard | 46.041 | .2044 | .6573 | .0478 | .0156 | 402.0 | .0788 | .4313 | .0156 | .1148 | 1.0 | .000 | 32+0 |
| Early-KD | C | Hard-highconf | 46.035 | .2044 | .6560 | .0474 | .0000 | 403.1 | .0789 | .4335 | .0000 | .1149 | 1.0 | .933 | 32+1 |
| Early-KD | C | Unlock-4 | 45.337 | .2040 | .6549 | .0478 | .0156 | 402.1 | .0793 | .4321 | .0156 | .1151 | 1.0 | .933 | 32+1 |

Despite `93--98%` anchor coverage, all SDE PPL changes are only `0--1.1`.
Thus the ODE gain is not sampler-independent; SDE anchoring is saturated and
nearly inert.

#### Unlock-4 timing and actual revision

Protocol: ODE-32, length 128, paired `n=64`, seed 42. Timing uses each branch's
own endpoint. The selection mask comes from Unlock-4 and is reused to define
the paired scopes in Standard.

| Checkpoint | Scope | `Delta tau_first` | `Delta tau_stable` | `Delta N_rev` | Endpoint agreement | Anchor changed after release |
|---|---|---:|---:|---:|---:|---:|
| ELF base | selected | -.0140 | -.0141 | -.1820 | .844 | .100 |
| ELF base | unselected | -.0102 | -.0121 | -.1963 | .351 | — |
| Control | selected | -.0140 | -.0150 | -.1682 | .848 | .090 |
| Control | unselected | -.0077 | -.0098 | -.1124 | .374 | — |
| Early-KD | selected | -.0107 | -.0100 | -.1366 | .871 | .081 |
| Early-KD | unselected | +.0039 | +.0061 | -.0885 | .415 | — |

Unlock-4 is genuinely revisable: `8.1--10.0%` of selected positions finish at
a different token from the anchor read at `t=.40`, after release at
`t=.5625`. But unselected timing changes are small and not directionally
universal. The supported method claim is therefore temporary reliable
conditioning in deterministic ODE, not a demonstrated global coordination
transition.

### 6.11 Late-coupled block denoising (EXP-79)

Protocol: ELF base, two 128-token blocks, native ODE-32, noise 2, SC-CFG 3,
`n=128`, seed 42. Blocks are decoded separately around their own EOS before
the text is concatenated. `Boundary` is GPT-2-large PPL on the first 32 suffix
evaluator tokens conditioned on the decoded prefix.

| Arm | PPL | PPL A | PPL B | Boundary | D1 | D2 | Rep-4 | Deg. | Prefix rev. | Suffix rev. | Calls | Token-calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel-32 | **169.5** | 228.9 | 215.8 | 148.6 | .3567 | .8318 | .0068 | .0156 | — | — | 32 | 8192 |
| Parallel-60 | **76.3** | 102.8 | 102.9 | 67.1 | .3248 | .7852 | .0231 | .0234 | — | — | 60 | 15360 |
| Semi-AR-64 | 311.9 | 296.7 | 394.2 | 648.8 | .4086 | .8628 | .0131 | .0000 | — | — | 64 | 12288 |
| Late reencoded m24 | 300.6 | 285.0 | 382.1 | 602.9 | .4011 | .8618 | .0127 | .0078 | .0573 | .0828 | 56 | 11264 |
| Late reencoded m28 | 309.1 | 290.8 | 392.8 | 634.8 | .4093 | .8648 | .0113 | .0000 | .0370 | .0519 | 60 | 11776 |
| Late m28 freeze-A | 312.1 | 297.7 | 395.5 | 630.5 | .4097 | .8651 | .0113 | .0000 | .0000 | .0514 | 60 | 11776 |

The `n=8` smoke first passed exact native-runner agreement (`1.0`), zero
condition-restore error, and zero freeze-A prefix revision. In the decisive
panel, late coupling is only marginally better than Semi-AR and dramatically
worse than parallel decoding on full, suffix, and boundary PPL. Full joint
refinement changes `3.7%` of prefix tokens at m28 but improves full PPL by only
`3.0` relative to freeze-A; boundary PPL is instead `4.2` worse. The method is
stopped at P0 and is not promoted across representations, checkpoints, lengths,
or architectures.

## 7. Post-hoc asynchronous sampling and cross-architecture evidence

### 7.1 GS19 asynchronous schedule ablation

On ELF, synchronous versus LTR/RTL/fixed-random/confidence-adaptive local
clocks gives:

- `tau_stable`: `16.79 -> 19.8--21.7` for the four asynchronous arms;
- revisions: `5.57 -> 8.2--9.7`;
- generation PPL: `76.8 -> 144--442`;
- RTL degeneration: `.75`.

All four schedules fail all three pre-registered signals. On Plaid, all four
also fail, with PPL increasing by `3.0x--14.4x`. Confidence-adaptive is the
only Plaid arm with earlier nominal stable time (`21.73 -> 19.70`) and is also
the worst-quality arm, demonstrating that faster token stabilization alone is
not a quality metric.

### 7.2 Corrected LangFlow comparison

Nominal diffusion time is not comparable across ELF and LangFlow. The safe
cross-architecture record uses each model's own trajectory or explicitly
matched statistics.

| Experiment | Metric | Corrected LangFlow result | Interpretation |
|---|---|---|---|
| EXP-21v2 | Native/backbone/skip/probe top-1 | At `t=.85`: native `.561`, backbone `.000`, skip `.0765`, probe-h `.505`; at `t=1`: native `.988`, backbone `.000`, skip `.924`, probe-h `.944`. | Native decoding is skip-dominated; the backbone residual is a corrector, so the original final-hidden probe-gap comparison was asymmetric. |
| EXP-25v2 | Occurrence-level hazard model | After frequency control, function-token odds ratio is `.26--.73` before the cliff; frequency odds ratio is `2--7.5`. | Frequency, not POS class, is the main early-token driver. |
| EXP-26v2 | Spatial dependence | Moran's `I=.26` at `t=.745` (`z=22.54`, `p<.001`); committed-neighbor hazard OR `2.44 [2.19,2.76]`. | Spatial clustering exists, but this observational statistic alone does not prove propagation. |
| EXP-27v2 | OWT token frequency | Pearson `r=-.651` (`p=4.7e-24`); partial correlation controlling POS `-.638`. | High-frequency tokens stabilize earlier across architectures. |
| EXP-30v2 | Layer-wise probe | Best intermediate layer B10 is about `+2.6 pp` over final at `t=.85`; MLP does not beat linear. | Late representation is linearly readable, but skip asymmetry remains essential. |
| EXP-53 | Stable commitment | Never-stable `.0479`, mean `T_stable=.840`; cumulative stable fraction `.171/.537/.737/.873/.952` at `t=.806/.884/.922/.961/1.0`. | LangFlow stabilizes much later in its own clock than ELF KD, but direct nominal-time ratios are not a mechanism claim. |
| GS14 replication | True-trajectory consensus | `C_topic=.958 -> .985 -> .992`; lexical consensus contracts later. | Coarse-basin-before-exact-lexical ordering survives. |

### 7.3 Plaid cross-architecture scorecard (GS20)

| Test | ELF | Plaid | Decision |
|---|---|---|---|
| Endpoint-specificity collapse | Early ambiguity, then narrow collapse | Same pattern, within one checkpoint | Confirmed across architectures |
| Local velocity dynamics | Endpoint alignment early; collapse after token stability | Early alignment low/non-monotone; event order reverses | Boundary: stochastic finite differences are confounded |
| Rank/energy control | Top-k beats matched alternatives | Same | Confirmed |
| Residualized collective coupling | 13/16 checkpoints beat all nulls | 1/16 | Boundary: ancestral step noise likely dilutes increments |
| Async denoising | All fail, PPL `2--6x` worse | All fail, PPL `3--14.4x` worse | Confirmed negative result |

Endpoint-based/static conclusions replicate more reliably than adjacent-state
finite-difference conclusions. Plaid injects Gaussian noise at every ancestral
step, unlike deterministic ELF/LangFlow Euler paths.

## 8. Historical result audit

These experiments remain important because they explain why earlier stories
or methods were withdrawn.

| Family | Main result | Current use |
|---|---|---|
| EXP-01/04/05 | Oracle-state cliff is not a free-rollout cliff; head-only null is negligible; batch-shuffled prior was invalid. | Measurement motivation only; do not present oracle construction as a scientific reversal. |
| EXP-07v2 | Baseline linear-probe gap remains about +41 pp under document-level split; KD checkpoints have a negative gap. | Representation is recoverable before the native decoder exposes it. |
| EXP-08v2/25v2/27v2 | Function tokens appear earlier, but real frequency dominates after control; LangFlow frequency partial correlation is about `-.638`. | Say frequency/coarse lexical prior, not POS-driven coarse-to-fine semantics. |
| EXP-09/26 | Spatial clustering exists, but early near/far analyses had risk-set collapse and common-cause confounds. | Use GS13/18-B causal/residualized tests instead. |
| EXP-11v2/14v2 | KD checkpoints are more stable under corrected perturbation and have fewer flips than baseline. | Supporting stability evidence, not the central transition test. |
| EXP-31--37 | Diffusion-forcing/self-conditioning effects reverse across checkpoints and several settings degenerate. | Dead-end lineage; no universal DF method claim. |
| EXP-49--51 | Synthetic auxiliary-loss fine-tuning reduces its training loss but collapses generation. | Do not revive without real OWT training and a new design. |
| EXP-58/59 | Pipeline looked strong on a legacy custom path. | Superseded by EXP-61/64 native-protocol failure. |
| EXP-62 | Noisy-head KD early window reaches low PPL but fragmented pseudo-text; ODE-64 early PPL `53.5` with degeneration `.098`. | Objective mismatch/metric gaming; replaced by EXP-63. |
| EXP-63 | Corrected clean-teacher Early-KD passes two training seeds. | Retained training-time result with conditioned-quality boundary. |
| EXP-64/65 | Old KD checkpoints expose PPL/repetition conflict. | Justification for the complete metric panel. |
| EXP-68/69 | ODE hard-commit gain vanishes or reverses under native SDE calibration. | Explicit solver-specific boundary. |

## 9. Claims currently safe for paper/slides

1. Early pooled “global signal” is not evidence that the model has already
   formed a meaningful global sentence state.
2. Free rollout maintains multiple lexical futures and undergoes a relatively
   narrow endpoint-affinity collapse; exact lexical commitment is later and
   more fragile than coarse basin structure.
3. Other positions causally affect a target token. On deterministic ELF
   rollout, position-correct anchors accelerate and stabilize unresolved
   decisions, while matched shuffled anchors reverse the effect.
4. Corrected Early-KD improves unconditional ODE generation and commitment
   timing in two training seeds, but conditioned improvement is not robust.
5. Hard commitment is an ODE-specific intervention, not a sampler-independent
   method. A four-step lock preserves the replicated ODE gain and permits later
   revision, while native SDE either makes the intervention inert or loses
   coherence when commitment is forced earlier.
6. Pipeline, post-hoc local clocks, and the current gated WFF pilot are not
   positive methods.

## 10. Provenance

Primary specs:

- mechanism: `EXP-GS11`--`EXP-GS20`, `EXP-67`;
- WFF/local clocks: `EXP-60`, `EXP-GS19`;
- Pipeline: `EXP-61`, `EXP-64`;
- KD: `EXP-62`, `EXP-63`, `EXP-66`;
- hard commitment and sampler boundary: `EXP-64`--`EXP-69`, `EXP-74`, `EXP-78`.

Primary server result directories:

```text
results/exp60_wff_pilot/
results/exp64_unified_method_eval/
results/exp65_hard_commit_calibration/
results/exp67_hard_commit_mechanism/
results/exp68_native_sde_commit/
results/exp69_native_sde_anchor_calibration/
results/exp78_robust_revisable_commit/
```

When adding a new formal evaluation, append its protocol and complete metric
row here at the same time as updating its spec. Never overwrite a historical
row after changing tokenizer, degeneration rule, sampling schedule, or sample
bank; add a new protocol block instead.
