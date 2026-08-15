# CCLF Major Experimental Results

**Last updated:** 2026-08-14
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
- **Prompt PPL**: evaluator PPL on the generated suffix while conditioning on
  the original observed prompt.
- **Prompt gain**: `log PPL(y|shuffled(c)) - log PPL(y|c)`; positive means the
  suffix is more compatible with its true prompt than a mismatched prompt.
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
| Does late clock-aligned coupling help prompted continuation? | EXP-79/87/94 | Only relative to Block-SAR. At identical `11264` token-calls, Plaid Parallel-44 beats late raw/continuous by `23.91` C-PPL and about `62--64` boundary-PPL points; extra parallel denoising explains the apparent gain. |
| Does real prefix conditioning rescue asynchronous ELF methods? | EXP-80 | No. Soft anchors lose to Standard-64 and local/canonical waves lose to Standard-136 in both scopes. Unlock-4's same-call PPL gain replicates across two new OWT panels and Gutenberg, but prompt-gain improvement is not robust and diversity/repetition trade-offs remain. |
| Is Unlock-4 actually using the prompt more strongly? | EXP-81 | Not robustly. It lowers NLL in every suffix band, but the pooled full-suffix prompt-gain delta is only `+.0045 [-.0085,.0181]`. |
| What part of temporary anchoring matters? | EXP-82 | Correct position-content and coverage matter more than high confidence: random 50% anchors beat top-confidence anchors on PPL in all three panels, while shuffled content is catastrophic. |
| Does the temporary-anchor sign scale? | EXP-89 | Yes for PPL through length 1024 and prefix ratios `.25/.50/.75`; the unconditional effect shrinks with length and the D1/Rep-4 trade-off remains. |
| Is temporary anchoring portable beyond ELF? | EXP-90 | Conditionally, yes: random correct anchors improve C-PPL in 3/3 LangFlow and 3/3 Plaid seeds, while shuffled content is catastrophic. LangFlow U-PPL worsens slightly, and diversity/degeneration trade-offs remain architecture-dependent. |
| Can adaptive rollback fix that trade-off? | EXP-88 | Not yet. Shadow disagreement releases about one third and improves PPL further, but D1 falls again. |
| Can subset-conditioned flow training internalize the anchor effect? | EXP-91 | No in the 200-step pilot. Across three paired inference seeds, mean random-anchor interaction is `+1.5/+2.3` U/C PPL (unfavorable), prompt-gain interaction is `-.0072`, and C-degeneration interaction is `+.0078` on every seed. |
| Does conditional/on-policy subset training fix that mismatch? | EXP-92 | No with the current target. Conditional-oracle is non-Pareto; on-policy is unfavorable at `+9.46/+11.58` U/C PPL interaction, and a fixed `.25` weight still worsens C-PPL in 3/3 seeds and prompt gain in 3/3. |
| Is random anchoring already subset-optimal? | EXP-93 | No. Best-of-16 lowers C-PPL from mean-random `335.67/390.12` to `210.19/240.52` on two independent banks. The gap is real, but static, lookahead, and additive influence selectors do not reliably predict it. |
| Does Plaid also contain anchor-subset headroom? | EXP-99 | Yes. Best-of-16 improves C-PPL over mean random by `43--49%` on two disjoint banks at both `.50` and `.75` density, with all paired-NLL intervals excluding zero. |
| Can a non-additive set model predict that Plaid utility? | EXP-100 | No with the tested model. With 320 training trajectories, final pair accuracy remains chance; only 1/3 optimization seeds has favorable pooled NLL, its CI crosses zero, and a fixed-index null matches the gain. |
| Does per-trajectory Plaid trigger timing matter? | EXP-101 | Yes diagnostically: best-of-eight improves fixed step-14 C-PPL by `45--49%` on two banks. No for instantaneous adaptation: the discovery-frozen q10-confidence rule reverses on validation. |
| Does a native short-horizon response predict trigger utility? | EXP-102 | Yes for likelihood. Four-step unresolved entropy reduction reaches pairwise accuracy `.609/.614` and significantly lowers NLL on discovery/validation, but raw selection misses degeneration and prompt-gain gates. |
| Can conservative abstention turn that signal into a final method? | EXP-103 | Not yet. It passes calibration with only `5/64` switches, but on an untouched bank switches `2/64`; C-PPL changes `94.91 -> 94.30` and paired NLL CI ends at zero. |
| Can a temporary-anchor policy improve the complete Plaid panel? | EXP-95 | Yes, with a bounded claim. Early one-step 75% confidence anchors reduce U/C-PPL `135.43/110.39 -> 99.32/80.28`, improve mean D1 and degeneration, and revise at `.699`; D2 falls by `.0051`. Readout sham is exact and shuffled content is harmful. |
| Does corrected temporal KD work? | EXP-63/66 | Early-window KD improves unconditional ODE quality and timing in two training seeds; conditioned gains are not robust. |
| Does ELF hard commitment work? | EXP-64--69/74/78 | Yes as an ELF ODE-specific intervention. Three-seed and conditioned gains replicate, and a four-step lock is sufficient; ELF native-SDE effects remain negligible. EXP-90 separately tests related native temporary anchors on other architectures. |

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

### 6.11 Late-coupled block denoising (EXP-79; unconditional P0)

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

These values are an unconditional mechanism screen, not the decisive test of
the intended prompt-conditioned use case. The `n=8` smoke first passed exact native-runner agreement (`1.0`), zero
condition-restore error, and zero freeze-A prefix revision. In the decisive
panel, late coupling is only marginally better than Semi-AR and dramatically
worse than parallel decoding on full, suffix, and boundary PPL. Full joint
refinement changes `3.7%` of prefix tokens at m28 but improves full PPL by only
`3.0` relative to freeze-A; boundary PPL is instead `4.2` worse. This rejects
an unconditional-quality claim. A fixed 64-token Gutenberg prefix with a
192-token continuation is now the decisive P1; no broader promotion occurs
until prompt-conditioned PPL, ROUGE-L, and A-to-B boundary PPL are available.

#### Decisive conditional P1

Protocol: the same ELF-base arms and paired seed-42 noise, now on 128 fixed
Gutenberg examples with an observed 64-token prefix and a 192-token generated
continuation. `Prompt PPL` evaluates the continuation under GPT-2-large while
conditioning on the original prompt. `Boundary` evaluates the first 32 B
tokens given the original prompt and generated A. All latent prompt-clamp
errors are zero.

| Arm | Cont. PPL | Prompt PPL | R-L | Boundary | D1 | D2 | Rep-4 | Deg. | Decoded prefix | A rev. | B rev. | Calls | Token-calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel-32 | 238.3 | 252.8 | .1020 | 242.9 | .3478 | .8354 | .0102 | .0078 | .9922 | — | — | 32 | 8192 |
| Parallel-60 | **128.3** | **134.5** | **.1027** | **125.5** | .3205 | .7931 | .0261 | .0156 | 1.0000 | — | — | 60 | 15360 |
| Semi-AR-64 | 394.9 | 421.2 | .0968 | 553.1 | .3650 | .8561 | .0104 | .0000 | .9922 | — | — | 64 | 12288 |
| Late reencoded m24 | 393.6 | 419.0 | .0958 | 566.3 | .3653 | .8537 | .0105 | .0000 | .9922 | .0660 | .0817 | 56 | 11264 |
| Late reencoded m28 | 394.2 | 420.2 | .0967 | 568.3 | .3620 | .8530 | .0111 | .0000 | 1.0000 | .0453 | .0502 | 60 | 11776 |
| Late m28 freeze-A | 398.1 | 423.3 | .0967 | 566.8 | .3631 | .8533 | .0100 | .0000 | 1.0000 | .0000 | .0494 | 60 | 11776 |

Conditional generation confirms the negative result on the task the method is
intended to help. Late-28 is effectively tied with Semi-AR (`-1.0` prompt PPL,
`-.0001` ROUGE-L) and is much worse than Parallel-32/60 (`+167/+286` prompt
PPL and lower ROUGE-L). Joint revision improves prompt PPL by only `3.0`
relative to freeze-A, leaves ROUGE-L unchanged, and makes boundary PPL `1.5`
worse. EXP-79 is therefore stopped on conditional evidence, not on the earlier
unconditional screen.

#### Post-stop unconditional portability/representation sweep

These runs audit the already implemented grid after the conditional stop. They
do not contain prompt-conditioned metrics and therefore do not reopen the P1
decision. ELF reports one `n=128`, seed-42 run. LangFlow and Plaid entries are
means over paired `n=64` runs at seeds 42/123/456. Native-reference agreement
was `1.0` throughout; Plaid step noise was paired across arms.

ELF full representation grid:

| Arm | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | Unique | Collapse | A rev. | High/low-conf. rev. | Hybrid frac. | Calls+R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel-32 | 168.13 | .3564 | .8314 | .0071 | .0156 | 164.4 | .0666 | .6165 | .0000 | — | — | — | 32+0 |
| Semi-AR-64 | 310.65 | .4090 | .8624 | .0132 | .0000 | 152.2 | .0588 | .6416 | .0000 | — | — | — | 64+1 |
| Continuous m20 | 311.36 | .4064 | .8690 | .0082 | .0000 | 154.6 | .0589 | .6553 | .0000 | .0809 | .0495/.6730 | .951 | 52+1 |
| Continuous m24 | 313.14 | .4041 | .8657 | .0086 | .0000 | 154.8 | .0580 | .6590 | .0000 | .0599 | .0356/.5742 | .956 | 56+1 |
| Continuous m28 | 325.02 | .4059 | .8675 | .0083 | .0000 | 154.8 | .0584 | .6625 | .0000 | .0391 | .0180/.5265 | .960 | 60+1 |
| Continuous m30 | 328.69 | .4058 | .8697 | .0083 | .0000 | 154.8 | .0588 | .6619 | .0000 | .0272 | .0091/.4608 | .961 | 62+1 |
| Reencoded m20 | **293.49** | .4093 | .8619 | .0123 | .0000 | 151.6 | .0583 | .6398 | .0000 | .0797 | .0486/.6655 | .951 | 52+1 |
| Reencoded m24 | 299.11 | .4016 | .8615 | .0127 | .0078 | 152.9 | .0588 | .6381 | .0000 | .0573 | .0328/.6069 | .956 | 56+1 |
| Reencoded m28 | 307.59 | .4096 | .8644 | .0113 | .0000 | 151.9 | .0581 | .6379 | .0000 | .0370 | .0165/.5139 | .960 | 60+1 |
| Reencoded m30 | 302.96 | .4108 | .8644 | .0129 | .0000 | 151.6 | .0587 | .6416 | .0000 | .0274 | .0089/.4721 | .961 | 62+1 |
| Hybrid m20 | 295.57 | .4078 | .8624 | .0124 | .0000 | 151.6 | .0585 | .6422 | .0000 | .0804 | .0502/.6576 | .951 | 52+1 |
| Hybrid m24 | 301.11 | .4081 | .8642 | .0106 | .0078 | 152.4 | .0582 | .6423 | .0000 | .0591 | .0342/.5994 | .956 | 56+1 |
| Hybrid m28 | 303.43 | .4086 | .8652 | .0118 | .0000 | 152.3 | .0589 | .6406 | .0000 | .0379 | .0173/.5008 | .960 | 60+1 |
| Hybrid m30 | 307.73 | .4072 | .8612 | .0127 | .0000 | 152.3 | .0592 | .6424 | .0000 | .0279 | .0089/.4888 | .961 | 62+1 |

LangFlow full grid (three-seed means):

| Arm | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | Unique | Collapse | A rev. | B rev. | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel | 12.15 | .2556 | .4887 | .2953 | .1979 | 154.6 | .1245 | .2816 | .0990 | — | — | 32 |
| Semi-AR | 23.69 | .2762 | .5447 | .2145 | .3385 | 142.2 | .1575 | .2730 | .2031 | — | — | 64 |
| Neutral m24 | 23.71 | .3547 | .5545 | .2609 | .2448 | 113.2 | .1153 | .3417 | .0781 | .0550 | .2728 | 56 |
| Raw m24 | 13.10 | .2532 | .4468 | .3291 | .2656 | 143.1 | .1271 | .2683 | .1146 | .0677 | .0803 | 56 |
| Continuous m24 | 26.07 | .2901 | .5481 | .2130 | .3073 | 141.0 | .1504 | .2959 | .1667 | .0651 | .3319 | 56 |
| Hard m24 | 25.20 | .2868 | .5441 | .2144 | .3021 | 140.6 | .1469 | .2949 | .1510 | .0616 | .3400 | 56 |
| Neutral m28 | 24.31 | .3585 | .5547 | .2657 | .2552 | 113.3 | .1154 | .3448 | .0938 | .0293 | .1769 | 60 |
| Raw m28 | 12.57 | .2448 | .4333 | .3378 | .3125 | 143.3 | .1332 | .2578 | .1406 | .0402 | .0411 | 60 |
| Continuous m28 | 26.29 | .2876 | .5529 | .2103 | .3125 | 141.7 | .1522 | .2943 | .1771 | .0371 | .2156 | 60 |
| Hard m28 | 25.98 | .2871 | .5508 | .2129 | .3281 | 141.7 | .1518 | .2912 | .1823 | .0349 | .2150 | 60 |
| Neutral m30 | 24.83 | .3614 | .5549 | .2675 | .2604 | 113.2 | .1152 | .3483 | .0990 | .0181 | .1194 | 62 |
| Raw m30 | **11.95** | .2411 | .4240 | .3451 | .3021 | 141.6 | .1388 | .2517 | .1458 | .0246 | .0219 | 62 |
| Continuous m30 | 25.20 | .2862 | .5512 | .2145 | .3229 | 141.6 | .1553 | .2866 | .1927 | .0224 | .1436 | 62 |
| Hard m30 | 24.69 | .2815 | .5439 | .2182 | .3229 | 141.8 | .1536 | .2829 | .1875 | .0205 | .1439 | 62 |

Plaid full grid (three-seed means):

| Arm | PPL | D1 | D2 | Rep-4 | Deg. | Words | MaxShare | Unique | Collapse | A rev. | B rev. | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel | 142.02 | .5082 | .9253 | .0012 | .0365 | 167.3 | .0542 | .7396 | .0000 | — | — | 32 |
| Semi-AR | 131.95 | .4834 | .9164 | .0010 | .0260 | 178.5 | .0549 | .7343 | .0000 | — | — | 64 |
| Neutral m24 | 183.56 | .4878 | .9217 | .0008 | .0312 | 172.4 | .0506 | .7556 | .0000 | .5007 | .5492 | 56 |
| Raw m24 | 131.78 | .4798 | .9169 | .0010 | .0156 | 178.8 | .0555 | .7332 | .0000 | .4923 | .4895 | 56 |
| Continuous m24 | 132.14 | .4795 | .9162 | .0008 | **.0104** | 178.7 | .0552 | .7322 | .0000 | .4897 | .4925 | 56 |
| Hard m24 | **130.01** | .4790 | .9169 | .0012 | .0156 | 178.6 | .0554 | .7305 | .0000 | .4857 | .4921 | 56 |
| Neutral m28 | 186.69 | .4901 | .9242 | .0007 | .0208 | 172.3 | .0505 | .7565 | .0000 | .3258 | .3847 | 60 |
| Raw m28 | 132.39 | .4819 | .9167 | .0008 | .0208 | 178.6 | .0554 | .7344 | .0000 | .3214 | .3188 | 60 |
| Continuous m28 | 131.40 | .4816 | .9158 | .0009 | .0260 | 178.6 | .0553 | .7328 | .0000 | .3197 | .3165 | 60 |
| Hard m28 | 130.25 | .4820 | .9154 | .0009 | .0156 | 178.6 | .0556 | .7328 | .0000 | .3149 | .3171 | 60 |
| Neutral m30 | 185.83 | .4879 | .9226 | .0008 | .0260 | 172.3 | .0503 | .7560 | .0000 | .2207 | .2711 | 62 |
| Raw m30 | 132.26 | .4826 | .9158 | .0010 | .0208 | 178.6 | .0550 | .7320 | .0000 | .2192 | .2096 | 62 |
| Continuous m30 | 131.74 | .4828 | .9166 | .0011 | .0208 | 178.5 | .0554 | .7332 | .0000 | .2189 | .2101 | 62 |
| Hard m30 | 132.43 | .4825 | .9165 | .0012 | .0208 | 178.5 | .0554 | .7343 | .0000 | .2171 | .2141 | 62 |

ELF's reencoded/hybrid `m=20` cells beat Semi-AR by `15--17` PPL with
12 fewer denoiser calls and otherwise similar quality, but still lose badly to
Parallel-32. LangFlow raw context reaches parallel-like PPL only by worsening
diversity, repetition, degeneration, and collapse. Plaid hard `m=24` is a weak
compute-quality lead over Semi-AR (PPL `130.01` versus `131.95`, 56 versus 64
calls), but paired PPL deltas are `-6.85/+1.38/-.35`; it is not yet a robust
method result. The cross-architecture sweep therefore supports, at most, an
architecture-specific cheaper Semi-AR approximation.

### 6.12 Paired unconditional/conditional main table (EXP-80)

Protocol: ELF base, ODE, length 128, paired `n_uncond=n_cond=64`, seed 42,
noise 2, SC-CFG 3. Conditional generation observes the first 64 positions and
generates the final 64. The panel uses a deterministic offset into the released
OWT train split and is therefore labeled **in-domain**, not train-disjoint.
All conditional schedules cover only free suffix positions. Latent prompt-clamp
error is zero for every arm.

This is the current unified main table. `C-PPL` uses the true prompt, `Shuffle`
uses a mismatched prompt with the same generated suffix, and `Gain` is their
log-PPL difference. `Calls+R` separates denoiser calls from lexical readouts.

| Arm | U-PPL | U-D1 | U-D2 | U-Rep4 | U-Deg | C-PPL | Shuffle | Gain | C-RL | C-D1 | C-D2 | C-Rep4 | C-Deg | Calls+R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard-32 | 278.7 | .4970 | .8879 | .0137 | .0000 | 587.1 | 770.9 | .2724 | .0851 | .5619 | .9224 | .0162 | .0000 | 32+0 |
| Standard-64 | 101.5 | .4438 | .8248 | .0346 | .0469 | 267.4 | 358.9 | .2944 | **.0886** | .5165 | .8779 | .0349 | .0469 | 64+0 |
| Standard-136 | **62.7** | .4160 | .7588 | .0853 | .1406 | **183.5** | **243.6** | .2833 | .0816 | .4965 | .8434 | .0492 | .0938 | 136+0 |
| Unlock-4 | 201.7 | .4857 | .8801 | .0123 | .0000 | 393.0 | 541.7 | **.3211** | .0877 | .5398 | .9077 | .0202 | .0156 | 32+1 |
| Soft LTR | 232.9 | .4840 | .8762 | .0165 | .0156 | 512.3 | 657.3 | .2492 | .0846 | .5448 | .9076 | .0200 | .0000 | 64+0 |
| Soft random | 213.3 | .4837 | .8698 | .0198 | .0000 | 506.0 | 662.0 | .2689 | .0860 | .5524 | .9098 | .0217 | .0000 | 64+0 |
| Local + refine 8 | 1583.6 | .6020 | .9569 | .0014 | .0000 | 1915.5 | 2346.0 | .2027 | .0643 | .6362 | .9609 | .0018 | .0000 | 136+0 |
| Canonical + refine 8 | 1395.6 | .6924 | .9787 | .0000 | .0469 | 764.9 | 849.4 | .1048 | .0520 | .6104 | .9245 | .0033 | .2188 | 136+0 |

The remaining generated-quality, preservation, and cost fields from the same
run are:

| Arm | U-Words | U-MaxShare | U-Unique | U-collapse | C-suffix PPL | C-Words | C-MaxShare | C-Unique | C-collapse | Decoded prefix | Clamp | Token-calls | U/C sec. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard-32 | 75.7 | .0768 | .6871 | .0000 | 590.3 | 42.9 | .0732 | .7840 | .0000 | .9688 | 0 | 4096 | 6.5/6.0 |
| Standard-64 | 72.0 | .0931 | .6100 | .0469 | 265.7 | 42.7 | .0822 | .7400 | .0000 | .9531 | 0 | 8192 | 11.8/11.7 |
| Standard-136 | 69.5 | .1137 | .5301 | .0781 | 183.2 | 43.0 | .0881 | .7115 | .0156 | .9375 | 0 | 17408 | 24.6/24.6 |
| Unlock-4 | 75.3 | .0793 | .6688 | .0000 | 405.9 | 42.6 | .0765 | .7632 | .0000 | .9688 | 0 | 4096 | 6.2/6.2 |
| Soft LTR | 76.8 | .0806 | .6652 | .0000 | 497.7 | 42.9 | .0760 | .7675 | .0000 | .9688 | 0 | 8192 | 11.8/11.9 |
| Soft random | 75.9 | .0780 | .6664 | .0000 | 509.2 | 42.9 | .0733 | .7718 | .0000 | .9844 | 0 | 8192 | 11.9/11.9 |
| Local + refine 8 | 84.4 | .0494 | .8279 | .0000 | 1777.6 | 43.7 | .0685 | .8470 | .0000 | .9375 | 0 | 17408 | 26.5/26.1 |
| Canonical + refine 8 | 52.4 | .0529 | .8905 | .0000 | 651.0 | 29.2 | .1207 | .8128 | .1406 | .9688 | 0 | 17408 | 25.8/26.0 |

`Decoded prefix` is an autoencoder reconstruction diagnostic. The actual
observed prompt latent is clamped exactly in every arm, as shown by `Clamp=0`.

The paired panel resolves the earlier coverage gap without changing the method
ranking. Soft random is slightly better than soft LTR but both lose badly to
Standard-64. Local-clock and canonical-context lose to Standard-136;
canonical-context also has the weakest prompt gain and `21.9%` conditioned
degeneration. More standard ODE calls lower PPL but increase repetition and
degeneration, so Standard-136 is not treated as an unqualified quality win.

Unlock-4 is the positive control that survives both scopes. Relative to
Standard-32 it improves unconditional PPL by `77.1`, prompt-conditioned PPL by
`194.1`, ROUGE-L by `.0026`, and prompt gain by `.0486`, with one extra
readout. The supported method claim remains temporary, revisable anchoring in
deterministic ELF ODE—not a general asynchronous wave.

#### P1 robustness panels

Three paired `n_uncond=n_cond=128` replications are complete: two independent
OWT noise/data blocks (seeds 43/44, offsets 11000/12000) and one Gutenberg
out-of-domain panel (seed 42). All prompt latent clamp errors remain zero.

| Panel | Arm | U-PPL | U-D1 | U-D2 | U-Rep4 | U-Deg | C-PPL | Shuffle | Gain | C-RL | C-D1 | C-D2 | C-Rep4 | C-Deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gutenberg | Standard-32 | 296.7 | .4670 | .8839 | .0085 | .0000 | 572.2 | 665.9 | .1517 | .0870 | .4514 | .9034 | .0075 | .0234 |
| Gutenberg | Standard-64 | 109.3 | .4246 | .8307 | .0274 | .0781 | 292.4 | 351.9 | .1853 | .0859 | .4159 | .8587 | .0231 | .0078 |
| Gutenberg | Standard-136 | 58.6 | .3860 | .7585 | .0720 | .1406 | 193.7 | 237.9 | .2054 | .0877 | .4060 | .8226 | .0449 | .0234 |
| Gutenberg | Unlock-4 | 214.7 | .4515 | .8759 | .0084 | .0000 | 427.8 | 502.5 | .1609 | .0879 | .4291 | .8886 | .0124 | .0234 |
| OWT-43 | Standard-32 | 287.0 | .4698 | .8887 | .0078 | .0312 | 584.9 | 748.8 | .2470 | .0839 | .5093 | .9048 | .0157 | .0469 |
| OWT-43 | Standard-64 | 112.8 | .4389 | .8373 | .0239 | .0781 | 313.2 | 427.1 | .3103 | .0820 | .4720 | .8636 | .0335 | .0469 |
| OWT-43 | Standard-136 | 52.7 | .4011 | .7694 | .0648 | .1250 | 177.1 | 243.9 | .3200 | .0850 | .4425 | .8277 | .0521 | .0625 |
| OWT-43 | Unlock-4 | 213.1 | .4536 | .8747 | .0102 | .0547 | 429.3 | 546.8 | .2420 | .0857 | .4913 | .8921 | .0200 | .0547 |
| OWT-44 | Standard-32 | 265.2 | .4587 | .8754 | .0131 | .0078 | 512.9 | 667.5 | .2635 | .0898 | .5009 | .9109 | .0138 | .0156 |
| OWT-44 | Standard-64 | 106.2 | .4239 | .8253 | .0374 | .1016 | 266.3 | 367.7 | .3229 | .0949 | .4609 | .8681 | .0303 | .0547 |
| OWT-44 | Standard-136 | 48.1 | .3827 | .7466 | .0840 | .1562 | 138.9 | 197.0 | .3493 | .0958 | .4411 | .8242 | .0602 | .0938 |
| OWT-44 | Unlock-4 | 197.0 | .4415 | .8635 | .0152 | .0078 | 378.8 | 496.8 | .2711 | .0938 | .4874 | .9005 | .0173 | .0391 |

The remaining quality, preservation, and compute fields are:

| Panel | Arm | U-Words | U-MaxShare | U-Unique | U-collapse | C-suffix PPL | C-Words | C-MaxShare | C-Unique | C-collapse | Decoded prefix | Clamp | Calls+R | Token-calls | U/C sec. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gutenberg | Standard-32 | 74.0 | .0779 | .7002 | .0000 | 510.2 | 44.1 | .0747 | .7943 | .0000 | .9844 | 0 | 32+0 | 4096 | 12.6/12.6 |
| Gutenberg | Standard-64 | 69.3 | .0989 | .6185 | .0625 | 263.4 | 43.6 | .0821 | .7508 | .0000 | .9844 | 0 | 64+0 | 8192 | 24.8/25.6 |
| Gutenberg | Standard-136 | 67.9 | .1158 | .5313 | .0938 | 173.3 | 43.0 | .0877 | .7128 | .0000 | .9844 | 0 | 136+0 | 17408 | 53.8/56.5 |
| Gutenberg | Unlock-4 | 73.8 | .0799 | .6807 | .0000 | 378.9 | 44.0 | .0771 | .7776 | .0000 | .9844 | 0 | 32+1 | 4096 | 14.1/14.1 |
| OWT-43 | Standard-32 | 70.7 | .0800 | .6912 | .0078 | 532.3 | 43.1 | .0792 | .7728 | .0078 | .9453 | 0 | 32+0 | 4096 | 12.5/13.2 |
| OWT-43 | Standard-64 | 65.4 | .0967 | .6169 | .0312 | 299.3 | 42.7 | .0887 | .7299 | .0234 | .9453 | 0 | 64+0 | 8192 | 24.1/24.0 |
| OWT-43 | Standard-136 | 61.7 | .1147 | .5230 | .0781 | 171.4 | 42.8 | .0897 | .6968 | .0156 | .9531 | 0 | 136+0 | 17408 | 50.3/50.4 |
| OWT-43 | Unlock-4 | 70.4 | .0815 | .6742 | .0078 | 389.6 | 42.9 | .0797 | .7614 | .0078 | .9453 | 0 | 32+1 | 4096 | 12.7/12.8 |
| OWT-44 | Standard-32 | 72.8 | .0838 | .6855 | .0234 | 504.2 | 43.5 | .0792 | .7882 | .0000 | .9766 | 0 | 32+0 | 4096 | 12.4/12.5 |
| OWT-44 | Standard-64 | 66.3 | .1026 | .5987 | .0547 | 274.1 | 43.7 | .0813 | .7470 | .0078 | .9922 | 0 | 64+0 | 8192 | 24.2/24.5 |
| OWT-44 | Standard-136 | 62.8 | .1175 | .5175 | .1094 | 145.8 | 42.9 | .0870 | .7000 | .0000 | .9922 | 0 | 136+0 | 17408 | 50.6/50.7 |
| OWT-44 | Unlock-4 | 72.4 | .0869 | .6648 | .0234 | 378.4 | 43.2 | .0820 | .7764 | .0000 | .9922 | 0 | 32+1 | 4096 | 12.8/12.9 |

Relative to Standard-32, Unlock-4 improves U-PPL by `68.2--82.0` and C-PPL by
`134.0--155.7` in all three panels; ROUGE-L also rises by `.0009--.0040`.
Prompt-gain deltas are `+.0092/-.0050/+.0076`, so stronger prompt utilization
does not robustly replicate. C-D1 falls in every panel (mean `-.0180`) and
C-Rep4 rises (mean `+.0042`); mean C-degeneration rises by `.0104`. The robust
claim is therefore a same-denoiser-call PPL improvement with modest lexical
diversity/repetition trade-offs, not a general improvement on every metric.
Standard-64 still has substantially lower PPL, while using twice as many
denoiser calls.

### 6.13 Temporary-anchor diagnosis and scale (EXP-81/82/88/89)

These experiments diagnose the replicated ELF ODE Unlock signal without
changing the base model or the 32 denoiser-call budget.

#### Where does the conditional gain occur? (EXP-81)

EXP-81 rescored the exact `384` EXP-80 P1 continuations with GPT-2-large. The
reported quantity is Unlock-4 minus Standard-32; negative true NLL is better,
while positive prompt gain means stronger preference for the true prompt over
a shuffled prompt.

| suffix band | `Delta` true NLL | paired-bootstrap 95% CI | `Delta` prompt gain | paired-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| GPT-2 tokens 1--8 | `-.2735` | `[-.3528,-.1945]` | `+.0465` | `[-.0071,+.0985]` |
| GPT-2 tokens 9--32 | `-.2369` | `[-.2930,-.1815]` | `-.0076` | `[-.0300,+.0152]` |
| GPT-2 tokens 33+ | `-.3749` | `[-.4292,-.3224]` | `-.0002` | `[-.0161,+.0162]` |
| full suffix | `-.3009` | `[-.3363,-.2653]` | `+.0045` | `[-.0085,+.0181]` |

Unlock lowers NLL throughout the continuation, but no pooled prompt-gain band
is significant. Full-suffix prompt-gain deltas across Gutenberg and two OWT
panels are `+.0096/-.0023/+.0063`, all with intervals crossing zero. The
replicated conditional PPL improvement is therefore mainly generic sample
likelihood, not demonstrably stronger prompt utilization.

#### Confidence, coverage, and correct content (EXP-82)

The P0 sweep freezes trigger `t=.30`, anchor density `.50`, and horizon `H=4`.
Formal results use three paired `n_U=n_C=128` panels. All arms have 32 denoiser
calls; anchor arms add one lexical readout.

| Panel | Arm | U-PPL | C-PPL | Gain | C-RL | C-D1 | C-Rep4 | C-Deg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gutenberg | Standard-32 | 296.7 | 572.2 | .1517 | .0870 | .4514 | .0075 | .0234 |
|  | top-confidence | 243.0 | 424.6 | .1783 | .0899 | .4372 | .0128 | .0156 |
|  | random position | **206.5** | **392.4** | **.1838** | **.0920** | .4281 | .0120 | .0078 |
|  | shuffled content | 805.5 | 1554.1 | .0111 | .0791 | .4879 | .0009 | .0078 |
| OWT-42 | Standard-32 | 296.7 | 728.6 | .2251 | .0812 | .5262 | .0090 | .0234 |
|  | top-confidence | 243.0 | 517.9 | .2483 | .0854 | .5176 | .0129 | .0312 |
|  | random position | **206.5** | **512.7** | **.2642** | **.0876** | .5145 | .0103 | .0391 |
|  | shuffled content | 805.5 | 1971.7 | .0607 | .0590 | .5622 | .0035 | .0156 |
| OWT-43 | Standard-32 | 287.0 | 572.7 | .2535 | .0846 | .4916 | .0133 | .0547 |
|  | top-confidence | 240.5 | 430.9 | **.2754** | .0879 | .4800 | .0175 | .0625 |
|  | random position | **203.7** | **380.2** | .2739 | **.0888** | .4744 | .0173 | .0547 |
|  | shuffled content | 830.8 | 1580.3 | .0495 | .0705 | .5346 | .0039 | .0078 |

Random anchors have lower trigger confidence than top-confidence anchors in
P0 (`.887` versus `.999`) yet are consistently better or near-better. Correct
position-content is essential because matched shuffled content destroys both
U/C PPL. Thus broad coverage, rather than selecting only already-confident
tokens, is the stronger mechanism clue. The caveat is consistent: C-D1 falls
and C-Rep4 often rises.

#### Shadow-validated rollback (EXP-88)

An unconditioned shadow forward at the lock midpoint gives a valid disagreement
signal. Shadow-null exactly matches Standard and shadow-keep matches the fixed
anchor arm, ruling out an extra-compute explanation.

| Arm | U-PPL | C-PPL | Gain | C-RL | C-D1 | C-Rep4 | Released |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard-32 | 278.7 | 812.8 | .1845 | .0778 | .5945 | .0086 | -- |
| fixed random anchor | 192.1 | 561.3 | .2149 | .0874 | .5754 | .0121 | .000 |
| identity rollback | 183.2 | 515.1 | **.2208** | **.0884** | .5669 | .0111 | .325 |
| confidence rollback | 194.4 | 550.5 | .2121 | .0873 | .5734 | .0125 | .076 |
| combined rollback | **182.6** | **511.1** | .2167 | .0877 | .5680 | .0108 | .341 |

Rollback is active and further improves PPL, but it does not recover lexical
diversity. It fails the preregistered Pareto gate and is not promoted.

#### Length and prefix-ratio scaling (EXP-89)

The frozen random-anchor policy improves PPL in all nine new cells. Entries are
random-anchor minus Standard-32 deltas; length-1024 uses the preregistered
preliminary `n_U=n_C=16` budget.

| length | prefix ratio | `Delta` U-PPL | `Delta` C-PPL | `Delta` gain | `Delta` C-RL | `Delta` C-D1 | `Delta` C-Rep4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | .25 | -25.2 | -41.7 | -.006 | +.004 | -.011 | +.003 |
| 256 | .50 | -25.2 | -96.5 | +.012 | +.004 | -.016 | +.003 |
| 256 | .75 | -25.2 | -175.3 | +.018 | +.003 | -.020 | +.001 |
| 512 | .25 | -9.8 | -18.7 | .000 | +.003 | -.010 | +.001 |
| 512 | .50 | -9.8 | -52.9 | +.005 | +.005 | -.011 | .000 |
| 512 | .75 | -9.8 | -126.0 | +.011 | +.003 | -.005 | .000 |
| 1024 | .25 | -2.5 | -13.0 | .000 | +.003 | .000 | -.001 |
| 1024 | .50 | -2.5 | -42.4 | +.027 | +.001 | -.012 | -.002 |
| 1024 | .75 | -2.5 | -184.0 | -.002 | +.002 | -.013 | +.003 |

The unconditional effect decays with sequence length, whereas conditional
benefit grows with observed-prefix ratio. No new degeneration appears. This
supports scale robustness of the PPL sign, but the diversity trade-off remains
and prevents an unqualified method claim.

### 6.14 Triggered subset-flow training (EXP-91)

A matched continued-training control and a subset-flow checkpoint each receive
200 real-OWT steps. In half of subset-flow examples, a random 50% subset is
replaced by the frozen teacher's predicted-clean state and loss is applied only
to unresolved positions. The held-out paired panel uses `n_U=n_C=128` and no
training-document overlap.

| Checkpoint | Arm | U-PPL | U-D1 | U-Rep4 | C-PPL | Gain | C-RL | C-D1 | C-Rep4 | C-Deg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched control | Standard | 296.3 | .4719 | .0076 | 599.1 | .2353 | .0762 | .5257 | .0098 | .0156 |
| matched control | Random anchor | 211.7 | .4489 | .0116 | **416.9** | **.2943** | .0810 | .5078 | .0131 | .0234 |
| subset-flow | Standard | 301.8 | .4694 | .0063 | 610.0 | .2334 | .0745 | .5301 | .0120 | .0234 |
| subset-flow | Random anchor | **208.6** | **.4506** | .0096 | 419.9 | .2817 | **.0816** | **.5097** | .0134 | .0391 |

The single seed-42 interaction is not robust. Two additional inference seeds
use the same trained checkpoints with independent noise and OWT offsets:

| inference seed | `Delta Delta` U-PPL | `Delta Delta` C-PPL | `Delta Delta` gain | `Delta Delta` C-RL | `Delta Delta` C-D1 | `Delta Delta` C-Rep4 | `Delta Delta` C-Deg. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -8.6 | -8.0 | -.0107 | +.0024 | -.0024 | -.0019 | +.0078 |
| 123 | +11.5 | +28.9 | +.0056 | -.0012 | -.0014 | -.0027 | +.0078 |
| 456 | +1.7 | -14.0 | -.0166 | -.0016 | +.0013 | +.0024 | +.0078 |
| mean | **+1.5** | **+2.3** | **-.0072** | **-.0001** | **-.0008** | **-.0007** | **+.0078** |

`Delta Delta` is the subset-flow random-minus-Standard effect minus the
matched-control effect; negative PPL is favorable. Neither U nor C interaction
is sign-stable, and both mean PPL interactions are slightly unfavorable.
Prompt gain falls on two seeds and conditioned degeneration worsens by one of
128 samples on every seed. The negative decision therefore does not rest on a
single worse Standard baseline: the targeted asynchronous capability does not
replicate across inference panels.

### 6.15 Conditional Plaid late coupling (EXP-87)

EXP-79's conditional negative result is ELF-specific. EXP-87 repeats the
64-token-prefix/192-token-continuation comparison with Plaid's native ancestral
sampler, shared step noise, exact prompt clamping, `n=128` per seed, and seeds
`42/123/456`. Values are three-seed means.

| Arm | Calls | Suffix PPL | C-PPL | Shuffle | Gain | Boundary | R-L | D1 | D2 | Rep-4 | Deg. | A/B revision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel | 32 | 107.36 | 105.32 | 127.10 | .1881 | 130.27 | .1034 | .4380 | .8967 | .0001 | .0260 | .000/.000 |
| Block-SAR | 64 | 104.41 | 100.87 | 124.38 | .2096 | 149.82 | .1080 | .4326 | .8947 | .0010 | .0130 | .000/.000 |
| Late raw | 56 | **98.93** | **95.38** | **118.44** | .2166 | 133.52 | **.1087** | .4304 | .8916 | .0010 | .0104 | .441/.446 |
| Late continuous | 56 | 99.98 | 96.18 | 119.68 | **.2185** | 126.58 | .1086 | .4295 | .8921 | .0012 | **.0078** | .441/.446 |
| Late hard | 56 | 100.19 | 96.06 | 119.27 | .2164 | **122.51** | .1085 | .4290 | .8917 | .0010 | .0156 | .437/.448 |

All three late arms beat Block-SAR C-PPL in every seed while using eight fewer
calls. Raw gives the best mean suffix/C-PPL, continuous the best prompt gain
and degeneration, and hard the best boundary PPL. D1/D2 are slightly lower,
so this is a robust compute-quality lead over Block-SAR, not dominance on every
metric. The large post-coupling revision rates confirm that the gain does not
come from irreversibly freezing the first block.

### 6.16 Cross-architecture temporary-anchor portability (EXP-90)

LangFlow and Plaid use their own endpoint-calibrated trigger step and native
32-step solver. Every result is the mean of seeds `42/123/456` with
`n_U=n_C=32`. Plaid shares the exact ancestral noise at every step. Duplicate
native baselines agree token-for-token (`1.0`) in both scopes, anchor density
is exactly `.50`, and observed prompt latent clamp error is zero.

| Model | Arm | U-PPL | C-PPL | Gain | C-RL | C-D1 | C-Rep4 | U-Deg. | C-Deg. | Revision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LangFlow | Standard | 13.33 | 82.01 | .4932 | .0461 | .4471 | .1209 | .3021 | .2604 | -- |
|  | random correct | 14.52 | **73.40** | **.5118** | .0462 | .4316 | .1228 | .3125 | .2708 | .040 |
|  | top-confidence | 16.72 | 87.95 | .4898 | .0457 | .4527 | .1087 | .3229 | .2708 | .000 |
|  | shuffled content | 423.43 | 1180.01 | .3238 | .0392 | .5974 | .0031 | .1562 | .0833 | .036 |
| Plaid | Standard | 136.90 | 97.81 | .5303 | .0964 | .6183 | .0000 | .0417 | .0521 | -- |
|  | random correct | 121.80 | **85.51** | **.5414** | **.1003** | .6096 | .0000 | .0625 | .0417 | .627 |
|  | top-confidence | **114.46** | 90.42 | .5128 | .0998 | .6173 | .0002 | .0417 | .0417 | .389 |
|  | shuffled content | 952.03 | 682.19 | .3082 | .0845 | .6619 | .0002 | .0208 | .0521 | .794 |

Random position-correct anchors improve C-PPL in all six architecture-seed
panels. On LangFlow the mean conditional delta is `-8.61` PPL and prompt gain
improves in 3/3 seeds, but U-PPL worsens by `+1.18` in 3/3. On Plaid the mean
random-anchor deltas are `-15.10` U-PPL and `-12.29` C-PPL, both favorable in
3/3 seeds. Top confidence is harmful in both LangFlow scopes; on Plaid it is
best unconditionally but improves C-PPL in only 2/3 seeds.

Correct content is the strongest portable control: shuffled anchors increase
mean C-PPL by `+1098.0` on LangFlow and `+584.4` on Plaid. Their low repetition
and degeneration detector rates do not make them coherent; evaluator PPL and
the texts expose catastrophic corruption. Temporary anchors are also genuinely
revisable: random-anchor final revision is about `4%` on deterministic
LangFlow and `63%` on ancestral Plaid.

The safe cross-architecture claim is thus conditional and mechanistic, not an
all-metric method claim: broad, correct temporary context improves prompted
continuations, while unconditional gains, confidence selection, revision, and
diversity/degeneration trade-offs depend on architecture and solver.

### 6.17 Conditional/on-policy subset-flow factorization (EXP-92)

EXP-92 corrects three EXP-91 mismatches: half of training examples contain a
real clamped prefix, every update includes a paired synchronous preservation
loss, and the on-policy arm obtains its mixed state from a frozen-teacher ODE
trajectory with a random subset held for one, two, or four steps. All arms use
the same OWT documents, update count, trainable parameters, and first
synchronous objective. Prompt-clamp error is exactly zero.

Three-seed means (`n_U=n_C=32`) are:

| Training arm | Standard U-PPL | Random U-PPL | Standard C-PPL | Random C-PPL | Standard gain | Random gain | Random C-RL | Random C-D1 | Random C-Rep4 | Random C-Deg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paired control | 287.36 | **201.02** | **765.59** | **524.99** | .1598 | **.2133** | .0740 | **.6377** | **.0158** | .0208 |
| conditional oracle | 287.63 | 206.45 | 787.24 | 532.94 | .1651 | .2004 | **.0753** | .6340 | .0164 | .0208 |
| conditional on-policy | **286.99** | 210.11 | 779.13 | 550.11 | **.1668** | .2015 | .0750 | .6398 | .0165 | .0208 |

The random-minus-Standard interaction relative to the paired control is:

| Training arm | `DeltaDelta` U-PPL | `DeltaDelta` C-PPL | `DeltaDelta` gain | `DeltaDelta` C-RL | `DeltaDelta` C-D1 | `DeltaDelta` C-Rep4 | `DeltaDelta` C-Deg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditional oracle | +5.16 | -13.70 | -.0182 | +.0004 | -.0065 | +.0003 | +.0104 |
| conditional on-policy | +9.46 | +11.58 | -.0189 | +.0020 | +.0061 | -.0001 | +.0104 |

Negative PPL is favorable. Conditional oracle has a favorable C-PPL
interaction in two seeds but worsens absolute random C-PPL, prompt gain, D1,
and degeneration. On-policy is unfavorable on mean U/C PPL interactions and
passes C-PPL in only one seed. Neither arm passes the gate. Because the initial
on-policy transition loss is about three times the synchronous loss, one fixed
`lambda_mix=.25` run tests loss imbalance before the straight-to-endpoint
target is retired. That follow-up improves synchronous validation from `.998`
to `.889`, but its mean U/C PPL interactions remain unfavorable at
`+7.08/+16.31`; C-PPL interaction is worse in all three seeds and prompt gain
falls in all three. The current target is therefore retired. A successor must
preserve native curved velocity and add only a normalized residual correction,
not force the rollout state directly toward the endpoint.

### 6.18 Compute-matched Plaid late-coupling audit (EXP-94)

EXP-87 compared denoiser calls, but clamped suffix maturation still evaluates
the full two-block context. Late-24 therefore costs `11264` token-calls, exactly
matching Parallel-44 rather than Parallel-32. The paired seed-42 `n=32` audit
finds:

| Arm | Token-calls | C-PPL | Boundary PPL | Gain | D1 | Deg. |
|---|---:|---:|---:|---:|---:|---:|
| Parallel-32 | 8192 | 107.95 | 148.89 | .1680 | .5622 | .0000 |
| Parallel-44 | 11264 | **90.63** | **113.76** | **.1786** | .5633 | .0625 |
| Block-SAR-64 | 12288 | 122.27 | 218.33 | .1536 | .5539 | .0000 |
| Late raw-24 | 11264 | 114.54 | 175.75 | .1740 | .5629 | .0000 |
| Late continuous-24 | 11264 | 114.54 | 177.39 | .1672 | .5621 | .0000 |
| Late hard-24 | 11264 | 122.57 | 179.55 | .1726 | .5585 | .0313 |

Late raw/continuous are cleaner Block-SAR replacements, but their EXP-87 lead
is explained by comparison against an inefficient baseline: they lose to
Parallel-44 by `+23.91` C-PPL and `+61.99/+63.63` boundary PPL at identical
token compute. Do not claim an asynchronous compute-allocation advantage, and
do not expand this schedule into an adaptive/multi-block method without a
material algorithmic change.

### 6.19 Plaid temporary-anchor Pareto screen (EXP-95)

The paired seed-42 `n_U=n_C=16` screen covers 48 random-anchor cells across
native trigger steps `14/18/22`, densities `.125/.25/.50/.75`, and horizons
`1/2/4/8`. Eight cells improve C-PPL and D1 without larger degeneration or a
material Rep-4/prompt-gain regression. The signal is concentrated at the early
trigger (`step=14`, `t_native=.4652`):

| Trigger | Density | H | Delta U-PPL | Delta C-PPL | Delta C-D1 | Delta Deg. | Delta gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | .50 | 1 | **-34.56** | **-45.58** | +.0082 | .0000 | +.0869 |
| 14 | .75 | 1 | -36.09 | -36.32 | +.0098 | .0000 | +.0110 |
| 14 | .50 | 4 | -18.19 | -33.32 | **+.0158** | .0000 | +.0331 |
| 14 | .75 | 4 | -25.06 | -29.93 | +.0105 | .0000 | +.0361 |
| 18 | .75 | 2 | -26.49 | -16.13 | +.0060 | .0000 | +.0089 |

Leading post-transition cells also improve PPL but consistently reduce D1.
Four complementary early/transition cells are being promoted to larger panels
with Standard, readout-sham, top-confidence, and shuffled-content controls;
the small screen is not yet a formal method result.

All four larger seed-42 panels are positive. Standard and readout sham match
exactly (`C-PPL=111.83`, C-D1 `.6281`). At `t14,d=.50,H=1`, random anchoring
reaches C-PPL `81.85`, gain `.5600`, and D1 `.6413`; at `t14,d=.75,H=1`,
top-confidence reaches the best C-PPL `78.88` with D1 `.6439`. Random
`t14,d=.50,H=4` reaches `87.04`, and confidence `t18,d=.75,H=2` reaches
`90.78`. Degeneration remains `.0313` for these correct-content arms, while
shuffled-content C-PPL ranges from `134.21` to `330.47`. Anchor revision is
`.57--.78`, confirming that the intervention is strongly revisable. The two
single-step early settings are frozen for seeds 123/456 formal replication.

The frozen three-seed replication is complete. For the cleanest setting
(`step=14`, `t_native=.4652`, density `.75`, horizon `1`), means are:

| Arm | U-PPL | C-PPL | Shuffled | Gain | C-RL | C-D1 | C-D2 | Rep-4 | Deg. | Revision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 135.43 | 110.39 | 178.49 | .4804 | .1042 | .6330 | .9603 | .0000 | .0417 | -- |
| Readout sham | 135.43 | 110.39 | 178.49 | .4804 | .1042 | .6330 | .9603 | .0000 | .0417 | -- |
| Random correct | **95.98** | 80.87 | 141.28 | **.5570** | **.1083** | .6301 | .9507 | .0002 | .0313 | .768 |
| Top-confidence correct | 99.32 | **80.28** | **139.50** | .5522 | .1042 | **.6363** | .9552 | .0002 | .0313 | .699 |
| Shuffled content | 234.94 | 138.46 | 221.00 | .4680 | .0902 | .6594 | .9723 | .0000 | .0208 | .856 |

Top-confidence C-PPL improves in all seeds (`-32.95/-36.00/-21.38`); C-D1
improves in two seeds and by `+.0033` on average, while D2 falls by `.0051`.
This is a robust PPL and prompt-gain result with a favorable balanced quality
panel, not strict dominance on every metric. The density-.50 random policy also
improves C-PPL in all seeds (`110.39 -> 79.73`) but does not preserve D1 as
reliably. The paper-facing method is early one-step temporary anchoring, not
late coupling or irreversible hard commitment.

### 6.20 ELF random-subset selector headroom (EXP-93 Stage 1)

This conditional seed-42 `n=64` panel freezes ODE-32, prompt and suffix noise,
native trigger `.30`, exact anchor density `.50`, predicted-clean content, and
hold horizon `4`. Only the identity of the random anchored subset changes.
Sixteen masks are evaluated for every trajectory. The oracle chooses the mask
with the lowest final suffix GPT-2 NLL separately for each trajectory; it is an
analysis upper bound and uses information unavailable at inference.

| Arm | C-PPL | Mean seq. NLL | D1 | D2 | Rep-4 | Deg. | R-L | Anchor revision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard-32 | 553.43 | 6.3321 | .5710 | .9161 | .0148 | .0938 | .0770 | — |
| Top-confidence 50% | 379.43 | 5.9551 | .5580 | .9066 | .0185 | .0625 | .0852 | .0703 |
| Mean of 16 random masks | 335.67 | 5.8387 | .5473 | .9016 | .0230 | .1074 | .0821 | .3427 |
| Oracle best-of-16 | **210.19** | **5.3620** | .5273 | .8870 | .0294 | .1250 | .0818 | — |
| Oracle worst-of-16 | 553.30 | 6.3422 | .5666 | .9126 | .0164 | .0625 | .0853 | — |

Across all `1024` paired mask/trajectory cells, a random mask beats the
top-confidence mask with probability `.6113`. Per-mask aggregate C-PPL ranges
from `317.68` to `357.29`, while within-trajectory utility IQR averages `.3547`
nats. Best-of-16 improves C-PPL by `37.38%` over mean-random and has paired
headroom `.4767 [.4409,.5126]` nats under trajectory bootstrap. No fixed mask
index dominates (winner counts range `1--8`), as expected for independently
sampled subsets.

The Stage-1 gate is therefore passed by a wide margin: selection identity is
a major variable at fixed schedule and content. It does **not** yet show that
the good mask is predictable. Oracle-best also reduces D1 and worsens Rep-4
and degeneration, so a learned selector must be frozen on held-out
trajectories and judged as a multi-metric Pareto method, not optimized against
the final-panel GPT-2 score. The next test records trigger-time reliability,
spatial/latent coverage, redundancy, and shadow-step influence for candidate
subsets and measures out-of-sample ranking accuracy.

Stage 2 is negative. A grouped four-fold trajectory-OOF ridge model has
Spearman `.0003/.0367`, pairwise ranking accuracy `.5013/.5128`, and oracle
mask top-1 accuracy `.0313/.0625` on seed-42/123 utility banks. It selects
C-PPL `339.74` versus mean-random `335.67` on seed 42, and `381.69` versus
`390.12` on seed 123, while the oracle values are `210.19/240.52`. Static
confidence/coverage/redundancy proxies stay near chance; one-step shadow
entropy is only `.523/.527` pairwise accurate. Thus the oracle gap is real but
not deployable with the tested features.

Two stronger follow-ups preserve that boundary. First, candidate-specific
lookahead evolves every mask for `2/4/8` steps and scores unresolved positions
before and after release. On the discovery bank, the best frozen eight-step
scores reduce C-PPL from mean-random `335.67` to `303.77` (confidence gain) or
`308.84` (entropy reduction). On the independent bank, however, they reach
only `384.17/373.37` versus mean-random `390.12`. Paired mean-NLL improvements
are `-.0158 [-.0859,+.0513]` and `-.0436 [-.1093,+.0199]`; both intervals cross
zero. The apparent discovery gain therefore does not validate, and each of 16
candidates additionally costs nine denoiser calls.

Second, a single-position intervention estimates a `64 x 64` causal graph for
each trajectory by holding each candidate source for four steps and measuring
target confidence, entropy, and top-1 changes. Additive graph coverage,
redundancy, and cross-influence features are then scored against the same
candidate utilities. The discovery leader, mean selected-to-unresolved
confidence influence, has Spearman `.113` and pairwise accuracy `.540` on 16
trajectories, but falls to `.006/.504` on an independent 16-trajectory bank;
its selected C-PPL is `458.76`, versus `438.60` for mean random on the same 16
held-out trajectories. The graph OOF model is likewise non-predictive
(`.041/.520`) and selects C-PPL `436.48`, essentially tied with that subset
baseline. The corresponding discovery-only mean-random value is `368.53`, so
the frozen feature's `323.99` there does not transfer.

The resulting conclusion is narrower than “random is best.” Subset identity
has a large, replicated oracle effect, but utility is trajectory-specific and
not recovered by marginal confidence, spatial or latent coverage, short-path
consistency, or an additive pairwise influence graph. A successor must model
non-additive subset interactions and must be trained and frozen before a new
quality bank; Stage 3 is closed for the current proxy family.

### 6.21 Plaid subset headroom and matched-size quality audit (EXP-99)

EXP-99 freezes Plaid native step 14, horizon 1, and one density per run. Every
trajectory evaluates 16 random subsets with a shared prompt, initial latent,
and every ancestral-noise draw; a separate mask seed changes only subset
identity. Discovery and validation use disjoint data offsets and sampler seeds.

| Bank | Density | Standard | Top conf. | Mean random | Oracle best-of-16 | Oracle gain | Paired NLL CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 42, offset 0 | .50 | 107.65 | 90.04 | 84.33 | **44.35** | 47.41% | [.595,.693] |
| seed 42, offset 0 | .75 | 107.65 | **78.44** | 81.70 | 46.66 | 42.89% | [.508,.621] |
| seed 123, offset 1000 | .50 | 100.42 | 96.91 | 92.06 | 46.90 | 49.05% | [.617,.735] |
| seed 123, offset 1000 | .75 | 100.42 | 86.85 | 85.80 | **46.32** | 46.01% | [.572,.669] |

All 64 validation trajectories have positive best-of-16 headroom at each
density. The first runner version flattened all `16 x 64` random texts before
computing corpus-level D1/D2, which made those two values incomparable with
64-text arms. PPL, per-sequence NLL, bootstrap, and the headroom decision were
unaffected. A corrected run computes D1/D2 for each 64-text mask bank and then
averages those matched-size statistics.

| Validation arm, density .75 | C-PPL | Gain | D1 | D2 | Rep-4 | Deg. | R-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | 100.42 | .5303 | .5733 | .9418 | .0000 | .0156 | .1063 |
| Top confidence | 86.85 | .5725 | .5712 | .9350 | .0010 | .0000 | .1051 |
| Mean random | 85.80 | .5675 | .5700 | .9394 | .0001 | .0127 | .1030 |
| Oracle best-of-16 | **46.32** | **.6435** | .5665 | .9356 | .0003 | .0313 | .1025 |

The oracle gap is not merely degeneration: likelihood and prompt gain improve
substantially, while D1/D2 decline modestly. Degeneration does rise at density
`.75`; at density `.50` it instead falls from mean-random `.0039` to zero while
D1 changes `.5707 -> .5537`. A deployable selector must therefore optimize and
validate the complete quality panel rather than final NLL alone.

### 6.22 Non-additive Plaid subset selector (EXP-100, negative)

Trigger replay reconstructs every EXP-99 mask and attaches its final NLL to 53
inference-time token features: noisy state, self-conditioning, predicted-clean
state, confidence, entropy, top-1/top-2 margin, position, and prefix status. A
two-layer joint sequence Transformer receives the entire candidate membership
mask and scores selected--selected and selected--unresolved interactions with a
within-trajectory listwise loss.

The 64-trajectory pilot is negative: validation pairwise accuracy is
`.484/.508/.465` across three optimization seeds and selected C-PPL is
`85.77/86.50/88.58`, versus mean-random `85.80`. The architecture and
hyperparameters are then held fixed while the training bank grows to 320
trajectory-disjoint examples (5120 candidate subsets):

| Opt. seed | Best epoch | Train pair acc. | Val. pair acc. | Val. Spearman | Selected C-PPL | Mean random | NLL delta [95% CI] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 44 | .795 | .494 | -.022 | 79.75 | 85.80 | -.071 [-.148,.002] |
| 123 | 2 | .519 | .500 | .005 | 80.32 | 85.80 | -.070 [-.151,.011] |
| 456 | 43 | .785 | .502 | .004 | 80.20 | 85.80 | -.070 [-.151,.003] |

The PPL column alone is misleading. The candidate index with lowest training
NLL is index 4; frozen on validation it worsens PPL to `88.80` and mean NLL by
`+.034`. The validation-local best fixed index is instead 14, at PPL `79.67`
and `-.075` nats---nearly identical to the early-stopped selectors despite
their chance ranking. The current interpretation is validation selection bias,
not learned utility. Three frozen checkpoints are evaluated once on three
unopened seed/offset banks. The primary checkpoint selected on validation gives:

| Final data seed | Mean random | Frozen index 4 | Joint selector | Pair acc. | NLL delta [95% CI] |
|---:|---:|---:|---:|---:|---:|
| 2026 | 95.68 | **94.00** | 94.43 | .492 | -.014 [-.089,.062] |
| 2027 | 92.55 | **90.47** | 91.24 | .494 | -.015 [-.114,.081] |
| 2028 | 87.57 | 85.75 | **85.73** | .502 | -.021 [-.106,.064] |

Pooling 192 final trajectories yields `-.0167 [-.0658,.0316]` nats for the
primary optimization seed. Its favorable mean sign appears in 3/3 data banks
but never excludes zero and is matched by the training-frozen fixed index. The
other two optimization seeds improve in 0/3 banks and have pooled deltas
`+.0137` and `+.0166` nats. Final Spearman correlations remain near zero.

EXP-100 therefore fails the final ranking/NLL gate. The large Plaid oracle gap
is real, but this joint Transformer does not identify it out of sample. No
selected-text quality panel is run after the deployment gate fails; doing so
would create another opportunity to select a favorable metric post hoc. The
next adaptive method should change a lower-dimensional decision such as trigger
timing, or learn a native trajectory-preserving utility signal rather than
reranking random masks from final-NLL supervision.

### 6.23 Adaptive trigger timing (EXP-101)

EXP-101 freezes Plaid top-confidence density `.75` and horizon one, then varies
only native trigger step in `8,10,12,14,16,18,20,22`. Prompts, initial latents,
and every ancestral-noise draw are paired. Per-trajectory best-of-trigger has
large replicated headroom:

| Bank | Fixed step 14 | Oracle best-of-8 | Improvement | Paired NLL [95% CI] |
|---|---:|---:|---:|---:|
| seed 42 / offset 0 | 78.44 | **43.32** | 44.77% | `-.597 [-.732,-.477]` |
| seed 123 / offset 1000 | 86.85 | **44.52** | 48.74% | `-.664 [-.794,-.537]` |

Every candidate trigger wins at least two trajectories on each bank. Aggregate
fixed timing is also unstable: step 18 is best on discovery (C-PPL `75.20`),
whereas step 8 is best on validation (`76.64`). Timing is therefore a real
trajectory-level control variable rather than a globally mistuned constant.

Eight inference-time summaries are recorded on an unmodified replay. The
discovery-frozen one-statistic threshold uses suffix q10 confidence. It lowers
discovery C-PPL `78.44 -> 73.61`, but the paired interval crosses zero, and it
reverses on validation to `86.85 -> 98.91`, NLL `+.135 [-.023,.292]`.
Degeneration also rises by `.0313`. Instantaneous confidence, entropy, lexical
revision, margin, and predicted-clean instability do not explain which trigger
will be useful.

### 6.24 Native short-horizon trigger utility (EXP-102)

For each trigger, EXP-102 forks the native state into an unmodified control and
the one-step anchor intervention, pairs ancestral noise, releases the anchors,
and reads unresolved positions after `0/1/2/4` further native updates. Signal
selection uses discovery only. The winner is entropy reduction relative to the
paired control after four updates:

| Bank | Fixed step 14 | Signal-selected | NLL delta [95% CI] | Pair acc. | Spearman |
|---|---:|---:|---:|---:|---:|
| discovery | 78.44 | **64.62** | `-.194 [-.339,-.047]` | .609 | .291 |
| validation | 86.85 | **69.93** | `-.212 [-.362,-.065]` | .614 | .279 |

This is the first trigger signal that transfers. It uses neither final text nor
an external LM, showing that utility becomes visible in the intervention's
short-term causal effect rather than the pre-intervention maturity state. Raw
argmax is not yet a complete method: on validation D1 changes by only `-.0032`
and Rep-4 improves, but degeneration increases by `.0313` and prompt gain falls
by `.0159`. It is also an expensive teacher because it enumerates all triggers.

### 6.25 Selective local-utility fallback (EXP-103)

A calibration-only abstention threshold keeps fixed step 14 unless the frozen
four-step entropy signal has a large advantage. At `gamma=1.53414`, calibration
switches `5/64` trajectories and changes C-PPL `86.85 -> 79.65`, with paired
NLL `-.088 [-.191,-.012]`. D1 is unchanged, D2 and prompt gain improve, and
degeneration does not increase.

The threshold is then frozen before opening seed 2026 / offset 6000. Trigger
oracle headroom remains large (`94.91 -> 54.23`), but the selective policy
switches only `2/64` trajectories. Final C-PPL is `94.30` versus `94.91`, and
paired NLL is `-.0064 [-.0175,0]`. The quality panel remains healthy---D1
`-.0002`, D2 `+.0011`, no Rep-4 or degeneration increase, and prompt gain
`+.0044`---but the likelihood interval does not exclude zero. The safe fallback
therefore fails the final method gate by becoming too sparse. The actionable
next target is to distill the validated local entropy response into one
controller evaluation or a training-time transition objective, not to retune
another inference threshold.

### 6.26 Distilled current-state trigger controller (EXP-104)

A two-layer controller is trained on 192 trajectories from seeds 42, 123, and
2026 to predict EXP-102's four-step entropy response from eight instantaneous
event statistics and normalized trigger time. It attains training pairwise
accuracy `.628` and Pearson `.385`, but on the unopened seed-2027 calibration
bank the values fall to `.504` and `.036`. The response-prediction gate fails,
so no threshold is selected and seed 2028 remains unopened for this method.

Thus the intervention's *observed short-term causal effect* transfers, whereas
the tested pre-intervention summaries do not contain a transferable proxy for
that effect. This closes summary-state distillation without weakening the
EXP-102 teacher result.

### 6.27 Causal online response trigger (EXP-105)

At steps 8, 10, and 12, the policy may run a paired four-step anchor/control
probe from the current state. It triggers at the first measured entropy
response above a threshold and otherwise falls back to step 14. Unlike the
EXP-102/103 argmax teacher, it never evaluates a later candidate before making
an earlier decision.

On seed 2027 / offset 7000, the frozen threshold `gamma=.92161` chooses steps
`8/10/12/14` for `1/1/2/60` trajectories. C-PPL changes `84.63 -> 83.63` and
paired NLL is `-.0124 [-.0266,-.0022]`. D1 changes by `-.0013`, D2 by `+.0007`,
Rep-4 by `-.0011`, degeneration by `0`, and prompt gain by `-.0054`. All
calibration gates pass. Seed 2028 is the single untouched final test; this is
still a compute-heavy proof of signal validity rather than an efficient
sampler.

On that untouched final bank, the frozen rule again fires early on only `4/64`
trajectories (`2/1/1/60` at steps `8/10/12/14`). C-PPL changes
`85.23 -> 85.05`, but paired NLL is only `-.0022 [-.0161,.0109]`. Quality is
safe---D1 `-.0025`, D2 `+.0001`, no Rep-4 or degeneration increase, and prompt
gain `+.0015`---yet the likelihood gate fails. The method is not promoted.
EXP-103 and EXP-105 independently show the same bottleneck: a quality-safe
fallback isolates too few high-response cases to produce a stable aggregate
gain.

### 6.28 Noise-averaged response diagnostic (EXP-106)

Four independent paired probe futures do not denoise the response into a
better predictor. Pairwise accuracy changes `.605 -> .543` on seed 2027 and
`.571 -> .544` on seed 2028; pooled accuracy changes `.588 -> .544`, while
pooled Spearman falls `.226 -> .122`. The harmful final event's response is
attenuated, but so are truly beneficial events.

The original one-probe signal uses the same route-29 ancestral noise that the
actual conditional generation subsequently realizes. Its predictive content
is therefore path-specific rather than an estimate of expected intervention
utility over arbitrary future noise. This closes Monte-Carlo response
averaging and motivates direct pathwise shadow-branch selection in EXP-107.

### 6.29 Pathwise shadow branch selection (EXP-107)

At fixed step 14, EXP-107 forks anchor and control through the same five native
updates and continues from the branch with lower unresolved entropy. The
smoke test confirms that this is exact selection between the fixed-anchor and
Standard states rather than extra refinement of the chosen trajectory.

On the new seed-2029/offset-9000 panel, shadow selection chooses anchor on
`57.8%` of trajectories. It remains better than Standard C-PPL
(`89.28` versus `119.42`) but loses to always anchoring (`82.79`). Relative to
fixed anchoring, paired NLL is `+.0728 [-.0168,.1685]`; D1 changes `-.0073`
and prompt gain `-.0204`, so both likelihood and quality gates fail. The
response sign is not an anchor-versus-control value function.

A final offline upper-bound check selects among only trigger steps
`8/10/12/14` using the response teacher. The gain is significant only on seed
42 and violates D1 there; seed 2028 reverses from C-PPL `85.23` to `91.33`.
Across all five banks, fixed step 14 remains the best aggregate fixed trigger
(pooled C-PPL `85.83`). This closes multi-trigger shadow-beam implementation.

### 6.30 Deterministic ELF trigger-time headroom (EXP-108)

EXP-108 returns method discovery to ELF's native deterministic ODE. For each
real OWT prefix and exactly paired initial latent, it sweeps Unlock-4 trigger
times `.25,.30,.35,.40,.45,.50,.55,.60` around the fixed `.40` reference.

| bank / trigger | C-PPL | D1 | D2 | Rep-4 | degeneration | prompt gain | anchor fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 42 / `.25` | 349.74 | .5295 | .9026 | .0134 | .0625 | .3421 | .5945 |
| seed 42 / `.30` | 324.84 | .5278 | .9013 | .0142 | .0625 | .3861 | .7527 |
| seed 42 / `.35` | 366.07 | .5325 | .9061 | .0158 | .0312 | .3712 | .8352 |
| seed 42 / `.40` | 391.88 | .5404 | .9054 | .0210 | .0312 | .3158 | .8679 |
| seed 42 / `.45` | 449.21 | .5475 | .9116 | .0167 | .0156 | .3229 | .9072 |
| seed 42 / `.50` | 465.21 | .5496 | .9132 | .0148 | .0156 | .3094 | .9238 |
| seed 42 / `.55` | 481.93 | .5489 | .9110 | .0165 | .0156 | .3128 | .9424 |
| seed 42 / `.60` | 505.93 | .5504 | .9131 | .0151 | .0156 | .3006 | .9524 |
| seed 123 / `.25` | 352.80 | .5196 | .8880 | .0270 | .0469 | .2869 | .5850 |
| seed 123 / `.30` | 334.65 | .5259 | .8886 | .0314 | .0312 | .2848 | .7356 |
| seed 123 / `.35` | 381.29 | .5326 | .8906 | .0297 | .0312 | .2543 | .8311 |
| seed 123 / `.40` | 389.88 | .5327 | .8905 | .0291 | .0156 | .2462 | .8704 |
| seed 123 / `.45` | 455.61 | .5424 | .9018 | .0233 | .0156 | .2206 | .9036 |
| seed 123 / `.50` | 473.79 | .5413 | .9022 | .0245 | .0156 | .2179 | .9172 |
| seed 123 / `.55` | 526.68 | .5432 | .9053 | .0235 | .0156 | .2063 | .9373 |
| seed 123 / `.60` | 523.15 | .5442 | .9062 | .0231 | .0156 | .1996 | .9446 |

The unrestricted per-trajectory oracle reduces fixed-`.40` C-PPL
`391.88 -> 255.99` and `389.88 -> 247.58`, with paired mean-NLL CIs wholly
below zero. It fails quality because final-NLL selection overuses `.25/.30`:
D1 falls by `.0195/.0176`, and degeneration rises by `.0156/.0313`.

A preregistered quality-constrained audit then allows only `.40--.60`. It
retains C-PPL gains of `7.90%` and `6.59%`, again with paired CIs below zero.
The seed-123 bank passes the full gate. Seed 42 misses only the D1 threshold by
`.000176`; Rep-4 and degeneration do not increase, while prompt gain improves.
Winner histograms are `43/10/5/3/3` and `41/13/3/2/5` for
`.40/.45/.50/.55/.60`. Therefore adaptive timing is real in deterministic ELF,
but the action space must exclude premature anchors and the deployable signal
must be learned on new banks.

The narrower `.40`-versus-`.45` oracle is the cleanest target. It changes
C-PPL `391.88 -> 369.70` and `389.88 -> 368.02`; paired NLL CIs are
`[-.0967,-.0311]` and `[-.1001,-.0296]`. Both banks pass every quality gate,
with winner counts `46/18` and `44/20`. The next ELF experiment therefore asks
whether a current-state or short-horizon deterministic signal can predict this
single-checkpoint delay, rather than reopening a broad trigger sweep.

### 6.31 ELF late-trigger signal screen (EXP-110 Stage A)

The EXP-108 `.40/.45` trajectories are regenerated with exact text agreement
before extracting any feature. Mean entropy, confidence, margin, `.90` anchor
fraction, one-checkpoint token stability, and predicted-clean displacement at
the `.40` state all fail, with pooled sign AUC `.472--.539` and inconsistent
directions across banks.

The deterministic short-horizon comparison is different. At a common `.625`
checkpoint, the `.40`-minus-`.45` entropy response has pooled AUC `.664`, while
the `.45`-minus-`.40` confidence response reaches `.682`. Confidence-response
AUC is `.675` on seed 42 and `.691` on seed 123, with positive Spearman
correlation in both. This passes the preregistered signal gate and fixes the
only Stage-B score before opening seed 456. It is a causal diagnostic with
extra shadow-branch compute, not yet an efficient sampler.

Stage B rejects that signal. The new seed-456 bank has even larger binary
oracle headroom (`392.06 -> 364.14`, `7.12%`) and passes every quality gate, but
confidence-response AUC reverses to `.438` and entropy-response AUC falls to
`.465`. The seven frozen top-response policies delay `8--32` trajectories and
all worsen C-PPL to `394.92--409.66`; every paired NLL CI crosses zero. No
threshold is selected and seed 789 remains unopened for EXP-110. The replicated
fact is timing headroom, not transferability of the tested response score.

### 6.32 Token-level overlapping Unlock waves (EXP-111)

EXP-111 replaces the brittle sequence-level `.40/.45` choice with two
position-level waves. Confidence-`.90` tokens anchor at `.40625`; previously
unselected tokens that cross the same threshold join at `.46875`. Each token's
anchor expires after four native intervals. Native fixed-`.40` and the ignored
second-readout sham agree exactly.

| bank | arm | C-PPL | D1 | D2 | Rep-4 | degeneration | prompt gain | wave-2 density | revision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 789 | fixed `.40` | 358.77 | .5602 | .9038 | .0182 | .0313 | .2676 | .0000 | .1026 |
| seed 789 | Two-Wave-New | **342.87** | .5558 | .9010 | .0195 | .0469 | .2733 | .0786 | .1158 |
| seed 1011 | fixed `.40` | 485.81 | .5707 | .9077 | .0186 | .0469 | .2728 | .0000 | .1074 |
| seed 1011 | Two-Wave-New | **473.50** | .5673 | .9058 | .0208 | .0469 | .2683 | .0701 | .1205 |

Likelihood improves `4.43%` and `2.53%`. Seed 789 has paired NLL
`-.0429 [-.0789,-.0064]` but one extra degenerate sample (`+.015625`), just
above the written `.015` cutoff. Seed 1011 is quality-safe, but its paired NLL
CI `[-.0659,.0102]` crosses zero. EXP-111 therefore does not pass its strict
two-bank gate. The frozen pooled diagnostic is nevertheless significant:
C-PPL `417.00 -> 402.45` (`3.49%`) and paired NLL `-.0354
[-.0610,-.0090]`. Two-Wave-Refresh is weaker and inconsistent, so EXP-112
tests the unchanged new-token wave once at `n=128` rather than tuning it.

### 6.33 Frozen Two-Wave-New confirmation (EXP-112)

On the untouched seed-2026/offset-45000 `n=128` conditional bank, fixed `.40`
versus Two-Wave-New changes C-PPL `412.73 -> 401.58` (`2.70%`). Paired mean
NLL is `-.0264 [-.0521,-.00078]`. D1 changes `-.00019`, D2 `-.00094`, Rep-4
`+.00089`, degeneration `0`, and prompt gain `+.00517`; second-wave density is
`.0780`. The native control and ignored-readout sham both agree exactly.
Every confirmation gate passes, authorizing the frozen unconditional panel.

The paired unconditional panel keeps the same direction but does not pass the
promotion gate. Standard/fixed `.40`/Two-Wave-New U-PPL is
`276.30/208.71/204.75`; the wave improves fixed `.40` by `1.90%`, while paired
NLL is `-.0175 [-.0371,.00134]`. D1 changes `-.00169`, D2 `-.00060`, Rep-4
`+.00079`, and degeneration `+2/128=.015625` (paired exact `p=.50`); wave-2
density is `.0682`.
Therefore the safe method claim is a confirmed real-prefix conditional ELF
ODE improvement, with favorable but inconclusive unconditional evidence.

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
| Endpoint-specificity collapse | Early ambiguity, then narrow collapse | Earlier Plaid result was noise-confounded. Corrected paired-noise `n=2` smoke has self rank reach 1 only around `t=.55--.74` and a gradual entropy decline; `n=32` is running. | Pending corrected formal replication |
| Local velocity dynamics | Endpoint alignment early; collapse after token stability | Early alignment low/non-monotone; event order reverses | Boundary: stochastic finite differences are confounded |
| Rank/energy control | Top-k beats matched alternatives | Same | Confirmed |
| Residualized collective coupling | 13/16 checkpoints beat all nulls | 1/16 | Boundary: ancestral step noise likely dilutes increments |
| Async denoising | All fail, PPL `2--6x` worse | All fail, PPL `3--14.4x` worse | Confirmed negative result |

Endpoint-based/static conclusions had appeared to replicate more reliably than
adjacent-state finite differences. That statement is now provisional: Plaid
injects Gaussian noise at every ancestral step, unlike deterministic
ELF/LangFlow Euler paths, and the original endpoint-bank calibration also
failed to pair that noise. EXP-83 rebuilds the bank with common random numbers
for base and perturbed branches before any cross-architecture endpoint claim is
retained.

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
5. On ELF, hard commitment is solver-specific: a four-step lock preserves the
   replicated ODE gain and permits later revision, while ELF native SDE either
   makes it inert or loses coherence when commitment is forced earlier. This
   boundary must not be generalized to every stochastic architecture.
6. Plaid now provides the cleanest method result. Early one-step 75\%
   confidence anchors improve unconditional and real-prefix conditional PPL in
   three seeds, preserve mean D1, reduce degeneration, and revise heavily after
   release. D2 falls slightly, so this is a balanced quality result rather than
   strict dominance on every metric.
7. The exact readout sham, same-mask shuffled-content control, shared prompt
   and ancestral noise, and real 64-token prefix protocol isolate the gain as
   correct temporary lexical context rather than extra compute or a generic
   perturbation.
8. Pipeline, post-hoc local clocks, compute-matched late coupling, subset-flow
   training, and the current gated WFF pilot are not positive methods.
9. Anchor subset identity has large replicated oracle headroom on both ELF and
   Plaid, but static, lookahead, additive-graph, and the tested non-additive
   Transformer selectors all fail independent deployment gates.
10. Plaid trigger timing also has large oracle headroom. Instantaneous maturity
    statistics fail, whereas a four-step paired entropy response predicts final
    utility across banks; enumerative and selective versions still fail the
    complete deployment gate.
11. A controller trained on instantaneous summaries cannot distill the local
    response across banks. Direct causal probing is quality-safe but too sparse
    to pass its final likelihood gate.
12. Plaid response is path-specific: independent-future averaging weakens it,
    while pathwise shadow selection still cannot beat the robust fixed anchor.
    Treat response as a mechanism diagnostic, not a deployable controller.
13. Deterministic ELF ODE has replicated per-trajectory Unlock-4 timing
    headroom. Unrestricted NLL selection is quality-unsafe, while a late-only
    action space preserves roughly `7%` C-PPL headroom with minimal quality
    movement; an online selector still remains to be validated.
14. A fixed token-level overlapping `.40/.45` Unlock wave converts part of
    that headroom into a deployable conditional method: the frozen `n=128`
    confirmation improves C-PPL `2.70%` with all quality gates. Unconditional
    PPL improves `1.90%`, but its CI crosses zero, so the current claim remains
    specific to real-prefix deterministic ELF ODE.

## 10. Provenance

Primary specs:

- mechanism: `EXP-GS11`--`EXP-GS20`, `EXP-67`;
- WFF/local clocks: `EXP-60`, `EXP-GS19`;
- Pipeline: `EXP-61`, `EXP-64`;
- KD: `EXP-62`, `EXP-63`, `EXP-66`;
- hard commitment and sampler boundary: `EXP-64`--`EXP-69`, `EXP-74`, `EXP-78`.
- selector and training dead ends: `EXP-91`--`EXP-93`, `EXP-100`, `EXP-101`;
- compute-matched coupling and the Plaid method result: `EXP-94`, `EXP-95`.
- Plaid subset headroom and its failed learned selector: `EXP-99`, `EXP-100`.
- Plaid trigger headroom, native local utility, and selective fallback:
  `EXP-101`--`EXP-103`.
- Plaid response distillation and causal online probing: `EXP-104`--`EXP-105`.
- deterministic ELF Unlock-4 timing headroom: `EXP-108`.
- deterministic ELF late-trigger selector and two-wave method:
  `EXP-110`--`EXP-112`.

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
