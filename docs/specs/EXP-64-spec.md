# EXP-64 Spec: Unified Native-Recipe Method Evaluation

**Status:** DONE
**Date:** 2026-08-08
**Script:** `models/ELF-torch/experiments/probe_elf/unified_method_eval_exp64.py`

## Why this rerun is necessary

The unified comparison slide exposed missing metric cells, but simply filling
those cells from historical outputs would still be invalid. The historical
panels used different generation recipes:

- the two-pass and progressive-commit scripts used noise scale 1 and SC-CFG 1;
- the native ELF configs use noise scale 2 and SC-CFG 3;
- the Pipeline result changed sign when its noise scale was corrected;
- different scripts reported different subsets of diversity, degeneration,
  repetition, and conditioned-continuation metrics.

EXP-64 therefore revalidates every row retained on the method-comparison slide
under one matched native protocol.

## Fixed protocol

- checkpoints: ELF baseline, Broad-KD (`kd2`), Commit-KD (`kd_cr`);
- seed 42 with paired initial noise across arms;
- native initial noise scale 2;
- SC-CFG 3;
- sequence length 128, uniform ODE-32;
- 256 unconditional samples;
- 128 fixed Gutenberg prefix/continuation pairs, with a 64-token prefix;
- GPT-2-large generation perplexity for every arm;
- identical metric implementations for every arm.

## Arms

| Checkpoint | Methods |
|---|---|
| ELF baseline | standard, local-clock LTR, hard commit at `t=0.50`, confidence 0.70 |
| Broad-KD | standard, two-pass prefix, hard commit at `t=0.30`, Pipeline |
| Commit-KD | standard, hard commit at `t=0.40`, Pipeline |

Historical commit times are kept fixed. This is a native-recipe revalidation,
not a new hyperparameter search.

## Metrics

Every method reports the following on unconditional generations:

- Gen.PPL;
- corpus Distinct-1 and Distinct-2;
- mean within-sample repeated 4-gram rate;
- degeneration fraction using the established 20% maximum-unigram share,
  empty-output, and non-ASCII corruption rule;
- mean maximum unigram share, per-sample unique-word ratio, and an explicit
  unigram-collapse rate, because repeated high-frequency words can evade a
  repeated-4-gram detector;
- mean decoded length;
- method call count and four audit samples.

Every method is additionally run on the same prefix-conditioned continuation
panel. The same metrics are computed on the generated suffix, together with
token-level ROUGE-L F1 against the held-out true suffix.

## Decision rule

A method is a credible positive result only if, relative to its own matched
standard arm:

1. Gen.PPL improves;
2. diversity does not collapse and repetition/degeneration do not increase
   materially;
3. conditioned ROUGE-L does not decrease materially;
4. the sign is obtained under the native noise and guidance recipe.

If the old hard-commit gain reverses under the native recipe, remove it as a
positive result from the main talk. If PPL improves but diversity or
conditioned quality collapses, retain it only as evidence of metric hacking or
premature locking.

## Results

All values below are means over generation seeds 42, 123, and 456. The
parenthesized PPL change is computed against the matched standard arm for the
same checkpoint and seed before averaging.

| Checkpoint · method | Gen.PPL (delta) | D1 | D2 | Deg. | Cond. R-L | Calls |
|---|---:|---:|---:|---:|---:|---:|
| ELF base · standard | 273 (0) | .419 | .864 | .020 | .087 | 32 |
| ELF base · local-clock LTR | 2253 (+1979) | .537 | .882 | .038 | .076 | 32 |
| ELF base · hard commit | 238 (-35) | .416 | .861 | .020 | .087 | 33 |
| Broad-KD · standard | 1371 (0) | .459 | .950 | .211 | .060 | 32 |
| Broad-KD · two-pass prefix | 1343 (-27) | .430 | .949 | .212 | .055 | 64 |
| Broad-KD · hard commit | 677 (-694) | .363 | .847 | .085 | .096 | 33 |
| Broad-KD · Pipeline | 1557 (+186) | .467 | .959 | .086 | .026 | 31 |
| Commit-KD · standard | 989 (0) | .443 | .963 | .296 | .064 | 32 |
| Commit-KD · hard commit | 727 (-262) | .423 | .911 | .112 | .088 | 33 |
| Commit-KD · Pipeline | 1234 (+246) | .454 | .961 | .112 | .027 | 31 |

The PPL deltas are stable across seeds:

- ELF hard commit: `-35.4 +/- 1.8`;
- Broad-KD hard commit: `-693.9 +/- 20.0`;
- Commit-KD hard commit: `-261.5 +/- 13.2`;
- Broad-KD Pipeline: `+186.3 +/- 28.4`;
- Commit-KD Pipeline: `+245.9 +/- 42.1`.

The plus/minus values above are standard deviations across the three seeds.

## Quality audit

The complete panel changes the interpretation of hard commitment rather than
eliminating it:

- **ELF base:** the modest PPL gain is clean. D1/D2, degeneration, and
  conditioned ROUGE-L are effectively unchanged.
- **Broad-KD:** PPL and conditioned ROUGE-L improve strongly, but D1 drops
  `.459 -> .363`, D2 drops `.950 -> .847`, per-sample unique-word ratio drops
  `.879 -> .742`, and unigram-collapse rate rises `.9% -> 4.0%`. Sample audit
  finds visible high-frequency loops. This is a quality trade-off, not a clean
  win.
- **Commit-KD:** PPL and conditioned ROUGE-L improve, while D1/D2 decrease more
  modestly and no sample crosses the 20% unigram-collapse threshold. This is
  the cleanest current positive method signal.
- **Two-pass prefix:** its small unconditional PPL gain does not justify twice
  the model calls; conditioned PPL worsens and ROUGE-L drops.
- **Pipeline:** both KD checkpoints worsen in unconditional PPL and lose more
  than half of conditioned ROUGE-L. The positive method claim is closed.
- **Local-clock LTR:** fails catastrophically under the matched native panel.

The stricter 20% unigram criterion was necessary: the former 35% rule missed
obvious high-frequency loops. The raw JSON outputs, including every generated
text for audit, are stored on the experiment server under
`models/ELF-torch/results/exp64_unified_method_eval/`.

## Final verdict

Hard commitment is the only tested family that survives native-recipe,
three-seed, full-panel evaluation. Its effect is checkpoint-dependent:
Commit-KD is the strongest clean signal, while Broad-KD motivates calibrated,
revisable commitment rather than irreversible freezing. Pipeline, two-pass
prefix, and post-hoc local clocks should not be presented as positive methods.
