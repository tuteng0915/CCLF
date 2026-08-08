# EXP-67 Spec — Hard-Commit Mechanism Audit

**Status:** ODE MECHANISM PANEL COMPLETE / NATIVE-SDE GENERALIZATION OPEN
**Priority:** P1
**Script:** `models/ELF-torch/experiments/probe_elf/hard_commit_mechanism_exp67.py`

## Question

When hard commitment improves final text quality, does it help because reliable
local proposals become useful context for unresolved positions, or because it
prematurely locks easy high-frequency tokens and happens to lower PPL?

## Paired trajectories

Use the frozen EXP-65 configuration for every checkpoint. From the same
initial-noise bank, collect standard and hard-commit trajectories on the same
33-point time grid. The primary panel uses length 128 and at least 48
trajectories; any headline mechanism result must be checked on the surviving
length-1024 configuration.

Record at every checkpoint:

- sampler state and predicted-clean state;
- token posterior, top-1 token, confidence, entropy, and final-token margin;
- the hard-commit mask and token written at `t_c`;
- final decoded sequence and text-quality diagnostics.

## Primary timing estimands

For position `i`, define first agreement with its own final endpoint,
three-checkpoint stable agreement, and revision count:

```text
tau_first(i)  = min {t_k : y_i(t_k) = y_i(1)}
tau_stable(i) = min {t_k : y_i(t_j) = y_i(1) for j=k,k+1,k+2}
N_rev(i)      = sum_k 1[y_i(t_k) != y_i(t_{k-1})].
```

Report paired differences for all positions and separately for positions that
are committed at `t_c` versus those that remain unresolved. The desired
signature is

```text
tau_stable decreases, while tau_first changes little.
```

Earlier first guesses without lower stable time do not support the
coordination account.

## Anchor-to-suffix causal test

At the commit checkpoint, construct three matched continuations from the same
state and future solver path:

1. **natural:** no tokens are written;
2. **true anchors:** write the selected high-confidence proposals;
3. **shuffled anchors:** preserve the number, confidence distribution, token
   frequency bins, and positions, but permute selected token identities across
   sequences.

Measure only positions that were uncommitted at `t_c`:

- change in margin toward their eventual endpoint;
- stable-time shift;
- final token agreement and text quality.

True anchors must beat both natural and frequency-matched shuffled anchors to
support a contextual-coordination mechanism.

## Premature-locking controls

Report for committed positions:

- unigram frequency percentile and function/content-token composition;
- fraction disagreeing with the paired standard endpoint;
- unigram collapse, repeated 4-grams, D1/D2, and maximum word share;
- results stratified by confidence decile.

This stratification tests whether the Broad-KD diversity loss is concentrated
in lower-confidence or high-frequency commitments and directly informs a
future rollback/remasking policy.

## Decision rule

- **Coordination support:** true anchors selectively improve unresolved
  positions, reduce stable time/revisions, and beat shuffled anchors without a
  collapse penalty.
- **Premature locking:** apparent gains come mainly from earlier first guesses,
  high-frequency commitments, or reduced diversity, with no true-anchor
  advantage over the matched control.
- **Mixed result:** report checkpoint-specific trade-offs and use the observed
  confidence/frequency strata to define revisable commitment.

## Completed ODE mechanism panel

The primary length-128 panel uses 48 paired trajectories, ODE-32, and the
frozen `t_c=0.40`, `gamma=0.60` intervention. At the fork, approximately 95%
of positions are selected as anchors. Shuffled controls preserve anchor
positions and match confidence and frequency quartiles, while permuting the
continuous predicted-clean anchor vectors. Only 3.5--4.7% of shuffled anchors
retain the same decoded token.

### Effects on positions unresolved at the fork

All entries are paired changes relative to natural rollout; negative timing
and revision deltas are favorable.

| Checkpoint | Intervention | `Delta tau_first` | `Delta tau_stable` | `Delta N_rev` |
|---|---|---:|---:|---:|
| ELF baseline | true anchors | **-0.032** | **-0.014** | **-0.62** |
| ELF baseline | shuffled anchors | +0.069 | +0.068 | +1.36 |
| continued-training control | true anchors | **-0.038** | **-0.025** | **-0.57** |
| continued-training control | shuffled anchors | +0.050 | +0.047 | +1.21 |
| Early-KD | true anchors | **-0.028** | **-0.031** | **-0.42** |
| Early-KD | shuffled anchors | +0.080 | +0.066 | +1.44 |

True anchors therefore accelerate both first and stable commitment and reduce
revision, whereas matched but misplaced anchors consistently delay resolution.
The effect is not explained by freezing an arbitrary set of confident or
high-frequency states.

### Endpoint identity and immediate lexical evidence

The intervention often changes the unresolved positions' eventual token:
agreement with the natural endpoint is only `.265`, `.296`, and `.353` after
true anchoring, versus `.006`, `.007`, and `.020` after shuffled anchoring.
Consequently, margin to the *natural* endpoint alone is not a valid causal
estimand after the fork. We additionally measure each branch's first-step
margin toward its own eventual endpoint.

| Checkpoint | True minus natural own-endpoint margin | Shuffled minus natural | True minus shuffled |
|---|---:|---:|---:|
| ELF baseline | **+4.56** | -2.17 | **+6.73** |
| continued-training control | **+3.89** | -2.59 | **+6.47** |
| Early-KD | **+3.44** | -3.92 | **+7.35** |

Thus a correct anchor set immediately strengthens evidence for the coherent
future induced by that context; shuffled anchors weaken it. This supports
contextual coordination, but not the stronger claim that commitment merely
speeds transport toward a predetermined natural-rollout endpoint.

### Text quality sanity check

| Checkpoint | Natural PPL | True-anchor PPL | Shuffled-anchor PPL |
|---|---:|---:|---:|
| ELF baseline | 281 | **209** | 4,176 |
| continued-training control | 262 | **206** | 4,139 |
| Early-KD | 206 | **166** | 3,761 |

True anchoring improves the within-panel PPL comparison without material
change in D1/D2, repeated 4-grams, or unigram-collapse rate. Shuffled text has
extremely poor PPL despite superficially high lexical diversity, showing why
diversity alone cannot diagnose coherence. These length-128 PPL values should
not be compared directly with the length-1024 method table.

## Decision and scope

The deterministic ODE panel supports the coordination account: meaningful,
position-correct anchors causally accelerate and stabilize unresolved lexical
decisions, while matched shuffled anchors do the opposite. The more precise
claim is **context-dependent endpoint selection**, not faster travel to a
fixed pre-existing endpoint.

This remains an ODE mechanism result. EXP-68 showed that the same frozen policy
commits about 99% of positions at the first native logit-normal SDE crossing
and has negligible quality effect, so no sampler-independent mechanism claim
is currently warranted. A native-SDE mechanism test requires recalibrating the
commit time so that a meaningful unresolved set remains; it must be treated as
a new intervention rather than retroactively upgrading this result.
