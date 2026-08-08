# EXP-67 Spec — Hard-Commit Mechanism Audit

**Status:** QUEUED AFTER EXP-66
**Priority:** P1

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
