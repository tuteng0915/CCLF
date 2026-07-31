# EXP-GS20 Spec — Rank- and Energy-Matched Residual Controls (P1)

**Status**: PLANNED  
**Priority**: P1 — make the GS12 high-rank residual claim identifiable  
**Models**: ELF baseline; optional LangFlow replication  
**Proposed script**: `experiments/global_state/analyze_rank_matched_modes.py`  
**Output**: `results/global_state/<model>/<checkpoint>/rank_matched_modes_<label>.{json,npz}`

## 1. Confound

GS12 compares a rank-8 component with its much larger complementary residual.
The residual keeps most dimensions and often most energy, so higher token
recovery is not by itself evidence that lexical identity is specifically
high-rank.

## 2. Decomposition

For centered representation `U_c = U - mean_position(U)` with SVD
`U_c = U S V^T`, evaluate equal-dimensional subspaces:

- top-`k`;
- bottom-`k`;
- middle-`k`;
- random `k`-dimensional subspace;
- token-direction-matched random subspace, if feasible.

Rank sweep:

```text
k in {1, 2, 4, 8, 16, 32, 64, 128}
```

Run separately on raw oracle states and predicted-clean states.

## 3. Two complementary tests

### A. Reconstruction test

Add the same position mean to every projected component and feed the
reconstructed state through the native model/readout.

Report:

- native token accuracy;
- terminal-token margin;
- POS-histogram R2;
- reconstruction energy.

### B. Causal removal test

Starting from the native state, remove an equal-dimensional component:

```text
U_drop_top_k
U_drop_bottom_k
U_drop_random_k
```

Measure change in native token margin and recovery. This is more informative
than asking whether an OOD reconstruction works.

## 4. Matching rules

Run both:

1. **rank matched**: every arm retains/removes exactly `k` dimensions;
2. **energy matched**: choose component widths or rescale components to match
   retained Frobenius energy within 2%.

Never compare top-8 against the entire `d-8` residual as the main result.

## 5. Statistics

- `n_sequences>=128`;
- complete preregistered time grid;
- 10 random subspaces per `(sequence,t,k)`;
- sequence-level bootstrap confidence intervals;
- full curves, not two representative checkpoints.

## 6. Decision rule

The strong "high-rank lexical code" claim is supported only if:

1. top-`k` underperforms rank- and energy-matched random/middle/bottom
   subspaces over a substantial `k` range; and
2. removing distributed non-top components harms token margins more than
   removing energy-matched top components.

Otherwise use the narrower GS12 claim:

> the leading rank-8 centered component is insufficient, while the
> complementary residual retains native token readability.

