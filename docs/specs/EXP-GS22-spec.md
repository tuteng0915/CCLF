# EXP-GS22 Spec — Order-Controlled Asynchronous Denoising Ablation (P2)

**Status**: PLANNED  
**Priority**: P2 — causal test before Wavefront Flow Forcing training  
**Models**: ELF baseline first; LangFlow only after the intervention is defined natively  
**Proposed script**: `experiments/interventions/eval_async_schedule.py`  
**Output**: `results/interventions/<model>/<checkpoint>/async_schedule_<label>.{json,npz}`

## 1. Hypothesis

If synchronous denoising creates a coordination bottleneck, then assigning
different effective progress levels to positions should help even when the
ordering is not linguistic. If only left-to-right helps, language direction is
important. If confidence ordering helps most, reliable anchors are the key.

## 2. Exact inference-time intervention

Existing ELF checkpoints accept a scalar global time, not a vector of local
times. Therefore this experiment must **not** claim that the pretrained model
natively evaluates `v(z, tau_vector)`.

At global step `s_k`:

1. run the ordinary model at scalar `s_k` and obtain `xhat_k`;
2. estimate the noise component under the native scalar schedule;
3. perform the ordinary global solver update;
4. reconstruct each position to its scheduled local progress
   `tau_i(s_{k+1})` using the same `(xhat_i, epshat_i)` pair;
5. feed the resulting heterogeneous state to the next scalar-time model call.

This is a controlled state intervention with train--test mismatch, not the
final WFF sampler. Save the exact reconstruction equations in code comments
for each architecture.

## 3. Schedule family

For sequence ordering rank `q_i` and width `Delta`:

```text
tau_i(s) = clip(s + Delta * (1 - 2*q_i/(L-1)), 0, 1)
```

Arms with the same multiset of local times:

1. synchronous (`Delta=0`);
2. left-to-right;
3. right-to-left;
4. fixed random permutation per sequence;
5. reversed fixed random permutation;
6. block-random ordering;
7. confidence-adaptive ordering.

For adaptive ordering, freeze the ranking for a minimum of 2 solver steps to
avoid jitter. Confidence must be measured by native entropy, not correctness.

`Delta` sweep: `{0.05, 0.10, 0.20, 0.30}` on normalized log-SNR progress.

## 4. Fairness controls

- identical initial noise across all arms;
- identical number of backbone calls;
- identical local-time multiset for all non-adaptive orderings;
- three generation seeds;
- paired evaluation at sequence level;
- no hard token freezing;
- final 4 synchronous global-refinement steps in every asynchronous arm;
- include a state-norm-matched random perturbation arm to separate asynchrony
  from generic off-manifold noise.

## 5. Metrics

Generation:

- Gen.PPL with a documented external LM;
- MAUVE or equivalent distributional metric;
- distinct-n, repetition, sequence length, degenerate-output rate;
- wall-clock and number of model evaluations.

Mechanism:

- `tau_first` and `tau_stable`;
- oracle--rollout token gap;
- endpoint velocity specificity from GS17;
- branch entropy from GS19;
- controlled collective peak from GS21;
- premature locking and later token revision.

Primary desired signature:

```text
tau_stable decreases
tau_first changes little
generation quality does not degrade
```

## 6. Scale

Smoke test:

- 64 samples, one seed, check state norms and decoded text.

Pilot:

- 256 paired samples, 3 seeds;
- 32 solver steps;
- all orderings at `Delta in {0.10,0.20}`.

Formal:

- 1000 samples per retained arm;
- at least two step budgets;
- hierarchical paired bootstrap by initial-noise sample and seed.

## 7. Decision table

| result | interpretation |
|---|---|
| LTR only wins | linguistic directionality matters |
| LTR, RTL, and random all win | breaking synchrony/symmetry is sufficient |
| confidence-adaptive wins | reliable anchors matter more than fixed order |
| quality improves but dynamics do not | generic sampler heuristic, not mechanism validation |
| dynamics improve but quality degrades | coordination changed, but intervention is off-manifold |
| all async arms fail | no evidence for inference-time asynchrony; do not train WFF yet |

Only after a positive paired result should heterogeneous-time training be
implemented as Wavefront Flow Forcing.

