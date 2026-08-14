# EXP-100 Spec — Joint Temporary-Anchor Selector

**Status:** DONE / NON-ADDITIVE SELECTOR FAILS FINAL GATE
**Purpose:** if Plaid has replicated subset headroom, learn a deployable,
non-additive scorer that selects a jointly useful temporary-anchor subset from
the current trigger state without access to final text.

## Scope isolation

Freeze the Plaid intervention before learning selection:

```text
trigger = native step 14
horizon = one solver interval
density = EXP-99's frozen passing density
content = position-aligned predicted-clean state
release = full joint refinement
```

EXP-100 changes only subset identity. It does not jointly tune trigger,
density, horizon, anchor content, or the Plaid backbone.

## Utility banks

Build trajectory-disjoint train, validation, and final-test banks. For each
trajectory, branch `M` candidate masks from one shared trigger state. Initial
noise, prompt, and every Plaid ancestral-noise draw are identical across masks.
Offline labels may use final conditional NLL,

```text
U(A; s) = NLL_standard(s) - NLL_anchor(A; s),
```

but inference features must be available at the trigger. Save compact token
features rather than full vocabulary logits:

- noisy state, self-conditioning state, and predicted-clean state;
- top-1 confidence, entropy, and top-1/top-2 margin;
- normalized position and prefix/eligible indicators;
- candidate mask and final candidate NLL;
- candidate revision, ROUGE-L, D1/D2, Rep-4, and degeneration.

No trajectory or alternate mask from the same trajectory may cross splits.
The final test bank remains unopened until architecture and hyperparameters are
frozen.

## Model

An independent token score is explicitly disallowed as the main model. Score
the full candidate set with selected/unselected mask embeddings and a small
Set Transformer or sequence Transformer:

```text
token features + candidate-membership embedding
    -> joint self-attention over the sequence
    -> selected/unresolved pooled interaction
    -> scalar subset utility
```

The training loss is listwise within trajectory:

```text
L_rank = KL(softmax(-NLL(A_m)/T_y) || softmax(Score(A_m)/T_s)).
```

Tune only on train/validation. Report grouped Spearman, tie-aware pairwise
accuracy, selected conditional NLL, and the fraction of trajectories where the
selected mask beats one random mask and top confidence.

## Baselines and final gate

At identical trigger, density, horizon, and denoiser calls, compare:

1. Standard Plaid;
2. readout-only sham;
3. one random mask;
4. top confidence;
5. spatially stratified random;
6. frozen joint selector;
7. shuffled content using the selector's mask;
8. oracle best-of-M, diagnostic only.

Promote only if the frozen selector:

- beats one random mask in at least `2/3` final seeds;
- improves mean conditional NLL over random with a trajectory-bootstrap 95%
  interval excluding zero;
- does not materially regress prompt gain, D1, Rep-4, or degeneration relative
  to the best non-oracle anchor baseline;
- adds no denoiser step and reports selector FLOPs/readout calls;
- retains substantial post-release revision, ruling out irreversible locking.

If EXP-99 passes but this model fails, close confidence/static/additive/joint
reranking as a method family and move to adaptive trigger timing. Do not use
the oracle subset as a reported inference method.

## Stage-1 implementation

- `build_plaid_selector_features_exp100.py` replays only the shared prefix of
  an EXP-99 bank, reconstructs its masks from independent mask seeds, and saves
  compact trigger-time features with the existing final-NLL labels;
- `train_joint_anchor_selector_exp100.py` trains the fixed two-layer joint
  sequence scorer with a within-trajectory listwise objective and early stopping
  on a trajectory-disjoint validation bank;
- `merge_plaid_selector_banks_exp100.py` checks protocol invariants and exact
  token-panel disjointness before merging training shards;
- `eval_joint_anchor_selector_exp100.py` is the only path allowed to open the
  frozen final bank after a checkpoint has been selected; it also reports the
  candidate index frozen from training data and the selector's chosen-index
  histogram as selection-bias nulls;
- density `.75` is primary because it is EXP-95's frozen balanced operating
  point and passes EXP-99 on discovery and validation. Density `.50` remains a
  headroom control, not a simultaneously tuned hyperparameter.

The first `n_train=64` pilot stays at chance on validation. Scaling to 320
trajectory-disjoint training examples produces an apparent `-.070` nats / about
`-5.5` C-PPL selected-mask improvement, but pairwise accuracy and Spearman stay
at chance and all three bootstrap intervals touch zero. A fixed-index audit
shows the selected PPL nearly matches the validation-local best candidate index,
not the index frozen on training. Three checkpoints are frozen and a three-bank
final audit is negative. The only optimization seed with favorable mean NLL
has pooled delta `-.0167 [-.0658,.0316]` nats and chance pair accuracy on all
three banks; the other two seeds worsen pooled NLL and improve in `0/3` banks.
The training-frozen fixed candidate index matches or beats the apparent primary
selector gain. Close this selector family and do not run a selected-text quality
panel after the NLL/ranking gate failure.
