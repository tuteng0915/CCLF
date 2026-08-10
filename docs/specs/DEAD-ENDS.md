# CCLF Dead Ends, Invalid Protocols, and Superseded Results

This ledger prevents old exploratory results from silently returning to the
paper. “Dead end” does not always mean the experiment was useless: several
negative results were scientifically valuable. It means the original claim or
protocol should not be treated as active.

## 1. Invalid numbers — never cite

| experiment | reason | replacement / action |
|---|---|---|
| EXP-33/34/35 | ungated decode self-conditioning produced degenerate repetitive text; PPL became meaningless | use gated and factorial EXP-36v2 |
| original EXP-14 headline | used cosine-nearest `xhat` rather than the correct decode path; top-1 flips were mislabeled commitment | use EXP-14v2 |
| original LangFlow topic nearest-centroid results in GS2/GS4/GS6/GS14 | squared Euclidean nearest centroid was dominated by clean/rollout norm mismatch; some scripts also had inlined stale code | use cosine-based reruns documented in the current specs |
| original EXP-27 frequency proxy | GPT-2 token ID is not a frequency measure | use EXP-27v2 with measured OWT frequency |

## 2. Superseded protocols — retain only as history

| old experiment | problem | corrected result |
|---|---|---|
| EXP-05 batch-shuffle prior | mixed wrong-instance posterior into the prior | EXP-05v3 global null prior |
| EXP-07 original split | position-level train/test leakage | EXP-07v2 document-level split |
| EXP-11 original branching scale | perturbation scaling was mislabeled and no eta sweep was performed | EXP-11v2 per-position scaling and eta sweep |
| EXP-15 parameter L2 story | parameter norm change is not functional importance; original layer ratios were inaccurate | EXP-15v2 module decomposition plus later patching experiments |
| EXP-25 function-word timing | frequency/surprisal confounded the apparent coarse-to-fine effect | EXP-25v2/27v2 frequency-controlled results |
| EXP-26 neighbor analysis | far risk set collapsed and common causes were uncontrolled | EXP-26v2; causal coordination remains unresolved |
| EXP-30 original LangFlow probe | skip-input asymmetry and missing MLP/CI controls | EXP-30v2 |
| EXP-31/31b single-seed DF | seed and degeneration concerns | EXP-31v2 multi-seed result |
| EXP-36 original interaction | incomplete factorial design | EXP-36v2 |
| EXP-47 intermediate SC | custom pipeline and metric bugs | EXP-48/54 standard pipeline |

## 3. Rejected hypotheses — valid negative evidence

| hypothesis | decisive evidence | current interpretation |
|---|---|---|
| global/topic structure forms before lexical structure | GS1 ordering failed; GS11 showed pooling confound; GS12 removed low-rank structure claim | do not continue topic/sentence probe hierarchy |
| structure lives specifically in a centered low-rank mode | GS12 mean-only matched or exceeded low-rank POS prediction | coarse statistics are mean-explainable; exact token readout survives in the complement |
| function words reveal a universal coarse-to-fine transition | EXP-25v2/27v2 showed frequency dominates after control | token frequency is the safer explanation |
| directional spatial bootstrapping from the observational risk sets | EXP-09v2/28 had incomparable or tiny risk sets and inconsistent direction | do not claim left-to-right propagation from these experiments |
| signed topic direction linearly adds lexical evidence | GS13 formal run was U-shaped; LangFlow was only one-sided significant | retain only axis-specific context effect |
| GS15 negative chord excess proves late endpoint selection | curved transport can also remain below the chord | use GS15 descriptively; GS16--GS17 decide the mechanism |
| synthetic D1/D3 fine-tuning is a viable method | EXP-51 showed D1 collapse and D3 partial degeneration | do not resume without real OWT data and a fresh training design |
| Pipeline mainly fails because it queries a shared average clock | EXP-70 true-local-clock oracle is no better, while mixed-state error is 3--4x clock error | heterogeneous context is the dominant failure; do not sweep the discrete Pipeline schedule |
| repeated synchronized soft prefix leadership is an efficient sampler | EXP-71 correct-content arms all lose to compute-matched ODE-64 and LTR never beats RTL/random | keep the shuffled-content contrast as mechanism evidence only |
| current deep local-time injection learns a native wave | EXP-72 LTR/RTL velocity cosine remains 1.000 and the LTR interaction worsens by +20.5 PPL | stop at step 500; do not diagnose exposure bias or launch EXP-73 until a model first learns a functional local clock |
| predicted-clean input replacement canonicalizes heterogeneous Pipeline context | EXP-75 lowers PPL partially but leaves vector error unchanged/worse and remains far from coherent generation | correct content is useful, but simple replacement is not a shared dynamical coordinate system |
| isolated asynchronous block transitions compose after clock bootstrapping | EXP-76 learns a functional clock, yet all EXP-77 fill/drain arms have PPL `3400--3900` | close block-wave distillation in the current architecture; do not sweep schedules or unroll length |

## 4. Low-value or appendix-only branches

These are not invalid, but should not consume core-paper time:

- EXP-09v2 and EXP-28 directional summaries;
- EXP-29 kNN word visualization, due centroid and cherry-picking sensitivity;
- topic/sentence cosine probes whose dynamic range is saturated;
- nominal-`t` comparisons between ELF and LangFlow;
- raw-state CKA on LangFlow when it is saturated near one;
- more static probe-capacity or representation visualizations without a new
  causal decision;
- EXP-45/46 unless the self-conditioning method line is explicitly revived as
  a separate paper.

## 5. Negative results worth retaining

Do not delete these from the record:

- GS15's negative chord-relative result motivated the current geometric audit;
- EXP-37c and ungated-SC failures reveal severe train--test mismatch;
- EXP-51 shows that synthetic fine-tuning can destroy generation while making
  an auxiliary loss look successful;
- PT3's frequency-matched control shows that an apparently token-specific
  direction may be explained by frequency;
- GS13's U-shaped response prevents an unjustified linear semantic-direction
  story.

The rule is: cite a negative result for the boundary it establishes, but do not
rebrand the rejected hypothesis as an active mechanism.
