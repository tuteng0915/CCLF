"""EXP-PT10: Transition Failure Predictors.

Turns the 6-way failure taxonomy from EXP-PT2 into a falsifiable predictive
model: a simple, interpretable multinomial logistic regression over
per-position features, with a SEQUENCE-grouped (not position-grouped)
train/val split -- position-level splits leak information across positions
in the same document (this exact mistake, and its fix, is documented in this
project's EXP-07 -> EXP-07v2 history; applying the lesson proactively here).

This script does NOT run the model -- it's pure post-hoc analysis over
already-computed EXP-PT1/PT2/PT3 outputs, so it needs no GPU.

Predictors implemented (of the suite doc's longer list):
  - token frequency (proxy: frequency within this probing sample itself, NOT
    the true training-corpus frequency -- same caveat as EXP-20)
  - function/content word (heuristic closed-class word list on the decoded
    token string, matching the approach used in EXP-08/EXP-08v2)
  - token string length (subtoken-fragmentation proxy)
  - position (fraction of sequence length)
  - prior-mode advantage: m_raw(t_min) = ell(y) - ell(f) at the earliest t
    (from EXP-PT2's margin_trajectory output)
  - initial rank (rank of y at t_min, from EXP-PT2)
  - initial velocity alignment: a_clean(t_min), a_tok(t_min) (from EXP-PT3;
    only defined where EXP-PT3 had a valid centroid direction -- positions
    without one are dropped from the regression, see summary for the
    resulting sample size)

NOT implemented (need infrastructure this suite doesn't have yet):
  contextual surprisal (needs an external LM), local context strength (needs
  EXP-PT4), oracle-rollout state distance (needs EXP-PT7), self-conditioning
  norm (not meaningful in this Protocol-A-only, zero-SC setup), Jacobian /
  local gain estimate (needs EXP-PT6's perturbation branching).

Usage:
    conda run -n elf python experiments/phase_transition/analyze_failure_predictors.py \\
        --labels_npz results/phase_transition/elf/baseline/transition_failure_labels_full.npz \\
        --margin_npz results/phase_transition/elf/baseline/margin_trajectory_raw_full.npz \\
        --velocity_npz results/phase_transition/elf/baseline/velocity_alignment_raw_full.npz \\
        --tokenizer t5-small \\
        --out_dir results/phase_transition/elf/baseline --label full
"""

import argparse
import json
from pathlib import Path

import numpy as np

FUNCTION_WORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    "in", "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "of", "off", "over", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "so", "yet", "if", "because", "as", "until", "while",
    "not", "no", "yes", "there", "here", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "than", "too", "very", "s", "t", "just",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_npz", required=True)
    p.add_argument("--margin_npz", required=True)
    p.add_argument("--velocity_npz", default=None, help="omit to skip velocity-alignment features")
    p.add_argument("--context_ablation_json", default=None,
                    help="EXP-PT4's context_ablation_<label>.json, for the local-context-strength "
                         "feature (omit to skip it)")
    p.add_argument("--tokenizer", required=True, help="e.g. t5-small or gpt2")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--min_class_frac", type=float, default=0.01,
                    help="failure categories below this fraction are merged into 'other'")
    p.add_argument("--val_frac", type=float, default=0.3, help="fraction of SEQUENCES held out for validation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.labels_npz).stem

    lab = np.load(args.labels_npz, allow_pickle=True)
    labels = lab["labels"]  # (N,L) str
    tau_e, tau_b = lab["tau_e"], lab["tau_b"]
    gt_ids, f1 = lab["gt_ids"], lab["f1"]
    N, L = labels.shape

    marg = np.load(args.margin_npz)
    ell_gt0, ell_f10 = marg["t0_ell_gt"], marg["t0_ell_f1"]
    rank_raw0 = marg["t0_rank_raw"]
    m_raw0 = ell_gt0 - ell_f10

    has_velocity = args.velocity_npz is not None
    if has_velocity:
        vel = np.load(args.velocity_npz)
        a_clean0 = vel["t0_a_clean"]
        a_tok0 = vel["t0_a_tok"]
        valid_mask = vel["valid_mask"].astype(bool)
    else:
        valid_mask = np.ones((N, L), dtype=bool)

    # Feature: local-context-strength, merged in from EXP-PT4's already-computed
    # context-ablation data (results/phase_transition/<model>/<ckpt>/context_ablation_<label>.json).
    # PT4 measures accuracy only at a sparse set of PROBE positions (n_probes,
    # spaced `probe_spacing` apart) and only reports it aggregated to
    # PER-SEQUENCE granularity (acc_per_seq: (T, N), already averaged across
    # all probes in that sequence) -- there is no per-individual-probe-position
    # accuracy saved. So this is necessarily a PER-SEQUENCE feature (how much
    # does this sequence as a whole rely on local context), broadcast to every
    # position in that sequence, NOT a genuine per-position predictor like the
    # other features here. PT4 and this script's underlying samples are the
    # same n_samples=N sequences in the same order (both call the same
    # adapter's `load_owt_sequences`, which streams the first N examples from
    # a fixed dataset in a fixed order, independent of seq_len -- verified by
    # inspecting elf_adapter.py/langflow_adapter.py), so per-sequence index
    # alignment is valid. ⚠️ For LangFlow specifically, PT4's context-ablation
    # rerun used seq_len=1024 (post rigor-audit fix, for 26 probes) while this
    # script's LangFlow data uses seq_len=128 -- the underlying documents are
    # still the same (same deterministic stream order), but PT4's "local
    # context gap" for LangFlow reflects probes spread across up to 1024
    # tokens, a longer effective span than the L=128 this script analyzes;
    # this doesn't invalidate the per-sequence value but means it isn't
    # measuring "context strength within the exact same window" for LangFlow
    # as it does for ELF (whose PT4 data is also seq_len=1024, matching this
    # script's own seq_len=1024 -- no such mismatch there).
    has_context = args.context_ablation_json is not None
    if has_context:
        with open(args.context_ablation_json) as f:
            ctx = json.load(f)
        full_acc = np.array(ctx["conditions"]["full_context"]["acc_per_seq"])  # (T_ctx, N)
        local_acc = np.array(ctx["conditions"]["local_window_r1"]["acc_per_seq"])  # (T_ctx, N)
        assert full_acc.shape[1] == N, (
            f"EXP-PT4 context-ablation N={full_acc.shape[1]} != this script's N={N}; "
            "cannot align sequences, re-check that both used the same --n_samples")
        local_context_gap_per_seq = (full_acc - local_acc).mean(axis=0)  # (N,) -- avg over t
        local_context_gap = np.tile(local_context_gap_per_seq[:, None], (1, L))  # (N,L) broadcast
        print(f"[PT10] Loaded local-context-strength feature from {args.context_ablation_json} "
              f"(per-sequence, broadcast to all L={L} positions; mean gap="
              f"{local_context_gap_per_seq.mean():.4f})")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Feature: frequency of gt token WITHIN this sample (proxy, not true
    # training-corpus frequency -- see docstring / EXP-20 precedent).
    flat_gt = gt_ids.reshape(-1)
    uniq, counts = np.unique(flat_gt, return_counts=True)
    freq_map = dict(zip(uniq.tolist(), counts.tolist()))
    log_freq = np.vectorize(lambda t: np.log(freq_map.get(int(t), 1) + 1))(gt_ids)

    # Feature: function/content word + token string length, via decoded token text.
    uniq_ids = np.unique(gt_ids)
    decode_cache = {}
    for tid in uniq_ids:
        s = tok.convert_ids_to_tokens(int(tid))
        clean = s.replace("▁", "").replace("Ġ", "").strip().lower()
        decode_cache[int(tid)] = clean
    is_func = np.vectorize(lambda t: float(decode_cache.get(int(t), "") in FUNCTION_WORDS))(gt_ids)
    tok_len = np.vectorize(lambda t: float(len(decode_cache.get(int(t), ""))))(gt_ids)

    position_frac = np.tile(np.linspace(0, 1, L), (N, 1))

    feature_names = ["log_freq", "is_function_word", "token_len", "position_frac",
                      "prior_mode_advantage", "initial_rank"]
    feats = [log_freq, is_func, tok_len, position_frac, m_raw0, rank_raw0.astype(np.float64)]
    if has_velocity:
        feature_names += ["a_clean_t0", "a_tok_t0"]
        feats += [a_clean0, a_tok0]
    if has_context:
        feature_names += ["local_context_gap"]
        feats += [local_context_gap]

    X_full = np.stack([f.reshape(-1) for f in feats], axis=-1)  # (N*L, n_feat)
    y_full = labels.reshape(-1)
    seq_idx_full = np.repeat(np.arange(N), L)
    row_valid = valid_mask.reshape(-1) if has_velocity else np.ones(N * L, dtype=bool)

    X, y, seq_idx = X_full[row_valid], y_full[row_valid], seq_idx_full[row_valid]
    print(f"[PT10] {row_valid.sum()}/{N*L} positions usable "
          f"({'with' if has_velocity else 'without'} velocity features)")

    # Merge rare classes into 'other'
    uniq_y, counts_y = np.unique(y, return_counts=True)
    frac_y = counts_y / len(y)
    rare = set(uniq_y[frac_y < args.min_class_frac])
    y_merged = np.array(["other" if c in rare else c for c in y])
    print(f"[PT10] class support (merged rare<{args.min_class_frac*100:.0f}% into 'other'):")
    for c, n in zip(*np.unique(y_merged, return_counts=True)):
        print(f"    {c:>26}: {n} ({n/len(y_merged)*100:.2f}%)")

    # Sequence-grouped train/val split
    rng = np.random.default_rng(args.seed)
    all_seqs = np.arange(N)
    rng.shuffle(all_seqs)
    n_val_seqs = max(1, int(N * args.val_frac))
    val_seqs = set(all_seqs[:n_val_seqs].tolist())
    is_val = np.array([s in val_seqs for s in seq_idx])

    X_train, y_train = X[~is_val], y_merged[~is_val]
    X_val, y_val = X[is_val], y_merged[is_val]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, log_loss

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=2000, C=1.0)  # multinomial by default for >2 classes (sklearn>=1.5)
    clf.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_s))
    val_pred = clf.predict(X_val_s)
    val_acc = accuracy_score(y_val, val_pred)
    val_probs = clf.predict_proba(X_val_s)
    val_ll = log_loss(y_val, val_probs, labels=clf.classes_)

    # Baseline: always predict the majority class (sanity check that the
    # model is doing better than a trivial constant predictor).
    majority = max(set(y_train.tolist()), key=list(y_train).count)
    majority_val_acc = float((y_val == majority).mean())

    # Per-position val correctness + which sequence each position belongs
    # to, for post-hoc sequence-level bootstrap CI (rigor-audit follow-up,
    # same "free win" pattern as PT1/PT2/PT3 -- no refit needed, just
    # resample which val sequences contribute to the accuracy average).
    val_seq_idx = seq_idx[is_val]
    val_correct = (val_pred == y_val)
    val_is_majority = (y_val == majority)
    npz_path = out_dir / f"failure_predictors_raw_{label}.npz"
    np.savez_compressed(
        npz_path, val_seq_idx=val_seq_idx, val_correct=val_correct, val_is_majority=val_is_majority,
    )
    print(f"[PT10] Saved per-position val raw arrays to {npz_path}")

    coef_table = {
        cls: {feat: float(coef) for feat, coef in zip(feature_names, clf.coef_[i])}
        for i, cls in enumerate(clf.classes_)
    } if len(clf.classes_) > 2 else {
        clf.classes_[1]: {feat: float(coef) for feat, coef in zip(feature_names, clf.coef_[0])}
    }

    summary = {
        "labels_npz": str(args.labels_npz), "label": label,
        "n_positions_total": int(N * L), "n_positions_usable": int(row_valid.sum()),
        "has_velocity_features": has_velocity,
        "feature_names": feature_names,
        "class_support": {c: int(n) for c, n in zip(*np.unique(y_merged, return_counts=True))},
        "n_train_seqs": int(N - n_val_seqs), "n_val_seqs": int(n_val_seqs),
        "train_accuracy": float(train_acc), "val_accuracy": float(val_acc),
        "val_log_loss": float(val_ll), "majority_class_val_accuracy": majority_val_acc,
        "coefficients_by_class": coef_table,
        "notes": [
            "Token frequency is a within-sample proxy, not true training-corpus frequency (EXP-20 caveat).",
            "function/content word uses a fixed closed-class list on the decoded token string, "
            "not a real POS tagger.",
            "Missing predictors (need infra not built yet): contextual surprisal (external LM), "
            "local context strength (EXP-PT4), oracle-rollout distance (EXP-PT7), "
            "self-conditioning norm (not meaningful in this zero-SC oracle-only protocol), "
            "Jacobian/local gain (EXP-PT6).",
            "Sequence-grouped (not position-grouped) train/val split, following the "
            "EXP-07->EXP-07v2 document-level-split lesson.",
        ] + ([
            "local_context_gap is a PER-SEQUENCE feature merged in from EXP-PT4 "
            "(full_context minus local_window_r1 accuracy, averaged over PT4's t-grid, "
            "broadcast to every position in that sequence) -- NOT a genuine per-position "
            "predictor like the others here. For LangFlow, PT4's context-ablation data "
            "used seq_len=1024 while this analysis uses seq_len=128; same underlying "
            "documents (verified deterministic sample ordering) but a longer effective "
            "context span than LangFlow's own L, see script comments.",
        ] if has_context else []),
    }
    json_path = out_dir / f"failure_predictors_{label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT10] train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
          f"majority_baseline_val_acc={majority_val_acc:.3f}  val_log_loss={val_ll:.3f}")
    print("[PT10] Top |coefficient| per class (standardized features):")
    for cls, coefs in coef_table.items():
        top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  {cls}: " + ", ".join(f"{k}={v:+.2f}" for k, v in top))
    print(f"[PT10] Saved {json_path}")


if __name__ == "__main__":
    main()
