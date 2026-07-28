"""EXP-PT8 (part 1): Build controlled minimal-pair evidence-source data.

Uses BLiMP (nyu-mll/blimp on HF hub) as the natural minimal-pairs source,
instead of hand-curating sentences -- BLiMP already covers most of the
categories the suite doc suggests (subject-verb agreement, negation/NPI
licensing, number/determiner-noun agreement, argument structure as a
semantic-role proxy) with pairs that differ in exactly one grammatical cue.
"Named entity location" and "local collocation" (two of the doc's suggested
categories) aren't naturally covered by BLiMP and are NOT included.

For each BLiMP pair (sentence_good, sentence_bad), tokenizes both with the
target model's tokenizer and keeps ONLY pairs where:
  - both sentences tokenize to the SAME length (keeps position alignment trivial)
  - they differ at EXACTLY ONE token position (the "critical position" --
    this is what doc calls "the target token remains at the same position,
    while one controlled cue changes")
Pairs that don't satisfy this (different subword segmentation length, or the
edit spans multiple tokens) are dropped -- this is a real, fairly aggressive
filter; the retained count is reported and should be checked before treating
results as representative of each BLiMP category as a whole.

Usage:
    conda run -n elf python experiments/phase_transition/build_minimal_pairs.py \\
        --tokenizer t5-small --out results/phase_transition/minimal_pairs_t5.json \\
        --n_per_uid 60
"""

import argparse
import json


UIDS = [
    "irregular_plural_subject_verb_agreement_1",   # subject-verb agreement (number)
    "distractor_agreement_relational_noun",         # subject-verb agreement (harder)
    "determiner_noun_agreement_1",                  # number agreement
    "npi_present_1",                                # negation / NPI licensing
    "existential_there_subject_raising",            # argument structure / semantic role proxy
    "wh_vs_that_no_gap",                            # filler-gap / semantic role proxy
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="e.g. t5-small or gpt2")
    p.add_argument("--out", required=True)
    p.add_argument("--n_per_uid", type=int, default=60)
    p.add_argument("--seq_len", type=int, default=32, help="pad/truncate length for both sentences")
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pairs = []
    per_uid_counts = {}
    for uid in UIDS:
        try:
            ds = load_dataset("nyu-mll/blimp", uid, split="train")
        except Exception as e:
            print(f"[build_minimal_pairs] skipping {uid}: {e}")
            continue
        kept = 0
        for ex in ds:
            if kept >= args.n_per_uid:
                break
            good_ids = tok(ex["sentence_good"], truncation=True, max_length=args.seq_len)["input_ids"]
            bad_ids = tok(ex["sentence_bad"], truncation=True, max_length=args.seq_len)["input_ids"]
            if len(good_ids) != len(bad_ids):
                continue
            diffs = [i for i, (g, b) in enumerate(zip(good_ids, bad_ids)) if g != b]
            if len(diffs) != 1:
                continue
            crit = diffs[0]
            pad_id = tok.pad_token_id
            good_padded = good_ids + [pad_id] * (args.seq_len - len(good_ids))
            bad_padded = bad_ids + [pad_id] * (args.seq_len - len(bad_ids))
            pairs.append({
                "uid": uid, "critical_position": crit, "seq_len_real": len(good_ids),
                "good_ids": good_padded, "bad_ids": bad_padded,
                "sentence_good": ex["sentence_good"], "sentence_bad": ex["sentence_bad"],
            })
            kept += 1
        per_uid_counts[uid] = kept
        print(f"[build_minimal_pairs] {uid}: kept {kept}/{min(len(ds), args.n_per_uid)} candidates")

    print(f"[build_minimal_pairs] total pairs kept: {len(pairs)}")
    with open(args.out, "w") as f:
        json.dump({
            "tokenizer": args.tokenizer, "seq_len": args.seq_len,
            "per_uid_counts": per_uid_counts, "pairs": pairs,
        }, f)
    print(f"[build_minimal_pairs] Saved {args.out}")


if __name__ == "__main__":
    main()
