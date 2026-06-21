"""
Offline MAUVE evaluation for Idea A experiment.
Uses GPT-2 featurization (standard for LM generation eval).
Reference texts sampled from OWT train set (T5-tokenized → decoded).

Usage:
    python eval_mauve_offline.py <output_dir1> [<output_dir2> ...]

Each dir is searched recursively for all_generated_*.jsonl files.
"""
import sys, json, glob, os
from pathlib import Path

import torch
import mauve
from transformers import AutoTokenizer
from datasets import load_dataset


# ── Reference text loading ───────────────────────────────────────────────────

def load_reference_texts(n=1000, seed=1234, max_chars=2048):
    """Sample n texts from OWT train set, decode via T5 tokenizer."""
    print(f"Loading {n} OWT reference texts (seed={seed})...")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    ds = load_dataset(
        "embedded-language-flows/openwebtext-t5",
        split="train",
        streaming=True,
    )
    # Skip seed * n examples to get a different slice than the generation seed
    ds = ds.skip(seed * n)

    texts = []
    for ex in ds:
        ids = ex["input_ids"]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        text = text[:max_chars]
        if len(text.split()) >= 20:  # skip very short texts
            texts.append(text)
        if len(texts) >= n:
            break

    print(f"  Loaded {len(texts)} reference texts.")
    return texts


# ── Generated text loading ───────────────────────────────────────────────────

def load_generated_texts(jsonl_path):
    texts = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["generated"])
    return texts


# ── MAUVE ────────────────────────────────────────────────────────────────────

def compute_mauve_score(ref_texts, gen_texts, device_id, max_text_length=256):
    out = mauve.compute_mauve(
        p_text=ref_texts,
        q_text=gen_texts,
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=False,
        featurize_model_name="gpt2",
        batch_size=32,
    )
    return out.mauve


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("Usage: python eval_mauve_offline.py <output_dir1> [...]")
        sys.exit(1)

    device_id = 0 if torch.cuda.is_available() else -1
    device_str = f"cuda:{device_id}" if device_id >= 0 else "cpu"
    print(f"MAUVE featurization on {device_str}")

    ref_texts = load_reference_texts(n=1000, seed=9999)

    results = []
    for d in dirs:
        jsonl_files = sorted(
            glob.glob(os.path.join(d, "**", "all_generated_*.jsonl"), recursive=True)
        )
        if not jsonl_files:
            print(f"No .jsonl files found in {d}")
            continue
        for jf in jsonl_files:
            condition = Path(jf).parent.name
            gen_texts = load_generated_texts(jf)
            parent_name = Path(jf).parent.parent.name
            print(f"\n[{condition}] {len(gen_texts)} samples from {parent_name}/")
            score = compute_mauve_score(ref_texts, gen_texts, device_id)
            print(f"  MAUVE = {score:.4f}")
            results.append((condition, len(gen_texts), score, parent_name))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for condition, n, score, label in results:
        print(f"  [{label}] {condition}  n={n}  MAUVE={score:.4f}")


if __name__ == "__main__":
    main()
