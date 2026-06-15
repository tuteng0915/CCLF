"""
Offline Gen.PPL evaluation for Idea A experiment.
Uses PyTorch GPT-2 Large to score texts saved in .jsonl files.

Usage:
    python eval_ppl_offline.py <dir1> [<dir2> ...]

Each dir should contain all_generated_*.jsonl files.
"""
import sys, json, math, glob, os
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm


def load_texts(jsonl_path):
    texts = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["generated"])
    return texts


def compute_ppl(texts, model, tokenizer, device, max_length=1024, batch_size=16):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="  PPL batches", leave=False):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)
        with torch.no_grad():
            # labels=input_ids → HF computes CE loss internally, averaged over tokens
            out = model(input_ids, attention_mask=attn_mask, labels=input_ids)
        # out.loss is mean NLL over non-padding tokens
        n_tokens = attn_mask.sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("nan")


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("Usage: python eval_ppl_offline.py <output_dir1> [<output_dir2> ...]")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 Large on {device}...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)

    results = []
    for d in dirs:
        jsonl_files = sorted(glob.glob(os.path.join(d, "**", "all_generated_*.jsonl"), recursive=True))
        if not jsonl_files:
            print(f"No .jsonl files found in {d}")
            continue
        for jf in jsonl_files:
            condition = Path(jf).parent.name
            texts = load_texts(jf)
            print(f"\n[{condition}] {len(texts)} samples from {Path(jf).parent.parent.name}/")
            ppl = compute_ppl(texts, model, tokenizer, device)
            print(f"  Gen.PPL = {ppl:.2f}")
            results.append((condition, len(texts), ppl, jf))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for condition, n, ppl, jf in results:
        label = Path(jf).parent.parent.name
        print(f"  [{label}] {condition}  n={n}  Gen.PPL={ppl:.2f}")


if __name__ == "__main__":
    main()
