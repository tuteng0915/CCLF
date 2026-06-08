"""
compute_token_centroids_pt.py — PyTorch version of compute_token_centroids.py

Computes per-token contextual centroids using PyTorch T5EncoderModel,
avoiding the JAX/cuDNN compatibility issue.

ELF uses vanilla T5-small as its encoder (weights from
embedded-language-flows/t5_small_encoder_jax — same as t5-small).

For each token id v in the vocabulary:
    E_ctx[v] = mean over all positions in OWT where token_id == v of
               (T5_encoder(sequence)[last_hidden_state][position] / latent_std)

Output .npz is compatible with the JAX version:
    centroids  [V, d]  float32
    counts     [V]     int64
    vocab_size, hidden_dim, n_texts, latent_std

Usage:
    CUDA_VISIBLE_DEVICES=1 python compute_token_centroids_pt.py \\
        --n_texts 4096 --seq_len 256 --batch_size 16 --latent_std 0.2 \\
        --out_dir ~/elf_centroids
"""

import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import torch
from transformers import T5EncoderModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_owt_texts(n: int) -> list:
    from datasets import load_dataset
    def _stream(name, **kw):
        ds = load_dataset(name, split="train", streaming=True, **kw)
        texts = []
        for ex in ds:
            t = ex["text"].strip()
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= n:
                break
        return texts[:n]
    for name, kw in [("Skylion007/openwebtext", {}),
                     ("stas/openwebtext-10k",   {}),
                     ("wikitext", {"name": "wikitext-103-raw-v1"})]:
        try:
            texts = _stream(name, **kw)
            if texts:
                print(f"[data] loaded from {name}")
                return texts
        except Exception as e:
            print(f"[data] {name} failed: {e}")
    raise RuntimeError("Could not load any text dataset.")


def compute_centroids_pt(
    model,
    tokenizer,
    texts: list,
    seq_len: int,
    batch_size: int,
    latent_std: float,
    vocab_size: int,
    hidden_dim: int,
    progress_every: int = 256,
) -> tuple:
    sum_arr   = np.zeros((vocab_size, hidden_dim), dtype=np.float64)
    count_arr = np.zeros(vocab_size, dtype=np.int64)
    n_texts   = len(texts)

    for batch_start in range(0, n_texts, batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        ids  = enc["input_ids"].to(device)       # [B, L]
        mask = enc["attention_mask"].to(device)  # [B, L]

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        # last_hidden_state: [B, L, d]
        x = out.last_hidden_state.float().cpu().numpy()   # [B, L, d]
        x_norm = x / latent_std

        ids_np  = ids.cpu().numpy()
        mask_np = mask.cpu().numpy()
        B = ids_np.shape[0]

        for b in range(B):
            for l in range(seq_len):
                if mask_np[b, l] == 1:
                    vid = int(ids_np[b, l])
                    if 0 <= vid < vocab_size:
                        sum_arr[vid]   += x_norm[b, l].astype(np.float64)
                        count_arr[vid] += 1

        processed = batch_start + B
        if processed % progress_every == 0 or processed == n_texts:
            coverage = int((count_arr > 0).sum())
            print(f"  {processed}/{n_texts} texts  "
                  f"coverage {coverage}/{vocab_size} tokens")

    return sum_arr, count_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="t5-small")
    p.add_argument("--n_texts",     type=int,   default=4096)
    p.add_argument("--seq_len",     type=int,   default=256)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--latent_std",  type=float, default=0.2)
    p.add_argument("--out_dir",     default="~/elf_centroids")
    args = p.parse_args()

    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[model] loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = T5EncoderModel.from_pretrained(args.model).to(device).eval()

    vocab_size = tokenizer.vocab_size
    hidden_dim = model.config.d_model
    print(f"[model] vocab={vocab_size}  d_model={hidden_dim}  latent_std={args.latent_std}")

    print(f"[data] loading {args.n_texts} OWT texts…")
    texts = load_owt_texts(args.n_texts)
    print(f"[data] {len(texts)} texts ready")

    print(f"\n[centroid] accumulating (batch_size={args.batch_size}, seq_len={args.seq_len})…")
    sum_arr, count_arr = compute_centroids_pt(
        model, tokenizer, texts,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        latent_std=args.latent_std,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    )

    n_zero = int((count_arr == 0).sum())
    centroids = (sum_arr / np.maximum(count_arr[:, None], 1)).astype(np.float32)
    if n_zero:
        print(f"[centroid] {n_zero} tokens unseen — centroid set to zero vector")

    # Stats
    seen = count_arr > 0
    norms = np.linalg.norm(centroids[seen], axis=-1)
    print(f"[stats] seen={seen.sum()}  L2 norms: "
          f"p10={np.percentile(norms,10):.3f}  p50={np.percentile(norms,50):.3f}  "
          f"p90={np.percentile(norms,90):.3f}")

    out_path = out_dir / "token_centroids.npz"
    np.savez(
        out_path,
        centroids  = centroids,
        counts     = count_arr,
        vocab_size = np.array(vocab_size,   dtype=np.int64),
        hidden_dim = np.array(hidden_dim,   dtype=np.int64),
        n_texts    = np.array(len(texts),   dtype=np.int64),
        latent_std = np.array(args.latent_std, dtype=np.float32),
    )
    print(f"\n[saved] {out_path}")
    print(f"        shape={centroids.shape}  dtype={centroids.dtype}")
    print("[done]")


if __name__ == "__main__":
    main()
