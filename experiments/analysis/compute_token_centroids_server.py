"""
Compute per-token contextual centroids from the ELF T5 encoder.

For each token id v in the vocabulary, compute:

    E_ctx[v] = mean over all positions in OWT where token_id == v of
               (T5_encoder(sequence)[position] / latent_std)

This places centroids in the same normalized space as x_hat_t, enabling
proper geometric anchoring tests for Conjecture 5 (rather than using the
raw T5 input-embedding matrix, which lives in a different space).

Usage:
    cd ~/ELF
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \\
    python experiments/probe_anchor/compute_token_centroids.py \\
        --config src/configs/training_configs/train_owt_ELF-B.yml \\
        --checkpoint embedded-language-flows/ELF-B-owt \\
        --n_texts 4096 --seq_len 256 --batch_size 8 --latent_std 0.2 \\
        --out_dir ~/elf_centroids
"""

import sys, os, argparse, copy
from pathlib import Path

import numpy as np

ELF_SRC = os.path.expanduser("~/ELF/src")
if ELF_SRC not in sys.path:
    sys.path.insert(0, ELF_SRC)

import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass
import jax.numpy as jnp
import optax


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading  (same streaming pattern as probe_anchor_v4.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_owt_texts(n):
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. ELF encoder loading  (pattern from probe_anchor_v4.py: load_elf)
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(config_path, checkpoint_path, latent_std_override=None):
    """Load T5 encoder, tokenizer, and config from an ELF checkpoint."""
    from modules.t5_encoder import get_encoder
    from utils.checkpoint_utils import load_encoder_checkpoint
    from configs.config import load_config_from_yaml
    from transformers import AutoTokenizer

    _orig_cwd = os.getcwd()
    os.chdir(ELF_SRC)
    abs_cfg = (os.path.join(_orig_cwd, config_path)
               if not os.path.isabs(config_path) else config_path)
    config = load_config_from_yaml(abs_cfg)
    os.chdir(_orig_cwd)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name)
    enc_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)

    latent_std = (latent_std_override
                  if latent_std_override is not None
                  else getattr(config, "latent_std", 0.2))

    print(f"[encoder] model={config.encoder_model_name}  "
          f"d={enc_config.d_model}  vocab={tokenizer.vocab_size}  "
          f"latent_std={latent_std}")
    return encoder_model, encoder_params, tokenizer, enc_config, latent_std


# ─────────────────────────────────────────────────────────────────────────────
# 3. Centroid accumulation
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroids(
    encoder_model,
    encoder_params,
    tokenizer,
    hidden_dim: int,
    vocab_size: int,
    texts: list,
    seq_len: int,
    batch_size: int,
    latent_std: float,
    progress_every: int = 256,
) -> tuple:
    """
    Returns (sum_arr, count_arr) both [V, d] / [V] respectively.

    Accumulates in float64 to avoid precision loss when summing many vectors.
    """
    sum_arr   = np.zeros((vocab_size, hidden_dim), dtype=np.float64)
    count_arr = np.zeros(vocab_size,               dtype=np.int64)
    n_texts   = len(texts)

    for batch_start in range(0, n_texts, batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        B = len(batch_texts)

        # Tokenize batch
        enc = tokenizer(
            batch_texts,
            return_tensors="np",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        ids  = enc["input_ids"]          # [B, L]
        mask = enc["attention_mask"]     # [B, L]

        # Run T5 encoder
        out  = encoder_model.apply(
            {"params": encoder_params},
            input_ids=ids,
            attention_mask=mask,
            deterministic=True,
        )
        # x_norm: [B, L, d]  — same space as x_hat_t
        x_norm = np.array(out[0], dtype=np.float32) / latent_std

        # Accumulate per-token
        for b in range(B):
            for l in range(seq_len):
                if mask[b, l] == 1:
                    vid = int(ids[b, l])
                    if 0 <= vid < vocab_size:
                        sum_arr[vid]   += x_norm[b, l].astype(np.float64)
                        count_arr[vid] += 1

        processed = batch_start + B
        if processed % progress_every == 0 or processed == n_texts:
            coverage = int((count_arr > 0).sum())
            print(f"  Processed {processed}/{n_texts} texts, "
                  f"coverage: {coverage}/{vocab_size} tokens seen")

    return sum_arr, count_arr


# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_centroid_stats(centroids, count_arr, tokenizer):
    """Print top-5 and bottom-5 tokens by count, plus L2 norm stats."""
    seen_mask = count_arr > 0
    n_zero    = int((~seen_mask).sum())
    n_seen    = int(seen_mask.sum())
    print(f"\n[stats] tokens seen: {n_seen}  zero-count (filled with 0): {n_zero}")

    # L2 norms of non-zero centroids
    norms = np.linalg.norm(centroids[seen_mask], axis=-1)
    print(f"[stats] centroid L2 norms  "
          f"min={norms.min():.4f}  p10={np.percentile(norms,10):.4f}  "
          f"p50={np.percentile(norms,50):.4f}  p90={np.percentile(norms,90):.4f}  "
          f"max={norms.max():.4f}")

    seen_ids    = np.where(seen_mask)[0]
    seen_counts = count_arr[seen_ids]
    order       = np.argsort(seen_counts)[::-1]

    def tok_str(vid):
        try:
            return repr(tokenizer.convert_ids_to_tokens(int(vid)))
        except Exception:
            return f"id={vid}"

    print("\n[stats] Top-5 tokens by count:")
    for rank in range(min(5, len(order))):
        vid = seen_ids[order[rank]]
        print(f"  {rank+1:>2}.  {tok_str(vid):>20}  count={count_arr[vid]:>8d}  "
              f"||c||={np.linalg.norm(centroids[vid]):.4f}")

    print("[stats] Bottom-5 tokens by count (excluding zero-count):")
    for rank in range(1, min(6, len(order) + 1)):
        vid = seen_ids[order[-rank]]
        print(f"  {rank:>2}.  {tok_str(vid):>20}  count={count_arr[vid]:>8d}  "
              f"||c||={np.linalg.norm(centroids[vid]):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-token contextual centroids from the ELF T5 encoder.")
    p.add_argument("--config",      type=str,   required=True,
                   help="Path to ELF training config YAML.")
    p.add_argument("--checkpoint",  type=str,   default="embedded-language-flows/ELF-B-owt",
                   help="HuggingFace repo id or local path for the ELF checkpoint.")
    p.add_argument("--n_texts",     type=int,   default=4096,
                   help="Number of OWT texts to process.")
    p.add_argument("--seq_len",     type=int,   default=256,
                   help="Tokenized sequence length (padding/truncation target).")
    p.add_argument("--batch_size",  type=int,   default=8,
                   help="Texts per encoder forward pass.")
    p.add_argument("--latent_std",  type=float, default=None,
                   help="Override latent_std (default: read from config, fallback 0.2).")
    p.add_argument("--out_dir",     type=str,   default="~/elf_centroids",
                   help="Directory for token_centroids.npz output.")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load encoder ────────────────────────────────────────────────────────
    encoder_model, encoder_params, tokenizer, enc_config, latent_std = load_encoder(
        args.config, args.checkpoint, latent_std_override=args.latent_std)

    vocab_size = tokenizer.vocab_size
    hidden_dim = enc_config.d_model

    # ── Load texts ──────────────────────────────────────────────────────────
    print(f"[data] loading {args.n_texts} OWT texts…")
    texts = load_owt_texts(args.n_texts)
    print(f"[data] got {len(texts)} texts")

    # ── Accumulate centroids ─────────────────────────────────────────────────
    print(f"\n[centroid] accumulating over {len(texts)} texts  "
          f"(batch_size={args.batch_size}, seq_len={args.seq_len})…")
    sum_arr, count_arr = compute_centroids(
        encoder_model, encoder_params, tokenizer,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        texts=texts,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        latent_std=latent_std,
        progress_every=256,
    )

    # ── Compute centroids; zero-fill unseen tokens ───────────────────────────
    n_zero = int((count_arr == 0).sum())
    with np.errstate(invalid="ignore"):
        centroids = (sum_arr / np.maximum(count_arr[:, None], 1)).astype(np.float32)
    # Tokens never seen stay at zero (numerator was already 0)
    if n_zero:
        print(f"[centroid] {n_zero} tokens never seen in OWT — centroids set to zero vector")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = out_dir / "token_centroids.npz"
    np.savez(
        out_path,
        centroids  = centroids,
        counts     = count_arr,
        vocab_size = np.array(vocab_size,  dtype=np.int64),
        hidden_dim = np.array(hidden_dim,  dtype=np.int64),
        n_texts    = np.array(len(texts),  dtype=np.int64),
        latent_std = np.array(latent_std,  dtype=np.float32),
    )
    print(f"\n[save] → {out_path}")
    print(f"        centroids shape : {centroids.shape}  dtype={centroids.dtype}")
    print(f"        counts shape    : {count_arr.shape}  dtype={count_arr.dtype}")
    print(f"        vocab_size={vocab_size}  hidden_dim={hidden_dim}  "
          f"n_texts={len(texts)}  latent_std={latent_std}")

    # ── Summary stats ────────────────────────────────────────────────────────
    print_centroid_stats(centroids, count_arr, tokenizer)
    print("\n[done]")


if __name__ == "__main__":
    main()
