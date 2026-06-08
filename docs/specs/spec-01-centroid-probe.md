# Spec 01 — Centroid Anchor Distance Probe

**Type**: code modification + experiment run  
**Priority**: high (blocks Conjecture 5 verification)  
**Output**: `results/elf/probe_v3_centroid/`

---

## Background

`probe_anchor_v3.py` computes three geometric anchoring metrics (D_soft, D_NN, margin)
to test whether ELF's denoising output x̂_t geometrically approaches the token manifold.

In the current implementation, the anchor matrix E is loaded from the **T5 input embedding
table** (line 656–659 in `experiments/probe_elf/probe_anchor_v3.py`):

```python
E_raw = np.array(encoder_params["shared"]["embedding"])  # [V, 512] input embeddings
E = E_raw[:vocab_size] / latent_std
```

This is wrong: x̂_t lives in **T5 contextual embedding space** (output of 6 Transformer
layers), not the input embedding space. As a result, probe v3 showed flat d_nn≈1216 and
flat margin≈71 across all t — the distance to the "nearest token" is meaningless in the
wrong space.

The fix: use pre-computed contextual centroids as E:
```
E_ctx[v] = mean_{positions where token==v in OWT} T5-encoder-output[position]
```
These are in `results/data/token_centroids.npz`, key `centroids`, shape `[32100, 512]`.

---

## Task

### 1. Modify `experiments/probe_elf/probe_anchor_v3.py`

Add a `--centroid_path` CLI argument. When provided, load E from the npz file instead
of the model's raw embedding table.

**Change location**: the block at lines 655–659:

```python
# BEFORE (load raw input embeddings):
E_raw      = np.array(encoder_params["shared"]["embedding"])
E = E_raw[:vocab_size] / latent_std
print(f"[embed] E: {E.shape}  latent_std={latent_std}  "
      f"(WARNING: T5 input embeddings, not contextual centroids)")
```

Replace with:

```python
# AFTER (support contextual centroid as drop-in replacement):
if args.centroid_path:
    data = np.load(args.centroid_path)
    E_raw = data["centroids"].astype(np.float32)   # [V, 512], T5 contextual space
    print(f"[embed] E: {E_raw.shape}  source=contextual_centroid ({args.centroid_path})")
else:
    E_raw = np.array(encoder_params["shared"]["embedding"])
    print(f"[embed] E: {E_raw.shape}  source=raw_input_embedding  "
          f"(WARNING: space mismatch — use --centroid_path for correct geometry)")
E = E_raw[:vocab_size] / latent_std
```

Add to `argparse` (near the end of the file, in the `parse_args` function):

```python
p.add_argument("--centroid_path", type=str, default=None,
               help="Path to token_centroids.npz (contextual centroids). "
                    "If omitted, falls back to raw T5 input embeddings (space mismatch).")
```

### 2. Also apply the same change to `experiments/probe_elf/probe_anchor_v4.py`

The pattern is identical — find the E loading block (around line 874–877) and apply the
same `--centroid_path` logic.

### 3. Run on server

```bash
# on new-ncl, from ~/tt_workspace/model/CCLF/
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
python experiments/probe_elf/probe_anchor_v3.py \
    --checkpoint embedded-language-flows/ELF-B-owt \
    --n_samples 64 --seq_len 256 --n_t_steps 21 \
    --tau_list 1.0 \
    --centroid_path results/data/token_centroids.npz \
    --out_dir results/elf/probe_v3_centroid
```

### 4. Expected output

`results/elf/probe_v3_centroid/`
- `anchor_probe_v3.json` — same schema as `results/elf/probe_v3/anchor_probe_v3.json`
- `anchor_probe_commitment.png`
- `anchor_probe_tau_sweep.png`

---

## Success criteria

The run is successful if the JSON shows **non-flat** d_nn and margin curves, i.e.:
- `d_nn` is **monotonically decreasing** from t=0 to t=1 (geometric approach to manifold)
- `margin` is **monotonically increasing** (growing token separation)

Compare against the flat baselines from probe v3:
- Old: `d_nn ≈ 1216` flat, `margin ≈ 71` flat → space mismatch confirmed
- Expected new: `d_nn` decreases meaningfully (expected Δd_nn similar to LangFlow: −2.77 in normalized units)

---

## Notes

- `token_centroids.npz` was computed with `compute_token_centroids_pt.py` using PyTorch
  T5-small encoder on 4096 OWT texts (22250 / 32100 tokens seen). Tokens with zero count
  have centroid = zero vector — these will show artificially large d_nn.
- The centroid normalization (`/ latent_std`) is correct: ELF trains with
  `x_norm = x_T5 / 0.2`, so both x̂_t and E must be in the same normalized space.
- `latent_std` is read from the checkpoint config (0.2 for ELF-B-owt).
