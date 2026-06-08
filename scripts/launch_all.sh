#!/bin/bash
PYTHON=~/miniforge3/envs/elf/bin/python
ELF_DIR=~/ELF

echo "=== 1. compute_token_centroids (GPU 4, JAX MEM=0.25) ==="
kill $(pgrep -f "compute_token_centroids") 2>/dev/null || true
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  nohup $PYTHON ~/compute_token_centroids.py \
    --config $ELF_DIR/src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint embedded-language-flows/ELF-B-owt \
    --n_texts 4096 --seq_len 256 --batch_size 8 --latent_std 0.2 \
    --out_dir ~/elf_centroids \
  > ~/centroid.log 2>&1 &
echo "  centroid PID=$!"

echo "=== 2. probe_mdlm (GPU 4, HF checkpoint) ==="
kill $(pgrep -f "probe_mdlm") 2>/dev/null || true
CUDA_VISIBLE_DEVICES=4 \
  nohup $PYTHON ~/probe_mdlm.py \
    --checkpoint kuleshov-group/mdlm-owt \
    --n_samples 64 --seq_len 128 --n_t_steps 21 --n_noise 4 \
    --out_dir ~/probe_mdlm_v1 \
  > ~/probe_mdlm.log 2>&1 &
echo "  mdlm PID=$!"

echo "=== 3. probe_duo (GPU 4, HF checkpoint) ==="
kill $(pgrep -f "probe_duo") 2>/dev/null || true
CUDA_VISIBLE_DEVICES=4 \
  nohup $PYTHON ~/probe_duo.py \
    --checkpoint s-sahoo/duo \
    --n_samples 64 --seq_len 128 --n_t_steps 21 --n_noise 4 \
    --out_dir ~/probe_duo_v1 \
  > ~/probe_duo.log 2>&1 &
echo "  duo PID=$!"

echo "=== All launched. Waiting 30s for startup errors... ==="
sleep 30
echo "--- centroid log ---"; tail -8 ~/centroid.log
echo "--- mdlm log ---";     tail -8 ~/probe_mdlm.log
echo "--- duo log ---";      tail -8 ~/probe_duo.log
