#!/bin/bash
kill $(pgrep -f "compute_token_centroids") 2>/dev/null || true
sleep 1
cd ~/ELF
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  nohup ~/miniforge3/envs/elf/bin/python ~/compute_token_centroids.py \
    --config src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint embedded-language-flows/ELF-B-owt \
    --n_texts 4096 --seq_len 256 --batch_size 8 --latent_std 0.2 \
    --out_dir ~/elf_centroids \
  > ~/centroid.log 2>&1 &
echo "centroid PID=$!"
