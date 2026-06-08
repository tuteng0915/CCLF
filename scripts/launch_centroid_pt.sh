#!/bin/bash
kill $(pgrep -f compute_token_centroids) 2>/dev/null
sleep 1
rm -f ~/centroid_pt.log
CUDA_VISIBLE_DEVICES=1 \
  nohup ~/miniforge3/envs/elf/bin/python ~/compute_token_centroids_pt.py \
    --n_texts 4096 --seq_len 256 --batch_size 16 --latent_std 0.2 \
    --out_dir ~/elf_centroids \
  > ~/centroid_pt.log 2>&1 &
echo "centroid_pt PID=$!"
