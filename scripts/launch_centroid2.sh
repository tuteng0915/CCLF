#!/bin/bash
kill $(pgrep -f "compute_token_centroids") 2>/dev/null || true
sleep 1
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  nohup ~/miniforge3/envs/elf/bin/python ~/compute_token_centroids.py \
    --config ~/ELF/src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint embedded-language-flows/ELF-B-owt \
    --n_texts 4096 --seq_len 256 --batch_size 8 --latent_std 0.2 \
    --out_dir ~/elf_centroids \
  > ~/centroid.log 2>&1 &
echo "centroid PID=$!"

kill $(pgrep -f "probe_mdlm") 2>/dev/null || true
sleep 1
CUDA_VISIBLE_DEVICES=4 \
  nohup ~/miniforge3/envs/elf/bin/python ~/probe_mdlm.py \
    --checkpoint kuleshov-group/mdlm-owt \
    --n_samples 64 --seq_len 128 --n_t_steps 21 --n_noise 4 \
    --out_dir ~/probe_mdlm_v1 \
  > ~/probe_mdlm.log 2>&1 &
echo "mdlm PID=$!"
