#!/bin/bash
kill $(pgrep -f "probe_anchor_v4") 2>/dev/null || true
sleep 1
cd ~/ELF
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  nohup ~/miniforge3/envs/elf/bin/python probe_anchor_v4.py \
    --config src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint embedded-language-flows/ELF-B-owt \
    --n_samples 64 --seq_len 256 --n_t_steps 21 \
    --n_noise 4 --tau_list "0.5,1.0,2.0" \
    --out_dir ~/probe_results_v4 > ~/probe_v4.log 2>&1 &
echo "PID=$!"
