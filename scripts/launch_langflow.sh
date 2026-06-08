#!/bin/bash
kill $(pgrep -f "probe_langflow") 2>/dev/null || true
sleep 2
cd ~/LangFlow
CUDA_VISIBLE_DEVICES=1 nohup ~/miniforge3/envs/elf/bin/python probe_langflow.py \
    --checkpoint Continuous-Rivals-Discrete/langflow-owt \
    --n_samples 64 --seq_len 256 --n_t_steps 21 --n_noise 4 \
    --out_dir ~/probe_langflow_v1 > ~/probe_langflow_v1.log 2>&1 &
echo "PID=$!"
