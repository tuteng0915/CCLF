#!/bin/bash
# Upgrade nvidia-cudnn-cu12 to match jaxlib 0.10.1's requirement (cuDNN 9.8.x)
~/miniforge3/envs/elf/bin/pip install "nvidia-cudnn-cu12>=9.8.0" --quiet
echo "Install exit: $?"
# Verify
~/miniforge3/envs/elf/bin/pip show nvidia-cudnn-cu12 | grep Version
# Quick JAX smoke test
CUDA_VISIBLE_DEVICES=3 ~/miniforge3/envs/elf/bin/python -c "import jax; import jax.numpy as jnp; k = jax.random.PRNGKey(42); x = jax.random.normal(k, (4,4)); print('JAX OK, device:', x.device()); print(x.sum())"
