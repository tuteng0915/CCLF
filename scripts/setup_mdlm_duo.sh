#!/bin/bash
set -e
PYTHON=~/miniforge3/envs/elf/bin/python
PIP=~/miniforge3/envs/elf/bin/pip

echo "=== Cloning MDLM ==="
if [ ! -d ~/mdlm ]; then
    git clone https://github.com/kuleshov-group/mdlm ~/mdlm
else
    echo "mdlm already cloned"
fi

echo "=== Cloning DUO ==="
if [ ! -d ~/duo ]; then
    git clone https://github.com/s-sahoo/duo ~/duo
else
    echo "duo already cloned"
fi

echo "=== Installing MDLM ==="
cd ~/mdlm && $PIP install -e . --quiet 2>&1 | tail -5

echo "=== Installing DUO ==="
cd ~/duo && $PIP install -e . --quiet 2>&1 | tail -5

echo "=== Checking HuggingFace MDLM checkpoint ==="
$PYTHON -c "
from huggingface_hub import list_models
models = list(list_models(author='kuleshov-group', limit=20))
for m in models:
    print(m.id)
" 2>/dev/null || echo "(hf list failed)"

echo "=== Checking HuggingFace DUO checkpoint ==="
$PYTHON -c "
from huggingface_hub import list_models
models = list(list_models(author='s-sahoo', limit=20))
for m in models:
    print(m.id)
" 2>/dev/null || echo "(hf list failed)"

echo "=== Done ==="
