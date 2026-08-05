#!/usr/bin/env bash
# run_gpu_experiments_v2.sh — Launch EXP-30v2 / EXP-31v2 multi-seed / EXP-36 factorial
#
# Usage (from models/ELF-torch):
#   GPU=0 bash scripts/run_gpu_experiments_v2.sh [exp30v2|exp31v2_kd_cr|exp31v2_kd2|exp36|all]
#
# Each experiment is sequential by default. Set PARALLEL=1 to spawn in background
# (requires enough GPU memory).

set -euo pipefail

GPU=${GPU:-0}
PARALLEL=${PARALLEL:-0}
CKPT_DIR="converted"
CCLF_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)/CCLF/CCLF"

BASELINE_CKPT="${CKPT_DIR}/elf_b-owt-baseline_torch.pt"
KD_CR_CKPT="${CKPT_DIR}/elf_b-owt-kd-cr_torch.pt"
KD2_CKPT="${CKPT_DIR}/elf_b-owt-kd2_torch.pt"

TARGET=${1:-all}

run_exp30v2() {
    echo "===== EXP-30v2: LangFlow layer-wise probe v2 ====="
    cd "${CCLF_ROOT}"
    CUDA_VISIBLE_DEVICES=${GPU} conda run -n elf python \
        experiments/probe_langflow/probe_layerwise_langflow_v2.py \
        --checkpoint Continuous-Rivals-Discrete/langflow-owt \
        --n_samples 64 --seq_len 128 --n_noise 4 \
        --n_probe_seeds 5 \
        --out_dir results/exp30v2_langflow \
        2>&1 | tee /tmp/exp30v2.log
    echo "[EXP-30v2] Done. Log: /tmp/exp30v2.log"
}

run_exp31v2_kd_cr() {
    echo "===== EXP-31v2 multi-seed: kd_cr (seeds 0-4) ====="
    for seed in 0 1 2 3 4; do
        echo "--- kd_cr seed=${seed} ---"
        CUDA_VISIBLE_DEVICES=${GPU} conda run -n elf \
            bash scripts/launch.sh eval \
            src/configs/training_configs/eval_spec31v2_kd_cr_seed${seed}.yml \
            --checkpoint_path ${KD_CR_CKPT} \
            2>&1 | tee /tmp/exp31v2_kd_cr_seed${seed}.log
        echo "  [seed=${seed}] Done."
    done
    echo "[EXP-31v2 kd_cr] All 5 seeds done."
}

run_exp31v2_kd2() {
    echo "===== EXP-31v2 multi-seed: kd2 (seeds 0-4) ====="
    for seed in 0 1 2 3 4; do
        echo "--- kd2 seed=${seed} ---"
        CUDA_VISIBLE_DEVICES=${GPU} conda run -n elf \
            bash scripts/launch.sh eval \
            src/configs/training_configs/eval_spec31v2_kd2_seed${seed}.yml \
            --checkpoint_path ${KD2_CKPT} \
            2>&1 | tee /tmp/exp31v2_kd2_seed${seed}.log
        echo "  [seed=${seed}] Done."
    done
    echo "[EXP-31v2 kd2] All 5 seeds done."
}

run_exp36_factorial() {
    echo "===== EXP-36 FULL FACTORIAL ====="
    for ckpt_name in baseline kd_cr kd2; do
        case "${ckpt_name}" in
            baseline) CKPT="${BASELINE_CKPT}" ;;
            kd_cr)    CKPT="${KD_CR_CKPT}" ;;
            kd2)      CKPT="${KD2_CKPT}" ;;
        esac
        echo "--- ${ckpt_name} ---"
        CUDA_VISIBLE_DEVICES=${GPU} conda run -n elf \
            bash scripts/launch.sh eval \
            src/configs/training_configs/eval_spec36_factorial_${ckpt_name}.yml \
            --checkpoint_path ${CKPT} \
            2>&1 | tee /tmp/exp36_factorial_${ckpt_name}.log
        echo "  [${ckpt_name}] Done."
    done
    echo "[EXP-36 factorial] All 3 checkpoints done."
}

case "${TARGET}" in
    exp30v2)          run_exp30v2 ;;
    exp31v2_kd_cr)    run_exp31v2_kd_cr ;;
    exp31v2_kd2)      run_exp31v2_kd2 ;;
    exp36)            run_exp36_factorial ;;
    all)
        run_exp30v2
        run_exp31v2_kd_cr
        run_exp31v2_kd2
        run_exp36_factorial
        ;;
    *)
        echo "Unknown target: ${TARGET}"
        echo "Valid: exp30v2 | exp31v2_kd_cr | exp31v2_kd2 | exp36 | all"
        exit 1
        ;;
esac

echo "[DONE] All requested experiments completed."
