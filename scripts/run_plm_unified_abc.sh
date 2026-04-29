#!/bin/bash
# Unified A/B/C training for PLM-1B + SplitDualHeadExpert + TemporalQueryRouter.
#
# Phase A: Expert training with aux losses (video-only)
# Phase B: Router training with QA loss (expert frozen)
# Phase C: LoRA training with QA loss (expert + router frozen)
#
# Usage:
#   bash scripts/run_plm_unified_abc.sh                    # single GPU
#   torchrun --nproc_per_node=4 scripts/train_plm_unified_abc.py ...  # multi-GPU

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Configurable ──
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plm_unified_abc}"
NUM_FRAMES="${NUM_FRAMES:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
VIDEO_TOKEN_ID="${VIDEO_TOKEN_ID:-128003}"
BATCH_SIZE="${BATCH_SIZE:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
DATASET_PRESET="${DATASET_PRESET:-full+eval}"
VIDEO_ONLY_DATA="${VIDEO_ONLY_DATA:-datasets/video_only_stage1.jsonl}"

# Phase steps
PHASE_A="${PHASE_A:-800}"
PHASE_B="${PHASE_B:-800}"
PHASE_C="${PHASE_C:-600}"

# Learning rates
EXPERT_LR="${EXPERT_LR:-1e-3}"
ROUTER_LR="${ROUTER_LR:-1e-3}"
LORA_LR="${LORA_LR:-5e-5}"
LORA_RANK="${LORA_RANK:-16}"

# Expert / Router
EXPERT_BOTTLENECK="${EXPERT_BOTTLENECK:-128}"
ROUTER_HIDDEN="${ROUTER_HIDDEN:-256}"

# GPU count
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPUS="${NGPUS:-1}"

echo "=== PLM-1B Unified A/B/C Training ==="
echo "  output_dir:    ${OUTPUT_DIR}"
echo "  num_frames:    ${NUM_FRAMES}"
echo "  image_size:    ${IMAGE_SIZE}"
echo "  video_token_id:${VIDEO_TOKEN_ID}"
echo "  phases:        A=${PHASE_A} B=${PHASE_B} C=${PHASE_C}"
echo "  lrs:           expert=${EXPERT_LR} router=${ROUTER_LR} lora=${LORA_LR}"
echo "  batch_size:    ${BATCH_SIZE} x grad_accum=${GRAD_ACCUM} x gpus=${NGPUS}"
echo "  dataset:       ${DATASET_PRESET}"
echo ""

COMMON_ARGS=(
    --output_dir "${OUTPUT_DIR}"
    --num_frames "${NUM_FRAMES}"
    --image_size "${IMAGE_SIZE}"
    --video_token_id "${VIDEO_TOKEN_ID}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum "${GRAD_ACCUM}"
    --dataset_preset "${DATASET_PRESET}"
    --video_only_data "${VIDEO_ONLY_DATA}"
    --phase_a_steps "${PHASE_A}"
    --phase_b_steps "${PHASE_B}"
    --phase_c_steps "${PHASE_C}"
    --expert_lr "${EXPERT_LR}"
    --router_lr "${ROUTER_LR}"
    --lora_lr "${LORA_LR}"
    --lora_rank "${LORA_RANK}"
    --expert_bottleneck "${EXPERT_BOTTLENECK}"
    --router_hidden "${ROUTER_HIDDEN}"
    --save_interval 200
    --log_interval 10
    --warmup_steps 50
    --num_workers 4
)

if [ "${NGPUS}" -gt 1 ]; then
    torchrun \
        --nproc_per_node="${NGPUS}" \
        --master_port="${MASTER_PORT:-29500}" \
        scripts/train_plm_unified_abc.py \
        "${COMMON_ARGS[@]}"
else
    python scripts/train_plm_unified_abc.py \
        "${COMMON_ARGS[@]}"
fi

echo "=== Done: ${OUTPUT_DIR} ==="
