#!/bin/bash
# Unified A/B/C training for VideoLLaMA3-2B + Expert + Router + LoRA
#
# Usage:
#   bash scripts/run_vl3_unified_abc.sh                    # single GPU
#   torchrun --nproc_per_node=4 scripts/train_videollama3_unified_abc.py ...  # multi-GPU
#
# Override any arg via env vars, e.g.:
#   PHASE_A=1000 PHASE_B=1000 PHASE_C=800 bash scripts/run_vl3_unified_abc.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Defaults ──
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vl3_unified_abc}"
BASE_MODEL="${BASE_MODEL:-DAMO-NLP-SG/VideoLLaMA3-2B}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

PHASE_A="${PHASE_A:-800}"
PHASE_B="${PHASE_B:-800}"
PHASE_C="${PHASE_C:-600}"

EXPERT_LR="${EXPERT_LR:-1e-3}"
ROUTER_LR="${ROUTER_LR:-1e-3}"
LORA_LR="${LORA_LR:-5e-5}"
LORA_RANK="${LORA_RANK:-16}"

EXPERT_BOTTLENECK="${EXPERT_BOTTLENECK:-128}"
EXPERT_KERNEL_SIZE="${EXPERT_KERNEL_SIZE:-3}"

NUM_FRAMES="${NUM_FRAMES:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
PATCH_SIZE="${PATCH_SIZE:-14}"
MERGE_SIZE="${MERGE_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

DATASET_PRESET="${DATASET_PRESET:-full+eval}"
VIDEO_ONLY_DATA="${VIDEO_ONLY_DATA:-datasets/video_only_stage1.jsonl}"
NUM_WORKERS="${NUM_WORKERS:-4}"

SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

NGPU="${NGPU:-1}"

# ── Run ──
CMD="python scripts/train_videollama3_unified_abc.py"
if [ "${NGPU}" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=${NGPU} scripts/train_videollama3_unified_abc.py"
fi

exec ${CMD} \
    --output_dir "${OUTPUT_DIR}" \
    --base_model_path "${BASE_MODEL}" \
    --attn_implementation "${ATTN_IMPL}" \
    --phase_a_steps "${PHASE_A}" \
    --phase_b_steps "${PHASE_B}" \
    --phase_c_steps "${PHASE_C}" \
    --expert_lr "${EXPERT_LR}" \
    --router_lr "${ROUTER_LR}" \
    --lora_lr "${LORA_LR}" \
    --lora_rank "${LORA_RANK}" \
    --expert_bottleneck "${EXPERT_BOTTLENECK}" \
    --expert_kernel_size "${EXPERT_KERNEL_SIZE}" \
    --num_frames "${NUM_FRAMES}" \
    --image_size "${IMAGE_SIZE}" \
    --patch_size "${PATCH_SIZE}" \
    --merge_size "${MERGE_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --dataset_preset "${DATASET_PRESET}" \
    --video_only_data "${VIDEO_ONLY_DATA}" \
    --num_workers "${NUM_WORKERS}" \
    --save_interval "${SAVE_INTERVAL}" \
    --log_interval "${LOG_INTERVAL}"
