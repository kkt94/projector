#!/bin/bash
# VL3 Stage 2: TemporalQueryRouter 학습
# IV3 Stage 2와 동일 설정: init_bias=-1.0, router_lr=5e-4, align_lambda=0.0

OUTPUT_DIR="outputs/vl3_stage2_router"
STAGE1_CKPT="outputs/vl3_stage1_video_only/ckpt_step_800"

torchrun --nproc_per_node=4 --master_port=29502 \
    scripts/train_videollama3_udh_unified_curriculum.py \
    --stage2 \
    --stage2_checkpoint "$STAGE1_CKPT" \
    --stage2_init_bias -1.0 \
    --output_dir "$OUTPUT_DIR" \
    --router_hidden 128 \
    --router_lr 5e-4 \
    --align_lambda 0.0 \
    --batch_size 8 \
    --grad_accum 1 \
    --max_steps 1200 \
    --save_interval 200 \
    --log_interval 10 \
    --num_frames 32 \
    --dataset_preset full+eval \
    --num_workers 4 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
