#!/bin/bash
# Stage 1 video-only expert 학습 — output_norm 제거 + up_proj gain=0.1
# enrichment가 자연스러운 스케일로 학습되는지 확인

OUTPUT_DIR="outputs/internvl3_stage1_skip_norm"

torchrun --nproc_per_node=4 --master_port=29501 \
    scripts/train_internvl3_udh_unified_curriculum.py \
    --video_only \
    --skip_output_norm \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 10 \
    --grad_accum 1 \
    --max_steps 800 \
    --save_interval 200 \
    --log_interval 10 \
    --num_frames 32 \
    --expert_bottleneck 128 \
    --expert_kernel_size 3 \
    --align_lambda 0.1 \
    --state_mask_lambda 0.03 \
    --local_corr_lambda 0.01 \
    --dataset_preset full \
    --num_workers 4 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
