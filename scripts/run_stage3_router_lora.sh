#!/bin/bash
# Stage 3: Router(step800) frozen + LoRA rank=32 학습
# Expert(Stage1 video-only) + Router(Stage2 step800) 모두 frozen, LoRA만 학습

OUTPUT_DIR="outputs/internvl3_stage3_router800_lora_r32"
STAGE2_CKPT="outputs/internvl3_stage2_router/ckpt_step_800"

torchrun --nproc_per_node=4 --master_port=29501 \
    scripts/train_internvl3_udh_unified_curriculum.py \
    --stage3 \
    --stage3_checkpoint "$STAGE2_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank 32 \
    --lora_lr 2e-4 \
    --batch_size 6 \
    --grad_accum 2 \
    --max_steps 600 \
    --save_interval 200 \
    --log_interval 10 \
    --num_frames 32 \
    --dataset_preset full+eval \
    --align_lambda 0.0 \
    --num_workers 4 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
