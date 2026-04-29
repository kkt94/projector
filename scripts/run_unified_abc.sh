#!/bin/bash
# Unified A/B/C training: Expert → Router → LoRA in one loop
# Phase A: expert (aux only, skip_output_norm, 800 steps)
# Phase B: router (QA loss, NO normalize, 800 steps)
# Phase C: LoRA (QA loss, 600 steps)

OUTPUT_DIR="outputs/internvl3_unified_abc_v1"

torchrun --nproc_per_node=4 --master_port=29501 \
    scripts/train_internvl3_unified_abc.py \
    --output_dir "$OUTPUT_DIR" \
    --phase_a_steps 800 \
    --phase_b_steps 800 \
    --phase_c_steps 600 \
    --expert_lr 1e-3 \
    --router_lr 5e-4 \
    --router_hidden 128 \
    --router_init_bias -1.0 \
    --lora_rank 32 \
    --lora_lr 2e-4 \
    --batch_size 6 \
    --grad_accum 2 \
    --num_frames 32 \
    --save_interval 200 \
    --log_interval 10 \
    --dataset_preset full+eval \
    --video_only_data datasets/video_only_stage1.jsonl \
    --num_workers 4 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
