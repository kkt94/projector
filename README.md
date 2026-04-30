# VideoLLaMA3-7B Evaluation Bundle

Eval/train code for the dual-head expert (Phase A) on VideoLLaMA3-7B with all
recently applied fixes (patch-order, bicubic+antialias resize, scoring, sampling).

## Layout

```
scripts/
  eval_mvbench_vl3.py            # MVBench MC eval (20 subtasks × 200q)
  eval_tempcompass_vl3.py        # TempCompass MC eval (1580q)
  eval_worldbench_vl3_simple.py  # WorldBench eval (181q, 4 categories)
  train_videollama3_unified_abc.py            # Phase A/B/C training main
  train_videollama3_udh_unified_curriculum.py # dataset / loaders
core/vision_projector/
  dual_proj.py                   # SplitDualHeadExpert
  temporal_query_router.py       # router (Phase B; not used here)
tools/
  run_mvbench_full.sh            # batch runner reference
```

## Required deps (host env)

- python 3.12
- torch (with cuda + flash_attn2)
- transformers, datasets, peft, accelerate
- decord, pillow, numpy
- VideoLLaMA3-7B weights at `models/VideoLLaMA3-7B` (or `--base_model PATH`)

## Eval commands

`HF_HOME` must point to a cache that holds the model + datasets.

```bash
export HF_HOME=/path/to/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# TempCompass (1580q, ~30min)
python scripts/eval_tempcompass_vl3.py \
  --variant phase_a --gpu 0 \
  --base_model models/VideoLLaMA3-7B \
  --checkpoint_dir <your_ckpt_dir> \
  --output out/tc.json

# WorldBench (181q, ~10min)
python scripts/eval_worldbench_vl3_simple.py \
  --variant phase_a --gpu 0 \
  --base_model models/VideoLLaMA3-7B \
  --checkpoint_dir <your_ckpt_dir> \
  --output out/wb.json

# MVBench full (4000q, ~3h)  — set --max_per_task 0
python scripts/eval_mvbench_vl3.py \
  --variant phase_a --gpu 0 \
  --max_per_task 0 \
  --base_model models/VideoLLaMA3-7B \
  --checkpoint_dir <your_ckpt_dir> \
  --output out/mb.json

# vanilla baseline: --variant vanilla, omit --checkpoint_dir
```

## Key fixes baked into this bundle

1. **Patch-order fix** in `train_videollama3_unified_abc.py:_build_vision_features`
   – matches Videollama3ImageProcessor's meta-block-grouped order.
2. **Bicubic + antialias resize** in `_preprocess_frames` /
   `VideoOnlyDataset.__getitem__` (matches the official processor).
3. **--square_size 392** default in eval scripts (no double resize).
4. **Stratified sampling** in `eval_mvbench_vl3.py` (MVBench json files are
   answer-biased; first-30 slice is unrepresentative).
5. **WB scoring** strips trailing `.` before comparing (`pred.rstrip(".")`).

## Datasets expected on disk

- MVBench: `datasets/eval/MVBench/{test.json, json/, <video_root>/}`
- TempCompass: HF datasets `lmms-lab/TempCompass` + videos under
  `.cache/huggingface/tempcompass/videos/`.
- WorldBench: `datasets/eval/WorldBench/{scenes/, textual_questions/}`
  (TIFF/PNG frame sequences).

## Checkpoint format expected

`<ckpt_dir>/` must contain:
- `adapter_config.json`  — { proj_dim, expert_bottleneck, expert_kernel_size,
                              max_local_gate, max_state_gate, pooling_ratio,
                              skip_output_norm, video_only, ... }
- `diff_expert.pt`        — `SplitDualHeadExpert.state_dict()`

(For Phase B/C the bundle also expects `tq_router.pt` and `lora/` — not used
in vanilla / phase_a evals.)
