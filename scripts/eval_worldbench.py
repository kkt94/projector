"""
Evaluate PLM on WorldBench text-based questions.

WorldBench has 181 questions across 4 physics categories:
  - motion_physics (31)
  - obj_perm (84)
  - scale_perspective (38)
  - support_rel (28)

Videos are stored as TIFF frame sequences in scenes/ directory.
This script converts them to frames on-the-fly.

Usage:
  python scripts/eval_worldbench.py --data_dir datasets/worldbench
  python scripts/eval_worldbench.py --data_dir datasets/worldbench --categories motion_physics obj_perm
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATEGORIES = {
    "motion_physics": "motion_physics.json",
    "obj_perm": "obj_perm.json",
    "scale_perspective": "scale_perspective.json",
    "support_rel": "support_rel.json",
}


def load_video_frames_from_tiffs(scene_dir, num_frames=32):
    """Load frames from TIFF sequence in a scene directory."""
    rgb_dir = os.path.join(scene_dir, "rgb")
    if not os.path.exists(rgb_dir):
        # Try direct scene dir
        rgb_dir = scene_dir

    tiff_files = sorted(glob.glob(os.path.join(rgb_dir, "rgba_*.png")) +
                        glob.glob(os.path.join(rgb_dir, "rgb_*.tiff")) +
                        glob.glob(os.path.join(rgb_dir, "rgb_*.tif")) +
                        glob.glob(os.path.join(rgb_dir, "*.png")))
    if not tiff_files:
        return None

    total = len(tiff_files)
    if total < num_frames:
        indices = list(range(total))
        while len(indices) < num_frames:
            indices.append(indices[-1])
    else:
        indices = [int(i * total / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        img = Image.open(tiff_files[idx]).convert("RGB")
        frames.append(np.array(img))
    return frames


def extract_answer(response, answer_type="binary"):
    """Extract answer from model response."""
    response = response.strip()

    if answer_type == "binary":
        # Yes/No extraction
        resp_lower = response.lower()
        if resp_lower.startswith("yes") or resp_lower == "y":
            return "Yes"
        if resp_lower.startswith("no") or resp_lower == "n":
            return "No"
        # Search in longer response
        if "yes" in resp_lower and "no" not in resp_lower:
            return "Yes"
        if "no" in resp_lower and "yes" not in resp_lower:
            return "No"
        # First word heuristic
        first = resp_lower.split()[0] if resp_lower.split() else ""
        if first in ("yes", "yes.", "yes,"):
            return "Yes"
        if first in ("no", "no.", "no,"):
            return "No"
        return response
    else:
        # MCQ — extract letter or option text
        return response


def is_correct(pred, gt):
    """Compare prediction with ground truth."""
    pred_norm = pred.strip().rstrip(".").lower()
    gt_norm = gt.strip().rstrip(".").lower()
    return pred_norm == gt_norm


def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}")
    num_frames = args.num_frames

    print("Loading PLM-1B...")
    processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B", local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        "facebook/Perception-LM-1B", torch_dtype=torch.bfloat16, device_map={"": device}, local_files_only=True
    )
    model.eval()

    # Load checkpoint adapter if specified
    adapter_design = None
    adapter_objs = None
    adapter_cfg = None
    if args.checkpoint_dir:
        cfg_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
        if os.path.exists(cfg_path):
            print(f"Loading checkpoint: {args.checkpoint_dir}")
            lmms_path = "/tmp/lmms-eval"
            if lmms_path not in sys.path:
                sys.path.insert(0, lmms_path)
            from lmms_eval.models.simple.plm_dual_proj import _load_dual_proj_adapter, _load_llm_lora
            param_dtype = next(model.parameters()).dtype
            _load_llm_lora(args.checkpoint_dir, model)
            adapter_design, adapter_objs, adapter_cfg = _load_dual_proj_adapter(
                args.checkpoint_dir, model, device, param_dtype)
            print(f"  Loaded adapter: design={adapter_design}")

    categories = args.categories or list(CATEGORIES.keys())
    scenes_dir = os.path.join(args.data_dir, "scenes")
    questions_dir = os.path.join(args.data_dir, "textual_questions")

    cat_correct = {}
    cat_total = {}
    all_results = []

    for cat in categories:
        qa_file = os.path.join(questions_dir, CATEGORIES[cat])
        if not os.path.exists(qa_file):
            print(f"[WARN] QA file not found: {qa_file}")
            continue

        with open(qa_file) as f:
            questions = json.load(f)

        print(f"\n=== {cat} ({len(questions)} questions) ===")
        cat_correct[cat] = 0
        cat_total[cat] = 0

        for qi, q in enumerate(questions):
            video_name = q["video_name"]  # e.g., "motion_phys/ball_bounce/1.mp4"
            question_text = q["question"]
            gt_answer = q["answer"]

            # Resolve scene directory (strip .mp4, look in scenes/)
            scene_path = os.path.join(scenes_dir, video_name.replace(".mp4", ""))
            if not os.path.exists(scene_path):
                # Try alternative paths
                alt = os.path.join(scenes_dir, video_name)
                if os.path.exists(alt):
                    scene_path = alt
                else:
                    print(f"  [SKIP] Scene not found: {scene_path}")
                    continue

            frames = load_video_frames_from_tiffs(scene_path, num_frames)
            if frames is None:
                print(f"  [SKIP] No frames in {scene_path}")
                continue

            # Determine answer type
            is_binary = gt_answer.lower() in ("yes", "no")

            # Build prompt
            if is_binary:
                prompt = f"{question_text}\nAnswer with Yes or No."
            else:
                prompt = f"{question_text}\nAnswer briefly."

            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt}
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, videos=[frames], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                if adapter_design and adapter_design in ("dbrouter", "diff_delta_expert", "lora_sft_expert", "udh_v2_res2"):
                    from scripts.eval_cosmos_reason1 import _generate_with_adapter
                    output_ids = _generate_with_adapter(
                        model, processor, inputs, adapter_design, adapter_objs, adapter_cfg, device, num_frames)
                else:
                    output_ids = model.generate(
                        **inputs, max_new_tokens=32, do_sample=False, temperature=None, top_p=None)

            input_len = inputs["input_ids"].shape[1]
            if output_ids.shape[1] > input_len:
                response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
            else:
                response = processor.decode(output_ids[0], skip_special_tokens=True).strip()

            pred = extract_answer(response, "binary" if is_binary else "other")
            correct = is_correct(pred, gt_answer)
            cat_correct[cat] += int(correct)
            cat_total[cat] += 1

            all_results.append({
                "category": cat, "video": video_name, "question": question_text,
                "gt": gt_answer, "pred": pred, "response": response, "correct": correct,
            })

            if (qi + 1) % 10 == 0:
                print(f"  [{qi+1}/{len(questions)}] {cat_correct[cat]}/{cat_total[cat]}")

    # Summary
    print("\n" + "=" * 60)
    print("WorldBench Results")
    print("=" * 60)
    total_c, total_t = 0, 0
    for cat in sorted(cat_correct.keys()):
        c, t = cat_correct[cat], cat_total[cat]
        print(f"  {cat:<25} {c:>3}/{t:<3} ({100*c/max(t,1):.1f}%)")
        total_c += c
        total_t += t
    print(f"  {'TOTAL':<25} {total_c:>3}/{total_t:<3} ({100*total_c/max(total_t,1):.1f}%)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"categories": {c: {"correct": cat_correct[c], "total": cat_total[c]} for c in cat_correct},
                        "total_correct": total_c, "total": total_t, "details": all_results}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate on WorldBench")
    p.add_argument("--data_dir", type=str, default="datasets/worldbench")
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default=None, help="Temporal expert checkpoint")
    p.add_argument("--output", type=str, default=None)
    evaluate(p.parse_args())
