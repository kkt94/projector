#!/usr/bin/env python3
"""WorldBench evaluation for VideoLLaMA3 (vanilla / Phase A).

Mirrors eval_worldbench_plm3b.py: same projector hook flow so vanilla and
phase_a use IDENTICAL code path; the hook only adds expert enrichment when
expert is loaded.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.vision_projector.dual_proj import SplitDualHeadExpert  # noqa: E402

CATEGORIES = {
    "motion_physics": "motion_physics.json",
    "obj_perm": "obj_perm.json",
    "scale_perspective": "scale_perspective.json",
    "support_rel": "support_rel.json",
}


def load_video_frames_from_dir(scene_dir: str, num_frames: int = 16, square_size: int = 384):
    exts = ("rgba_*.png", "normal_*.png", "rgb_*.tiff", "rgb_*.tif", "*.png")
    files = []
    for ext in exts:
        files = sorted(glob.glob(os.path.join(scene_dir, ext)))
        if files:
            break
    if not files:
        return None
    total = len(files)
    if total < num_frames:
        idx = list(range(total)) + [total - 1] * (num_frames - total)
    else:
        idx = [int(i * total / num_frames) for i in range(num_frames)]
    from PIL import Image
    # Force square frames so VL3 + expert internal spatial pool stays consistent.
    return [Image.open(files[i]).convert("RGB").resize((square_size, square_size), Image.BILINEAR)
            for i in idx]


def maybe_load_expert(checkpoint_dir: str, proj_dim: int, device, dtype):
    cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None, None
    with open(cfg_path) as f:
        cfg = json.load(f)
    expert = SplitDualHeadExpert(
        proj_dim=proj_dim,
        bottleneck=cfg.get("expert_bottleneck", 128),
        kernel_size=cfg.get("expert_kernel_size", 3),
        max_local_gate=cfg.get("max_local_gate", 0.35),
        max_state_gate=cfg.get("max_state_gate", 0.35),
        pooling_ratio=cfg.get("pooling_ratio", 1),
        skip_output_norm=cfg.get("skip_output_norm", True),
    ).to(device=device, dtype=dtype)
    expert.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"),
                   map_location=device, weights_only=True), strict=False)
    expert.eval()
    expert.no_gate = True
    print(f"[expert] loaded from {checkpoint_dir}", flush=True)
    return expert, cfg


def extract_yes_no(text: str) -> str:
    t = text.strip().lower()
    if not t:
        return ""
    if t.startswith("yes") or t == "y":
        return "Yes"
    if t.startswith("no") or t == "n":
        return "No"
    if "yes" in t and "no" not in t:
        return "Yes"
    if "no" in t and "yes" not in t:
        return "No"
    return text.strip()


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["vanilla", "phase_a"], required=True)
    ap.add_argument("--checkpoint_dir", default="")
    ap.add_argument("--base_model", default="models/VideoLLaMA3-7B")
    ap.add_argument("--data_dir", default="datasets/eval/WorldBench")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=32,
                    help="Match training num_frames (default 32 for VL3 run01).")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--merge_size", type=int, default=2)
    ap.add_argument("--square_size", type=int, default=392,
                    help="Pre-resize size; should match training target_size = "
                         "round(image_size/(patch_size*merge_size)) * (patch_size*merge_size). "
                         "Default 392 matches training image_size=384, patch=14, merge=2.")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.variant != "vanilla" and not args.checkpoint_dir:
        ap.error("--checkpoint_dir required for non-vanilla variants")

    os.environ.setdefault("HF_HOME", "/home/dev/projects/projector/.cache/huggingface")
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16

    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    print(f"[load] {args.base_model} on cuda:{args.gpu}", flush=True)
    cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    if hasattr(cfg, "vision_encoder_config"):
        cfg.vision_encoder_config._attn_implementation_internal = "sdpa"
    cfg._attn_implementation_internal = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, config=cfg, trust_remote_code=True,
        torch_dtype=dtype, device_map={"": device})
    model.eval()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    proj_dim = int(model.config.hidden_size)

    expert = None
    if args.variant != "vanilla":
        expert, _ = maybe_load_expert(args.checkpoint_dir, proj_dim, device, dtype)

    # VL3 mm_projector hook (used by `_build_vision_features` flow).
    proj = model.model.mm_projector
    T = args.num_frames
    proj_orig = proj.forward

    def projector_hook(features, *a, **kw):
        z_out = proj_orig(features, *a, **kw)
        if expert is None:
            return z_out
        # VL3 mm_projector returns flat (B*T*M, D); reshape to (B, T, M, D)
        D = z_out.shape[-1]
        if z_out.dim() == 2:
            n = z_out.shape[0]
            # B=1 for eval, T from outer scope; M = n / (B*T)
            BT = max(1, T)
            if n % BT != 0:
                return z_out  # cannot reshape; bail
            M = n // BT
            try:
                z_base = z_out.view(1, BT, M, D)
                with torch.no_grad():
                    _, lr, sr, _ = expert(z_base, z_base, return_aux=True)
                return (z_base + lr + sr).view(n, D)
            except Exception as e:
                print(f"  [hook err] {type(e).__name__}: {e}", flush=True)
                return z_out
        elif z_out.dim() == 3:
            n = z_out.shape[0]
            B = max(1, n // T)
            M = z_out.shape[1]
            try:
                z_base = z_out.view(B, T, M, D)
                with torch.no_grad():
                    _, lr, sr, _ = expert(z_base, z_base, return_aux=True)
                return (z_base + lr + sr).view(n, M, D)
            except Exception:
                return z_out
        return z_out

    proj.forward = projector_hook
    print(f"[hook] expert={'on' if expert else 'off'} T={T}", flush=True)

    scenes_dir = os.path.join(args.data_dir, "scenes")
    questions_dir = os.path.join(args.data_dir, "textual_questions")
    categories = args.categories or list(CATEGORIES.keys())

    cat_correct = {}
    cat_total = {}
    rows = []
    n_skip = 0
    t_start = time.time()

    for cat in categories:
        qa_file = os.path.join(questions_dir, CATEGORIES[cat])
        if not os.path.exists(qa_file):
            print(f"[skip-cat] {qa_file}", flush=True); continue
        with open(qa_file) as f:
            questions = json.load(f)
        print(f"\n=== {cat} ({len(questions)} questions) ===", flush=True)
        cat_correct[cat] = 0; cat_total[cat] = 0

        for qi, q in enumerate(questions):
            video_name = q["video_name"]
            scene_path = os.path.join(scenes_dir, video_name.replace(".mp4", ""))
            if not os.path.exists(scene_path):
                alt = os.path.join(scenes_dir, video_name)
                if os.path.exists(alt):
                    scene_path = alt
                else:
                    n_skip += 1; continue
            pil_frames = load_video_frames_from_dir(scene_path, args.num_frames, args.square_size)
            if pil_frames is None:
                n_skip += 1; continue

            gold = str(q["answer"]).strip()
            is_binary = gold.lower() in ("yes", "no")
            prompt = (q["question"] + ("\nAnswer with Yes or No." if is_binary
                                       else "\nAnswer briefly."))
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "num_frames": len(pil_frames)},
                    {"type": "text", "text": prompt},
                ],
            }]
            try:
                inputs = processor(conversation=messages, return_tensors="pt", add_generation_prompt=True)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                with torch.amp.autocast("cuda", dtype=dtype):
                    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                # VL3's generate returns only the new tokens (not input + new),
                # whereas PLM returns the full sequence. Detect by length.
                gen_ids = out[0]
                if gen_ids.shape[0] >= inputs["input_ids"].shape[1] + 1:
                    gen_ids = gen_ids[inputs["input_ids"].shape[1]:]
                response = processor.tokenizer.decode(
                    gen_ids, skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_skip += 1; continue
            except Exception as e:
                n_skip += 1
                if n_skip <= 3:
                    print(f"  [err] {video_name}: {type(e).__name__}: {e}", flush=True)
                continue

            pred = extract_yes_no(response) if is_binary else response.strip()
            # Match WorldBench reference scoring: strip trailing periods.
            # Without this, "Right." vs gold "Right" is marked wrong.
            ok = (pred.strip().rstrip(".").lower() == gold.strip().rstrip(".").lower())
            cat_total[cat] += 1
            if ok: cat_correct[cat] += 1
            rows.append({"category": cat, "video_name": video_name,
                         "question": q["question"], "answer": gold,
                         "pred": response, "ok": ok})
            if (qi + 1) % 10 == 0:
                acc = cat_correct[cat] / max(1, cat_total[cat])
                print(f"  [{qi+1}/{len(questions)}] cat_acc={acc:.3f}", flush=True)

        print(f"=== {cat}: {cat_correct[cat]}/{cat_total[cat]} = "
              f"{cat_correct[cat]/max(1, cat_total[cat]):.3f} ===", flush=True)

    total_correct = sum(cat_correct.values())
    total_eval = sum(cat_total.values())
    overall = total_correct / max(1, total_eval)
    out = {
        "variant": args.variant,
        "checkpoint_dir": args.checkpoint_dir,
        "categories": {c: {"correct": cat_correct[c], "total": cat_total[c],
                           "accuracy": cat_correct[c]/max(1, cat_total[c])}
                       for c in cat_correct},
        "overall_accuracy": overall,
        "total_correct": total_correct,
        "total_eval": total_eval,
        "n_skip": n_skip,
        "duration_sec": time.time() - t_start,
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== {args.variant} OVERALL ===", flush=True)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
