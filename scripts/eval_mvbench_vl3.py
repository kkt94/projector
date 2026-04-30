#!/usr/bin/env python3
"""MVBench MC eval for VideoLLaMA3-7B (vanilla / phase_a) — projector hook.

Mirrors eval_tempcompass_vl3.py: identical code path between vanilla and
phase_a (hook adds expert enrichment only when expert is loaded).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vision_projector.dual_proj import SplitDualHeadExpert  # noqa: E402

# Order matters only for printing.
TASK_VIDEO_ROOT = {}  # populated from test.json


def load_frames(video_path: str, num_frames: int, square_size: int,
                start: float | None = None, end: float | None = None):
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, num_threads=1)
    fps = float(vr.get_avg_fps()) or 30.0
    total = len(vr)
    if start is not None and end is not None and end > start:
        s = max(0, int(start * fps))
        e = min(total, int(end * fps))
        if e - s < num_frames:
            s, e = 0, total
    else:
        s, e = 0, total
    span = max(1, e - s)
    if span < num_frames:
        idx = list(range(s, e)) + [e - 1] * (num_frames - span)
    else:
        idx = [s + int(i * span / num_frames) for i in range(num_frames)]
    frames = vr.get_batch(idx)
    frames = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else np.stack(
        [frames[i].asnumpy() for i in range(num_frames)], axis=0)
    from PIL import Image
    return [Image.fromarray(f.astype(np.uint8)).resize((square_size, square_size), Image.BILINEAR)
            for f in frames]


def maybe_load_expert(checkpoint_dir, proj_dim, device, dtype):
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
    return expert, cfg


LETTERS = ["A", "B", "C", "D", "E"]


def extract_letter(text: str) -> str:
    t = text.strip().upper()
    m = re.search(r"\b([A-E])\b", t)
    return m.group(1) if m else ""


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["vanilla", "phase_a"], required=True)
    ap.add_argument("--checkpoint_dir", default="")
    ap.add_argument("--base_model", default="models/VideoLLaMA3-7B")
    ap.add_argument("--data_dir", default="datasets/eval/MVBench")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--square_size", type=int, default=392,
                    help="Pre-resize size; should match training target_size = "
                         "round(image_size/(patch_size*merge_size)) * (patch_size*merge_size). "
                         "Default 392 matches training image_size=384, patch=14, merge=2.")
    ap.add_argument("--max_per_task", type=int, default=30,
                    help="0 = use all questions per subtask")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.variant != "vanilla" and not args.checkpoint_dir:
        ap.error("--checkpoint_dir required for non-vanilla")

    os.environ.setdefault("HF_HOME", "/home/dev/projects/projector/.cache/huggingface")
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16

    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
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

    proj = model.model.mm_projector
    T = args.num_frames
    proj_orig = proj.forward

    def projector_hook(features, *a, **kw):
        z_out = proj_orig(features, *a, **kw)
        if expert is None:
            return z_out
        D = z_out.shape[-1]
        if z_out.dim() == 2:
            n = z_out.shape[0]
            if n % T != 0:
                return z_out
            M = n // T
            try:
                z_base = z_out.view(1, T, M, D)
                with torch.no_grad():
                    _, lr, sr, _ = expert(z_base, z_base, return_aux=True)
                return (z_base + lr + sr).view(n, D)
            except Exception:
                return z_out
        return z_out

    proj.forward = projector_hook
    print(f"[hook] expert={'on' if expert else 'off'} T={T}", flush=True)

    # Load MVBench task index from local data dir.
    test_idx = json.load(open(os.path.join(args.data_dir, "test.json")))
    task_root = test_idx["root"]
    task_meta = test_idx["meta"]
    json_dir = os.path.join(args.data_dir, "json")

    tasks = args.tasks or sorted(task_root.keys())
    print(f"[data] tasks={len(tasks)} max_per_task={args.max_per_task}", flush=True)

    per_task_correct, per_task_total = {}, {}
    n_skip = 0
    rows = []
    t_start = time.time()

    for ti, task in enumerate(tasks):
        if task not in task_root:
            print(f"[skip-task] {task}", flush=True); continue
        per_task_correct[task] = 0
        per_task_total[task] = 0
        # Prefer json/<task>.json, fallback to test.json meta.
        qpath = os.path.join(json_dir, f"{task}.json")
        if os.path.exists(qpath):
            questions = json.load(open(qpath))
        else:
            questions = task_meta.get(task, [])
        # Stratified sampling: even spread across the (sorted) subtask file so the
        # selected subset reflects the full answer distribution. The first N rows
        # of MVBench json files are answer-biased (e.g., action_localization first
        # 30 are all "Throughout the entire video.").
        if args.max_per_task > 0 and len(questions) > args.max_per_task:
            n = len(questions)
            step = n / args.max_per_task
            questions = [questions[int(i * step)] for i in range(args.max_per_task)]
        video_root = os.path.join(args.data_dir, task_root[task])
        print(f"\n=== {task} ({len(questions)} q) root={task_root[task]} ===", flush=True)

        for qi, q in enumerate(questions):
            vp = os.path.join(video_root, q["video"])
            if not os.path.exists(vp):
                n_skip += 1; continue
            try:
                pil = load_frames(vp, args.num_frames, args.square_size,
                                  start=q.get("start"), end=q.get("end"))
            except Exception:
                n_skip += 1; continue

            cands = q["candidates"]
            opts = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(cands))
            prompt = (q["question"].strip() + "\n" + opts +
                      "\nAnswer with the option's letter from the given choices directly.")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": pil, "num_frames": len(pil)},
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
                gen_ids = out[0]
                if gen_ids.shape[0] >= inputs["input_ids"].shape[1] + 1:
                    gen_ids = gen_ids[inputs["input_ids"].shape[1]:]
                response = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); n_skip += 1; continue
            except Exception:
                n_skip += 1; continue

            pred = extract_letter(response)
            try:
                gold_idx = cands.index(q["answer"])
                gold_letter = LETTERS[gold_idx]
            except ValueError:
                gold_letter = ""
            ok = (pred == gold_letter) and gold_letter != ""
            per_task_total[task] += 1
            if ok:
                per_task_correct[task] += 1
            rows.append({"task": task, "video": q["video"],
                         "question": q["question"], "answer": q["answer"],
                         "pred": response, "ok": ok})
            if (qi + 1) % 25 == 0:
                acc = per_task_correct[task] / max(1, per_task_total[task])
                print(f"  [{qi+1}/{len(questions)}] acc={acc:.3f}", flush=True)
        print(f"=== {task}: {per_task_correct[task]}/{per_task_total[task]} = "
              f"{per_task_correct[task]/max(1, per_task_total[task]):.3f} ===", flush=True)

    total_correct = sum(per_task_correct.values())
    total_eval = sum(per_task_total.values())
    out = {
        "variant": args.variant,
        "checkpoint_dir": args.checkpoint_dir,
        "tasks": {t: {"correct": per_task_correct[t], "total": per_task_total[t],
                      "accuracy": per_task_correct[t]/max(1, per_task_total[t])}
                  for t in per_task_correct},
        "overall_accuracy": total_correct / max(1, total_eval),
        "total_correct": total_correct, "total_eval": total_eval,
        "n_skip": n_skip, "duration_sec": time.time() - t_start, "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== {args.variant} OVERALL ===", flush=True)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
