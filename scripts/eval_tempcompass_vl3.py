#!/usr/bin/env python3
"""TempCompass MC eval for VideoLLaMA3-7B (vanilla / Phase A) — projector hook.

Identical code path between vanilla and phase_a (hook adds expert enrichment
only when expert is loaded). Pre-resizes frames to a square `--square_size`
to keep VL3 mm_projector output reshape consistent for the expert.
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

DIM_ORDER = ["action", "direction", "speed", "order", "attribute_change"]


def load_frames(video_path: str, num_frames: int, square_size: int):
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    if total < num_frames:
        idx = list(range(total)) + [total - 1] * (num_frames - total)
    else:
        idx = [int(i * total / num_frames) for i in range(num_frames)]
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


def extract_letter(text: str) -> str:
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else ""


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["vanilla", "phase_a"], required=True)
    ap.add_argument("--checkpoint_dir", default="")
    ap.add_argument("--base_model", default="models/VideoLLaMA3-7B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--square_size", type=int, default=392,
                    help="Pre-resize size; should match training target_size = "
                         "round(image_size/(patch_size*merge_size)) * (patch_size*merge_size). "
                         "Default 392 matches training image_size=384, patch=14, merge=2.")
    ap.add_argument("--max_samples", type=int, default=0)
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

    from datasets import load_dataset
    ds = load_dataset("lmms-lab/TempCompass", "multi-choice", split="test")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    cache_dir = "/home/dev/projects/projector/.cache/huggingface/tempcompass/videos"
    print(f"[data] {len(ds)} questions", flush=True)

    correct = {d: 0 for d in DIM_ORDER}
    total = {d: 0 for d in DIM_ORDER}
    n_skip = 0
    t_start = time.time()
    rows = []

    for i, ex in enumerate(ds):
        vp = os.path.join(cache_dir, f"{ex['video_id']}.mp4")
        if not os.path.exists(vp):
            n_skip += 1
            continue
        try:
            pil = load_frames(vp, args.num_frames, args.square_size)
        except Exception:
            n_skip += 1
            continue
        prompt = ex["question"] + "\nPlease directly give the best option:"
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": pil, "num_frames": len(pil)},
                {"type": "text", "text": prompt},
            ],
        }]
        try:
            inputs = processor(conversation=messages, return_tensors="pt", add_generation_prompt=True)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            gen_ids = out[0]
            if gen_ids.shape[0] >= inputs["input_ids"].shape[1] + 1:
                gen_ids = gen_ids[inputs["input_ids"].shape[1]:]
            response = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_skip += 1
            continue
        except Exception:
            n_skip += 1
            continue

        pred = extract_letter(response)
        gold = str(ex["answer"]).strip().upper()
        gold_letter = extract_letter(gold) or gold[:1]
        ok = pred == gold_letter
        d = ex.get("dim", "speed")
        if d in total:
            total[d] += 1
            if ok:
                correct[d] += 1
        rows.append({"video_id": ex["video_id"], "dim": d, "question": ex["question"],
                     "answer": gold, "pred": response, "ok": ok})
        if (i + 1) % 50 == 0:
            done = sum(total.values())
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(ds)}] done={done} skip={n_skip} t={elapsed:.0f}s "
                  f"acc={sum(correct.values())/max(1,done):.3f}", flush=True)

    out = {"variant": args.variant, "checkpoint_dir": args.checkpoint_dir,
           "total": total, "correct": correct,
           "per_dim_accuracy": {d: (correct[d]/total[d] if total[d] else None) for d in DIM_ORDER},
           "overall_accuracy": sum(correct.values()) / max(1, sum(total.values())),
           "n_skip": n_skip, "n_eval": sum(total.values()),
           "duration_sec": time.time() - t_start, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
