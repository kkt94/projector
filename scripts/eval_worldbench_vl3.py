"""
Evaluate VideoLLaMA3 + UDH expert on WorldBench.

Uses the same native hook approach as the lmms-eval wrapper:
expert is injected via encode_images hook so the full VL3 pipeline
(including _get_valid_visual_tokens, token compression, etc.) is preserved.

Usage:
  python scripts/eval_worldbench_vl3.py --output outputs/wb_result.json
  python scripts/eval_worldbench_vl3.py --checkpoint_dir outputs/videollama3_udh_v2_unified_phaseA/best_phase_a --output outputs/wb_expert.json
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATEGORIES = {
    "motion_physics": "motion_physics.json",
    "obj_perm": "obj_perm.json",
    "scale_perspective": "scale_perspective.json",
    "support_rel": "support_rel.json",
}


def load_video_frames_from_dir(scene_dir, num_frames=32):
    """Load frames from PNG/TIFF sequence in a scene directory."""
    exts = ("rgba_*.png", "rgb_*.tiff", "rgb_*.tif", "*.png")
    files = []
    for ext in exts:
        files = sorted(glob.glob(os.path.join(scene_dir, ext)))
        if files:
            break
    if not files:
        return None

    total = len(files)
    if total < num_frames:
        indices = list(range(total))
        while len(indices) < num_frames:
            indices.append(indices[-1])
    else:
        indices = [int(i * total / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        img = Image.open(files[idx]).convert("RGB")
        frames.append(img)
    return frames


def extract_answer(response):
    resp_lower = response.strip().lower()
    if resp_lower.startswith("yes") or resp_lower == "y":
        return "Yes"
    if resp_lower.startswith("no") or resp_lower == "n":
        return "No"
    if "yes" in resp_lower and "no" not in resp_lower:
        return "Yes"
    if "no" in resp_lower and "yes" not in resp_lower:
        return "No"
    first = resp_lower.split()[0] if resp_lower.split() else ""
    if first in ("yes", "yes.", "yes,"):
        return "Yes"
    if first in ("no", "no.", "no,"):
        return "No"
    return response.strip()


def is_correct(pred, gt):
    return pred.strip().rstrip(".").lower() == gt.strip().rstrip(".").lower()


def load_expert(checkpoint_dir, model, device):
    """Load UDH expert + (optional) TemporalQueryRouter from checkpoint."""
    cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None, None, None

    from core.vision_projector.dual_proj import SplitDualHeadExpert
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Stage 2인 경우 stage1_config에서 expert 설정 가져옴
    is_stage2 = cfg.get("module_type", "") == "videollama3_udh_v2_stage2_tqrouter"
    expert_cfg = cfg.get("stage1_config", cfg) if is_stage2 else cfg

    proj_dim = cfg.get("proj_dim", model.config.hidden_size)
    expert = SplitDualHeadExpert(
        proj_dim=proj_dim,
        bottleneck=expert_cfg.get("expert_bottleneck", 128),
        kernel_size=expert_cfg.get("expert_kernel_size", 3),
        pooling_ratio=expert_cfg.get("pooling_ratio", 1),
        ortho_proj=expert_cfg.get("ortho_proj", False),
        max_local_gate=expert_cfg.get("max_local_gate", 0.35),
        max_state_gate=expert_cfg.get("max_state_gate", 0.35),
        skip_output_norm=expert_cfg.get("skip_output_norm", False),
    ).to(device=device, dtype=torch.bfloat16)

    expert_path = os.path.join(checkpoint_dir, "diff_expert.pt")
    expert.load_state_dict(torch.load(expert_path, map_location=device, weights_only=True))
    expert.eval()
    if expert_cfg.get("video_only", False):
        expert.no_gate = True
        print(f"  Expert no_gate=True (from stage1 video_only)")
    print(f"  Loaded expert from {checkpoint_dir}")

    # TemporalQueryRouter 로드 (Stage 2 checkpoint인 경우)
    tq_router = None
    tq_router_path = os.path.join(checkpoint_dir, "tq_router.pt")
    if os.path.exists(tq_router_path):
        from core.vision_projector.temporal_query_router import TemporalQueryRouter
        tq_router = TemporalQueryRouter(
            feat_dim=proj_dim,
            hidden_dim=cfg.get("router_hidden", 256),
            init_bias=cfg.get("stage2_init_bias", 0.0),
        ).to(device=device, dtype=torch.bfloat16)
        tq_router.load_state_dict(torch.load(tq_router_path, map_location=device, weights_only=True))
        tq_router.eval()
        print(f"  Loaded TemporalQueryRouter from {checkpoint_dir}")

    return expert, cfg, tq_router


def evaluate(args):
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = torch.device(f"cuda:{args.gpu}")
    num_frames = args.num_frames

    # Load VL3
    model_path = args.model_path
    print(f"Loading VideoLLaMA3 from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    # Load expert + tq_router
    expert = None
    expert_cfg = None
    tq_router = None
    if args.checkpoint_dir and os.path.isdir(args.checkpoint_dir):
        expert, expert_cfg, tq_router = load_expert(args.checkpoint_dir, model, device)

    # Fixed resolution for expert
    patch_size = 14
    merge_size = 2
    factor = patch_size * merge_size
    target_size = int(round(384.0 / factor) * factor)  # 392

    image_token_id = model.config.image_token_index

    scenes_dir = os.path.join(args.data_dir, "scenes")
    questions_dir = os.path.join(args.data_dir, "textual_questions")
    categories = args.categories or list(CATEGORIES.keys())

    cat_correct = {}
    cat_total = {}
    all_results = []

    for cat in categories:
        qa_file = os.path.join(questions_dir, CATEGORIES[cat])
        if not os.path.exists(qa_file):
            print(f"[WARN] {qa_file} not found")
            continue

        with open(qa_file) as f:
            questions = json.load(f)

        print(f"\n=== {cat} ({len(questions)} questions) ===")
        cat_correct[cat] = 0
        cat_total[cat] = 0

        for qi, q in enumerate(questions):
            video_name = q["video_name"]
            question_text = q["question"]
            gt_answer = q["answer"]

            scene_path = os.path.join(scenes_dir, video_name.replace(".mp4", ""))
            if not os.path.exists(scene_path):
                alt = os.path.join(scenes_dir, video_name)
                if os.path.exists(alt):
                    scene_path = alt
                else:
                    continue

            pil_frames = load_video_frames_from_dir(scene_path, num_frames)
            if pil_frames is None:
                continue

            # Resize to fixed resolution for expert compatibility
            if expert is not None:
                pil_frames = [f.resize((target_size, target_size), Image.BILINEAR) for f in pil_frames]

            is_binary = gt_answer.lower() in ("yes", "no")
            prompt = f"{question_text}\nAnswer with Yes or No." if is_binary else f"{question_text}\nAnswer briefly."

            message = [{"role": "user", "content": [
                {"type": "video", "video": pil_frames, "num_frames": len(pil_frames)},
                {"type": "text", "text": prompt}
            ]}]
            inputs = processor(conversation=message, return_tensors="pt", add_generation_prompt=True)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            input_ids = inputs["input_ids"]

            # Hook for expert
            original_encode = model.encode_images
            hook_fn = None
            if expert is not None:
                T = len(pil_frames)

                # Stage 2 router가 있으면 질문 embedding 미리 추출
                q_embed_for_router = None
                if tq_router is not None:
                    q_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        q_emb_all = model.get_model().embed_tokens(q_ids)  # (1, seq, D)
                        q_embed_for_router = q_emb_all.mean(dim=1)  # (1, D)

                def make_hook(T_val, q_embed=None):
                    def encode_with_expert(pixel_values, grid_sizes, merge_sizes):
                        mm_features = original_encode(pixel_values, grid_sizes, merge_sizes)
                        tpf = mm_features.shape[0] // T_val
                        z_base = mm_features.reshape(1, T_val, tpf, -1)

                        if tq_router is not None and q_embed is not None:
                            # Stage 2: router로 프레임별 gate 적용
                            _, local_res, state_res, _ = expert(z_base, z_base, return_aux=True)
                            z_base_summary = z_base.mean(dim=2)   # (1, T, D)
                            local_summary = local_res.mean(dim=2)  # (1, T, D)
                            state_summary = state_res.mean(dim=2)  # (1, T, D)
                            local_scale, state_scale = tq_router(
                                q_embed, z_base_summary, local_summary, state_summary,
                            )
                            # Enrichment normalize + scale
                            base_norm = z_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                            local_norm = local_res * (base_norm / local_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                            state_norm = state_res * (base_norm / state_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                            ls = local_scale.unsqueeze(-1)
                            ss = state_scale.unsqueeze(-1)
                            z_temp = (1.0 - ls - ss) * z_base + ls * local_norm + ss * state_norm
                        else:
                            # Stage 1: expert의 기본 출력 (내부 gate 사용)
                            z_temp, _, _, _ = expert(z_base, z_base, return_aux=True)

                        return z_temp.reshape(-1, z_temp.shape[-1])
                    return encode_with_expert
                hook_fn = make_hook(T, q_embed_for_router)

            with torch.no_grad():
                if args.decode_mode == "dual_select" and hook_fn is not None:
                    # DS: generate with expert
                    model.encode_images = hook_fn
                    try:
                        out_expert = model.generate(
                            **inputs, do_sample=False, max_new_tokens=32,
                            use_cache=True, pad_token_id=processor.tokenizer.eos_token_id,
                        )
                    finally:
                        model.encode_images = original_encode
                    # DS: generate vanilla
                    out_base = model.generate(
                        **inputs, do_sample=False, max_new_tokens=32,
                        use_cache=True, pad_token_id=processor.tokenizer.eos_token_id,
                    )
                    gen_expert = processor.tokenizer.decode(out_expert[0], skip_special_tokens=True).strip()
                    gen_base = processor.tokenizer.decode(out_base[0], skip_special_tokens=True).strip()
                    if gen_expert == gen_base:
                        response = gen_base
                    else:
                        # Score both via forward pass
                        def _score(out_ids, hfn=None):
                            prompt_len = inputs["input_ids"].shape[1]
                            gen_len = out_ids.shape[1] - prompt_len
                            if gen_len <= 0:
                                return -float("inf")
                            if hfn is not None:
                                model.encode_images = hfn
                            try:
                                out = model(**{k: v for k, v in inputs.items() if k != "input_ids"},
                                           input_ids=out_ids, use_cache=False, return_dict=True)
                            finally:
                                model.encode_images = original_encode
                            logits = out.logits[:, prompt_len-1:-1, :]
                            lp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                            gen_tok = out_ids[:, prompt_len:]
                            tlp = lp.gather(2, gen_tok.unsqueeze(-1)).squeeze(-1)
                            eos_mask = gen_tok.eq(processor.tokenizer.eos_token_id)
                            if eos_mask.any():
                                fe = eos_mask.float().argmax(dim=1).item()
                                if fe == 0: return -float("inf")
                                tlp = tlp[:, :fe]
                            return tlp.mean().item()
                        lp_e = _score(out_expert, hook_fn)
                        lp_b = _score(out_base, None)
                        response = gen_expert if lp_e >= lp_b else gen_base
                else:
                    # Normal: generate with or without expert
                    if hook_fn is not None:
                        model.encode_images = hook_fn
                    try:
                        output_ids = model.generate(
                            **inputs, do_sample=False, max_new_tokens=32,
                            use_cache=True, pad_token_id=processor.tokenizer.eos_token_id,
                        )
                    finally:
                        model.encode_images = original_encode
                    response = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

            pred = extract_answer(response)
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
            json.dump({
                "categories": {c: {"correct": cat_correct[c], "total": cat_total[c]} for c in cat_correct},
                "total_correct": total_c, "total": total_t, "details": all_results,
            }, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="datasets/worldbench")
    p.add_argument("--model_path", type=str, default="models/VideoLLaMA3-2B")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--decode_mode", type=str, default="greedy", choices=["greedy", "dual_select"])
    evaluate(p.parse_args())
