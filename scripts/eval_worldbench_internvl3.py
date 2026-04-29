"""Evaluate InternVL3 + UDH expert on WorldBench.

Expert is injected via extract_feature hook, same as lmms-eval wrapper.

Usage:
  python scripts/eval_worldbench_internvl3.py --output outputs/wb_internvl3_vanilla.json
  python scripts/eval_worldbench_internvl3.py --checkpoint_dir outputs/internvl3_udh_phaseA/best_phase_a --output outputs/wb_internvl3_expert.json
"""
import argparse, glob, json, os, sys, re
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

def load_frames(scene_dir, num_frames=32):
    exts = ("rgba_*.png", "rgb_*.tiff", "rgb_*.tif", "*.png")
    files = []
    for ext in exts:
        files = sorted(glob.glob(os.path.join(scene_dir, ext)))
        if files: break
    if not files: return None
    total = len(files)
    if total < num_frames:
        indices = list(range(total))
        while len(indices) < num_frames: indices.append(indices[-1])
    else:
        indices = [int(i * total / num_frames) for i in range(num_frames)]
    return [Image.open(files[i]).convert("RGB") for i in indices]

def extract_answer(response):
    r = response.strip()
    if r.lower() in ("yes", "no"): return r.lower()
    m = re.search(r"\b(yes|no)\b", r, re.IGNORECASE)
    if m: return m.group(1).lower()
    return r.split(".")[0].strip().lower()

def is_correct(pred, gt):
    return pred.lower().strip() == gt.lower().strip()

def load_expert(checkpoint_dir, device):
    from core.vision_projector.dual_proj import SplitDualHeadExpert
    cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None, None, None
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Stage 2인 경우 stage1_config에서 expert 설정 가져옴
    is_stage2_or_3 = cfg.get("module_type", "") in (
        "internvl3_udh_v2_stage2_tqrouter", "internvl3_udh_v2_stage3_lora",
    )
    expert_cfg = cfg.get("stage1_config", cfg) if is_stage2_or_3 else cfg

    expert = SplitDualHeadExpert(
        proj_dim=cfg.get("proj_dim", 2048),
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
        proj_dim = cfg.get("proj_dim", 2048)
        tq_router = TemporalQueryRouter(
            feat_dim=proj_dim,
            hidden_dim=cfg.get("router_hidden") or cfg.get("stage2_config", {}).get("router_hidden", 256),
            init_bias=cfg.get("stage2_init_bias", 0.0),
        ).to(device=device, dtype=torch.bfloat16)
        tq_router.load_state_dict(torch.load(tq_router_path, map_location=device, weights_only=True))
        tq_router.eval()
        print(f"  Loaded TemporalQueryRouter from {checkpoint_dir}")

    return expert, cfg, tq_router

def evaluate(args):
    from transformers import AutoModel, AutoTokenizer
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    device = torch.device(f"cuda:{args.gpu}")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device}).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    # Load expert + tq_router
    expert = None
    tq_router = None
    if args.checkpoint_dir and os.path.isdir(args.checkpoint_dir):
        expert, _, tq_router = load_expert(args.checkpoint_dir, device)
        # Load LoRA if present
        lora_dir = os.path.join(args.checkpoint_dir, "lora")
        if os.path.isdir(lora_dir):
            from peft import PeftModel
            model.language_model = PeftModel.from_pretrained(
                model.language_model, lora_dir
            ).to(device=device, dtype=torch.bfloat16)
            model.language_model.eval()
            print(f"  Loaded LoRA from {lora_dir}")

    scenes_dir = os.path.join(args.data_dir, "scenes")
    questions_dir = os.path.join(args.data_dir, "textual_questions")
    categories = list(CATEGORIES.keys())

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    cat_correct = {}; cat_total = {}; all_results = []

    for cat in categories:
        qa_file = os.path.join(questions_dir, CATEGORIES[cat])
        if not os.path.exists(qa_file): continue
        with open(qa_file) as f: questions = json.load(f)
        print(f"\n=== {cat} ({len(questions)} questions) ===")
        cat_correct[cat] = 0; cat_total[cat] = 0

        for qi, q in enumerate(questions):
            video_name = q["video_name"]
            question_text = q["question"]
            gt_answer = q["answer"]

            scene_path = os.path.join(scenes_dir, video_name.replace(".mp4", ""))
            if not os.path.exists(scene_path):
                alt = os.path.join(scenes_dir, video_name)
                if os.path.exists(alt): scene_path = alt
                else: continue

            pil_frames = load_frames(scene_path, args.num_frames)
            if pil_frames is None: continue

            is_binary = gt_answer.lower() in ("yes", "no")
            prompt = f"{question_text}\nAnswer with Yes or No." if is_binary else f"{question_text}\nAnswer briefly."

            pixel_values_list = []
            num_patches_list = []
            for frame in pil_frames:
                pv = transform(frame).unsqueeze(0)
                pixel_values_list.append(pv)
                num_patches_list.append(1)

            pixel_values = torch.cat(pixel_values_list, dim=0).to(device=device, dtype=torch.bfloat16)

            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(pil_frames))])
            question = video_prefix + prompt
            generation_config = dict(max_new_tokens=32, do_sample=False)

            # Hook expert if loaded
            original_extract = model.extract_feature
            if expert is not None:
                T = len(pil_frames)

                # Stage 2 router가 있으면 질문 embedding 미리 추출
                q_embed_for_router = None
                if tq_router is not None:
                    q_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        q_emb_all = model.get_input_embeddings()(q_ids)  # (1, seq, D)
                        q_embed_for_router = q_emb_all.mean(dim=1)  # (1, D)

                def make_hook(T_val, q_embed=None):
                    def extract_with_expert(pv):
                        features = original_extract(pv)
                        tpf = features.shape[1]
                        z_base = features.reshape(1, T_val, tpf, -1)

                        if tq_router is not None and q_embed is not None:
                            # Stage 2: router로 프레임별 gate 적용
                            _, local_res, state_res, _ = expert(z_base, z_base, return_aux=True)
                            z_base_summary = z_base.mean(dim=2)   # (1, T, D)
                            local_summary = local_res.mean(dim=2)  # (1, T, D)
                            state_summary = state_res.mean(dim=2)  # (1, T, D)
                            local_scale, state_scale = tq_router(
                                q_embed, z_base_summary, local_summary, state_summary,
                            )
                            # Enrichment normalize + scale 적용
                            base_norm = z_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                            local_norm = local_res * (base_norm / local_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                            state_norm = state_res * (base_norm / state_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                            ls = local_scale.unsqueeze(-1)
                            ss = state_scale.unsqueeze(-1)
                            z_temp = (1.0 - ls - ss) * z_base + ls * local_norm + ss * state_norm
                        else:
                            # Stage 1: expert의 기본 출력 (내부 gate 사용)
                            z_temp, _, _, _ = expert(z_base, z_base, return_aux=True)

                        return z_temp.reshape(T_val, tpf, -1)
                    return extract_with_expert
                model.extract_feature = make_hook(T, q_embed_for_router)

            with torch.no_grad():
                try:
                    response = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list)
                finally:
                    model.extract_feature = original_extract

            pred = extract_answer(response)
            correct = is_correct(pred, gt_answer)
            cat_correct[cat] += int(correct)
            cat_total[cat] += 1
            all_results.append({"category": cat, "video": video_name, "question": question_text, "gt": gt_answer, "pred": pred, "response": response, "correct": correct})

            if (qi + 1) % 10 == 0:
                print(f"  [{qi+1}/{len(questions)}] {cat_correct[cat]}/{cat_total[cat]}")

    print("\n" + "=" * 60)
    total_c, total_t = 0, 0
    for cat in sorted(cat_correct.keys()):
        c, t = cat_correct[cat], cat_total[cat]
        print(f"  {cat:<25} {c:>3}/{t:<3} ({100*c/max(t,1):.1f}%)")
        total_c += c; total_t += t
    total_label = "TOTAL"
    print(f"  {total_label:<25} {total_c:>3}/{total_t:<3} ({100*total_c/max(total_t,1):.1f}%)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"categories": {c: {"correct": cat_correct[c], "total": cat_total[c]} for c in cat_correct}, "total_correct": total_c, "total": total_t, "details": all_results}, f, indent=2)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="datasets/worldbench")
    p.add_argument("--model_path", type=str, default="models/InternVL3-2B")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    evaluate(p.parse_args())
