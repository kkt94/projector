"""
Evaluate PLM (with optional temporal expert) on Cosmos-Reason1-Benchmark.

Usage:
  # Vanilla PLM-1B
  python scripts/eval_cosmos_reason1.py --data_dir datasets/cosmos_reason1

  # With temporal expert checkpoint
  python scripts/eval_cosmos_reason1.py --data_dir datasets/cosmos_reason1 \
      --checkpoint_dir outputs/dbrouter_v928v2_s2/ckpt_step_500

  # Specify subsets
  python scripts/eval_cosmos_reason1.py --data_dir datasets/cosmos_reason1 \
      --subsets robovqa robofail
"""

import argparse
import json
import os
import re
import sys
import tarfile
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import decord
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

SUBSETS = ["robovqa", "robofail", "bridgev2", "agibot", "holoassist"]


def load_subset(data_dir, subset_name):
    """Load QA pairs and resolve video paths for a subset."""
    subset_dir = os.path.join(data_dir, subset_name)
    qa_file = os.path.join(subset_dir, f"{subset_name}_benchmark_qa_pairs.json")
    if not os.path.exists(qa_file):
        print(f"[WARN] QA file not found: {qa_file}")
        return []

    # Extract clips.tar.gz if needed
    clips_tar = os.path.join(subset_dir, "clips.tar.gz")
    clips_dir = os.path.join(subset_dir, "clips")
    if os.path.exists(clips_tar) and not os.path.exists(clips_dir):
        print(f"  Extracting {clips_tar}...")
        with tarfile.open(clips_tar, "r:gz") as tar:
            tar.extractall(subset_dir)

    with open(qa_file) as f:
        samples = json.load(f)

    # Resolve video paths
    resolved = []
    for sample in samples:
        video_path = os.path.join(subset_dir, sample["video"])
        if not os.path.exists(video_path):
            # Try without "clips/" prefix
            alt = os.path.join(subset_dir, os.path.basename(sample["video"]))
            if os.path.exists(alt):
                video_path = alt
            else:
                continue
        resolved.append({
            "video": video_path,
            "subset": subset_name,
            **sample["qa_pairs"],
        })
    return resolved


def load_video_frames(video_path, num_frames=32):
    """Load frames from video using decord."""
    if not HAS_DECORD:
        return None
    try:
        vr = decord.VideoReader(video_path, num_threads=1)
        total = len(vr)
        if total == 0:
            return None
        if total < num_frames:
            indices = list(range(total))
            while len(indices) < num_frames:
                indices.append(indices[-1])
        else:
            indices = [int(i * total / num_frames) for i in range(num_frames)]
        frames = vr.get_batch(indices)
        return [frames[i].numpy() if isinstance(frames, torch.Tensor) else frames[i].asnumpy()
                for i in range(len(indices))]
    except Exception as e:
        print(f"[WARN] Failed to load {video_path}: {e}")
        return None


def format_prompt(question, options):
    """Format MCQ prompt."""
    option_text = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
    return f"{question}\n{option_text}\nOnly give the best option."


def extract_answer_letter(response, options):
    """Extract option letter from model response."""
    response = response.strip()

    # Direct single letter
    if response.upper() in options:
        return response.upper()

    # "A)", "A.", "A ", "(A)", "(A) text"
    m = re.match(r"^\(?([A-Za-z])\)?[\).\s:,-]", response)
    if m and m.group(1).upper() in options:
        return m.group(1).upper()

    # Search anywhere for "(A)" pattern
    for letter in options:
        if re.search(rf"\({letter}\)", response, re.IGNORECASE):
            return letter

    # Exact option text match
    resp_lower = response.lower().strip().rstrip(".")
    for letter, text in options.items():
        if resp_lower == text.lower().strip().rstrip("."):
            return letter

    # Substring match (best Jaccard, >= 0.5 for short options)
    resp_words = set(resp_lower.split())
    best_letter, best_score = None, 0.0
    for letter, text in options.items():
        opt_words = set(text.lower().split())
        if not opt_words:
            continue
        jaccard = len(resp_words & opt_words) / len(resp_words | opt_words)
        if jaccard > best_score:
            best_score = jaccard
            best_letter = letter
    if best_score >= 0.5:
        return best_letter

    return None


VIDEO_TOKEN_ID = 128003
PAD_ID = 128001


def _generate_with_adapter(model, processor, inputs, design, adapter_objs, adapter_cfg, device, num_frames):
    """Generate with temporal expert applied to video features."""
    import torch.nn.functional as F

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values_videos = inputs.get("pixel_values_videos")
    if pixel_values_videos is None:
        return model.generate(**inputs, max_new_tokens=32, do_sample=False)

    B = input_ids.shape[0]
    T = num_frames

    # Everything in bf16 autocast
    dtype = torch.bfloat16

    # Vision encoder
    frames_flat = pixel_values_videos.flatten(0, 1)
    vision_out = model.model.vision_tower(frames_flat)
    hidden = vision_out.last_hidden_state[:, 1:, :].to(dtype)

    # Projector
    proj = model.model.multi_modal_projector
    features = hidden.permute(1, 0, 2)
    features = proj.linear_1(features.to(dtype))
    features = proj.gelu(features)
    features = proj.linear_2(features)
    features = features.permute(1, 0, 2)
    z_base_flat = proj.pooling(features)
    M, D = z_base_flat.shape[1], z_base_flat.shape[2]
    z_base_4d = z_base_flat.reshape(B, T, M, D)
    z_pre_pool_4d = z_base_4d  # training uses pooled for both

    # Expert + router/calibrator
    expert = adapter_objs[0]
    if design in ("diff_delta_expert", "lora_sft_expert"):
        expert_out = expert(z_pre_pool_4d, z_base_4d)
        z_temp = expert_out[0] if isinstance(expert_out, tuple) else expert_out
        projected = z_temp.reshape(B * T, M, D)
    elif design == "dbrouter":
        router = adapter_objs[1]
        _, _, _, aux = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
        local_res = aux["local_residual"]
        state_res = aux["state_residual"]
        # Question embedding
        all_embeds = model.model.get_input_embeddings()(input_ids)
        text_mask = (input_ids != VIDEO_TOKEN_ID) & (input_ids != PAD_ID)
        q_embed = (all_embeds * text_mask.unsqueeze(-1).float()).sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        # Stats
        base_pool = z_base_4d.detach().mean(dim=2)
        base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1) if T > 1 else torch.zeros(B, device=device)
        local_mag = local_res.detach().norm(dim=-1).mean(dim=(1, 2))
        state_mag = state_res.detach().norm(dim=-1).mean(dim=(1, 2))
        total_mag = (local_res + state_res).detach().norm(dim=-1).mean(dim=(1, 2))
        stats = torch.log1p(torch.stack([base_motion, local_mag, state_mag, total_mag], dim=-1).float()).to(dtype=z_base_4d.dtype)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w = router(q_embed, video_stats=stats)
        w_local = w[:, 0].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
        w_state = w[:, 1].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
        z_final = z_base_4d + w_local * local_res + w_state * state_res
        projected = z_final.reshape(B * T, M, D)
    else:
        projected = z_base_flat  # fallback

    # Build inputs_embeds with adapter features
    inputs_embeds = model.model.get_input_embeddings()(input_ids)
    projected_flat = projected.reshape(B, -1, D).to(inputs_embeds.dtype)
    _, special_video_mask = model.model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, video_features=projected_flat)
    inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, projected_flat)

    # Generate from inputs_embeds
    output_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=32,
        do_sample=False,
    )
    return output_ids


def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}")
    num_frames = args.num_frames

    # Load model
    print("Loading PLM-1B...")
    processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B", local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        "facebook/Perception-LM-1B", torch_dtype=torch.bfloat16, device_map={"": device}, local_files_only=True
    )
    model.eval()

    # Load temporal expert if checkpoint specified
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

    # Load data
    subsets_to_eval = args.subsets or SUBSETS
    all_samples = []
    for subset in subsets_to_eval:
        samples = load_subset(args.data_dir, subset)
        print(f"  {subset}: {len(samples)} samples")
        all_samples.extend(samples)

    if not all_samples:
        print("No samples found!")
        return

    print(f"\nTotal: {len(all_samples)} samples")

    # Evaluate
    results = {}
    subset_correct = {}
    subset_total = {}

    for i, sample in enumerate(all_samples):
        subset = sample["subset"]
        if subset not in subset_correct:
            subset_correct[subset] = 0
            subset_total[subset] = 0

        frames = load_video_frames(sample["video"], num_frames)
        if frames is None:
            print(f"[WARN] Skipping {sample['video']} — cannot load frames")
            continue

        question = sample["question"]
        options = sample["index2ans"]
        gt_answer = sample["answer"]
        prompt_text = format_prompt(question, options)

        # Build input
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": prompt_text}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, videos=[frames], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate — with adapter if loaded
        with torch.inference_mode():
            if adapter_design and adapter_design in ("dbrouter", "diff_delta_expert", "lora_sft_expert", "udh_v2_res2"):
                # Manual forward: vision → projector → expert/router → LLM generate
                output_ids = _generate_with_adapter(
                    model, processor, inputs, adapter_design, adapter_objs, adapter_cfg, device, num_frames)
            else:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        if output_ids.shape[1] > input_len:
            response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        else:
            # inputs_embeds generate may return only new tokens
            response = processor.decode(output_ids[0], skip_special_tokens=True).strip()

        # Score
        pred_letter = extract_answer_letter(response, options)
        is_correct = (pred_letter == gt_answer)
        subset_correct[subset] += int(is_correct)
        subset_total[subset] += 1

        if i < 5 or pred_letter is None:
            print(f"    #{i} gt={gt_answer} pred={pred_letter!r} resp={response!r}")

        if (i + 1) % 10 == 0 or i == len(all_samples) - 1:
            total_c = sum(subset_correct.values())
            total_t = sum(subset_total.values())
            print(f"  [{i+1}/{len(all_samples)}] {total_c}/{total_t} ({100*total_c/max(total_t,1):.1f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("Cosmos-Reason1-Benchmark Results")
    print("=" * 60)
    total_correct_all = 0
    total_all = 0
    for subset in sorted(subset_correct.keys()):
        c = subset_correct[subset]
        t = subset_total[subset]
        pct = 100 * c / max(t, 1)
        print(f"  {subset:<15} {c:>4}/{t:<4} ({pct:.1f}%)")
        total_correct_all += c
        total_all += t
    print(f"  {'TOTAL':<15} {total_correct_all:>4}/{total_all:<4} ({100*total_correct_all/max(total_all,1):.1f}%)")

    # Save results
    if args.output:
        results = {
            "subsets": {s: {"correct": subset_correct[s], "total": subset_total[s]}
                       for s in subset_correct},
            "total_correct": total_correct_all,
            "total": total_all,
            "accuracy": total_correct_all / max(total_all, 1),
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate on Cosmos-Reason1-Benchmark")
    p.add_argument("--data_dir", type=str, default="datasets/cosmos_reason1")
    p.add_argument("--checkpoint_dir", type=str, default=None, help="Temporal expert checkpoint (optional)")
    p.add_argument("--subsets", nargs="*", default=None, help="Subsets to evaluate (default: all available)")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="Save results JSON")
    evaluate(p.parse_args())
