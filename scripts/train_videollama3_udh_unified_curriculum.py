"""
Unified UDH v2 curriculum training on top of VideoLLaMA3-2B.

This keeps one shared train pool and replaces the hard stage1/stage2 split
with three phases inside one run:

  Phase A: expert shaping
    - train expert + auxiliary heads
    - optimize QA + stage1 auxiliary losses

  Phase B: router burn-in
    - freeze expert
    - train router only on QA-mixed output + QA-derived router targets

  Phase C: partial joint refine
    - train router + a small subset of expert readout/gate params
    - keep weak stage1 auxiliary losses active to preserve branch separation

Unlike the Perception-LM version, this script uses:
  - VideoLLaMA3-2B as the frozen base VLM
  - fixed-resolution video preprocessing for stable per-frame token counts
  - manual <image> token packing so the UDH expert can control projected tokens
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vision_projector.dual_proj import DualBranchRouter, GatedAttentionDualRouter, SplitDualHeadExpert
from train_dual_proj_v9_2_6_stage1 import TRAIN_DATASETS as FULL_TRAIN_DATASETS
from train_udh_v2_stage1 import TRAIN_DATASETS as UDH_CAPPED_TRAIN_DATASETS
from train_udh_v2_stage1 import log_rank0

try:
    import decord

    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


IMAGE_TOKEN = "<image>"
# Qwen2 chat template special tokens
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

# ---------------------------------------------------------------------------
# MVBench constants – mapping subtask → (sub_root, video_folder, ext_remap)
# sub_root is relative to MVBENCH_BASE; ext_remap is (old, new) or None
# ---------------------------------------------------------------------------
MVBENCH_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "MVBench")
MVBENCH_JSON_DIR = os.path.join(MVBENCH_BASE, "json")

MVBENCH_SUBTASK_PATHS = {
    "object_interaction": ("video", "star/Charades_v1_480", None),
    "action_sequence": ("video", "star/Charades_v1_480", None),
    "action_prediction": ("video", "star/Charades_v1_480", None),
    "action_localization": ("video", "sta/sta_video", None),
    "moving_count": ("video", "clevrer/video_validation", None),
    "fine_grained_pose": ("", "nturgbd_convert", (".avi", ".mp4")),
    "character_order": ("", "perception/videos", None),
    "object_shuffle": ("", "perception/videos", None),
    "egocentric_navigation": ("video", "vlnqa", None),
    "moving_direction": ("video", "clevrer/video_validation", None),
    "episodic_reasoning": ("video", "tvqa/frames_fps3_hq", None),
    "fine_grained_action": ("video", "Moments_in_Time_Raw/videos", None),
    "scene_transition": ("video", "scene_qa/video", None),
    "state_change": ("", "perception/videos", None),
    "moving_attribute": ("video", "clevrer/video_validation", None),
    "action_antonym": ("video", "ssv2_video", None),
    "unexpected_action": ("video", "FunQA_test/test", None),
    "counterfactual_inference": ("video", "clevrer/video_validation", None),
    "object_existence": ("video", "clevrer/video_validation", None),
    "action_count": ("", "perception/videos", None),
}


class VideoLLaMA3FixedVideoQADataset(Dataset):
    def __init__(
        self,
        data_path,
        video_root,
        tokenizer,
        num_frames=16,
        max_length=0,
        max_samples=0,
        image_size=384,
        patch_size=14,
        merge_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    ):
        self.video_root = video_root
        self.tokenizer = tokenizer
        self.num_frames = int(num_frames)
        self.max_length = int(max_length)
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.factor = self.patch_size * self.merge_size
        self.target_size = int(round(float(image_size) / float(self.factor)) * self.factor)
        self.hm = self.target_size // self.patch_size // self.merge_size
        self.wm = self.hm
        self.tokens_per_frame = self.hm * self.wm
        self.num_visual_tokens = self.num_frames * self.tokens_per_frame
        self.image_prefix = IMAGE_TOKEN * self.num_visual_tokens
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.samples = []

        if self.max_length > 0 and self.max_length <= self.num_visual_tokens:
            raise ValueError(
                f"max_length={self.max_length} is too small for "
                f"{self.num_visual_tokens} visual tokens"
            )

        with open(data_path) as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                vid = item.get("video") or item.get("video_path")
                if not vid:
                    continue
                convs = item.get("conversations", [])
                if len(convs) < 2:
                    continue
                q = convs[0].get("value", "")
                a = convs[1].get("value", "")
                if not q or not a:
                    continue
                full = os.path.join(video_root, vid)
                if os.path.exists(full):
                    self.samples.append({"video": full, "question": q, "answer": a})
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, path):
        if HAS_DECORD:
            try:
                vr = decord.VideoReader(path, num_threads=1)
                total = len(vr)
                if total == 0:
                    return None
                if total < self.num_frames:
                    indices = list(range(total))
                    while len(indices) < self.num_frames:
                        indices.append(indices[-1])
                else:
                    indices = [int(i * total / self.num_frames) for i in range(self.num_frames)]
                frames = vr.get_batch(indices)
                if isinstance(frames, torch.Tensor):
                    frames = frames.cpu().numpy()
                else:
                    frames = np.stack([frames[i].asnumpy() for i in range(len(indices))], axis=0)
                return [frames[i] for i in range(len(indices))]
            except Exception:
                return None
        return None

    def _preprocess_frames(self, frames):
        arr = np.stack(frames, axis=0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        if tensor.shape[-2] != self.target_size or tensor.shape[-1] != self.target_size:
            tensor = F.interpolate(
                tensor,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        tensor = (tensor - self.image_mean) / self.image_std
        return tensor

    def _build_text_tensors(self, question, answer):
        q_clean = re.sub(r"<video>\s*", "", question).strip()
        user_content = self.image_prefix + "\n" + q_clean
        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).squeeze(0)
        input_ids = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            return_tensors="pt",
        ).squeeze(0)

        if self.max_length > 0 and input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]

        prompt_len = min(prompt_ids.shape[0], input_ids.shape[0])
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        prompt_mask[:prompt_len] = True
        return input_ids, attention_mask, labels, prompt_mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        frames = None
        for attempt in range(3):
            target = item if attempt == 0 else self.samples[random.randint(0, len(self.samples) - 1)]
            frames = self._load_frames(target["video"])
            if frames is not None:
                item = target
                break
        if frames is None:
            frames = [np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)] * self.num_frames

        pixel_values = self._preprocess_frames(frames)
        input_ids, attention_mask, labels, prompt_mask = self._build_text_tensors(
            item["question"], item["answer"]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_mask": prompt_mask,
            "pixel_values": pixel_values,
        }


class MVBenchVideoQADataset(VideoLLaMA3FixedVideoQADataset):
    """Load MVBench evaluation data (20 subtasks x 200 MCQ) for diagnostic training.

    Inherits frame loading / preprocessing / text tensor building from the base
    class.  Only overrides sample construction: reads the MVBench JSON files,
    formats each MCQ as a conversation (user question with lettered options →
    assistant answers the correct letter), and resolves the per-subtask video
    paths.
    """

    OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]

    def __init__(
        self,
        tokenizer,
        num_frames=16,
        max_length=0,
        image_size=384,
        patch_size=14,
        merge_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        mvbench_base=None,
        subtasks=None,
    ):
        # Initialise base class fields without loading a JSONL
        self.tokenizer = tokenizer
        self.num_frames = int(num_frames)
        self.max_length = int(max_length)
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.factor = self.patch_size * self.merge_size
        self.target_size = int(round(float(image_size) / float(self.factor)) * self.factor)
        self.hm = self.target_size // self.patch_size // self.merge_size
        self.wm = self.hm
        self.tokens_per_frame = self.hm * self.wm
        self.num_visual_tokens = self.num_frames * self.tokens_per_frame
        self.image_prefix = IMAGE_TOKEN * self.num_visual_tokens
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.video_root = ""  # unused; paths are absolute
        self.samples = []

        base = mvbench_base or MVBENCH_BASE
        json_dir = os.path.join(base, "json")
        tasks = subtasks or list(MVBENCH_SUBTASK_PATHS.keys())

        skipped = 0
        for subtask in tasks:
            json_path = os.path.join(json_dir, f"{subtask}.json")
            if not os.path.exists(json_path):
                print(f"[MVBench] WARNING: {json_path} not found, skipping")
                continue
            sub_root, folder, ext_remap = MVBENCH_SUBTASK_PATHS[subtask]
            video_dir = os.path.join(base, sub_root, folder) if sub_root else os.path.join(base, folder)

            with open(json_path) as f:
                items = json.load(f)

            for item in items:
                vid = item["video"]
                if ext_remap:
                    vid = vid.replace(ext_remap[0], ext_remap[1])

                video_path = os.path.join(video_dir, vid)
                is_frame_dir = os.path.isdir(video_path)
                if not os.path.isfile(video_path) and not is_frame_dir:
                    # try data0613 fallback (star/clevrer)
                    alt = os.path.join(base, "video", "data0613", folder, vid)
                    if os.path.isfile(alt):
                        video_path = alt
                    elif os.path.isdir(alt):
                        video_path = alt
                        is_frame_dir = True
                    else:
                        skipped += 1
                        continue

                # Format MCQ as lettered options
                question = item["question"]
                candidates = item.get("candidates", [])
                answer_text = item["answer"]

                options_str = "\n".join(
                    f"{self.OPTION_LETTERS[i]}. {c}" for i, c in enumerate(candidates)
                )
                full_question = f"{question}\n{options_str}"

                # Determine correct letter
                answer_letter = None
                for i, c in enumerate(candidates):
                    if c == answer_text:
                        answer_letter = self.OPTION_LETTERS[i]
                        break
                if answer_letter is None:
                    # Fallback: substring match
                    for i, c in enumerate(candidates):
                        if answer_text in c or c in answer_text:
                            answer_letter = self.OPTION_LETTERS[i]
                            break
                if answer_letter is None:
                    answer_letter = "A"  # should not happen

                self.samples.append({
                    "video": video_path,
                    "question": full_question,
                    "answer": answer_letter,
                    "subtask": subtask,
                    "is_frame_dir": is_frame_dir,
                })

        print(f"[MVBench] Loaded {len(self.samples)} samples ({skipped} skipped due to missing videos)")

    def _load_frames(self, path):
        """Override to handle frame directories (e.g. episodic_reasoning)."""
        if os.path.isdir(path):
            return self._load_frames_from_dir(path)
        return super()._load_frames(path)

    def _load_frames_from_dir(self, frame_dir):
        """Load frames from a directory of images (jpg/png)."""
        try:
            from PIL import Image
            exts = (".jpg", ".jpeg", ".png", ".bmp")
            files = sorted(
                f for f in os.listdir(frame_dir)
                if f.lower().endswith(exts)
            )
            if not files:
                return None
            total = len(files)
            if total < self.num_frames:
                indices = list(range(total))
                while len(indices) < self.num_frames:
                    indices.append(indices[-1])
            else:
                indices = [int(i * total / self.num_frames) for i in range(self.num_frames)]
            frames = []
            for idx in indices:
                img = Image.open(os.path.join(frame_dir, files[idx])).convert("RGB")
                frames.append(np.array(img))
            return frames
        except Exception:
            return None


class VideoOnlyDataset(Dataset):
    """Video-only dataset for Stage 1 aux-loss training. No text needed."""

    def __init__(self, data_path, num_frames=32, image_size=392, patch_size=14, merge_size=2):
        super().__init__()
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        factor = patch_size * merge_size
        self.target_size = round(image_size / factor) * factor
        self.image_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
        self.samples = []
        with open(data_path) as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                vid = item.get("video")
                if vid and os.path.exists(vid):
                    self.samples.append(vid)
        print(f"[VideoOnly] Loaded {len(self.samples)} videos from {data_path}")

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, path):
        if HAS_DECORD:
            try:
                vr = decord.VideoReader(path, num_threads=1)
                total = len(vr)
                if total == 0:
                    return None
                if total < self.num_frames:
                    indices = list(range(total))
                    while len(indices) < self.num_frames:
                        indices.append(indices[-1])
                else:
                    indices = [int(i * total / self.num_frames) for i in range(self.num_frames)]
                frames = vr.get_batch(indices)
                if isinstance(frames, torch.Tensor):
                    frames = frames.cpu().numpy()
                else:
                    frames = np.stack([frames[i].asnumpy() for i in range(len(indices))], axis=0)
                from PIL import Image as PILImage
                return [PILImage.fromarray(f) for f in frames]
            except Exception:
                return None
        return None

    def __getitem__(self, idx):
        from PIL import Image
        for attempt in range(3):
            target = self.samples[idx] if attempt == 0 else self.samples[random.randint(0, len(self.samples) - 1)]
            frames = self._load_frames(target)
            if frames is not None:
                break
        if frames is None:
            frames = [Image.new("RGB", (self.target_size, self.target_size)) for _ in range(self.num_frames)]
        # Resize, to tensor, normalize
        processed = []
        for f in frames:
            f = f.resize((self.target_size, self.target_size))
            t = torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255.0
            t = (t - self.image_mean) / self.image_std
            processed.append(t)
        return {"pixel_values": torch.stack(processed)}


def video_only_collate_fn(batch):
    return {"pixel_values": torch.stack([item["pixel_values"] for item in batch])}


def build_dataset(dataset_configs, tokenizer, num_frames, max_length, image_size, patch_size, merge_size):
    datasets = []
    total = 0
    for cfg in dataset_configs:
        data_path = cfg.get("jsonl") or cfg.get("json")
        if not data_path or not os.path.exists(data_path):
            continue
        ds = VideoLLaMA3FixedVideoQADataset(
            data_path=data_path,
            video_root=cfg["video_root"],
            tokenizer=tokenizer,
            num_frames=num_frames,
            max_length=max_length,
            max_samples=cfg.get("max_samples", 0),
            image_size=image_size,
            patch_size=patch_size,
            merge_size=merge_size,
        )
        datasets.append(ds)
        total += len(ds)
    combined = ConcatDataset(datasets)
    print(f"[Dataset] Combined: {total} total samples from {len(datasets)} sources")
    return combined


def make_collate_fn(pad_id):
    def collate_fn(batch):
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids, attention_mask, labels, prompt_mask = [], [], [], []
        for item in batch:
            cur_len = item["input_ids"].shape[0]
            pad_len = max_len - cur_len
            input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_id))
            attention_mask.append(F.pad(item["attention_mask"], (0, pad_len), value=0))
            labels.append(F.pad(item["labels"], (0, pad_len), value=-100))
            prompt_mask.append(F.pad(item["prompt_mask"].long(), (0, pad_len), value=0).bool())
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "prompt_mask": torch.stack(prompt_mask),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        }

    return collate_fn


class VideoLLaMA3UDHUnifiedCurriculumModel(nn.Module):
    def __init__(
        self,
        model,
        expert,
        router,
        image_token_id,
        pad_token_id,
        patch_size=14,
        merge_size=2,
        align_lambda=0.1,
        state_mask_lambda=0.03,
        state_mask_ratio=0.5,
        state_pred_dropout=0.05,
        local_corr_lambda=0.01,
        local_corr_radius=1,
        local_corr_anchors=128,
        local_corr_dropout=0.05,
        router_lambda=0.10,
        router_tau=0.5,
        router_margin=0.02,
        entropy_lambda=0.005,
        balance_lambda=0.01,
        balance_target=(0.5, 0.5),
        router_pattern_lambda=0.10,
        router_var_lambda=0.10,
        video_only=False,
    ):
        super().__init__()
        self.model = model
        self.expert = expert
        self.router = router
        self.video_only = bool(video_only)
        self.image_token_id = int(image_token_id)
        self.pad_token_id = int(pad_token_id)
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        # Pre-compute grid dims for native QA loss
        self._hp = 0  # set by first _build_stage1_inputs call
        self._wp = 0
        self._use_native_qa_loss = False  # set via set_native_qa_loss()
        # Qwen2 chat template special token IDs (for question embedding filtering)
        self._im_start_id = 151644
        self._im_end_id = 151645
        # Vision encoder chunk size: 0 = all at once (flash_attn), >0 = chunk (SDPA/eager)
        attn_impl = getattr(model.config, "_attn_implementation", "sdpa")
        self._vision_chunk_size = 0 if attn_impl == "flash_attention_2" else 4

        self.align_lambda = float(align_lambda)
        self.state_mask_lambda = float(state_mask_lambda)
        self.state_mask_ratio = float(state_mask_ratio)
        self.local_corr_lambda = float(local_corr_lambda)
        self.local_corr_radius = int(local_corr_radius)
        self.local_corr_anchors = int(local_corr_anchors)
        self.router_lambda = float(router_lambda)
        self.router_tau = float(router_tau)
        self.router_margin = float(router_margin)
        self.entropy_lambda = float(entropy_lambda)
        self.balance_lambda = float(balance_lambda)
        self.router_pattern_lambda = float(router_pattern_lambda)
        self.router_var_lambda = float(router_var_lambda)
        self.register_buffer(
            "balance_target",
            torch.as_tensor(balance_target, dtype=torch.float32),
            persistent=False,
        )

        proj_dim = expert.proj_dim
        bottleneck = expert.bottleneck
        self.state_pred_dropout = nn.Dropout(state_pred_dropout) if state_pred_dropout > 0 else nn.Identity()
        self.local_corr_drop = nn.Dropout(local_corr_dropout) if local_corr_dropout > 0 else nn.Identity()

        self.state_bridge_head = nn.Sequential(
            nn.LayerNorm(bottleneck * 2 + 1),
            nn.Linear(bottleneck * 2 + 1, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, proj_dim),
        )
        nn.init.xavier_uniform_(self.state_bridge_head[1].weight, gain=0.1)
        nn.init.zeros_(self.state_bridge_head[1].bias)
        nn.init.xavier_uniform_(self.state_bridge_head[3].weight, gain=0.1)
        nn.init.zeros_(self.state_bridge_head[3].bias)

        self.local_corr_q = nn.Linear(bottleneck, bottleneck)
        self.local_corr_k = nn.Linear(proj_dim, bottleneck)
        nn.init.xavier_uniform_(self.local_corr_q.weight, gain=0.1)
        nn.init.zeros_(self.local_corr_q.bias)
        nn.init.xavier_uniform_(self.local_corr_k.weight, gain=0.1)
        nn.init.zeros_(self.local_corr_k.bias)

        self._last_phase = "A"
        self._last_total = 0.0
        self._last_qa = 0.0
        self._last_align = 0.0
        self._last_state_mask = 0.0
        self._last_state_acc = 0.0
        self._last_local_corr = 0.0
        self._last_local_acc = 0.0
        self._last_router = 0.0
        self._last_entropy = 0.0
        self._last_balance = 0.0
        self._last_w_l = 0.0
        self._last_w_s = 0.0
        self._last_w_l_std = 0.0
        self._last_w_s_std = 0.0
        self._last_tgt_l = 0.0
        self._last_tgt_s = 0.0
        self._last_tgt_l_std = 0.0
        self._last_tgt_s_std = 0.0
        self._last_qa_base = 0.0
        self._last_qa_local = 0.0
        self._last_qa_state = 0.0
        self._last_router_pattern = 0.0
        self._last_router_var = 0.0
        self._last_grad = 0.0

    @property
    def aux_params(self):
        params = list(self.state_bridge_head.parameters())
        params.extend(self.local_corr_q.parameters())
        params.extend(self.local_corr_k.parameters())
        return params

    def _build_stage1_inputs(self, pixel_values, chunk_size=0):
        """Encode frames through vision encoder, optionally in chunks.

        With flash_attention_2: flash_attn_varlen_func handles cu_seqlens natively
        with O(N) memory, so chunking is unnecessary (chunk_size=0 → all at once).
        With SDPA/eager: creates O(N²) attention mask → must chunk to avoid OOM.
        Set chunk_size=0 to disable chunking (flash_attn), or >0 for SDPA/eager.
        """
        bsz, t, _, h, w = pixel_values.shape
        hp = h // self.patch_size
        wp = w // self.patch_size
        hm = hp // self.merge_size
        wm = wp // self.merge_size
        patches_per_frame = hp * wp
        tokens_per_frame = hm * wm
        self._hp = hp
        self._wp = wp

        n_total = bsz * t
        effective_chunk = chunk_size if chunk_size > 0 else n_total

        all_projected = []
        with torch.no_grad():
            flat_frames = pixel_values.reshape(n_total, 3, h, w)
            for ci in range(0, n_total, effective_chunk):
                end = min(ci + effective_chunk, n_total)
                n_chunk = end - ci
                chunk_frames = flat_frames[ci:end]
                patches = chunk_frames.unfold(2, self.patch_size, self.patch_size).unfold(
                    3, self.patch_size, self.patch_size
                )
                patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
                patches = patches.reshape(n_chunk * patches_per_frame, 3 * self.patch_size * self.patch_size)
                grid_sizes = torch.tensor(
                    [[1, hp, wp]] * n_chunk, device=pixel_values.device, dtype=torch.long
                )
                msizes = torch.tensor(
                    [self.merge_size] * n_chunk, device=pixel_values.device, dtype=torch.long
                )
                merged = self.model.model.vision_encoder(patches, grid_sizes, msizes)
                projected = self.model.model.mm_projector(merged)
                all_projected.append(projected.reshape(n_chunk, tokens_per_frame, -1))

        z = torch.cat(all_projected, dim=0).reshape(bsz, t, tokens_per_frame, -1)
        return z, z

    def _run_expert(self, z_pre_pool, z_base, use_grad):
        if use_grad:
            return self.expert(z_pre_pool, z_base, return_aux=True)
        with torch.no_grad():
            return self.expert(z_pre_pool, z_base, return_aux=True)

    def _qa_loss(self, projected_flat, input_ids, attention_mask, labels, chunk_seq=1024):
        """QA loss with chunked CE to avoid materializing full (seq × vocab) logits."""
        inputs_embeds = self.model.get_model().embed_tokens(input_ids).detach()
        projected_flat = projected_flat.to(inputs_embeds.dtype)
        image_mask = input_ids.eq(self.image_token_id)
        expected = image_mask.sum(dim=1)
        if not torch.all(expected == projected_flat.shape[1]):
            raise ValueError(
                f"Image token count mismatch: expected {expected.tolist()}, "
                f"got {projected_flat.shape[1]}"
            )
        inputs_embeds[image_mask] = projected_flat.reshape(-1, projected_flat.shape[-1])
        # Get hidden states without computing logits (labels=None, no lm_head)
        outputs = self.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        # Chunked CE: apply lm_head in small chunks to avoid OOM on (seq × vocab)
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        qa_loss = torch.tensor(0.0, device=input_ids.device)
        total_tokens = 0
        for i in range(0, shift_hidden.shape[1], chunk_seq):
            h_chunk = shift_hidden[:, i : i + chunk_seq, :]
            l_chunk = shift_labels[:, i : i + chunk_seq]
            n_valid = (l_chunk != -100).sum().item()
            if n_valid > 0:
                logits_chunk = self.model.lm_head(h_chunk)
                chunk_loss = F.cross_entropy(
                    logits_chunk.view(-1, logits_chunk.size(-1)),
                    l_chunk.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                qa_loss = qa_loss + chunk_loss
                total_tokens += n_valid
        if total_tokens > 0:
            qa_loss = qa_loss / total_tokens
        return qa_loss.float()

    def _patchify_pixels(self, pixel_values):
        """Convert (B, T, 3, H, W) raw frames to (B*T*hp*wp, patch_dim) patches.

        This matches the format that VL3's processor outputs and that
        prepare_inputs_labels_for_multimodal / _get_compression_mask expect.
        """
        bsz, t, c, h, w = pixel_values.shape
        flat = pixel_values.reshape(bsz * t, c, h, w)
        # unfold into (N, hp, wp, C, ps, ps)
        patches = flat.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        n_total = bsz * t
        patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        patch_dim = c * self.patch_size * self.patch_size
        return patches.reshape(n_total * patches_per_frame, patch_dim)

    def _qa_loss_native(self, z_temp_4d, pixel_values, input_ids, attention_mask, labels, chunk_seq=1024):
        """QA loss using VL3's native multimodal preprocessing plus chunked CE.

        This keeps the eval-time native path:
          encode_images -> prepare_inputs_labels_for_multimodal ->
          token compression / truncation -> Qwen2 forward
        while still avoiding a full (seq x vocab) logits allocation.
        """
        bsz, t, tpf, dim = z_temp_4d.shape
        z_flat = z_temp_4d.reshape(-1, dim)  # (bsz*t*tpf, dim)

        # Convert raw frames to patchified format for compression mask
        pv_patches = self._patchify_pixels(pixel_values)

        original_encode = self.model.encode_images

        def encode_with_expert(pv, gs, ms):
            # Return pre-computed expert features instead of encoding from pixels
            return z_flat.to(pv.dtype)

        self.model.encode_images = encode_with_expert
        try:
            (
                mm_input_ids,
                mm_attention_mask,
                mm_position_ids,
                _,
                mm_inputs_embeds,
                mm_labels,
            ) = self.model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pv_patches,
                grid_sizes=torch.tensor(
                    [[t, self._hp, self._wp]] * bsz,
                    device=pixel_values.device, dtype=torch.long,
                ),
                merge_sizes=torch.tensor(
                    [self.merge_size] * bsz,
                    device=pixel_values.device, dtype=torch.long,
                ),
                modals=["video"] * bsz,
            )
        finally:
            self.model.encode_images = original_encode

        outputs = self.model.model(
            input_ids=mm_input_ids,
            attention_mask=mm_attention_mask,
            position_ids=mm_position_ids,
            inputs_embeds=mm_inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = mm_labels[:, 1:].contiguous()
        qa_loss = torch.tensor(0.0, device=input_ids.device)
        total_tokens = 0
        for i in range(0, shift_hidden.shape[1], chunk_seq):
            h_chunk = shift_hidden[:, i : i + chunk_seq, :]
            l_chunk = shift_labels[:, i : i + chunk_seq]
            n_valid = (l_chunk != -100).sum().item()
            if n_valid > 0:
                logits_chunk = self.model.lm_head(h_chunk)
                chunk_loss = F.cross_entropy(
                    logits_chunk.view(-1, logits_chunk.size(-1)),
                    l_chunk.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                qa_loss = qa_loss + chunk_loss
                total_tokens += n_valid
        if total_tokens > 0:
            qa_loss = qa_loss / total_tokens
        return qa_loss.float()

    def _qa_loss_per_sample(self, z_4d, input_ids, attention_mask, labels, chunk_seq=1024):
        bsz = z_4d.shape[0]
        z_flat = z_4d.reshape(bsz, -1, z_4d.shape[-1])
        with torch.inference_mode():
            inputs_embeds = self.model.get_model().embed_tokens(input_ids)
            projected = z_flat.to(inputs_embeds.dtype)
            image_mask = input_ids.eq(self.image_token_id)
            ie = inputs_embeds.clone()
            ie[image_mask] = projected.reshape(-1, projected.shape[-1])
            outputs = self.model.model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=ie,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            shift_hidden = hidden_states[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            losses = torch.zeros(bsz, device=input_ids.device)
            for i in range(bsz):
                sample_loss = 0.0
                n_valid = 0
                for ci in range(0, shift_hidden.shape[1], chunk_seq):
                    h_c = shift_hidden[i, ci : ci + chunk_seq, :]
                    l_c = shift_labels[i, ci : ci + chunk_seq]
                    nv = (l_c != -100).sum().item()
                    if nv > 0:
                        logits_c = self.model.lm_head(h_c.unsqueeze(0)).squeeze(0)
                        sample_loss = sample_loss + F.cross_entropy(
                            logits_c, l_c, ignore_index=-100, reduction="sum"
                        )
                        n_valid += nv
                if n_valid > 0:
                    losses[i] = sample_loss / n_valid
        return losses

    def _qa_loss_three_variants(self, z_base, z_local, z_state, input_ids, attention_mask, labels, chunk_seq=1024):
        """Compute per-sample QA losses for base/local/state in one batched LLM forward."""
        bsz = z_base.shape[0]
        z_b = z_base.reshape(bsz, -1, z_base.shape[-1])
        z_l = z_local.reshape(bsz, -1, z_local.shape[-1])
        z_s = z_state.reshape(bsz, -1, z_state.shape[-1])

        # concat 3 variants along batch dim → single LLM forward
        z_cat = torch.cat([z_b, z_l, z_s], dim=0)  # (3*bsz, seq, d)
        ids3 = input_ids.repeat(3, 1)
        mask3 = attention_mask.repeat(3, 1)
        lab3 = labels.repeat(3, 1)

        with torch.inference_mode():
            embeds = self.model.get_model().embed_tokens(ids3)
            proj = z_cat.to(embeds.dtype)
            img_mask = ids3.eq(self.image_token_id)
            ie = embeds.clone()
            ie[img_mask] = proj.reshape(-1, proj.shape[-1])
            out = self.model.model(
                input_ids=None,
                attention_mask=mask3,
                inputs_embeds=ie,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = out.last_hidden_state
            shift_hidden = hidden_states[:, :-1, :].contiguous()
            shift_labels = lab3[:, 1:].contiguous()
            losses = torch.zeros(3 * bsz, device=input_ids.device)
            for i in range(3 * bsz):
                sample_loss = 0.0
                n_valid = 0
                for ci in range(0, shift_hidden.shape[1], chunk_seq):
                    h_c = shift_hidden[i, ci : ci + chunk_seq, :]
                    l_c = shift_labels[i, ci : ci + chunk_seq]
                    nv = (l_c != -100).sum().item()
                    if nv > 0:
                        logits_c = self.model.lm_head(h_c.unsqueeze(0)).squeeze(0)
                        sample_loss = sample_loss + F.cross_entropy(
                            logits_c, l_c, ignore_index=-100, reduction="sum"
                        )
                        n_valid += nv
                if n_valid > 0:
                    losses[i] = sample_loss / n_valid

        return losses[:bsz], losses[bsz : 2 * bsz], losses[2 * bsz:]

    @staticmethod
    def _alignment_loss(z_final, z_base):
        return 1.0 - F.cosine_similarity(z_final.float(), z_base.float(), dim=-1).mean()

    @staticmethod
    def _cosine_aux_loss(pred, target):
        pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
        target = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
        return 1.0 - (pred * target).sum(dim=-1).mean()

    def _masked_state_loss(self, aux):
        state_h = aux["state_h"]
        state = aux["state"]
        t = state.shape[1]
        if t <= 2:
            return torch.tensor(0.0, device=state.device), 0.0

        n_mid = t - 2
        n_target = max(1, int(round(self.state_mask_ratio * n_mid)))
        target_idx = torch.randperm(n_mid, device=state.device)[:n_target] + 1
        target_idx, _ = torch.sort(target_idx)

        first_h = self.state_pred_dropout(state_h[:, :1]).expand(-1, n_target, -1)
        last_h = self.state_pred_dropout(state_h[:, -1:]).expand(-1, n_target, -1)
        alpha = (target_idx.float() / (t - 1)).view(1, n_target, 1).expand(state.shape[0], -1, -1)
        alpha = alpha.to(dtype=state.dtype)

        bridge_input = torch.cat([first_h, last_h, alpha], dim=-1)
        pred_state = self.state_bridge_head(bridge_input)
        target_state = state[:, target_idx]
        loss = self._cosine_aux_loss(pred_state, target_state)

        with torch.no_grad():
            pred_norm = F.normalize(pred_state.float(), dim=-1)
            all_states_norm = F.normalize(state.float(), dim=-1)
            sim = torch.bmm(pred_norm, all_states_norm.transpose(1, 2))
            nn_idx = sim.argmax(dim=-1)
            diff = (nn_idx - target_idx.unsqueeze(0)).abs()
            correct = (diff <= 2).float().mean().item()

        return loss, correct

    def _local_correspondence_loss(self, z_pre_pool, aux):
        if "local_h" not in aux:
            zero = torch.tensor(0.0, device=z_pre_pool.device)
            return zero, zero
        b, t, s_pre, d = z_pre_pool.shape
        if t <= 1:
            zero = torch.tensor(0.0, device=z_pre_pool.device)
            return zero, zero
        h = int(math.sqrt(s_pre))
        if h * h != s_pre:
            zero = torch.tensor(0.0, device=z_pre_pool.device)
            return zero, zero

        bt = b * (t - 1)
        prev_tokens = z_pre_pool[:, :-1].reshape(bt, s_pre, d).float()
        curr_tokens = z_pre_pool[:, 1:].reshape(bt, s_pre, d).float()
        local_h = aux["local_h"].reshape(bt, s_pre, self.expert.bottleneck)

        anchor_count = min(self.local_corr_anchors, s_pre)
        anchor_idx = torch.randperm(s_pre, device=z_pre_pool.device)[:anchor_count]

        prev_a = prev_tokens[:, anchor_idx]
        local_a = local_h[:, anchor_idx]

        radius = self.local_corr_radius
        ksz = 2 * radius + 1
        curr_grid = curr_tokens.reshape(bt, h, h, d).permute(0, 3, 1, 2)
        curr_grid = F.pad(curr_grid, (radius, radius, radius, radius), mode="replicate")
        neigh = F.unfold(curr_grid, kernel_size=ksz, padding=0)
        k2 = ksz * ksz
        neigh = neigh.transpose(1, 2).reshape(bt, s_pre, k2, d)
        curr_a = neigh[:, anchor_idx]

        prev_n = F.normalize(prev_a, dim=-1, eps=1e-6)
        curr_n = F.normalize(curr_a, dim=-1, eps=1e-6)
        pseudo_sim = (prev_n.unsqueeze(2) * curr_n).sum(dim=-1)
        target_idx = pseudo_sim.argmax(dim=-1)

        q = F.normalize(self.local_corr_q(self.local_corr_drop(local_a.float())), dim=-1, eps=1e-6)
        k = F.normalize(self.local_corr_k(curr_a.detach()), dim=-1, eps=1e-6)
        logits = (q.unsqueeze(2) * k).sum(dim=-1) / 0.07
        loss = F.cross_entropy(logits.reshape(-1, k2), target_idx.reshape(-1))
        acc = (logits.argmax(dim=-1) == target_idx).float().mean()
        return loss, acc

    def _get_question_embedding(self, input_ids, prompt_mask):
        with torch.no_grad():
            all_embeds = self.model.get_model().embed_tokens(input_ids)
        q_embeds = []
        for i in range(input_ids.shape[0]):
            text_mask = (
                prompt_mask[i]
                & input_ids[i].ne(self.image_token_id)
                & input_ids[i].ne(self.pad_token_id)
                & input_ids[i].ne(self._im_start_id)
                & input_ids[i].ne(self._im_end_id)
            )
            if text_mask.any():
                q_embeds.append(all_embeds[i][text_mask].mean(dim=0))
            else:
                q_embeds.append(
                    torch.zeros(
                        all_embeds.shape[-1],
                        device=all_embeds.device,
                        dtype=all_embeds.dtype,
                    )
                )
        return torch.stack(q_embeds)

    @staticmethod
    def _centered_match_loss(pred, target):
        pred = pred.float()
        target = target.detach().float()
        pred_c = pred - pred.mean()
        target_c = target - target.mean()
        return F.mse_loss(pred_c, target_c)

    def _compute_router_stats(self, z_base, local_res, state_res):
        base_pool = z_base.detach().mean(dim=2)
        if z_base.shape[1] > 1:
            base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
        else:
            base_motion = torch.zeros(z_base.shape[0], device=z_base.device, dtype=z_base.dtype)
        local_mag = local_res.detach().norm(dim=-1).mean(dim=(1, 2))
        state_mag = state_res.detach().norm(dim=-1).mean(dim=(1, 2))
        total_mag = (local_res + state_res).detach().norm(dim=-1).mean(dim=(1, 2))
        stats = torch.stack([base_motion, local_mag, state_mag, total_mag], dim=-1)
        return torch.log1p(stats.float()).to(device=z_base.device, dtype=z_base.dtype)

    def _router_mix(
        self,
        z_base,
        local_res,
        state_res,
        input_ids,
        attention_mask,
        labels,
        prompt_mask,
        router_loss_scale=1.0,
    ):
        q_embed = self._get_question_embedding(input_ids, prompt_mask)
        stats = self._compute_router_stats(z_base, local_res, state_res)
        v_frames = (z_base + local_res + state_res).mean(dim=2)
        weights, logits = self.router(q_embed, video_stats=stats, v_frames=v_frames, return_logits=True)
        w_local = weights[:, 0]
        w_state = weights[:, 1]
        logit_local = logits[:, 0]
        logit_state = logits[:, 1]

        z_final = z_base + w_local.view(-1, 1, 1, 1) * local_res + w_state.view(-1, 1, 1, 1) * state_res
        qa_loss = self._qa_loss(
            z_final.reshape(z_final.shape[0], -1, z_final.shape[-1]),
            input_ids,
            attention_mask,
            labels,
        )
        align_loss = self._alignment_loss(z_final, z_base)

        qa_base_ps, qa_local_ps, qa_state_ps = self._qa_loss_three_variants(
            z_base.detach(),
            (z_base + local_res).detach(),
            (z_base + state_res).detach(),
            input_ids, attention_mask, labels,
        )

        gain_local = qa_base_ps - qa_local_ps - self.router_margin
        gain_state = qa_base_ps - qa_state_ps - self.router_margin
        target_local = torch.sigmoid(gain_local / self.router_tau)
        target_state = torch.sigmoid(gain_state / self.router_tau)
        router_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(logit_local, target_local.detach())
            + F.binary_cross_entropy_with_logits(logit_state, target_state.detach())
        )
        pattern_loss = 0.5 * (
            self._centered_match_loss(w_local, target_local)
            + self._centered_match_loss(w_state, target_state)
        )
        var_loss = 0.5 * (
            F.mse_loss(
                w_local.float().std(unbiased=False),
                target_local.detach().float().std(unbiased=False),
            )
            + F.mse_loss(
                w_state.float().std(unbiased=False),
                target_state.detach().float().std(unbiased=False),
            )
        )
        entropy = -(
            w_local.clamp(min=1e-6) * w_local.clamp(min=1e-6).log()
            + w_state.clamp(min=1e-6) * w_state.clamp(min=1e-6).log()
        ).mean()
        balance = F.mse_loss(
            torch.stack([w_local.mean(), w_state.mean()]).float(),
            self.balance_target.to(w_local.device).float(),
        )

        total = (
            qa_loss
            + self.align_lambda * align_loss
            + (self.router_lambda * router_loss_scale) * router_loss
            + self.router_pattern_lambda * pattern_loss
            + self.router_var_lambda * var_loss
            - self.entropy_lambda * entropy
            + self.balance_lambda * balance
        )
        metrics = {
            "qa": qa_loss,
            "align": align_loss,
            "router": router_loss,
            "router_pattern": pattern_loss,
            "router_var": var_loss,
            "entropy": entropy,
            "balance": balance,
            "w_local": w_local,
            "w_state": w_state,
            "target_local": target_local,
            "target_state": target_state,
            "qa_base": qa_base_ps,
            "qa_local": qa_local_ps,
            "qa_state": qa_state_ps,
        }
        return total, metrics

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        prompt_mask,
        labels=None,
        phase="A",
        router_loss_scale=1.0,
        aux_loss_scale=1.0,
    ):
        self._last_phase = phase
        z_pre_pool, z_base = self._build_stage1_inputs(pixel_values, chunk_size=self._vision_chunk_size)

        if phase == "A":
            z_temp, _, _, aux = self._run_expert(z_pre_pool, z_base, use_grad=True)

            if self.video_only:
                qa_loss = torch.tensor(0.0, device=pixel_values.device)
                align_loss = torch.tensor(0.0, device=pixel_values.device)
            else:
                if self._use_native_qa_loss:
                    qa_loss = self._qa_loss_native(
                        z_temp, pixel_values, input_ids, attention_mask, labels,
                    )
                else:
                    qa_loss = self._qa_loss(
                        z_temp.reshape(z_temp.shape[0], -1, z_temp.shape[-1]),
                        input_ids,
                        attention_mask,
                        labels,
                    )
                align_loss = self._alignment_loss(z_temp, z_base)

            state_mask_loss, state_mask_acc = self._masked_state_loss(aux)
            local_corr_loss, local_corr_acc = self._local_correspondence_loss(z_pre_pool, aux)

            if self.video_only:
                total_loss = (
                    self.state_mask_lambda * state_mask_loss
                    + self.local_corr_lambda * local_corr_loss
                )
            else:
                total_loss = (
                    qa_loss
                    + self.align_lambda * align_loss
                    + self.state_mask_lambda * state_mask_loss
                    + self.local_corr_lambda * local_corr_loss
                )

            self._last_total = total_loss.item()
            self._last_qa = qa_loss.item()
            self._last_align = align_loss.item()
            self._last_state_mask = state_mask_loss.item()
            self._last_state_acc = state_mask_acc
            self._last_local_corr = local_corr_loss.item()
            self._last_local_acc = local_corr_acc.item()
            self._last_router = 0.0
            self._last_entropy = 0.0
            self._last_balance = 0.0
            self._last_w_l = 0.0
            self._last_w_s = 0.0
            self._last_w_l_std = 0.0
            self._last_w_s_std = 0.0
            self._last_tgt_l = 0.0
            self._last_tgt_s = 0.0
            self._last_tgt_l_std = 0.0
            self._last_tgt_s_std = 0.0
            self._last_qa_base = 0.0
            self._last_qa_local = 0.0
            self._last_qa_state = 0.0
            self._last_router_pattern = 0.0
            self._last_router_var = 0.0
            return total_loss

        use_grad = phase == "C"
        _, local_res, state_res, aux = self._run_expert(z_pre_pool, z_base, use_grad=use_grad)
        total_loss, metrics = self._router_mix(
            z_base,
            local_res,
            state_res,
            input_ids,
            attention_mask,
            labels,
            prompt_mask,
            router_loss_scale=router_loss_scale,
        )

        state_mask_loss = torch.tensor(0.0, device=z_base.device)
        local_corr_loss = torch.tensor(0.0, device=z_base.device)
        local_corr_acc = torch.tensor(0.0, device=z_base.device)
        if phase == "C" and aux_loss_scale > 0:
            state_mask_loss, state_mask_acc = self._masked_state_loss(aux)
            local_corr_loss, local_corr_acc = self._local_correspondence_loss(z_pre_pool, aux)
            total_loss = total_loss + aux_loss_scale * (
                self.state_mask_lambda * state_mask_loss
                + self.local_corr_lambda * local_corr_loss
            )

        self._last_total = total_loss.item()
        self._last_qa = metrics["qa"].item()
        self._last_align = metrics["align"].item()
        self._last_state_mask = state_mask_loss.item()
        self._last_local_corr = local_corr_loss.item()
        self._last_state_acc = state_mask_acc
        self._last_local_acc = local_corr_acc.item()
        self._last_router = metrics["router"].item()
        self._last_entropy = metrics["entropy"].item()
        self._last_balance = metrics["balance"].item()
        self._last_w_l = metrics["w_local"].detach().mean().item()
        self._last_w_s = metrics["w_state"].detach().mean().item()
        self._last_w_l_std = metrics["w_local"].detach().std(unbiased=False).item()
        self._last_w_s_std = metrics["w_state"].detach().std(unbiased=False).item()
        self._last_tgt_l = metrics["target_local"].mean().item()
        self._last_tgt_s = metrics["target_state"].mean().item()
        self._last_tgt_l_std = metrics["target_local"].detach().std(unbiased=False).item()
        self._last_tgt_s_std = metrics["target_state"].detach().std(unbiased=False).item()
        self._last_qa_base = metrics["qa_base"].mean().item()
        self._last_qa_local = metrics["qa_local"].mean().item()
        self._last_qa_state = metrics["qa_state"].mean().item()
        self._last_router_pattern = metrics["router_pattern"].item()
        self._last_router_var = metrics["router_var"].item()
        return total_loss


MVBENCH_TRAIN_DATASETS = [
    {"jsonl": "datasets/mvbench_train/action_antonym.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/ssv2_video"},
    {"jsonl": "datasets/mvbench_train/action_count.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/perception/videos"},
    {"jsonl": "datasets/mvbench_train/action_localization.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/sta/sta_video"},
    {"jsonl": "datasets/mvbench_train/action_prediction.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/star/Charades_v1_480"},
    {"jsonl": "datasets/mvbench_train/action_sequence.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/star/Charades_v1_480"},
    {"jsonl": "datasets/mvbench_train/character_order.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/perception/videos"},
    {"jsonl": "datasets/mvbench_train/counterfactual_inference.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/clevrer/video_validation"},
    {"jsonl": "datasets/mvbench_train/egocentric_navigation.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/vlnqa"},
    {"jsonl": "datasets/mvbench_train/fine_grained_action.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/Moments_in_Time_Raw/videos"},
    {"jsonl": "datasets/mvbench_train/moving_attribute.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/clevrer/video_validation"},
    {"jsonl": "datasets/mvbench_train/moving_count.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/clevrer/video_validation"},
    {"jsonl": "datasets/mvbench_train/moving_direction.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/clevrer/video_validation"},
    {"jsonl": "datasets/mvbench_train/object_existence.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/clevrer/video_validation"},
    {"jsonl": "datasets/mvbench_train/object_interaction.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/star/Charades_v1_480"},
    {"jsonl": "datasets/mvbench_train/object_shuffle.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/perception/videos"},
    {"jsonl": "datasets/mvbench_train/scene_transition.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/scene_qa/video"},
    {"jsonl": "datasets/mvbench_train/state_change.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/perception/videos"},
    {"jsonl": "datasets/mvbench_train/unexpected_action.jsonl", "video_root": "/root/.cache/huggingface/mvbench_video/FunQA_test/test"},
]


TEMPCOMPASS_TRAIN_DATASETS = [
    {"jsonl": "datasets/tempcompass_train/action.jsonl", "video_root": ""},
    {"jsonl": "datasets/tempcompass_train/attribute_change.jsonl", "video_root": ""},
    {"jsonl": "datasets/tempcompass_train/direction.jsonl", "video_root": ""},
    {"jsonl": "datasets/tempcompass_train/order.jsonl", "video_root": ""},
    {"jsonl": "datasets/tempcompass_train/speed.jsonl", "video_root": ""},
]

NEXTQA_EVAL_TRAIN_DATASETS = [
    {"jsonl": "datasets/nextqa_eval_train/nextqa_mc_test.jsonl", "video_root": "/root/.cache/huggingface/nextqa"},
]


def resolve_dataset_configs(dataset_preset):
    if dataset_preset == "full":
        return list(FULL_TRAIN_DATASETS)
    if dataset_preset == "udh_capped":
        return list(UDH_CAPPED_TRAIN_DATASETS)
    if dataset_preset == "full+mvbench":
        return list(FULL_TRAIN_DATASETS) + list(MVBENCH_TRAIN_DATASETS)
    if dataset_preset == "full+mvbench+tc":
        return list(FULL_TRAIN_DATASETS) + list(MVBENCH_TRAIN_DATASETS) + list(TEMPCOMPASS_TRAIN_DATASETS)
    if dataset_preset == "full+eval":
        return list(FULL_TRAIN_DATASETS) + list(MVBENCH_TRAIN_DATASETS) + list(TEMPCOMPASS_TRAIN_DATASETS) + list(NEXTQA_EVAL_TRAIN_DATASETS)
    if dataset_preset == "mvbench":
        return "mvbench"  # sentinel – handled by build_dataset caller
    if dataset_preset == "eval_bench":
        # MVBench + TempCompass combined JSONL
        return [{"jsonl": "datasets/eval_benchmark_train.jsonl", "video_root": "/"}]
    raise ValueError(f"Unknown dataset_preset: {dataset_preset!r}")


def resolve_model_path(base_model_path):
    candidates = [
        base_model_path,
        os.path.join("models", os.path.basename(base_model_path)),
        os.path.join("DAMO-NLP-SG", os.path.basename(base_model_path)),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return base_model_path


def save_checkpoint(output_dir, step, wrapped, optimizer, adapter_config):
    ckpt_dir = os.path.join(output_dir, f"ckpt_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    inner = wrapped._orig_mod if hasattr(wrapped, "_orig_mod") else wrapped
    torch.save(inner.expert.state_dict(), os.path.join(ckpt_dir, "diff_expert.pt"))
    torch.save(inner.router.state_dict(), os.path.join(ckpt_dir, "router.pt"))
    torch.save(
        {
            "state_bridge_head": inner.state_bridge_head.state_dict(),
            "local_corr_q": inner.local_corr_q.state_dict(),
            "local_corr_k": inner.local_corr_k.state_dict(),
        },
        os.path.join(ckpt_dir, "aux_heads.pt"),
    )
    with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict()},
        os.path.join(ckpt_dir, "training_state.pt"),
    )


def save_best_checkpoint(output_dir, tag, step, wrapped, optimizer, adapter_config, meta):
    ckpt_dir = os.path.join(output_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    inner = wrapped._orig_mod if hasattr(wrapped, "_orig_mod") else wrapped
    torch.save(inner.expert.state_dict(), os.path.join(ckpt_dir, "diff_expert.pt"))
    torch.save(inner.router.state_dict(), os.path.join(ckpt_dir, "router.pt"))
    torch.save(
        {
            "state_bridge_head": inner.state_bridge_head.state_dict(),
            "local_corr_q": inner.local_corr_q.state_dict(),
            "local_corr_k": inner.local_corr_k.state_dict(),
        },
        os.path.join(ckpt_dir, "aux_heads.pt"),
    )
    with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "meta": meta},
        os.path.join(ckpt_dir, "training_state.pt"),
    )
    with open(os.path.join(ckpt_dir, "best_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def build_base_model(args, device, rank):
    model_path = resolve_model_path(args.base_model_path)
    log_rank0(rank, f"  base_model={model_path} attn={args.attn_implementation}")

    # Liger kernel: must be applied BEFORE model loading to monkey-patch Qwen2 classes
    if not args.no_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2

            apply_liger_kernel_to_qwen2(
                fused_linear_cross_entropy=False,  # we compute CE manually
            )
            log_rank0(rank, "[Liger] Applied Qwen2 kernel patches (RoPE, RMSNorm, SwiGLU)")
        except Exception as e:
            log_rank0(rank, f"[Liger] Not available, skipping: {e}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    for p in base_model.parameters():
        p.requires_grad = False
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        # NOTE: keep train() mode so gradient checkpointing actually activates.
        # Qwen2 + SigLIP both have dropout=0, so train/eval behave identically.
        base_model.train()
    else:
        base_model.eval()
    return base_model, processor, model_path


def should_train_phase_c_param(name, mode):
    if mode == "all":
        return True
    if mode == "gates":
        return name.endswith("local_gate.weight") or name.endswith("local_gate.bias") or name.endswith("state_gate.weight") or name.endswith("state_gate.bias")
    if mode == "readout":
        return name.startswith("local_up") or name.startswith("state_up")
    if mode == "readout_gates":
        return (
            name.startswith("local_up")
            or name.startswith("state_up")
            or name.startswith("local_gate")
            or name.startswith("state_gate")
        )
    raise ValueError(f"Unknown phase_c_unfreeze mode: {mode!r}")


def set_phase_trainability(wrapped, phase, phase_c_unfreeze):
    for p in wrapped.router.parameters():
        p.requires_grad = phase in ("B", "C")

    for p in wrapped.aux_params:
        p.requires_grad = phase in ("A", "C")

    if phase == "A":
        for p in wrapped.expert.parameters():
            p.requires_grad = True
        return

    if phase == "B":
        for p in wrapped.expert.parameters():
            p.requires_grad = False
        return

    for name, p in wrapped.expert.named_parameters():
        p.requires_grad = should_train_phase_c_param(name, phase_c_unfreeze)


def get_phase(global_step, phase_a_steps, phase_b_steps):
    if global_step < phase_a_steps:
        return "A", global_step
    if global_step < phase_a_steps + phase_b_steps:
        return "B", global_step - phase_a_steps
    return "C", global_step - phase_a_steps - phase_b_steps


def linear_ramp(step_in_phase, ramp_steps):
    if ramp_steps <= 0:
        return 1.0
    return max(0.0, min(1.0, float(step_in_phase + 1) / float(ramp_steps)))


def warmup_cosine_scale(step, total_steps, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def cosine_decay_scale(step_in_phase, total_phase_steps):
    if total_phase_steps <= 1:
        return 1.0
    progress = min(1.0, max(0.0, step_in_phase / float(total_phase_steps - 1)))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def resolve_phase_batch_size(args, phase):
    value = getattr(args, f"phase_{phase.lower()}_batch_size")
    return value if value and value > 0 else args.batch_size


def resolve_phase_grad_accum(args, phase):
    value = getattr(args, f"phase_{phase.lower()}_grad_accum")
    return value if value and value > 0 else args.grad_accum


def set_group_lrs(optimizer, phase, phase_lr_scale, phase_scale, args):
    lr_map = {"expert": 0.0, "aux": 0.0, "router": 0.0}
    if phase == "A":
        lr_map["expert"] = args.expert_lr * phase_lr_scale
        lr_map["aux"] = args.expert_lr * phase_lr_scale
    elif phase == "B":
        router_scale = max(args.phase_b_min_router_scale, phase_scale)
        lr_map["router"] = args.router_lr * phase_lr_scale * router_scale
    else:
        expert_scale = max(args.phase_c_min_expert_scale, phase_scale)
        lr_map["expert"] = args.joint_expert_lr * phase_lr_scale * expert_scale
        lr_map["aux"] = args.joint_expert_lr * phase_lr_scale * expert_scale
        lr_map["router"] = args.router_lr * phase_lr_scale

    for group in optimizer.param_groups:
        group["lr"] = lr_map[group["name"]]


def compute_phase_score(wrapped, phase):
    if phase == "A":
        return -(
            wrapped._last_qa
            + wrapped.align_lambda * wrapped._last_align
            + wrapped.state_mask_lambda * wrapped._last_state_mask
            + wrapped.local_corr_lambda * wrapped._last_local_corr
        )

    router_health = wrapped._last_w_l_std + wrapped._last_w_s_std
    score = -(
        wrapped._last_qa
        + wrapped.align_lambda * wrapped._last_align
        + wrapped.router_lambda * wrapped._last_router
        + wrapped.router_pattern_lambda * wrapped._last_router_pattern
        + wrapped.router_var_lambda * wrapped._last_router_var
        + wrapped.balance_lambda * wrapped._last_balance
        - wrapped.entropy_lambda * wrapped._last_entropy
    )
    return score + 0.01 * router_health


def best_eligible(phase, phase_step, wrapped, args):
    if phase == "A":
        return phase_step >= args.best_phase_a_start
    if phase == "B":
        return (
            phase_step >= args.phase_b_router_ramp_steps
            and wrapped._last_w_l_std >= args.best_min_router_std
            and wrapped._last_w_s_std >= args.best_min_router_std
        )
    return (
        phase_step >= args.phase_c_joint_ramp_steps
        and wrapped._last_w_l_std >= args.best_min_router_std
        and wrapped._last_w_s_std >= args.best_min_router_std
    )


def maybe_update_best(output_dir, tag, step, phase, phase_step, wrapped, optimizer, adapter_config, best_track, args):
    raw_score = compute_phase_score(wrapped, phase)
    prev_ema = best_track.get("ema")
    ema_score = raw_score if prev_ema is None else args.best_ema_beta * prev_ema + (1.0 - args.best_ema_beta) * raw_score
    best_track["ema"] = ema_score
    best_track["raw"] = raw_score

    if not best_eligible(phase, phase_step, wrapped, args):
        return False

    best_score = best_track.get("best_score", float("-inf"))
    if ema_score <= best_score + args.best_delta:
        return False

    best_track["best_score"] = ema_score
    best_track["best_step"] = step
    meta = {
        "tag": tag,
        "phase": phase,
        "phase_step": phase_step,
        "global_step": step,
        "score_raw": raw_score,
        "score_ema": ema_score,
        "metrics": {
            "loss_total": wrapped._last_total,
            "qa": wrapped._last_qa,
            "align": wrapped._last_align,
            "state_mask": wrapped._last_state_mask,
            "local_corr": wrapped._last_local_corr,
            "local_acc": wrapped._last_local_acc,
            "router": wrapped._last_router,
            "router_pattern": wrapped._last_router_pattern,
            "router_var": wrapped._last_router_var,
            "entropy": wrapped._last_entropy,
            "balance": wrapped._last_balance,
            "w_local": wrapped._last_w_l,
            "w_state": wrapped._last_w_s,
            "w_local_std": wrapped._last_w_l_std,
            "w_state_std": wrapped._last_w_s_std,
            "target_local": wrapped._last_tgt_l,
            "target_state": wrapped._last_tgt_s,
            "target_local_std": wrapped._last_tgt_l_std,
            "target_state_std": wrapped._last_tgt_s_std,
            "qa_base": wrapped._last_qa_base,
            "qa_local": wrapped._last_qa_local,
            "qa_state": wrapped._last_qa_state,
            "expert_local_gate": wrapped.expert._last_local_gate_mean,
            "expert_state_gate": wrapped.expert._last_state_gate_mean,
        },
    }
    save_best_checkpoint(output_dir, tag, step, wrapped, optimizer, adapter_config, meta)
    return True


def save_phase_final(output_dir, tag, step, phase, phase_step, wrapped, optimizer, adapter_config):
    meta = {
        "tag": tag,
        "phase": phase,
        "phase_step": phase_step,
        "global_step": step,
        "metrics": {
            "loss_total": wrapped._last_total,
            "qa": wrapped._last_qa,
            "align": wrapped._last_align,
            "state_mask": wrapped._last_state_mask,
            "local_corr": wrapped._last_local_corr,
            "local_acc": wrapped._last_local_acc,
            "router": wrapped._last_router,
            "router_pattern": wrapped._last_router_pattern,
            "router_var": wrapped._last_router_var,
            "entropy": wrapped._last_entropy,
            "balance": wrapped._last_balance,
            "w_local": wrapped._last_w_l,
            "w_state": wrapped._last_w_s,
            "w_local_std": wrapped._last_w_l_std,
            "w_state_std": wrapped._last_w_s_std,
            "target_local": wrapped._last_tgt_l,
            "target_state": wrapped._last_tgt_s,
            "target_local_std": wrapped._last_tgt_l_std,
            "target_state_std": wrapped._last_tgt_s_std,
            "qa_base": wrapped._last_qa_base,
            "qa_local": wrapped._last_qa_local,
            "qa_state": wrapped._last_qa_state,
            "expert_local_gate": wrapped.expert._last_local_gate_mean,
            "expert_state_gate": wrapped.expert._last_state_gate_mean,
        },
    }
    save_best_checkpoint(output_dir, tag, step, wrapped, optimizer, adapter_config, meta)


def build_loader(dataset, batch_size, sampler, shuffle, num_workers, collate_fn):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def sync_and_step(wrapped, optimizer, world_size, device):
    if world_size > 1:
        for p in wrapped.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    active_params = [p for p in wrapped.parameters() if p.requires_grad and p.grad is not None]
    grad_norm = torch.nn.utils.clip_grad_norm_(active_params, 1.0) if active_params else torch.tensor(0.0, device=device)
    wrapped._last_grad = float(grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train(args):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        rank, world_size = 0, 1
        device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)
    phase_c_steps = args.max_steps - args.phase_a_steps - args.phase_b_steps
    if phase_c_steps <= 0:
        raise ValueError("max_steps must be larger than phase_a_steps + phase_b_steps")
    phase_total_steps = {
        "A": args.phase_a_steps,
        "B": args.phase_b_steps,
        "C": phase_c_steps,
    }

    factor = args.patch_size * args.merge_size
    target_size = int(round(float(args.image_size) / float(factor)) * factor)
    tokens_per_frame = (target_size // args.patch_size // args.merge_size) ** 2
    max_length = args.max_length if args.max_length > 0 else args.num_frames * tokens_per_frame + args.text_budget
    warmup_steps = min(args.global_warmup_steps, max(1, args.max_steps // 10))
    dataset_configs = resolve_dataset_configs(args.dataset_preset)

    log_rank0(rank, "=== VideoLLaMA3 UDH v2 Unified Curriculum ===")
    log_rank0(rank, f"  phases=A:{args.phase_a_steps} B:{args.phase_b_steps} C:{phase_c_steps}")
    log_rank0(rank, f"  lrs expert={args.expert_lr:.2e} router={args.router_lr:.2e} joint_expert={args.joint_expert_lr:.2e}")
    log_rank0(rank, f"  phase_c_unfreeze={args.phase_c_unfreeze}")
    log_rank0(
        rank,
        f"  router_reg pattern={args.router_pattern_lambda:.3f} var={args.router_var_lambda:.3f} "
        f"scale_floor(B={args.phase_b_min_router_scale:.2f}, C={args.phase_c_min_expert_scale:.2f})",
    )
    n_sources = len(dataset_configs) if isinstance(dataset_configs, list) else dataset_configs
    log_rank0(rank, f"  dataset_preset={args.dataset_preset} ({n_sources} sources)")
    log_rank0(rank, f"  num_frames={args.num_frames} image_size={args.image_size} aligned_size={target_size}")
    log_rank0(rank, f"  tokens_per_frame={tokens_per_frame} max_length={max_length}")
    for phase_name in ("A", "B", "C"):
        phase_bs = resolve_phase_batch_size(args, phase_name)
        phase_ga = resolve_phase_grad_accum(args, phase_name)
        eff_batch = phase_bs * phase_ga * world_size
        log_rank0(rank, f"  phase_{phase_name.lower()} batch={phase_bs}x{phase_ga}x{world_size}={eff_batch}")

    base_model, processor, resolved_model_path = build_base_model(args, device, rank)
    proj_dim = base_model.config.hidden_size
    pooling_ratio = 1
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    expert = SplitDualHeadExpert(
        proj_dim=proj_dim,
        bottleneck=args.expert_bottleneck,
        kernel_size=args.expert_kernel_size,
        max_local_gate=args.max_local_gate,
        max_state_gate=args.max_state_gate,
        pooling_ratio=pooling_ratio,
        ortho_proj=args.ortho_proj,
    ).to(device=device, dtype=torch.bfloat16)

    if args.video_only:
        expert.no_gate = True
        log_rank0(rank, "  Expert no_gate=True (video-only stage 1)")

    q_dim = base_model.get_model().embed_tokens.weight.shape[1]
    if args.router_type == "gated_attn":
        router = GatedAttentionDualRouter(
            q_dim=q_dim,
            video_dim=proj_dim,
            hidden_dim=args.router_hidden,
            num_heads=4,
            stats_dim=4,
            init_local=args.init_local,
            init_state=args.init_state,
        ).to(device=device, dtype=torch.bfloat16)
    else:
        router = DualBranchRouter(
            embed_dim=q_dim,
            hidden_dim=args.router_hidden,
            stats_dim=4,
            init_local=args.init_local,
            init_state=args.init_state,
        ).to(device=device, dtype=torch.bfloat16)

    wrapped = VideoLLaMA3UDHUnifiedCurriculumModel(
        base_model,
        expert,
        router,
        image_token_id=base_model.config.image_token_index,
        pad_token_id=pad_token_id,
        patch_size=args.patch_size,
        merge_size=args.merge_size,
        align_lambda=args.align_lambda,
        state_mask_lambda=args.state_mask_lambda,
        state_mask_ratio=args.state_mask_ratio,
        state_pred_dropout=args.state_pred_dropout,
        local_corr_lambda=args.local_corr_lambda,
        local_corr_radius=args.local_corr_radius,
        local_corr_anchors=args.local_corr_anchors,
        local_corr_dropout=args.local_corr_dropout,
        router_lambda=args.router_lambda,
        router_tau=args.router_tau,
        router_margin=args.router_margin,
        entropy_lambda=args.entropy_lambda,
        balance_lambda=args.balance_lambda,
        balance_target=(args.balance_target_local, args.balance_target_state),
        router_pattern_lambda=args.router_pattern_lambda,
        router_var_lambda=args.router_var_lambda,
        video_only=args.video_only,
    ).to(device=device)

    if args.native_qa_loss:
        wrapped._use_native_qa_loss = True
        log_rank0(rank, "  *** Using native QA loss (matches eval-time token compression) ***")

    optimizer = torch.optim.AdamW(
        [
            {"params": list(wrapped.expert.parameters()), "lr": 0.0, "name": "expert"},
            {"params": wrapped.aux_params, "lr": 0.0, "name": "aux"},
            {"params": list(wrapped.router.parameters()), "lr": 0.0, "name": "router"},
        ],
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    if dataset_configs == "mvbench":
        dataset = MVBenchVideoQADataset(
            tokenizer=tokenizer,
            num_frames=args.num_frames,
            max_length=max_length,
            image_size=args.image_size,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
            mvbench_base=getattr(args, "mvbench_base", None),
        )
    else:
        dataset = build_dataset(
            dataset_configs,
            tokenizer=tokenizer,
            num_frames=args.num_frames,
            max_length=max_length,
            image_size=args.image_size,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
        )
    if args.video_only:
        dataset = VideoOnlyDataset(
            data_path=args.video_only_data,
            num_frames=args.num_frames,
            image_size=args.image_size,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
        )
        collate_fn = video_only_collate_fn
    else:
        collate_fn = make_collate_fn(pad_token_id)
    phase_loaders, phase_samplers, phase_iters = {}, {}, {}
    phase_epochs = {"A": 0, "B": 0, "C": 0}
    for phase_name in ("A", "B", "C"):
        batch_size = resolve_phase_batch_size(args, phase_name)
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.data_seed)
            if world_size > 1
            else None
        )
        loader = build_loader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        phase_loaders[phase_name] = loader
        phase_samplers[phase_name] = sampler
        phase_iters[phase_name] = None

    adapter_config = {
        "module_type": "videollama3_udh_v2_dbrouter_unified",
        "stage1_module_type": "videollama3_udh_v2_stage1_unified",
        "curriculum_style": "videollama3_udh_v2_unified_router_curriculum",
        "base_model_path": resolved_model_path,
        "attn_implementation": args.attn_implementation,
        "liger_kernel": not args.no_liger,
        "native_qa_loss": bool(args.native_qa_loss),
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "aligned_size": target_size,
        "patch_size": args.patch_size,
        "merge_size": args.merge_size,
        "tokens_per_frame": tokens_per_frame,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "phase_a_batch_size": resolve_phase_batch_size(args, "A"),
        "phase_b_batch_size": resolve_phase_batch_size(args, "B"),
        "phase_c_batch_size": resolve_phase_batch_size(args, "C"),
        "phase_a_grad_accum": resolve_phase_grad_accum(args, "A"),
        "phase_b_grad_accum": resolve_phase_grad_accum(args, "B"),
        "phase_c_grad_accum": resolve_phase_grad_accum(args, "C"),
        "proj_dim": proj_dim,
        "pooling_ratio": pooling_ratio,
        "expert_bottleneck": args.expert_bottleneck,
        "expert_kernel_size": args.expert_kernel_size,
        "max_local_gate": args.max_local_gate,
        "max_state_gate": args.max_state_gate,
        "align_lambda": args.align_lambda,
        "ortho_proj": args.ortho_proj,
        "video_only": args.video_only,
        "state_mask_lambda": args.state_mask_lambda,
        "state_mask_ratio": args.state_mask_ratio,
        "local_corr_lambda": args.local_corr_lambda,
        "local_corr_radius": args.local_corr_radius,
        "local_corr_anchors": args.local_corr_anchors,
        "router_lambda": args.router_lambda,
        "router_tau": args.router_tau,
        "router_margin": args.router_margin,
        "router_pattern_lambda": args.router_pattern_lambda,
        "router_var_lambda": args.router_var_lambda,
        "entropy_lambda": args.entropy_lambda,
        "balance_lambda": args.balance_lambda,
        "balance_target_local": args.balance_target_local,
        "balance_target_state": args.balance_target_state,
        "phase_a_steps": args.phase_a_steps,
        "phase_b_steps": args.phase_b_steps,
        "phase_b_min_router_scale": args.phase_b_min_router_scale,
        "phase_c_min_expert_scale": args.phase_c_min_expert_scale,
        "phase_c_unfreeze": args.phase_c_unfreeze,
        "joint_aux_scale": args.phase_c_aux_scale,
        "router_type": args.router_type,
        "router_hidden": args.router_hidden,
        "expert_lr": args.expert_lr,
        "router_lr": args.router_lr,
        "joint_expert_lr": args.joint_expert_lr,
        "dataset_preset": args.dataset_preset,
        "train_datasets": dataset_configs,
    }

    global_step = 0
    accum_counter = 0
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    current_phase = None
    best_track = {
        "phase_a": {"ema": None, "raw": None, "best_score": float("-inf"), "best_step": -1},
        "phase_b": {"ema": None, "raw": None, "best_score": float("-inf"), "best_step": -1},
        "phase_c": {"ema": None, "raw": None, "best_score": float("-inf"), "best_step": -1},
        "final": {"ema": None, "raw": None, "best_score": float("-inf"), "best_step": -1},
    }

    while global_step < args.max_steps:
        phase, phase_step = get_phase(global_step, args.phase_a_steps, args.phase_b_steps)
        if phase != current_phase:
            if accum_counter > 0:
                sync_and_step(wrapped, optimizer, world_size, device)
                accum_counter = 0
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            set_phase_trainability(wrapped, phase, args.phase_c_unfreeze)
            current_phase = phase
            phase_bs = resolve_phase_batch_size(args, phase)
            phase_ga = resolve_phase_grad_accum(args, phase)
            eff_batch = phase_bs * phase_ga * world_size
            log_rank0(rank, f"--- Switch to Phase {phase} at step {global_step} (batch={phase_bs}x{phase_ga}x{world_size}={eff_batch}) ---")

        loader = phase_loaders[phase]
        sampler = phase_samplers[phase]
        if phase_iters[phase] is None:
            if sampler is not None:
                sampler.set_epoch(phase_epochs[phase])
            phase_epochs[phase] += 1
            phase_iters[phase] = iter(loader)

        try:
            batch = next(phase_iters[phase])
        except StopIteration:
            phase_iters[phase] = None
            continue

        if phase == "B":
            phase_scale = linear_ramp(phase_step, args.phase_b_router_ramp_steps)
            aux_scale = 0.0
            phase_lr_scale = cosine_decay_scale(phase_step, phase_total_steps[phase])
        elif phase == "C":
            phase_scale = linear_ramp(phase_step, args.phase_c_joint_ramp_steps)
            aux_scale = args.phase_c_aux_scale * phase_scale
            phase_lr_scale = cosine_decay_scale(phase_step, phase_total_steps[phase])
        else:
            phase_scale = 1.0
            aux_scale = 1.0
            phase_lr_scale = warmup_cosine_scale(phase_step, phase_total_steps[phase], warmup_steps)
        set_group_lrs(optimizer, phase, phase_lr_scale, phase_scale, args)

        grad_accum = resolve_phase_grad_accum(args, phase)
        if args.video_only:
            input_ids = None
            attention_mask = None
            labels = None
            prompt_mask = None
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                prompt_mask=prompt_mask,
                labels=labels,
                phase=phase,
                router_loss_scale=phase_scale,
                aux_loss_scale=aux_scale,
            )
        (loss / grad_accum).backward()
        accum_counter += 1

        if accum_counter >= grad_accum or global_step == args.max_steps - 1:
            sync_and_step(wrapped, optimizer, world_size, device)
            accum_counter = 0

        global_step += 1

        if rank == 0:
            phase_final_tags = []
            if phase == "A" and global_step == args.phase_a_steps:
                save_phase_final(args.output_dir, "phase_a_final", global_step, phase, phase_step, wrapped, optimizer, adapter_config)
                phase_final_tags.append("phase_a_final")
            if phase == "B" and global_step == args.phase_a_steps + args.phase_b_steps:
                save_phase_final(args.output_dir, "phase_b_final", global_step, phase, phase_step, wrapped, optimizer, adapter_config)
                phase_final_tags.append("phase_b_final")
            if phase == "C" and global_step == args.max_steps:
                save_phase_final(args.output_dir, "phase_c_final", global_step, phase, phase_step, wrapped, optimizer, adapter_config)
                phase_final_tags.append("phase_c_final")
            if phase_final_tags:
                log_rank0(rank, f"[Phase Final @ {global_step}] saved: {', '.join(phase_final_tags)}")

        if rank == 0:
            improved_tags = []
            if phase == "A":
                if maybe_update_best(args.output_dir, "best_phase_a", global_step, phase, phase_step, wrapped, optimizer, adapter_config, best_track["phase_a"], args):
                    improved_tags.append("best_phase_a")
            elif phase == "B":
                if maybe_update_best(args.output_dir, "best_phase_b", global_step, phase, phase_step, wrapped, optimizer, adapter_config, best_track["phase_b"], args):
                    improved_tags.append("best_phase_b")
            else:
                if maybe_update_best(args.output_dir, "best_phase_c", global_step, phase, phase_step, wrapped, optimizer, adapter_config, best_track["phase_c"], args):
                    improved_tags.append("best_phase_c")
                if maybe_update_best(args.output_dir, "best_final", global_step, phase, phase_step, wrapped, optimizer, adapter_config, best_track["final"], args):
                    improved_tags.append("best_final")

            if improved_tags:
                with open(os.path.join(args.output_dir, "best_summary.json"), "w") as f:
                    json.dump(best_track, f, indent=2)
                log_rank0(rank, f"[Best @ {global_step}] updated: {', '.join(improved_tags)}")

        if rank == 0 and global_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            speed = elapsed / args.log_interval
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            phase_bs = resolve_phase_batch_size(args, phase)
            phase_ga = resolve_phase_grad_accum(args, phase)
            log_rank0(
                rank,
                f"[Step {global_step}/{args.max_steps}][{wrapped._last_phase}] "
                f"bs={phase_bs} ga={phase_ga} "
                f"loss={wrapped._last_total:.4f} qa={wrapped._last_qa:.4f} "
                f"align={wrapped._last_align:.4f} rtr={wrapped._last_router:.4f} "
                f"r_pat={wrapped._last_router_pattern:.4f} r_var={wrapped._last_router_var:.4f} "
                f"s_mask={wrapped._last_state_mask:.4f} s_acc={wrapped._last_state_acc:.3f} l_corr={wrapped._last_local_corr:.4f} l_acc={wrapped._last_local_acc:.3f} "
                f"w_l={wrapped._last_w_l:.3f} w_s={wrapped._last_w_s:.3f} "
                f"w_l_std={wrapped._last_w_l_std:.3f} w_s_std={wrapped._last_w_s_std:.3f} "
                f"tgt_l={wrapped._last_tgt_l:.3f} tgt_s={wrapped._last_tgt_s:.3f} "
                f"tgt_l_std={wrapped._last_tgt_l_std:.3f} tgt_s_std={wrapped._last_tgt_s_std:.3f} "
                f"qa_b={wrapped._last_qa_base:.3f} qa_l={wrapped._last_qa_local:.3f} qa_s={wrapped._last_qa_state:.3f} "
                f"l_gate={expert._last_local_gate_mean:.4f} s_gate={expert._last_state_gate_mean:.4f} "
                f"lg_std={expert._last_local_gate_std:.4f} lg_max={expert._last_local_gate_max:.4f} lg_sp={expert._last_local_gate_sparse:.2f} "
                f"sg_std={expert._last_state_gate_std:.4f} sg_max={expert._last_state_gate_max:.4f} sg_sp={expert._last_state_gate_sparse:.2f} "
                f"la_m={expert._last_local_alpha_mean:.3f} la_s={expert._last_local_alpha_std:.3f} la_min={expert._last_local_alpha_min:.3f} "
                f"sa_m={expert._last_state_alpha_mean:.3f} sa_s={expert._last_state_alpha_std:.3f} sa_min={expert._last_state_alpha_min:.3f} "
                f"le_n={expert._last_local_enrich_norm:.3f} le_s={expert._last_local_enrich_std:.3f} lf_s={expert._last_local_frame_std:.3f} "
                f"se_n={expert._last_state_enrich_norm:.3f} se_s={expert._last_state_enrich_std:.3f} sf_s={expert._last_state_frame_std:.3f} "
                f"ort={expert._last_ortho_ratio:.2f} par={expert._last_para_ratio:.2f} "
                f"g={wrapped._last_grad:.3f} "
                f"lr_e={optimizer.param_groups[0]['lr']:.2e} lr_a={optimizer.param_groups[1]['lr']:.2e} lr_r={optimizer.param_groups[2]['lr']:.2e} "
                f"spd={speed:.2f}s mem={mem:.1f}GB"
            )
            t_start = time.time()
            torch.cuda.reset_peak_memory_stats(device)

        if args.save_interval > 0 and (global_step % args.save_interval == 0 or global_step == args.max_steps):
            if rank == 0:
                save_checkpoint(args.output_dir, global_step, wrapped, optimizer, adapter_config)

    if dist.is_initialized():
        dist.destroy_process_group()


# =====================================================================
# Stage 2: TemporalQueryRouter 학습
# =====================================================================

class VL3Stage2RouterModel(nn.Module):
    """VideoLLaMA3 Stage 2 학습용 wrapper.

    Expert(frozen)로 enrichment 생성 → TemporalQueryRouter로 프레임별 gate 결정
    → z_final 구성 → QA loss로 router만 학습.
    """

    def __init__(
        self,
        model,         # VideoLLaMA3 base (frozen)
        expert,        # SplitDualHeadExpert (frozen)
        tq_router,     # TemporalQueryRouter (trainable)
        image_token_id,
        pad_token_id,
        patch_size=14,
        merge_size=2,
        align_lambda=0.1,
    ):
        super().__init__()
        self.model = model
        self.expert = expert
        self.tq_router = tq_router
        self.image_token_id = int(image_token_id)
        self.pad_token_id = int(pad_token_id)
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.align_lambda = float(align_lambda)

        # 캐시 초기화 (lazy property용)
        self._lm_inner = None
        self._vision_enc = None
        self._mm_proj = None

        # Qwen2 chat template special token IDs
        self._im_start_id = 151644
        self._im_end_id = 151645

        # Vision encoder chunk size: flash_attn이면 전체 한번에, SDPA/eager면 chunk
        attn_impl = getattr(model.config, "_attn_implementation", "sdpa")
        self._vision_chunk_size = 0 if attn_impl == "flash_attention_2" else 4

        # 로깅용 캐시
        self._last_total = 0.0
        self._last_qa = 0.0
        self._last_align = 0.0
        self._last_local_scale_mean = 0.0
        self._last_local_scale_std = 0.0
        self._last_state_scale_mean = 0.0
        self._last_state_scale_std = 0.0
        self._last_grad = 0.0

    def _get_raw_model(self):
        """PeftModel 래핑 여부에 관계없이 원본 CausalLM 반환."""
        m = self.model
        if hasattr(m, "peft_config"):  # PeftModelForCausalLM
            return m.base_model.model  # Videollama3Qwen2ForCausalLM
        return m

    @property
    def _lm_transformer(self):
        """LLM transformer (LoRA 적용된 layers 포함)."""
        if self._lm_inner is None:
            m = self.model
            if hasattr(m, "peft_config"):
                # PeftModel → LoraModel.model.model = Qwen2Model (LoRA layers 포함)
                self._lm_inner = m.base_model.model.model
            else:
                self._lm_inner = m.model
        return self._lm_inner

    @property
    def _lm_head(self):
        return self._get_raw_model().lm_head

    @property
    def _vision_encoder(self):
        if self._vision_enc is None:
            self._vision_enc = self._get_raw_model().model.vision_encoder
        return self._vision_enc

    @property
    def _mm_projector(self):
        if self._mm_proj is None:
            self._mm_proj = self._get_raw_model().model.mm_projector
        return self._mm_proj

    def _build_stage1_inputs(self, pixel_values, chunk_size=0):
        """Vision encoder forward: (B, T, 3, H, W) → z_base (B, T, S, D)."""
        bsz, t, _, h, w = pixel_values.shape
        hp = h // self.patch_size
        wp = w // self.patch_size
        hm = hp // self.merge_size
        wm = wp // self.merge_size
        patches_per_frame = hp * wp
        tokens_per_frame = hm * wm

        n_total = bsz * t
        effective_chunk = chunk_size if chunk_size > 0 else n_total

        all_projected = []
        with torch.no_grad():
            flat_frames = pixel_values.reshape(n_total, 3, h, w)
            for ci in range(0, n_total, effective_chunk):
                end = min(ci + effective_chunk, n_total)
                n_chunk = end - ci
                chunk_frames = flat_frames[ci:end]
                patches = chunk_frames.unfold(2, self.patch_size, self.patch_size).unfold(
                    3, self.patch_size, self.patch_size
                )
                patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
                patches = patches.reshape(n_chunk * patches_per_frame, 3 * self.patch_size * self.patch_size)
                grid_sizes = torch.tensor(
                    [[1, hp, wp]] * n_chunk, device=pixel_values.device, dtype=torch.long
                )
                msizes = torch.tensor(
                    [self.merge_size] * n_chunk, device=pixel_values.device, dtype=torch.long
                )
                merged = self._vision_encoder(patches, grid_sizes, msizes)
                projected = self._mm_projector(merged)
                all_projected.append(projected.reshape(n_chunk, tokens_per_frame, -1))

        z = torch.cat(all_projected, dim=0).reshape(bsz, t, tokens_per_frame, -1)
        return z, z  # z_pre_pool, z_base 동일

    def _get_question_embedding(self, input_ids, prompt_mask):
        """질문 텍스트 부분의 embedding mean pool → (B, D)."""
        with torch.no_grad():
            all_embeds = self._get_raw_model().get_model().embed_tokens(input_ids)
        q_embeds = []
        for i in range(input_ids.shape[0]):
            text_mask = (
                prompt_mask[i]
                & input_ids[i].ne(self.image_token_id)
                & input_ids[i].ne(self.pad_token_id)
                & input_ids[i].ne(self._im_start_id)
                & input_ids[i].ne(self._im_end_id)
            )
            if text_mask.any():
                q_embeds.append(all_embeds[i][text_mask].mean(dim=0))
            else:
                q_embeds.append(
                    torch.zeros(all_embeds.shape[-1], device=all_embeds.device, dtype=all_embeds.dtype)
                )
        return torch.stack(q_embeds)

    def _qa_loss(self, projected_flat, input_ids, attention_mask, labels, chunk_seq=1024):
        """QA loss (chunked CE)."""
        inputs_embeds = self._get_raw_model().get_model().embed_tokens(input_ids).detach()
        projected_flat = projected_flat.to(inputs_embeds.dtype)
        image_mask = input_ids.eq(self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, projected_flat)
        outputs = self._lm_transformer(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        lm_head = self._lm_head
        qa_loss = torch.tensor(0.0, device=input_ids.device)
        total_tokens = 0
        for i in range(0, shift_hidden.shape[1], chunk_seq):
            h_chunk = shift_hidden[:, i : i + chunk_seq, :]
            l_chunk = shift_labels[:, i : i + chunk_seq]
            n_valid = (l_chunk != -100).sum().item()
            if n_valid > 0:
                logits_chunk = lm_head(h_chunk)
                chunk_loss = F.cross_entropy(
                    logits_chunk.view(-1, logits_chunk.size(-1)),
                    l_chunk.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                qa_loss = qa_loss + chunk_loss
                total_tokens += n_valid
        if total_tokens > 0:
            qa_loss = qa_loss / total_tokens
        return qa_loss.float()

    @staticmethod
    def _alignment_loss(z_final, z_base):
        return 1.0 - F.cosine_similarity(z_final.float(), z_base.float(), dim=-1).mean()

    def forward(self, input_ids, attention_mask, pixel_values, prompt_mask, labels):
        """
        Stage 2 forward:
        1. Vision encoder → z_base (frozen)
        2. Expert → local_res, state_res (frozen)
        3. 질문 embedding 추출
        4. TemporalQueryRouter → 프레임별 local_scale, state_scale
        5. z_final = z_base + local_scale * local_res + state_scale * state_res
        6. QA loss
        """
        # 1) Vision features
        z_pre_pool, z_base = self._build_stage1_inputs(pixel_values, chunk_size=self._vision_chunk_size)

        # 2) Expert (frozen) → enrichment
        with torch.no_grad():
            _, local_res, state_res, _ = self.expert(z_pre_pool, z_base, return_aux=True)

        # 3) 질문 embedding
        q_embed = self._get_question_embedding(input_ids, prompt_mask)  # (B, D)

        # 4) 프레임별 summary 생성 (spatial mean)
        z_base_summary = z_base.mean(dim=2)          # (B, T, D)
        local_summary = local_res.mean(dim=2)         # (B, T, D)
        state_summary = state_res.mean(dim=2)         # (B, T, D)

        # 5) Router → 프레임별 gate
        local_scale, state_scale = self.tq_router(
            q_embed, z_base_summary, local_summary, state_summary,
        )

        # 6) 최종 feature 합성
        # Enrichment normalize + scale
        base_norm = z_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        local_norm = local_res * (base_norm / local_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        state_norm = state_res * (base_norm / state_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        ls = local_scale.unsqueeze(-1)
        ss = state_scale.unsqueeze(-1)
        z_final = (1.0 - ls - ss) * z_base + ls * local_norm + ss * state_norm

        # 7) QA loss
        qa_loss = self._qa_loss(
            z_final.reshape(z_final.shape[0], -1, z_final.shape[-1]),
            input_ids, attention_mask, labels,
        )
        align_loss = self._alignment_loss(z_final, z_base)
        total_loss = qa_loss + self.align_lambda * align_loss

        # 로깅 캐시 업데이트
        self._last_total = total_loss.item()
        self._last_qa = qa_loss.item()
        self._last_align = align_loss.item()
        self._last_local_scale_mean = self.tq_router._last_local_scale_mean
        self._last_local_scale_std = self.tq_router._last_local_scale_std
        self._last_state_scale_mean = self.tq_router._last_state_scale_mean
        self._last_state_scale_std = self.tq_router._last_state_scale_std

        return total_loss


def _save_vl3_stage2_checkpoint(output_dir, tag, step, wrapped, optimizer, adapter_config):
    """Stage 2 체크포인트 저장: router + config + (optional) LoRA."""
    ckpt_dir = os.path.join(output_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    inner = wrapped._orig_mod if hasattr(wrapped, "_orig_mod") else wrapped

    # Router weights
    torch.save(inner.tq_router.state_dict(), os.path.join(ckpt_dir, "tq_router.pt"))
    # Expert weights (참조용 복사 — frozen이므로 stage1과 동일)
    torch.save(inner.expert.state_dict(), os.path.join(ckpt_dir, "diff_expert.pt"))
    # Config
    adapter_config_save = dict(adapter_config)
    adapter_config_save["saved_step"] = step
    with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config_save, f, indent=2)
    # Training state
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict()},
        os.path.join(ckpt_dir, "training_state.pt"),
    )
    # LoRA (있으면)
    bm = getattr(inner, "_base_model_ref", None)
    if bm is not None and hasattr(bm, "save_pretrained"):
        bm.save_pretrained(os.path.join(ckpt_dir, "lora"))


def train_stage2(args):
    """Stage 2 학습 메인 루프: expert freeze → TemporalQueryRouter만 학습."""
    from core.vision_projector.temporal_query_router import TemporalQueryRouter

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        rank, world_size = 0, 1
        device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1 checkpoint 경로 확인
    s1_dir = args.stage2_checkpoint
    if not os.path.isdir(s1_dir):
        raise ValueError(f"Stage 1 checkpoint not found: {s1_dir}")
    s1_config_path = os.path.join(s1_dir, "adapter_config.json")
    if not os.path.exists(s1_config_path):
        raise ValueError(f"adapter_config.json not found in {s1_dir}")
    with open(s1_config_path) as f:
        s1_config = json.load(f)

    # 기본 모델 로드
    base_model, processor, resolved_model_path = build_base_model(args, device, rank)
    tokenizer = processor.tokenizer
    image_token_id = base_model.config.image_token_index
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # proj_dim 결정
    proj_dim = base_model.config.hidden_size
    factor = args.patch_size * args.merge_size
    target_size = int(round(float(args.image_size) / float(factor)) * factor)
    hm = target_size // args.patch_size // args.merge_size
    tokens_per_frame = hm * hm
    max_length = args.max_length if args.max_length > 0 else args.num_frames * tokens_per_frame + args.text_budget

    # Expert 로드 + freeze
    expert = SplitDualHeadExpert(
        proj_dim=proj_dim,
        bottleneck=s1_config.get("expert_bottleneck", args.expert_bottleneck),
        kernel_size=s1_config.get("expert_kernel_size", args.expert_kernel_size),
        max_local_gate=s1_config.get("max_local_gate", args.max_local_gate),
        max_state_gate=s1_config.get("max_state_gate", args.max_state_gate),
        pooling_ratio=s1_config.get("pooling_ratio", 1),
        ortho_proj=s1_config.get("ortho_proj", False),
        skip_output_norm=s1_config.get("skip_output_norm", False),
    ).to(device=device, dtype=torch.bfloat16)

    expert_path = os.path.join(s1_dir, "diff_expert.pt")
    expert.load_state_dict(torch.load(expert_path, map_location=device, weights_only=True))
    for p in expert.parameters():
        p.requires_grad = False
    expert.eval()
    if s1_config.get("video_only", False):
        expert.no_gate = True
        log_rank0(rank, f"  Expert no_gate=True (from stage1 video_only)")
    log_rank0(rank, f"Loaded & froze expert from {s1_dir}")

    # LoRA 로드 (Stage 1에서 학습된 것이 있으면)
    lora_params = []
    lora_dir = os.path.join(s1_dir, "lora")
    if os.path.isdir(lora_dir) and args.lora_rank > 0:
        from peft import PeftModel
        # VL3: model 자체가 CausalLM이므로 직접 LoRA 적용
        base_model = PeftModel.from_pretrained(
            base_model, lora_dir
        ).to(device=device, dtype=torch.bfloat16)
        if args.lora_lr > 0:
            lora_params = [p for p in base_model.parameters() if p.requires_grad]
            n_lora = sum(p.numel() for p in lora_params)
            log_rank0(rank, f"LoRA loaded from {lora_dir}, trainable params={n_lora:,}")
        else:
            for p in base_model.parameters():
                p.requires_grad = False
            log_rank0(rank, f"LoRA loaded from {lora_dir} (frozen)")

    # TemporalQueryRouter 생성
    tq_router = TemporalQueryRouter(
        feat_dim=proj_dim,
        hidden_dim=args.router_hidden,
        init_bias=args.stage2_init_bias,
    ).to(device=device, dtype=torch.bfloat16)
    n_router = sum(p.numel() for p in tq_router.parameters())
    log_rank0(rank, f"TemporalQueryRouter: {n_router:,} params (hidden={args.router_hidden})")

    # Stage2 wrapper
    wrapped = VL3Stage2RouterModel(
        model=base_model,
        expert=expert,
        tq_router=tq_router,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
        patch_size=args.patch_size,
        merge_size=args.merge_size,
        align_lambda=args.align_lambda,
    ).to(device=device)

    wrapped._base_model_ref = base_model if lora_params else None

    # torch.compile (선택)
    run_model = wrapped
    if args.torch_compile:
        backend = args.compile_backend or "inductor"
        mode = args.compile_mode or "default"
        run_model = torch.compile(wrapped, backend=backend, mode=mode, dynamic=args.compile_dynamic)
        log_rank0(rank, f"[torch.compile] backend={backend} mode={mode}")

    # Optimizer
    param_groups = [
        {"params": list(tq_router.parameters()), "lr": args.router_lr, "name": "tq_router"},
    ]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lora_lr, "name": "lora"})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.999))

    # Dataset
    dataset_configs = resolve_dataset_configs(args.dataset_preset)
    if dataset_configs == "mvbench":
        dataset = MVBenchVideoQADataset(
            tokenizer=tokenizer,
            num_frames=args.num_frames,
            max_length=max_length,
            image_size=args.image_size,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
            mvbench_base=getattr(args, "mvbench_base", None),
        )
    else:
        dataset = build_dataset(
            dataset_configs,
            tokenizer=tokenizer,
            num_frames=args.num_frames,
            max_length=max_length,
            image_size=args.image_size,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
        )
    collate = make_collate_fn(pad_token_id)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.data_seed) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    warmup_steps = min(args.global_warmup_steps, max(1, args.max_steps // 10))

    # Adapter config (저장용)
    adapter_config = {
        "module_type": "videollama3_udh_v2_stage2_tqrouter",
        "stage1_checkpoint": s1_dir,
        "stage1_config": s1_config,
        "base_model_path": resolved_model_path,
        "proj_dim": proj_dim,
        "router_hidden": args.router_hidden,
        "stage2_init_bias": args.stage2_init_bias,
        "align_lambda": args.align_lambda,
        "router_lr": args.router_lr,
        "lora_lr": args.lora_lr,
        "lora_rank": args.lora_rank,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "dataset_preset": args.dataset_preset,
    }

    log_rank0(rank, "=== VL3 Stage 2: TemporalQueryRouter Training ===")
    log_rank0(rank, f"  stage1_ckpt={s1_dir}")
    log_rank0(rank, f"  router_lr={args.router_lr:.2e} lora_lr={args.lora_lr:.2e}")
    log_rank0(rank, f"  max_steps={args.max_steps} batch={args.batch_size} grad_accum={args.grad_accum}")
    log_rank0(rank, f"  align_lambda={args.align_lambda} init_bias={args.stage2_init_bias}")

    global_step = 0
    accum_counter = 0
    epoch = 0
    data_iter = None
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    best_qa = float("inf")

    while global_step < args.max_steps:
        if data_iter is None:
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch += 1
            data_iter = iter(loader)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = None
            continue

        # LR warmup + cosine decay
        lr_scale = warmup_cosine_scale(global_step, args.max_steps, warmup_steps)
        for group in optimizer.param_groups:
            if group["name"] == "tq_router":
                group["lr"] = args.router_lr * lr_scale
            elif group["name"] == "lora":
                group["lr"] = args.lora_lr * lr_scale

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = run_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                prompt_mask=prompt_mask,
                labels=labels,
            )
        (loss / args.grad_accum).backward()
        accum_counter += 1

        if accum_counter >= args.grad_accum or global_step == args.max_steps - 1:
            # Gradient sync + step
            if world_size > 1:
                for p in wrapped.parameters():
                    if p.requires_grad and p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            active_params = [p for p in wrapped.parameters() if p.requires_grad and p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(active_params, 1.0) if active_params else torch.tensor(0.0, device=device)
            wrapped._last_grad = float(grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

        global_step += 1

        # 로깅
        if rank == 0 and global_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            speed = elapsed / args.log_interval
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            log_rank0(
                rank,
                f"[VL3-S2 Step {global_step}/{args.max_steps}] "
                f"loss={wrapped._last_total:.4f} qa={wrapped._last_qa:.4f} "
                f"align={wrapped._last_align:.4f} "
                f"l_scale={wrapped._last_local_scale_mean:.3f}±{wrapped._last_local_scale_std:.3f} "
                f"s_scale={wrapped._last_state_scale_mean:.3f}±{wrapped._last_state_scale_std:.3f} "
                f"g={wrapped._last_grad:.3f} "
                f"lr_r={optimizer.param_groups[0]['lr']:.2e} "
                + (f"lr_l={optimizer.param_groups[1]['lr']:.2e} " if len(optimizer.param_groups) > 1 else "")
                + f"spd={speed:.2f}s mem={mem:.1f}GB",
            )
            t_start = time.time()
            torch.cuda.reset_peak_memory_stats(device)

        # Best checkpoint 저장
        if rank == 0 and wrapped._last_qa < best_qa:
            best_qa = wrapped._last_qa
            _save_vl3_stage2_checkpoint(args.output_dir, "best", global_step, wrapped, optimizer, adapter_config)
            log_rank0(rank, f"[Best @ {global_step}] qa={best_qa:.4f}")

        # 정기 저장
        if args.save_interval > 0 and (global_step % args.save_interval == 0 or global_step == args.max_steps):
            if rank == 0:
                _save_vl3_stage2_checkpoint(args.output_dir, f"ckpt_step_{global_step}", global_step, wrapped, optimizer, adapter_config)

    # 최종 저장
    if rank == 0:
        _save_vl3_stage2_checkpoint(args.output_dir, "final", global_step, wrapped, optimizer, adapter_config)
        log_rank0(rank, f"[VL3 Stage 2 Done] final step={global_step} qa={wrapped._last_qa:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


def train_stage3(args):
    """Stage 3: Router frozen + LoRA only. Stage 2 checkpoint에서 expert+router 로드, LoRA만 학습."""
    from core.vision_projector.temporal_query_router import TemporalQueryRouter

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        rank, world_size = 0, 1
        device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 2 checkpoint 로드
    s2_dir = args.stage3_checkpoint
    if not os.path.isdir(s2_dir):
        raise ValueError(f"Stage 2 checkpoint not found: {s2_dir}")
    with open(os.path.join(s2_dir, "adapter_config.json")) as f:
        s2_config = json.load(f)
    s1_config = s2_config.get("stage1_config", s2_config)

    # 기본 모델 로드
    base_model, processor, resolved_model_path = build_base_model(args, device, rank)
    tokenizer = processor.tokenizer
    image_token_id = base_model.config.image_token_index
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # proj_dim 결정
    proj_dim = base_model.config.hidden_size
    factor = args.patch_size * args.merge_size
    target_size = int(round(float(args.image_size) / float(factor)) * factor)
    hm = target_size // args.patch_size // args.merge_size
    tokens_per_frame = hm * hm
    max_length = args.max_length if args.max_length > 0 else args.num_frames * tokens_per_frame + args.text_budget

    # Expert 로드 + freeze
    expert = SplitDualHeadExpert(
        proj_dim=proj_dim,
        bottleneck=s1_config.get("expert_bottleneck", args.expert_bottleneck),
        kernel_size=s1_config.get("expert_kernel_size", args.expert_kernel_size),
        max_local_gate=s1_config.get("max_local_gate", args.max_local_gate),
        max_state_gate=s1_config.get("max_state_gate", args.max_state_gate),
        pooling_ratio=s1_config.get("pooling_ratio", 1),
        ortho_proj=s1_config.get("ortho_proj", False),
        skip_output_norm=s1_config.get("skip_output_norm", False),
    ).to(device=device, dtype=torch.bfloat16)
    expert.load_state_dict(torch.load(os.path.join(s2_dir, "diff_expert.pt"), map_location=device, weights_only=True))
    for p in expert.parameters():
        p.requires_grad = False
    expert.eval()
    if s1_config.get("video_only", False):
        expert.no_gate = True
    log_rank0(rank, f"Loaded & froze expert from {s2_dir} (no_gate={expert.no_gate})")

    # Router 로드 + freeze
    tq_router = TemporalQueryRouter(
        feat_dim=proj_dim,
        hidden_dim=s2_config.get("router_hidden", 256),
        init_bias=s2_config.get("stage2_init_bias", 0.0),
    ).to(device=device, dtype=torch.bfloat16)
    tq_router.load_state_dict(torch.load(os.path.join(s2_dir, "tq_router.pt"), map_location=device, weights_only=True))
    for p in tq_router.parameters():
        p.requires_grad = False
    tq_router.eval()
    log_rank0(rank, f"Loaded & froze router from {s2_dir}")

    # LoRA 생성
    if args.lora_rank <= 0:
        raise ValueError("Stage 3 requires --lora_rank > 0")
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    # gradient checkpointing + LoRA 호환성: inputs_embeds에 requires_grad 강제
    base_model.enable_input_require_grads()
    lora_params = [p for p in base_model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    log_rank0(rank, f"LoRA created: rank={args.lora_rank} params={n_lora:,}")

    # Stage 3 wrapper (Stage 2와 동일한 모델, router frozen)
    wrapped = VL3Stage2RouterModel(
        model=base_model,
        expert=expert,
        tq_router=tq_router,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
        patch_size=args.patch_size,
        merge_size=args.merge_size,
        align_lambda=args.align_lambda,
    ).to(device=device)
    wrapped._base_model_ref = base_model

    # Optimizer: LoRA만
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lora_lr, "name": "lora"}],
        weight_decay=0.01, betas=(0.9, 0.999),
    )

    # Dataset
    dataset_configs = resolve_dataset_configs(args.dataset_preset)
    if dataset_configs == "mvbench":
        dataset = MVBenchVideoQADataset(
            tokenizer=tokenizer,
            num_frames=args.num_frames,
            max_length=max_length,
            image_size=args.image_size,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
            mvbench_base=getattr(args, "mvbench_base", None),
        )
    else:
        dataset = build_dataset(
            dataset_configs,
            tokenizer=tokenizer,
            num_frames=args.num_frames,
            max_length=max_length,
            image_size=args.image_size,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
        )
    collate = make_collate_fn(pad_token_id)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.data_seed) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    warmup_steps = min(args.global_warmup_steps, max(1, args.max_steps // 10))

    adapter_config = {
        "module_type": "videollama3_udh_v2_stage3_lora",
        "stage2_checkpoint": s2_dir,
        "stage2_config": s2_config,
        "stage1_config": s1_config,
        "base_model_path": resolved_model_path,
        "proj_dim": proj_dim,
        "lora_rank": args.lora_rank,
        "lora_lr": args.lora_lr,
        "align_lambda": args.align_lambda,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "dataset_preset": args.dataset_preset,
    }

    log_rank0(rank, "=== VL3 Stage 3: LoRA Training (Expert+Router frozen) ===")
    log_rank0(rank, f"  stage2_ckpt={s2_dir}")
    log_rank0(rank, f"  lora_rank={args.lora_rank} lora_lr={args.lora_lr:.2e}")
    log_rank0(rank, f"  max_steps={args.max_steps} batch={args.batch_size} grad_accum={args.grad_accum}")

    global_step = 0
    accum_counter = 0
    epoch = 0
    data_iter = None
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    best_qa = float("inf")

    while global_step < args.max_steps:
        if data_iter is None:
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch += 1
            data_iter = iter(loader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = None
            continue

        lr_scale = warmup_cosine_scale(global_step, args.max_steps, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = args.lora_lr * lr_scale

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = wrapped(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, prompt_mask=prompt_mask, labels=labels,
            )
        (loss / args.grad_accum).backward()
        accum_counter += 1

        if accum_counter >= args.grad_accum or global_step == args.max_steps - 1:
            if world_size > 1:
                for p in lora_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            wrapped._last_grad = float(grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

        global_step += 1

        if rank == 0 and global_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            speed = elapsed / args.log_interval
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            log_rank0(rank,
                f"[VL3-S3 Step {global_step}/{args.max_steps}] "
                f"qa={wrapped._last_qa:.4f} "
                f"align={wrapped._last_align:.4f} "
                f"l_sc={wrapped._last_local_scale_mean:.3f}±{wrapped._last_local_scale_std:.3f} "
                f"s_sc={wrapped._last_state_scale_mean:.3f}±{wrapped._last_state_scale_std:.3f} "
                f"g={wrapped._last_grad:.3f} "
                f"lr_l={optimizer.param_groups[0]['lr']:.2e} "
                f"spd={speed:.2f}s mem={mem:.1f}GB",
            )
            t_start = time.time()
            torch.cuda.reset_peak_memory_stats(device)

        if rank == 0 and wrapped._last_qa < best_qa:
            best_qa = wrapped._last_qa
            _save_vl3_stage2_checkpoint(args.output_dir, "best", global_step, wrapped, optimizer, adapter_config)
            log_rank0(rank, f"[Best @ {global_step}] qa={best_qa:.4f}")

        if args.save_interval > 0 and (global_step % args.save_interval == 0 or global_step == args.max_steps):
            if rank == 0:
                _save_vl3_stage2_checkpoint(args.output_dir, f"ckpt_step_{global_step}", global_step, wrapped, optimizer, adapter_config)

    if rank == 0:
        _save_vl3_stage2_checkpoint(args.output_dir, "final", global_step, wrapped, optimizer, adapter_config)
        log_rank0(rank, f"[VL3 Stage 3 Done] final step={global_step} qa={wrapped._last_qa:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VideoLLaMA3-2B + UDH v2 with a unified A/B/C curriculum")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="models/VideoLLaMA3-2B")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--merge_size", type=int, default=2)
    parser.add_argument("--text_budget", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--phase_a_batch_size", type=int, default=0)
    parser.add_argument("--phase_b_batch_size", type=int, default=0)
    parser.add_argument("--phase_c_batch_size", type=int, default=0)
    parser.add_argument("--phase_a_grad_accum", type=int, default=0)
    parser.add_argument("--phase_b_grad_accum", type=int, default=0)
    parser.add_argument("--phase_c_grad_accum", type=int, default=0)
    parser.add_argument("--native_qa_loss", action="store_true",
                        help="Use VL3 native forward path for QA loss (matches eval-time token compression)")
    parser.add_argument("--dataset_preset", type=str, default="full", choices=["full", "udh_capped", "mvbench", "eval_bench", "full+mvbench", "full+mvbench+tc", "full+eval"])
    parser.add_argument("--mvbench_base", type=str, default=None,
                        help="Override MVBench data root (default: data/MVBench relative to repo root)")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--phase_a_steps", type=int, default=1999)
    parser.add_argument("--phase_b_steps", type=int, default=0)
    parser.add_argument("--phase_b_router_ramp_steps", type=int, default=150)
    parser.add_argument("--phase_c_joint_ramp_steps", type=int, default=150)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--global_warmup_steps", type=int, default=100)
    parser.add_argument("--best_ema_beta", type=float, default=0.9)
    parser.add_argument("--best_delta", type=float, default=1e-4)
    parser.add_argument("--best_phase_a_start", type=int, default=200)
    parser.add_argument("--best_min_router_std", type=float, default=0.002)
    parser.add_argument("--expert_lr", type=float, default=1e-3)
    parser.add_argument("--router_lr", type=float, default=1e-3)
    parser.add_argument("--joint_expert_lr", type=float, default=1e-4)
    parser.add_argument("--phase_b_min_router_scale", type=float, default=0.25)
    parser.add_argument("--phase_c_min_expert_scale", type=float, default=0.35)
    parser.add_argument("--phase_c_unfreeze", type=str, default="readout_gates", choices=["gates", "readout", "readout_gates", "all"])
    parser.add_argument("--phase_c_aux_scale", type=float, default=0.30)
    parser.add_argument("--expert_bottleneck", type=int, default=128)
    parser.add_argument("--expert_kernel_size", type=int, default=3)
    parser.add_argument("--max_local_gate", type=float, default=0.35)
    parser.add_argument("--max_state_gate", type=float, default=0.35)
    parser.add_argument("--align_lambda", type=float, default=0.1)
    parser.add_argument("--ortho_proj", action="store_true", default=False)
    parser.add_argument("--video_only", action="store_true", default=False)
    parser.add_argument("--video_only_data", type=str, default="datasets/video_only_stage1.jsonl")
    parser.add_argument("--state_mask_lambda", type=float, default=0.03)
    parser.add_argument("--state_mask_ratio", type=float, default=0.5)
    parser.add_argument("--state_pred_dropout", type=float, default=0.05)
    parser.add_argument("--local_corr_lambda", type=float, default=0.01)
    parser.add_argument("--local_corr_radius", type=int, default=1)
    parser.add_argument("--local_corr_anchors", type=int, default=128)
    parser.add_argument("--local_corr_dropout", type=float, default=0.05)
    parser.add_argument("--router_type", type=str, default="mlp", choices=["mlp", "gated_attn"])
    parser.add_argument("--router_hidden", type=int, default=128)
    parser.add_argument("--init_local", type=float, default=0.5)
    parser.add_argument("--init_state", type=float, default=0.5)
    parser.add_argument("--router_lambda", type=float, default=0.10)
    parser.add_argument("--router_tau", type=float, default=0.5)
    parser.add_argument("--router_margin", type=float, default=0.02)
    parser.add_argument("--router_pattern_lambda", type=float, default=0.10)
    parser.add_argument("--router_var_lambda", type=float, default=0.10)
    parser.add_argument("--entropy_lambda", type=float, default=0.005)
    parser.add_argument("--balance_lambda", type=float, default=0.01)
    parser.add_argument("--balance_target_local", type=float, default=0.5)
    parser.add_argument("--balance_target_state", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--no_liger", action="store_true")
    # Stage 2: TemporalQueryRouter 학습 모드
    parser.add_argument("--stage2", action="store_true", default=False, help="Stage 2 mode: train TemporalQueryRouter only.")
    parser.add_argument("--stage2_checkpoint", type=str, default="", help="Path to Stage 1 checkpoint dir (expert + optional LoRA).")
    parser.add_argument("--stage2_init_bias", type=float, default=0.0, help="Router output bias init. 0.0 → sigmoid(0)=0.5.")
    # Stage 3: Router frozen + LoRA only
    parser.add_argument("--stage3", action="store_true", default=False, help="Stage 3 mode: freeze router, train LoRA only.")
    parser.add_argument("--stage3_checkpoint", type=str, default="", help="Path to Stage 2 checkpoint dir (expert + router).")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--compile_backend", type=str, default="")
    parser.add_argument("--compile_mode", type=str, default="")
    parser.add_argument("--compile_dynamic", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank. 0=disabled.")
    parser.add_argument("--lora_lr", type=float, default=5e-5)

    args = parser.parse_args()
    if args.stage3:
        train_stage3(args)
    elif args.stage2:
        train_stage2(args)
    else:
        train(args)
