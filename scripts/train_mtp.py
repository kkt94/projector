"""
Train MTP v9: Pre-Projector Temporal Adapter.

v9b: future segment classification + temporal order prediction
     Regression targets (v6-v9a) collapse in EVA-CLIP space → classification.

Usage (v9b contrastive):
  torchrun --nproc_per_node=4 scripts/train_mtp.py \
      --output_dir outputs/mtp_v9b --gate_type frame --pred_type contrastive \
      --segment_size 2 --n_candidates 4 --contrastive_temp 0.1 \
      --order_pairs 2 --order_lambda 0.67 --pred_lambda 0.3 --max_steps 150
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
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.vision_projector.mtp import (
    TemporalAdapter, TokenLevelPredictor, DeltaTokenPredictor,
    SegmentContrastivePredictor, AttentionContrastivePredictor,
    DummyPredictor,
)
from core.vision_projector.hier_module import apply_lora_to_llm

try:
    import decord
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

END_HEADER_ID = 128007
EOT_ID = 128009
VIDEO_TOKEN_ID = 128003
PAD_ID = 128001

# ════════════════════════════════════════════════════════════════
#  Dataset configs (same as HIER)
# ════════════════════════════════════════════════════════════════

TRAIN_DATASETS = [
    {"jsonl": "datasets/PE-Video/pe_video_train.jsonl",
     "video_root": "datasets/PE-Video/train_videos"},
    {"jsonl": "datasets/llava_video_instruction_full.jsonl",
     "video_root": "datasets/llava_video_plm/videos"},
    {"jsonl": "datasets/llava_nextqa_qa.jsonl",
     "video_root": "datasets/llava_video_plm/videos"},
    {"jsonl": "datasets/llava_activitynetqa_qa.jsonl",
     "video_root": "datasets/llava_video_plm/videos"},
    {"jsonl": "datasets/llava_perceptiontest_qa.jsonl",
     "video_root": "datasets/llava_video_plm/videos"},
    {"jsonl": "datasets/multitask/youcook2/train_plm.jsonl",
     "video_root": "datasets/multitask/youcook2/videos"},
    {"jsonl": "datasets/multitask/nextgqa/train_plm.jsonl",
     "video_root": "datasets/multitask/nextgqa/videos"},
    {"jsonl": "datasets/multitask/charades/train_plm.jsonl",
     "video_root": "datasets/multitask/charades/videos"},
    {"jsonl": "datasets/multitask/egoschema/train_plm.jsonl",
     "video_root": "datasets/multitask/egoschema/videos"},
]

TOKENS_PER_FRAME = 256

# ════════════════════════════════════════════════════════════════
#  Dataset (standard PLM format, NO token expansion)
# ════════════════════════════════════════════════════════════════


class VideoQADataset(Dataset):
    def __init__(self, data_path, video_root, processor, num_frames=16,
                 max_length=4608, max_samples=0):
        self.video_root = video_root
        self.processor = processor
        self.num_frames = num_frames
        self.max_length = max_length
        self.samples = []

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
                # With set_bridge("torch"), get_batch returns torch.Tensor
                # Use .numpy() not .asnumpy() (which is decord-native only)
                return [frames[i].numpy() if isinstance(frames, torch.Tensor)
                        else frames[i].asnumpy() for i in range(len(indices))]
            except Exception:
                return None
        return None

    def __getitem__(self, idx):
        item = self.samples[idx]
        for attempt in range(3):
            target = item if attempt == 0 else self.samples[random.randint(0, len(self.samples) - 1)]
            frames = self._load_frames(target["video"])
            if frames is not None:
                item = target
                break
        if frames is None:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames

        q_clean = re.sub(r"<video>\s*", "", item["question"]).strip()
        answer = item["answer"]

        messages = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": q_clean},
        ]}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        text = prompt_text + answer + "<|eot_id|>"

        proc = self.processor(text=text, videos=[frames], return_tensors="pt")
        input_ids = proc["input_ids"].squeeze(0)
        attention_mask = proc["attention_mask"].squeeze(0)
        pixel_values_videos = proc["pixel_values_videos"].squeeze(0)

        # Build labels: mask everything up to and including assistant header
        labels = input_ids.clone()
        eot_positions = (input_ids == EOT_ID).nonzero(as_tuple=True)[0]
        if len(eot_positions) >= 2:
            mask_end = eot_positions[1].item() + 1
            end_header_positions = (input_ids == END_HEADER_ID).nonzero(as_tuple=True)[0]
            for pos in end_header_positions:
                if pos.item() > mask_end:
                    mask_end = pos.item() + 2
                    break
        elif len(eot_positions) >= 1:
            mask_end = eot_positions[0].item() + 1
        else:
            mask_end = 0
        labels[:mask_end] = -100
        labels[input_ids == PAD_ID] = -100

        # Pad/truncate to max_length
        L = input_ids.shape[0]
        if L < self.max_length:
            pad_len = self.max_length - L
            input_ids = F.pad(input_ids, (0, pad_len), value=PAD_ID)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            labels = F.pad(labels, (0, pad_len), value=-100)
        elif L > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "pixel_values_videos": pixel_values_videos, "labels": labels}


def build_dataset(dataset_configs, processor, num_frames, max_length):
    datasets = []
    total = 0
    for cfg in dataset_configs:
        data_path = cfg.get("jsonl") or cfg.get("json")
        if not data_path or not os.path.exists(data_path):
            print(f"[WARN] Skipping missing: {data_path}")
            continue
        ds = VideoQADataset(data_path=data_path, video_root=cfg["video_root"],
                            processor=processor, num_frames=num_frames,
                            max_length=max_length,
                            max_samples=cfg.get("max_samples", 0))
        datasets.append(ds)
        total += len(ds)
    combined = ConcatDataset(datasets)
    print(f"[Dataset] Combined: {total} total samples from {len(datasets)} sources")
    return combined


def collate_fn(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


# ════════════════════════════════════════════════════════════════
#  Model: Pre-projector adapter + Vision Predictor + LoRA
# ════════════════════════════════════════════════════════════════

class PLMWithMTP(nn.Module):
    """
    MTP v6: Pre-projector adapter with token-level prediction.

    Flow: pixels → vision_tower(frozen) → TemporalAdapter(trainable)
                 → projector(frozen) → LLM(frozen + LoRA)

    Training: L_qa + λ * L_pred
      - Adapter enriches vision features before projector (projector normalizes)
      - Pred: full adapter forward on visible frames → predict masked spatial tokens
      - Gradient flows through FiLM/gate (not just temporal transformer)
    Inference: No predictor, only adapter + LoRA.
    """

    def __init__(self, model, adapter, predictor, pred_lambda=0.5, pred_k=32,
                 mask_ratio=0.25, preserve_lambda=0.0, eff_mod_lambda=0.0,
                 eff_mod_target=3.0, pred_type="absolute",
                 gate_entropy_lambda=0.0, preserve_decay_steps=None,
                 order_lambda=0.67, motion_lambda=1.0,
                 motion_gate_lambda=0.5):
        super().__init__()
        self.model = model
        self.adapter = adapter
        self.predictor = predictor
        self.pred_lambda = pred_lambda
        self.pred_k = pred_k
        self.mask_ratio = mask_ratio
        self.preserve_lambda = preserve_lambda
        self._initial_preserve_lambda = preserve_lambda
        self.eff_mod_lambda = eff_mod_lambda
        self.eff_mod_target = eff_mod_target
        self.pred_type = pred_type
        self.gate_entropy_lambda = gate_entropy_lambda
        self.preserve_decay_steps = preserve_decay_steps
        self.order_lambda = order_lambda
        self.motion_lambda = motion_lambda
        self.motion_gate_lambda = motion_gate_lambda
        self._current_step = 0

    @torch.no_grad()
    def _compute_motion_targets(self, pixel_values_videos, num_frames):
        """Compute pixel-level motion targets from raw frames.

        Returns:
            frame_energy: (B, T) per-frame motion magnitude
            spatial_map: (B, T, 1024) per-position motion, globally normalized
        """
        B = pixel_values_videos.shape[0]
        T = num_frames
        # pixel_values_videos: (B, T, 3, 448, 448), normalized mean=0.5 std=0.5
        pixels = pixel_values_videos.float() * 0.5 + 0.5  # → [0, 1]
        gray = pixels.mean(dim=2)  # (B, T, 448, 448)

        # Frame diff (pad first frame with zeros)
        diff = (gray[:, 1:] - gray[:, :-1]).abs()  # (B, T-1, 448, 448)
        diff = F.pad(diff, (0, 0, 0, 0, 1, 0))  # (B, T, 448, 448)

        # No camera motion compensation — camera motion IS temporal signal
        # Previous versions subtracted frame mean which killed almost all signal

        # Frame-level energy
        frame_energy = diff.mean(dim=(-2, -1))  # (B, T)

        # Spatial map: downsample to 32×32 → flatten to 1024
        spatial_map = F.adaptive_avg_pool2d(diff, (32, 32))  # (B, T, 32, 32)
        spatial_map = spatial_map.reshape(B, T, 1024)
        # Global normalization: preserve cross-frame magnitude differences
        # Use per-video global max (not per-frame max)
        global_max = spatial_map.reshape(B, -1).max(dim=-1).values  # (B,)
        global_max = global_max.clamp(min=1e-6).reshape(B, 1, 1)
        spatial_map = spatial_map / global_max

        return frame_energy, spatial_map

    def _compute_motion_gate_loss(self, pixel_values_videos, num_frames):
        """Motion gate loss: KL on gate distribution + modulation ranking.

        Two components:
          1. KL: gate distribution should match motion energy distribution
          2. Ranking: high-motion frames should have larger gate*||scale||
        """
        frame_energy, _ = self._compute_motion_targets(pixel_values_videos, num_frames)

        _ad = self.adapter
        if hasattr(_ad, '_orig_mod'):
            _ad = _ad._orig_mod
        gate_logit = _ad._last_gate_logit_live  # (B, T, 1), WITH grad
        if gate_logit is None:
            return torch.tensor(0.0, device=pixel_values_videos.device)
        gate_logit_flat = gate_logit.squeeze(-1)  # (B, T)

        # 1. KL divergence: gate distribution → motion distribution
        gate_dist = F.log_softmax(gate_logit_flat, dim=-1)
        motion_dist = F.softmax(frame_energy, dim=-1)  # already detached (no_grad)
        kl_loss = F.kl_div(gate_dist, motion_dist, reduction='batchmean')

        # 2. Modulation ranking: high-motion → larger gate*||scale||
        # Margin ranking: for pairs (i,j) where energy[i] > energy[j],
        # modulation[i] should be > modulation[j]
        gate_val = torch.sigmoid(gate_logit_flat)  # (B, T), WITH grad
        # Effective modulation strength ≈ gate (scale is shared across frames)
        # We use gate directly as proxy for modulation strength
        B, T = frame_energy.shape
        rank_loss = torch.tensor(0.0, device=pixel_values_videos.device)
        if T >= 4:
            # Sample pairs: high vs low motion
            n_pairs = min(4, T // 2)
            _, high_idx = frame_energy.topk(n_pairs, dim=-1)  # (B, n_pairs)
            _, low_idx = frame_energy.topk(n_pairs, dim=-1, largest=False)
            high_gate = gate_val.gather(1, high_idx)  # (B, n_pairs)
            low_gate = gate_val.gather(1, low_idx)  # (B, n_pairs)
            # Margin ranking: high_gate should exceed low_gate by margin
            margin = 0.1
            rank_loss = F.relu(margin - (high_gate - low_gate)).mean()

        total_loss = kl_loss + 0.5 * rank_loss

        # Cache for logging
        self._last_motion_energy = frame_energy.mean().item()
        self._last_motion_gate_loss = total_loss.item()
        self._last_rank_loss = rank_loss.item()

        return total_loss

    def _compute_spatial_motion_loss(self, pixel_values_videos, num_frames):
        """Spatial saliency loss + frame-level gate loss."""
        _, spatial_map = self._compute_motion_targets(pixel_values_videos, num_frames)

        _ad = self.adapter
        if hasattr(_ad, '_orig_mod'):
            _ad = _ad._orig_mod

        # Spatial saliency loss
        sal_logit = _ad._last_sal_logit_live  # (B, T, N), WITH grad
        if sal_logit is None:
            zero = torch.tensor(0.0, device=pixel_values_videos.device)
            return zero, zero
        sal_pred = torch.sigmoid(sal_logit)  # (B, T, N)
        spatial_loss = F.smooth_l1_loss(sal_pred, spatial_map.to(sal_pred.dtype))

        # Frame-level gate loss
        frame_energy = spatial_map.mean(dim=-1)  # (B, T)
        gate_logit = _ad._last_gate_logit_live  # (B, T, 1)
        if gate_logit is None:
            gate_loss = torch.tensor(0.0, device=pixel_values_videos.device)
        else:
            gate_logit_flat = gate_logit.squeeze(-1)
            gate_dist = F.log_softmax(gate_logit_flat, dim=-1)
            motion_dist = F.softmax(frame_energy, dim=-1)
            gate_loss = F.kl_div(gate_dist, motion_dist, reduction='batchmean')

        # Cache for logging
        self._last_spatial_loss = spatial_loss.item()
        self._last_spatial_gate_loss = gate_loss.item()
        self._last_mean_saliency = sal_pred.detach().mean().item()

        return spatial_loss, gate_loss

    def _get_features(self, pixel_values_videos, num_frames):
        B = pixel_values_videos.shape[0]
        T = num_frames
        frames_flat = pixel_values_videos.flatten(0, 1)  # (B*T, C, H, W)

        # Vision tower (frozen, no_grad)
        with torch.no_grad():
            vision_out = self.model.model.vision_tower(frames_flat)
            features = vision_out.last_hidden_state[:, 1:, :].float()  # (BT, N=1024, C=1024)

        _, N, C = features.shape
        features_4d = features.reshape(B, T, N, C)  # (B, T, 1024, 1024)

        # Save frozen features with FULL spatial for token-level prediction
        self._frozen_features_4d = features_4d.detach()  # (B, T, N, C)

        # Detach from vision tower, enable grad for adapter backward
        feat_for_adapter = features_4d.detach().requires_grad_(True)

        # Adapter (trainable, PRE-projector) — 1st forward: all frames for QA
        enriched_4d = self.adapter(feat_for_adapter)  # (B, T, 1024, 1024)

        # Projector (frozen, in compute graph for gradient flow to adapter)
        enriched_flat = enriched_4d.reshape(B * T, N, C)
        projected = self.model.model.multi_modal_projector(enriched_flat)  # (BT, 256, 2048)

        M, D = projected.shape[1], projected.shape[2]
        return projected.reshape(B, T * M, D)  # (B, T*256, 2048)

    def _compute_pred_loss(self):
        """
        Prediction loss through full adapter path.

        Modes:
          - "contrastive" (v9b): segment classification + temporal order
          - "delta" (v9a): regression on delta features (collapsed — reference only)
          - "absolute" (v6-v8): regression on absolute features
        """
        frozen = self._frozen_features_4d  # (B, T, N, C)
        B, T, N, C = frozen.shape
        device = frozen.device

        # ── Contrastive mode: classification tasks ──
        if self.pred_type in ("contrastive", "attn_contrastive"):
            # 50% visible for context, 50% for candidate segments
            T_vis = T // 2
            if T_vis < 2:
                return torch.tensor(0.0, device=device)

            vis_features = frozen[:, :T_vis].detach().requires_grad_(True)
            enriched_vis = self.adapter(vis_features)  # (B, T_vis, N, C)

            cls_loss, order_loss, metrics = self.predictor(
                enriched_vis, frozen, T_vis)

            # Cache for logging
            self._last_cls_acc = metrics['cls_acc']
            self._last_order_acc = metrics['order_acc']
            self._last_cls_loss = cls_loss.item()
            self._last_order_loss = order_loss.item()
            self._last_pred_cos = 0.0

            return cls_loss + self.order_lambda * order_loss

        # ── Regression modes (delta / absolute) ──
        num_mask = max(1, int(T * self.mask_ratio))
        vis_idx = torch.arange(0, T - num_mask, device=device)
        mask_idx = torch.arange(T - num_mask, T, device=device)
        T_vis = len(vis_idx)

        if T_vis == 0:
            return torch.tensor(0.0, device=device)

        vis_features = frozen[:, vis_idx].detach().requires_grad_(True)
        enriched_vis = self.adapter(vis_features)

        K = self.pred_k
        spatial_idx = torch.stack([
            torch.randperm(N, device=device)[:K] for _ in range(num_mask)
        ])

        pred = self.predictor(enriched_vis, mask_idx, spatial_idx)

        if self.pred_type == "delta":
            import random
            gap = random.choice([2, 4, 6])
            gap = min(gap, len(vis_idx))
            ref_frame = frozen[:, vis_idx[-gap]]

            target_parts = []
            for i in range(num_mask):
                abs_feat = frozen[:, mask_idx[i], spatial_idx[i]]
                ref_feat = ref_frame[:, spatial_idx[i]]
                target_parts.append(abs_feat - ref_feat)
            target = torch.cat(target_parts, dim=1).detach()

            delta_norm = target.norm(dim=-1)
            total_tokens = target.shape[1]
            k_keep = max(1, int(total_tokens * 0.25))
            avg_delta_norm = delta_norm.mean(dim=0)
            _, topk_idx = avg_delta_norm.topk(k_keep)

            pred_kept = pred[:, topk_idx]
            target_kept = target[:, topk_idx]
            delta_norm_kept = delta_norm[:, topk_idx]

            w = (delta_norm_kept / (delta_norm_kept.mean() + 1e-6)).clamp(max=3.0).detach()
            raw_loss = F.smooth_l1_loss(
                pred_kept, target_kept, reduction='none').mean(dim=-1)
            loss = (raw_loss * w).mean()

            with torch.no_grad():
                cos = F.cosine_similarity(pred_kept, target_kept, dim=-1).mean().item()
                self._last_pred_cos = cos
                self._last_delta_norm = delta_norm_kept.mean().item()
        else:
            target_parts = []
            for i in range(num_mask):
                target_parts.append(frozen[:, mask_idx[i], spatial_idx[i]])
            target = torch.cat(target_parts, dim=1).detach()
            loss = F.smooth_l1_loss(pred, target)
            with torch.no_grad():
                self._last_pred_cos = F.cosine_similarity(
                    pred, target, dim=-1).mean().item()

        return loss

    def _compute_preserve_loss(self):
        """
        Preservation loss: projector(adapter(x)) should stay close to projector(x).
        Uses visible frames from prediction masking to avoid extra computation.
        Prevents adapter from distorting features beyond LLM's adaptation range.
        """
        frozen = self._frozen_features_4d  # (B, T, N, C)
        B, T, N, C = frozen.shape
        device = frozen.device

        # Use first 75% frames as visible (same as prediction)
        num_mask = max(1, int(T * self.mask_ratio))
        T_vis = T - num_mask

        # Clean projector output (no adapter)
        vis_frozen = frozen[:, :T_vis]  # (B, T_vis, N, C)
        with torch.no_grad():
            clean_proj = self.model.model.multi_modal_projector(
                vis_frozen.reshape(B * T_vis, N, C))  # (BT_vis, M, D)
            clean_mean = clean_proj.reshape(B, T_vis, -1, clean_proj.shape[-1]).mean(dim=2)
            clean_mean = clean_mean.detach()  # (B, T_vis, D)

        # Adapted projector output
        vis_for_adapter = vis_frozen.detach().requires_grad_(True)
        enriched_vis = self.adapter(vis_for_adapter)  # (B, T_vis, N, C)
        adapted_proj = self.model.model.multi_modal_projector(
            enriched_vis.reshape(B * T_vis, N, C))
        adapted_mean = adapted_proj.reshape(B, T_vis, -1, adapted_proj.shape[-1]).mean(dim=2)

        return F.mse_loss(adapted_mean, clean_mean)  # raw MSE, weight applied in forward()

    def forward(self, input_ids, attention_mask, pixel_values_videos,
                labels=None, num_frames=16):
        projected_flat = self._get_features(pixel_values_videos, num_frames)

        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        projected_flat = projected_flat.to(inputs_embeds.dtype)

        _, special_video_mask = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=projected_flat)
        inputs_embeds = inputs_embeds.masked_scatter(
            special_video_mask, projected_flat)

        outputs = self.model.model.language_model(
            attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            return_dict=True)

        # QA loss (chunked CE to avoid OOM)
        qa_loss = torch.tensor(0.0, device=input_ids.device)
        pred_loss = torch.tensor(0.0, device=input_ids.device)

        if labels is not None:
            hidden_states = outputs.last_hidden_state
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            total_tokens = 0
            chunk_seq = 1024
            for i in range(0, shift_hidden.shape[1], chunk_seq):
                h_chunk = shift_hidden[:, i:i + chunk_seq, :]
                l_chunk = shift_labels[:, i:i + chunk_seq]
                logits_chunk = self.model.lm_head(h_chunk)
                n_valid = (l_chunk != -100).sum().item()
                if n_valid > 0:
                    chunk_loss = F.cross_entropy(
                        logits_chunk.view(-1, logits_chunk.size(-1)),
                        l_chunk.reshape(-1), ignore_index=-100, reduction="sum")
                    qa_loss = qa_loss + chunk_loss
                    total_tokens += n_valid
            if total_tokens > 0:
                qa_loss = qa_loss / total_tokens

            # Prediction loss (skip for motion-supervised modes)
            if self.pred_type not in ("motion_gate", "motion_spatial"):
                pred_loss = self._compute_pred_loss()

        # Motion supervision losses (v10a/v10b)
        motion_loss = torch.tensor(0.0, device=input_ids.device)
        if labels is not None:
            if self.pred_type == "motion_gate" and self.motion_lambda > 0:
                motion_loss = self.motion_lambda * self._compute_motion_gate_loss(
                    pixel_values_videos, num_frames)
            elif self.pred_type == "motion_spatial":
                spatial_loss, gate_loss = self._compute_spatial_motion_loss(
                    pixel_values_videos, num_frames)
                motion_loss = (self.motion_lambda * spatial_loss
                               + self.motion_gate_lambda * gate_loss)
        self._last_motion_loss = motion_loss.item() if torch.is_tensor(motion_loss) else motion_loss

        # Effective modulation regularization: keep eff_scale near target
        eff_mod_loss = torch.tensor(0.0, device=input_ids.device)
        if self.eff_mod_lambda > 0:
            _ad = self.adapter
            if hasattr(_ad, '_orig_mod'):
                _ad = _ad._orig_mod
            eff_s = getattr(_ad, '_last_eff_scale_norm', 0.0)
            if eff_s > self.eff_mod_target:
                eff_mod_loss = self.eff_mod_lambda * (eff_s - self.eff_mod_target) ** 2

        # Preservation loss with decay schedule
        preserve_loss = torch.tensor(0.0, device=input_ids.device)
        if self._initial_preserve_lambda > 0 and labels is not None:
            if self.preserve_decay_steps and self.preserve_decay_steps > 0:
                decay = max(0.0, 1.0 - self._current_step / self.preserve_decay_steps)
                effective_preserve = self._initial_preserve_lambda * decay
            else:
                effective_preserve = self.preserve_lambda
            self._effective_preserve_lambda = effective_preserve
            if effective_preserve > 0:
                preserve_loss = effective_preserve * self._compute_preserve_loss()

        # Gate entropy regularization (gradient flows through live logit)
        gate_ent_loss = torch.tensor(0.0, device=input_ids.device)
        if self.gate_entropy_lambda > 0 and labels is not None:
            _ad = self.adapter
            if hasattr(_ad, '_orig_mod'):
                _ad = _ad._orig_mod
            gate_logit_live = getattr(_ad, '_last_gate_logit_live', None)
            if gate_logit_live is not None:
                gate_prob = torch.sigmoid(gate_logit_live)
                entropy = -(gate_prob * torch.log(gate_prob + 1e-8)
                            + (1 - gate_prob) * torch.log(1 - gate_prob + 1e-8))
                # Penalize low entropy (saturated gates). Max entropy = ln(2) ≈ 0.693
                gate_ent_loss = self.gate_entropy_lambda * F.relu(0.5 - entropy.mean())

        combined_loss = (qa_loss + self.pred_lambda * pred_loss + motion_loss
                         + preserve_loss + eff_mod_loss + gate_ent_loss)
        self._last_preserve_loss = preserve_loss.item() if torch.is_tensor(preserve_loss) else preserve_loss
        self._last_eff_mod_loss = eff_mod_loss.item() if torch.is_tensor(eff_mod_loss) else eff_mod_loss
        self._last_gate_ent_loss = gate_ent_loss.item() if torch.is_tensor(gate_ent_loss) else gate_ent_loss
        return combined_loss, qa_loss, pred_loss


# ════════════════════════════════════════════════════════════════
#  Training utilities
# ════════════════════════════════════════════════════════════════

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def log_rank0(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def save_checkpoint(output_dir, step, adapter, predictor, lora_params,
                    optimizer, scheduler):
    ckpt_dir = os.path.join(output_dir, f"ckpt_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save adapter
    ad = adapter
    if hasattr(ad, "_orig_mod"):
        ad = ad._orig_mod
    torch.save(ad.state_dict(), os.path.join(ckpt_dir, "adapter.pt"))

    # Save predictor (for resume only, not needed at inference)
    pr = predictor
    if hasattr(pr, "_orig_mod"):
        pr = pr._orig_mod
    torch.save(pr.state_dict(), os.path.join(ckpt_dir, "predictor.pt"))

    # Save LoRA weights
    lora_state = {f"lora_{i}": p.data.cpu() for i, p in enumerate(lora_params)}
    torch.save(lora_state, os.path.join(ckpt_dir, "lora.pt"))

    # Save training state
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, os.path.join(ckpt_dir, "training_state.pt"))


def apply_liger_kernel():
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        apply_liger_kernel_to_llama(
            rope=True, swiglu=True, cross_entropy=True,
            fused_linear_cross_entropy=False, rms_norm=True)
        print("[Liger] Applied kernel patches to Llama")
        return True
    except Exception as e:
        print(f"[Liger] Failed: {e}")
        return False


# ════════════════════════════════════════════════════════════════
#  Training
# ════════════════════════════════════════════════════════════════

def train(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if not args.no_liger:
        apply_liger_kernel()

    num_frames = 16
    max_length = num_frames * TOKENS_PER_FRAME + 512  # 4608
    adapter_lr = args.adapter_lr or 1e-3
    lora_lr = args.lora_lr or 5e-5
    lr = args.lr or adapter_lr  # default lr used for display
    total_steps = args.max_steps or 14800
    warmup_steps = min(500, total_steps // 5)
    batch_size = args.batch_size or 10
    grad_accum = args.grad_accum or 1
    lora_rank = args.lora_rank or 8
    lora_alpha = args.lora_alpha or 16.0
    pred_lambda = args.pred_lambda or 0.5
    adapter_bottleneck = args.adapter_bottleneck or 128
    adapter_layers = args.adapter_layers or 2

    scale_bound = args.scale_bound if args.scale_bound is not None else 0.1
    shift_bound = args.shift_bound  # None = same as scale_bound
    gate_type = args.gate_type or "dot_product"
    mask_ratio = args.mask_ratio or 0.25
    preserve_lambda = args.preserve_lambda or 0.0
    eff_mod_lambda = args.eff_mod_lambda or 0.0
    eff_mod_target = args.eff_mod_target or 3.0
    pred_k = args.pred_k or 32
    pred_type = args.pred_type or "absolute"
    k_ctx = args.k_ctx or 32
    gate_entropy_lambda = args.gate_entropy_lambda or 0.0
    preserve_decay_steps = args.preserve_decay_steps
    lora_delay = args.lora_delay
    if lora_delay is None:
        lora_delay = min(500, total_steps // 3)
    # Contrastive-specific args
    segment_size = args.segment_size or 2
    n_candidates = args.n_candidates or 4
    contrastive_temp = args.contrastive_temp or 0.1
    order_pairs = args.order_pairs or 2
    order_lambda = args.order_lambda if args.order_lambda is not None else 0.67
    # Motion-specific args (v10a/v10b)
    motion_lambda = args.motion_lambda if args.motion_lambda is not None else 1.0
    motion_gate_lambda = args.motion_gate_lambda if args.motion_gate_lambda is not None else 0.5

    eff_batch = batch_size * grad_accum * world_size
    pred_desc = {"absolute": "Absolute Token", "delta": "Delta Token",
                 "contrastive": "Segment Contrastive",
                 "attn_contrastive": "Attention Contrastive",
                 "motion_gate": "Frame Motion Gate",
                 "motion_spatial": "Spatial Motion Saliency"}
    log_rank0(rank, f"=== MTP v10: Pre-Projector Adapter + {pred_desc.get(pred_type, pred_type)} ===")
    log_rank0(rank, f"  adapter: bottleneck={adapter_bottleneck}, layers={adapter_layers}, scale_bound={scale_bound}, shift_bound={shift_bound}, gate={gate_type}")
    if pred_type in ("contrastive", "attn_contrastive"):
        log_rank0(rank, f"  prediction: type={pred_type}, cls_lambda={pred_lambda}, order_lambda={pred_lambda*order_lambda:.2f}, seg_size={segment_size}, n_cands={n_candidates}, temp={contrastive_temp}, order_pairs={order_pairs}, k_ctx={k_ctx}")
    elif pred_type in ("motion_gate", "motion_spatial"):
        log_rank0(rank, f"  motion: type={pred_type}, motion_lambda={motion_lambda}, motion_gate_lambda={motion_gate_lambda}")
    else:
        log_rank0(rank, f"  prediction: type={pred_type}, lambda={pred_lambda}, causal_mask={int(mask_ratio*100)}%, K={pred_k}, k_ctx={k_ctx}")
    if preserve_lambda > 0:
        log_rank0(rank, f"  preserve_lambda={preserve_lambda}, decay_steps={preserve_decay_steps}")
    if gate_entropy_lambda > 0:
        log_rank0(rank, f"  gate_entropy_lambda={gate_entropy_lambda}")
    if eff_mod_lambda > 0:
        log_rank0(rank, f"  eff_mod_lambda={eff_mod_lambda}, target={eff_mod_target}")
    log_rank0(rank, f"  num_frames={num_frames}, max_length={max_length}")
    log_rank0(rank, f"  adapter_lr={adapter_lr}, lora_lr={lora_lr}, lora_delay={lora_delay}")
    log_rank0(rank, f"  lora_rank={lora_rank}, lora_alpha={lora_alpha}")
    log_rank0(rank, f"  batch={batch_size}x{grad_accum}x{world_size} = {eff_batch}")
    log_rank0(rank, f"  total_steps={total_steps}, warmup={warmup_steps}")

    # ── Load model ──
    log_rank0(rank, "Loading PLM-1B...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        "facebook/Perception-LM-1B", dtype=torch.bfloat16, device_map={"": device})
    processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B")

    # Freeze everything initially
    for param in base_model.parameters():
        param.requires_grad = False

    # ── Apply LoRA to LLM ──
    log_rank0(rank, f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})...")
    lora_params, n_replaced = apply_lora_to_llm(
        base_model.model.language_model,
        rank=lora_rank, alpha=lora_alpha,
        target_modules=("q_proj", "v_proj"))
    for p in lora_params:
        p.data = p.data.to(device=device, dtype=torch.bfloat16)
    lora_param_count = sum(p.numel() for p in lora_params)
    log_rank0(rank, f"  LoRA: {n_replaced} modules, {lora_param_count:,} params")

    # ── Create MTP modules ──
    # Infer vision dim from vision tower config
    vision_dim = getattr(base_model.model.vision_tower.config, 'hidden_size', 1024)
    log_rank0(rank, f"  vision_dim={vision_dim}")

    k_motion = args.k_motion or 0
    use_spatial_saliency = (pred_type == "motion_spatial")
    adapter = TemporalAdapter(
        vision_dim=vision_dim, bottleneck=adapter_bottleneck,
        num_heads=4, num_layers=adapter_layers, max_frames=32,
        scale_bound=scale_bound, shift_bound=shift_bound, gate_type=gate_type,
        k_motion=k_motion, spatial_saliency=use_spatial_saliency,
    ).to(device=device, dtype=torch.bfloat16)
    adapter_params = adapter.count_trainable_params()
    motion_str = f", k_motion={k_motion}" if k_motion > 0 else ""
    spatial_str = ", spatial_saliency" if use_spatial_saliency else ""
    log_rank0(rank, f"  Adapter: {adapter_params:,} params (scale={scale_bound}, shift={adapter.shift_bound}, gate={gate_type}{motion_str}{spatial_str})")

    if pred_type in ("motion_gate", "motion_spatial"):
        predictor = DummyPredictor().to(device=device, dtype=torch.bfloat16)
    elif pred_type == "attn_contrastive":
        predictor = AttentionContrastivePredictor(
            vision_dim=vision_dim, bottleneck=adapter_bottleneck,
            num_heads=4, k_ctx=k_ctx, segment_size=segment_size,
            n_candidates=n_candidates, temperature=contrastive_temp,
            order_pairs=order_pairs,
        ).to(device=device, dtype=torch.bfloat16)
    elif pred_type == "contrastive":
        predictor = SegmentContrastivePredictor(
            vision_dim=vision_dim, bottleneck=adapter_bottleneck,
            num_heads=4, segment_size=segment_size, n_candidates=n_candidates,
            temperature=contrastive_temp, order_pairs=order_pairs,
        ).to(device=device, dtype=torch.bfloat16)
    elif pred_type == "delta":
        predictor = DeltaTokenPredictor(
            vision_dim=vision_dim, bottleneck=adapter_bottleneck,
            num_heads=4, max_frames=32, num_spatial=1024, k_ctx=k_ctx,
        ).to(device=device, dtype=torch.bfloat16)
    else:
        predictor = TokenLevelPredictor(
            vision_dim=vision_dim, bottleneck=adapter_bottleneck,
            num_heads=4, max_frames=32, num_spatial=1024,
        ).to(device=device, dtype=torch.bfloat16)
    predictor_params = predictor.count_trainable_params()
    log_rank0(rank, f"  Predictor: {predictor_params:,} params ({pred_type})")

    total_trainable = adapter_params + predictor_params + lora_param_count
    log_rank0(rank, f"  TOTAL trainable: {total_trainable:,} params "
              f"({adapter_params + lora_param_count:,} at inference)")

    # ── Gradient checkpointing ──
    base_model.model.language_model.gradient_checkpointing_enable()

    # ── Wrap model ──
    wrapped = PLMWithMTP(base_model, adapter, predictor,
                         pred_lambda=pred_lambda, pred_k=pred_k,
                         mask_ratio=mask_ratio,
                         preserve_lambda=preserve_lambda,
                         eff_mod_lambda=eff_mod_lambda,
                         eff_mod_target=eff_mod_target,
                         pred_type=pred_type,
                         gate_entropy_lambda=gate_entropy_lambda,
                         preserve_decay_steps=preserve_decay_steps,
                         order_lambda=order_lambda,
                         motion_lambda=motion_lambda,
                         motion_gate_lambda=motion_gate_lambda)

    # ── Dataset ──
    dataset = build_dataset(TRAIN_DATASETS, processor, num_frames, max_length)
    samples_per_epoch = len(dataset)
    steps_per_epoch = samples_per_epoch // eff_batch
    log_rank0(rank, f"  dataset={samples_per_epoch}, ~{steps_per_epoch} steps/epoch")

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
    ) if world_size > 1 else None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=8, pin_memory=True,
        collate_fn=collate_fn, drop_last=True, prefetch_factor=2,
        persistent_workers=True)

    # ── Optimizer: separate LRs for adapter/predictor vs LoRA ──
    adapter_params_list = list(adapter.parameters())
    predictor_params_list = list(predictor.parameters())
    param_groups = [
        {"params": adapter_params_list, "lr": adapter_lr, "name": "adapter"},
        {"params": predictor_params_list, "lr": adapter_lr, "name": "predictor"},
        {"params": lora_params, "lr": lora_lr, "name": "lora"},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # ── Resume ──
    start_step = 0
    if args.resume:
        log_rank0(rank, f"Resuming from {args.resume}")
        state_path = os.path.join(args.resume, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = state["step"]

        adapter_path = os.path.join(args.resume, "adapter.pt")
        if os.path.exists(adapter_path):
            ad = wrapped.adapter
            if hasattr(ad, "_orig_mod"):
                ad = ad._orig_mod
            ad.load_state_dict(torch.load(adapter_path, map_location=device))

        predictor_path = os.path.join(args.resume, "predictor.pt")
        if os.path.exists(predictor_path):
            pr = wrapped.predictor
            if hasattr(pr, "_orig_mod"):
                pr = pr._orig_mod
            pr.load_state_dict(torch.load(predictor_path, map_location=device))

        lora_path = os.path.join(args.resume, "lora.pt")
        if os.path.exists(lora_path):
            lora_state = torch.load(lora_path, map_location=device)
            for i, p in enumerate(lora_params):
                key = f"lora_{i}"
                if key in lora_state:
                    p.data.copy_(lora_state[key].to(p.device))

        log_rank0(rank, f"Resumed from step {start_step}")

    output_dir = args.output_dir or "outputs/mtp_v2"
    os.makedirs(output_dir, exist_ok=True)

    # ── Training loop ──
    log_rank0(rank, f"Starting training from step {start_step}...")
    wrapped.train()
    wrapped.model.model.vision_tower.eval()

    all_trainable = adapter_params_list + predictor_params_list + lora_params
    total_numel = sum(p.numel() for p in all_trainable)
    grad_buffer = torch.zeros(total_numel, device=device, dtype=torch.float32)
    log_rank0(rank, f"Manual gradient sync: {len(all_trainable)} tensors, "
              f"{total_numel:,} params, buffer={total_numel * 4 / 1e6:.1f}MB")

    optimizer.zero_grad()
    global_step = start_step
    micro_step = 0
    epoch = 0
    running_loss = 0.0
    running_qa_loss = 0.0
    running_pred_loss = 0.0
    loss_count = 0
    log_interval = 50
    save_interval = args.save_interval or 3000
    t_start = time.time()

    while global_step < total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dataloader:
            if global_step >= total_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                wrapped._current_step = global_step
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    combined_loss, qa_loss, pred_loss = wrapped(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values_videos=batch["pixel_values_videos"],
                        labels=batch["labels"])
                loss = combined_loss / grad_accum
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[rank {rank}] OOM at step ~{global_step}, skipping batch",
                          flush=True)
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise
            running_loss += combined_loss.item()
            running_qa_loss += qa_loss.item()
            running_pred_loss += pred_loss.item()
            micro_step += 1

            if micro_step % grad_accum == 0:
                if world_size > 1:
                    offset = 0
                    for p in all_trainable:
                        n = p.numel()
                        if p.grad is not None:
                            grad_buffer[offset:offset + n].copy_(
                                p.grad.data.view(-1).float())
                        else:
                            grad_buffer[offset:offset + n].zero_()
                        offset += n
                    dist.all_reduce(grad_buffer, op=dist.ReduceOp.AVG)
                    offset = 0
                    for p in all_trainable:
                        n = p.numel()
                        if p.grad is not None:
                            p.grad.data.copy_(
                                grad_buffer[offset:offset + n].view_as(
                                    p.grad).to(p.grad.dtype))
                        offset += n

                torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)

                # Compute per-component grad norms before zero_grad
                _g = {}
                if (global_step + 1) % log_interval == 0 and rank == 0:
                    _ad = wrapped.adapter
                    if hasattr(_ad, "_orig_mod"):
                        _ad = _ad._orig_mod

                    def _gnorm(params):
                        return sum(
                            p.grad.data.norm().item() ** 2
                            for p in params if p.grad is not None
                        ) ** 0.5

                    _g["down"] = _gnorm(_ad.down.parameters())
                    _g["layers"] = _gnorm(_ad.layers.parameters())
                    if hasattr(_ad, 'gate_linear'):
                        _g["gate"] = _gnorm(list(_ad.gate_linear.parameters()))
                    else:
                        _g["gate"] = _gnorm(
                            list(_ad.gate_proj.parameters()) + [_ad.gate_bias])
                    _g["film"] = _gnorm(
                        list(_ad.film_scale.parameters()) +
                        list(_ad.film_shift.parameters()))
                    if hasattr(_ad, 'spatial_proj'):
                        _g["sal"] = _gnorm(
                            list(_ad.spatial_proj.parameters()) + [_ad.spatial_bias])
                    _g["ad"] = (_g["down"]**2 + _g["layers"]**2 +
                                _g["gate"]**2 + _g["film"]**2 +
                                _g.get("sal", 0)**2) ** 0.5

                    _pr = wrapped.predictor
                    if hasattr(_pr, "_orig_mod"):
                        _pr = _pr._orig_mod
                    _g["pr"] = _gnorm(_pr.parameters())
                    _g["lr"] = _gnorm(lora_params)

                # LoRA delay: zero out LoRA gradients during delay period
                if global_step < lora_delay:
                    for p in lora_params:
                        if p.grad is not None:
                            p.grad.zero_()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                loss_count += 1

                # Black frame sanity check (first step only)
                if global_step == 1 and rank == 0:
                    pv = batch["pixel_values_videos"]
                    pmin, pmax = pv.min().item(), pv.max().item()
                    if pmin == pmax:
                        raise RuntimeError(
                            f"BLACK FRAME BUG! All pixels = {pmin:.4f}. "
                            f"Check _load_frames() .asnumpy() vs .numpy()")
                    print(f"[FRAME CHECK] pixel range: [{pmin:.4f}, {pmax:.4f}] — OK")

                if global_step % log_interval == 0 and rank == 0:
                    avg_loss = running_loss / loss_count
                    avg_qa = running_qa_loss / loss_count
                    avg_pred = running_pred_loss / loss_count

                    _ad = wrapped.adapter
                    if hasattr(_ad, "_orig_mod"):
                        _ad = _ad._orig_mod
                    gate_mean = getattr(_ad, '_last_gate_mean', 0.0)
                    gate_std = getattr(_ad, '_last_gate_std', 0.0)
                    scale_norm = getattr(_ad, '_last_scale_norm', 0.0)
                    shift_norm = getattr(_ad, '_last_shift_norm', 0.0)
                    eff_scale = getattr(_ad, '_last_eff_scale_norm', 0.0)
                    eff_shift = getattr(_ad, '_last_eff_shift_norm', 0.0)
                    pred_cos = getattr(wrapped, '_last_pred_cos', 0.0)

                    lr_ad = optimizer.param_groups[0]["lr"]
                    lr_lr = optimizer.param_groups[2]["lr"]
                    lora_active = "ON" if global_step >= lora_delay else "OFF"
                    mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                    cur_epoch = global_step * eff_batch / samples_per_epoch
                    elapsed = time.time() - t_start
                    steps_per_sec = loss_count / elapsed

                    extra = ""
                    pres = getattr(wrapped, '_last_preserve_loss', 0.0)
                    emod = getattr(wrapped, '_last_eff_mod_loss', 0.0)
                    g_ent = getattr(wrapped, '_last_gate_ent_loss', 0.0)
                    e_pres = getattr(wrapped, '_effective_preserve_lambda', 0.0)
                    if pres > 0:
                        extra += f" pres={pres:.4f}"
                    if e_pres > 0:
                        extra += f" e_pres={e_pres:.4f}"
                    if emod > 0:
                        extra += f" emod={emod:.4f}"
                    if g_ent > 0:
                        extra += f" g_ent={g_ent:.4f}"
                    # Contrastive mode: cls_acc, order_acc
                    cls_a = getattr(wrapped, '_last_cls_acc', None)
                    if cls_a is not None:
                        cls_l = getattr(wrapped, '_last_cls_loss', 0.0)
                        ord_a = getattr(wrapped, '_last_order_acc', 0.5)
                        ord_l = getattr(wrapped, '_last_order_loss', 0.0)
                        extra += (f" cls_l={cls_l:.3f} cls_a={cls_a:.2f}"
                                  f" ord_l={ord_l:.3f} ord_a={ord_a:.2f}")
                    # Delta mode: delta norm
                    d_norm = getattr(wrapped, '_last_delta_norm', 0.0)
                    if d_norm > 0:
                        d_kept = getattr(wrapped, '_last_delta_kept', 0)
                        extra += f" d_n={d_norm:.3f} d_kept={d_kept}"
                    # Motion mode: motion loss, energy, saliency
                    mot_l = getattr(wrapped, '_last_motion_loss', 0.0)
                    if mot_l > 0:
                        mot_e = getattr(wrapped, '_last_motion_energy', 0.0)
                        rnk_l = getattr(wrapped, '_last_rank_loss', 0.0)
                        extra += f" mot_l={mot_l:.4f} mot_e={mot_e:.2e}"
                        if rnk_l > 0:
                            extra += f" rnk={rnk_l:.4f}"
                    sal_l = getattr(wrapped, '_last_spatial_loss', None)
                    if sal_l is not None and sal_l > 0:
                        gat_l = getattr(wrapped, '_last_spatial_gate_loss', 0.0)
                        sal_m = getattr(wrapped, '_last_mean_saliency', 0.0)
                        extra += f" sal_l={sal_l:.4f} gat_l={gat_l:.4f} sal_m={sal_m:.3f}"

                    print(
                        f"[Step {global_step}/{total_steps} ep={cur_epoch:.2f}] "
                        f"loss={avg_loss:.4f} qa={avg_qa:.4f} pred={avg_pred:.4f} "
                        f"lr_ad={lr_ad:.2e} lora={lora_active} "
                        f"gate={gate_mean:.4f}±{gate_std:.4f} "
                        f"scale={scale_norm:.4f} shift={shift_norm:.4f} "
                        f"eff_s={eff_scale:.4f} eff_sh={eff_shift:.4f} "
                        f"cos={pred_cos:.4f}{extra} "
                        f"g_dn={_g.get('down',0):.3f} g_ly={_g.get('layers',0):.3f} "
                        f"g_gt={_g.get('gate',0):.3f} g_fm={_g.get('film',0):.3f} "
                        f"g_pr={_g.get('pr',0):.3f} g_lr={_g.get('lr',0):.3f} "
                        + (f"g_sal={_g['sal']:.3f} " if 'sal' in _g else "") +
                        f"speed={steps_per_sec:.2f}s/s mem={mem_gb:.1f}GB",
                        flush=True)
                    running_loss = 0.0
                    running_qa_loss = 0.0
                    running_pred_loss = 0.0
                    loss_count = 0
                    t_start = time.time()

                if global_step % save_interval == 0 and rank == 0:
                    log_rank0(rank, f"Saving checkpoint at step {global_step}...")
                    save_checkpoint(output_dir, global_step, wrapped.adapter,
                                    wrapped.predictor, lora_params,
                                    optimizer, scheduler)
        epoch += 1

    if rank == 0:
        log_rank0(rank, "Saving final checkpoint...")
        save_checkpoint(output_dir, global_step, wrapped.adapter,
                        wrapped.predictor, lora_params, optimizer, scheduler)
        # Convenience copies
        ad = wrapped.adapter
        if hasattr(ad, "_orig_mod"):
            ad = ad._orig_mod
        torch.save(ad.state_dict(), os.path.join(output_dir, "adapter_final.pt"))
        lora_state = {f"lora_{i}": p.data.cpu() for i, p in enumerate(lora_params)}
        torch.save(lora_state, os.path.join(output_dir, "lora_final.pt"))
        log_rank0(rank, f"Training complete. Output: {output_dir}")

    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train MTP")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Legacy; use --adapter_lr/--lora_lr")
    parser.add_argument("--adapter_lr", type=float, default=None, help="LR for adapter+predictor (default 1e-3)")
    parser.add_argument("--lora_lr", type=float, default=None, help="LR for LoRA params (default 5e-5)")
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--pred_lambda", type=float, default=None)
    parser.add_argument("--adapter_bottleneck", type=int, default=None)
    parser.add_argument("--adapter_layers", type=int, default=None)
    parser.add_argument("--scale_bound", type=float, default=None, help="FiLM scale/shift bound via tanh (default 0.1, 0=unbounded)")
    parser.add_argument("--pred_k", type=int, default=None, help="Spatial tokens per masked frame for prediction (default 32)")
    parser.add_argument("--lora_delay", type=int, default=None, help="Freeze LoRA for first N steps (default min(500, steps//3))")
    parser.add_argument("--gate_type", type=str, default=None, choices=["dot_product", "frame"], help="Gate type (default dot_product)")
    parser.add_argument("--mask_ratio", type=float, default=None, help="Fraction of frames to mask (default 0.25)")
    parser.add_argument("--preserve_lambda", type=float, default=None, help="Preservation loss weight (default 0, disabled)")
    parser.add_argument("--eff_mod_lambda", type=float, default=None, help="Eff modulation reg weight (default 0, disabled)")
    parser.add_argument("--eff_mod_target", type=float, default=None, help="Target eff_scale norm (default 3.0)")
    parser.add_argument("--pred_type", type=str, default=None, choices=["absolute", "delta", "contrastive", "attn_contrastive", "motion_gate", "motion_spatial"], help="Prediction target type")
    parser.add_argument("--k_ctx", type=int, default=None, help="Spatial tokens per visible frame for predictor context (delta mode)")
    parser.add_argument("--shift_bound", type=float, default=None, help="Separate bound for FiLM shift (default: same as scale_bound)")
    parser.add_argument("--preserve_decay_steps", type=int, default=None, help="Steps over which preserve_lambda decays to 0")
    parser.add_argument("--gate_entropy_lambda", type=float, default=None, help="Gate entropy regularization weight (default 0)")
    # Contrastive prediction args (v9b)
    parser.add_argument("--segment_size", type=int, default=None, help="Frames per segment for contrastive (default 2)")
    parser.add_argument("--n_candidates", type=int, default=None, help="Number of candidate segments (default 4)")
    parser.add_argument("--contrastive_temp", type=float, default=None, help="Temperature for contrastive loss (default 0.1)")
    parser.add_argument("--order_pairs", type=int, default=None, help="Number of order prediction pairs (default 2)")
    parser.add_argument("--order_lambda", type=float, default=None, help="Order loss weight relative to cls loss (default 0.67)")
    parser.add_argument("--k_motion", type=int, default=None, help="Motion tokens per frame for adapter (0=disabled, default 0)")
    # Motion supervision args (v10a/v10b)
    parser.add_argument("--motion_lambda", type=float, default=None, help="Motion supervision loss weight (default 1.0)")
    parser.add_argument("--motion_gate_lambda", type=float, default=None, help="Frame-level gate KL loss weight for motion_spatial mode (default 0.5)")
    parser.add_argument("--no_liger", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train(args)


if __name__ == "__main__":
    main()
