"""
Train UDH v2 Stage 1 — Split Dual Head Temporal Expert.

Two fully independent branches, each optimized for its own aux loss:
  - local branch: per-token(1024) conv-only, L_local_corr
  - state branch: spatial-mean DeltaNet(1 pos), L_state_mask

No shared encoder. No gradient interference.
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
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.vision_projector.dual_proj import SplitDualHeadExpert

try:
    import decord
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

END_HEADER_ID = 128007
EOT_ID = 128009
PAD_ID = 128001
TOKENS_PER_FRAME = 256

TRAIN_DATASETS = [
    {"jsonl": "datasets/PE-Video/pe_video_train.jsonl", "video_root": "datasets/PE-Video/train_videos", "max_samples": 20000},
    {"jsonl": "datasets/llava_video_instruction_full.jsonl", "video_root": "datasets/llava_video_plm/videos", "max_samples": 8000},
    {"jsonl": "datasets/llava_nextqa_qa.jsonl", "video_root": "datasets/llava_video_plm/videos"},
    {"jsonl": "datasets/llava_activitynetqa_qa.jsonl", "video_root": "datasets/llava_video_plm/videos"},
    {"jsonl": "datasets/llava_perceptiontest_qa.jsonl", "video_root": "datasets/llava_video_plm/videos"},
    {"jsonl": "datasets/multitask/youcook2/train_plm.jsonl", "video_root": "datasets/multitask/youcook2/videos", "max_samples": 4000},
    {"jsonl": "datasets/multitask/nextgqa/train_plm.jsonl", "video_root": "datasets/multitask/nextgqa/videos"},
    {"jsonl": "datasets/multitask/charades/train_plm.jsonl", "video_root": "datasets/multitask/charades/videos"},
    {"jsonl": "datasets/multitask/egoschema/train_plm.jsonl", "video_root": "datasets/multitask/egoschema/videos"},
]


class VideoQADataset(Dataset):
    def __init__(self, data_path, video_root, processor, num_frames=16, max_length=4608, max_samples=0):
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
                return [frames[i].numpy() if isinstance(frames, torch.Tensor) else frames[i].asnumpy() for i in range(len(indices))]
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
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q_clean}]}]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = prompt_text + item["answer"] + "<|eot_id|>"
        proc = self.processor(text=text, videos=[frames], return_tensors="pt")
        input_ids = proc["input_ids"].squeeze(0)
        attention_mask = proc["attention_mask"].squeeze(0)
        pixel_values_videos = proc["pixel_values_videos"].squeeze(0)
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
            continue
        ds = VideoQADataset(data_path=data_path, video_root=cfg["video_root"], processor=processor,
                            num_frames=num_frames, max_length=max_length, max_samples=cfg.get("max_samples", 0))
        datasets.append(ds)
        total += len(ds)
    combined = ConcatDataset(datasets)
    print(f"[Dataset] Combined: {total} total samples from {len(datasets)} sources")
    return combined


def collate_fn(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


class PLMVideoOnlyDataset(Dataset):
    """Video-only dataset for PLM Stage 1. No text, just pixel_values_videos."""

    def __init__(self, data_path, processor, num_frames=32):
        super().__init__()
        self.processor = processor
        self.num_frames = num_frames
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
        print(f"[PLMVideoOnly] Loaded {len(self.samples)} videos from {data_path}")

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
                return [frames[i].numpy() if isinstance(frames, torch.Tensor) else frames[i].asnumpy() for i in range(len(indices))]
            except Exception:
                return None
        return None

    def __getitem__(self, idx):
        for attempt in range(3):
            target = self.samples[idx] if attempt == 0 else self.samples[random.randint(0, len(self.samples) - 1)]
            frames = self._load_frames(target)
            if frames is not None:
                break
        if frames is None:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
        proc = self.processor(text="", videos=[frames], return_tensors="pt")
        pixel_values_videos = proc["pixel_values_videos"].squeeze(0)
        return {"pixel_values_videos": pixel_values_videos}


class PLMWithUDHExpert(nn.Module):
    def __init__(
        self,
        model,
        expert,
        align_lambda=0.03,
        state_mask_lambda=0.03,
        state_mask_ratio=0.5,
        state_pred_dropout=0.05,
        local_corr_lambda=0.01,
        local_corr_radius=1,
        local_corr_anchors=64,
        local_corr_dropout=0.05,
        local_corr_warmup_steps=200,
    ):
        super().__init__()
        self.model = model
        self.expert = expert
        self.align_lambda = align_lambda
        self.state_mask_lambda = state_mask_lambda
        self.state_mask_ratio = state_mask_ratio
        self.local_corr_lambda = local_corr_lambda
        self.local_corr_radius = local_corr_radius
        self.local_corr_anchors = local_corr_anchors
        self.local_corr_warmup_steps = int(local_corr_warmup_steps)

        proj_dim = expert.proj_dim
        bn = expert.bottleneck
        self.state_pred_dropout = nn.Dropout(state_pred_dropout) if state_pred_dropout > 0 else nn.Identity()
        self.local_corr_drop = nn.Dropout(local_corr_dropout) if local_corr_dropout > 0 else nn.Identity()

        # State mask head: [first_h, last_h, alpha] → pred state
        self.state_bridge_head = nn.Sequential(
            nn.LayerNorm(bn * 2 + 1),
            nn.Linear(bn * 2 + 1, bn),
            nn.GELU(),
            nn.Linear(bn, proj_dim),
        )
        nn.init.xavier_uniform_(self.state_bridge_head[1].weight, gain=0.1)
        nn.init.zeros_(self.state_bridge_head[1].bias)
        nn.init.xavier_uniform_(self.state_bridge_head[3].weight, gain=0.1)
        nn.init.zeros_(self.state_bridge_head[3].bias)

        # Local corr heads
        self.local_corr_q = nn.Linear(bn, bn)
        self.local_corr_k = nn.Linear(proj_dim, bn)
        nn.init.xavier_uniform_(self.local_corr_q.weight, gain=0.1)
        nn.init.zeros_(self.local_corr_q.bias)
        nn.init.xavier_uniform_(self.local_corr_k.weight, gain=0.1)
        nn.init.zeros_(self.local_corr_k.bias)

        self._last_qa_loss = 0.0
        self._last_align_loss = 0.0
        self._last_state_mask = 0.0
        self._last_local_corr = 0.0
        self._last_local_acc = 0.0

    def _get_vision_features(self, pixel_values_videos, num_frames):
        bsz = pixel_values_videos.shape[0]
        frames_flat = pixel_values_videos.flatten(0, 1)
        with torch.no_grad():
            vision_out = self.model.model.vision_tower(frames_flat)
            features = vision_out.last_hidden_state[:, 1:, :].float()
        _, n, c = features.shape
        return features.reshape(bsz, num_frames, n, c)

    def _qa_loss(self, projected_flat, input_ids, attention_mask, labels):
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        projected_flat = projected_flat.to(inputs_embeds.dtype)
        _, special_video_mask = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=projected_flat)
        inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, projected_flat)
        outputs = self.model.model.language_model(attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=True)
        qa_loss = torch.tensor(0.0, device=input_ids.device)
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
                    chunk_loss = F.cross_entropy(logits_chunk.view(-1, logits_chunk.size(-1)), l_chunk.reshape(-1), ignore_index=-100, reduction="sum")
                    qa_loss = qa_loss + chunk_loss
                    total_tokens += n_valid
            if total_tokens > 0:
                qa_loss = qa_loss / total_tokens
        return qa_loss

    @staticmethod
    def _alignment_loss(z_final, z_base):
        return 1.0 - F.cosine_similarity(z_final.float(), z_base.float(), dim=-1).mean()

    @staticmethod
    def _cosine_aux_loss(pred, target):
        pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
        target = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
        return 1.0 - (pred * target).sum(dim=-1).mean()

    def _masked_state_loss(self, aux):
        state_h = aux["state_h"]  # (B, T, 128) — state branch DeltaNet output
        state = aux["state"]      # (B, T, 2048) — base_pool
        t = state.shape[1]
        if t <= 2:
            return torch.tensor(0.0, device=state.device)

        n_mid = t - 2
        n_target = max(1, int(round(self.state_mask_ratio * n_mid)))
        target_idx = torch.randperm(n_mid, device=state.device)[:n_target] + 1
        target_idx, _ = torch.sort(target_idx)

        # state_h is T (full frames from state DeltaNet); use frame 0 and frame T-1
        first_h = self.state_pred_dropout(state_h[:, :1]).expand(-1, n_target, -1)
        last_h = self.state_pred_dropout(state_h[:, -1:]).expand(-1, n_target, -1)
        alpha = (target_idx.float() / (t - 1)).view(1, n_target, 1).expand(state.shape[0], -1, -1).to(dtype=state.dtype)

        bridge_input = torch.cat([first_h, last_h, alpha], dim=-1)
        pred_state = self.state_bridge_head(bridge_input)
        target_state = state[:, target_idx]
        return self._cosine_aux_loss(pred_state, target_state)

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

    def forward(self, input_ids, attention_mask, pixel_values_videos, labels=None, num_frames=32, global_step=0, **kwargs):
        vision_4d = self._get_vision_features(pixel_values_videos, num_frames)
        b, t, n, c = vision_4d.shape
        proj = self.model.model.multi_modal_projector
        flat = vision_4d.reshape(b * t, n, c)
        with torch.no_grad():
            features = flat.permute(1, 0, 2)
            features = proj.linear_1(features)
            features = proj.gelu(features)
            features = proj.linear_2(features)
            features = features.permute(1, 0, 2)
            z_base_flat = proj.pooling(features)
        m, d = z_base_flat.shape[1], z_base_flat.shape[2]
        z_base = z_base_flat.reshape(b, t, m, d)

        z_temp, local_res, state_res, aux = self.expert(z_base, z_base, return_aux=True)
        projected_flat = z_temp.reshape(b, t * m, d)
        qa_loss = self._qa_loss(projected_flat, input_ids, attention_mask, labels)
        align_loss = self._alignment_loss(z_temp, z_base) if self.align_lambda > 0 else torch.tensor(0.0, device=qa_loss.device)
        state_mask_loss = self._masked_state_loss(aux) if self.state_mask_lambda > 0 else torch.tensor(0.0, device=qa_loss.device)
        local_corr_loss, local_corr_acc = self._local_correspondence_loss(z_base, aux) if self.local_corr_lambda > 0 else (
            torch.tensor(0.0, device=qa_loss.device), torch.tensor(0.0, device=qa_loss.device))

        corr_scale = min(1.0, max(0.0, (global_step - self.local_corr_warmup_steps) / max(1, self.local_corr_warmup_steps))) if self.local_corr_warmup_steps > 0 else 1.0
        total_loss = (
            qa_loss
            + self.align_lambda * align_loss
            + self.state_mask_lambda * state_mask_loss
            + (corr_scale * self.local_corr_lambda) * local_corr_loss
        )

        self._last_qa_loss = qa_loss.item()
        self._last_align_loss = align_loss.item()
        self._last_state_mask = state_mask_loss.item()
        self._last_local_corr = local_corr_loss.item()
        self._last_local_acc = local_corr_acc.item()
        return total_loss


def save_checkpoint(output_dir, step, expert, optimizer, scheduler, adapter_config):
    ckpt_dir = os.path.join(output_dir, f"ckpt_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    exp = expert._orig_mod if hasattr(expert, "_orig_mod") else expert
    torch.save(exp.state_dict(), os.path.join(ckpt_dir, "diff_expert.pt"))
    with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, os.path.join(ckpt_dir, "training_state.pt"))


def log_rank0(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def train(args):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)
    num_frames = args.num_frames
    max_length = num_frames * TOKENS_PER_FRAME + 512
    warmup_steps = min(50, args.max_steps // 5)
    eff_batch = args.batch_size * args.grad_accum * world_size

    log_rank0(rank, "=== UDH v2 Stage1: Split Dual Head (local DeltaNet + state DeltaNet) ===")
    log_rank0(rank, f"  batch={args.batch_size}x{args.grad_accum}x{world_size}={eff_batch}")
    log_rank0(rank, f"  bottleneck={args.expert_bottleneck}, kernel={args.expert_kernel_size}")
    log_rank0(rank, f"  max_local_gate={args.max_local_gate}, max_state_gate={args.max_state_gate}")
    log_rank0(rank, f"  local_corr_lambda={args.local_corr_lambda}, state_mask_lambda={args.state_mask_lambda}")
    log_rank0(rank, f"  local_corr_warmup={args.local_corr_warmup_steps}")

    base_model = AutoModelForImageTextToText.from_pretrained(
        "facebook/Perception-LM-1B", dtype=torch.bfloat16, device_map={"": device}
    )
    processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B")

    if not args.no_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama()
            log_rank0(rank, "[Liger] Applied kernel patches (post-load)")
        except ImportError:
            pass
    for p in base_model.parameters():
        p.requires_grad = False

    pooling_ratio = 1  # Use post-pool z_base (256 tokens) for both branches
    expert = SplitDualHeadExpert(
        proj_dim=base_model.model.multi_modal_projector.linear_2.out_features,
        bottleneck=args.expert_bottleneck,
        kernel_size=args.expert_kernel_size,
        max_local_gate=args.max_local_gate,
        max_state_gate=args.max_state_gate,
        pooling_ratio=pooling_ratio,
    ).to(device=device, dtype=torch.bfloat16)

    wrapped = PLMWithUDHExpert(
        base_model,
        expert,
        align_lambda=args.align_lambda,
        state_mask_lambda=args.state_mask_lambda,
        state_mask_ratio=args.state_mask_ratio,
        state_pred_dropout=args.state_pred_dropout,
        local_corr_lambda=args.local_corr_lambda,
        local_corr_radius=args.local_corr_radius,
        local_corr_anchors=args.local_corr_anchors,
        local_corr_dropout=args.local_corr_dropout,
        local_corr_warmup_steps=args.local_corr_warmup_steps,
    ).to(device=device)

    trainable_params = [p for p in wrapped.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": trainable_params, "lr": args.adapter_lr}], weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.max_steps)

    dataset = build_dataset(TRAIN_DATASETS, processor, num_frames, max_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True, persistent_workers=True)

    base_model.model.language_model.gradient_checkpointing_enable()
    base_model.model.language_model.train()

    adapter_config = {
        "module_type": "dual_proj_deltanet_udh_v2_stage1",
        "version": "udh_v2_stage1",
        "num_frames": num_frames,
        "proj_dim": base_model.model.multi_modal_projector.linear_2.out_features,
        "expert_bottleneck": args.expert_bottleneck,
        "expert_kernel_size": args.expert_kernel_size,
        "max_local_gate": args.max_local_gate,
        "max_state_gate": args.max_state_gate,
        "pooling_ratio": pooling_ratio,
        "align_lambda": args.align_lambda,
        "state_mask_lambda": args.state_mask_lambda,
        "state_mask_ratio": args.state_mask_ratio,
        "state_pred_dropout": args.state_pred_dropout,
        "local_corr_lambda": args.local_corr_lambda,
        "local_corr_radius": args.local_corr_radius,
        "local_corr_anchors": args.local_corr_anchors,
        "local_corr_dropout": args.local_corr_dropout,
        "local_corr_warmup_steps": args.local_corr_warmup_steps,
        "use_llm_lora": False,
    }

    global_step = 0
    running_loss = 0.0
    loss_count = 0
    t_start = time.time()
    epoch = 0

    while global_step < args.max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= args.max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = wrapped(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values_videos=batch["pixel_values_videos"],
                    labels=batch["labels"],
                    num_frames=num_frames,
                    global_step=global_step,
                )
                loss = loss / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0 or global_step == args.max_steps - 1:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * args.grad_accum
            loss_count += 1
            global_step += 1

            if rank == 0 and global_step % args.log_interval == 0:
                avg_loss = running_loss / max(loss_count, 1)
                elapsed = time.time() - t_start
                speed = elapsed / args.log_interval
                mem = torch.cuda.max_memory_allocated(device) / 1e9
                print(
                    f"[Step {global_step}/{args.max_steps}] "
                    f"loss={avg_loss:.4f} qa={wrapped._last_qa_loss:.4f} "
                    f"align={wrapped._last_align_loss:.4f} "
                    f"l_corr={wrapped._last_local_corr:.4f} l_acc={wrapped._last_local_acc:.3f} "
                    f"s_mask={wrapped._last_state_mask:.4f} "
                    f"diff_n={expert._last_diff_norm:.3f} l_hid={expert._last_local_hidden_norm:.3f} s_hid={expert._last_state_hidden_norm:.3f} "
                    f"l_gate={expert._last_local_gate_mean:.4f} s_gate={expert._last_state_gate_mean:.4f} "
                    f"l_enrich={expert._last_local_enrich_norm:.3f} s_enrich={expert._last_state_enrich_norm:.3f} "
                    f"speed={speed:.2f}s/s mem={mem:.1f}GB",
                    flush=True,
                )
                running_loss = 0.0
                loss_count = 0
                t_start = time.time()
                torch.cuda.reset_peak_memory_stats(device)

            if global_step % args.save_interval == 0 or global_step == args.max_steps:
                if rank == 0:
                    save_checkpoint(args.output_dir, global_step, expert, optimizer, scheduler, adapter_config)

    if dist.is_initialized():
        dist.destroy_process_group()


def build_parser():
    p = argparse.ArgumentParser(description="Train UDH v1 Stage 1")
    p.add_argument("--output_dir", type=str, default="outputs/udh_v2_stage1")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--expert_bottleneck", type=int, default=128)
    p.add_argument("--expert_kernel_size", type=int, default=3)
    p.add_argument("--max_local_gate", type=float, default=0.35)
    p.add_argument("--max_state_gate", type=float, default=0.35)
    p.add_argument("--align_lambda", type=float, default=0.03)
    p.add_argument("--state_mask_lambda", type=float, default=0.03)
    p.add_argument("--state_mask_ratio", type=float, default=0.5)
    p.add_argument("--state_pred_dropout", type=float, default=0.05)
    p.add_argument("--local_corr_lambda", type=float, default=0.01)
    p.add_argument("--local_corr_radius", type=int, default=1)
    p.add_argument("--local_corr_anchors", type=int, default=128)
    p.add_argument("--local_corr_dropout", type=float, default=0.05)
    p.add_argument("--local_corr_warmup_steps", type=int, default=200)
    p.add_argument("--adapter_lr", type=float, default=1e-3)
    p.add_argument("--no_liger", action="store_true")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
