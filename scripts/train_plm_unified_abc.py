"""
Unified A/B/C training for PLM-1B + SplitDualHeadExpert + TemporalQueryRouter.

Three phases in one training loop:
  Phase A: Expert-only training with aux losses (video-only, no QA)
  Phase B: Router training with QA loss (expert frozen)
  Phase C: LoRA training with QA loss (expert + router frozen)

Adapted from train_internvl3_unified_abc.py for PLM-1B (facebook/Perception-LM-1B).
Key PLM differences:
  - AutoModelForImageTextToText + AutoProcessor
  - video_token_id = 128003
  - Vision: model.model.vision_model → multi_modal_projector → pooling
  - LLM: model.model.language_model (Llama)
  - proj_dim = 2048
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vision_projector.dual_proj import SplitDualHeadExpert
from core.vision_projector.temporal_query_router import TemporalQueryRouter
from train_udh_v2_unified_curriculum import (
    warmup_cosine_scale,
    log_rank0,
    resolve_dataset_configs,
)
from train_udh_v2_stage1 import (
    END_HEADER_ID,
    EOT_ID,
    PAD_ID,
    TOKENS_PER_FRAME,
    build_dataset,
    collate_fn as plm_qa_collate_fn,
    PLMVideoOnlyDataset,
)
from train_udh_v2_unified_curriculum import build_base_model as _build_base_model_unified


# ════════════════════════════════════════════════════════════════
#  PLM-specific helpers
# ════════════════════════════════════════════════════════════════

VIDEO_TOKEN_ID = 128003


def plm_video_only_collate_fn(batch):
    return {"pixel_values_videos": torch.stack([item["pixel_values_videos"] for item in batch])}


def build_base_model(args, device, rank):
    """Build PLM-1B base model with frozen weights."""
    base_model, processor = _build_base_model_unified(device, no_liger=getattr(args, "no_liger", False))
    return base_model, processor


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


# ════════════════════════════════════════════════════════════════
#  Unified A/B/C Model for PLM-1B
# ════════════════════════════════════════════════════════════════

class PLMUnifiedABCModel(nn.Module):
    """Unified wrapper for Phase A (expert), B (router), C (LoRA) training on PLM-1B.

    Phase A: video-only, aux losses only (s_mask + l_corr).
    Phase B/C: QA dataset, router scales expert enrichment → QA loss.
    """

    def __init__(
        self,
        model,
        expert,
        tq_router,
        processor,
        video_token_id=VIDEO_TOKEN_ID,
        state_mask_lambda=0.03,
        state_mask_ratio=0.5,
        state_pred_dropout=0.05,
        local_corr_lambda=0.01,
        local_corr_radius=1,
        local_corr_anchors=128,
        local_corr_dropout=0.05,
    ):
        super().__init__()
        self.model = model
        self.expert = expert
        self.tq_router = tq_router
        self.processor = processor
        self.video_token_id = int(video_token_id)
        self._lm_inner = None

        self.state_mask_lambda = float(state_mask_lambda)
        self.state_mask_ratio = float(state_mask_ratio)
        self.local_corr_lambda = float(local_corr_lambda)
        self.local_corr_radius = int(local_corr_radius)
        self.local_corr_anchors = int(local_corr_anchors)

        proj_dim = expert.proj_dim
        bottleneck = expert.bottleneck
        self.state_pred_dropout = nn.Dropout(state_pred_dropout) if state_pred_dropout > 0 else nn.Identity()
        self.local_corr_drop = nn.Dropout(local_corr_dropout) if local_corr_dropout > 0 else nn.Identity()

        # Aux heads for Phase A
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

        # Logging caches
        self._last_total = 0.0
        self._last_qa = 0.0
        self._last_qa_base = 0.0
        self._last_state_mask = 0.0
        self._last_state_acc = 0.0
        self._last_local_corr = 0.0
        self._last_local_acc = 0.0
        self._last_local_scale_mean = 0.0
        self._last_state_scale_mean = 0.0
        self._last_grad = 0.0

    @property
    def aux_params(self):
        params = list(self.state_bridge_head.parameters())
        params.extend(self.local_corr_q.parameters())
        params.extend(self.local_corr_k.parameters())
        return params

    def _get_vision_features(self, pixel_values_videos, num_frames):
        """Vision encoder → features (B, T, N, C). Frozen.

        PLM: model.model.vision_model → last_hidden_state[:, 1:, :] (drop CLS).
        """
        bsz = pixel_values_videos.shape[0]
        frames_flat = pixel_values_videos.flatten(0, 1)
        with torch.no_grad():
            vision_out = self.model.model.vision_model(frames_flat)
            features = vision_out.last_hidden_state[:, 1:, :].float()
        _, n, c = features.shape
        return features.reshape(bsz, num_frames, n, c)

    def _build_vision_features(self, pixel_values_videos, num_frames):
        """Vision encoder → projector → pooling → z_base (B, T, S, D). Frozen.

        PLM pipeline: vision_model → multi_modal_projector (linear_1 → gelu → linear_2 → pooling)
        Returns z_base with 256 tokens per frame (after 2×2 pool from 1024).
        """
        vision_4d = self._get_vision_features(pixel_values_videos, num_frames)
        bsz, t, n, c = vision_4d.shape
        proj = self.model.model.multi_modal_projector
        flat = vision_4d.reshape(bsz * t, n, c)
        with torch.no_grad():
            features = flat.permute(1, 0, 2)
            features = proj.linear_1(features)
            features = proj.gelu(features)
            features = proj.linear_2(features)
            features = features.permute(1, 0, 2)
            z_base_flat = proj.pooling(features)
        m, d = z_base_flat.shape[1], z_base_flat.shape[2]
        z_base = z_base_flat.reshape(bsz, t, m, d)
        return z_base

    def _qa_loss(self, projected_flat, input_ids, attention_mask, labels, chunk_seq=1024):
        """QA loss using PLM's language model with chunked CE.

        Uses model.model.get_placeholder_mask for video token scatter.
        """
        with torch.no_grad():
            inputs_embeds = self.model.model.get_input_embeddings()(input_ids).detach()
        projected_flat = projected_flat.to(inputs_embeds.dtype)
        _, special_video_mask = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=projected_flat
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, projected_flat)
        outputs = self.model.model.language_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

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

    @staticmethod
    def _cosine_aux_loss(pred, target):
        pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
        target = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
        return 1.0 - (pred * target).sum(dim=-1).mean()

    def _masked_state_loss(self, aux):
        """Masked state prediction loss for Phase A."""
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
        """Local correspondence loss for Phase A."""
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
        target_idx_t = pseudo_sim.argmax(dim=-1)

        q = F.normalize(self.local_corr_q(self.local_corr_drop(local_a.float())), dim=-1, eps=1e-6)
        k = F.normalize(self.local_corr_k(curr_a.detach()), dim=-1, eps=1e-6)
        logits = (q.unsqueeze(2) * k).sum(dim=-1) / 0.07
        loss = F.cross_entropy(logits.reshape(-1, k2), target_idx_t.reshape(-1))
        acc = (logits.argmax(dim=-1) == target_idx_t).float().mean()
        return loss, acc

    def _get_question_embedding(self, input_ids):
        """Question text embedding mean pool → (B, D).

        PLM: uses model.model.get_input_embeddings() and filters out
        video tokens, pad tokens, and special header/eot tokens.
        """
        with torch.no_grad():
            all_embeds = self.model.model.get_input_embeddings()(input_ids)
        q_embeds = []
        for i in range(input_ids.shape[0]):
            seq = input_ids[i].tolist()
            valid_len = len(seq)
            while valid_len > 0 and seq[valid_len - 1] == PAD_ID:
                valid_len -= 1
            start, end = 0, valid_len
            eot_positions = (input_ids[i] == EOT_ID).nonzero(as_tuple=True)[0].tolist()
            if eot_positions:
                user_end = eot_positions[0]
                header_positions = (input_ids[i] == END_HEADER_ID).nonzero(as_tuple=True)[0].tolist()
                header_positions = [pos for pos in header_positions if pos < user_end]
                if header_positions:
                    start = header_positions[-1] + 1
                end = user_end
            tokens_i = input_ids[i, start:end]
            embeds_i = all_embeds[i, start:end]
            text_mask = (
                (tokens_i != self.video_token_id)
                & (tokens_i != PAD_ID)
                & (tokens_i != END_HEADER_ID)
                & (tokens_i != EOT_ID)
            )
            if text_mask.any():
                q_embeds.append(embeds_i[text_mask].mean(dim=0))
            else:
                q_embeds.append(
                    torch.zeros(all_embeds.shape[-1], device=all_embeds.device, dtype=all_embeds.dtype)
                )
        return torch.stack(q_embeds)

    def forward_phase_a(self, pixel_values_videos, num_frames):
        """Phase A: expert training with aux losses only (video-only)."""
        z_base = self._build_vision_features(pixel_values_videos, num_frames)
        z_pre_pool = z_base  # PLM: z_pre_pool == z_base (post-projector)

        z_temp, local_res, state_res, aux = self.expert(z_pre_pool, z_base, return_aux=True)

        state_mask_loss, state_mask_acc = self._masked_state_loss(aux)
        local_corr_loss, local_corr_acc = self._local_correspondence_loss(z_pre_pool, aux)

        total_loss = (
            self.state_mask_lambda * state_mask_loss
            + self.local_corr_lambda * local_corr_loss
        )

        self._last_total = total_loss.item()
        self._last_qa = 0.0
        self._last_qa_base = 0.0
        self._last_state_mask = state_mask_loss.item()
        self._last_state_acc = state_mask_acc
        self._last_local_corr = local_corr_loss.item()
        self._last_local_acc = local_corr_acc.item() if isinstance(local_corr_acc, (int, float)) else local_corr_acc.item()
        self._last_local_scale_mean = 0.0
        self._last_state_scale_mean = 0.0
        return total_loss

    def forward_phase_bc(self, input_ids, attention_mask, pixel_values_videos, labels, num_frames):
        """Phase B/C: router-scaled enrichment → QA loss (NO normalize)."""
        z_base = self._build_vision_features(pixel_values_videos, num_frames)
        z_pre_pool = z_base

        # Expert (frozen in B/C)
        with torch.no_grad():
            _, local_res, state_res, _ = self.expert(z_pre_pool, z_base, return_aux=True)

        # Question embedding for router
        q_embed = self._get_question_embedding(input_ids)  # (B, D)

        # Frame-level summaries (spatial mean)
        z_base_summary = z_base.mean(dim=2)       # (B, T, D)
        local_summary = local_res.mean(dim=2)      # (B, T, D)
        state_summary = state_res.mean(dim=2)      # (B, T, D)

        # Router → per-frame gates
        local_scale, state_scale = self.tq_router(
            q_embed, z_base_summary, local_summary, state_summary,
        )
        # local_scale: (B, T, 1), state_scale: (B, T, 1)

        # NO normalize — use expert's natural output directly
        ls = local_scale.unsqueeze(-1)   # (B, T, 1, 1)
        ss = state_scale.unsqueeze(-1)   # (B, T, 1, 1)
        z_final = z_base + ls * local_res + ss * state_res

        # QA loss
        qa_loss = self._qa_loss(
            z_final.reshape(z_final.shape[0], -1, z_final.shape[-1]),
            input_ids, attention_mask, labels,
        )

        # QA baseline (z_base only) for monitoring
        with torch.no_grad():
            qa_base = self._qa_loss(
                z_base.reshape(z_base.shape[0], -1, z_base.shape[-1]),
                input_ids, attention_mask, labels,
            )
            self._last_qa_base = qa_base.item()

        total_loss = qa_loss

        self._last_total = total_loss.item()
        self._last_qa = qa_loss.item()
        self._last_state_mask = 0.0
        self._last_state_acc = 0.0
        self._last_local_corr = 0.0
        self._last_local_acc = 0.0
        self._last_local_scale_mean = self.tq_router._last_local_scale_mean
        self._last_state_scale_mean = self.tq_router._last_state_scale_mean
        return total_loss


# ════════════════════════════════════════════════════════════════
#  Checkpointing
# ════════════════════════════════════════════════════════════════

def save_abc_checkpoint(output_dir, tag, step, wrapped, optimizer, adapter_config, phase):
    """Save checkpoint for the unified A/B/C pipeline."""
    ckpt_dir = os.path.join(output_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    inner = wrapped._orig_mod if hasattr(wrapped, "_orig_mod") else wrapped

    # Expert weights (always saved)
    torch.save(inner.expert.state_dict(), os.path.join(ckpt_dir, "diff_expert.pt"))

    # Aux heads
    torch.save(
        {
            "state_bridge_head": inner.state_bridge_head.state_dict(),
            "local_corr_q": inner.local_corr_q.state_dict(),
            "local_corr_k": inner.local_corr_k.state_dict(),
        },
        os.path.join(ckpt_dir, "aux_heads.pt"),
    )

    # Router weights (Phase B onwards)
    if phase in ("B", "C"):
        torch.save(inner.tq_router.state_dict(), os.path.join(ckpt_dir, "tq_router.pt"))

    # LoRA weights (Phase C onwards)
    bm = getattr(inner, "_base_model_ref", None)
    if phase == "C" and bm is not None:
        # PLM: LoRA is on base_model.model.language_model
        lm = bm.model.language_model
        if hasattr(lm, "save_pretrained"):
            lm.save_pretrained(os.path.join(ckpt_dir, "lora"))

    # Config
    adapter_config_save = dict(adapter_config)
    adapter_config_save["saved_step"] = step
    adapter_config_save["saved_phase"] = phase
    with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config_save, f, indent=2)

    # Training state
    torch.save(
        {"step": step, "phase": phase, "optimizer": optimizer.state_dict()},
        os.path.join(ckpt_dir, "training_state.pt"),
    )


# ════════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════════

def train(args):
    # DDP setup
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
    total_steps = args.phase_a_steps + args.phase_b_steps + args.phase_c_steps

    # ── Build base model ──
    base_model, processor = build_base_model(args, device, rank)

    # PLM constants
    proj_dim = 2048  # Llama hidden size
    tokens_per_frame = TOKENS_PER_FRAME  # 256 (1024/4 after 2x2 pool)
    num_frames = args.num_frames
    video_token_id = args.video_token_id
    max_length = args.max_length if args.max_length > 0 else num_frames * tokens_per_frame + args.text_budget

    # ── Expert ──
    expert = SplitDualHeadExpert(
        proj_dim=proj_dim,
        bottleneck=args.expert_bottleneck,
        kernel_size=args.expert_kernel_size,
        max_local_gate=args.max_local_gate,
        max_state_gate=args.max_state_gate,
        pooling_ratio=1,
        ortho_proj=False,
        skip_output_norm=True,  # always True in this script
    ).to(device=device, dtype=torch.bfloat16)
    expert.no_gate = True  # Phase A: raw enrichment, no gate
    n_expert = sum(p.numel() for p in expert.parameters())

    # ── Router ──
    tq_router = TemporalQueryRouter(
        feat_dim=proj_dim,
        hidden_dim=args.router_hidden,
        init_bias=args.router_init_bias,
    ).to(device=device, dtype=torch.bfloat16)
    n_router = sum(p.numel() for p in tq_router.parameters())

    # ── Wrapped model ──
    wrapped = PLMUnifiedABCModel(
        model=base_model,
        expert=expert,
        tq_router=tq_router,
        processor=processor,
        video_token_id=video_token_id,
        state_mask_lambda=args.state_mask_lambda,
        state_mask_ratio=args.state_mask_ratio,
        state_pred_dropout=args.state_pred_dropout,
        local_corr_lambda=args.local_corr_lambda,
        local_corr_radius=args.local_corr_radius,
        local_corr_anchors=args.local_corr_anchors,
        local_corr_dropout=args.local_corr_dropout,
    ).to(device=device)
    wrapped._base_model_ref = None  # set during Phase C

    log_rank0(rank, "=== PLM-1B Unified A/B/C Training ===")
    log_rank0(rank, f"  phases: A={args.phase_a_steps} B={args.phase_b_steps} C={args.phase_c_steps} total={total_steps}")
    log_rank0(rank, f"  lrs: expert={args.expert_lr:.2e} router={args.router_lr:.2e} lora={args.lora_lr:.2e}")
    log_rank0(rank, f"  expert: {n_expert:,} params  router: {n_router:,} params")
    log_rank0(rank, f"  lora_rank={args.lora_rank}  skip_output_norm=True  no_gate=True")
    log_rank0(rank, f"  num_frames={num_frames} image_size={args.image_size} proj_dim={proj_dim}")
    log_rank0(rank, f"  tokens_per_frame={tokens_per_frame} max_length={max_length}")
    log_rank0(rank, f"  video_token_id={video_token_id}")
    log_rank0(rank, f"  batch_size={args.batch_size} grad_accum={args.grad_accum}")

    # ── Datasets ──
    # Phase A: video-only (PLM uses processor-based VideoOnlyDataset)
    video_only_ds = PLMVideoOnlyDataset(
        data_path=args.video_only_data,
        processor=processor,
        num_frames=num_frames,
    )
    # Phase B/C: QA
    dataset_configs = resolve_dataset_configs(args.dataset_preset)
    qa_ds = build_dataset(
        dataset_configs,
        processor=processor,
        num_frames=num_frames,
        max_length=max_length,
    )
    log_rank0(rank, f"  video_only_data: {len(video_only_ds)} samples")
    log_rank0(rank, f"  qa_data: {len(qa_ds)} samples ({args.dataset_preset})")

    # ── Adapter config ──
    adapter_config = {
        "module_type": "plm_unified_abc",
        "base_model_path": "facebook/Perception-LM-1B",
        "proj_dim": proj_dim,
        "expert_bottleneck": args.expert_bottleneck,
        "expert_kernel_size": args.expert_kernel_size,
        "max_local_gate": args.max_local_gate,
        "max_state_gate": args.max_state_gate,
        "skip_output_norm": True,
        "no_gate": True,
        "router_hidden": args.router_hidden,
        "router_init_bias": args.router_init_bias,
        "lora_rank": args.lora_rank,
        "phase_a_steps": args.phase_a_steps,
        "phase_b_steps": args.phase_b_steps,
        "phase_c_steps": args.phase_c_steps,
        "expert_lr": args.expert_lr,
        "router_lr": args.router_lr,
        "lora_lr": args.lora_lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "num_frames": num_frames,
        "image_size": args.image_size,
        "video_token_id": video_token_id,
        "dataset_preset": args.dataset_preset,
        "video_only": True,
    }

    # ═══════════════════════════════════════
    #  Phase A: Expert training (aux only)
    # ═══════════════════════════════════════
    log_rank0(rank, "--- Phase A: Expert training (video-only, aux loss) ---")

    # Trainability: expert + aux heads
    for p in expert.parameters():
        p.requires_grad = True
    for p in wrapped.aux_params:
        p.requires_grad = True
    for p in tq_router.parameters():
        p.requires_grad = False

    # Optimizer for Phase A
    param_groups_a = [
        {"params": list(expert.parameters()), "lr": args.expert_lr, "name": "expert"},
        {"params": wrapped.aux_params, "lr": args.expert_lr, "name": "aux"},
    ]
    optimizer = torch.optim.AdamW(param_groups_a, weight_decay=0.01, betas=(0.9, 0.999))

    # DataLoader for Phase A
    sampler_a = DistributedSampler(video_only_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.data_seed) if world_size > 1 else None
    loader_a = build_loader(video_only_ds, batch_size=args.batch_size, sampler=sampler_a, shuffle=sampler_a is None, num_workers=args.num_workers, collate_fn=plm_video_only_collate_fn)

    global_step = 0
    accum_counter = 0
    epoch = 0
    data_iter = None
    t_start = time.time()
    warmup_a = min(args.warmup_steps, max(1, args.phase_a_steps // 10))
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.phase_a_steps:
        if data_iter is None:
            if sampler_a is not None:
                sampler_a.set_epoch(epoch)
            epoch += 1
            data_iter = iter(loader_a)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = None
            continue

        # LR schedule
        lr_scale = warmup_cosine_scale(global_step, args.phase_a_steps, warmup_a)
        for group in optimizer.param_groups:
            group["lr"] = args.expert_lr * lr_scale

        pixel_values_videos = batch["pixel_values_videos"].to(device=device, dtype=torch.bfloat16)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = wrapped.forward_phase_a(pixel_values_videos, num_frames)
        (loss / args.grad_accum).backward()
        accum_counter += 1

        if accum_counter >= args.grad_accum or global_step == args.phase_a_steps - 1:
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

        # Logging
        if rank == 0 and global_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            speed = elapsed / args.log_interval
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            log_rank0(
                rank,
                f"[PhA Step {global_step}/{args.phase_a_steps}] "
                f"s_mask={wrapped._last_state_mask:.4f} s_acc={wrapped._last_state_acc:.3f} "
                f"l_corr={wrapped._last_local_corr:.4f} l_acc={wrapped._last_local_acc:.3f} "
                f"le_n={expert._last_local_enrich_norm:.3f} "
                f"se_n={expert._last_state_enrich_norm:.3f} "
                f"g={wrapped._last_grad:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"spd={speed:.2f}s mem={mem:.1f}GB",
            )
            t_start = time.time()
            torch.cuda.reset_peak_memory_stats(device)

        # Save interval
        if args.save_interval > 0 and global_step % args.save_interval == 0 and rank == 0:
            save_abc_checkpoint(args.output_dir, f"ckpt_step_{global_step}", global_step, wrapped, optimizer, adapter_config, "A")

    # ── Phase A checkpoint ──
    if rank == 0:
        save_abc_checkpoint(args.output_dir, "phase_a_final", global_step, wrapped, optimizer, adapter_config, "A")
        log_rank0(rank, f"[Phase A Done] step={global_step} s_mask={wrapped._last_state_mask:.4f} l_corr={wrapped._last_local_corr:.4f}")

    # Cleanup Phase A loader
    del loader_a, data_iter
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════
    #  Phase B: Router training (QA loss)
    # ═══════════════════════════════════════
    log_rank0(rank, "--- Phase B: Router training (expert frozen, QA loss) ---")

    # Freeze expert + aux, unfreeze router
    for p in expert.parameters():
        p.requires_grad = False
    expert.eval()
    for p in wrapped.aux_params:
        p.requires_grad = False
    for p in tq_router.parameters():
        p.requires_grad = True

    # Rebuild optimizer for Phase B
    param_groups_b = [
        {"params": list(tq_router.parameters()), "lr": args.router_lr, "name": "tq_router"},
    ]
    optimizer = torch.optim.AdamW(param_groups_b, weight_decay=0.01, betas=(0.9, 0.999))

    # DataLoader for Phase B (QA)
    sampler_b = DistributedSampler(qa_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.data_seed) if world_size > 1 else None
    loader_b = build_loader(qa_ds, batch_size=args.batch_size, sampler=sampler_b, shuffle=sampler_b is None, num_workers=args.num_workers, collate_fn=plm_qa_collate_fn)

    phase_b_step = 0
    accum_counter = 0
    epoch = 0
    data_iter = None
    t_start = time.time()
    warmup_b = min(args.warmup_steps, max(1, args.phase_b_steps // 10))
    optimizer.zero_grad(set_to_none=True)

    while phase_b_step < args.phase_b_steps:
        if data_iter is None:
            if sampler_b is not None:
                sampler_b.set_epoch(epoch)
            epoch += 1
            data_iter = iter(loader_b)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = None
            continue

        lr_scale = warmup_cosine_scale(phase_b_step, args.phase_b_steps, warmup_b)
        for group in optimizer.param_groups:
            group["lr"] = args.router_lr * lr_scale

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        pixel_values_videos = batch["pixel_values_videos"].to(device=device, dtype=torch.bfloat16)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = wrapped.forward_phase_bc(input_ids, attention_mask, pixel_values_videos, labels, num_frames)
        (loss / args.grad_accum).backward()
        accum_counter += 1

        if accum_counter >= args.grad_accum or phase_b_step == args.phase_b_steps - 1:
            if world_size > 1:
                for p in tq_router.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            active_params = [p for p in tq_router.parameters() if p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(active_params, 1.0) if active_params else torch.tensor(0.0, device=device)
            wrapped._last_grad = float(grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

        global_step += 1
        phase_b_step += 1

        if rank == 0 and phase_b_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            speed = elapsed / args.log_interval
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            log_rank0(
                rank,
                f"[PhB Step {phase_b_step}/{args.phase_b_steps}] "
                f"qa={wrapped._last_qa:.4f} qa_b={wrapped._last_qa_base:.4f} "
                f"l_sc={wrapped._last_local_scale_mean:.3f} "
                f"s_sc={wrapped._last_state_scale_mean:.3f} "
                f"g={wrapped._last_grad:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"spd={speed:.2f}s mem={mem:.1f}GB",
            )
            t_start = time.time()
            torch.cuda.reset_peak_memory_stats(device)

        if args.save_interval > 0 and phase_b_step % args.save_interval == 0 and rank == 0:
            save_abc_checkpoint(args.output_dir, f"ckpt_step_{global_step}", global_step, wrapped, optimizer, adapter_config, "B")

    # ── Phase B checkpoint ──
    if rank == 0:
        save_abc_checkpoint(args.output_dir, "phase_b_final", global_step, wrapped, optimizer, adapter_config, "B")
        log_rank0(rank, f"[Phase B Done] step={global_step} qa={wrapped._last_qa:.4f} qa_b={wrapped._last_qa_base:.4f}")

    del loader_b, data_iter
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════
    #  Phase C: LoRA training (QA loss)
    # ═══════════════════════════════════════
    log_rank0(rank, "--- Phase C: LoRA training (expert+router frozen, QA loss) ---")

    if args.lora_rank <= 0:
        raise ValueError("Phase C requires --lora_rank > 0")

    # Freeze router
    for p in tq_router.parameters():
        p.requires_grad = False
    tq_router.eval()

    # Apply LoRA to PLM's LLM (language_model inside the ImageTextToText model)
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # PLM: LoRA goes on base_model.model.language_model (the Llama part)
    base_model.model.language_model = get_peft_model(base_model.model.language_model, lora_config)
    lora_params = [p for p in base_model.model.language_model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    n_total_lm = sum(p.numel() for p in base_model.model.language_model.parameters())
    log_rank0(rank, f"  LoRA applied: rank={args.lora_rank} params={n_lora:,} ({100*n_lora/n_total_lm:.2f}% of LLM)")
    wrapped._base_model_ref = base_model

    # Rebuild optimizer for Phase C
    param_groups_c = [
        {"params": lora_params, "lr": args.lora_lr, "name": "lora"},
    ]
    optimizer = torch.optim.AdamW(param_groups_c, weight_decay=0.01, betas=(0.9, 0.999))

    # DataLoader for Phase C (QA)
    sampler_c = DistributedSampler(qa_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.data_seed + 1) if world_size > 1 else None
    loader_c = build_loader(qa_ds, batch_size=args.batch_size, sampler=sampler_c, shuffle=sampler_c is None, num_workers=args.num_workers, collate_fn=plm_qa_collate_fn)

    phase_c_step = 0
    accum_counter = 0
    epoch = 0
    data_iter = None
    t_start = time.time()
    warmup_c = min(args.warmup_steps, max(1, args.phase_c_steps // 10))
    optimizer.zero_grad(set_to_none=True)

    while phase_c_step < args.phase_c_steps:
        if data_iter is None:
            if sampler_c is not None:
                sampler_c.set_epoch(epoch)
            epoch += 1
            data_iter = iter(loader_c)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = None
            continue

        lr_scale = warmup_cosine_scale(phase_c_step, args.phase_c_steps, warmup_c)
        for group in optimizer.param_groups:
            group["lr"] = args.lora_lr * lr_scale

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        pixel_values_videos = batch["pixel_values_videos"].to(device=device, dtype=torch.bfloat16)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = wrapped.forward_phase_bc(input_ids, attention_mask, pixel_values_videos, labels, num_frames)
        (loss / args.grad_accum).backward()
        accum_counter += 1

        if accum_counter >= args.grad_accum or phase_c_step == args.phase_c_steps - 1:
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
        phase_c_step += 1

        if rank == 0 and phase_c_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            speed = elapsed / args.log_interval
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            log_rank0(
                rank,
                f"[PhC Step {phase_c_step}/{args.phase_c_steps}] "
                f"qa={wrapped._last_qa:.4f} qa_b={wrapped._last_qa_base:.4f} "
                f"l_sc={wrapped._last_local_scale_mean:.3f} "
                f"s_sc={wrapped._last_state_scale_mean:.3f} "
                f"g={wrapped._last_grad:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"spd={speed:.2f}s mem={mem:.1f}GB",
            )
            t_start = time.time()
            torch.cuda.reset_peak_memory_stats(device)

        if args.save_interval > 0 and phase_c_step % args.save_interval == 0 and rank == 0:
            save_abc_checkpoint(args.output_dir, f"ckpt_step_{global_step}", global_step, wrapped, optimizer, adapter_config, "C")

    # ── Phase C final checkpoint ──
    if rank == 0:
        save_abc_checkpoint(args.output_dir, "phase_c_final", global_step, wrapped, optimizer, adapter_config, "C")
        log_rank0(rank, f"[Phase C Done] step={global_step} qa={wrapped._last_qa:.4f} qa_b={wrapped._last_qa_base:.4f}")

    log_rank0(rank, f"=== Training Complete: {global_step} total steps ===")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified A/B/C training for PLM-1B + Expert + Router + LoRA")

    # Output
    parser.add_argument("--output_dir", type=str, required=True)

    # Base model
    parser.add_argument("--no_liger", action="store_true")

    # Phase step counts
    parser.add_argument("--phase_a_steps", type=int, default=800)
    parser.add_argument("--phase_b_steps", type=int, default=800)
    parser.add_argument("--phase_c_steps", type=int, default=600)

    # Learning rates
    parser.add_argument("--expert_lr", type=float, default=1e-3)
    parser.add_argument("--router_lr", type=float, default=1e-3)
    parser.add_argument("--lora_lr", type=float, default=5e-5)

    # Expert config
    parser.add_argument("--expert_bottleneck", type=int, default=128)
    parser.add_argument("--expert_kernel_size", type=int, default=3)
    parser.add_argument("--max_local_gate", type=float, default=0.35)
    parser.add_argument("--max_state_gate", type=float, default=0.35)

    # Router config
    parser.add_argument("--router_hidden", type=int, default=256)
    parser.add_argument("--router_init_bias", type=float, default=0.0)

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=16)

    # Aux loss config (Phase A)
    parser.add_argument("--state_mask_lambda", type=float, default=0.03)
    parser.add_argument("--state_mask_ratio", type=float, default=0.5)
    parser.add_argument("--state_pred_dropout", type=float, default=0.05)
    parser.add_argument("--local_corr_lambda", type=float, default=0.01)
    parser.add_argument("--local_corr_radius", type=int, default=1)
    parser.add_argument("--local_corr_anchors", type=int, default=128)
    parser.add_argument("--local_corr_dropout", type=float, default=0.05)

    # Data
    parser.add_argument("--video_only_data", type=str, default="datasets/video_only_stage1.jsonl")
    parser.add_argument("--dataset_preset", type=str, default="full+eval",
                        choices=["full", "udh_capped", "full+mvbench", "full+mvbench+tc", "full+eval"])
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--video_token_id", type=int, default=128003)
    parser.add_argument("--text_budget", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_seed", type=int, default=42)

    # Training
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()
    train(args)
