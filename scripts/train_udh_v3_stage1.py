"""
UDH v3 Stage1: Improved auxiliary losses for temporal expert.

Changes from v2:
  1. l_corr → Wider soft correspondence (radius=3, 7×7=49 candidates, KL divergence)
  2. s_mask → Token-level masked frame reconstruction (not just spatial mean)
  3. New: Inter-frame relation prediction (change magnitude classification)

Phase A only (no router phases).

  Phase A: expert shaping
    - train expert + auxiliary heads
    - optimize QA + improved auxiliary losses

  Phase B: router burn-in (disabled by default, phase_b_steps=0)

  Phase C: partial joint refine (disabled by default)
    - train router + a small subset of expert readout/gate params
    - keep weak stage1 auxiliary losses active to preserve branch separation
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
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.vision_projector.dual_proj import DualBranchRouter, GatedAttentionDualRouter, SplitDualHeadExpert
import train_dual_proj as v2
from train_udh_v2_stage1 import (
    END_HEADER_ID,
    EOT_ID,
    PAD_ID,
    TOKENS_PER_FRAME,
    TRAIN_DATASETS as UDH_CAPPED_TRAIN_DATASETS,
    build_dataset,
    collate_fn,
    log_rank0,
)
from train_dual_proj_v9_2_6_stage1 import TRAIN_DATASETS as FULL_TRAIN_DATASETS


class UDHv2UnifiedCurriculumModel(nn.Module):
    def __init__(
        self,
        model,
        expert,
        router,
        align_lambda=0.03,
        state_mask_lambda=0.03,
        state_mask_ratio=0.5,
        state_pred_dropout=0.05,
        local_corr_lambda=0.01,
        local_corr_radius=1,
        local_corr_anchors=64,
        local_corr_dropout=0.05,
        router_lambda=0.10,
        router_tau=0.5,
        router_margin=0.02,
        entropy_lambda=0.005,
        balance_lambda=0.01,
        balance_target=(0.5, 0.5),
        router_pattern_lambda=0.10,
        router_var_lambda=0.10,
    ):
        super().__init__()
        self.model = model
        self.expert = expert
        self.router = router

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

        # ── Aux head 1: Token-level masked frame reconstruction ──
        # Reconstructs masked frame tokens from DeltaNet state_h context
        self.token_recon_head = nn.Sequential(
            nn.LayerNorm(bottleneck * 2 + 1),
            nn.Linear(bottleneck * 2 + 1, bottleneck * 2),
            nn.GELU(),
            nn.Linear(bottleneck * 2, proj_dim),
        )
        nn.init.xavier_uniform_(self.token_recon_head[1].weight, gain=0.1)
        nn.init.zeros_(self.token_recon_head[1].bias)
        nn.init.xavier_uniform_(self.token_recon_head[3].weight, gain=0.1)
        nn.init.zeros_(self.token_recon_head[3].bias)

        # ── Aux head 2: Wider soft correspondence (local branch) ──
        self.local_corr_q = nn.Linear(bottleneck, bottleneck)
        self.local_corr_k = nn.Linear(proj_dim, bottleneck)
        nn.init.xavier_uniform_(self.local_corr_q.weight, gain=0.1)
        nn.init.zeros_(self.local_corr_q.bias)
        nn.init.xavier_uniform_(self.local_corr_k.weight, gain=0.1)
        nn.init.zeros_(self.local_corr_k.bias)

        # ── Aux head 3: Inter-frame relation prediction ──
        # Classifies temporal change magnitude between frame pairs
        self.relation_head = nn.Sequential(
            nn.Linear(bottleneck * 3, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 3),  # 3 classes: no change / small / large
        )
        nn.init.xavier_uniform_(self.relation_head[0].weight, gain=0.1)
        nn.init.zeros_(self.relation_head[0].bias)
        nn.init.xavier_uniform_(self.relation_head[2].weight, gain=0.1)
        nn.init.zeros_(self.relation_head[2].bias)

        self._last_phase = "A"
        self._last_total = 0.0
        self._last_qa = 0.0
        self._last_align = 0.0
        self._last_state_mask = 0.0
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
        self._last_relation = 0.0
        self._last_router_pattern = 0.0
        self._last_router_var = 0.0
        self._last_grad = 0.0

    @property
    def aux_params(self):
        params = list(self.token_recon_head.parameters())
        params.extend(self.local_corr_q.parameters())
        params.extend(self.local_corr_k.parameters())
        params.extend(self.relation_head.parameters())
        return params

    def _get_vision_features(self, pixel_values_videos, num_frames):
        bsz = pixel_values_videos.shape[0]
        frames_flat = pixel_values_videos.flatten(0, 1)
        with torch.no_grad():
            vision_out = self.model.model.vision_tower(frames_flat)
            features = vision_out.last_hidden_state[:, 1:, :].float()
        _, n, c = features.shape
        return features.reshape(bsz, num_frames, n, c)

    def _build_stage1_inputs(self, pixel_values_videos, num_frames):
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
            z_pre_pool_flat = features
            z_base_flat = proj.pooling(features)
        n_pre = z_pre_pool_flat.shape[1]
        m, d = z_base_flat.shape[1], z_base_flat.shape[2]
        z_pre_pool = z_pre_pool_flat.reshape(bsz, t, n_pre, d)
        z_base = z_base_flat.reshape(bsz, t, m, d)
        return z_pre_pool, z_base

    def _run_expert(self, z_pre_pool, z_base, use_grad):
        if use_grad:
            return self.expert(z_pre_pool, z_base, return_aux=True)
        with torch.no_grad():
            return self.expert(z_pre_pool, z_base, return_aux=True)

    def _qa_loss(self, projected_flat, input_ids, attention_mask, labels):
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
        return qa_loss

    def _qa_loss_per_sample(self, z_4d, input_ids, attention_mask, labels):
        bsz = z_4d.shape[0]
        z_flat = z_4d.reshape(bsz, -1, z_4d.shape[-1])
        losses = torch.zeros(bsz, device=input_ids.device)
        with torch.inference_mode():
            for i in range(bsz):
                losses[i] = self._qa_loss(
                    z_flat[i:i + 1],
                    input_ids[i:i + 1],
                    attention_mask[i:i + 1],
                    labels[i:i + 1] if labels is not None else None,
                )
        return losses

    @staticmethod
    def _alignment_loss(z_final, z_base):
        return 1.0 - F.cosine_similarity(z_final.float(), z_base.float(), dim=-1).mean()

    @staticmethod
    def _cosine_aux_loss(pred, target):
        pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
        target = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
        return 1.0 - (pred * target).sum(dim=-1).mean()

    def _inter_frame_relation_loss(self, z_pre_pool, aux):
        """Inter-frame relation prediction.

        Classifies temporal change magnitude between random frame pairs.
        Forces DeltaNet to encode "how much changed" between frames.
        3 classes: no change (diff < τ_low), small, large (diff > τ_high).
        """
        local_h = aux.get("local_h")
        if local_h is None:
            return torch.tensor(0.0, device=z_pre_pool.device)
        b, t, s_pre, d = z_pre_pool.shape
        if t <= 2:
            return torch.tensor(0.0, device=z_pre_pool.device)

        # Sample random frame pairs (non-adjacent allowed)
        n_pairs = min(8, t - 1)
        t1_idx = torch.randint(0, t - 1, (n_pairs,), device=z_pre_pool.device)
        # gap 1~4 frames
        gaps = torch.randint(1, min(5, t), (n_pairs,), device=z_pre_pool.device)
        t2_idx = torch.clamp(t1_idx + gaps, max=t - 1)

        # Frame-level pooled DeltaNet hidden (local_h is T-1 frames)
        # Clamp to valid range for local_h
        t1_lh = torch.clamp(t1_idx, max=t - 2)
        t2_lh = torch.clamp(t2_idx - 1, min=0, max=t - 2)
        h1 = local_h[:, t1_lh].mean(dim=2)  # (B, n_pairs, bottleneck)
        h2 = local_h[:, t2_lh].mean(dim=2)

        # Predict relation from DeltaNet hidden
        rel_input = torch.cat([h1, h2, h1 - h2], dim=-1).float()  # (B, n_pairs, bn*3)
        rel_pred = self.relation_head(rel_input)  # (B, n_pairs, 3)

        # Pseudo GT: visual feature change magnitude
        with torch.no_grad():
            v1 = z_pre_pool[:, t1_idx].mean(dim=2).float()  # (B, n_pairs, D)
            v2 = z_pre_pool[:, t2_idx].mean(dim=2).float()
            diff_norm = (v2 - v1).norm(dim=-1)  # (B, n_pairs)
            # Adaptive thresholds: percentiles within batch
            tau_low = diff_norm.quantile(0.33)
            tau_high = diff_norm.quantile(0.67)
            labels = torch.zeros_like(diff_norm, dtype=torch.long)
            labels[diff_norm > tau_low] = 1
            labels[diff_norm > tau_high] = 2

        loss = F.cross_entropy(rel_pred.reshape(-1, 3), labels.reshape(-1))
        return loss

    def _masked_frame_reconstruction_loss(self, z_pre_pool, aux):
        """Token-level masked frame reconstruction.

        Masks random frames and reconstructs their spatial-mean tokens
        from surrounding DeltaNet state context (interpolation of state_h).
        Uses token-level targets instead of just spatial mean.
        """
        state_h = aux["state_h"]  # (B, T, bottleneck)
        b, t, s_pre, d = z_pre_pool.shape
        if t <= 2:
            return torch.tensor(0.0, device=z_pre_pool.device)

        n_mid = t - 2
        n_target = max(1, int(round(self.state_mask_ratio * n_mid)))
        target_idx = torch.randperm(n_mid, device=z_pre_pool.device)[:n_target] + 1
        target_idx, _ = torch.sort(target_idx)

        # Context: interpolate state_h from first and last
        first_h = self.state_pred_dropout(state_h[:, :1]).expand(-1, n_target, -1)
        last_h = self.state_pred_dropout(state_h[:, -1:]).expand(-1, n_target, -1)
        alpha = (target_idx.float() / (t - 1)).view(1, n_target, 1).expand(b, -1, -1)
        alpha = alpha.to(dtype=state_h.dtype)

        bridge_input = torch.cat([first_h, last_h, alpha], dim=-1)  # (B, n_target, bn*2+1)
        pred_tokens = self.token_recon_head(bridge_input)  # (B, n_target, proj_dim)

        # Target: spatial mean of masked frames (token-level average)
        target_tokens = z_pre_pool[:, target_idx].mean(dim=2)  # (B, n_target, proj_dim)

        return self._cosine_aux_loss(pred_tokens, target_tokens)

    def _local_correspondence_loss(self, z_pre_pool, aux):
        """Wider soft correspondence loss.

        Changes from v2:
          - radius=3 (7×7=49 candidates) instead of radius=1 (3×3=9)
          - Soft target (KL divergence with temperature-scaled similarity)
            instead of hard argmax (cross entropy)
          - Captures larger motions and gradual transitions
        """
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

        # Subsample frames to save memory (every other frame pair)
        frame_stride = max(1, (t - 1) // 16)
        frame_idx = list(range(0, t - 1, frame_stride))
        n_frames = len(frame_idx)
        bt = b * n_frames

        prev_tokens = z_pre_pool[:, frame_idx].reshape(bt, s_pre, d).float()
        next_idx = [min(i + 1, t - 1) for i in frame_idx]
        curr_tokens = z_pre_pool[:, next_idx].reshape(bt, s_pre, d).float()
        local_h_sub = aux["local_h"][:, frame_idx].reshape(bt, s_pre, self.expert.bottleneck)

        anchor_count = min(self.local_corr_anchors, s_pre)
        anchor_idx = torch.randperm(s_pre, device=z_pre_pool.device)[:anchor_count]

        prev_a = prev_tokens[:, anchor_idx].detach()
        local_a = local_h_sub[:, anchor_idx]

        # Neighborhood: radius=2 → 5×5 = 25 candidates (memory-efficient)
        radius = self.local_corr_radius
        ksz = 2 * radius + 1
        curr_grid = curr_tokens.reshape(bt, h, h, d).permute(0, 3, 1, 2)
        curr_grid = F.pad(curr_grid, (radius, radius, radius, radius), mode="replicate")
        neigh = F.unfold(curr_grid, kernel_size=ksz, padding=0)
        k2 = ksz * ksz
        neigh = neigh.transpose(1, 2).reshape(bt, s_pre, k2, d)
        curr_a = neigh[:, anchor_idx].detach()
        del curr_grid, neigh, prev_tokens, curr_tokens, local_h_sub

        # Soft target: temperature-scaled similarity distribution
        prev_n = F.normalize(prev_a, dim=-1, eps=1e-6)
        curr_n = F.normalize(curr_a, dim=-1, eps=1e-6)
        pseudo_sim = (prev_n.unsqueeze(2) * curr_n).sum(dim=-1)  # (bt, anchors, k2)
        soft_target = F.softmax(pseudo_sim / 0.1, dim=-1).detach()  # sharper soft target

        # Prediction from DeltaNet hidden
        q = F.normalize(self.local_corr_q(self.local_corr_drop(local_a.float())), dim=-1, eps=1e-6)
        k = F.normalize(self.local_corr_k(curr_a.detach()), dim=-1, eps=1e-6)
        pred_logits = (q.unsqueeze(2) * k).sum(dim=-1) / 0.07  # (bt, anchors, k2)
        pred_log_prob = F.log_softmax(pred_logits, dim=-1)

        # KL divergence: soft target distribution
        loss = F.kl_div(pred_log_prob.reshape(-1, k2), soft_target.reshape(-1, k2),
                        reduction="batchmean")

        # Accuracy: still track hard accuracy for monitoring
        target_idx = pseudo_sim.argmax(dim=-1)
        acc = (pred_logits.argmax(dim=-1) == target_idx).float().mean()
        return loss, acc

    def _get_question_embedding(self, input_ids):
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
                (tokens_i != v2.VIDEO_TOKEN_ID)
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

    @staticmethod
    def _centered_match_loss(pred, target):
        pred = pred.float()
        target = target.detach().float()
        pred_c = pred - pred.mean()
        target_c = target - target.mean()
        return F.mse_loss(pred_c, target_c)

    def _router_mix(
        self,
        z_base,
        local_res,
        state_res,
        input_ids,
        attention_mask,
        labels,
        router_loss_scale=1.0,
    ):
        q_embed = self._get_question_embedding(input_ids)
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

        qa_base_ps = self._qa_loss_per_sample(z_base.detach(), input_ids, attention_mask, labels)
        qa_local_ps = self._qa_loss_per_sample((z_base + local_res).detach(), input_ids, attention_mask, labels)
        qa_state_ps = self._qa_loss_per_sample((z_base + state_res).detach(), input_ids, attention_mask, labels)

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
        pixel_values_videos,
        labels=None,
        num_frames=32,
        phase="A",
        router_loss_scale=1.0,
        aux_loss_scale=1.0,
    ):
        self._last_phase = phase
        z_pre_pool, z_base = self._build_stage1_inputs(pixel_values_videos, num_frames)

        if phase == "A":
            z_temp, _, _, aux = self._run_expert(z_pre_pool, z_base, use_grad=True)
            qa_loss = self._qa_loss(z_temp.reshape(z_temp.shape[0], -1, z_temp.shape[-1]), input_ids, attention_mask, labels)
            align_loss = self._alignment_loss(z_temp, z_base)
            recon_loss = self._masked_frame_reconstruction_loss(z_pre_pool, aux)
            local_corr_loss, local_corr_acc = self._local_correspondence_loss(z_pre_pool, aux)
            relation_loss = self._inter_frame_relation_loss(z_pre_pool, aux)
            total_loss = (
                qa_loss
                + self.align_lambda * align_loss
                + self.state_mask_lambda * recon_loss
                + self.local_corr_lambda * local_corr_loss
                + self.local_corr_lambda * relation_loss
            )

            self._last_total = total_loss.item()
            self._last_qa = qa_loss.item()
            self._last_align = align_loss.item()
            self._last_state_mask = recon_loss.item()
            self._last_local_corr = local_corr_loss.item()
            self._last_local_acc = local_corr_acc.item()
            self._last_relation = relation_loss.item()
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
            router_loss_scale=router_loss_scale,
        )

        recon_loss = torch.tensor(0.0, device=z_base.device)
        local_corr_loss = torch.tensor(0.0, device=z_base.device)
        local_corr_acc = torch.tensor(0.0, device=z_base.device)
        relation_loss = torch.tensor(0.0, device=z_base.device)
        if phase == "C" and aux_loss_scale > 0:
            recon_loss = self._masked_frame_reconstruction_loss(z_pre_pool, aux)
            local_corr_loss, local_corr_acc = self._local_correspondence_loss(z_pre_pool, aux)
            relation_loss = self._inter_frame_relation_loss(z_pre_pool, aux)
            total_loss = total_loss + aux_loss_scale * (
                self.state_mask_lambda * recon_loss
                + self.local_corr_lambda * local_corr_loss
                + self.local_corr_lambda * relation_loss
            )

        self._last_total = total_loss.item()
        self._last_qa = metrics["qa"].item()
        self._last_align = metrics["align"].item()
        self._last_state_mask = recon_loss.item()
        self._last_local_corr = local_corr_loss.item()
        self._last_local_acc = local_corr_acc.item()
        self._last_relation = relation_loss.item()
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


def resolve_dataset_configs(dataset_preset: str):
    if dataset_preset == "full":
        return list(FULL_TRAIN_DATASETS)
    if dataset_preset == "udh_capped":
        return list(UDH_CAPPED_TRAIN_DATASETS)
    raise ValueError(f"Unknown dataset_preset: {dataset_preset!r}")


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


def build_base_model(device, no_liger=False):
    base_model = AutoModelForImageTextToText.from_pretrained(
        "facebook/Perception-LM-1B",
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B")

    if not no_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_mllama

            apply_liger_kernel_to_mllama()
            log_rank0(0, "[Liger] Applied mllama kernel patches")
        except Exception:
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_llama

                apply_liger_kernel_to_llama()
                log_rank0(0, "[Liger] Applied llama kernel patches")
            except Exception:
                pass

    for p in base_model.parameters():
        p.requires_grad = False
    base_model.model.language_model.gradient_checkpointing_enable()
    base_model.model.language_model.config.use_cache = False
    base_model.model.language_model.train()
    return base_model, processor


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
        p.requires_grad = (phase in ("B", "C"))

    for p in wrapped.aux_params:
        p.requires_grad = (phase in ("A", "C"))

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


def cosine_decay_scale(step_in_phase, total_phase_steps):
    if total_phase_steps <= 1:
        return 1.0
    progress = min(1.0, max(0.0, step_in_phase / float(total_phase_steps - 1)))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def warmup_cosine_scale(step, total_steps, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def resolve_phase_batch_size(args, phase):
    value = getattr(args, f"phase_{phase.lower()}_batch_size")
    return value if value and value > 0 else args.batch_size


def resolve_phase_grad_accum(args, phase):
    value = getattr(args, f"phase_{phase.lower()}_grad_accum")
    return value if value and value > 0 else args.grad_accum


def set_group_lrs(optimizer, phase, phase_lr_scale, phase_scale, args):
    lr_map = {
        "expert": 0.0,
        "aux": 0.0,
        "router": 0.0,
    }
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
    if prev_ema is None:
        ema_score = raw_score
    else:
        ema_score = args.best_ema_beta * prev_ema + (1.0 - args.best_ema_beta) * raw_score
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


def build_loader(dataset, batch_size, sampler, shuffle, num_workers):
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
    num_frames = args.num_frames
    max_length = num_frames * TOKENS_PER_FRAME + 512
    warmup_steps = min(args.global_warmup_steps, max(1, args.max_steps // 10))
    phase_c_steps = args.max_steps - args.phase_a_steps - args.phase_b_steps
    if phase_c_steps <= 0:
        raise ValueError("max_steps must be larger than phase_a_steps + phase_b_steps")
    phase_total_steps = {
        "A": args.phase_a_steps,
        "B": args.phase_b_steps,
        "C": phase_c_steps,
    }

    dataset_configs = resolve_dataset_configs(args.dataset_preset)

    log_rank0(rank, "=== UDH v2 Unified Curriculum ===")
    log_rank0(rank, f"  phases=A:{args.phase_a_steps} B:{args.phase_b_steps} C:{phase_c_steps}")
    log_rank0(rank, f"  lrs expert={args.expert_lr:.2e} router={args.router_lr:.2e} joint_expert={args.joint_expert_lr:.2e}")
    log_rank0(rank, f"  phase_c_unfreeze={args.phase_c_unfreeze}")
    log_rank0(
        rank,
        f"  router_reg pattern={args.router_pattern_lambda:.3f} var={args.router_var_lambda:.3f} "
        f"scale_floor(B={args.phase_b_min_router_scale:.2f}, C={args.phase_c_min_expert_scale:.2f})",
    )
    log_rank0(rank, f"  dataset_preset={args.dataset_preset} ({len(dataset_configs)} sources)")
    for phase_name in ("A", "B", "C"):
        phase_bs = resolve_phase_batch_size(args, phase_name)
        phase_ga = resolve_phase_grad_accum(args, phase_name)
        eff_batch = phase_bs * phase_ga * world_size
        log_rank0(rank, f"  phase_{phase_name.lower()} batch={phase_bs}x{phase_ga}x{world_size}={eff_batch}")

    base_model, processor = build_base_model(device, no_liger=args.no_liger)
    proj = base_model.model.multi_modal_projector
    pooling_ratio = proj.pooling.pooling_ratio
    proj_dim = proj.linear_2.out_features

    expert = SplitDualHeadExpert(
        proj_dim=proj_dim,
        bottleneck=args.expert_bottleneck,
        kernel_size=args.expert_kernel_size,
        max_local_gate=args.max_local_gate,
        max_state_gate=args.max_state_gate,
        pooling_ratio=pooling_ratio,
    ).to(device=device, dtype=torch.bfloat16)

    q_dim = base_model.model.get_input_embeddings().weight.shape[1]
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

    wrapped = UDHv2UnifiedCurriculumModel(
        base_model,
        expert,
        router,
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
    ).to(device=device)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(wrapped.expert.parameters()), "lr": 0.0, "name": "expert"},
            {"params": wrapped.aux_params, "lr": 0.0, "name": "aux"},
            {"params": list(wrapped.router.parameters()), "lr": 0.0, "name": "router"},
        ],
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    dataset = build_dataset(dataset_configs, processor, num_frames, max_length)
    phase_loaders = {}
    phase_samplers = {}
    phase_iters = {}
    phase_epochs = {"A": 0, "B": 0, "C": 0}
    for phase_name in ("A", "B", "C"):
        batch_size = resolve_phase_batch_size(args, phase_name)
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.data_seed)
            if world_size > 1 else None
        )
        loader = build_loader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=args.num_workers,
        )
        phase_loaders[phase_name] = loader
        phase_samplers[phase_name] = sampler
        phase_iters[phase_name] = None

    adapter_config = {
        "module_type": "dual_proj_deltanet_udh_v2_dbrouter",
        "stage1_module_type": "dual_proj_deltanet_udh_v2_stage1",
        "curriculum_style": "udh_v2_unified_router_curriculum",
        "num_frames": num_frames,
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
        "use_llm_lora": False,
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values_videos = batch["pixel_values_videos"].to(device=device, dtype=torch.bfloat16)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                labels=labels,
                num_frames=num_frames,
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
                f"recon={wrapped._last_state_mask:.4f} l_corr={wrapped._last_local_corr:.4f} l_acc={wrapped._last_local_acc:.3f} rel={wrapped._last_relation:.4f} "
                f"w_l={wrapped._last_w_l:.3f} w_s={wrapped._last_w_s:.3f} "
                f"w_l_std={wrapped._last_w_l_std:.3f} w_s_std={wrapped._last_w_s_std:.3f} "
                f"tgt_l={wrapped._last_tgt_l:.3f} tgt_s={wrapped._last_tgt_s:.3f} "
                f"tgt_l_std={wrapped._last_tgt_l_std:.3f} tgt_s_std={wrapped._last_tgt_s_std:.3f} "
                f"qa_b={wrapped._last_qa_base:.3f} qa_l={wrapped._last_qa_local:.3f} qa_s={wrapped._last_qa_state:.3f} "
                f"l_gate={expert._last_local_gate_mean:.4f} s_gate={expert._last_state_gate_mean:.4f} "
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UDH v2 with a unified A/B/C curriculum")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--phase_a_batch_size", type=int, default=0)
    parser.add_argument("--phase_b_batch_size", type=int, default=0)
    parser.add_argument("--phase_c_batch_size", type=int, default=0)
    parser.add_argument("--phase_a_grad_accum", type=int, default=0)
    parser.add_argument("--phase_b_grad_accum", type=int, default=0)
    parser.add_argument("--phase_c_grad_accum", type=int, default=0)
    parser.add_argument("--dataset_preset", type=str, default="full", choices=["full", "udh_capped"])
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--phase_a_steps", type=int, default=2000)
    parser.add_argument("--phase_b_steps", type=int, default=500)
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
    parser.add_argument("--align_lambda", type=float, default=0.03)
    parser.add_argument("--state_mask_lambda", type=float, default=0.03)
    parser.add_argument("--state_mask_ratio", type=float, default=0.5)
    parser.add_argument("--state_pred_dropout", type=float, default=0.05)
    parser.add_argument("--local_corr_lambda", type=float, default=0.01)
    parser.add_argument("--local_corr_radius", type=int, default=2)
    parser.add_argument("--local_corr_anchors", type=int, default=64)
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
    train(parser.parse_args())
