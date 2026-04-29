"""
MTP v10: Pre-Projector Temporal Adapter

v10: Attention-pooled contrastive (no mean-pooling) + motion-augmented adapter
v9b: Future segment classification + temporal order prediction (mean-pool → random accuracy)
v9a/v6-v8: Regression targets (collapsed in EVA-CLIP space)

Architecture:
  Vision(frozen) → TemporalAdapter(trainable, optional motion tokens)
                 → Projector(frozen) → LLM(frozen + LoRA ~0.85M)

FiLM modulation:
  1. Mean-pool → bottleneck + (optional) top-K motion tokens → temporal transformer
  2. FiLM: scale = scale_bound * tanh(linear(h)), shift = shift_bound * tanh(linear(h))
  3. Gate: dot_product(x, h) or frame-level sigmoid(linear(h))
  4. enriched = (1 + gate * scale) * x + gate * shift

Prediction (training only, discarded at inference):
  v10 attn_contrastive: spatial token sampling + attention pooling (no mean-pool)
  v9b contrastive: segment classification (mean-pooled → failed)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAdapter(nn.Module):
    """
    Pre-projector adapter: FiLM from temporal context with configurable gate.

    Gate types:
      - "dot_product" (v6): gate = sigmoid(dot(proj(x), h) / sqrt(d) + bias)
      - "frame" (v7): gate = sigmoid(linear(h))

    Input:  (B, T, N=1024, C=1024) vision features
    Output: (B, T, N=1024, C=1024) enriched features
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        bottleneck: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_frames: int = 32,
        scale_bound: float = 0.1,
        shift_bound: float = None,
        gate_type: str = "dot_product",
        k_motion: int = 0,
        spatial_saliency: bool = False,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.bottleneck = bottleneck
        self.scale_bound = scale_bound
        self.shift_bound = shift_bound if shift_bound is not None else scale_bound
        self.gate_type = gate_type
        self.k_motion = k_motion
        self.spatial_saliency = spatial_saliency

        # ── Temporal context (frame-level) ──
        self.down = nn.Linear(vision_dim, bottleneck)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, bottleneck))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Motion tokens (approach B: frame diff as input) ──
        if k_motion > 0:
            self.motion_down = nn.Linear(vision_dim, bottleneck)
            # Type embedding to distinguish motion tokens from appearance tokens
            self.motion_type_embed = nn.Parameter(
                torch.zeros(1, 1, 1, bottleneck))
            nn.init.trunc_normal_(self.motion_type_embed, std=0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=bottleneck,
                nhead=num_heads,
                dim_feedforward=bottleneck * 4,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        # ── Gate ──
        if gate_type == "frame":
            self.gate_linear = nn.Linear(bottleneck, 1)
        else:
            self.gate_proj = nn.Linear(vision_dim, bottleneck)
            self.gate_bias = nn.Parameter(torch.tensor(-2.0))

        # ── FiLM parameters (from temporal context) ──
        self.film_scale = nn.Linear(bottleneck, vision_dim)
        self.film_shift = nn.Linear(bottleneck, vision_dim)

        # ── Spatial saliency (v10b): per-token modulation weight ──
        if spatial_saliency:
            self.spatial_proj = nn.Linear(vision_dim, bottleneck)
            self.spatial_bias = nn.Parameter(torch.tensor(-1.0))

        self._init_weights()

    def _init_weights(self):
        # FiLM near-zero init
        nn.init.normal_(self.film_scale.weight, std=0.01)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.normal_(self.film_shift.weight, std=0.01)
        nn.init.zeros_(self.film_shift.bias)
        # Gate init
        if self.gate_type == "frame":
            nn.init.zeros_(self.gate_linear.weight)
            nn.init.constant_(self.gate_linear.bias, -2.0)
        else:
            nn.init.normal_(self.gate_proj.weight, std=0.01)
            nn.init.zeros_(self.gate_proj.bias)
        # Motion projection init
        if self.k_motion > 0:
            nn.init.normal_(self.motion_down.weight, std=0.01)
            nn.init.zeros_(self.motion_down.bias)
        # Spatial saliency init
        if self.spatial_saliency:
            nn.init.normal_(self.spatial_proj.weight, std=0.01)
            nn.init.zeros_(self.spatial_proj.bias)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, C) vision features
        Returns:
            enriched: (B, T, N, C) temporally enriched features
        """
        B, T, N, C = x.shape

        if T <= 1:
            self._last_gate_mean = 0.0
            self._last_gate_std = 0.0
            self._last_scale_norm = 0.0
            self._last_shift_norm = 0.0
            self._last_eff_scale_norm = 0.0
            self._last_eff_shift_norm = 0.0
            self._last_gate_logit_live = None
            return x

        # Frame-level temporal context via mean pooling → transformer
        desc = x.mean(dim=2)  # (B, T, C)
        app_h = self.down(desc) + self.pos_embed[:, :T]  # (B, T, bn)

        # Optional: motion tokens from temporal diffs (approach B)
        if self.k_motion > 0 and T > 1:
            # Compute per-position temporal diff magnitude
            diff_mag = torch.zeros(B, T, N, device=x.device, dtype=x.dtype)
            diff_mag[:, 1:] = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
            _, topk_idx = diff_mag.topk(self.k_motion, dim=-1)  # (B, T, K)

            # Gather original features at most-changing positions
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, C)
            motion_tokens = x.gather(2, idx_exp)  # (B, T, K, C)
            motion_h = self.motion_down(motion_tokens)  # (B, T, K, bn)

            # Add frame PE (reuse pos_embed) + motion type embedding
            motion_h = (motion_h
                        + self.pos_embed[:, :T].unsqueeze(2)  # (1, T, 1, bn)
                        + self.motion_type_embed)              # (1, 1, 1, bn)
            motion_flat = motion_h.reshape(B, T * self.k_motion, self.bottleneck)

            # Concatenate: [T appearance tokens, T*K motion tokens]
            combined = torch.cat([app_h, motion_flat], dim=1)
        else:
            combined = app_h

        for layer in self.layers:
            combined = layer(combined)

        # Extract frame-level context (first T tokens)
        h = combined[:, :T]  # (B, T, bn)

        # Bounded FiLM parameters (separate bounds for scale/shift)
        if self.scale_bound > 0:
            scale = self.scale_bound * torch.tanh(self.film_scale(h))  # (B, T, C)
            shift = self.shift_bound * torch.tanh(self.film_shift(h))  # (B, T, C)
        else:
            scale = self.film_scale(h)
            shift = self.film_shift(h)

        # Gate (cache logit WITH grad for entropy reg, detached for logging)
        if self.gate_type == "frame":
            gate_logit = self.gate_linear(h)  # (B, T, 1)
            self._last_gate_logit_live = gate_logit  # WITH grad — for entropy reg
            gate = torch.sigmoid(gate_logit)  # (B, T, 1)
        else:
            x_gate = self.gate_proj(desc)  # (B, T, bn)
            raw_logit = (x_gate * h).sum(dim=-1, keepdim=True) / math.sqrt(self.bottleneck)
            gate_logit = raw_logit + self.gate_bias
            self._last_gate_logit_live = gate_logit  # WITH grad
            gate = torch.sigmoid(gate_logit)  # (B, T, 1)
        gate = gate.unsqueeze(2)  # (B, T, 1, 1)

        # FiLM modulation with gate (+ optional spatial saliency)
        if self.spatial_saliency:
            # Per-token saliency: dot(spatial_proj(x), h) / sqrt(bn)
            x_proj = self.spatial_proj(x)  # (B, T, N, bn)
            sal_logit = (x_proj * h.unsqueeze(2)).sum(dim=-1) / math.sqrt(self.bottleneck)
            sal_logit = sal_logit + self.spatial_bias  # (B, T, N)
            saliency = torch.sigmoid(sal_logit)  # (B, T, N)
            self._last_saliency = saliency.detach()
            self._last_sal_logit_live = sal_logit  # WITH grad for supervision
            sal = saliency.unsqueeze(-1)  # (B, T, N, 1)
            enriched = (1.0 + gate * sal * scale.unsqueeze(2)) * x + gate * sal * shift.unsqueeze(2)
        else:
            self._last_saliency = None
            self._last_sal_logit_live = None
            enriched = (1.0 + gate * scale.unsqueeze(2)) * x + gate * shift.unsqueeze(2)

        # Cache for logging (detached)
        self._last_gate_mean = gate.detach().mean().item()
        self._last_gate_std = gate.detach().std().item()
        self._last_scale_norm = scale.detach().norm(dim=-1).mean().item()
        self._last_shift_norm = shift.detach().norm(dim=-1).mean().item()
        self._last_eff_scale_norm = self._last_gate_mean * self._last_scale_norm
        self._last_eff_shift_norm = self._last_gate_mean * self._last_shift_norm

        return enriched

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DummyPredictor(nn.Module):
    """No-op predictor for motion-supervised modes (v10a/v10b)."""

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0, device=self._dummy.device)

    def count_trainable_params(self):
        return 1


class TokenLevelPredictor(nn.Module):
    """
    Original v6-v8 predictor: absolute token targets, mean-pooled context.
    Kept for backward compatibility.
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        bottleneck: int = 128,
        num_heads: int = 4,
        max_frames: int = 32,
        num_spatial: int = 1024,
    ):
        super().__init__()
        self.bottleneck = bottleneck
        self.context_down = nn.Linear(vision_dim, bottleneck)
        self.frame_queries = nn.Parameter(
            torch.randn(max_frames, bottleneck) * 0.02)
        self.spatial_embed = nn.Embedding(num_spatial, bottleneck)
        self.cross_attn = nn.MultiheadAttention(
            bottleneck, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(bottleneck)
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck, bottleneck * 4),
            nn.GELU(),
            nn.Linear(bottleneck * 4, bottleneck),
        )
        self.norm2 = nn.LayerNorm(bottleneck)
        self.up = nn.Linear(bottleneck, vision_dim)

    def forward(self, enriched_vis, mask_indices, spatial_indices):
        B, T_vis, N, C = enriched_vis.shape
        T_mask = len(mask_indices)
        K = spatial_indices.shape[1]

        kv = self.context_down(enriched_vis.mean(dim=2))
        frame_emb = self.frame_queries[mask_indices]
        spatial_emb = self.spatial_embed(spatial_indices)
        queries = frame_emb.unsqueeze(1) + spatial_emb
        queries = queries.reshape(T_mask * K, self.bottleneck)
        queries = queries.unsqueeze(0).expand(B, -1, -1)

        h, _ = self.cross_attn(queries, kv, kv)
        h = self.norm1(queries + h)
        h = self.norm2(h + self.ffn(h))
        return self.up(h)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeltaTokenPredictor(nn.Module):
    """
    v9a predictor: delta token targets with full spatial-token context.

    Two key changes from TokenLevelPredictor:
      1. Context: subsampled K_ctx spatial tokens per visible frame (not mean-pooled)
      2. Target: delta = frozen[mask] - frozen[last_visible] (computed in train script)

    Input: enriched visible features (B, T_vis, N, C) from adapter.forward()
    Output: predicted delta tokens (B, T_mask * K, vision_dim)
    Discarded at inference.
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        bottleneck: int = 128,
        num_heads: int = 4,
        max_frames: int = 32,
        num_spatial: int = 1024,
        k_ctx: int = 32,
    ):
        super().__init__()
        self.bottleneck = bottleneck
        self.k_ctx = k_ctx

        # Down-project enriched spatial tokens for context
        self.context_down = nn.Linear(vision_dim, bottleneck)

        # Learnable queries for masked frame positions
        self.frame_queries = nn.Parameter(
            torch.randn(max_frames, bottleneck) * 0.02)

        # Spatial position embedding (shared between query and context)
        self.spatial_embed = nn.Embedding(num_spatial, bottleneck)

        # Frame position embedding for context KV
        self.ctx_frame_embed = nn.Parameter(
            torch.randn(max_frames, bottleneck) * 0.02)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            bottleneck, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(bottleneck)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck, bottleneck * 4),
            nn.GELU(),
            nn.Linear(bottleneck * 4, bottleneck),
        )
        self.norm2 = nn.LayerNorm(bottleneck)

        # Project to vision space
        self.up = nn.Linear(bottleneck, vision_dim)

    def forward(self, enriched_vis, mask_indices, spatial_indices):
        """
        Args:
            enriched_vis: (B, T_vis, N, C) enriched visible features
            mask_indices: (T_mask,) masked frame indices
            spatial_indices: (T_mask, K) spatial token indices per frame
        Returns:
            pred: (B, T_mask * K, vision_dim)
        """
        B, T_vis, N, C = enriched_vis.shape
        T_mask = len(mask_indices)
        K = spatial_indices.shape[1]
        device = enriched_vis.device

        # ── Context: subsample K_ctx spatial tokens per visible frame ──
        ctx_spatial_idx = torch.stack([
            torch.randperm(N, device=device)[:self.k_ctx]
            for _ in range(T_vis)
        ])  # (T_vis, K_ctx)

        # Gather: (B, T_vis, K_ctx, C)
        idx_exp = ctx_spatial_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, C)
        ctx_tokens = enriched_vis.gather(2, idx_exp)

        # Down-project + position embeddings
        ctx_down = self.context_down(ctx_tokens)  # (B, T_vis, K_ctx, bn)
        ctx_frame_pe = self.ctx_frame_embed[:T_vis].unsqueeze(1)  # (T_vis, 1, bn)
        ctx_spatial_pe = self.spatial_embed(ctx_spatial_idx)  # (T_vis, K_ctx, bn)
        ctx_down = ctx_down + ctx_frame_pe.unsqueeze(0) + ctx_spatial_pe.unsqueeze(0)

        kv = ctx_down.reshape(B, T_vis * self.k_ctx, self.bottleneck)

        # ── Queries: frame position + spatial position ──
        frame_emb = self.frame_queries[mask_indices]
        spatial_emb = self.spatial_embed(spatial_indices)
        queries = frame_emb.unsqueeze(1) + spatial_emb
        queries = queries.reshape(T_mask * K, self.bottleneck)
        queries = queries.unsqueeze(0).expand(B, -1, -1)

        # ── Cross-attention + FFN ──
        h, _ = self.cross_attn(queries, kv, kv)
        h = self.norm1(queries + h)
        h = self.norm2(h + self.ffn(h))

        return self.up(h)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionContrastivePredictor(nn.Module):
    """
    v10 predictor: Attention-pooled spatial-temporal contrastive prediction.

    Key improvement over SegmentContrastivePredictor (v9b):
      - NO mean-pooling: samples K spatial tokens directly
      - Attention-based aggregation with learned CLS queries
      - Both context and segments preserve spatial-temporal token structure

    Tasks (training only, discarded at inference):
      1. Future Classification: CE loss, chance = 1/n_candidates
      2. Temporal Order: BCE loss, chance = 50%
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        bottleneck: int = 128,
        num_heads: int = 4,
        max_frames: int = 32,
        k_ctx: int = 32,
        segment_size: int = 2,
        n_candidates: int = 4,
        temperature: float = 0.1,
        order_pairs: int = 2,
    ):
        super().__init__()
        self.bottleneck = bottleneck
        self.k_ctx = k_ctx
        self.segment_size = segment_size
        self.n_candidates = n_candidates
        self.temperature = temperature
        self.order_pairs = order_pairs

        # Shared token projection (1024 → 128)
        self.token_down = nn.Linear(vision_dim, bottleneck)

        # Context frame position embedding
        self.ctx_frame_pe = nn.Parameter(
            torch.zeros(1, max_frames, 1, bottleneck))
        nn.init.trunc_normal_(self.ctx_frame_pe, std=0.02)

        # Context: self-attention over spatial-temporal tokens
        self.ctx_self_attn = nn.TransformerEncoderLayer(
            d_model=bottleneck,
            nhead=num_heads,
            dim_feedforward=bottleneck * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        # CLS query for context attention pooling
        self.ctx_cls = nn.Parameter(torch.randn(1, 1, bottleneck) * 0.02)
        self.ctx_cross_attn = nn.MultiheadAttention(
            bottleneck, num_heads, batch_first=True)
        self.ctx_proj = nn.Linear(bottleneck, bottleneck)

        # Segment: attention pooling (NO position info → content-based)
        self.seg_cls = nn.Parameter(torch.randn(1, 1, bottleneck) * 0.02)
        self.seg_cross_attn = nn.MultiheadAttention(
            bottleneck, num_heads, batch_first=True)
        self.seg_proj = nn.Linear(bottleneck, bottleneck)

        # Order head
        self.order_head = nn.Sequential(
            nn.Linear(bottleneck * 3, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)
        nn.init.xavier_uniform_(self.seg_proj.weight)
        nn.init.zeros_(self.seg_proj.bias)

    def forward(self, enriched_vis, frozen_all, T_vis):
        """
        Args:
            enriched_vis: (B, T_vis, N, C) adapter output for visible frames
            frozen_all: (B, T, N, C) all frozen features
            T_vis: int, number of visible frames
        Returns:
            cls_loss, order_loss, metrics dict
        """
        B, T = frozen_all.shape[:2]
        N, C = frozen_all.shape[2], frozen_all.shape[3]
        device = frozen_all.device

        remaining = T - T_vis
        n_segs = remaining // self.segment_size
        n_cands = min(self.n_candidates, n_segs)

        if n_cands < 2:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, {'cls_acc': 0.0, 'order_acc': 0.5}

        # Sample K spatial positions (shared for context and segments)
        spatial_idx = torch.randperm(N, device=device)[:self.k_ctx]

        # ── Context from enriched visible features ──
        ctx_tokens = enriched_vis[:, :, spatial_idx]  # (B, T_vis, K, C)
        ctx_down = self.token_down(ctx_tokens)  # (B, T_vis, K, bn)
        # Add frame PE (no spatial PE — rely on content)
        ctx_down = ctx_down + self.ctx_frame_pe[:, :T_vis]  # broadcast K
        ctx_flat = ctx_down.reshape(B, T_vis * self.k_ctx, self.bottleneck)

        # Self-attention among all spatial-temporal tokens
        ctx_enc = self.ctx_self_attn(ctx_flat)  # (B, T_vis*K, bn)

        # Attention pool with learned CLS query
        cls_q = self.ctx_cls.expand(B, -1, -1)  # (B, 1, bn)
        ctx_agg, _ = self.ctx_cross_attn(cls_q, ctx_enc, ctx_enc)
        ctx_pre_norm = self.ctx_proj(ctx_agg.squeeze(1))  # (B, bn)
        ctx_repr = F.normalize(ctx_pre_norm, dim=-1)  # (B, bn)

        # ── Segment keys from frozen features (NO position info) ──
        seg_reprs = []
        seg_pre_norms = []
        for i in range(n_cands):
            start = T_vis + i * self.segment_size
            end = min(start + self.segment_size, T)
            seg_T = end - start

            seg_tokens = frozen_all[:, start:end, :, :]
            seg_tokens = seg_tokens[:, :, spatial_idx].detach()
            seg_down = self.token_down(seg_tokens)  # (B, seg_T, K, bn)
            seg_flat = seg_down.reshape(
                B, seg_T * self.k_ctx, self.bottleneck)

            # Attention pool
            seg_q = self.seg_cls.expand(B, -1, -1)
            seg_agg, _ = self.seg_cross_attn(seg_q, seg_flat, seg_flat)
            seg_pre = self.seg_proj(seg_agg.squeeze(1))  # (B, bn)
            seg_repr = F.normalize(seg_pre, dim=-1)

            seg_reprs.append(seg_repr)
            seg_pre_norms.append(seg_pre)

        seg_keys = torch.stack(seg_reprs, dim=1)  # (B, n_cands, bn)

        # ── Future classification: positive = segment 0 ──
        logits = torch.einsum(
            'bd,bnd->bn', ctx_repr, seg_keys) / self.temperature
        cls_target = torch.zeros(B, dtype=torch.long, device=device)
        cls_loss = F.cross_entropy(logits, cls_target)

        with torch.no_grad():
            cls_acc = (logits.argmax(dim=1) == cls_target).float().mean().item()

        # ── Temporal order prediction ──
        order_loss = torch.tensor(0.0, device=device)
        order_acc = 0.5

        if n_cands >= 2:
            n_pairs = min(self.order_pairs,
                          n_cands * (n_cands - 1) // 2)
            o_losses = []
            o_correct = 0
            o_total = 0

            for _ in range(n_pairs):
                idx = torch.randperm(n_cands, device=device)[:2]
                i, j = idx[0].item(), idx[1].item()
                if i > j:
                    i, j = j, i

                seg_a = seg_pre_norms[i]
                seg_b = seg_pre_norms[j]

                swap = (torch.rand(B, device=device) > 0.5)
                inp_a = torch.where(swap.unsqueeze(-1), seg_b, seg_a)
                inp_b = torch.where(swap.unsqueeze(-1), seg_a, seg_b)

                order_target = (~swap).float()
                o_input = torch.cat(
                    [inp_a, inp_b, ctx_pre_norm], dim=-1)
                o_logit = self.order_head(o_input).squeeze(-1)
                o_losses.append(
                    F.binary_cross_entropy_with_logits(
                        o_logit, order_target))

                with torch.no_grad():
                    o_correct += (
                        (o_logit > 0).float() == order_target
                    ).sum().item()
                    o_total += B

            if o_losses:
                order_loss = torch.stack(o_losses).mean()
                order_acc = o_correct / o_total

        return cls_loss, order_loss, {'cls_acc': cls_acc, 'order_acc': order_acc}

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SegmentContrastivePredictor(nn.Module):
    """
    v9b predictor: Future segment classification + temporal order prediction.

    Two tasks (training only, discarded at inference):
      1. Future Classification: given enriched visible context, identify which of
         n_candidates segments comes immediately after the visible prefix.
         CE loss, chance accuracy = 1/n_candidates.
      2. Order Prediction: given two segments (50% swapped), predict correct
         temporal order. Content-based only (no absolute position info).
         BCE loss, chance accuracy = 50%.

    Uses 50% masking: first half visible (context), second half divided into segments.
    All candidates from same video — negatives are nearby temporal segments (hard).
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        bottleneck: int = 128,
        num_heads: int = 4,
        segment_size: int = 2,
        n_candidates: int = 4,
        temperature: float = 0.1,
        order_pairs: int = 2,
    ):
        super().__init__()
        self.bottleneck = bottleneck
        self.segment_size = segment_size
        self.n_candidates = n_candidates
        self.temperature = temperature
        self.order_pairs = order_pairs

        # ── Context encoder (from enriched visible features) ──
        self.ctx_down = nn.Linear(vision_dim, bottleneck)
        self.ctx_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck,
            nhead=num_heads,
            dim_feedforward=bottleneck * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.ctx_proj = nn.Linear(bottleneck, bottleneck)

        # ── Segment encoder (from frozen features, no position info) ──
        self.seg_down = nn.Linear(vision_dim, bottleneck)
        self.seg_proj = nn.Linear(bottleneck, bottleneck)

        # ── Order head: content-based, no position info ──
        self.order_head = nn.Sequential(
            nn.Linear(bottleneck * 3, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)
        nn.init.xavier_uniform_(self.seg_proj.weight)
        nn.init.zeros_(self.seg_proj.bias)

    def forward(self, enriched_vis, frozen_all, T_vis):
        """
        Args:
            enriched_vis: (B, T_vis, N, C) adapter output for visible frames
            frozen_all: (B, T, N, C) all frozen features
            T_vis: int, number of visible frames
        Returns:
            cls_loss, order_loss, metrics dict
        """
        B, T = frozen_all.shape[:2]
        device = frozen_all.device

        remaining = T - T_vis
        n_segs = remaining // self.segment_size
        n_cands = min(self.n_candidates, n_segs)

        if n_cands < 2:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, {'cls_acc': 0.0, 'order_acc': 0.5}

        # ── Segment frame indices (all future, temporally ordered) ──
        seg_frame_lists = []
        for i in range(n_cands):
            start = T_vis + i * self.segment_size
            end = min(start + self.segment_size, T)
            seg_frame_lists.append(list(range(start, end)))

        # ── 1. Context query from enriched visible features ──
        ctx_pooled = enriched_vis.mean(dim=2)  # (B, T_vis, C) spatial pool
        ctx_h = self.ctx_down(ctx_pooled)  # (B, T_vis, bn)
        ctx_h = self.ctx_layer(ctx_h)
        ctx_q = self.ctx_proj(ctx_h.mean(dim=1))  # (B, bn) temporal pool
        ctx_q = F.normalize(ctx_q, dim=-1)

        # ── 2. Segment keys from frozen features ──
        seg_features = []
        for seg_idx in seg_frame_lists:
            seg_feat = frozen_all[:, seg_idx].detach().mean(dim=(1, 2))  # (B, C)
            seg_features.append(seg_feat)
        seg_stack = torch.stack(seg_features, dim=1)  # (B, n_cands, C)
        seg_h = self.seg_down(seg_stack)  # (B, n_cands, bn)
        seg_keys = F.normalize(self.seg_proj(seg_h), dim=-1)  # (B, n_cands, bn)

        # ── Future classification: positive = segment 0 (immediately after visible) ──
        logits = torch.einsum('bd,bnd->bn', ctx_q, seg_keys) / self.temperature
        cls_target = torch.zeros(B, dtype=torch.long, device=device)
        cls_loss = F.cross_entropy(logits, cls_target)

        with torch.no_grad():
            cls_acc = (logits.argmax(dim=1) == cls_target).float().mean().item()

        # ── 3. Temporal order prediction ──
        order_loss = torch.tensor(0.0, device=device)
        order_acc = 0.5

        if n_cands >= 2:
            n_pairs = min(self.order_pairs, n_cands * (n_cands - 1) // 2)
            o_losses = []
            o_correct = 0
            o_total = 0

            for _ in range(n_pairs):
                idx = torch.randperm(n_cands, device=device)[:2]
                i, j = idx[0].item(), idx[1].item()
                if i > j:
                    i, j = j, i  # correct order: i < j

                seg_a = seg_h[:, i]  # (B, bn) — earlier segment
                seg_b = seg_h[:, j]  # (B, bn) — later segment

                # 50% random swap
                swap = (torch.rand(B, device=device) > 0.5)
                inp_a = torch.where(swap.unsqueeze(-1), seg_b, seg_a)
                inp_b = torch.where(swap.unsqueeze(-1), seg_a, seg_b)

                order_target = (~swap).float()  # 1 = correct order

                o_input = torch.cat([inp_a, inp_b, ctx_q], dim=-1)  # (B, bn*3)
                o_logit = self.order_head(o_input).squeeze(-1)
                o_losses.append(
                    F.binary_cross_entropy_with_logits(o_logit, order_target))

                with torch.no_grad():
                    o_correct += ((o_logit > 0).float() == order_target).sum().item()
                    o_total += B

            if o_losses:
                order_loss = torch.stack(o_losses).mean()
                order_acc = o_correct / o_total

        return cls_loss, order_loss, {'cls_acc': cls_acc, 'order_acc': order_acc}

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
