# Copyright (c) Meta Platforms, Inc. and affiliates.
# A3 projector with learned anchor-state, anchored-delta, optional future
# stream, locality enhancement, and convex alignment head.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vision_projector.base import BaseProjector


class A3Projector(BaseProjector):
    def __init__(self, args):
        super().__init__()

        self.vision_dim = args.vision_model.get("width", 1024)
        self.llm_dim = args.dim

        self.bottleneck_dim = getattr(args, "a3_bottleneck_dim", 256)
        self.state_spatial = getattr(args, "a3_state_spatial", 4)
        self.delta_temporal = getattr(args, "a3_delta_temporal", 3)
        self.delta_spatial = getattr(args, "a3_delta_spatial", 4)
        self.reg_weight = getattr(args, "a3_reg_weight", 0.01)
        self.reg_smooth_weight = getattr(args, "a3_reg_smooth_weight", 1.0)
        self.reg_ortho_weight = getattr(args, "a3_reg_ortho_weight", 0.25)
        self.use_adaptive_gate = getattr(args, "a3_adaptive_gate", True)

        self.use_locality_block = getattr(args, "a3_use_locality_block", True)

        # Blend between anchored delta and frame-to-frame delta.
        blend_init = float(getattr(args, "a3_delta_anchor_blend", 0.65))
        blend_init = min(max(blend_init, 1e-4), 1.0 - 1e-4)
        self.delta_anchor_blend_init = blend_init

        self.use_future_stream = getattr(args, "a3_use_future_stream", False)
        self.future_temporal = max(1, int(getattr(args, "a3_future_temporal", 2)))
        self.future_spatial = max(1, int(getattr(args, "a3_future_spatial", 2)))

        self.align_use_convex = getattr(args, "a3_align_use_convex", True)
        self.align_codebook_size = max(8, int(getattr(args, "a3_align_codebook_size", 512)))
        self.align_mix_ratio = float(getattr(args, "a3_align_mix_ratio", 0.2))

        scale_init = float(getattr(args, "a3_post_ln_scale_init", 1.0 / math.sqrt(self.llm_dim)))
        self.post_ln_scale_init = max(scale_init, 1e-6)

        self._reg_loss = None
        self.setup_projector(args)

    def setup_projector(self, args=None):
        d = self.bottleneck_dim

        self.channel_reduce = nn.Conv3d(
            self.vision_dim,
            d,
            kernel_size=1,
            bias=True,
            dtype=torch.get_default_dtype(),
        )
        self.channel_act = nn.GELU()

        if self.use_locality_block:
            self.locality_dw = nn.Conv3d(
                d,
                d,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=d,
                bias=True,
                dtype=torch.get_default_dtype(),
            )
            self.locality_pw = nn.Conv3d(
                d,
                d,
                kernel_size=1,
                bias=True,
                dtype=torch.get_default_dtype(),
            )
            self.locality_act = nn.GELU()
            self.locality_gate = nn.Parameter(torch.zeros(1))

        self.temporal_attn = nn.Linear(
            d,
            1,
            bias=True,
            dtype=torch.get_default_dtype(),
        )

        self.state_conv = nn.Conv2d(
            d,
            d,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            dtype=torch.get_default_dtype(),
        )
        self.state_act = nn.GELU()
        self.state_pool = nn.AdaptiveAvgPool2d(self.state_spatial)

        self.delta_temporal_conv = nn.Conv3d(
            d,
            d,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=True,
            dtype=torch.get_default_dtype(),
        )
        self.delta_act = nn.GELU()
        self.delta_pool = nn.AdaptiveAvgPool3d(
            (self.delta_temporal, self.delta_spatial, self.delta_spatial)
        )

        if self.use_future_stream:
            self.future_temporal_conv = nn.Conv3d(
                d,
                d,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=True,
                dtype=torch.get_default_dtype(),
            )
            self.future_act = nn.GELU()
            self.future_pool = nn.AdaptiveAvgPool3d(
                (self.future_temporal, self.future_spatial, self.future_spatial)
            )

        if self.use_adaptive_gate:
            self.motion_gate = nn.Sequential(
                nn.Linear(d, max(1, d // 4), dtype=torch.get_default_dtype()),
                nn.GELU(),
                nn.Linear(max(1, d // 4), 1, dtype=torch.get_default_dtype()),
            )
            self.gate_strength = nn.Parameter(torch.zeros(1))

        self.output_proj = nn.Linear(
            d,
            self.llm_dim,
            bias=True,
            dtype=torch.get_default_dtype(),
        )
        self.output_ln = nn.LayerNorm(self.llm_dim)
        self.post_ln_scale = nn.Parameter(
            torch.tensor([self.post_ln_scale_init], dtype=torch.get_default_dtype())
        )

        self.type_state = nn.Parameter(torch.zeros(self.llm_dim))
        self.type_delta = nn.Parameter(torch.zeros(self.llm_dim))
        if self.use_future_stream:
            self.type_future = nn.Parameter(torch.zeros(self.llm_dim))

        self.delta_anchor_blend_logit = nn.Parameter(
            torch.tensor(
                [math.log(self.delta_anchor_blend_init / (1.0 - self.delta_anchor_blend_init))],
                dtype=torch.get_default_dtype(),
            )
        )

        if self.align_use_convex:
            self.align_logits = nn.Linear(
                self.llm_dim,
                self.align_codebook_size,
                bias=True,
                dtype=torch.get_default_dtype(),
            )
            self.align_codebook = nn.Parameter(
                torch.zeros(self.align_codebook_size, self.llm_dim, dtype=torch.get_default_dtype())
            )
            mix = min(max(self.align_mix_ratio, 1e-4), 1.0 - 1e-4)
            self.align_mix_logit = nn.Parameter(
                torch.tensor([math.log(mix / (1.0 - mix))], dtype=torch.get_default_dtype())
            )

    def init_tensors(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        nn.init.xavier_uniform_(self.temporal_attn.weight)
        nn.init.zeros_(self.temporal_attn.bias)

        nn.init.normal_(self.type_state, std=0.02)
        nn.init.normal_(self.type_delta, std=0.02)
        if self.use_future_stream:
            nn.init.normal_(self.type_future, std=0.02)

        if self.use_locality_block:
            nn.init.zeros_(self.locality_gate)

        if self.use_adaptive_gate:
            for m in self.motion_gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            nn.init.zeros_(self.gate_strength)

        if self.align_use_convex:
            nn.init.xavier_uniform_(self.align_logits.weight)
            nn.init.zeros_(self.align_logits.bias)
            nn.init.normal_(self.align_codebook, std=0.02)

    def _safe_temporal_conv(self, conv: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
        k_t = int(conv.kernel_size[0]) if isinstance(conv.kernel_size, tuple) else int(conv.kernel_size)
        if x.shape[2] < k_t:
            pad = k_t - x.shape[2]
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        return conv(x)

    def _predict_future_tokens(self, f_feat: torch.Tensor, delta_f: torch.Tensor) -> torch.Tensor:
        vel_last = delta_f[:, :, -1:, :, :]
        if delta_f.shape[2] >= 2:
            accel_last = delta_f[:, :, -1:, :, :] - delta_f[:, :, -2:-1, :, :]
        else:
            accel_last = torch.zeros_like(vel_last)

        pred = f_feat[:, :, -1:, :, :] + vel_last + 0.5 * accel_last
        preds = [pred]
        for _ in range(self.future_temporal - 1):
            vel_last = vel_last + accel_last
            pred = preds[-1] + vel_last
            preds.append(pred)
        future = torch.cat(preds, dim=2)
        future = self.future_act(self._safe_temporal_conv(self.future_temporal_conv, future))
        future = self.future_pool(future)
        return future.flatten(2).permute(0, 2, 1)

    def forward(self, grid_tokens: torch.Tensor) -> torch.Tensor:
        _, t, _, _, _ = grid_tokens.shape
        if t < 2:
            pad = grid_tokens[:, -1:, :, :, :].expand(-1, 2 - t, -1, -1, -1)
            grid_tokens = torch.cat([grid_tokens, pad], dim=1)

        x = grid_tokens.permute(0, 4, 1, 2, 3).contiguous()
        f_feat = self.channel_act(self.channel_reduce(x))

        if self.use_locality_block:
            local = self.locality_act(self.locality_dw(f_feat))
            local = self.locality_pw(local)
            f_feat = f_feat + torch.sigmoid(self.locality_gate) * local

        # Learned temporal anchor
        f_avg = f_feat.mean(dim=[3, 4]).permute(0, 2, 1)  # (B, T, d)
        alpha_logits = self.temporal_attn(f_avg)
        alpha = F.softmax(alpha_logits, dim=1)
        alpha_w = alpha.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        anchor = (f_feat * alpha_w).sum(dim=2)  # (B, d, H, W)

        # State tokens
        s = self.state_act(self.state_conv(anchor))
        s = self.state_pool(s)
        state_tokens = s.flatten(2).permute(0, 2, 1)

        # Anchored-delta + frame-diff hybrid
        delta_diff = f_feat[:, :, 1:, :, :] - f_feat[:, :, :-1, :, :]
        anchor_t = anchor.unsqueeze(2).expand(-1, -1, delta_diff.shape[2], -1, -1)
        delta_anchor = f_feat[:, :, 1:, :, :] - anchor_t
        blend = torch.sigmoid(self.delta_anchor_blend_logit)
        delta_f = blend * delta_anchor + (1.0 - blend) * delta_diff

        d = self.delta_act(self._safe_temporal_conv(self.delta_temporal_conv, delta_f))
        d = self.delta_pool(d)
        delta_tokens = d.flatten(2).permute(0, 2, 1)

        future_tokens = None
        if self.use_future_stream:
            future_tokens = self._predict_future_tokens(f_feat, delta_f)

        if self.use_adaptive_gate:
            state_tokens, delta_tokens, future_tokens = self._apply_adaptive_gate(
                f_feat, state_tokens, delta_tokens, future_tokens
            )

        tokens = [state_tokens, delta_tokens]
        if future_tokens is not None:
            tokens.append(future_tokens)
        v = torch.cat(tokens, dim=1)
        e = self.output_proj(v)

        n_state = self.state_spatial ** 2
        n_delta = self.delta_temporal * self.delta_spatial ** 2
        e[:, :n_state, :] = e[:, :n_state, :] + self.type_state
        e[:, n_state:n_state + n_delta, :] = e[:, n_state:n_state + n_delta, :] + self.type_delta
        if future_tokens is not None:
            e[:, n_state + n_delta :, :] = e[:, n_state + n_delta :, :] + self.type_future

        e = self.output_ln(e)
        e = e * self.post_ln_scale

        if self.align_use_convex:
            probs = F.softmax(self.align_logits(e), dim=-1)
            aligned = probs @ self.align_codebook
            mix = torch.sigmoid(self.align_mix_logit)
            e = (1.0 - mix) * e + mix * aligned

        self._reg_loss = self._compute_reg_loss(alpha, anchor, delta_f)
        return e

    def _apply_adaptive_gate(
        self,
        f_feat: torch.Tensor,
        state_tokens: torch.Tensor,
        delta_tokens: torch.Tensor,
        future_tokens: torch.Tensor = None,
    ):
        motion_feat = f_feat.var(dim=2).mean(dim=[2, 3])
        gate = torch.sigmoid(self.motion_gate(motion_feat))
        strength = torch.clamp(self.gate_strength, min=0.0, max=1.0)

        state_scale = (1.0 + strength * (1.0 - 2.0 * gate)).unsqueeze(-1)
        dyn_scale = (1.0 + strength * (2.0 * gate - 1.0)).unsqueeze(-1)

        state_tokens = state_tokens * state_scale
        delta_tokens = delta_tokens * dyn_scale
        if future_tokens is not None:
            future_tokens = future_tokens * dyn_scale
        return state_tokens, delta_tokens, future_tokens

    def _compute_reg_loss(
        self,
        alpha: torch.Tensor,
        anchor: torch.Tensor,
        delta_f: torch.Tensor,
    ) -> torch.Tensor:
        alpha_diff = alpha[:, 1:, :] - alpha[:, :-1, :]
        smooth_loss = (alpha_diff ** 2).mean()

        bsz = delta_f.shape[0]
        anchor_flat = anchor.flatten(1)  # (B, d*H*W)
        delta_flat = delta_f.permute(0, 2, 1, 3, 4).reshape(bsz, -1, anchor_flat.shape[1])
        anchor_norm = F.normalize(anchor_flat, dim=-1, eps=1e-6)
        delta_norm = F.normalize(delta_flat, dim=-1, eps=1e-6)
        cos_sim = (delta_norm * anchor_norm.unsqueeze(1)).sum(dim=-1)
        ortho_loss = (cos_sim ** 2).mean()

        reg = self.reg_smooth_weight * smooth_loss + self.reg_ortho_weight * ortho_loss
        return self.reg_weight * reg

    def get_reg_loss(self) -> torch.Tensor:
        if self._reg_loss is not None:
            return self._reg_loss
        return self.output_proj.weight.new_zeros(())

    def count_tokens(self) -> int:
        state = self.state_spatial ** 2
        delta = self.delta_temporal * self.delta_spatial ** 2
        future = self.future_temporal * self.future_spatial ** 2 if self.use_future_stream else 0
        return state + delta + future

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
