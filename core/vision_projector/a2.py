# Copyright (c) Meta Platforms, Inc. and affiliates.
# Dynamics-aware A2 projector with locality enhancement and optional
# future stream / convex alignment head.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vision_projector.base import BaseProjector


class DynamicsAwareA2Projector(BaseProjector):
    def __init__(self, args):
        super().__init__()

        self.vision_dim = args.vision_model.get("width", 1024)
        self.llm_dim = args.dim

        self.bottleneck_dim = getattr(args, "a2_bottleneck_dim", 256)
        self.state_spatial = getattr(args, "a2_state_spatial", 4)
        self.delta_temporal = getattr(args, "a2_delta_temporal", 3)
        self.delta_spatial = getattr(args, "a2_delta_spatial", 4)

        # Locality-enhanced abstractor block (Honeybee-inspired).
        self.use_locality_block = getattr(args, "a2_use_locality_block", True)

        # Predictor-like future tokens (rollout in feature space).
        self.use_future_stream = getattr(args, "a2_use_future_stream", False)
        self.future_temporal = max(1, int(getattr(args, "a2_future_temporal", 2)))
        self.future_spatial = max(1, int(getattr(args, "a2_future_spatial", 2)))

        # AlignVLM-style convex alignment in LLM input space.
        self.align_use_convex = getattr(args, "a2_align_use_convex", True)
        self.align_codebook_size = max(8, int(getattr(args, "a2_align_codebook_size", 512)))
        self.align_mix_ratio = float(getattr(args, "a2_align_mix_ratio", 0.2))

        # Token norm alignment.
        scale_init = float(getattr(args, "a2_post_ln_scale_init", 1.0 / math.sqrt(self.llm_dim)))
        self.post_ln_scale_init = max(scale_init, 1e-6)

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
            # Start close to no-op for stable resume from old bridges.
            self.locality_gate = nn.Parameter(torch.zeros(1))

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

        if self.use_locality_block:
            nn.init.zeros_(self.locality_gate)

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
        # velocity-like term
        vel_last = delta_f[:, :, -1:, :, :]
        # acceleration-like term
        if delta_f.shape[2] >= 2:
            accel_last = (delta_f[:, :, -1:, :, :] - delta_f[:, :, -2:-1, :, :])
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

        # State
        s = f_feat.mean(dim=2)
        s = self.state_act(self.state_conv(s))
        s = self.state_pool(s)
        state_tokens = s.flatten(2).permute(0, 2, 1)

        # Delta
        delta_f = f_feat[:, :, 1:, :, :] - f_feat[:, :, :-1, :, :]
        d = self.delta_act(self._safe_temporal_conv(self.delta_temporal_conv, delta_f))
        d = self.delta_pool(d)
        delta_tokens = d.flatten(2).permute(0, 2, 1)

        tokens = [state_tokens, delta_tokens]
        if self.use_future_stream:
            tokens.append(self._predict_future_tokens(f_feat, delta_f))

        v = torch.cat(tokens, dim=1)
        e = self.output_ln(self.output_proj(v))
        e = e * self.post_ln_scale

        if self.align_use_convex:
            probs = F.softmax(self.align_logits(e), dim=-1)
            aligned = probs @ self.align_codebook
            mix = torch.sigmoid(self.align_mix_logit)
            e = (1.0 - mix) * e + mix * aligned

        return e

    def count_tokens(self) -> int:
        state = self.state_spatial ** 2
        delta = self.delta_temporal * self.delta_spatial ** 2
        future = self.future_temporal * self.future_spatial ** 2 if self.use_future_stream else 0
        return state + delta + future

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
