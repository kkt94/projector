# Copyright (c) Meta Platforms, Inc. and affiliates.
# A4 projector: single-pass static/dynamic/global decomposition with fixed
# visual token budget.  Supports both video and image inputs.
#
# v2 improvements over v1:
#   - Spatial-aware EMA gate (per-pixel alpha via lightweight temporal conv net)
#   - Exponentially-weighted static pooling (preserves EMA convergence)
#   - Energy-biased cross-attention for event selection (fully differentiable)
#   - Temporal sinusoidal encoding on dynamic residuals
#   - Optional lightweight locality enhancement (Honeybee-inspired)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vision_projector.base import BaseProjector


class A4Projector(BaseProjector):
    """
    A4 single-pass projector (video & image):
      - Static memory tokens (S): persistent scene/object content via spatial-aware EMA.
      - Event tokens (E): energy-biased cross-attention over dynamic residuals (video)
        or spatial features as detail tokens (image).
      - Global tokens (G): clip-level temporal summary.

    For images (T=1), event queries attend to spatial features directly instead
    of near-zero dynamic residuals, so all 128 tokens carry useful information.

    Token order is configurable (default: G -> S -> E) to better match
    autoregressive decoding where earlier prefix tokens condition later ones.
    """

    def __init__(self, args):
        super().__init__()

        self.vision_dim = args.vision_model.get("width", 1024)
        self.llm_dim = args.dim

        self.bottleneck_dim = int(getattr(args, "a4_bottleneck_dim", 320))
        self.num_static_tokens = int(getattr(args, "a4_num_static_tokens", 48))
        self.num_event_tokens = int(getattr(args, "a4_num_event_tokens", 64))
        self.num_global_tokens = int(getattr(args, "a4_num_global_tokens", 16))
        self.token_order = str(getattr(args, "a4_token_order", "gse")).lower()

        self.reg_weight = float(getattr(args, "a4_reg_weight", 0.01))
        self.reg_smooth_weight = float(getattr(args, "a4_reg_smooth_weight", 1.0))
        self.reg_sparse_weight = float(getattr(args, "a4_reg_sparse_weight", 0.05))

        self.use_locality_block = bool(getattr(args, "a4_use_locality_block", False))

        self._reg_loss = None
        self.setup_projector(args)

    def setup_projector(self, args=None):
        d = self.bottleneck_dim

        # --- Channel reduction ---
        self.channel_reduce = nn.Conv3d(
            self.vision_dim,
            d,
            kernel_size=1,
            bias=True,
            dtype=torch.get_default_dtype(),
        )
        self.channel_act = nn.GELU()

        # --- Optional locality enhancement (Honeybee-inspired DW+PW) ---
        if self.use_locality_block:
            self.locality_dw = nn.Conv3d(
                d, d, kernel_size=3, stride=1, padding=1,
                groups=d, bias=True, dtype=torch.get_default_dtype(),
            )
            self.locality_pw = nn.Conv3d(
                d, d, kernel_size=1, bias=True,
                dtype=torch.get_default_dtype(),
            )
            self.locality_act = nn.GELU()
            self.locality_gate = nn.Parameter(torch.zeros(1))

        # --- Spatial-aware motion gate ---
        # Temporal kernel=3 gives the network a 3-frame receptive field to
        # implicitly detect motion without explicit frame differencing.
        self.motion_net = nn.Sequential(
            nn.Conv3d(
                d, d // 4, kernel_size=(3, 1, 1), padding=(1, 0, 0),
                bias=True, dtype=torch.get_default_dtype(),
            ),
            nn.GELU(),
            nn.Conv3d(
                d // 4, 1, kernel_size=1,
                bias=True, dtype=torch.get_default_dtype(),
            ),
        )

        # --- Event cross-attention ---
        self.event_queries = nn.Parameter(
            torch.zeros(1, self.num_event_tokens, d, dtype=torch.get_default_dtype())
        )
        self.energy_bias_scale = nn.Parameter(
            torch.tensor([1.0], dtype=torch.get_default_dtype())
        )

        # --- Output projections (separate per stream) ---
        self.proj_static = nn.Linear(
            d, self.llm_dim, bias=True, dtype=torch.get_default_dtype()
        )
        self.proj_event = nn.Linear(
            d, self.llm_dim, bias=True, dtype=torch.get_default_dtype()
        )
        self.proj_global = nn.Linear(
            d, self.llm_dim, bias=True, dtype=torch.get_default_dtype()
        )

        self.type_static = nn.Parameter(torch.zeros(self.llm_dim))
        self.type_event = nn.Parameter(torch.zeros(self.llm_dim))
        self.type_global = nn.Parameter(torch.zeros(self.llm_dim))

        self.output_ln = nn.LayerNorm(self.llm_dim)
        self.post_ln_scale = nn.Parameter(
            torch.tensor([1.0 / (self.llm_dim ** 0.5)], dtype=torch.get_default_dtype())
        )

    def init_tensors(self):
        nn.init.xavier_uniform_(self.channel_reduce.weight)
        nn.init.zeros_(self.channel_reduce.bias)

        for layer in (self.proj_static, self.proj_event, self.proj_global):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.type_static, std=0.02)
        nn.init.normal_(self.type_event, std=0.02)
        nn.init.normal_(self.type_global, std=0.02)

        # Motion net: xavier init, last layer bias → sigmoid ≈ 0.31 (mild smoothing)
        for m in self.motion_net.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.motion_net[-1].bias, -0.8)

        nn.init.normal_(self.event_queries, std=0.02)

        if self.use_locality_block:
            nn.init.zeros_(self.locality_gate)

    # ------------------------------------------------------------------
    # Static memory via spatial-aware EMA
    # ------------------------------------------------------------------

    def _build_static_memory(
        self, f_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build static/dynamic decomposition using spatial-aware EMA.

        Args:
            f_feat: (B, d, T, H, W)
        Returns:
            s_seq: (B, d, T, H, W) static memory trajectory
            d_seq: (B, d, T, H, W) dynamic residual trajectory
            smooth_loss: scalar
        """
        bsz, _, t, _, _ = f_feat.shape

        if t < 2:
            pad = f_feat[:, :, -1:].expand(-1, -1, 2 - t, -1, -1)
            f_feat = torch.cat([f_feat, pad], dim=2)
            t = f_feat.shape[2]

        # Per-pixel, per-frame gate: (B, 1, T, H, W)
        alpha = torch.sigmoid(self.motion_net(f_feat))

        s_prev = f_feat[:, :, 0]
        s_steps = [s_prev]
        d_steps = [f_feat.new_zeros(s_prev.shape)]

        for idx in range(1, t):
            a = alpha[:, :, idx]  # (B, 1, H, W) — spatial-aware
            h_t = f_feat[:, :, idx]
            s_prev = (1.0 - a) * s_prev + a * h_t
            s_steps.append(s_prev)
            d_steps.append(h_t - s_prev)

        s_seq = torch.stack(s_steps, dim=2)
        d_seq = torch.stack(d_steps, dim=2)

        smooth_loss = (s_seq[:, :, 1:] - s_seq[:, :, :-1]).pow(2).mean()
        return s_seq, d_seq, smooth_loss

    # ------------------------------------------------------------------
    # Token builders
    # ------------------------------------------------------------------

    def _pool_to_tokens(self, x: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """x: (B, d, L) -> (B, N, d)"""
        if x.shape[-1] == n_tokens:
            out = x
        else:
            out = F.adaptive_avg_pool1d(x, n_tokens)
        return out.permute(0, 2, 1).contiguous()

    def _make_global_tokens(self, f_feat: torch.Tensor) -> torch.Tensor:
        g_seq = f_feat.mean(dim=(3, 4))  # (B, d, T)
        return self._pool_to_tokens(g_seq, self.num_global_tokens)

    def _make_static_tokens(self, s_seq: torch.Tensor) -> torch.Tensor:
        """Exponentially-weighted temporal pooling preserving EMA convergence."""
        t = s_seq.shape[2]
        # Later EMA states are more converged → give them higher weight.
        weights = torch.exp(
            torch.linspace(-2.0, 0.0, t, device=s_seq.device, dtype=s_seq.dtype)
        )
        weights = weights / weights.sum()
        s_map = (s_seq * weights.view(1, 1, t, 1, 1)).sum(dim=2)  # (B, d, H, W)

        bsz, d, h, w = s_map.shape
        s_flat = s_map.reshape(bsz, d, h * w)
        return self._pool_to_tokens(s_flat, self.num_static_tokens)

    def _make_event_tokens(
        self, d_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy-biased cross-attention over dynamic residuals."""
        bsz, d, t, h, w = d_seq.shape
        L = t * h * w
        d_flat = d_seq.permute(0, 2, 3, 4, 1).reshape(bsz, L, d)  # (B, L, d)

        # Temporal sinusoidal PE so queries can distinguish when events occur
        pe = self._temporal_sinusoidal_pe(t, h, w, d, d_seq.device, d_seq.dtype)
        d_flat_pe = d_flat + pe.unsqueeze(0)  # (B, L, d)

        # Energy bias: steer attention toward high-energy dynamic residuals
        energy = d_seq.pow(2).mean(dim=1).reshape(bsz, 1, L)  # (B, 1, L)
        energy_bias = self.energy_bias_scale * torch.log1p(energy)

        # Cross-attention: learned queries attend to dynamic residuals
        queries = self.event_queries.expand(bsz, -1, -1)  # (B, E, d)
        attn_logits = torch.bmm(queries, d_flat_pe.transpose(1, 2))  # (B, E, L)
        attn_logits = attn_logits / (d ** 0.5) + energy_bias
        attn = F.softmax(attn_logits, dim=-1)  # (B, E, L)

        # Attend to original features (without PE) to avoid leaking PE into values
        events = torch.bmm(attn, d_flat)  # (B, E, d)

        sparse_loss = d_seq.abs().mean()
        return events, sparse_loss

    @staticmethod
    def _temporal_sinusoidal_pe(
        T: int, H: int, W: int, d: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sinusoidal PE along the temporal axis, broadcast over spatial dims."""
        t_pos = torch.arange(T, device=device, dtype=dtype)
        half = (d + 1) // 2
        freq = 1.0 / (
            10000.0 ** (torch.arange(half, device=device, dtype=dtype) / max(half, 1))
        )
        angles = t_pos.unsqueeze(1) * freq.unsqueeze(0)  # (T, half)

        pe = torch.zeros(T, d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(angles[:, : (d + 1) // 2])
        pe[:, 1::2] = torch.cos(angles[:, : d // 2])

        # Expand: (T, d) -> (T, H, W, d) -> (T*H*W, d)
        return pe[:, None, None, :].expand(T, H, W, d).reshape(T * H * W, d)

    # ------------------------------------------------------------------
    # Compose & forward
    # ------------------------------------------------------------------

    def _compose_tokens(
        self, g_tok: torch.Tensor, s_tok: torch.Tensor, e_tok: torch.Tensor
    ) -> torch.Tensor:
        g = self.proj_global(g_tok) + self.type_global
        s = self.proj_static(s_tok) + self.type_static
        e = self.proj_event(e_tok) + self.type_event

        if self.token_order == "seg":
            out = torch.cat([s, e, g], dim=1)
        elif self.token_order == "sge":
            out = torch.cat([s, g, e], dim=1)
        else:  # default gse
            out = torch.cat([g, s, e], dim=1)

        out = self.output_ln(out)
        out = out * self.post_ln_scale
        return out

    def forward(self, grid_tokens: torch.Tensor) -> torch.Tensor:
        # grid_tokens: (B, T, H, W, C)
        is_image = grid_tokens.shape[1] == 1

        x = grid_tokens.permute(0, 4, 1, 2, 3).contiguous()
        f_feat = self.channel_act(self.channel_reduce(x))  # (B, d, T, H, W)

        if self.use_locality_block:
            local = self.locality_act(self.locality_dw(f_feat))
            local = self.locality_pw(local)
            f_feat = f_feat + torch.sigmoid(self.locality_gate) * local

        s_seq, d_seq, smooth_loss = self._build_static_memory(f_feat)
        g_tok = self._make_global_tokens(f_feat)
        s_tok = self._make_static_tokens(s_seq)

        if is_image:
            # Images: event queries attend to spatial features as detail tokens,
            # since dynamic residuals are near-zero for single frames.
            e_tok, _ = self._make_event_tokens(f_feat)
            sparse_loss = f_feat.new_zeros(())
        else:
            # Videos: event queries attend to dynamic residuals as intended.
            e_tok, sparse_loss = self._make_event_tokens(d_seq)

        self._reg_loss = self.reg_weight * (
            self.reg_smooth_weight * smooth_loss + self.reg_sparse_weight * sparse_loss
        )

        return self._compose_tokens(g_tok, s_tok, e_tok)

    def get_reg_loss(self) -> torch.Tensor:
        if self._reg_loss is not None:
            return self._reg_loss
        return self.channel_reduce.weight.new_zeros(())

    def count_tokens(self) -> int:
        return self.num_global_tokens + self.num_static_tokens + self.num_event_tokens

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
