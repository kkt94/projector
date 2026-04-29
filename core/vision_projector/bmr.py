# Copyright (c) Meta Platforms, Inc. and affiliates.
# BMR projector (Bootstrapped Motion Router):
#   - P_base branch: static semantics from base features.
#   - P_motion branch: motion from base-feature temporal differences.
#   - Learned router: adaptively allocates static/event token budget per sample.
#   - Pretrained MLP warm-start: reuses vision→LLM mapping from pretrained projector.
#
# Token layout (fixed total):
#   [anchor tokens][mixed static/event tokens][recap tokens]

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vision_projector.base import BaseProjector

logger = logging.getLogger(__name__)


def _safe_logit(p: float) -> float:
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


class BMRProjector(BaseProjector):
    """
    Bootstrapped Motion Router (BMR):
      - Static branch (P_base): low-temporal-variation scene/object information.
      - Motion branch (P_motion): first/second-order temporal differences
        from P_base-space features + lightweight residual adapter.
      - Router: predicts per-sample event token budget in a bounded range.

    Supports warm-starting from a pretrained 2-layer MLP projector
    (e.g. MLPBaselineProjector, CRTProjector) via `load_pretrained_mlp()`.
    The pretrained MLP's vision→LLM mapping is decomposed into:
      - channel_reduce: first-layer weight reshaped to Conv3d
      - proj_*: second-layer weight shared across stream projections
    This bootstraps the costly vision→LLM alignment learned in prior stages.
    """

    def __init__(self, args):
        super().__init__()

        self.vision_dim = args.vision_model.get("width", 1024)
        self.llm_dim = args.dim

        self.bottleneck_dim = int(getattr(args, "bmr_bottleneck_dim", 320))
        self.total_tokens = int(getattr(args, "bmr_total_tokens", 128))
        self.num_anchor_tokens = int(getattr(args, "bmr_num_anchor_tokens", 8))
        self.num_recap_tokens = int(getattr(args, "bmr_num_recap_tokens", 16))

        free_tokens = self.total_tokens - self.num_anchor_tokens - self.num_recap_tokens
        if free_tokens <= 0:
            raise ValueError(
                "Invalid BMR token config: bmr_total_tokens must be larger than "
                "bmr_num_anchor_tokens + bmr_num_recap_tokens"
            )
        self.num_free_tokens = int(free_tokens)

        ev_min = int(getattr(args, "bmr_event_min_tokens", 24))
        ev_max = int(getattr(args, "bmr_event_max_tokens", 80))
        self.event_min_tokens = max(1, min(ev_min, self.num_free_tokens))
        self.event_max_tokens = max(
            self.event_min_tokens, min(ev_max, self.num_free_tokens)
        )
        self.event_init_ratio = float(getattr(args, "bmr_event_init_ratio", 0.45))

        self.router_hidden_dim = int(getattr(args, "bmr_router_hidden_dim", 256))
        self.camera_suppress_init = float(
            getattr(args, "bmr_camera_suppress_init", 0.5)
        )

        self.reg_weight = float(getattr(args, "bmr_reg_weight", 0.01))
        self.reg_smooth_weight = float(getattr(args, "bmr_reg_smooth_weight", 1.0))
        self.reg_sparse_weight = float(getattr(args, "bmr_reg_sparse_weight", 0.05))
        self.reg_ortho_weight = float(getattr(args, "bmr_reg_ortho_weight", 0.1))

        self._reg_loss = None
        self._last_event_count = None
        self._last_event_ratio = None

        self.setup_projector(args)

    def setup_projector(self, args=None):
        d = self.bottleneck_dim

        # Shared base feature extractor in projector space.
        self.channel_reduce = nn.Conv3d(
            self.vision_dim,
            d,
            kernel_size=1,
            bias=True,
            dtype=torch.get_default_dtype(),
        )
        self.channel_act = nn.GELU()

        # Spatially-aware EMA gate for static memory.
        self.static_gate = nn.Sequential(
            nn.Conv3d(
                d,
                max(1, d // 4),
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=True,
                dtype=torch.get_default_dtype(),
            ),
            nn.GELU(),
            nn.Conv3d(
                max(1, d // 4),
                1,
                kernel_size=1,
                bias=True,
                dtype=torch.get_default_dtype(),
            ),
        )

        # Motion specialization in P_base space: delta + small residual adapter.
        self.motion_adapter = nn.Sequential(
            nn.Conv3d(
                2 * d,
                d,
                kernel_size=1,
                bias=True,
                dtype=torch.get_default_dtype(),
            ),
            nn.GELU(),
            nn.Conv3d(
                d,
                d,
                kernel_size=1,
                bias=True,
                dtype=torch.get_default_dtype(),
            ),
        )

        # Per-frame camera suppress gate (temporal-aware).
        # Learns a per-frame suppression weight via temporal conv, replacing the
        # single scalar logit for better handling of varying camera motion.
        self.camera_suppress_net = nn.Sequential(
            nn.Conv3d(
                d, max(1, d // 8), kernel_size=(3, 1, 1), padding=(1, 0, 0),
                bias=True, dtype=torch.get_default_dtype(),
            ),
            nn.GELU(),
            nn.Conv3d(
                max(1, d // 8), 1, kernel_size=1,
                bias=True, dtype=torch.get_default_dtype(),
            ),
        )

        # Content-adaptive router.
        self.router = nn.Sequential(
            nn.Linear(2 * d, self.router_hidden_dim, dtype=torch.get_default_dtype()),
            nn.GELU(),
            nn.Linear(self.router_hidden_dim, 1, dtype=torch.get_default_dtype()),
        )

        # Event cross-attention queries (differentiable selection replacing top-k).
        self.event_queries = nn.Parameter(
            torch.zeros(1, self.num_free_tokens, d, dtype=torch.get_default_dtype())
        )
        self.energy_bias_scale = nn.Parameter(
            torch.tensor([1.0], dtype=torch.get_default_dtype())
        )

        # Learnable temperature for recap softmax.
        self.recap_log_tau = nn.Parameter(
            torch.tensor([0.0], dtype=torch.get_default_dtype())
        )

        # Stream-specific projections to LLM space.
        self.proj_anchor = nn.Linear(d, self.llm_dim, bias=True, dtype=torch.get_default_dtype())
        self.proj_static = nn.Linear(d, self.llm_dim, bias=True, dtype=torch.get_default_dtype())
        self.proj_event = nn.Linear(d, self.llm_dim, bias=True, dtype=torch.get_default_dtype())
        self.proj_recap = nn.Linear(d, self.llm_dim, bias=True, dtype=torch.get_default_dtype())

        self.type_anchor = nn.Parameter(torch.zeros(self.llm_dim))
        self.type_static = nn.Parameter(torch.zeros(self.llm_dim))
        self.type_event = nn.Parameter(torch.zeros(self.llm_dim))
        self.type_recap = nn.Parameter(torch.zeros(self.llm_dim))

        self.output_ln = nn.LayerNorm(self.llm_dim)
        # Small initial scale for stable LLM integration (zero-init style):
        # after LayerNorm (unit variance), this shrinks projected tokens so the
        # vision signal starts small and grows during training, preventing
        # early gradient explosion in the frozen/LoRA'd LLM layers.
        self.post_ln_scale = nn.Parameter(
            torch.tensor([1.0 / math.sqrt(self.llm_dim)], dtype=torch.get_default_dtype())
        )

    def init_tensors(self):
        nn.init.xavier_uniform_(self.channel_reduce.weight)
        nn.init.zeros_(self.channel_reduce.bias)

        for m in self.static_gate.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Start with conservative update gate.
        if isinstance(self.static_gate[-1], nn.Conv3d):
            nn.init.constant_(self.static_gate[-1].bias, -1.0)

        for m in self.motion_adapter.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Make motion adapter start close to identity path (delta1).
        if isinstance(self.motion_adapter[-1], nn.Conv3d):
            nn.init.zeros_(self.motion_adapter[-1].weight)
            nn.init.zeros_(self.motion_adapter[-1].bias)

        # Camera suppress: init bias so sigmoid ≈ camera_suppress_init.
        for m in self.camera_suppress_net.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if isinstance(self.camera_suppress_net[-1], nn.Conv3d):
            nn.init.zeros_(self.camera_suppress_net[-1].weight)
            nn.init.constant_(
                self.camera_suppress_net[-1].bias,
                _safe_logit(self.camera_suppress_init),
            )

        for m in self.router.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Bias router toward configured initial event ratio.
        target_event = self.event_init_ratio * float(self.num_free_tokens)
        norm = (target_event - self.event_min_tokens) / max(
            1e-6, float(self.event_max_tokens - self.event_min_tokens)
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, _safe_logit(norm))

        nn.init.normal_(self.event_queries, std=0.02)

        for layer in (self.proj_anchor, self.proj_static, self.proj_event, self.proj_recap):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.type_anchor, std=0.02)
        nn.init.normal_(self.type_static, std=0.02)
        nn.init.normal_(self.type_event, std=0.02)
        nn.init.normal_(self.type_recap, std=0.02)

    # ------------------------------------------------------------------
    # Pretrained MLP projector warm-start
    # ------------------------------------------------------------------

    def load_pretrained_mlp(
        self,
        mlp_state_dict: dict[str, torch.Tensor],
        *,
        key_prefix: str = "projector.",
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """
        Warm-start BMR from a pretrained 2-layer MLP projector.

        The standard MLP projector has the structure:
            projector.0 : Linear(vision_dim → llm_dim)  + bias
            projector.2 : Linear(llm_dim  → llm_dim)  + bias

        This method decomposes the pretrained weights as follows:

        1. First layer (vision_dim → llm_dim) is factored through SVD into:
           - channel_reduce (vision_dim → bottleneck_dim): top-d singular vectors
             of the first-layer weight, reshaped as 1×1×1 Conv3d.
             This preserves the pretrained feature extraction directions.

        2. Second layer (llm_dim → llm_dim) is copied to all stream projections
           (proj_anchor, proj_static, proj_event, proj_recap) via an intermediate
           projection from bottleneck_dim.  Since the pretrained second layer
           operates in llm_dim space, we compose:
             bottleneck(d) → llm_dim  ≈  V_top^T (bottleneck→llm) then W2 (llm→llm)
           For each stream proj, we set: proj_X.weight = W2 @ V_top^T[:llm_dim, :d]

        This ensures the pretrained vision→LLM alignment is preserved at init,
        while the new BMR-specific components (static gate, motion adapter,
        router, camera suppress) start from their own careful initializations.

        Args:
            mlp_state_dict: State dict from a pretrained MLP projector.
            key_prefix: Prefix for the MLP keys (default "projector.").
            strict: If True, raise on missing keys.

        Returns:
            Dict with "loaded" and "skipped" key lists.
        """
        # Strip prefix if present.
        sd = {}
        for k, v in mlp_state_dict.items():
            clean = k[len(key_prefix):] if k.startswith(key_prefix) else k
            sd[clean] = v

        w1_key, b1_key = "0.weight", "0.bias"
        w2_key, b2_key = "2.weight", "2.bias"
        loaded, skipped = [], []

        # --- First layer: vision_dim → llm_dim, factor via SVD ---
        if w1_key in sd:
            w1 = sd[w1_key].float()  # (llm_dim, vision_dim)
            if w1.shape[1] != self.vision_dim:
                logger.warning(
                    f"MLP first-layer input dim {w1.shape[1]} != vision_dim "
                    f"{self.vision_dim}, skipping first-layer warm-start"
                )
                skipped.append(w1_key)
            else:
                # SVD: W1 = U @ diag(S) @ Vh, keep top-d components
                # channel_reduce should map vision_dim → d preserving max variance
                U, S, Vh = torch.linalg.svd(w1, full_matrices=False)
                d = self.bottleneck_dim
                # Vh[:d] are the top-d right singular vectors (d, vision_dim)
                # Scale by sqrt(S) so information splits between reduce and project
                s_sqrt = S[:d].sqrt()
                reduce_w = (s_sqrt.unsqueeze(1) * Vh[:d])  # (d, vision_dim)
                # Reshape for Conv3d: (out_ch, in_ch, 1, 1, 1)
                self.channel_reduce.weight.data.copy_(
                    reduce_w.reshape(d, self.vision_dim, 1, 1, 1).to(
                        self.channel_reduce.weight.dtype
                    )
                )
                # Store U and S for stream projection init (used below)
                _U_top = U[:, :d]  # (llm_dim, d)
                _s_sqrt = s_sqrt    # (d,)
                loaded.append(w1_key)
                logger.info(
                    f"BMR warm-start: channel_reduce from SVD top-{d} of "
                    f"MLP first layer ({w1.shape}), "
                    f"explained variance ratio: "
                    f"{(S[:d].pow(2).sum() / S.pow(2).sum()):.3f}"
                )
        else:
            _U_top, _s_sqrt = None, None
            if strict:
                raise KeyError(f"Missing {key_prefix}{w1_key} in pretrained state dict")
            skipped.append(w1_key)

        if b1_key in sd:
            # First-layer bias is less critical; zero the Conv3d bias
            # (the bias semantics change with the SVD factorization).
            loaded.append(b1_key)
        else:
            skipped.append(b1_key)

        # --- Second layer: llm_dim → llm_dim, compose with SVD basis ---
        if w2_key in sd and _U_top is not None:
            w2 = sd[w2_key].float()  # (llm_dim, llm_dim)
            # The effective stream projection is:
            #   bottleneck(d) --[diag(s_sqrt) @ U_top^T]--> llm_dim --[W2]--> llm_dim
            # Compose: proj_X.weight = W2 @ U_top @ diag(s_sqrt)
            #   shape: (llm_dim, llm_dim) @ (llm_dim, d) @ (d, d) = (llm_dim, d)
            composed = w2 @ (_U_top * _s_sqrt.unsqueeze(0))  # (llm_dim, d)
            composed = composed.to(self.proj_anchor.weight.dtype)
            for proj in (self.proj_anchor, self.proj_static, self.proj_event, self.proj_recap):
                proj.weight.data.copy_(composed)
            loaded.append(w2_key)
            logger.info(
                f"BMR warm-start: stream projections from composed "
                f"MLP second layer ({w2.shape}) @ SVD basis"
            )
        elif w2_key in sd:
            skipped.append(w2_key)
        else:
            if strict:
                raise KeyError(f"Missing {key_prefix}{w2_key} in pretrained state dict")
            skipped.append(w2_key)

        if b2_key in sd:
            b2 = sd[b2_key].to(self.proj_anchor.bias.dtype)
            for proj in (self.proj_anchor, self.proj_static, self.proj_event, self.proj_recap):
                proj.bias.data.copy_(b2)
            loaded.append(b2_key)
        else:
            skipped.append(b2_key)

        result = {"loaded": loaded, "skipped": skipped}
        logger.info(
            f"BMR pretrained MLP warm-start: loaded {len(loaded)} params, "
            f"skipped {len(skipped)}"
        )
        return result

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _temporal_diff(x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        if x.shape[2] <= 1:
            return torch.zeros_like(x)
        delta = x[:, :, 1:] - x[:, :, :-1]
        pad = torch.zeros_like(x[:, :, :1])
        return torch.cat([pad, delta], dim=2)

    @staticmethod
    def _pool_to_tokens(x: torch.Tensor, n_tokens: int) -> torch.Tensor:
        # x: (B, C, L) -> (B, N, C)
        if x.shape[-1] == n_tokens:
            out = x
        else:
            out = F.adaptive_avg_pool1d(x, n_tokens)
        return out.permute(0, 2, 1).contiguous()

    @staticmethod
    def _sinusoidal_time_pe(
        t: int,
        d: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pos = torch.arange(t, device=device, dtype=dtype)
        half = (d + 1) // 2
        freq = 1.0 / (10000.0 ** (torch.arange(half, device=device, dtype=dtype) / max(half, 1)))
        ang = pos[:, None] * freq[None, :]
        pe = torch.zeros(t, d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(ang[:, : (d + 1) // 2])
        pe[:, 1::2] = torch.cos(ang[:, : d // 2])
        return pe

    @staticmethod
    def _temporal_sinusoidal_pe(
        T: int, H: int, W: int, d: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sinusoidal PE along temporal axis, broadcast over spatial dims."""
        t_pos = torch.arange(T, device=device, dtype=dtype)
        half = (d + 1) // 2
        freq = 1.0 / (
            10000.0 ** (torch.arange(half, device=device, dtype=dtype) / max(half, 1))
        )
        angles = t_pos.unsqueeze(1) * freq.unsqueeze(0)  # (T, half)
        pe = torch.zeros(T, d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(angles[:, : (d + 1) // 2])
        pe[:, 1::2] = torch.cos(angles[:, : d // 2])
        return pe[:, None, None, :].expand(T, H, W, d).reshape(T * H * W, d)

    def _build_static_and_motion(
        self, f_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # f_feat: (B, d, T, H, W)
        bsz, _, t, _, _ = f_feat.shape

        if t <= 1:
            s_seq = f_feat
            motion = torch.zeros_like(f_feat)
            smooth_loss = f_feat.new_zeros(())
            sparse_loss = f_feat.new_zeros(())
            return s_seq, motion, smooth_loss, sparse_loss

        alpha = torch.sigmoid(self.static_gate(f_feat))  # (B,1,T,H,W)
        s_prev = f_feat[:, :, 0]
        s_steps = [s_prev]
        for idx in range(1, t):
            a = alpha[:, :, idx]
            h_t = f_feat[:, :, idx]
            s_prev = (1.0 - a) * s_prev + a * h_t
            s_steps.append(s_prev)
        s_seq = torch.stack(s_steps, dim=2)

        d1 = self._temporal_diff(f_feat)
        d2 = self._temporal_diff(d1)
        motion_res = self.motion_adapter(torch.cat([d1, d2], dim=1))
        motion = d1 + motion_res

        # Temporal-aware camera motion suppression: per-frame, per-pixel gate
        # controls how much of the spatial-mean motion is subtracted.
        # Handles varying camera motion across frames (pan, zoom, rotate).
        beta = torch.sigmoid(self.camera_suppress_net(f_feat))  # (B,1,T,H,W)
        spatial_mean = motion.mean(dim=(3, 4), keepdim=True)  # (B,d,T,1,1)
        motion = motion - beta * spatial_mean

        smooth_loss = (s_seq[:, :, 1:] - s_seq[:, :, :-1]).pow(2).mean()
        sparse_loss = motion.abs().mean()
        return s_seq, motion, smooth_loss, sparse_loss

    def _build_anchor_tokens(self, f_feat: torch.Tensor) -> torch.Tensor:
        # Temporal summaries with sinusoidal time signal.
        temporal = f_feat.mean(dim=(3, 4))  # (B, d, T)
        t = temporal.shape[-1]
        pe = self._sinusoidal_time_pe(t, temporal.shape[1], temporal.device, temporal.dtype)
        temporal = temporal + pe.transpose(0, 1).unsqueeze(0)  # (B,d,T)
        return self._pool_to_tokens(temporal, self.num_anchor_tokens)

    def _build_static_candidates(self, s_seq: torch.Tensor) -> torch.Tensor:
        # Exponential weighting favors later (more converged) EMA states.
        t = s_seq.shape[2]
        weights = torch.exp(
            torch.linspace(-2.0, 0.0, t, device=s_seq.device, dtype=s_seq.dtype)
        )
        weights = weights / weights.sum()
        s_map = (s_seq * weights.view(1, 1, t, 1, 1)).sum(dim=2)  # (B,d,H,W)
        bsz, d, h, w = s_map.shape
        return self._pool_to_tokens(s_map.reshape(bsz, d, h * w), self.num_free_tokens)

    def _build_event_candidates(
        self, source: torch.Tensor, is_image: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build event candidate tokens via energy-biased cross-attention.

        For video: attends to motion features (temporal differences).
        For image: attends to spatial features directly (no motion available).

        Uses learned queries + energy bias for fully differentiable selection,
        replacing the previous non-differentiable top-k approach.
        """
        bsz, d, t, h, w = source.shape
        seq_len = t * h * w
        flat = source.permute(0, 2, 3, 4, 1).reshape(bsz, seq_len, d)  # (B,L,d)
        energy = source.pow(2).mean(dim=1).reshape(bsz, seq_len)  # (B,L)

        # Add temporal PE so queries can distinguish when events occur
        pe = self._temporal_sinusoidal_pe(t, h, w, d, source.device, source.dtype)
        flat_pe = flat + pe.unsqueeze(0)  # (B,L,d)

        # Cross-attention: learned queries attend to source features
        queries = self.event_queries.expand(bsz, -1, -1)  # (B, F, d)
        attn_logits = torch.bmm(queries, flat_pe.transpose(1, 2))  # (B, F, L)
        attn_logits = attn_logits / (d ** 0.5)

        # Energy bias steers attention toward high-energy locations
        if not is_image:
            energy_bias = self.energy_bias_scale * torch.log1p(
                energy.unsqueeze(1)  # (B, 1, L)
            )
            attn_logits = attn_logits + energy_bias

        attn = F.softmax(attn_logits, dim=-1)  # (B, F, L)
        # Attend to original features (without PE) to avoid leaking PE into values
        events = torch.bmm(attn, flat)  # (B, F, d)

        return events, energy.view(bsz, t, h, w)

    def _build_recap_tokens(self, f_feat: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        temporal = f_feat.mean(dim=(3, 4))  # (B,d,T)
        energy_t = motion.pow(2).mean(dim=(1, 3, 4))  # (B,T)
        # Learnable temperature prevents attention collapse on single high-energy frame
        tau = self.recap_log_tau.exp().clamp(min=0.1)
        weights = F.softmax(energy_t / tau, dim=-1)
        weighted = temporal * weights.unsqueeze(1)
        return self._pool_to_tokens(weighted, self.num_recap_tokens)

    def _route_budget(
        self,
        static_cand: torch.Tensor,
        event_cand: torch.Tensor,
        is_image: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # static_cand/event_cand: (B, free, d)
        s_stat = static_cand.mean(dim=1)
        e_stat = event_cand.abs().mean(dim=1)
        ratio = torch.sigmoid(self.router(torch.cat([s_stat, e_stat], dim=-1))).squeeze(-1)

        if is_image:
            # For images, force all budget to static (no motion to route).
            ratio = torch.zeros_like(ratio)

        e_float = self.event_min_tokens + (
            self.event_max_tokens - self.event_min_tokens
        ) * ratio
        e_count = torch.round(e_float).long()
        e_count = e_count.clamp(self.event_min_tokens, self.event_max_tokens)
        s_count = self.num_free_tokens - e_count
        return s_count, e_count, e_float

    def _compose_dynamic_tokens(
        self,
        s_tok: torch.Tensor,
        e_tok: torch.Tensor,
        s_count: torch.Tensor,
        e_count: torch.Tensor,
        e_float: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compose static and event tokens with per-sample budget allocation.
        Fully vectorized — no Python loop over batch dimension.
        """
        bsz = s_tok.shape[0]
        F_tok = self.num_free_tokens
        dim = s_tok.shape[-1]

        # Apply soft scaling so router receives gradient from token amplitudes.
        e_ratio = (e_float / float(F_tok)).clamp(0.0, 1.0).view(-1, 1, 1)
        s_ratio = 1.0 - e_ratio
        s_tok = s_tok * (0.5 + s_ratio)
        e_tok = e_tok * (0.5 + e_ratio)

        # Build position indices: [0, 1, ..., F-1] for each sample
        pos = torch.arange(F_tok, device=s_tok.device)  # (F,)

        # Static tokens occupy positions [0, s_count), event tokens [s_count, s_count+e_count)
        s_mask = (pos.unsqueeze(0) < s_count.unsqueeze(1))      # (B, F)
        e_pos = pos.unsqueeze(0) - s_count.unsqueeze(1)          # (B, F) offset for event
        e_mask = (e_pos >= 0) & (e_pos < e_count.unsqueeze(1))  # (B, F)

        # Clamp indices for safe gather (masked positions will be zeroed out)
        s_idx = pos.unsqueeze(0).expand(bsz, -1).clamp(0, F_tok - 1)  # (B, F)
        e_idx = e_pos.clamp(0, F_tok - 1)                              # (B, F)

        # Gather and mask
        s_gathered = torch.gather(
            s_tok, 1, s_idx.unsqueeze(-1).expand(-1, -1, dim)
        )  # (B, F, dim)
        e_gathered = torch.gather(
            e_tok, 1, e_idx.unsqueeze(-1).expand(-1, -1, dim)
        )  # (B, F, dim)

        out = s_gathered * s_mask.unsqueeze(-1) + e_gathered * e_mask.unsqueeze(-1)
        return out

    def forward(self, grid_tokens: torch.Tensor) -> torch.Tensor:
        # grid_tokens: (B, T, H, W, C)
        is_image = grid_tokens.shape[1] == 1

        x = grid_tokens.permute(0, 4, 1, 2, 3).contiguous()
        f_feat = self.channel_act(self.channel_reduce(x))  # (B,d,T,H,W)

        s_seq, motion, smooth_loss, sparse_loss = self._build_static_and_motion(f_feat)
        a_tok = self._build_anchor_tokens(f_feat)
        s_cand = self._build_static_candidates(s_seq)

        if is_image:
            # Images: event queries attend to spatial features directly,
            # since motion is zero for single frames.
            e_cand, _ = self._build_event_candidates(f_feat, is_image=True)
            sparse_loss = f_feat.new_zeros(())
        else:
            e_cand, _ = self._build_event_candidates(motion, is_image=False)

        r_tok = self._build_recap_tokens(f_feat, motion)

        s_count, e_count, e_float = self._route_budget(s_cand, e_cand, is_image=is_image)
        self._last_event_count = e_count.detach()
        self._last_event_ratio = (e_float / float(self.num_free_tokens)).detach()

        # Stream-specific projection to LLM space.
        a = self.proj_anchor(a_tok) + self.type_anchor
        s = self.proj_static(s_cand) + self.type_static
        e = self.proj_event(e_cand) + self.type_event
        r = self.proj_recap(r_tok) + self.type_recap

        m = self._compose_dynamic_tokens(s, e, s_count, e_count, e_float)
        out = torch.cat([a, m, r], dim=1)
        out = self.output_ln(out)
        out = out * self.post_ln_scale

        # Regularization: static smoothness + motion sparsity + stream separation.
        s_mean = F.normalize(s_cand.mean(dim=1), dim=-1, eps=1e-6)
        e_mean = F.normalize(e_cand.mean(dim=1), dim=-1, eps=1e-6)
        ortho_loss = (s_mean * e_mean).sum(dim=-1).pow(2).mean()
        self._reg_loss = self.reg_weight * (
            self.reg_smooth_weight * smooth_loss
            + self.reg_sparse_weight * sparse_loss
            + self.reg_ortho_weight * ortho_loss
        )

        return out

    def get_reg_loss(self) -> torch.Tensor:
        if self._reg_loss is not None:
            return self._reg_loss
        return self.channel_reduce.weight.new_zeros(())

    def count_tokens(self) -> int:
        return self.total_tokens

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
