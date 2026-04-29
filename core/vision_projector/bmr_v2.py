# BMRv3: Dual-Path MLP with Temporal Feature Recycling (TFR)
#
# Architecture:
#   P_base:   frozen pretrained MLP — preserves vision→LLM alignment
#   P_motion: trainable MLP copy + TFR between layers — adds temporal awareness
#   Fusion:   per-token motion-aware gate blends P_base and P_motion
#
# Key changes from BMRv2:
#   1. No pooling: keeps VLM native token count (e.g. 5408 = 32×169)
#   2. P_motion MLP is directly trainable (no LoRA — small enough at ~4.1M)
#   3. TFR replaces TIL: depthwise temporal Conv1d + motion-energy gating (~6K params)
#   4. Per-token fusion gate driven by motion energy (2 params vs 393K)
#
# Catastrophic forgetting prevention:
#   - P_base completely frozen → zero forgetting on spatial alignment
#   - Fusion gate init ≈ 0 → starts as pure P_base
#   - TFR gate init ≈ 0 → temporal injection starts dormant
#   - KD loss on static regions: MSE(P_motion, P_base) * (1 - motion_energy)
#   - Image (T=1): alpha forced to 0 → pure P_base
#
# Output: (B, T*S, llm_dim) — same per-token format as original VLM projector

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vision_projector.base import BaseProjector

logger = logging.getLogger(__name__)


class BMRv2Projector(BaseProjector):
    """
    Dual-Path MLP with Temporal Feature Recycling (BMR v3).

    P_base:   Frozen pretrained 2-layer MLP. Produces per-frame tokens.
    P_motion: Trainable MLP copy + TFR. Produces temporally-aware tokens.
    Fusion:   per-token gate driven by motion energy.

    For images (T=1): alpha = 0 → pure P_base output.
    For video:        alpha is per-token, driven by motion energy.
    """

    def __init__(self, args):
        super().__init__()

        self.vision_dim = args.vision_model.get("width", 1024)
        self.llm_dim = args.dim
        self.num_tokens = int(getattr(args, "bmr_v2_num_tokens", 5408))
        self.til_kernel_size = int(getattr(args, "bmr_v2_til_kernel", 3))
        self.kd_weight = float(getattr(args, "bmr_v2_kd_weight", 0.1))

        self._reg_loss: Optional[torch.Tensor] = None

        self.setup_projector(args)

    def setup_projector(self, args=None):
        dtype = torch.get_default_dtype()
        V, D = self.vision_dim, self.llm_dim

        # ═══════════════════════════════════════════════
        # P_base: Frozen 2-layer MLP (loaded from VLM)
        # ═══════════════════════════════════════════════
        self.mlp_base = nn.Sequential(
            nn.Linear(V, D, bias=True, dtype=dtype),
            nn.GELU(),
            nn.Linear(D, D, bias=True, dtype=dtype),
        )

        # ═══════════════════════════════════════════════
        # P_motion: Trainable 2-layer MLP (copied from VLM)
        # ═══════════════════════════════════════════════
        self.mlp_motion_layer1 = nn.Linear(V, D, bias=True, dtype=dtype)
        self.mlp_motion_act = nn.GELU()
        self.mlp_motion_layer2 = nn.Linear(D, D, bias=True, dtype=dtype)

        # ═══════════════════════════════════════════════
        # TFR: Temporal Feature Recycling
        # Depthwise temporal Conv1d + motion-energy gating
        # Inserted between P_motion Layer1 and Layer2.
        # ═══════════════════════════════════════════════
        k = self.til_kernel_size
        self.tfr_temporal_conv = nn.Conv1d(
            D, D, kernel_size=k, padding=k // 2,
            groups=D, bias=True, dtype=dtype,
        )
        self.tfr_gate_bias = nn.Parameter(torch.tensor([-3.0], dtype=dtype))
        self.tfr_energy_scale = nn.Parameter(torch.tensor([1.0], dtype=dtype))

        # ═══════════════════════════════════════════════
        # Per-Token Fusion Gate (motion-energy driven)
        # ═══════════════════════════════════════════════
        self.fusion_bias = nn.Parameter(torch.tensor([-3.0], dtype=dtype))
        self.fusion_energy_scale = nn.Parameter(torch.tensor([1.0], dtype=dtype))

        # ═══════════════════════════════════════════════
        # Output normalization
        # ═══════════════════════════════════════════════
        self.output_ln = nn.LayerNorm(D, dtype=dtype)

    def init_tensors(self):
        """Initialize trainable parameters.

        Called AFTER load_pretrained_mlp() in the training pipeline,
        so we only init the new BMRv3-specific components here.
        MLP base weights are loaded from VLM separately.
        """
        # TFR temporal conv: identity init (center=1, rest=0)
        nn.init.zeros_(self.tfr_temporal_conv.weight)
        center = self.til_kernel_size // 2
        with torch.no_grad():
            for c in range(self.tfr_temporal_conv.weight.shape[0]):
                self.tfr_temporal_conv.weight[c, 0, center] = 1.0
        nn.init.zeros_(self.tfr_temporal_conv.bias)

        # TFR gate: init to near-zero injection
        self.tfr_gate_bias.data.fill_(-3.0)  # sigmoid(-3) ≈ 0.047
        self.tfr_energy_scale.data.fill_(1.0)

        # Fusion gate: init to near-zero (pure P_base at start)
        self.fusion_bias.data.fill_(-3.0)  # sigmoid(-3) ≈ 0.047
        self.fusion_energy_scale.data.fill_(1.0)

        logger.info(
            "BMRv3 init: TFR conv=identity, TFR gate≈0.047, "
            "fusion gate≈0.047 → pure P_base at init"
        )

    # ------------------------------------------------------------------
    # Pretrained MLP loading
    # ------------------------------------------------------------------

    def load_pretrained_mlp(
        self,
        mlp_state_dict: dict[str, torch.Tensor],
        *,
        key_prefix: str = "projector.",
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """
        Load pretrained 2-layer MLP weights into both P_base and P_motion.

        P_base: loaded + frozen (requires_grad=False)
        P_motion: loaded + trainable (requires_grad=True)
        """
        sd = {}
        for k, v in mlp_state_dict.items():
            clean = k[len(key_prefix):] if k.startswith(key_prefix) else k
            sd[clean] = v

        loaded, skipped = [], []

        key_targets = {
            "0.weight": (self.mlp_base[0], "weight", self.mlp_motion_layer1, "weight"),
            "0.bias": (self.mlp_base[0], "bias", self.mlp_motion_layer1, "bias"),
            "2.weight": (self.mlp_base[2], "weight", self.mlp_motion_layer2, "weight"),
            "2.bias": (self.mlp_base[2], "bias", self.mlp_motion_layer2, "bias"),
        }

        for key, (base_mod, base_attr, motion_mod, motion_attr) in key_targets.items():
            if key in sd:
                tensor = sd[key]
                base_param = getattr(base_mod, base_attr)
                motion_param = getattr(motion_mod, motion_attr)

                if tensor.shape != base_param.shape:
                    logger.warning(
                        f"Shape mismatch for {key}: "
                        f"checkpoint={tensor.shape}, model={base_param.shape}"
                    )
                    skipped.append(key)
                    continue

                base_param.data.copy_(tensor.to(base_param.dtype))
                motion_param.data.copy_(tensor.to(motion_param.dtype))
                loaded.append(key)
            else:
                if strict:
                    raise KeyError(f"Missing {key_prefix}{key} in pretrained state dict")
                skipped.append(key)

        # Freeze P_base entirely; P_motion stays trainable
        for p in self.mlp_base.parameters():
            p.requires_grad = False

        result = {"loaded": loaded, "skipped": skipped}
        logger.info(
            f"BMRv3 MLP loaded: {len(loaded)} params into P_base+P_motion, "
            f"skipped {len(skipped)}. P_base frozen, P_motion trainable."
        )
        return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, grid_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_tokens: (B, T, H, W, C) from vision encoder

        Returns:
            (B, T*S, llm_dim) visual tokens for LLM (no pooling)
        """
        B, T, H, W, C = grid_tokens.shape
        is_image = (T == 1)
        S = H * W
        L = T * S

        # ═══ P_base: frozen per-frame MLP ═══
        flat = grid_tokens.reshape(B * L, C)
        with torch.no_grad():
            base_out = self.mlp_base(flat)
        base_out = base_out.reshape(B, L, self.llm_dim)

        # ═══ P_motion: trainable MLP + TFR ═══
        h = self.mlp_motion_layer1(flat)
        h = self.mlp_motion_act(h)
        h = h.reshape(B, T, S, self.llm_dim)

        # TFR: temporal feature recycling (skip for images)
        energy = None
        if not is_image:
            h, energy = self._tfr(h)

        h = h.reshape(B * L, self.llm_dim)
        motion_out = self.mlp_motion_layer2(h)
        motion_out = motion_out.reshape(B, L, self.llm_dim)

        # ═══ Per-Token Fusion Gate ═══
        if is_image or energy is None:
            alpha = base_out.new_zeros(B, L, 1)
        else:
            energy_flat = energy.reshape(B, L, 1)
            alpha = torch.sigmoid(
                self.fusion_bias + self.fusion_energy_scale * energy_flat
            )

        # Residual gating: starts as P_base, learns to blend in P_motion
        out = base_out + alpha * (motion_out - base_out)
        out = self.output_ln(out)

        # ═══ KD regularization loss (on static regions) ═══
        self._reg_loss = None
        if self.training and self.kd_weight > 0 and not is_image and energy is not None:
            energy_flat = energy.reshape(B, L, 1)
            energy_norm = torch.sigmoid(energy_flat - energy_flat.mean())
            kd_mask = 1.0 - energy_norm  # strong on static, weak on motion
            self._reg_loss = self.kd_weight * (
                (motion_out - base_out.detach()).pow(2) * kd_mask
            ).mean()

        return out

    # ------------------------------------------------------------------
    # Temporal Feature Recycling (TFR)
    # ------------------------------------------------------------------

    def _tfr(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Temporal Feature Recycling in MLP intermediate space.

        Applies depthwise temporal Conv1d with motion-energy gating.
        Static regions (low energy) → near-zero injection.
        Dynamic regions (high energy) → temporal context injected.

        Args:
            h: (B, T, S, D) features after MLP layer 1 + activation

        Returns:
            h_out: (B, T, S, D) temporally-enhanced features
            energy: (B, T, S, 1) per-position motion energy
        """
        B, T, S, D = h.shape

        # Motion energy: per-position temporal difference magnitude
        dh = h[:, 1:] - h[:, :-1]  # (B, T-1, S, D)
        energy = dh.norm(dim=-1, keepdim=True)  # (B, T-1, S, 1)
        energy = F.pad(energy, (0, 0, 0, 0, 1, 0))  # (B, T, S, 1) — first frame = 0

        # Depthwise temporal Conv1d: each spatial position processed along T
        # (B, T, S, D) → (B*S, D, T) for Conv1d
        h_perm = h.permute(0, 2, 3, 1).reshape(B * S, D, T)
        h_conv = self.tfr_temporal_conv(h_perm)  # (B*S, D, T)
        h_conv = h_conv.reshape(B, S, D, T).permute(0, 3, 1, 2)  # (B, T, S, D)

        # Motion-aware gated injection
        gate = torch.sigmoid(self.tfr_gate_bias + self.tfr_energy_scale * energy)
        h_out = h + gate * h_conv

        return h_out, energy

    # ------------------------------------------------------------------
    # API (compatible with other projectors)
    # ------------------------------------------------------------------

    def get_reg_loss(self) -> torch.Tensor:
        """Return KD regularization loss for this forward pass."""
        if self._reg_loss is not None:
            return self._reg_loss
        return self.mlp_base[0].weight.new_zeros(())

    def count_tokens(self) -> int:
        return self.num_tokens

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gate_alpha(self) -> Optional[torch.Tensor]:
        """Return last fusion gate value for logging (call after forward)."""
        return getattr(self, "_last_alpha", None)
