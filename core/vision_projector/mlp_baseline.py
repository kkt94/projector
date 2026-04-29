# Copyright (c) Meta Platforms, Inc. and affiliates.
# MLP Baseline Projector for V-JEPA2 grid tokens
#
# Simple 2-layer MLP baseline for comparison with A2 projector.
# Flattens spatiotemporal grid → pool to N tokens → MLP → LLM dim.

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vision_projector.base import BaseProjector


class MLPBaselineProjector(BaseProjector):
    """
    Simple 2-layer MLP projector for V-JEPA2 grid tokens (baseline).

    Architecture:
      1. Flatten grid: (B, T, H, W, C) → (B, T*H*W, C)
      2. Adaptive average pool to N tokens: (B, N, C)
      3. MLP: Linear(C, llm_dim) → GELU → Linear(llm_dim, llm_dim)

    This serves as a baseline to measure the benefit of A2's
    State/Delta decomposition over naive pooling + MLP.
    """

    def __init__(self, args):
        super().__init__()

        self.vision_dim = args.vision_model.get("width", 1024)
        self.llm_dim = args.dim
        self.num_tokens = getattr(args, "mlp_num_tokens", 64)

        self.setup_projector(args)

    def setup_projector(self, args=None):
        self.projector = nn.Sequential(
            nn.Linear(
                self.vision_dim, self.llm_dim,
                bias=True, dtype=torch.get_default_dtype(),
            ),
            nn.GELU(),
            nn.Linear(
                self.llm_dim, self.llm_dim,
                bias=True, dtype=torch.get_default_dtype(),
            ),
        )

    def init_tensors(self):
        nn.init.xavier_uniform_(self.projector[0].weight)
        nn.init.zeros_(self.projector[0].bias)
        nn.init.xavier_uniform_(self.projector[2].weight)
        nn.init.zeros_(self.projector[2].bias)

    def forward(self, grid_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_tokens: (B, T, H, W, C) from V-JEPA2 grid encoder

        Returns:
            (B, N, llm_dim) visual tokens for LLM
        """
        B, T, H, W, C = grid_tokens.shape

        # Flatten spatiotemporal grid to sequence
        x = grid_tokens.reshape(B, T * H * W, C)  # (B, L, C)

        # Pool to fixed token count
        # Reshape for avg pool: (B, C, L) → pool → (B, C, N) → (B, N, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = F.adaptive_avg_pool1d(x, self.num_tokens)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # 2-layer MLP
        x = self.projector(x)  # (B, N, llm_dim)

        return x

    def count_tokens(self) -> int:
        return self.num_tokens

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
