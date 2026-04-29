# Copyright (c) Meta Platforms, Inc. and affiliates.
# CRT (Causal Rollout Tokenization) enabled LMTransformer for V-JEPA2 + LLM

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
try:
    from xformers.ops import AttentionBias
except ImportError:
    AttentionBias = None

# BlockMask is only available in PyTorch 2.5+
try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None  # Fallback for older PyTorch versions

from core.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    TiedLinear,
    cross_entropy,
)
from core.utils import InitArgs

# Import CRT components
from core.vision_encoder.vjepa2 import VJEPA2VisionEncoder
from core.vision_projector.crt import CRTProjector

# Import original components for fallback
from apps.plm.transformer import (
    create_causal_mask,
    attention_flops_per_token,
    get_num_flop_per_token,
    causal_mask,
)

logger = logging.getLogger(__name__)


@dataclass
class CRTTransformerArgs(BaseTransformerArgs):
    """Arguments for CRT-enabled transformer."""

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False

    sliding_window: Optional[int] = None

    freeze_language_model: Optional[bool] = False
    freeze_vision_model: Optional[bool] = True  # V-JEPA2 always frozen

    # V-JEPA2 + CRT configuration
    vision_model: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "width": 1024,  # V-JEPA2 embed_dim
    })
    vjepa2_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "encoder_name": "vit_large",
        "num_segments": 8,
        "rollout_steps": 3,
        "spatial_grid": 6,
        "num_frames": 64,
        "pretrained": True,
    })

    mlp_init: InitArgs = field(default_factory=InitArgs)
    pooling_ratio: int = 1
    remove_vision_class_token: bool = True

    attn_impl: str = "sdpa"

    # CRT-specific args (can override vjepa2_config)
    num_segments: int = 8
    rollout_steps: int = 3
    spatial_grid: int = 6


class CRTLMTransformer(BaseTransformer):
    """
    LMTransformer with CRT (Causal Rollout Tokenization) support.

    Uses V-JEPA2 as the vision encoder and CRTProjector as the vision projector.
    The CRT approach decomposes video into:
      - CAUSE: observed states (spatial grid preserved)
      - PRED: predicted future states
      - INNO: innovation (unexpected events)

    This enables temporal reasoning by explicitly encoding:
      1. What was observed
      2. What was expected to happen
      3. What actually happened (surprise/innovation)
    """

    def __init__(self, args: CRTTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(
                args.dim,
                args.vocab_size,
                bias=False,
            )

        # Initialize V-JEPA2 vision model (lazy loading)
        self.vision_model = None
        self.vision_projector = None
        self.vjepa2_config = args.vjepa2_config or {}

        # Override with explicit args if provided
        self.num_segments = args.num_segments
        self.rollout_steps = args.rollout_steps
        self.spatial_grid = args.spatial_grid

        # Store args for later initialization
        self._args = args

        self.freeze_vision_model = args.freeze_vision_model
        self.freeze_language_model = args.freeze_language_model

    def init_vision_model(self, device: str = "cuda"):
        """
        Initialize V-JEPA2 vision model and CRT projector.
        Called after model is moved to device.
        """
        if self.vision_model is not None:
            return  # Already initialized

        logger.info("Initializing V-JEPA2 Vision Encoder...")

        # Create V-JEPA2 encoder
        vjepa2_args = {
            "encoder_name": self.vjepa2_config.get("encoder_name", "vit_large"),
            "num_segments": self.num_segments,
            "rollout_steps": self.rollout_steps,
            "spatial_grid": self.spatial_grid,
            "num_frames": self.vjepa2_config.get("num_frames", 64),
            "device": device,
        }

        self.vision_model = VJEPA2VisionEncoder(**vjepa2_args)
        self.vision_model.load_from_checkpoint(None)  # Uses default pretrained

        # Create CRT projector
        # Create args object with required attributes
        class ProjectorArgs:
            def __init__(self, args, vision_width):
                self.dim = args.dim
                self.vision_model = {"width": vision_width}
                self.mlp_init = args.mlp_init
                self.num_segments = args.num_segments
                self.rollout_steps = args.rollout_steps
                self.spatial_grid = args.spatial_grid

        proj_args = ProjectorArgs(self._args, self.vision_model.width)
        self.vision_projector = CRTProjector(proj_args)

        logger.info(f"CRT Projector initialized: {self.vision_projector.count_params():,} params")
        logger.info(f"Token count per video: {self.vision_projector.count_tokens()}")

    def train(self, mode: bool = True):
        super().train(mode=mode)
        for name, param in self.named_parameters():
            if "vision_model" in name:
                # V-JEPA2 is always frozen
                param.requires_grad = False
            elif "vision_projector" in name:
                # CRT projector is always trained
                param.requires_grad = mode
            else:
                param.requires_grad = mode and not self.freeze_language_model
        return self

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[AttentionBias, torch.Tensor, str]] = None,
        videos: Optional[torch.Tensor] = None,  # [B, T, C, H, W]
        video_pos_index: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        num_chunks: List[int] = [1],
        media_type: List[str] = ["video"],
        attn_impl: str = "sdpa",
        # Legacy image support
        images: Optional[torch.Tensor] = None,
        image_pos_index: Optional[torch.Tensor] = None,
        aspect_ratios: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with CRT video embeddings.

        Args:
            token_values: [B, SeqLen] token IDs
            target: [B, SeqLen] target token IDs for loss
            videos: [B, T, C, H, W] video frames
            video_pos_index: [B, SeqLen] position indices for video tokens
            loss_mask: [B, SeqLen] mask for loss computation
            ...
        """
        _, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)

        # Handle video input with CRT
        if videos is not None and self.vision_model is not None:
            # Extract CRT embeddings: {cause, pred, inno}
            crt_embeddings = self.vision_model(videos)

            # Project to LLM space
            h_video = self.vision_projector(crt_embeddings)  # [B, P, dim]

            # Stitch video tokens into text
            pos_index = video_pos_index if video_pos_index is not None else image_pos_index
            h = self.stitch_video_into_text(
                h,
                h_video,
                pos_index,
                num_chunks=num_chunks,
                media_type=media_type,
            )

        # Legacy image support (fallback)
        elif images is not None and self.vision_model is not None:
            # Treat images as single-frame videos
            if images.dim() == 4:
                images = images.unsqueeze(1)  # [B, 1, C, H, W]
            crt_embeddings = self.vision_model(images)
            h_img = self.vision_projector(crt_embeddings)
            h = self.stitch_video_into_text(
                h,
                h_img,
                image_pos_index,
                num_chunks=num_chunks,
                media_type=media_type,
            )

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        logits = self.output(self.norm(h))
        if target is not None:
            logits = logits[loss_mask]
            target = target[loss_mask]
            return cross_entropy(logits, target)
        else:
            return logits

    def stitch_video_into_text(
        self,
        h_tok: torch.Tensor,
        h_video: torch.Tensor,
        video_pos_index: torch.Tensor,
        num_chunks: List[int],
        media_type: List[str],
    ):
        """
        Insert video embeddings into text token sequence.

        Args:
            h_tok: [B, SeqLen, dim] text token embeddings
            h_video: [B, P, dim] video token embeddings (P = K*G*G + 2R)
            video_pos_index: [B, SeqLen] position indices (-1 for non-video)
        """
        # Generate cumulative indices for each sample
        cumulative_indices = list(itertools.accumulate(num_chunks, initial=0))

        # Get indices for non-text samples
        non_text_indices = [
            idx
            for start, end, m_type in zip(
                cumulative_indices[:-1], cumulative_indices[1:], media_type
            )
            if m_type != "text"
            for idx in range(start, end)
        ]

        video_indices_B, video_indices_L = torch.where(video_pos_index >= 0)
        valid_index_filter = video_indices_L < h_tok.shape[1]
        video_indices_L = video_indices_L[valid_index_filter]
        video_indices_B = video_indices_B[valid_index_filter]

        h_tok[video_indices_B, video_indices_L] = h_video[non_text_indices].flatten(0, 1)[
            valid_index_filter
        ]
        return h_tok

    def reset_parameters(self, init_std=None):
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


# FSDP grouping plan for CRT model
def build_fsdp_grouping_plan(model_args: CRTTransformerArgs):
    group_plan: List[Tuple[str, bool]] = []

    group_plan.append(("tok_embeddings", False))
    group_plan.append(("vision_model", False))
    group_plan.append(("vision_projector", False))

    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", True))

    group_plan.append(("output", True))

    return group_plan


# Tensor parallelism (same as original)
def tp_parallelize(model, tp_mesh, model_args: CRTTransformerArgs, distributed_args):
    # Import from original
    from apps.plm.transformer import tp_parallelize as original_tp_parallelize
    return original_tp_parallelize(model, tp_mesh, model_args, distributed_args)


def get_no_recompute_ops():
    return None
