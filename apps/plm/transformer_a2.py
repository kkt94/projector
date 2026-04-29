# Copyright (c) Meta Platforms, Inc. and affiliates.
# A2 (Dynamics-aware) LMTransformer for V-JEPA2 + LLM
#
# Uses V-JEPA2 grid tokens → A2 Projector (State/Delta) → Llama/Qwen LLM

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

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None

from core.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    TiedLinear,
    cross_entropy,
)
from core.utils import InitArgs

from core.vision_encoder.vjepa2_a2 import VJEPA2GridEncoder
from core.vision_encoder.hf_encoder_wrapper import HFVisionEncoderWrapper
from core.vision_projector.a2 import DynamicsAwareA2Projector
from core.vision_projector.a3 import A3Projector
from core.vision_projector.a4 import A4Projector
from core.vision_projector.bmr import BMRProjector
from core.vision_projector.mlp_baseline import MLPBaselineProjector

from apps.plm.transformer import (
    create_causal_mask,
    attention_flops_per_token,
    get_num_flop_per_token,
    causal_mask,
)

logger = logging.getLogger(__name__)


def _get_arg(args, name: str, default):
    return getattr(args, name, default)


def compute_num_visual_tokens(args) -> int:
    """Compute visual token budget from projector config."""
    ptype = _get_arg(args, "projector_type", "a2")
    if ptype == "mlp":
        return int(_get_arg(args, "mlp_num_tokens", 64))
    if ptype == "bmr":
        return int(_get_arg(args, "bmr_total_tokens", 128))
    if ptype == "bmr_v2":
        return int(_get_arg(args, "bmr_v2_num_tokens", 128))
    if ptype == "tep":
        return int(_get_arg(args, "tep_num_tokens", 3136))
    if ptype == "tdi":
        return int(_get_arg(args, "tdi_num_tokens", 3151))
    if ptype == "a4":
        return (
            int(_get_arg(args, "a4_num_global_tokens", 16))
            + int(_get_arg(args, "a4_num_static_tokens", 48))
            + int(_get_arg(args, "a4_num_event_tokens", 64))
        )
    if ptype == "a3":
        state = int(_get_arg(args, "a3_state_spatial", 4)) ** 2
        delta = int(_get_arg(args, "a3_delta_temporal", 3)) * int(_get_arg(args, "a3_delta_spatial", 4)) ** 2
        future = 0
        if bool(_get_arg(args, "a3_use_future_stream", False)):
            future = int(_get_arg(args, "a3_future_temporal", 2)) * int(_get_arg(args, "a3_future_spatial", 2)) ** 2
        return state + delta + future
    state = int(_get_arg(args, "a2_state_spatial", 4)) ** 2
    delta = int(_get_arg(args, "a2_delta_temporal", 3)) * int(_get_arg(args, "a2_delta_spatial", 4)) ** 2
    future = 0
    if bool(_get_arg(args, "a2_use_future_stream", False)):
        future = int(_get_arg(args, "a2_future_temporal", 2)) * int(_get_arg(args, "a2_future_spatial", 2)) ** 2
    return state + delta + future


@dataclass
class A2TransformerArgs(BaseTransformerArgs):
    """Arguments for A2-enabled transformer."""

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False

    sliding_window: Optional[int] = None

    freeze_language_model: Optional[bool] = False
    freeze_vision_model: Optional[bool] = True

    # V-JEPA2 config (encoder only, no predictor)
    vision_model: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "width": 1024,
    })
    vjepa2_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "encoder_name": "vit_large",
        "num_frames": 64,
        "native_preproc": True,
        "input_is_plm_normalized": True,
    })

    mlp_init: InitArgs = field(default_factory=InitArgs)
    pooling_ratio: int = 1
    remove_vision_class_token: bool = True

    attn_impl: str = "sdpa"

    # Projector type: "a2", "a3", "a4", "bmr", "bmr_v2", "tep", or "mlp"
    projector_type: str = "a2"

    # A2 projector parameters
    a2_bottleneck_dim: int = 256
    a2_state_spatial: int = 4      # 4×4 = 16 state tokens
    a2_delta_temporal: int = 3     # 3 temporal steps
    a2_delta_spatial: int = 4      # 4×4 spatial → 3×16 = 48 delta tokens
    a2_use_locality_block: bool = True
    a2_use_future_stream: bool = False
    a2_future_temporal: int = 2
    a2_future_spatial: int = 2
    a2_align_use_convex: bool = True
    a2_align_codebook_size: int = 512
    a2_align_mix_ratio: float = 0.2
    a2_post_ln_scale_init: float = 0.022
    # Total: 16 + 48 = 64 visual tokens

    # A3 projector parameters (learned anchor-state / anchored-delta)
    a3_bottleneck_dim: int = 256
    a3_state_spatial: int = 4      # 4×4 = 16 state tokens
    a3_delta_temporal: int = 3     # 3 temporal steps
    a3_delta_spatial: int = 4      # 4×4 spatial → 3×16 = 48 delta tokens
    a3_reg_weight: float = 0.01   # Decomposition regularization weight
    a3_reg_smooth_weight: float = 1.0
    a3_reg_ortho_weight: float = 0.25
    a3_adaptive_gate: bool = True  # Motion-based adaptive state/delta emphasis
    a3_use_locality_block: bool = True
    a3_delta_anchor_blend: float = 0.65
    a3_use_future_stream: bool = False
    a3_future_temporal: int = 2
    a3_future_spatial: int = 2
    a3_align_use_convex: bool = True
    a3_align_codebook_size: int = 512
    a3_align_mix_ratio: float = 0.2
    a3_post_ln_scale_init: float = 0.022

    # MLP baseline parameters
    mlp_num_tokens: int = 64       # Number of visual tokens for MLP baseline

    # A4 projector parameters (single-pass static/event/global decomposition)
    a4_bottleneck_dim: int = 320
    a4_num_static_tokens: int = 48
    a4_num_event_tokens: int = 64
    a4_num_global_tokens: int = 16
    a4_token_order: str = "gse"    # gse|sge|seg
    a4_reg_weight: float = 0.01
    a4_reg_smooth_weight: float = 1.0
    a4_reg_sparse_weight: float = 0.05

    # BMR projector parameters (bootstrapped motion router)
    bmr_bottleneck_dim: int = 320
    bmr_total_tokens: int = 128
    bmr_num_anchor_tokens: int = 8
    bmr_num_recap_tokens: int = 16
    bmr_event_min_tokens: int = 24
    bmr_event_max_tokens: int = 80
    bmr_event_init_ratio: float = 0.45
    bmr_router_hidden_dim: int = 256
    bmr_camera_suppress_init: float = 0.5
    bmr_reg_weight: float = 0.01
    bmr_reg_smooth_weight: float = 1.0
    bmr_reg_sparse_weight: float = 0.05
    bmr_reg_ortho_weight: float = 0.1
    bmr_pretrained_mlp_path: Optional[str] = None  # Path to pretrained MLP projector bridge for warm-start

    # BMRv2 projector parameters (dual-MLP + temporal residual gating)
    bmr_v2_num_tokens: int = 128
    bmr_v2_lora_rank: int = 64
    bmr_v2_lora_alpha: float = 16.0
    bmr_v2_til_kernel: int = 3
    bmr_v2_gate_mode: str = "per_sample"  # "per_sample" | "per_token"
    bmr_v2_kd_weight: float = 0.1

    # TEP projector parameters (Temporal Explicitation Projector)
    tep_bottleneck: int = 128           # Bottleneck dim for temporal attention
    tep_num_layers: int = 3             # Number of temporal transformer layers
    tep_num_heads: int = 8              # Attention heads in temporal transformer
    tep_max_frames: int = 32            # Max frames for temporal PE
    tep_use_frame_pe: bool = True       # Post-MLP frame position encoding
    tep_num_tokens: int = 3136          # Output token count (T*S)

    # TDI projector parameters (Temporal Difference Injection)
    tdi_num_tokens: int = 3151          # Output token count: T*196 + (T-1) = 3151

    # VLM base model (load vision encoder + MLP projector + LLM from open-source VLM)
    vlm_base_path: Optional[str] = None       # HF model path (e.g., "DAMO-NLP-SG/VideoLLaMA3-2B")
    vlm_type: Optional[str] = None            # Auto-detected if None
    vlm_load_vision_encoder: bool = True       # Load vision encoder from VLM
    vlm_load_llm: bool = True                  # Load LLM weights from VLM
    vlm_load_projector_for_bmr: bool = True    # Use VLM's MLP projector for BMR warm-start

    # LoRA parameters (Stage 2 SFT)
    lora_rank: int = 0             # 0 = disabled, >0 = enable LoRA
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_targets: str = "wq,wv"   # Comma-separated list of target module names


class A2LMTransformer(BaseTransformer):
    """
    LMTransformer with Dynamics-aware A2 Projector.

    V-JEPA2 grid tokens → State/Delta 2-stream decomposition → LLM.

    State stream captures static scene composition (objects, background).
    Delta stream captures temporal dynamics (actions, events, changes).

    This preserves temporal information structurally (via convolutions)
    rather than relying on the LLM to infer it from a flat token sequence.
    """

    def __init__(self, args: A2TransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Lazy-initialized after model is on device
        self.vision_model = None
        self.vision_projector = None
        self.vjepa2_config = args.vjepa2_config or {}

        # Store args for later initialization
        self._args = args

        self.freeze_vision_model = args.freeze_vision_model
        self.freeze_language_model = args.freeze_language_model

    def init_vision_model(self, device: str = "cuda"):
        """Initialize vision encoder and projector.

        If vlm_base_path is set, loads from an open-source VLM (LLaVA, InternVL, etc.).
        Otherwise, uses V-JEPA2 grid encoder (original path).
        """
        if self.vision_model is not None:
            return

        vlm_base_path = getattr(self._args, "vlm_base_path", None)
        if vlm_base_path:
            self._init_from_vlm(device)
        else:
            self._init_from_vjepa2(device)

    def _init_from_vjepa2(self, device: str):
        """Original V-JEPA2 initialization path."""
        logger.info("Initializing V-JEPA2 Grid Encoder for A2...")

        vjepa2_args = {
            "encoder_name": self.vjepa2_config.get("encoder_name", "vit_large"),
            "num_frames": self.vjepa2_config.get("num_frames", 64),
            "device": device,
            "native_preproc": self.vjepa2_config.get("native_preproc", True),
            "input_is_plm_normalized": self.vjepa2_config.get("input_is_plm_normalized", True),
        }

        self.vision_model = VJEPA2GridEncoder(**vjepa2_args)
        self.vision_model.load_from_checkpoint(None)

        self._init_projector(self.vision_model.width)

    def _init_from_vlm(self, device: str):
        """Initialize from a pretrained open-source VLM."""
        from core.vlm_loader import load_vlm_weights, extract_mlp_for_bmr, detect_vlm_type

        vlm_path = self._args.vlm_base_path
        vlm_type = getattr(self._args, "vlm_type", None)
        load_vision = getattr(self._args, "vlm_load_vision_encoder", True)
        load_proj = getattr(self._args, "vlm_load_projector_for_bmr", True)
        # LLM loading is handled in train_a2.py, not here
        load_llm = False

        # Auto-detect VLM type for native encoder decision
        if vlm_type is None:
            vlm_type = detect_vlm_type(vlm_path)

        logger.info(f"Loading VLM components from {vlm_path} (type={vlm_type})")
        vlm_weights = load_vlm_weights(
            vlm_path,
            vlm_type=vlm_type,
            load_vision=load_vision,
            load_projector=load_proj,
            load_llm=load_llm,
        )

        # Set up vision encoder
        vision_config = vlm_weights.vision_encoder_config or {}
        encoder_family = vision_config.get("encoder_family", "clip")
        vision_width = vision_config.get("hidden_size", 1024)

        self.vision_model = HFVisionEncoderWrapper(
            encoder_family=encoder_family,
            vision_config=vision_config,
            device=device,
        )

        # Use NATIVE encoder for VLMs with custom architectures (rotary pos emb, etc.)
        # to ensure exact feature match with the pretrained MLP projector.
        _NATIVE_VLM_TYPES = ("videollama3",)
        if load_vision and vlm_type in _NATIVE_VLM_TYPES:
            self.vision_model.load_native_from_vlm(vlm_path)
            logger.info(f"VLM NATIVE vision encoder loaded: {vlm_type}, width={vision_width}")
        elif load_vision and vlm_weights.vision_encoder_sd:
            self.vision_model.load_from_state_dict(
                vlm_weights.vision_encoder_sd, vision_config
            )
            logger.info(f"VLM vision encoder loaded: {encoder_family}, width={vision_width}")
        else:
            logger.info("VLM vision encoder weights not loaded (will need manual init)")

        # Override vision_model width for projector
        self._args.vision_model = {"width": vision_width}

        # Create projector
        self._init_projector(vision_width)

        # BMR/BMRv2/TEP warm-start from VLM's MLP projector
        if (
            load_proj
            and vlm_weights.mlp_projector_sd
            and self._args.projector_type in ("bmr", "bmr_v2", "tep")
        ):
            mlp_sd = extract_mlp_for_bmr(vlm_weights)
            result = self.vision_projector.load_pretrained_mlp(mlp_sd, key_prefix="")
            ptype_name = self._args.projector_type.upper()
            logger.info(
                f"{ptype_name} warm-start from VLM MLP: "
                f"loaded={result['loaded']}, skipped={result['skipped']}"
            )

        # TDI: load frozen MLP as a separate module (not inside TDI)
        if (
            load_proj
            and vlm_weights.mlp_projector_sd
            and self._args.projector_type == "tdi"
        ):
            mlp_sd = extract_mlp_for_bmr(vlm_weights)
            self._load_tdi_frozen_mlp(mlp_sd, vision_width)

    def _load_tdi_frozen_mlp(self, mlp_sd: dict, vision_width: int):
        """Load the VLM's MLP projector as a separate frozen module for TDI."""
        D = self._args.dim
        C = vision_width
        dtype = torch.get_default_dtype()

        self.tdi_frozen_mlp = nn.Sequential(
            nn.Linear(C, D, bias=True, dtype=dtype),
            nn.GELU(),
            nn.Linear(D, D, bias=True, dtype=dtype),
        )

        key_map = {
            "0.weight": (self.tdi_frozen_mlp[0], "weight"),
            "0.bias": (self.tdi_frozen_mlp[0], "bias"),
            "2.weight": (self.tdi_frozen_mlp[2], "weight"),
            "2.bias": (self.tdi_frozen_mlp[2], "bias"),
        }
        loaded = []
        for key, (module, attr) in key_map.items():
            if key in mlp_sd:
                getattr(module, attr).data.copy_(mlp_sd[key].to(dtype))
                loaded.append(key)

        for p in self.tdi_frozen_mlp.parameters():
            p.requires_grad = False

        logger.info(f"TDI frozen MLP loaded as separate module: {len(loaded)} params")

    def _init_projector(self, vision_width: int):
        """Create the vision projector based on projector_type."""
        # Create projector based on type
        class ProjectorArgs:
            def __init__(self, args, vision_width):
                self.dim = args.dim
                self.vision_model = {"width": vision_width}
                self.mlp_init = args.mlp_init
                # A2 params
                self.a2_bottleneck_dim = args.a2_bottleneck_dim
                self.a2_state_spatial = args.a2_state_spatial
                self.a2_delta_temporal = args.a2_delta_temporal
                self.a2_delta_spatial = args.a2_delta_spatial
                self.a2_use_locality_block = args.a2_use_locality_block
                self.a2_use_future_stream = args.a2_use_future_stream
                self.a2_future_temporal = args.a2_future_temporal
                self.a2_future_spatial = args.a2_future_spatial
                self.a2_align_use_convex = args.a2_align_use_convex
                self.a2_align_codebook_size = args.a2_align_codebook_size
                self.a2_align_mix_ratio = args.a2_align_mix_ratio
                self.a2_post_ln_scale_init = args.a2_post_ln_scale_init
                # A3 params
                self.a3_bottleneck_dim = args.a3_bottleneck_dim
                self.a3_state_spatial = args.a3_state_spatial
                self.a3_delta_temporal = args.a3_delta_temporal
                self.a3_delta_spatial = args.a3_delta_spatial
                self.a3_reg_weight = args.a3_reg_weight
                self.a3_reg_smooth_weight = args.a3_reg_smooth_weight
                self.a3_reg_ortho_weight = args.a3_reg_ortho_weight
                self.a3_adaptive_gate = args.a3_adaptive_gate
                self.a3_use_locality_block = args.a3_use_locality_block
                self.a3_delta_anchor_blend = args.a3_delta_anchor_blend
                self.a3_use_future_stream = args.a3_use_future_stream
                self.a3_future_temporal = args.a3_future_temporal
                self.a3_future_spatial = args.a3_future_spatial
                self.a3_align_use_convex = args.a3_align_use_convex
                self.a3_align_codebook_size = args.a3_align_codebook_size
                self.a3_align_mix_ratio = args.a3_align_mix_ratio
                self.a3_post_ln_scale_init = args.a3_post_ln_scale_init
                # MLP params
                self.mlp_num_tokens = args.mlp_num_tokens
                # A4 params
                self.a4_bottleneck_dim = args.a4_bottleneck_dim
                self.a4_num_static_tokens = args.a4_num_static_tokens
                self.a4_num_event_tokens = args.a4_num_event_tokens
                self.a4_num_global_tokens = args.a4_num_global_tokens
                self.a4_token_order = args.a4_token_order
                self.a4_reg_weight = args.a4_reg_weight
                self.a4_reg_smooth_weight = args.a4_reg_smooth_weight
                self.a4_reg_sparse_weight = args.a4_reg_sparse_weight
                # BMR params
                self.bmr_bottleneck_dim = args.bmr_bottleneck_dim
                self.bmr_total_tokens = args.bmr_total_tokens
                self.bmr_num_anchor_tokens = args.bmr_num_anchor_tokens
                self.bmr_num_recap_tokens = args.bmr_num_recap_tokens
                self.bmr_event_min_tokens = args.bmr_event_min_tokens
                self.bmr_event_max_tokens = args.bmr_event_max_tokens
                self.bmr_event_init_ratio = args.bmr_event_init_ratio
                self.bmr_router_hidden_dim = args.bmr_router_hidden_dim
                self.bmr_camera_suppress_init = args.bmr_camera_suppress_init
                self.bmr_reg_weight = args.bmr_reg_weight
                self.bmr_reg_smooth_weight = args.bmr_reg_smooth_weight
                self.bmr_reg_sparse_weight = args.bmr_reg_sparse_weight
                self.bmr_reg_ortho_weight = args.bmr_reg_ortho_weight
                # BMRv2 params
                self.bmr_v2_num_tokens = args.bmr_v2_num_tokens
                self.bmr_v2_lora_rank = args.bmr_v2_lora_rank
                self.bmr_v2_lora_alpha = args.bmr_v2_lora_alpha
                self.bmr_v2_til_kernel = args.bmr_v2_til_kernel
                self.bmr_v2_gate_mode = args.bmr_v2_gate_mode
                self.bmr_v2_kd_weight = args.bmr_v2_kd_weight
                # TEP params
                self.tep_bottleneck = args.tep_bottleneck
                self.tep_num_layers = args.tep_num_layers
                self.tep_num_heads = args.tep_num_heads
                self.tep_max_frames = args.tep_max_frames
                self.tep_use_frame_pe = args.tep_use_frame_pe
                # TDI params (lightweight — no bottleneck/layers/heads)

        proj_args = ProjectorArgs(self._args, vision_width)

        if self._args.projector_type == "mlp":
            self.vision_projector = MLPBaselineProjector(proj_args)
            logger.info(
                f"MLP Baseline Projector initialized: "
                f"{self.vision_projector.count_params():,} params, "
                f"{self.vision_projector.count_tokens()} tokens/video"
            )
        elif self._args.projector_type == "a3":
            self.vision_projector = A3Projector(proj_args)
            logger.info(
                f"A3 Projector initialized: {self.vision_projector.count_params():,} params, "
                f"{self.vision_projector.count_tokens()} tokens/video "
                f"(state={self._args.a3_state_spatial**2}, "
                f"delta={self._args.a3_delta_temporal * self._args.a3_delta_spatial**2}, "
                f"future={self._args.a3_future_temporal * self._args.a3_future_spatial**2 if self._args.a3_use_future_stream else 0}, "
                f"reg_weight={self._args.a3_reg_weight})"
            )
        elif self._args.projector_type == "a4":
            self.vision_projector = A4Projector(proj_args)
            logger.info(
                f"A4 Projector initialized: {self.vision_projector.count_params():,} params, "
                f"{self.vision_projector.count_tokens()} tokens/video "
                f"(global={self._args.a4_num_global_tokens}, "
                f"static={self._args.a4_num_static_tokens}, "
                f"event={self._args.a4_num_event_tokens}, "
                f"order={self._args.a4_token_order})"
            )
        elif self._args.projector_type == "bmr":
            self.vision_projector = BMRProjector(proj_args)
            logger.info(
                f"BMR Projector initialized: {self.vision_projector.count_params():,} params, "
                f"{self.vision_projector.count_tokens()} tokens/video "
                f"(anchor={self._args.bmr_num_anchor_tokens}, "
                f"recap={self._args.bmr_num_recap_tokens}, "
                f"event_range=[{self._args.bmr_event_min_tokens}, {self._args.bmr_event_max_tokens}])"
            )
            # Warm-start from pretrained MLP projector if path is provided.
            mlp_path = getattr(self._args, "bmr_pretrained_mlp_path", None)
            if mlp_path:
                logger.info(f"Loading pretrained MLP projector for BMR warm-start: {mlp_path}")
                import os
                if not os.path.exists(mlp_path):
                    raise FileNotFoundError(f"BMR pretrained MLP path not found: {mlp_path}")
                ckpt = torch.load(mlp_path, map_location="cpu", weights_only=False)
                # Support both raw state_dict and bridge format
                if "bridge_state_dict" in ckpt:
                    mlp_sd = ckpt["bridge_state_dict"]
                elif "state_dict" in ckpt:
                    mlp_sd = ckpt["state_dict"]
                else:
                    mlp_sd = ckpt
                # Convert DTensor if from FSDP checkpoint
                for key in list(mlp_sd.keys()):
                    v = mlp_sd[key]
                    if hasattr(v, '_local_tensor'):
                        mlp_sd[key] = v._local_tensor.detach().clone()
                result = self.vision_projector.load_pretrained_mlp(mlp_sd)
                logger.info(
                    f"BMR warm-start complete: loaded={result['loaded']}, "
                    f"skipped={result['skipped']}"
                )
        elif self._args.projector_type == "bmr_v2":
            from core.vision_projector.bmr_v2 import BMRv2Projector
            self.vision_projector = BMRv2Projector(proj_args)
            logger.info(
                f"BMRv2 Projector initialized: "
                f"{self.vision_projector.count_tokens()} tokens/video "
                f"(lora_rank={self._args.bmr_v2_lora_rank}, "
                f"til_kernel={self._args.bmr_v2_til_kernel}, "
                f"gate={self._args.bmr_v2_gate_mode}, "
                f"kd_weight={self._args.bmr_v2_kd_weight})"
            )
        elif self._args.projector_type == "tep":
            from core.vision_projector.tep import TemporalExplicitationProjector
            self.vision_projector = TemporalExplicitationProjector(proj_args)
            logger.info(
                f"TEP Projector initialized: "
                f"{self.vision_projector.count_params():,} trainable params, "
                f"{self.vision_projector.count_frozen_params():,} frozen params "
                f"(bottleneck={self._args.tep_bottleneck}, "
                f"layers={self._args.tep_num_layers}, "
                f"heads={self._args.tep_num_heads}, "
                f"frame_pe={self._args.tep_use_frame_pe})"
            )
        elif self._args.projector_type == "tdi":
            from core.vision_projector.tdi import TemporalDifferenceInjector
            self.vision_projector = TemporalDifferenceInjector(proj_args)
            logger.info(
                f"TDI Projector initialized: "
                f"{self.vision_projector.count_params():,} trainable params "
                f"(diff_norm + diff_adapter, frozen MLP is separate module)"
            )
        else:
            self.vision_projector = DynamicsAwareA2Projector(proj_args)
            logger.info(
                f"A2 Projector initialized: {self.vision_projector.count_params():,} params, "
                f"{self.vision_projector.count_tokens()} tokens/video "
                f"(state={self._args.a2_state_spatial**2}, "
                f"delta={self._args.a2_delta_temporal * self._args.a2_delta_spatial**2}, "
                f"future={self._args.a2_future_temporal * self._args.a2_future_spatial**2 if self._args.a2_use_future_stream else 0})"
            )

        # Ensure projector-specific explicit init logic is applied in training path.
        if hasattr(self.vision_projector, "init_tensors"):
            self.vision_projector.init_tensors()

    def train(self, mode: bool = True):
        super().train(mode=mode)
        ptype = getattr(self._args, "projector_type", "")
        for name, param in self.named_parameters():
            if "vision_model" in name:
                param.requires_grad = mode and not self.freeze_vision_model
            elif "vision_projector" in name:
                if ptype == "bmr_v2" and "mlp_base" in name:
                    # P_base is always frozen in BMRv2/v3
                    param.requires_grad = False
                elif ptype == "tep" and "frozen_mlp" in name:
                    # Frozen MLP is always frozen in TEP
                    param.requires_grad = False
                else:
                    param.requires_grad = mode
            elif "tdi_frozen_mlp" in name:
                # TDI's frozen MLP is always frozen (separate module)
                param.requires_grad = False
            else:
                param.requires_grad = mode and not self.freeze_language_model
        return self

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[AttentionBias, torch.Tensor, str]] = None,
        videos: Optional[torch.Tensor] = None,
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
        Forward pass with A2 video embeddings.

        Args:
            token_values: [B, SeqLen] token IDs
            target: [B, SeqLen] target token IDs for loss
            videos: [B, T, C, H, W] video frames
            video_pos_index: [B, SeqLen] position indices for video tokens
            loss_mask: [B, SeqLen] mask for loss computation
        """
        _, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)

        # Handle video input with A2 projector
        if videos is not None and self.vision_model is not None:
            # Extract grid tokens: (B, T_p, H_p, W_p, embed_dim)
            grid_tokens = self.vision_model(videos)

            # Match dtype to projector weights (V-JEPA2 returns float32)
            grid_tokens = grid_tokens.to(dtype=h.dtype)

            # Project through projector: (B, M, dim)
            if self._args.projector_type == "tdi":
                # TDI: frozen MLP → visual tokens → TDI (diff + interleave)
                B_v, T_v, H_v, W_v, C_v = grid_tokens.shape
                flat = grid_tokens.reshape(B_v * T_v * H_v * W_v, C_v)
                visual_all = self.tdi_frozen_mlp(flat)
                visual_tokens = visual_all.reshape(B_v, T_v, H_v * W_v, self._args.dim)
                h_video = self.vision_projector(visual_tokens)
            else:
                h_video = self.vision_projector(grid_tokens)

            # Stitch video tokens into text
            pos_index = video_pos_index if video_pos_index is not None else image_pos_index
            h = self.stitch_video_into_text(
                h, h_video, pos_index,
                num_chunks=num_chunks,
                media_type=media_type,
            )

        # Legacy image support
        elif images is not None and self.vision_model is not None:
            if images.dim() == 4:
                images = images.unsqueeze(1)  # [B, 1, C, H, W]
            grid_tokens = self.vision_model(images)
            grid_tokens = grid_tokens.to(dtype=h.dtype)
            h_img = self.vision_projector(grid_tokens)
            h = self.stitch_video_into_text(
                h, h_img, image_pos_index,
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
            loss = cross_entropy(logits, target)
            # Add projector regularization loss (A3)
            if self.training and hasattr(self.vision_projector, 'get_reg_loss'):
                loss = loss + self.vision_projector.get_reg_loss()
            return loss
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
            h_video: [B, M, dim] video token embeddings (M = 64 for A2)
            video_pos_index: [B, SeqLen] position indices (-1 for non-video)
        """
        if video_pos_index is None:
            return h_tok

        cumulative_indices = list(itertools.accumulate(num_chunks, initial=0))

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

        if len(non_text_indices) == 0:
            return h_tok

        flat_video = h_video[non_text_indices].flatten(0, 1)
        n_slots = video_indices_B.shape[0]
        n_feats = flat_video.shape[0]
        if n_slots == 0 or n_feats == 0:
            return h_tok

        n = min(n_slots, n_feats)
        h_tok[video_indices_B[:n], video_indices_L[:n]] = flat_video[:n]
        return h_tok

    def reset_parameters(self, init_std=None):
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0, std=init_std,
            a=-3 * init_std, b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0, std=init_std,
                a=-3 * init_std, b=3 * init_std,
            )


# ── FSDP grouping ───────────────────────────────────────────────────────────

def build_fsdp_grouping_plan(model_args: A2TransformerArgs):
    group_plan: List[Tuple[str, bool]] = []
    group_plan.append(("tok_embeddings", False))
    group_plan.append(("vision_model", False))
    group_plan.append(("vision_projector", False))
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", True))
    group_plan.append(("output", True))
    return group_plan


def tp_parallelize(model, tp_mesh, model_args: A2TransformerArgs, distributed_args):
    from apps.plm.transformer import tp_parallelize as original_tp_parallelize
    return original_tp_parallelize(model, tp_mesh, model_args, distributed_args)


def get_no_recompute_ops():
    return None
