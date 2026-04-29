# Copyright (c) Meta Platforms, Inc. and affiliates.

from core.vision_encoder.pe import VisionTransformer as PE_VisionTransformer

# CRT/V-JEPA2 support
try:
    from core.vision_encoder.vjepa2 import VJEPA2VisionEncoder, VJEPA2Config
except ImportError:
    # V-JEPA2 dependencies not installed
    VJEPA2VisionEncoder = None
    VJEPA2Config = None

# A2 grid encoder (raw spatiotemporal grid, no CRT)
try:
    from core.vision_encoder.vjepa2_a2 import VJEPA2GridEncoder
except ImportError:
    VJEPA2GridEncoder = None

# HuggingFace vision encoder wrapper (for VLM base models)
from core.vision_encoder.hf_encoder_wrapper import HFVisionEncoderWrapper

__all__ = [
    "PE_VisionTransformer",
    "VJEPA2VisionEncoder",
    "VJEPA2Config",
    "VJEPA2GridEncoder",
    "HFVisionEncoderWrapper",
]
