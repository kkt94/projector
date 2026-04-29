# Copyright (c) Meta Platforms, Inc. and affiliates.
# VLM Loader: load pretrained open-source VLMs and extract components
# for BMR warm-start.
#
# Supported VLMs:
#   - LLaVA-NeXT-Video (lmms-lab/LLaVA-NeXT-Video-*)
#   - InternVL2.5 (OpenGVLab/InternVL2_5-*)
#   - VideoLLaMA3 (DAMO-NLP-SG/VideoLLaMA3-*)
#   - Qwen2.5-VL (Qwen/Qwen2.5-VL-*)
#   - Custom MLP bridge files (project's own format)

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VLM type detection and weight prefix configuration
# ---------------------------------------------------------------------------

# Maps HuggingFace architectures field → internal vlm_type
_ARCH_TO_TYPE = {
    "LlavaNextVideoForConditionalGeneration": "llava_next_video",
    "LlavaNextForConditionalGeneration": "llava_next",
    "LlavaForConditionalGeneration": "llava",
    "InternVLChatModel": "internvl",
    "Qwen2VLForConditionalGeneration": "qwen2_vl",
    "Qwen2_5_VLForConditionalGeneration": "qwen2_vl",
    # VideoLLaMA3 variants
    "Videollama3Qwen2ForCausalLM": "videollama3",
    "VideoLLaMA3ForCausalLM": "videollama3",
}

# Weight key prefix routing per VLM type.
# Each entry maps: component → list of prefixes to try (first match wins).
_PREFIX_MAP = {
    "llava_next_video": {
        "vision": ["vision_tower.vision_model.", "vision_tower."],
        "projector": ["multi_modal_projector."],
        "llm": ["language_model."],
    },
    "llava_next": {
        "vision": ["vision_tower.vision_model.", "vision_tower."],
        "projector": ["multi_modal_projector."],
        "llm": ["language_model."],
    },
    "llava": {
        "vision": ["vision_tower.vision_model.", "vision_tower."],
        "projector": ["multi_modal_projector."],
        "llm": ["language_model."],
    },
    "internvl": {
        "vision": ["vision_model."],
        "projector": ["mlp1."],
        "llm": ["language_model."],
    },
    "qwen2_vl": {
        "vision": ["visual."],
        "projector": ["visual.merger."],  # merger is the projector in Qwen2-VL
        "llm": ["model."],
    },
    "videollama3": {
        "vision": ["model.vision_encoder."],
        "projector": ["model.mm_projector."],
        # LLM keys share `model.` prefix with vision/projector; use specific sub-prefixes
        "llm": ["model.layers.", "model.embed_tokens.", "model.norm.", "lm_head."],
    },
}

# MLP projector key normalization: source key → canonical key.
# Canonical format: "0.weight", "0.bias", "2.weight", "2.bias"
_MLP_KEY_MAPS = {
    "llava_next_video": {
        "linear_1.weight": "0.weight",
        "linear_1.bias": "0.bias",
        "linear_2.weight": "2.weight",
        "linear_2.bias": "2.bias",
    },
    "llava_next": {
        "linear_1.weight": "0.weight",
        "linear_1.bias": "0.bias",
        "linear_2.weight": "2.weight",
        "linear_2.bias": "2.bias",
    },
    "llava": {
        "linear_1.weight": "0.weight",
        "linear_1.bias": "0.bias",
        "linear_2.weight": "2.weight",
        "linear_2.bias": "2.bias",
    },
    "internvl": {
        "0.weight": "0.weight",
        "0.bias": "0.bias",
        "2.weight": "2.weight",
        "2.bias": "2.bias",
    },
    "qwen2_vl": {
        # Qwen2-VL merger may have different structure; fallback to direct copy
        "0.weight": "0.weight",
        "0.bias": "0.bias",
        "2.weight": "2.weight",
        "2.bias": "2.bias",
    },
    "videollama3": {
        # model.mm_projector.readout.{0,2}.{weight,bias}
        "readout.0.weight": "0.weight",
        "readout.0.bias": "0.bias",
        "readout.2.weight": "2.weight",
        "readout.2.bias": "2.bias",
    },
    "custom_mlp": {
        "projector.0.weight": "0.weight",
        "projector.0.bias": "0.bias",
        "projector.2.weight": "2.weight",
        "projector.2.bias": "2.bias",
        # Also handle already-normalized keys
        "0.weight": "0.weight",
        "0.bias": "0.bias",
        "2.weight": "2.weight",
        "2.bias": "2.bias",
    },
}

# Vision encoder normalization constants
_ENCODER_NORM = {
    "clip": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "siglip": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "internvit": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "qwen2_vl": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}


# ---------------------------------------------------------------------------
# VLMWeights container
# ---------------------------------------------------------------------------


@dataclass
class VLMWeights:
    """Container for extracted VLM component weights."""

    vlm_type: str
    vision_encoder_sd: Optional[Dict[str, torch.Tensor]] = None
    vision_encoder_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    mlp_projector_sd: Optional[Dict[str, torch.Tensor]] = None
    llm_sd: Optional[Dict[str, torch.Tensor]] = None
    llm_config: Optional[Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_vlm_type(model_path: str) -> str:
    """Auto-detect VLM type from config.json or file structure.

    Args:
        model_path: Path to HuggingFace model directory or bridge file.

    Returns:
        VLM type string.
    """
    path = Path(model_path)

    # Custom bridge file (.pt)
    if path.is_file() and path.suffix == ".pt":
        return "custom_mlp"

    # HuggingFace model directory
    config_file = path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

        architectures = config.get("architectures", [])
        for arch in architectures:
            if arch in _ARCH_TO_TYPE:
                vlm_type = _ARCH_TO_TYPE[arch]
                logger.info(f"Detected VLM type: {vlm_type} (arch={arch})")
                return vlm_type

        # Heuristic fallbacks
        model_type = config.get("model_type", "")
        if "videollama3" in model_type.lower():
            return "videollama3"
        if "llava" in model_type.lower():
            return "llava_next_video"
        if "internvl" in model_type.lower():
            return "internvl"
        if "qwen2_vl" in model_type.lower():
            return "qwen2_vl"

    raise ValueError(
        f"Cannot detect VLM type from {model_path}. "
        "Please specify vlm_type explicitly."
    )


def load_vlm_weights(
    model_path: str,
    vlm_type: Optional[str] = None,
    load_vision: bool = True,
    load_projector: bool = True,
    load_llm: bool = True,
) -> VLMWeights:
    """Load and partition VLM checkpoint into components.

    Args:
        model_path: Path to HuggingFace model directory or bridge file.
        vlm_type: VLM type override (auto-detected if None).
        load_vision: Whether to load vision encoder weights.
        load_projector: Whether to load MLP projector weights.
        load_llm: Whether to load LLM weights.

    Returns:
        VLMWeights with separated component state dicts.
    """
    if vlm_type is None:
        vlm_type = detect_vlm_type(model_path)

    if vlm_type == "custom_mlp":
        return _load_custom_mlp_bridge(model_path)

    path = Path(model_path)

    # Read config for vision encoder info
    vision_config = _extract_vision_config(path, vlm_type)

    # Load weight files
    sd = _load_hf_state_dict(path)
    logger.info(f"Loaded {len(sd)} weight tensors from {model_path}")

    # Route weights by prefix
    prefixes = _PREFIX_MAP.get(vlm_type)
    if prefixes is None:
        raise ValueError(f"Unknown VLM type: {vlm_type}")

    vision_sd = None
    projector_sd = None
    llm_sd = None

    # Extract projector first so we can exclude its keys from vision
    # (handles Qwen2-VL where visual.merger.* is a subset of visual.*)
    proj_raw_keys = set()
    if load_projector:
        projector_sd = _extract_by_prefix(sd, prefixes["projector"])
        for k in sd:
            for p in prefixes["projector"]:
                if k.startswith(p):
                    proj_raw_keys.add(k)
                    break
        logger.info(f"MLP projector: {len(projector_sd)} tensors")

    if load_vision:
        vision_sd = _extract_by_prefix(sd, prefixes["vision"], exclude_keys=proj_raw_keys)
        logger.info(f"Vision encoder: {len(vision_sd)} tensors")

    if load_llm:
        if vlm_type == "videollama3":
            # VideoLLaMA3 LLM keys don't share a single common prefix
            # (model.layers.*, model.embed_tokens.*, model.norm.*, lm_head.*).
            # Stripping individual prefixes causes key collisions.
            # Instead, extract everything NOT vision/projector and keep keys as-is.
            exclude = list(prefixes["vision"]) + list(prefixes["projector"])
            llm_sd = _extract_by_exclusion(sd, exclude)
        else:
            llm_sd = _extract_by_prefix(sd, prefixes["llm"])
        logger.info(f"LLM: {len(llm_sd)} tensors")

    # Release the full state dict to free memory
    del sd

    return VLMWeights(
        vlm_type=vlm_type,
        vision_encoder_sd=vision_sd,
        vision_encoder_config=vision_config,
        mlp_projector_sd=projector_sd,
        llm_sd=llm_sd,
    )


def extract_mlp_for_bmr(vlm_weights: VLMWeights) -> Dict[str, torch.Tensor]:
    """Normalize MLP projector keys to canonical format for BMR warm-start.

    Canonical keys: "0.weight", "0.bias", "2.weight", "2.bias"

    Args:
        vlm_weights: VLMWeights with mlp_projector_sd populated.

    Returns:
        Dict with canonical MLP keys.
    """
    if vlm_weights.mlp_projector_sd is None:
        raise ValueError("No MLP projector weights available for BMR warm-start")

    key_map = _MLP_KEY_MAPS.get(vlm_weights.vlm_type, _MLP_KEY_MAPS["custom_mlp"])
    src_sd = vlm_weights.mlp_projector_sd
    canonical = {}

    for src_key, dst_key in key_map.items():
        if src_key in src_sd:
            canonical[dst_key] = src_sd[src_key]

    if not canonical:
        # Fallback: try direct numeric keys (already in canonical form)
        for k, v in src_sd.items():
            if k in ("0.weight", "0.bias", "2.weight", "2.bias"):
                canonical[k] = v

    if "0.weight" not in canonical:
        logger.warning(
            f"Could not find first-layer weight in MLP projector. "
            f"Available keys: {list(src_sd.keys())}"
        )

    logger.info(
        f"Extracted MLP for BMR: {list(canonical.keys())} "
        f"(from {vlm_weights.vlm_type})"
    )
    return canonical


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_custom_mlp_bridge(path: str) -> VLMWeights:
    """Load project's own MLP bridge file format."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "bridge_state_dict" in ckpt:
        bridge_sd = ckpt["bridge_state_dict"]
    elif "state_dict" in ckpt:
        bridge_sd = ckpt["state_dict"]
    else:
        bridge_sd = ckpt

    # Convert DTensor if from FSDP
    for key in list(bridge_sd.keys()):
        v = bridge_sd[key]
        if hasattr(v, "_local_tensor"):
            bridge_sd[key] = v._local_tensor.detach().clone()

    return VLMWeights(
        vlm_type="custom_mlp",
        mlp_projector_sd=bridge_sd,
    )


def _extract_vision_config(path: Path, vlm_type: str) -> Dict[str, Any]:
    """Extract vision encoder config from HuggingFace model config."""
    config_file = path / "config.json"
    if not config_file.exists():
        return {}

    with open(config_file) as f:
        config = json.load(f)

    result = {}

    if vlm_type == "videollama3":
        # VideoLLaMA3 uses 'vision_encoder_config' key (SigLIP-based)
        vision_cfg = config.get("vision_encoder_config", config.get("vision_config", {}))
        result["image_size"] = vision_cfg.get("image_size", 384)
        result["patch_size"] = vision_cfg.get("patch_size", 14)
        result["hidden_size"] = vision_cfg.get("hidden_size", 1152)
        result["has_cls_token"] = True
        result["encoder_family"] = "siglip"
        result["num_hidden_layers"] = vision_cfg.get("num_hidden_layers", 27)
        result["intermediate_size"] = vision_cfg.get("intermediate_size", 4304)
        result["num_attention_heads"] = vision_cfg.get("num_attention_heads", 16)
        result["vision_feature_layer"] = config.get("vision_feature_layer", -1)
        # VideoLLaMA3 applies spatial merge after vision encoding
        # video_merge_size=2 (27×27 → 13×13), image_merge_size=1
        result["spatial_merge_size"] = 2

    elif vlm_type in ("llava_next_video", "llava_next", "llava"):
        vision_cfg = config.get("vision_config", {})
        result["image_size"] = vision_cfg.get("image_size", 336)
        result["patch_size"] = vision_cfg.get("patch_size", 14)
        result["hidden_size"] = vision_cfg.get("hidden_size", 1024)
        result["has_cls_token"] = True
        result["encoder_family"] = _detect_encoder_family(vision_cfg, config)
        result["num_hidden_layers"] = vision_cfg.get("num_hidden_layers", 24)
        result["intermediate_size"] = vision_cfg.get("intermediate_size", 4096)
        result["num_attention_heads"] = vision_cfg.get("num_attention_heads", 16)
        result["vision_feature_layer"] = config.get("vision_feature_layer", -2)

    elif vlm_type == "internvl":
        vision_cfg = config.get("vision_config", {})
        result["image_size"] = vision_cfg.get("image_size", 448)
        result["patch_size"] = vision_cfg.get("patch_size", 14)
        result["hidden_size"] = vision_cfg.get("hidden_size", 1024)
        result["has_cls_token"] = True
        result["encoder_family"] = "internvit"
        result["num_hidden_layers"] = vision_cfg.get("num_hidden_layers", 24)
        result["intermediate_size"] = vision_cfg.get("intermediate_size", 4096)
        result["num_attention_heads"] = vision_cfg.get("num_attention_heads", 16)

    elif vlm_type == "qwen2_vl":
        vision_cfg = config.get("vision_config", {})
        result["image_size"] = vision_cfg.get("image_size", 384)
        result["patch_size"] = vision_cfg.get("patch_size", 14)
        # Qwen2-VL uses temporal_patch_size for video
        result["temporal_patch_size"] = vision_cfg.get("temporal_patch_size", 2)
        result["hidden_size"] = vision_cfg.get("embed_dim", vision_cfg.get("hidden_size", 1280))
        result["has_cls_token"] = False  # Qwen2-VL does not use CLS token
        result["encoder_family"] = "qwen2_vl"
        result["num_hidden_layers"] = vision_cfg.get("depth", 32)

    logger.info(f"Vision config: {result}")
    return result


def _detect_encoder_family(vision_cfg: dict, full_config: dict) -> str:
    """Detect whether the vision encoder is CLIP, SigLIP, etc."""
    model_type = vision_cfg.get("model_type", "")
    if "siglip" in model_type.lower():
        return "siglip"
    if "clip" in model_type.lower():
        return "clip"

    # Check full config for hints
    mm_vision_tower = full_config.get("mm_vision_tower", "")
    if "siglip" in mm_vision_tower.lower():
        return "siglip"
    if "clip" in mm_vision_tower.lower():
        return "clip"

    # Default to CLIP for LLaVA-family models
    return "clip"


def _load_hf_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Load all weight shards from a HuggingFace model directory.

    Supports safetensors and pytorch .bin formats.
    """
    sd = {}

    # Try safetensors first
    safetensor_files = sorted(path.glob("*.safetensors"))
    if safetensor_files:
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors package required for loading VLM checkpoints. "
                "Install with: pip install safetensors"
            )

        for sf in safetensor_files:
            shard = load_file(str(sf), device="cpu")
            sd.update(shard)
            logger.info(f"  Loaded shard {sf.name}: {len(shard)} tensors")
        return sd

    # Fallback to pytorch .bin files
    bin_files = sorted(path.glob("pytorch_model*.bin"))
    if not bin_files:
        bin_files = sorted(path.glob("model*.bin"))

    if bin_files:
        for bf in bin_files:
            shard = torch.load(bf, map_location="cpu", weights_only=True)
            sd.update(shard)
            logger.info(f"  Loaded shard {bf.name}: {len(shard)} tensors")
        return sd

    # Single consolidated file
    for pat in ("consolidated.pth", "consolidated.00.pth"):
        cp = path / pat
        if cp.exists():
            sd = torch.load(cp, map_location="cpu", weights_only=True)
            if "model" in sd:
                sd = sd["model"]
            return sd

    raise FileNotFoundError(
        f"No weight files found in {path}. "
        "Expected .safetensors, pytorch_model*.bin, or consolidated.pth"
    )


def _extract_by_prefix(
    sd: Dict[str, torch.Tensor],
    prefixes: List[str],
    exclude_keys: Optional[set] = None,
) -> Dict[str, torch.Tensor]:
    """Extract keys matching any prefix, stripping the matched prefix."""
    extracted = {}
    exclude = exclude_keys or set()
    for key, value in sd.items():
        if key in exclude:
            continue
        for prefix in prefixes:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                extracted[clean_key] = value
                break
    return extracted


def _extract_by_exclusion(
    sd: Dict[str, torch.Tensor],
    exclude_prefixes: List[str],
) -> Dict[str, torch.Tensor]:
    """Extract all keys NOT matching any exclude prefix. Keys kept as-is."""
    extracted = {}
    for key, value in sd.items():
        if not any(key.startswith(p) for p in exclude_prefixes):
            extracted[key] = value
    return extracted


def get_encoder_normalization(encoder_family: str) -> Dict[str, List[float]]:
    """Get normalization constants for a vision encoder family."""
    return _ENCODER_NORM.get(encoder_family, _ENCODER_NORM["clip"])
