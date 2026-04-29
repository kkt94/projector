# HuggingFace Vision Encoder Wrapper
#
# Wraps HF vision encoders (CLIP, SigLIP, InternViT) to produce
# the same (B, T, H_p, W_p, embed_dim) output as VJEPA2GridEncoder.
#
# Videos are encoded per-frame and stacked along the time axis.
# BMR reconstructs temporal signals from the resulting grid.

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# PLM normalization: mean=0.5, std=0.5
_PLM_MEAN = [0.5, 0.5, 0.5]
_PLM_STD = [0.5, 0.5, 0.5]

# Encoder-specific normalization constants
_NORM_CONSTANTS = {
    "clip": {"mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]},
    "siglip": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "internvit": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "qwen2_vl": {"mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]},
}


class HFVisionEncoderWrapper(nn.Module):
    """
    Wraps a HuggingFace vision encoder to produce VJEPA2GridEncoder-compatible output.

    Output shape: (B, T, H_p, W_p, embed_dim)

    For video input, each frame is encoded independently and results are
    stacked along the temporal dimension. BMR's temporal convolutions and
    motion adapter then reconstruct temporal dynamics from the per-frame features.

    Supports two modes:
      - Standard HF: builds SigLIP/CLIP from config + loads weights (generic)
      - Native VLM:  loads the actual VLM encoder (e.g. VideoLLaMA3's custom
        SigLIP with rotary pos embeddings) for exact feature matching
    """

    def __init__(
        self,
        encoder_family: str,
        vision_config: Dict[str, Any],
        device: str = "cuda",
    ):
        super().__init__()
        self.encoder_family = encoder_family
        self.image_size = vision_config.get("image_size", 336)
        self.patch_size = vision_config.get("patch_size", 14)
        self.width = vision_config.get("hidden_size", 1024)
        self.has_cls_token = vision_config.get("has_cls_token", True)
        self.vision_feature_layer = vision_config.get("vision_feature_layer", -1)
        self.device = device

        self.spatial_merge_size = vision_config.get("spatial_merge_size", 1)

        # Native patch grid (encoder runs at original resolution)
        self.h_patches = self.image_size // self.patch_size
        self.w_patches = self.image_size // self.patch_size

        # Post-merge grid size (e.g., 27→13 with merge_size=2, using floor division)
        m = max(self.spatial_merge_size, 1)
        self.h_patches_merged = self.h_patches // m
        self.w_patches_merged = self.w_patches // m

        # Encoder will be loaded via load_from_state_dict, load_from_pretrained, or load_native_from_vlm
        self.encoder: Optional[nn.Module] = None
        self._native_mode = False  # True when using VLM's native encoder

        # Register normalization buffers
        norm = _NORM_CONSTANTS.get(encoder_family, _NORM_CONSTANTS["clip"])
        self.register_buffer(
            "_enc_mean",
            torch.tensor(norm["mean"], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_enc_std",
            torch.tensor(norm["std"], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # PLM normalization constants for un-normalizing
        self.register_buffer(
            "_plm_mean",
            torch.tensor(_PLM_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_plm_std",
            torch.tensor(_PLM_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        logger.info(
            f"HFVisionEncoderWrapper: family={encoder_family}, "
            f"image_size={self.image_size}, patch_size={self.patch_size}, "
            f"width={self.width}, grid={self.h_patches}x{self.w_patches}"
            f", spatial_merge={self.spatial_merge_size}"
            f" -> {self.h_patches_merged}x{self.w_patches_merged}"
        )

    # ── Loading methods ──

    def load_native_from_vlm(self, vlm_path: str):
        """Load the ACTUAL VLM vision encoder (not a generic HF model).

        This preserves the original encoder architecture exactly (e.g.,
        VideoLLaMA3's custom rotary pos embeddings, frame-level masking).
        """
        from transformers import AutoModelForCausalLM

        # Use flash_attention_2 if available (much faster + less memory)
        attn_impl = "sdpa"
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

        logger.info(f"Loading NATIVE vision encoder from {vlm_path} (attn={attn_impl})")
        hf_model = AutoModelForCausalLM.from_pretrained(
            vlm_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )

        # Extract the vision encoder module
        self.encoder = hf_model.model.vision_encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self._native_mode = True
        # Native encoder has no CLS token (VideoLLaMA3)
        self.has_cls_token = False

        # Recalculate patch grid for aligned resolution (native mode)
        m = max(self.spatial_merge_size, 1)
        factor = self.patch_size * m
        target_size = round(self.image_size / factor) * factor
        self.h_patches = target_size // self.patch_size
        self.w_patches = target_size // self.patch_size
        self.h_patches_merged = self.h_patches // m
        self.w_patches_merged = self.w_patches // m

        n_params = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            f"Loaded NATIVE vision encoder: {n_params:,} params (native_mode=True), "
            f"aligned_res={target_size}, grid={self.h_patches}x{self.w_patches} "
            f"-> {self.h_patches_merged}x{self.w_patches_merged}"
        )

        # Free the LLM weights
        del hf_model
        torch.cuda.empty_cache()

    def load_from_state_dict(self, state_dict: Dict[str, torch.Tensor], vision_config: Dict[str, Any]):
        """Build encoder architecture from config and load weights."""
        self.encoder = self._build_encoder(vision_config)

        # Auto-fix prefix mismatch: HF models wrap under 'vision_model.' but
        # vlm_loader already stripped the VLM prefix, leaving bare keys.
        sd = self._align_state_dict_keys(state_dict)

        missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
        # Filter out known-optional keys (pooler head, position embedding for NaViT)
        _OPTIONAL = (".head.", ".position_embedding.", ".position_ids")
        critical_missing = [k for k in missing if not any(o in k for o in _OPTIONAL)]
        if critical_missing:
            logger.warning(f"Missing keys in vision encoder ({len(critical_missing)}): {critical_missing[:5]}{'...' if len(critical_missing) > 5 else ''}")
        if missing and not critical_missing:
            logger.info(f"Vision encoder: {len(missing)} optional keys not in checkpoint (head/position_embedding) — OK")
        if unexpected:
            logger.info(f"Vision encoder: {len(unexpected)} unexpected keys ignored")
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info(f"Loaded vision encoder: {sum(p.numel() for p in self.encoder.parameters()):,} params")

    def load_from_pretrained(self, model_name_or_path: str):
        """Load encoder directly from HuggingFace model hub."""
        try:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        except Exception:
            from transformers import CLIPVisionModel, SiglipVisionModel
            if self.encoder_family == "siglip":
                self.encoder = SiglipVisionModel.from_pretrained(model_name_or_path)
            else:
                self.encoder = CLIPVisionModel.from_pretrained(model_name_or_path)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info(f"Loaded pretrained vision encoder from {model_name_or_path}")

    # ── Forward ──

    def forward(
        self,
        videos: torch.Tensor,
        strip_cls_token: bool = True,
    ) -> torch.Tensor:
        """
        Encode video frames and return spatiotemporal grid.

        Args:
            videos: (B, T, C, H, W) — PLM-normalized frames
            strip_cls_token: ignored (API compat)

        Returns:
            grid: (B, T, H_p, W_p, embed_dim)
        """
        del strip_cls_token

        if videos.dim() == 4:
            videos = videos.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)

        if videos.shape[1] == 3 and videos.dim() == 5:
            # Handle (B, C, T, H, W) format
            videos = videos.permute(0, 2, 1, 3, 4).contiguous()

        if self._native_mode:
            return self._forward_native(videos)
        else:
            return self._forward_hf(videos)

    def _forward_native(self, videos: torch.Tensor) -> torch.Tensor:
        """Forward using the VLM's native encoder (exact feature match).

        Converts (B, T, C, H, W) frames to the native encoder's expected
        patchified format, runs through the full encoder forward (which
        includes the correct spatial merge with rearrangement + bilinear),
        and reshapes to (B, T, Hm, Wm, D).

        Matches VideoLLaMA3 processor behavior:
          - Aligns resolution to factor = patch_size * merge_size
          - For video (merge=2): 384→392 (28 patches), merge→14
          - For image (merge=1): 384→378 (27 patches), no merge
        """
        B, T, C, H, W = videos.shape

        # Re-normalize: PLM → encoder-specific
        frames = videos.reshape(B * T, C, H, W)
        frames = self._renormalize(frames)

        # Align resolution to factor = patch_size * merge_size
        # (matches VideoLLaMA3 processor's simple_batched_resize)
        m = self.spatial_merge_size
        factor = self.patch_size * m
        target_size = round(self.image_size / factor) * factor

        # Resize to aligned resolution
        if H != target_size or W != target_size:
            frames = F.interpolate(
                frames,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        # Compute actual patch grid at aligned resolution
        Hp = target_size // self.patch_size
        Wp = target_size // self.patch_size
        ps = self.patch_size

        # Convert to patchified format: (N_total_patches, C * patch_size * patch_size)
        patches = frames.unfold(2, ps, ps).unfold(3, ps, ps)  # (B*T, C, Hp, Wp, ps, ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B*T, Hp, Wp, C, ps, ps)
        patches = patches.reshape(B * T * Hp * Wp, C * ps * ps)  # (N_total, C*ps*ps)

        Hm = Hp // m
        Wm = Wp // m
        S = Hp * Wp  # patches per frame

        # With flash_attention_2, the encoder handles variable-length sequences
        # natively via cu_seqlens — no OOM from O(N²) attention matrices.
        # Without flash_attn (SDPA fallback), we chunk frames to keep N small.
        _has_flash = hasattr(self, '_has_flash_attn') and self._has_flash_attn
        if not hasattr(self, '_has_flash_attn'):
            try:
                from flash_attn import flash_attn_varlen_func  # noqa: F401
                _has_flash = True
            except ImportError:
                _has_flash = False
            self._has_flash_attn = _has_flash

        if _has_flash:
            # Flash attention: process all frames at once (efficient)
            grid_sizes = torch.tensor(
                [[1, Hp, Wp]] * (B * T),
                device=videos.device, dtype=torch.long,
            )
            merge_sizes = torch.tensor(
                [m] * (B * T),
                device=videos.device, dtype=torch.long,
            )
            with torch.no_grad():
                merged_tokens = self.encoder(patches, grid_sizes, merge_sizes)
        else:
            # SDPA fallback: chunk to avoid O(N²) attention OOM
            chunk_size = 4
            merged_chunks = []
            with torch.no_grad():
                for start in range(0, B * T, chunk_size):
                    end = min(start + chunk_size, B * T)
                    n_frames = end - start
                    chunk_patches = patches[start * S : end * S]
                    chunk_grid = torch.tensor(
                        [[1, Hp, Wp]] * n_frames,
                        device=videos.device, dtype=torch.long,
                    )
                    chunk_merge = torch.tensor(
                        [m] * n_frames,
                        device=videos.device, dtype=torch.long,
                    )
                    out = self.encoder(chunk_patches, chunk_grid, chunk_merge)
                    merged_chunks.append(out)
            merged_tokens = torch.cat(merged_chunks, dim=0)
        D = merged_tokens.shape[-1]
        tokens = merged_tokens.reshape(B, T, Hm, Wm, D)

        return tokens.float()

    def _forward_hf(self, videos: torch.Tensor) -> torch.Tensor:
        """Forward using standard HF encoder (CLIP/SigLIP)."""
        B, T, C, H, W = videos.shape

        # Flatten batch and time for per-frame encoding
        frames = videos.reshape(B * T, C, H, W)

        # Re-normalize: PLM → encoder-specific
        frames = self._renormalize(frames)

        # Resize to encoder's expected resolution
        if H != self.image_size or W != self.image_size:
            frames = F.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Encode
        with torch.no_grad():
            tokens = self._encode_frames(frames)  # (B*T, N, embed_dim)

        # Strip CLS token if present
        if self.has_cls_token and tokens.shape[1] == self.h_patches * self.w_patches + 1:
            tokens = tokens[:, 1:]  # (B*T, H_p*W_p, embed_dim)

        # Reshape to spatial grid
        tokens = tokens.reshape(B, T, self.h_patches, self.w_patches, -1)

        # Spatial merge (matching VideoLLaMA3's bilinear downsampling)
        if self.spatial_merge_size > 1:
            tokens = self._spatial_merge(tokens)

        return tokens

    def _spatial_merge(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Spatial merge matching VideoLLaMA3's encoder output.

        Downsamples H×W patch grid by merge_size using bilinear interpolation.
        E.g., 27×27 → 13×13 with merge_size=2.

        Args:
            tokens: (B, T, H, W, D) patch features

        Returns:
            (B, T, H//m, W//m, D) merged features
        """
        B, T, H, W, D = tokens.shape
        m = self.spatial_merge_size
        Hm, Wm = H // m, W // m

        # (B*T, D, H, W) for F.interpolate
        x = tokens.reshape(B * T, H, W, D).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(Hm, Wm), mode="bilinear", align_corners=False)
        # (B, T, Hm, Wm, D)
        x = x.permute(0, 2, 3, 1).reshape(B, T, Hm, Wm, D)
        return x

    def _renormalize(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert PLM-normalized frames to encoder-specific normalization."""
        # PLM: x_plm = (x_raw - plm_mean) / plm_std
        # Encoder: x_enc = (x_raw - enc_mean) / enc_std
        # Combined: x_enc = (x_plm * plm_std + plm_mean - enc_mean) / enc_std

        # Check if normalization is essentially the same (e.g., SigLIP = PLM)
        if self.encoder_family == "siglip":
            return frames  # Same normalization

        x = frames.float()
        x = x * self._plm_std + self._plm_mean  # back to [0, 1]
        x = x.clamp(0.0, 1.0)
        x = (x - self._enc_mean) / self._enc_std
        return x.to(frames.dtype)

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Run encoder on preprocessed frames.

        Args:
            frames: (N, C, H, W) preprocessed frames

        Returns:
            tokens: (N, num_patches, embed_dim)
        """
        if self.encoder is None:
            raise RuntimeError("Vision encoder not loaded. Call load_from_state_dict or load_from_pretrained first.")

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=frames.is_cuda):
            outputs = self.encoder(
                pixel_values=frames,
                output_hidden_states=(self.vision_feature_layer != -1),
            )

        # Extract the right hidden state
        if self.vision_feature_layer != -1 and hasattr(outputs, "hidden_states") and outputs.hidden_states:
            tokens = outputs.hidden_states[self.vision_feature_layer]
        elif hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        else:
            tokens = outputs[0]

        return tokens.float()

    # ── Internal helpers ──

    def _build_encoder(self, vision_config: Dict[str, Any]) -> nn.Module:
        """Build a HF vision encoder from config dict."""
        try:
            from transformers import AutoConfig, AutoModel
            hf_config = AutoConfig.for_model(
                model_type=self._get_hf_model_type(),
                **self._config_to_hf_kwargs(vision_config),
            )
            return AutoModel.from_config(hf_config)
        except Exception as e:
            logger.warning(f"AutoConfig failed ({e}), falling back to manual construction")
            return self._build_manual_encoder(vision_config)

    def _align_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Align state dict keys to match the HF model's expected key prefix."""
        if self.encoder is None:
            return state_dict

        model_keys = set(self.encoder.state_dict().keys())
        sd_keys = set(state_dict.keys())

        # Already matches — no fix needed
        if sd_keys & model_keys:
            overlap = len(sd_keys & model_keys)
            if overlap > len(model_keys) * 0.5:
                return state_dict

        # Try common prefixes that HF models use
        for prefix in ("vision_model.", "model.", "encoder."):
            # Case 1: model expects prefix, sd doesn't have it
            prefixed = {prefix + k: v for k, v in state_dict.items()}
            overlap = len(set(prefixed.keys()) & model_keys)
            if overlap > len(model_keys) * 0.5:
                logger.info(f"Added '{prefix}' prefix to {overlap} state dict keys")
                return prefixed

            # Case 2: sd has prefix, model doesn't
            stripped = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    stripped[k[len(prefix):]] = v
                else:
                    stripped[k] = v
            overlap = len(set(stripped.keys()) & model_keys)
            if overlap > len(model_keys) * 0.5:
                logger.info(f"Stripped '{prefix}' prefix from {overlap} state dict keys")
                return stripped

        return state_dict

    def _get_hf_model_type(self) -> str:
        return {
            "clip": "clip_vision_model",
            "siglip": "siglip_vision_model",
            "internvit": "intern_vit_6b",
        }.get(self.encoder_family, "clip_vision_model")

    def _config_to_hf_kwargs(self, vision_config: Dict[str, Any]) -> dict:
        return {
            "hidden_size": vision_config.get("hidden_size", self.width),
            "image_size": vision_config.get("image_size", self.image_size),
            "patch_size": vision_config.get("patch_size", self.patch_size),
            "num_hidden_layers": vision_config.get("num_hidden_layers", 24),
            "intermediate_size": vision_config.get("intermediate_size", 4096),
            "num_attention_heads": vision_config.get("num_attention_heads", 16),
        }

    def _build_manual_encoder(self, vision_config: Dict[str, Any]) -> nn.Module:
        """Fallback: build CLIP-style vision encoder manually."""
        from transformers import CLIPVisionConfig, CLIPVisionModel
        config = CLIPVisionConfig(
            hidden_size=vision_config.get("hidden_size", self.width),
            image_size=vision_config.get("image_size", self.image_size),
            patch_size=vision_config.get("patch_size", self.patch_size),
            num_hidden_layers=vision_config.get("num_hidden_layers", 24),
            intermediate_size=vision_config.get("intermediate_size", 4096),
            num_attention_heads=vision_config.get("num_attention_heads", 16),
        )
        return CLIPVisionModel(config)

    # -- VJEPA2GridEncoder-compatible API --

    def count_tokens(self) -> int:
        return -1  # Determined by projector

    def init_tensors(self):
        pass

    def load_ckpt(self, path: str):
        pass

    def load_from_checkpoint(self, checkpoint_path: Optional[str] = None):
        pass

    def get_clip_embeddings(self, *args, **kwargs):
        return None
