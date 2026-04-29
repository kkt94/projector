"""
BMRv2/v3 model wrapper for lmms-eval standard evaluation.

Usage:
    python -m lmms_eval \
        --model bmr_v2 \
        --model_args checkpoint_path=./bmr_v3_stage2_from_videollama3_2b/stage2_weights_step5000.pt \
        --tasks mvbench \
        --batch_size 1 \
        --output_path ./eval_results/

The model loads:
  1. VideoLLaMA3-2B as base (vision encoder + LLM)
  2. BMRv2/v3 projector from checkpoint
  3. LLM LoRA merged from checkpoint
"""

import gc
import os
import types
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2ForCausalLM


def _gpu_mem():
    """Return current GPU memory allocated in GiB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# ────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────


def _extract_mlp_for_bmr(hf_model) -> Dict[str, torch.Tensor]:
    """Extract MLP state dict from VideoLLaMA3's mm_projector.readout."""
    sd = {}
    for k, v in hf_model.model.mm_projector.readout.state_dict().items():
        sd[k] = v.clone()
    return sd


def _merge_lora_into_hf(
    hf_model, lora_sd: Dict[str, torch.Tensor], rank: int, alpha: float
):
    """Merge our custom LoRA weights into the HF model's LLM layers."""
    scale = alpha / rank
    n_merged = 0

    n_layers = len(hf_model.model.layers)
    for i in range(n_layers):
        for our_name, hf_attr in [("wq", "q_proj"), ("wv", "v_proj")]:
            a_key = f"layers.{i}.attention.{our_name}.lora_A.weight"
            b_key = f"layers.{i}.attention.{our_name}.lora_B.weight"

            if a_key in lora_sd and b_key in lora_sd:
                lora_A = lora_sd[a_key]  # (rank, in_dim)
                lora_B = lora_sd[b_key]  # (out_dim, rank)
                target = getattr(hf_model.model.layers[i].self_attn, hf_attr)
                delta = (lora_B @ lora_A).to(device=target.weight.device, dtype=target.weight.dtype) * scale
                target.weight.data += delta
                n_merged += 1

    eval_logger.info(
        f"Merged {n_merged} LoRA adapters into HF model "
        f"(rank={rank}, alpha={alpha}, scale={scale:.4f})"
    )


def _create_bmr_v2_args(ckpt: Dict) -> object:
    """Create args namespace for BMRv2Projector from checkpoint metadata."""

    class Args:
        pass

    args = Args()
    args.dim = ckpt.get("llm_dim", 1536)
    args.vision_model = {"width": ckpt.get("vision_dim", 1152)}
    args.bmr_v2_num_tokens = ckpt.get("bmr_v2_num_tokens", 3136)
    args.bmr_v2_til_kernel = ckpt.get("bmr_v2_til_kernel", 3)
    args.bmr_v2_kd_weight = 0.0  # No KD loss during evaluation
    return args


# ────────────────────────────────────────────────────────────────
# lmms-eval model registration
# ────────────────────────────────────────────────────────────────


@register_model("bmr_v2")
class BMRv2(lmms):
    """
    BMRv2 (Dual-MLP + Temporal Residual Gating) model for lmms-eval.

    Loads VideoLLaMA3-2B as base, replaces the projector with BMRv2,
    merges LLM LoRA, and uses generative evaluation.

    Example:
        python -m lmms_eval \\
            --model bmr_v2 \\
            --model_args checkpoint_path=./checkpoint.pt,vlm_base=DAMO-NLP-SG/VideoLLaMA3-2B \\
            --tasks mvbench \\
            --batch_size 1 \\
            --output_path ./logs/
    """

    def __init__(
        self,
        checkpoint_path: str,
        vlm_base: str = "DAMO-NLP-SG/VideoLLaMA3-2B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        max_num_frames: int = 32,
        use_flash_attention_2: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        self.max_num_frames = max_num_frames

        # ── 1. Load checkpoint ──
        eval_logger.info(f"Loading BMRv2 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # ── 2. Resolve VLM path ──
        vlm_path = vlm_base
        for p in [
            vlm_base,
            f"models/{os.path.basename(vlm_base)}",
            f"DAMO-NLP-SG/{os.path.basename(vlm_base)}",
        ]:
            if os.path.isdir(p):
                vlm_path = p
                break
        eval_logger.info(f"Loading base VLM: {vlm_path}")

        # ── 3. Load VideoLLaMA3 HF model ──
        attn_impl = "flash_attention_2" if use_flash_attention_2 else "sdpa"
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            vlm_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        self.processor = AutoProcessor.from_pretrained(
            vlm_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        # ── 4. Create BMRv2 projector ──
        from core.vision_projector.bmr_v2 import BMRv2Projector

        proj_args = _create_bmr_v2_args(ckpt)
        self.bmr_v2 = BMRv2Projector(proj_args)

        # Load pretrained MLP into P_base and P_motion
        mlp_sd = _extract_mlp_for_bmr(self._hf_model)
        self.bmr_v2.load_pretrained_mlp(mlp_sd, key_prefix="")

        # Load trained BMRv2 projector weights
        bridge_sd = ckpt.get("bridge_state_dict", {})
        if bridge_sd:
            result = self.bmr_v2.load_state_dict(bridge_sd, strict=False)
            eval_logger.info(
                f"BMRv2 projector weights loaded. "
                f"Missing: {len(result.missing_keys)}, "
                f"Unexpected: {len(result.unexpected_keys)}"
            )
        self.bmr_v2 = self.bmr_v2.to(device=self._device, dtype=torch.bfloat16)
        self.bmr_v2.eval()

        # ── 5. Merge LLM LoRA ──
        lora_sd = ckpt.get("lora_state_dict", {})
        if lora_sd:
            lora_rank = ckpt.get("lora_rank", 16)
            lora_alpha = ckpt.get("lora_alpha", 32)
            _merge_lora_into_hf(self._hf_model, lora_sd, lora_rank, lora_alpha)

        self._hf_model.eval()

        # ── 6. Store config ──
        self.num_visual_tokens = proj_args.bmr_v2_num_tokens
        self.image_token_id = self._hf_model.config.image_token_index

        eval_logger.info(
            f"BMRv2 model ready. "
            f"Visual tokens: {self.num_visual_tokens}, "
            f"Max frames: {self.max_num_frames}"
        )

    # ── Properties ──

    @property
    def model(self):
        return self._hf_model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # ── Vision encoding ──

    def _encode_visual(
        self, pixel_values, grid_sizes, merge_sizes
    ) -> torch.Tensor:
        """
        Run vision encoder + BMRv2 projector.

        Handles both grid_sizes formats:
          - Single entry:   [[T, H, W]]       (video as one batch)
          - Per-frame:      [[1, H, W]] * T   (multi-image / manual)

        Returns: (1, num_visual_tokens, llm_dim) visual embeddings.
        """
        encoder = self._hf_model.model.vision_encoder
        pixel_values = pixel_values.to(device=self._device, dtype=torch.bfloat16)
        grid_sizes = grid_sizes.to(device=self._device)
        merge_sizes = merge_sizes.to(device=self._device)

        eval_logger.info(
            f"  _encode_visual: pixel_values={pixel_values.shape} "
            f"grid_sizes={grid_sizes.tolist()} merge_sizes={merge_sizes.tolist()} "
            f"mem={_gpu_mem():.2f}GiB"
        )

        with torch.no_grad():
            merged_tokens = encoder(pixel_values, grid_sizes, merge_sizes)

        eval_logger.info(
            f"  _encode_visual: after encoder merged_tokens={merged_tokens.shape} "
            f"mem={_gpu_mem():.2f}GiB"
        )

        # Free encoder intermediates immediately
        del pixel_values
        torch.cuda.empty_cache()

        # Determine T, H, W from grid_sizes
        spatial_merge = merge_sizes[0].item()
        if grid_sizes.shape[0] == 1 and grid_sizes[0, 0].item() > 1:
            # Single entry: [[T, H, W]]
            T = grid_sizes[0, 0].item()
            H = grid_sizes[0, 1].item()
            W = grid_sizes[0, 2].item()
        else:
            # Per-frame entries: [[1, H, W]] * T
            T = grid_sizes.shape[0]
            H = grid_sizes[0, 1].item()
            W = grid_sizes[0, 2].item()

        Hm = H // max(spatial_merge, 1)
        Wm = W // max(spatial_merge, 1)
        D = merged_tokens.shape[-1]
        features_5d = merged_tokens.reshape(1, T, Hm, Wm, D)

        # Run BMRv2 projector
        with torch.no_grad():
            visual_tokens = self.bmr_v2(features_5d)  # (1, N, llm_dim)

        return visual_tokens

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        visual_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Replace <image> token positions in input_ids with BMRv2 visual tokens.
        If token count differs, re-build the sequence.
        """
        input_ids = input_ids.to(self._device)

        if visual_tokens is None:
            return self._hf_model.model.embed_tokens(input_ids)

        ids = input_ids[0]  # (L,)
        image_mask = ids == self.image_token_id
        image_positions = image_mask.nonzero(as_tuple=True)[0]

        if len(image_positions) == 0:
            # No image tokens — prepend visual tokens
            text_embeds = self._hf_model.model.embed_tokens(input_ids)
            return torch.cat([visual_tokens, text_embeds], dim=1)

        first_img = image_positions[0].item()
        last_img = image_positions[-1].item()

        # Text before image tokens
        ids_before = ids[:first_img]
        # Text after image tokens
        ids_after = ids[last_img + 1 :]

        embeds_before = self._hf_model.model.embed_tokens(
            ids_before.unsqueeze(0)
        )
        embeds_after = self._hf_model.model.embed_tokens(
            ids_after.unsqueeze(0)
        )

        # Combine: text_before + visual_tokens + text_after
        inputs_embeds = torch.cat(
            [embeds_before, visual_tokens, embeds_after], dim=1
        )
        return inputs_embeds

    # ── Evaluation interface ──

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for BMRv2")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="BMRv2 Generating",
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(
                *chunk
            )
            task = task[0]
            split = split[0]
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids])
                for ids in doc_id
            ]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Process one sample at a time (batch_size=1 for video)
            for i, context in enumerate(contexts):
                # Clear GPU cache before each sample to prevent fragmentation
                gc.collect()
                torch.cuda.empty_cache()

                visual = visuals[i] if i < len(visuals) else None
                visual_tokens = None
                sample_idx = pbar.n

                eval_logger.info(
                    f"[Sample {sample_idx}] mem={_gpu_mem():.2f}GiB "
                    f"visual_type={type(visual).__name__} "
                    f"visual_val={repr(visual)[:120] if isinstance(visual, str) else type(visual).__name__}"
                )

                # ── Process visual input ──
                if visual is not None:
                    try:
                        if isinstance(visual, str) and visual.endswith(
                            (".mp4", ".avi", ".mov", ".mkv", ".webm")
                        ):
                            visual_tokens = self._process_video(visual, context)
                        elif isinstance(visual, Image.Image):
                            visual_tokens = self._process_image(
                                visual, context
                            )
                        elif isinstance(visual, (list, tuple)) and all(
                            isinstance(v, Image.Image) for v in visual
                        ):
                            visual_tokens = self._process_image_list(
                                visual, context
                            )
                        else:
                            eval_logger.warning(
                                f"[Sample {sample_idx}] Unknown visual type: "
                                f"{type(visual).__name__}, value={repr(visual)[:200]}"
                            )
                    except Exception as e:
                        eval_logger.warning(
                            f"[Sample {sample_idx}] Failed to process visual "
                            f"(mem={_gpu_mem():.2f}GiB): {e}. Using text-only."
                        )
                        # Clean up leaked GPU tensors from failed processing
                        gc.collect()
                        torch.cuda.empty_cache()
                        eval_logger.info(
                            f"[Sample {sample_idx}] After cleanup: mem={_gpu_mem():.2f}GiB"
                        )

                if visual_tokens is not None:
                    eval_logger.info(
                        f"[Sample {sample_idx}] visual_tokens shape={visual_tokens.shape} "
                        f"mem={_gpu_mem():.2f}GiB"
                    )

                try:
                    # ── Build conversation for tokenization ──
                    if visual_tokens is not None:
                        # Build input with image placeholders
                        message = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": Image.new("RGB", (16, 16))},
                                    {"type": "text", "text": context},
                                ],
                            }
                        ]
                        text_inputs = self.processor(
                            conversation=message,
                            return_tensors="pt",
                            add_generation_prompt=True,
                        )
                        input_ids = text_inputs["input_ids"]
                    else:
                        # Text-only
                        message = [
                            {"role": "user", "content": [{"type": "text", "text": context}]}
                        ]
                        text_inputs = self.processor(
                            conversation=message,
                            return_tensors="pt",
                            add_generation_prompt=True,
                        )
                        input_ids = text_inputs["input_ids"]

                    # ── Build inputs_embeds ──
                    inputs_embeds = self._build_inputs_embeds(
                        input_ids, visual_tokens
                    )
                    attn_mask = torch.ones(
                        1,
                        inputs_embeds.shape[1],
                        device=self._device,
                        dtype=torch.long,
                    )

                    eval_logger.info(
                        f"[Sample {sample_idx}] inputs_embeds shape={inputs_embeds.shape} "
                        f"mem={_gpu_mem():.2f}GiB"
                    )

                    # ── Generate ──
                    do_sample = gen_kwargs.get("do_sample", False)
                    max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
                    temperature = gen_kwargs.get("temperature", 1.0)

                    with torch.inference_mode():
                        # Bypass VideoLLaMA3's generate override by calling
                        # Qwen2ForCausalLM's parent (GenerationMixin.generate)
                        output_ids = Qwen2ForCausalLM.generate(
                            self._hf_model,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attn_mask,
                            do_sample=do_sample,
                            temperature=temperature if do_sample else 1.0,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                        )

                    # Decode generated tokens only
                    answer = self.tokenizer.decode(
                        output_ids[0], skip_special_tokens=True
                    ).strip()

                except Exception as e:
                    eval_logger.warning(
                        f"[Sample {sample_idx}] Generation failed "
                        f"(mem={_gpu_mem():.2f}GiB): {e}. Using empty answer."
                    )
                    answer = ""
                    gc.collect()
                    torch.cuda.empty_cache()

                res.append(answer)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), answer
                )
                pbar.update(1)

                # Free GPU memory between samples to prevent accumulation
                visual_tokens = None
                gc.collect()
                torch.cuda.empty_cache()

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not supported")

    # ── Visual processing helpers ──

    def _process_video(
        self, video_path: str, context: str
    ) -> torch.Tensor:
        """Process video: load frames, patchify, encode with native encoder.

        Bypasses VideoLLaMA3's processor (which has Jinja template issues
        with {"type": "video"} content) and manually loads + patchifies frames
        to match the training pipeline exactly:
          - Resize to factor-aligned resolution (384 → 392 for merge_size=2)
          - SigLIP normalization (mean=0.5, std=0.5)
          - Patchify to (N_patches, C*ps*ps) format
          - Encode with merge_size=2 (video mode)
        """
        from decord import VideoReader

        vr = VideoReader(video_path)
        total = len(vr)
        n_frames = min(self.max_num_frames, total)
        if n_frames <= 0:
            return None
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)

        T = len(frames)
        ps = 14
        m = 2  # video merge_size
        factor = ps * m
        target_size = round(384 / factor) * factor  # 392

        # Preprocess: resize + normalize + patchify
        frames_t = torch.from_numpy(frames).float() / 255.0  # (T, H, W, 3)
        frames_t = frames_t.permute(0, 3, 1, 2)  # (T, 3, H, W)
        frames_t = (frames_t - 0.5) / 0.5  # SigLIP normalize (= PLM)
        frames_t = F.interpolate(
            frames_t,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

        Hp = target_size // ps  # 28
        Wp = Hp

        # Patchify: (T, 3, H, W) → (T*Hp*Wp, 3*ps*ps)
        patches = frames_t.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.reshape(T * Hp * Wp, 3 * ps * ps)

        pixel_values = patches.to(device=self._device, dtype=torch.bfloat16)
        grid_sizes = torch.tensor(
            [[1, Hp, Wp]] * T, device=self._device, dtype=torch.long
        )
        merge_sizes = torch.tensor(
            [m] * T, device=self._device, dtype=torch.long
        )

        return self._encode_visual(pixel_values, grid_sizes, merge_sizes)

    def _process_image(
        self, image: Image.Image, context: str
    ) -> torch.Tensor:
        """Process a single image."""
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": "placeholder"},
                ],
            }
        ]
        inputs = self.processor(
            conversation=message,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        return self._encode_visual(
            inputs["pixel_values"],
            inputs["grid_sizes"],
            inputs["merge_sizes"],
        )

    def _process_image_list(
        self, images: list, context: str
    ) -> torch.Tensor:
        """Process multiple images as video frames."""
        content = []
        for img in images:
            content.append({"type": "image", "image": img.convert("RGB")})
        content.append({"type": "text", "text": "placeholder"})
        message = [{"role": "user", "content": content}]
        inputs = self.processor(
            conversation=message,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        return self._encode_visual(
            inputs["pixel_values"],
            inputs["grid_sizes"],
            inputs["merge_sizes"],
        )
