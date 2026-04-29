"""
lmms-eval wrapper for VideoLLaMA3-2B + UDH v2 (SplitDualHeadExpert + DualBranchRouter).

Supports two modes:
  - vanilla: checkpoint_dir="none" → pure VideoLLaMA3-2B
  - udh:     checkpoint_dir=<path> → loads diff_expert.pt + router.pt (Phase A: expert only)

Registered as "videollama3_udh" for lmms-eval.

Usage:
    python -m lmms_eval \
        --model videollama3_udh \
        --model_args checkpoint_dir=none,vlm_base=models/VideoLLaMA3-2B \
        --tasks mvbench \
        --batch_size 1
"""

import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from lmms_eval import utils
except ImportError:
    from lmms_eval.api import utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import decord

    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

eval_logger = logging.getLogger("lmms-eval")

IMAGE_TOKEN = "<image>"
IM_START_ID = 151644
IM_END_ID = 151645


def _mcq_acc_match(response: str, gt_letter: str) -> bool:
    """Match MCQ answer letter from response (same as PLM eval)."""
    if not gt_letter:
        return False
    resp = response.strip().upper()
    gt = gt_letter.strip().upper()
    if resp == gt:
        return True
    if resp.startswith(gt + ".") or resp.startswith(gt + ")") or resp.startswith(gt + " "):
        return True
    patterns = [
        rf"\b{gt}\b",
        rf"^{gt}[.)\s]",
        rf"answer\s*(?:is|:)\s*{gt}\b",
        rf"option\s*{gt}\b",
    ]
    for pat in patterns:
        if re.search(pat, resp):
            return True
    return False


@register_model("videollama3_udh")
class VideoLLaMA3UDH(lmms):
    """VideoLLaMA3-2B + UDH v2 dual-branch expert/router evaluation wrapper."""

    def __init__(
        self,
        vlm_base: str = "models/VideoLLaMA3-2B",
        checkpoint_dir: str = "none",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        max_video_frames: Optional[Union[int, str]] = 32,
        attn_implementation: str = "flash_attention_2",
        progress_dir: str = "/tmp/plm_eval_progress",
        decode_mode: str = "greedy",
        **kwargs,
    ) -> None:
        super().__init__()
        from accelerate import Accelerator
        from transformers import AutoModelForCausalLM, AutoProcessor

        accelerator = Accelerator()
        self._device = torch.device(f"cuda:{accelerator.local_process_index}")

        # Resolve model path
        vlm_path = vlm_base
        if not os.path.isdir(vlm_path):
            for candidate in [
                os.path.join("models", os.path.basename(vlm_path)),
                os.path.join("DAMO-NLP-SG", os.path.basename(vlm_path)),
            ]:
                if os.path.isdir(candidate):
                    vlm_path = candidate
                    break

        eval_logger.info(f"Loading VideoLLaMA3 from {vlm_path} (attn={attn_implementation})")
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            vlm_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": self._device},
            attn_implementation=attn_implementation,
        )
        self._processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self._tokenizer = self._processor.tokenizer

        for p in self._hf_model.parameters():
            p.requires_grad = False
        self._hf_model.eval()

        self._image_token_id = self._hf_model.config.image_token_index
        self._pad_token_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else self._tokenizer.eos_token_id
        )

        # Vision config
        self._patch_size = 14
        self._merge_size = 2
        self._factor = self._patch_size * self._merge_size
        self._target_size = int(round(384.0 / self._factor) * self._factor)  # 392
        hp = self._target_size // self._patch_size
        self._hm = hp // self._merge_size  # 14
        self._tokens_per_frame = self._hm * self._hm  # 196
        self._image_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
        self._image_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
        self._attn_implementation = attn_implementation

        # Load adapter
        self._expert = None
        self._router = None
        self._tq_router = None
        self._adapter_cfg = {}
        vanilla_mode = str(checkpoint_dir).lower() in ("none", "null", "") or not os.path.isdir(
            str(checkpoint_dir)
        )

        if vanilla_mode:
            eval_logger.info("=== VANILLA MODE (no adapter) ===")
        else:
            self._load_adapter(checkpoint_dir)

        self.batch_size_per_gpu = int(batch_size)
        self._max_video_frames = int(max_video_frames) if max_video_frames else 32

        if accelerator.num_processes > 1:
            self._rank = accelerator.process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        self._decode_mode = str(decode_mode).lower()
        if self._decode_mode == "dual_select" and self._expert is None:
            eval_logger.warning("dual_select requires adapter; falling back to greedy")
            self._decode_mode = "greedy"
        if self._decode_mode == "dual_select":
            eval_logger.info("=== DUAL SELECT decode mode ===")

        self._progress_dir = progress_dir
        os.makedirs(self._progress_dir, exist_ok=True)
        self._progress_file = os.path.join(self._progress_dir, f"progress_rank{self._rank}.jsonl")
        with open(self._progress_file, "w"):
            pass

    def _load_adapter(self, checkpoint_dir):
        """Load UDH v2 expert + router (Phase A/B/C) 또는 tq_router (Stage 2) from checkpoint."""
        from core.vision_projector.dual_proj import (
            DualBranchRouter,
            GatedAttentionDualRouter,
            SplitDualHeadExpert,
        )

        config_path = os.path.join(checkpoint_dir, "adapter_config.json")
        if not os.path.exists(config_path):
            eval_logger.warning(f"No adapter_config.json in {checkpoint_dir}, running vanilla")
            return

        with open(config_path) as f:
            cfg = json.load(f)
        self._adapter_cfg = cfg

        # Stage 2/3인 경우 stage1_config에서 expert 설정 가져옴
        is_stage2_or_3 = cfg.get("module_type", "") in (
            "videollama3_udh_v2_stage2_tqrouter", "videollama3_udh_v2_stage3_lora",
        )
        expert_cfg = cfg.get("stage1_config", cfg) if is_stage2_or_3 else cfg

        dtype = torch.bfloat16
        proj_dim = cfg.get("proj_dim", self._hf_model.config.hidden_size)
        pooling_ratio = expert_cfg.get("pooling_ratio", cfg.get("pooling_ratio", 1))

        # Load expert
        expert_path = os.path.join(checkpoint_dir, "diff_expert.pt")
        if os.path.exists(expert_path):
            expert = SplitDualHeadExpert(
                proj_dim=proj_dim,
                bottleneck=expert_cfg.get("expert_bottleneck", 128),
                kernel_size=expert_cfg.get("expert_kernel_size", 3),
                max_local_gate=expert_cfg.get("max_local_gate", 0.35),
                max_state_gate=expert_cfg.get("max_state_gate", 0.35),
                pooling_ratio=pooling_ratio,
                ortho_proj=expert_cfg.get("ortho_proj", False),
            )
            expert.load_state_dict(
                torch.load(expert_path, map_location="cpu", weights_only=True),
                strict=False,
            )
            self._expert = expert.to(device=self._device, dtype=dtype).eval()
            eval_logger.info(f"Loaded SplitDualHeadExpert from {expert_path}")

        # Load router (skip if Phase A only — phase_b_steps==0 means router was never trained)
        phase_b_steps = cfg.get("phase_b_steps", 0)
        router_path = os.path.join(checkpoint_dir, "router.pt")
        if os.path.exists(router_path) and phase_b_steps > 0:
            q_dim = self._get_raw_model().get_model().embed_tokens.weight.shape[1]
            router_type = cfg.get("router_type", "mlp")
            if router_type == "gated_attn":
                router = GatedAttentionDualRouter(
                    q_dim=q_dim,
                    video_dim=proj_dim,
                    hidden_dim=cfg.get("router_hidden", 128),
                    num_heads=4,
                    stats_dim=4,
                    init_local=cfg.get("init_local", 0.5),
                    init_state=cfg.get("init_state", 0.5),
                )
            else:
                router = DualBranchRouter(
                    embed_dim=q_dim,
                    hidden_dim=cfg.get("router_hidden", 128),
                    stats_dim=4,
                    init_local=cfg.get("init_local", 0.5),
                    init_state=cfg.get("init_state", 0.5),
                )
            router.load_state_dict(
                torch.load(router_path, map_location="cpu", weights_only=True)
            )
            self._router = router.to(device=self._device, dtype=dtype).eval()
            eval_logger.info(f"Loaded router ({router_type}) from {router_path}")

        # Stage 2/3: TemporalQueryRouter 로드
        tq_router_path = os.path.join(checkpoint_dir, "tq_router.pt")
        if os.path.exists(tq_router_path):
            from core.vision_projector.temporal_query_router import TemporalQueryRouter
            s2_cfg = cfg.get("stage2_config", cfg)
            router_hidden = cfg.get("router_hidden") or s2_cfg.get("router_hidden", 256)
            router_init_bias = cfg.get("stage2_init_bias") or s2_cfg.get("stage2_init_bias", 0.0)
            tq_router = TemporalQueryRouter(
                feat_dim=proj_dim,
                hidden_dim=router_hidden,
                init_bias=router_init_bias,
            ).to(device=self._device, dtype=dtype)
            tq_router.load_state_dict(torch.load(tq_router_path, map_location="cpu", weights_only=True))
            tq_router.eval()
            self._tq_router = tq_router
            eval_logger.info(f"Loaded TemporalQueryRouter from {tq_router_path} (hidden={router_hidden})")

        # Stage 3: LoRA 로드
        lora_dir = os.path.join(checkpoint_dir, "lora")
        if os.path.isdir(lora_dir):
            from peft import PeftModel
            self._hf_model = PeftModel.from_pretrained(
                self._hf_model, lora_dir
            ).to(device=self._device, dtype=dtype)
            self._hf_model.eval()
            eval_logger.info(f"Loaded LoRA from {lora_dir}")

        if self._expert is None:
            eval_logger.warning("No diff_expert.pt found, running vanilla")

    def _get_raw_model(self):
        """PeftModel이면 원본 CausalLM 반환, 아니면 그대로."""
        m = self._hf_model
        if hasattr(m, "peft_config"):
            return m.base_model.model
        return m

    @property
    def model(self):
        return self._hf_model

    @property
    def tokenizer(self):
        return self._tokenizer

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

    def _load_video_frames(self, path):
        """Load video frames using decord with uniform sampling."""
        if not HAS_DECORD:
            return None
        try:
            vr = decord.VideoReader(path, num_threads=1)
            total = len(vr)
            if total == 0:
                return None
            max_frames = self._max_video_frames
            if total <= max_frames:
                indices = list(range(total))
            else:
                indices = [int(i * total / max_frames) for i in range(max_frames)]
            frames = vr.get_batch(indices)
            if isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()
            else:
                frames = np.stack([frames[i].asnumpy() for i in range(len(indices))], axis=0)
            return [frames[i] for i in range(len(indices))]
        except Exception as e:
            eval_logger.warning(f"Failed to load video {path}: {e}")
            return None

    def _preprocess_frames(self, frames):
        """Preprocess frames: resize to target_size, normalize."""
        arr = np.stack(frames, axis=0)  # (T, H, W, 3)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
        if tensor.shape[-2] != self._target_size or tensor.shape[-1] != self._target_size:
            tensor = F.interpolate(
                tensor,
                size=(self._target_size, self._target_size),
                mode="bilinear",
                align_corners=False,
            )
        tensor = (tensor - self._image_mean) / self._image_std
        return tensor

    def _encode_vision(self, pixel_values):
        """Encode video frames through vision encoder + mm_projector, with chunking for SDPA."""
        bsz, t = pixel_values.shape[:2]
        h, w = pixel_values.shape[3], pixel_values.shape[4]
        hp = h // self._patch_size
        wp = w // self._patch_size
        patches_per_frame = hp * wp
        tokens_per_frame = self._tokens_per_frame
        n_total = bsz * t

        # flash_attn: no chunking needed; SDPA/eager: chunk to avoid O(N²) OOM
        chunk_size = 0 if self._attn_implementation == "flash_attention_2" else 4
        effective_chunk = chunk_size if chunk_size > 0 else n_total

        all_projected = []
        flat_frames = pixel_values.reshape(n_total, 3, h, w)
        for ci in range(0, n_total, effective_chunk):
            end = min(ci + effective_chunk, n_total)
            n_chunk = end - ci
            chunk_frames = flat_frames[ci:end]
            patches = chunk_frames.unfold(2, self._patch_size, self._patch_size).unfold(
                3, self._patch_size, self._patch_size
            )
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.reshape(
                n_chunk * patches_per_frame, 3 * self._patch_size * self._patch_size
            )
            grid_sizes = torch.tensor(
                [[1, hp, wp]] * n_chunk, device=pixel_values.device, dtype=torch.long
            )
            msizes = torch.tensor(
                [self._merge_size] * n_chunk, device=pixel_values.device, dtype=torch.long
            )
            merged = self._get_raw_model().model.vision_encoder(patches, grid_sizes, msizes)
            projected = self._get_raw_model().model.mm_projector(merged)
            all_projected.append(projected.reshape(n_chunk, tokens_per_frame, -1))

        z = torch.cat(all_projected, dim=0).reshape(bsz, t, tokens_per_frame, -1)
        return z

    def _apply_expert_router(self, z_base, input_ids=None):
        """Apply UDH expert and optionally router/tq_router to get final visual features."""
        if self._expert is None:
            return z_base

        B = z_base.shape[0]
        z_pre_pool = z_base  # pooling_ratio=1, so z_pre_pool == z_base

        with torch.no_grad():
            if self._tq_router is not None:
                # Stage 2: TemporalQueryRouter로 프레임별 gate 적용
                _, local_res, state_res, _ = self._expert(
                    z_pre_pool, z_base, return_aux=True
                )
                q_embed = self._get_question_embedding(input_ids)  # (B, D)
                z_base_summary = z_base.mean(dim=2)    # (B, T, D)
                local_summary = local_res.mean(dim=2)   # (B, T, D)
                state_summary = state_res.mean(dim=2)   # (B, T, D)
                local_scale, state_scale = self._tq_router(
                    q_embed, z_base_summary, local_summary, state_summary,
                )
                # local_scale, state_scale: (B, T, 1) → unsqueeze for spatial
                # Enrichment normalize + scale
                base_norm = z_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                local_norm = local_res * (base_norm / local_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                state_norm = state_res * (base_norm / state_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                ls = local_scale.unsqueeze(-1)
                ss = state_scale.unsqueeze(-1)
                z_final = (1.0 - ls - ss) * z_base + ls * local_norm + ss * state_norm

            elif self._router is not None:
                # Phase B/C: expert + router
                _, local_res, state_res, _ = self._expert(
                    z_pre_pool, z_base, return_aux=True, return_aux_mode="residual"
                )

                # Compute question embedding for router
                q_embed = self._get_question_embedding(input_ids)

                # Compute router stats
                base_pool = z_base.detach().mean(dim=2)
                if z_base.shape[1] > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    base_motion = torch.zeros(B, device=z_base.device, dtype=z_base.dtype)
                local_mag = local_res.detach().norm(dim=-1).mean(dim=(1, 2))
                state_mag = state_res.detach().norm(dim=-1).mean(dim=(1, 2))
                total_mag = (local_res + state_res).detach().norm(dim=-1).mean(dim=(1, 2))
                stats = torch.log1p(
                    torch.stack([base_motion, local_mag, state_mag, total_mag], dim=-1).float()
                ).to(dtype=z_base.dtype)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    w = self._router(q_embed, video_stats=stats)
                w_local = w[:, 0].to(dtype=z_base.dtype).view(B, 1, 1, 1)
                w_state = w[:, 1].to(dtype=z_base.dtype).view(B, 1, 1, 1)
                z_final = z_base + w_local * local_res + w_state * state_res
            else:
                # Phase A only: expert without router, use z_temp directly
                z_temp, _, _, _ = self._expert(z_pre_pool, z_base, return_aux=True)
                z_final = z_temp

        return z_final

    def _get_question_embedding(self, input_ids):
        """Extract question text embedding (excluding image/special tokens)."""
        with torch.no_grad():
            all_embeds = self._get_raw_model().get_model().embed_tokens(input_ids)
        q_embeds = []
        for i in range(input_ids.shape[0]):
            mask = (
                input_ids[i].ne(self._image_token_id)
                & input_ids[i].ne(self._pad_token_id)
                & input_ids[i].ne(IM_START_ID)
                & input_ids[i].ne(IM_END_ID)
            )
            if mask.any():
                q_embeds.append(all_embeds[i][mask].mean(dim=0))
            else:
                q_embeds.append(
                    torch.zeros(all_embeds.shape[-1], device=all_embeds.device, dtype=all_embeds.dtype)
                )
        return torch.stack(q_embeds)

    @torch.no_grad()
    def _generate_with_features(self, input_ids, visual_features, max_new_tokens=256):
        """Generate text by injecting visual features at <image> token positions.

        Uses manual autoregressive loop (same as PLM eval wrapper) to bypass
        VideoLLaMA3's inputs_embeds restriction in generate().
        """
        inputs_embeds = self._get_raw_model().get_model().embed_tokens(input_ids)

        if visual_features is not None:
            vf = visual_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask = input_ids.eq(self._image_token_id)
            inputs_embeds[image_mask] = inputs_embeds[image_mask] * 0.0 + vf.reshape(-1, vf.shape[-1])

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Manual autoregressive generation (like PLM wrapper)
        generated_tokens = []
        past_key_values = None
        cur_embeds = inputs_embeds
        cur_mask = attention_mask
        eos_id = self._tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            outputs = self._get_raw_model().get_model()(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = self._hf_model.lm_head(outputs.last_hidden_state[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            token_id = next_token.item()
            if token_id == eos_id:
                break
            generated_tokens.append(token_id)
            cur_embeds = self._get_raw_model().get_model().embed_tokens(next_token)
            cur_mask = torch.cat([cur_mask, torch.ones(1, 1, dtype=torch.long, device=self._device)], dim=1)

        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return text

    def _generate_with_logprob(self, input_ids, visual_features, max_new_tokens=256):
        """Generate text and return (text, avg_logprob) for dual select."""
        inputs_embeds = self._get_raw_model().get_model().embed_tokens(input_ids)

        if visual_features is not None:
            vf = visual_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask = input_ids.eq(self._image_token_id)
            inputs_embeds[image_mask] = inputs_embeds[image_mask] * 0.0 + vf.reshape(-1, vf.shape[-1])

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        generated_tokens = []
        log_probs = []
        past_key_values = None
        cur_embeds = inputs_embeds
        cur_mask = attention_mask
        eos_id = self._tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            outputs = self._get_raw_model().get_model()(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = self._hf_model.lm_head(outputs.last_hidden_state[:, -1:, :])
            log_p = F.log_softmax(logits[:, 0, :], dim=-1)
            next_token = logits.argmax(dim=-1)
            token_id = next_token.item()
            if token_id == eos_id:
                break
            log_probs.append(log_p[0, token_id].item())
            generated_tokens.append(token_id)
            cur_embeds = self._get_raw_model().get_model().embed_tokens(next_token)
            cur_mask = torch.cat([cur_mask, torch.ones(1, 1, dtype=torch.long, device=self._device)], dim=1)

        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        avg_lp = sum(log_probs) / len(log_probs) if log_probs else -float("inf")
        return text, avg_lp

    def _build_prompt_ids(self, question, num_visual_tokens):
        """Build input_ids with <image> token placeholders for the question."""
        q_clean = re.sub(r"<video>\s*", "", question).strip()
        image_prefix = IMAGE_TOKEN * num_visual_tokens
        user_content = image_prefix + "\n" + q_clean

        messages = [{"role": "user", "content": user_content}]
        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        return input_ids

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        from tqdm import tqdm

        res = []

        with open(self._progress_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "type": "start",
                        "total": len(requests),
                        "world_size": self._world_size,
                        "rank": self._rank,
                        "timestamp": time.time(),
                    }
                )
                + "\n"
            )

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            lambda x: (-len(self._tokenizer.encode(x[0])), x[0]),
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            for i, context in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None
                gen_kwargs = all_gen_kwargs[i]
                max_new_tokens = gen_kwargs.get("max_new_tokens", 256)

                try:
                    gen = self._process_single(visual, context, max_new_tokens)
                except torch.cuda.OutOfMemoryError:
                    eval_logger.warning(f"OOM on sample {len(res)}, returning empty")
                    torch.cuda.empty_cache()
                    gen = ""

                if gen.endswith("."):
                    gen = gen[:-1]

                res.append(gen)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), gen)
                pbar.update(1)

                # Log progress
                task_name = task
                doc = self.task_dict[task][split][doc_id[i]]
                self._log_progress(task_name, gen, doc, len(res), len(requests))

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def _prepare_native_inputs(self, frames, context):
        """Build processor inputs with fixed 392×392 resolution."""
        ts = self._target_size  # 392
        pil_frames = [Image.fromarray(f).resize((ts, ts), Image.BILINEAR) for f in frames]
        q_text = re.sub(r"<video>\s*", "", context).strip()

        message = [{"role": "user", "content": [
            {"type": "video", "video": pil_frames, "num_frames": len(pil_frames)},
            {"type": "text", "text": q_text}
        ]}]
        inputs = self._processor(conversation=message, return_tensors="pt", add_generation_prompt=True)
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs, pil_frames

    def _native_generate(self, inputs, hook_fn=None, return_scores=False):
        """Run VL3 native generate, optionally with encode_images hook.

        Deep-copies inputs: VL3's generate() pops keys from kwargs AND
        the vision encoder may modify pixel_values in-place, so a shallow
        copy is not enough when generate() is called multiple times.
        """
        inputs_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        original_encode = self._hf_model.encode_images
        if hook_fn is not None:
            self._hf_model.encode_images = hook_fn
        try:
            extra = {}
            if return_scores:
                extra["output_scores"] = True
                extra["return_dict_in_generate"] = True
            outputs = self._hf_model.generate(
                **inputs_copy,
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
                **extra,
            )
        finally:
            self._hf_model.encode_images = original_encode
        if return_scores:
            return outputs  # GenerateOutput with .sequences and .scores
        return outputs

    def _score_sequence(self, inputs, output_ids, hook_fn=None):
        """Compute avg log-prob of generated tokens via a single forward pass."""
        prompt_len = inputs["input_ids"].shape[1]
        full_ids = output_ids[:, :prompt_len + 256]  # cap length
        gen_len = full_ids.shape[1] - prompt_len
        if gen_len <= 0:
            return -float("inf")

        original_encode = self._hf_model.encode_images
        if hook_fn is not None:
            self._hf_model.encode_images = hook_fn
        try:
            out = self._hf_model(
                **{k: v for k, v in inputs.items() if k != "input_ids"},
                input_ids=full_ids,
                use_cache=False,
                return_dict=True,
            )
        finally:
            self._hf_model.encode_images = original_encode

        logits = out.logits[:, prompt_len - 1:-1, :]  # align with generated tokens
        log_probs = F.log_softmax(logits.float(), dim=-1)
        gen_tokens = full_ids[:, prompt_len:]
        token_lps = log_probs.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
        # Stop at eos
        eos_mask = gen_tokens.eq(self._tokenizer.eos_token_id)
        if eos_mask.any():
            first_eos = eos_mask.float().argmax(dim=1).item()
            if first_eos == 0:
                return -float("inf")
            token_lps = token_lps[:, :first_eos]
        return token_lps.mean().item()

    def _process_single_native_expert(self, frames, context, max_new_tokens):
        """Apply expert to visual features, then use VL3's native generate pipeline."""
        inputs, pil_frames = self._prepare_native_inputs(frames, context)
        input_ids = inputs["input_ids"]
        original_encode = self._hf_model.encode_images

        def encode_with_expert(pixel_values, grid_sizes, merge_sizes):
            mm_features = original_encode(pixel_values, grid_sizes, merge_sizes)
            T = len(pil_frames)
            tpf = mm_features.shape[0] // T
            z_base = mm_features.reshape(1, T, tpf, -1)
            z_final = self._apply_expert_router(z_base, input_ids=input_ids)
            return z_final.reshape(-1, z_final.shape[-1])

        if self._decode_mode == "dual_select":
            with torch.no_grad():
                # Generate base FIRST (clean model state) — completely separate inputs
                inputs_base = self._prepare_native_inputs(frames, context)[0]
                gen_out_base = self._native_generate(inputs_base, hook_fn=None, return_scores=True)

                # Generate with expert hook — completely separate inputs
                inputs_expert = self._prepare_native_inputs(frames, context)[0]
                gen_out_expert = self._native_generate(inputs_expert, hook_fn=encode_with_expert, return_scores=True)

                gen_expert = self._tokenizer.decode(gen_out_expert.sequences[0], skip_special_tokens=True).strip()
                gen_base = self._tokenizer.decode(gen_out_base.sequences[0], skip_special_tokens=True).strip()

                # Compare by first option letter to handle format differences
                def _extract_letter(s):
                    s = s.strip().lstrip("(")
                    return s[0].upper() if s else ""
                expert_letter = _extract_letter(gen_expert)
                base_letter = _extract_letter(gen_base)

                # Same answer letter → pick base (safer, preserves vanilla)
                if expert_letter == base_letter:
                    return gen_base

                # Different answer letters → compare avg logprob from output_scores
                def _avg_logprob(gen_output):
                    if not hasattr(gen_output, 'scores') or not gen_output.scores:
                        return -float("inf")
                    total_lp = 0.0
                    n = 0
                    for step_scores, token_id in zip(gen_output.scores, gen_output.sequences[0, -len(gen_output.scores):]):
                        if token_id == self._tokenizer.eos_token_id:
                            break
                        log_p = F.log_softmax(step_scores[0].float(), dim=-1)
                        total_lp += log_p[token_id].item()
                        n += 1
                    return total_lp / n if n > 0 else -float("inf")

                lp_expert = _avg_logprob(gen_out_expert)
                lp_base = _avg_logprob(gen_out_base)

                # Debug
                if not hasattr(self, '_ds_debug_count'):
                    self._ds_debug_count = 0
                self._ds_debug_count += 1
                if self._ds_debug_count <= 10:
                    chosen = "expert" if lp_expert >= lp_base else "base"
                    eval_logger.info(f"[DS#{self._ds_debug_count}] e={expert_letter}({lp_expert:.2f}) b={base_letter}({lp_base:.2f}) → {chosen}")

                return gen_expert if lp_expert >= lp_base else gen_base

        # Normal mode: just generate with expert
        with torch.no_grad():
            output_ids = self._native_generate(inputs, hook_fn=encode_with_expert)
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _process_single(self, visual, context, max_new_tokens):
        """Process a single sample.

        - Vanilla (no expert): use VideoLLaMA3's native generate flow
        - UDH (expert loaded): use per-frame _encode_vision + expert + manual autoregressive
        """
        # Load frames
        frames = None
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            frames = self._load_video_frames(visual)
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            frames = [np.array(v.convert("RGB")) for v in visual]
        elif isinstance(visual, Image.Image):
            frames = [np.array(visual.convert("RGB"))]

        # UDH path: apply expert, then use native VL3 generate pipeline
        if self._expert is not None and frames is not None:
            return self._process_single_native_expert(frames, context, max_new_tokens)

            # Vanilla path: native VideoLLaMA3 generate (392 fixed for fair comparison)
        message = []
        q_text = re.sub(r"<video>\s*", "", context).strip()
        if frames is not None:
            ts = self._target_size  # 392
            pil_frames = [Image.fromarray(f).resize((ts, ts), Image.BILINEAR) for f in frames]
            message.append({"role": "user", "content": [
                {"type": "video", "video": pil_frames, "num_frames": len(pil_frames)},
                {"type": "text", "text": q_text}
            ]})
        elif isinstance(visual, Image.Image):
            message.append({"role": "user", "content": [
                {"type": "image", "image": visual},
                {"type": "text", "text": q_text}
            ]})
        else:
            message.append({"role": "user", "content": [
                {"type": "text", "text": context}
            ]})

        inputs = self._processor(conversation=message, return_tensors="pt", add_generation_prompt=True)
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.inference_mode():
            output_ids = self._hf_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        gen = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return gen

    def _log_progress(self, task_name, response, ground_truth, current, total):
        gt_letter = ""
        if ground_truth and isinstance(ground_truth, dict) and "candidates" in ground_truth:
            gt_text = ground_truth.get("answer", "")
            candidates = ground_truth["candidates"]
            for idx, c in enumerate(candidates):
                if c.strip() == gt_text.strip():
                    gt_letter = chr(65 + idx)
                    break
        is_correct = _mcq_acc_match(response, gt_letter) if gt_letter else False
        entry = {
            "type": "sample",
            "current": current,
            "total": total,
            "task": task_name,
            "response": response[:200],
            "ground_truth": str(
                ground_truth.get("answer", "") if isinstance(ground_truth, dict) else ground_truth
            )[:200],
            "correct": is_correct,
            "timestamp": time.time(),
            "rank": self._rank,
        }
        with open(self._progress_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for videollama3_udh")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("multi_round not implemented for videollama3_udh")
