"""
lmms-eval wrapper for InternVL3 + UDH v2 (SplitDualHeadExpert).

Supports:
  - vanilla: checkpoint_dir="none" → pure InternVL3
  - udh:     checkpoint_dir=<path> → loads diff_expert.pt, hooks extract_feature

Registered as "internvl3_udh" for lmms-eval.

Usage:
    python -m lmms_eval \
        --model internvl3_udh \
        --model_args checkpoint_dir=none,vlm_base=models/InternVL3-2B \
        --tasks mvbench \
        --batch_size 1
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

eval_logger = logging.getLogger("lmms-eval")


def _mcq_acc_match(response: str, gt_letter: str) -> bool:
    if not gt_letter:
        return False
    resp = response.strip()
    if not resp:
        return False
    first = resp.lstrip("(")[0].upper() if resp.lstrip("(") else ""
    return first == gt_letter.upper()


@register_model("internvl3_udh")
class InternVL3UDH(lmms):
    """InternVL3 + UDH v2 dual-branch expert evaluation wrapper."""

    def __init__(
        self,
        vlm_base: str = "models/InternVL3-2B",
        checkpoint_dir: str = "none",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        max_video_frames: Optional[Union[int, str]] = 32,
        progress_dir: str = "/tmp/plm_eval_progress",
        **kwargs,
    ) -> None:
        super().__init__()
        from accelerate import Accelerator
        from transformers import AutoModel, AutoTokenizer

        accelerator = Accelerator()
        self._device = torch.device(f"cuda:{accelerator.local_process_index}")

        vlm_path = vlm_base
        if not os.path.isdir(vlm_path):
            for candidate in [os.path.join("models", os.path.basename(vlm_path))]:
                if os.path.isdir(candidate):
                    vlm_path = candidate
                    break

        eval_logger.info(f"Loading InternVL3 from {vlm_path}")
        self._hf_model = AutoModel.from_pretrained(
            vlm_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map={"": self._device},
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            vlm_path, trust_remote_code=True,
        )
        for p in self._hf_model.parameters():
            p.requires_grad = False
        self._hf_model.eval()

        self._num_image_token = self._hf_model.num_image_token  # 256
        self._max_video_frames = int(max_video_frames) if max_video_frames else 32

        # Load adapter
        self._expert = None
        self._tq_router = None
        self._adapter_cfg = {}
        vanilla_mode = str(checkpoint_dir).lower() in ("none", "null", "") or not os.path.isdir(str(checkpoint_dir))

        if vanilla_mode:
            eval_logger.info("=== VANILLA MODE (no adapter) ===")
        else:
            self._load_adapter(checkpoint_dir)

        self.batch_size_per_gpu = int(batch_size)

        if accelerator.num_processes > 1:
            self._rank = accelerator.process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        self._progress_dir = progress_dir
        os.makedirs(self._progress_dir, exist_ok=True)
        self._progress_file = os.path.join(self._progress_dir, f"progress_rank{self._rank}.jsonl")
        with open(self._progress_file, "w"):
            pass

    def _load_adapter(self, checkpoint_dir):
        from core.vision_projector.dual_proj import SplitDualHeadExpert

        cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
        if not os.path.exists(cfg_path):
            eval_logger.warning(f"No adapter_config.json in {checkpoint_dir}")
            return

        with open(cfg_path) as f:
            cfg = json.load(f)
        self._adapter_cfg = cfg

        # Stage 1 config를 참조 (Stage 2인 경우 stage1_config에서 expert 설정 가져옴)
        is_stage2_or_3 = cfg.get("module_type", "") in (
            "internvl3_udh_v2_stage2_tqrouter", "internvl3_udh_v2_stage3_lora",
        )
        expert_cfg = cfg.get("stage1_config", cfg) if is_stage2_or_3 else cfg

        proj_dim = cfg.get("proj_dim", self._hf_model.config.hidden_size)
        expert = SplitDualHeadExpert(
            proj_dim=proj_dim,
            bottleneck=expert_cfg.get("expert_bottleneck", 128),
            kernel_size=expert_cfg.get("expert_kernel_size", 3),
            pooling_ratio=expert_cfg.get("pooling_ratio", 1),
            ortho_proj=expert_cfg.get("ortho_proj", False),
            max_local_gate=expert_cfg.get("max_local_gate", 0.35),
            max_state_gate=expert_cfg.get("max_state_gate", 0.35),
            skip_output_norm=expert_cfg.get("skip_output_norm", False),
        ).to(device=self._device, dtype=torch.bfloat16)

        expert_path = os.path.join(checkpoint_dir, "diff_expert.pt")
        expert.load_state_dict(torch.load(expert_path, map_location=self._device, weights_only=True))
        expert.eval()
        # Stage 1이 video_only(no_gate)로 학습됐으면 eval에서도 no_gate
        if expert_cfg.get("video_only", False):
            expert.no_gate = True
            eval_logger.info("  Expert no_gate=True (from stage1 video_only)")
        self._expert = expert
        eval_logger.info(f"Loaded UDH expert from {checkpoint_dir}")

        # Stage 2/3: TemporalQueryRouter 로드
        # Stage 3의 경우 router_hidden이 stage2_config 안에 있음
        self._tq_router = None
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
            ).to(device=self._device, dtype=torch.bfloat16)
            tq_router.load_state_dict(torch.load(tq_router_path, map_location=self._device, weights_only=True))
            tq_router.eval()
            self._tq_router = tq_router
            eval_logger.info(f"Loaded TemporalQueryRouter from {checkpoint_dir}")

        # Load LoRA if present
        lora_dir = os.path.join(checkpoint_dir, "lora")
        if os.path.isdir(lora_dir):
            from peft import PeftModel
            self._hf_model.language_model = PeftModel.from_pretrained(
                self._hf_model.language_model, lora_dir
            ).to(device=self._device, dtype=torch.bfloat16)
            self._hf_model.language_model.eval()
            eval_logger.info(f"Loaded LoRA from {lora_dir}")

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
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(path, ctx=cpu(0), num_threads=1)
            total = len(vr)
            if total == 0:
                return None
            n = min(self._max_video_frames, total)
            indices = [int(i * total / n) for i in range(n)]
            frames = vr.get_batch(indices).asnumpy()
            return [frames[i] for i in range(len(indices))]
        except Exception:
            return None

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = __import__("tqdm").tqdm(total=len(requests), disable=(self._rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [r.args for r in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            visual = visuals[0] if visuals else None
            max_new_tokens = gen_kwargs.get("max_new_tokens", 256)

            try:
                gen = self._process_single(visual, contexts, max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                eval_logger.warning(f"OOM on sample {len(res)}, returning empty")
                torch.cuda.empty_cache()
                gen = ""

            if gen.endswith("."):
                gen = gen[:-1]

            res.append(gen)
            pbar.update(1)

            # Log progress
            doc = self.task_dict[task][split][doc_id]
            self._log_progress(task, gen, doc, len(res), len(requests))

        pbar.close()
        return res

    def _process_single(self, visual, context, max_new_tokens):
        frames = None
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            frames = self._load_video_frames(visual)
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            frames = [np.array(v.convert("RGB")) for v in visual]
        elif isinstance(visual, Image.Image):
            frames = [np.array(visual.convert("RGB"))]

        if frames is None:
            # Text-only
            response = self._hf_model.chat(
                self._tokenizer, None, context,
                dict(max_new_tokens=max_new_tokens, do_sample=False),
            )
            return response if isinstance(response, str) else response[0]

        # Build pixel_values using InternVL3's transform
        from lmms_eval.models.simple.internvl3 import build_transform, dynamic_preprocess
        transform = build_transform(input_size=448)
        pil_frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]

        all_patches = []
        num_patches_list = []
        for frame in pil_frames:
            patches = dynamic_preprocess(frame, image_size=448, max_num=1)
            pixel_values_frame = torch.stack([transform(p) for p in patches])
            all_patches.append(pixel_values_frame)
            num_patches_list.append(pixel_values_frame.shape[0])

        pixel_values = torch.cat(all_patches, dim=0).to(dtype=torch.bfloat16, device=self._device)

        # Build question with video prefix
        q_text = re.sub(r"<video>\s*", "", context).strip()
        video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(pil_frames))])
        question = video_prefix + q_text

        gen_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

        if self._expert is not None:
            # Hook: apply expert to extract_feature output
            T = len(pil_frames)
            original_extract = self._hf_model.extract_feature
            tq_router = self._tq_router
            hf_model_ref = self._hf_model

            # Stage 2 router가 있으면 질문 embedding 미리 추출
            q_embed_for_router = None
            if tq_router is not None:
                from transformers import AutoTokenizer
                tokenizer = self._tokenizer
                q_text_clean = re.sub(r"<video>\s*", "", context).strip()
                q_ids = tokenizer(q_text_clean, return_tensors="pt").input_ids.to(self._device)
                with torch.no_grad():
                    q_emb_all = hf_model_ref.get_input_embeddings()(q_ids)  # (1, seq, D)
                    q_embed_for_router = q_emb_all.mean(dim=1)  # (1, D)

            def extract_with_expert(pv):
                features = original_extract(pv)
                # features: (N_total_patches, tokens_per_patch, dim)
                # For video with max_num=1: N_total_patches = T, tokens = 256
                tpf = features.shape[1]
                z_base = features.reshape(1, T, tpf, -1)

                if tq_router is not None:
                    # Stage 2: router로 프레임별 gate 적용
                    _, local_res, state_res, _ = self._expert(z_base, z_base, return_aux=True)
                    z_base_summary = z_base.mean(dim=2)   # (1, T, D)
                    local_summary = local_res.mean(dim=2)  # (1, T, D)
                    state_summary = state_res.mean(dim=2)  # (1, T, D)
                    local_scale, state_scale = tq_router(
                        q_embed_for_router, z_base_summary, local_summary, state_summary,
                    )
                    # Enrichment normalize + scale 적용
                    base_norm = z_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    local_norm = local_res * (base_norm / local_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                    state_norm = state_res * (base_norm / state_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                    ls = local_scale.unsqueeze(-1)
                    ss = state_scale.unsqueeze(-1)
                    z_temp = (1.0 - ls - ss) * z_base + ls * local_norm + ss * state_norm
                elif self._adapter_cfg.get("fixed_enrich_scale", 0) > 0:
                    # LoRA-only: expert enrichment를 고정 scale로 적용
                    _, local_res, state_res, _ = self._expert(z_base, z_base, return_aux=True)
                    sc = self._adapter_cfg["fixed_enrich_scale"]
                    base_norm = z_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    local_norm = local_res * (base_norm / local_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                    state_norm = state_res * (base_norm / state_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                    z_temp = (1.0 - 2 * sc) * z_base + sc * local_norm + sc * state_norm
                else:
                    # Stage 1: expert의 기본 출력 (내부 gate 사용)
                    z_temp, _, _, _ = self._expert(z_base, z_base, return_aux=True)

                return z_temp.reshape(T, tpf, -1)

            self._hf_model.extract_feature = extract_with_expert
            try:
                response, _ = self._hf_model.chat(
                    self._tokenizer, pixel_values, question, gen_config,
                    num_patches_list=num_patches_list, history=None, return_history=True,
                )
            finally:
                self._hf_model.extract_feature = original_extract
        else:
            response, _ = self._hf_model.chat(
                self._tokenizer, pixel_values, question, gen_config,
                num_patches_list=num_patches_list, history=None, return_history=True,
            )

        return response

    def _log_progress(self, task_name, response, doc, current, total):
        gt_letter = ""
        if isinstance(doc, dict) and "candidates" in doc:
            gt_text = doc.get("answer", "")
            candidates = doc["candidates"]
            for idx, c in enumerate(candidates):
                if c.strip() == gt_text.strip():
                    gt_letter = chr(65 + idx)
                    break
        is_correct = _mcq_acc_match(response, gt_letter) if gt_letter else False
        try:
            with open(self._progress_file, "a") as f:
                f.write(json.dumps({
                    "task": task_name, "current": current, "total": total,
                    "correct": is_correct, "response": response[:100],
                }) + "\n")
        except Exception:
            pass

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("multi_round not implemented")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
