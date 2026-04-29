"""
PLM-1B + MTP evaluation wrapper for lmms-eval.

Uses HF pipeline. MTP enriches vision features with temporal context via
pre-projector adapter (between vision encoder and projector MLP).
No token count changes — same T×256 tokens as vanilla.

Usage:
    accelerate launch --num_processes=4 -m lmms_eval \
        --model plm_mtp \
        --model_args pretrained=facebook/Perception-LM-1B,checkpoint_dir=./outputs/mtp_v5/ckpt_step_300 \
        --tasks mvbench --batch_size 1
"""

import json
import os
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    import decord
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


def _load_adapter(checkpoint_dir, device, dtype):
    """Load TemporalAdapter from checkpoint."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from core.vision_projector.mtp import TemporalAdapter

    adapter_path = os.path.join(checkpoint_dir, "adapter.pt")
    if not os.path.exists(adapter_path):
        eval_logger.warning(f"No adapter.pt at {adapter_path}")
        return None

    state = torch.load(adapter_path, map_location="cpu", weights_only=True)

    # Infer config from state dict
    bottleneck = state["down.weight"].shape[0]
    vision_dim = state["down.weight"].shape[1]
    max_frames = state["pos_embed"].shape[1]
    num_layers = sum(1 for k in state if k.startswith("layers.") and k.endswith(".self_attn.in_proj_weight"))

    # Auto-detect gate type and scale_bound from checkpoint keys
    has_gate_linear = "gate_linear.weight" in state
    has_gate_proj = "gate_proj.weight" in state
    if has_gate_linear:
        gate_type = "frame"
    else:
        gate_type = "dot_product"

    # Infer scale_bound from film_scale weight magnitude
    # Can't know exact value, use 0.1 as safe default (matches v6/v8)
    scale_bound = 0.1

    # Auto-detect spatial saliency from checkpoint keys
    has_spatial = "spatial_proj.weight" in state

    adapter = TemporalAdapter(
        vision_dim=vision_dim, bottleneck=bottleneck,
        num_heads=4, num_layers=num_layers, max_frames=max_frames,
        scale_bound=scale_bound, gate_type=gate_type,
        spatial_saliency=has_spatial)
    adapter.load_state_dict(state, strict=True)
    spatial_str = ", spatial_saliency" if has_spatial else ""
    eval_logger.info(f"Loaded adapter (gate={gate_type}{spatial_str}) from {adapter_path} "
                     f"(vision_dim={vision_dim}, bn={bottleneck}, "
                     f"scale_bound={scale_bound}, "
                     f"{sum(p.numel() for p in adapter.parameters()):,} params)")
    return adapter.to(device=device, dtype=dtype).eval()


def _load_lora_weights(checkpoint_dir, lora_params, device):
    """Load LoRA weights from checkpoint."""
    lora_path = os.path.join(checkpoint_dir, "lora.pt")
    if not os.path.exists(lora_path):
        eval_logger.warning(f"No lora.pt at {lora_path}")
        return False

    state = torch.load(lora_path, map_location="cpu", weights_only=True)
    loaded = 0
    for i, p in enumerate(lora_params):
        key = f"lora_{i}"
        if key in state:
            p.data = state[key].to(device=device, dtype=p.data.dtype)
            loaded += 1
    eval_logger.info(f"Loaded {loaded} LoRA parameter tensors")
    return True


@register_model("plm_mtp")
class PLMWithMTP(lmms):
    """PLM-1B + MTP: Masked Temporal Prediction adapter."""

    def __init__(
        self,
        pretrained: str = "facebook/Perception-LM-1B",
        checkpoint_dir: str = "none",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        max_video_frames: Optional[Union[int, str]] = 16,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        progress_dir: str = "/tmp/plm_eval_progress",
        **kwargs,
    ) -> None:
        super().__init__()

        from transformers import AutoModelForImageTextToText, AutoProcessor

        accelerator = Accelerator()
        self._device = torch.device(f"cuda:{accelerator.local_process_index}")

        # ── 1. Load PLM-1B ──
        eval_logger.info(f"Loading PLM-1B from {pretrained}")
        self._hf_model = AutoModelForImageTextToText.from_pretrained(
            pretrained, dtype=torch.bfloat16, device_map={"": self._device})
        self._hf_processor = AutoProcessor.from_pretrained(pretrained)

        for p in self._hf_model.parameters():
            p.requires_grad = False
        self._hf_model.eval()

        # ── 2. Load adapter + LoRA ──
        vanilla_mode = str(checkpoint_dir).lower() in ("none", "null", "") or \
            not os.path.isdir(str(checkpoint_dir))

        if vanilla_mode:
            eval_logger.info("=== VANILLA MODE: Pure PLM-1B (no MTP) ===")
            self._adapter = None
            self._lora_params = []
        else:
            eval_logger.info(f"Loading MTP from {checkpoint_dir}")
            param_dtype = next(self._hf_model.parameters()).dtype

            # Load adapter
            self._adapter = _load_adapter(
                checkpoint_dir, self._device, param_dtype)

            # Apply LoRA to LLM
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from core.vision_projector.hier_module import apply_lora_to_llm

            self._lora_params, n_replaced = apply_lora_to_llm(
                self._hf_model.model.language_model,
                rank=int(lora_rank), alpha=float(lora_alpha),
                target_modules=("q_proj", "v_proj"))
            eval_logger.info(f"Applied LoRA: {n_replaced} modules")

            # Load LoRA weights and cast to model dtype
            _load_lora_weights(checkpoint_dir, self._lora_params, self._device)
            for p in self._lora_params:
                p.data = p.data.to(dtype=param_dtype)

        # ── 3. Class variables ──
        self.batch_size_per_gpu = int(batch_size)
        self._max_video_frames = int(max_video_frames) if max_video_frames else 16

        if accelerator.num_processes > 1:
            self.accelerator = accelerator
            self._rank = accelerator.process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        self._progress_dir = progress_dir
        os.makedirs(self._progress_dir, exist_ok=True)
        self._progress_file = os.path.join(
            self._progress_dir, f"progress_rank{self._rank}.jsonl")
        with open(self._progress_file, "w"):
            pass

    @property
    def model(self):
        return self._hf_model

    @property
    def tokenizer(self):
        return self._hf_processor.tokenizer

    @property
    def eot_token_id(self):
        return self._hf_processor.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def max_video_frames(self):
        return self._max_video_frames

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # ── Video processing ──

    def _load_video_frames(self, video_path, max_frames=None):
        if not HAS_DECORD:
            raise RuntimeError("decord required")
        max_frames = max_frames or self._max_video_frames
        vr = decord.VideoReader(video_path, num_threads=1)
        total = len(vr)
        if total <= max_frames:
            indices = list(range(total))
            while len(indices) < max_frames:
                indices.append(total - 1)
        else:
            indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()
        frames = vr.get_batch(indices).asnumpy()
        return [frames[i] for i in range(frames.shape[0])]

    def _encode_video_with_mtp(self, pixel_values_videos):
        """Compute enriched video features using MTP adapter (pre-projector)."""
        B, T = pixel_values_videos.shape[:2]
        frames_flat = pixel_values_videos.flatten(0, 1)

        with torch.no_grad():
            # Vision tower → features
            vision_out = self._hf_model.model.vision_tower(frames_flat)
            hidden = vision_out.last_hidden_state
            if self._hf_model.config.vision_use_cls_token:
                hidden = hidden[:, 1:, :]

            # v5: Adapter BEFORE projector (in vision space)
            if self._adapter is not None:
                _, N, C = hidden.shape
                hidden_4d = hidden.reshape(B, T, N, C)
                enriched_4d = self._adapter(hidden_4d)
                hidden = enriched_4d.reshape(B * T, N, C)

            # Projector → projected tokens (always clean projector output)
            projected = self._hf_model.model.multi_modal_projector(hidden)

        M = projected.shape[1]
        D = projected.shape[2]
        return projected.reshape(B, T * M, D)

    @torch.no_grad()
    def _generate_greedy(self, input_ids, pixel_values_videos, max_new_tokens=256):
        """Greedy generation with MTP-enriched features."""
        inputs_embeds = self._hf_model.model.get_input_embeddings()(input_ids)

        if pixel_values_videos is not None:
            video_features = self._encode_video_with_mtp(pixel_values_videos)
            video_features = video_features.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            _, special_video_mask = self._hf_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds,
                video_features=video_features)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_video_mask, video_features)

        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=input_ids.device)

        generated_tokens = []
        past_key_values = None
        cur_embeds = inputs_embeds
        cur_mask = attention_mask

        for _ in range(max_new_tokens):
            outputs = self._hf_model.model.language_model(
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
            if token_id == self._hf_processor.tokenizer.eos_token_id:
                break
            generated_tokens.append(token_id)

            cur_embeds = self._hf_model.model.get_input_embeddings()(next_token)
            cur_mask = torch.cat([
                cur_mask,
                torch.ones((1, 1), dtype=torch.long, device=cur_mask.device)
            ], dim=1)

        return self._hf_processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

    # ── Live progress tracking ──

    @staticmethod
    def _extract_letter(text: str) -> str:
        text = text.strip().rstrip(".")
        if len(text) == 1 and text.upper() in "ABCDEFGH":
            return text.upper()
        if len(text) >= 3 and text[0] == "(" and text[1].upper() in "ABCDEFGH" and text[2] == ")":
            return text[1].upper()
        if len(text) >= 2 and text[0].upper() in "ABCDEFGH" and text[1] in ".):, ":
            return text[0].upper()
        return ""

    @staticmethod
    def _strip_letter_prefix(text: str) -> str:
        text = text.strip()
        if len(text) >= 4 and text[0] == "(" and text[1].upper() in "ABCDEFGH" and text[2] == ")":
            return text[3:].strip()
        if len(text) >= 2 and text[0].upper() in "ABCDEFGH" and text[1] in ".):, ":
            return text[2:].strip()
        return text

    @staticmethod
    def _mcq_acc_match(pred: str, gt_option_letter: str) -> bool:
        """Match lmms-eval's mcq_acc scoring exactly."""
        import re
        punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=",
                 "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

        def process(answer):
            option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
            match = option_regex.match(answer.strip())
            if match:
                return match.group(1).upper()
            answer = answer.replace("\n", " ").replace("\t", " ").strip()
            for p in punct:
                answer = answer.replace(p, " " if p + " " in answer or " " + p in answer else " ")
            answer = answer.strip("'").strip('"').strip(")").strip("(").strip().lower()
            letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()
            return answer

        return process(pred) == gt_option_letter

    def _log_progress(self, current, total, task, split, did, response):
        gt = None
        gt_option_letter = None
        try:
            doc = self.task_dict[task][split][did]
            gt = doc.get("answer", doc.get("correct_answer", None))
            if gt is not None:
                import string
                candidates = doc.get("candidates", [])
                option_letters = string.ascii_uppercase
                for i, candidate in enumerate(candidates):
                    if candidate == gt:
                        gt_option_letter = option_letters[i]
                        break
        except Exception:
            pass

        correct = None
        if gt_option_letter is not None:
            correct = self._mcq_acc_match(response, gt_option_letter)
        elif gt is not None:
            resp_letter = self._extract_letter(response)
            gt_letter = self._extract_letter(str(gt).strip())
            if resp_letter and gt_letter:
                correct = resp_letter == gt_letter

        entry = {
            "type": "sample",
            "current": current,
            "total": total,
            "task": task,
            "response": response,
            "ground_truth": str(gt) if gt is not None else None,
            "correct": correct,
            "timestamp": time.time(),
            "rank": self._rank,
        }
        with open(self._progress_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Eval interface ──

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for PLM")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        with open(self._progress_file, "w") as f:
            f.write(json.dumps({
                "type": "start",
                "total": len(requests),
                "world_size": self._world_size,
                "rank": self._rank,
                "timestamp": time.time(),
            }) + "\n")

        pbar = tqdm(total=len(requests), disable=(self.rank != 0),
                    desc="MTP Eval")
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            lambda x: (-len(self._hf_processor.tokenizer.encode(x[0])), x[0]),
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids])
                       for ids in doc_id]
            visuals = self.flatten(visuals)

            for i, context in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None
                gen_kwargs = all_gen_kwargs[i]
                max_new_tokens = gen_kwargs.get("max_new_tokens", 256)

                if isinstance(visual, str) and visual.endswith(
                        (".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    frames = self._load_video_frames(visual)
                elif isinstance(visual, (list, tuple)) and all(
                        isinstance(v, Image.Image) for v in visual):
                    frames = [np.array(v.convert("RGB")) for v in visual]
                elif isinstance(visual, Image.Image):
                    frames = [np.array(visual.convert("RGB"))]
                else:
                    frames = None

                if frames is not None:
                    messages = [{"role": "user", "content": [
                        {"type": "video"},
                        {"type": "text", "text": context},
                    ]}]
                    text = self._hf_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    inputs = self._hf_processor(
                        text=text, videos=[frames], return_tensors="pt")
                else:
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": context},
                    ]}]
                    text = self._hf_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    inputs = self._hf_processor(text=text, return_tensors="pt")

                input_ids = inputs["input_ids"].to(self._device)
                pixel_values_videos = inputs.get("pixel_values_videos")
                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.to(
                        self._device, dtype=torch.bfloat16)

                gen = self._generate_greedy(
                    input_ids, pixel_values_videos,
                    max_new_tokens=max_new_tokens)

                if gen.endswith("."):
                    gen = gen[:-1]

                res.append(gen)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), gen)
                pbar.update(1)
                self._log_progress(
                    len(res), len(requests), task, split, doc_id[i], gen)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented.")
