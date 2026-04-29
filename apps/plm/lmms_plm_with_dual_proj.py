"""
PLM-1B + Dual Projector evaluation wrapper for lmms-eval.

Supports both Design A (DeltaNet) and Design B (DPM).
Auto-detects module_type from adapter_config.json.

Usage:
    accelerate launch --num_processes=4 -m lmms_eval \
        --model plm_dual_proj \
        --model_args pretrained=facebook/Perception-LM-1B,checkpoint_dir=./outputs/dual_proj_v1/ckpt_step_300 \
        --tasks mvbench --batch_size 1
"""

import json
import math
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

PAD_ID = 128001
VIDEO_TOKEN_ID = 128003
END_HEADER_ID = 128007
EOT_ID = 128009
ROUTER_MARKER_TEXTS = [
    "Option:",
    "Options:",
    "Only give the best option.",
    "Please directly give the best option:",
]

QUESTION_TYPE_KEYWORDS = {
    "temporal": ["when", "before", "after", "first", "last", "order", "sequence"],
    "counting": ["how many", "count", "number of"],
    "spatial": ["where", "location", "position", "direction", "left", "right"],
    "action": ["what is", "what does", "doing", "action", "activity"],
    "state": ["change", "transition", "state", "happen"],
}


def _mcq_acc_match(pred_text: str, gt_letter: str) -> bool:
    pred = pred_text.strip().upper()
    gt = gt_letter.strip().upper()
    if not gt:
        return False
    if pred.startswith(f"({gt})"):
        return True
    if pred.startswith(gt) and (len(pred) == 1 or not pred[1].isalpha()):
        return True
    return False


def _apply_delta_clamp(z_base: torch.Tensor, z_target: torch.Tensor, max_ratio) -> torch.Tensor:
    """Clamp adapter delta magnitude relative to base feature norm.

    This keeps short screening runs from pushing projected features too far away
    from the base projector distribution at eval time.
    """
    if isinstance(max_ratio, (int, float)):
        if max_ratio <= 0:
            return z_target
    elif isinstance(max_ratio, torch.Tensor):
        if torch.all(max_ratio <= 0):
            return z_target
    delta = z_target - z_base
    base_norm = z_base.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
    delta_norm = delta.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ratio = delta_norm / base_norm
    scale = torch.where(ratio > max_ratio, max_ratio / ratio, torch.ones_like(ratio))
    return z_base + delta * scale.to(delta.dtype)


def _apply_relative_delta_clamp(delta: torch.Tensor, ref_delta: torch.Tensor, max_ratio: float) -> torch.Tensor:
    ref_norm = ref_delta.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
    delta_norm = delta.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ratio = delta_norm / ref_norm
    scale = torch.where(ratio > max_ratio, max_ratio / ratio, torch.ones_like(ratio))
    return delta * scale.to(delta.dtype)


def build_router_marker_ids(tokenizer):
    marker_ids = []
    seen = set()
    for text in ROUTER_MARKER_TEXTS:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        key = tuple(ids)
        if ids and key not in seen:
            marker_ids.append(ids)
            seen.add(key)
    return marker_ids


def build_question_type_keyword_ids(tokenizer):
    keyword_ids = {}
    for key, patterns in QUESTION_TYPE_KEYWORDS.items():
        ids_list = []
        seen = set()
        for text in patterns:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            key_ids = tuple(ids)
            if ids and key_ids not in seen:
                ids_list.append(ids)
                seen.add(key_ids)
        keyword_ids[key] = ids_list
    return keyword_ids


def _load_dual_proj_adapter(checkpoint_dir, base_model, device, dtype):
    """Load dual projector adapter (Design A or B) from checkpoint."""
    import sys
    from pathlib import Path
    # Add perception_models root — works both from original location and lmms-eval copy
    for candidate in [
        str(Path(__file__).resolve().parent.parent.parent),
        "/workspace/perception_models",
    ]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        eval_logger.warning(f"No adapter_config.json at {config_path}")
        return None, None, None

    with open(config_path) as f:
        cfg = json.load(f)

    module_type = cfg.get("module_type", "")
    base_proj = base_model.model.multi_modal_projector

    if module_type in ("dual_proj_deltanet", "dual_proj_deltanet_v12_stage1"):
        # Design A: ProjectorLoRA + DeltaNetMerger
        from core.vision_projector.dual_proj import ProjectorLoRA, DeltaNetMerger

        proj_lora = ProjectorLoRA(
            base_proj,
            rank=cfg.get("proj_lora_rank", 32),
            alpha=cfg.get("proj_lora_rank", 32) * 2.0,
        )
        proj_lora_path = os.path.join(checkpoint_dir, "proj_lora.pt")
        proj_lora.load_state_dict(
            torch.load(proj_lora_path, map_location="cpu", weights_only=True))
        proj_lora = proj_lora.to(device=device, dtype=dtype).eval()

        merger = DeltaNetMerger(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("merger_bottleneck", 128),
            max_alpha=cfg.get("max_alpha", 0.15),
            alpha_mode=cfg.get("alpha_mode", "scaled"),
            use_enrich_norm=cfg.get("use_enrich_norm", False),
        )
        merger_path = os.path.join(checkpoint_dir, "merger.pt")
        merger.load_state_dict(
            torch.load(merger_path, map_location="cpu", weights_only=True))
        merger = merger.to(device=device, dtype=dtype).eval()

        total_params = sum(p.numel() for p in proj_lora.parameters()) + \
                       sum(p.numel() for p in merger.parameters())
        label = "Design A v12 stage1" if module_type == "dual_proj_deltanet_v12_stage1" else "Design A (DeltaNet)"
        eval_logger.info(
            f"Loaded {label}: proj_lora + merger = {total_params:,} params "
            f"(alpha_mode={cfg.get('alpha_mode', 'scaled')})")
        return "deltanet", (proj_lora, merger), cfg

    elif module_type == "dual_proj_deltanet_v61":
        # Design A v6.1: ProjectorLoRA + DeltaNetMergerV61 (base-aware)
        from core.vision_projector.dual_proj import ProjectorLoRA, DeltaNetMerger
        import importlib.util, re as _re

        proj_lora = ProjectorLoRA(
            base_proj,
            rank=cfg.get("proj_lora_rank", 32),
            alpha=cfg.get("proj_lora_rank", 32) * 2.0,
        )
        proj_lora_path = os.path.join(checkpoint_dir, "proj_lora.pt")
        proj_lora.load_state_dict(
            torch.load(proj_lora_path, map_location="cpu", weights_only=True))
        proj_lora = proj_lora.to(device=device, dtype=dtype).eval()

        # Dynamically load DeltaNetMergerV61 from training script
        v61_path = os.path.join(
            str(Path(__file__).resolve().parent.parent.parent),
            "scripts", "train_dual_proj_v6_1.py")
        if not os.path.exists(v61_path):
            v61_path = "/workspace/perception_models/scripts/train_dual_proj_v6_1.py"
        exec_globals = {
            "nn": torch.nn, "torch": torch, "F": torch.nn.functional,
            "DeltaNetMerger": DeltaNetMerger,
        }
        with open(v61_path) as f:
            source = f.read()
        match = _re.search(
            r'(class DeltaNetMergerV61\(DeltaNetMerger\):.*?)(?=\n# ════|\nclass (?!DeltaNetMergerV61))',
            source, _re.DOTALL)
        if match:
            exec(match.group(1), exec_globals)
        DeltaNetMergerV61 = exec_globals.get("DeltaNetMergerV61")
        if DeltaNetMergerV61 is None:
            raise RuntimeError("Could not load DeltaNetMergerV61 from train_dual_proj_v6_1.py")

        merger = DeltaNetMergerV61(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("merger_bottleneck", 128),
            max_alpha=cfg.get("max_alpha", 0.15),
            alpha_mode=cfg.get("alpha_mode", "frame_mlp"),
            use_enrich_norm=cfg.get("use_enrich_norm", False),
        )
        merger_path = os.path.join(checkpoint_dir, "merger.pt")
        merger.load_state_dict(
            torch.load(merger_path, map_location="cpu", weights_only=True))
        merger = merger.to(device=device, dtype=dtype).eval()

        total_params = sum(p.numel() for p in proj_lora.parameters()) + \
                       sum(p.numel() for p in merger.parameters())
        eval_logger.info(
            f"Loaded Design A v6.1 (DeltaNet+BaseAware): proj_lora + merger = {total_params:,} params")
        return "deltanet", (proj_lora, merger), cfg

    elif module_type == "dual_proj_deltanet_v62":
        # Design A v6.2: ProjectorLoRA + DeltaNetMergerV62 (base-aware + group-wise alpha)
        from core.vision_projector.dual_proj import ProjectorLoRA, DeltaNetMerger
        import importlib.util, re as _re

        proj_lora = ProjectorLoRA(
            base_proj,
            rank=cfg.get("proj_lora_rank", 32),
            alpha=cfg.get("proj_lora_rank", 32) * 2.0,
        )
        proj_lora_path = os.path.join(checkpoint_dir, "proj_lora.pt")
        proj_lora.load_state_dict(
            torch.load(proj_lora_path, map_location="cpu", weights_only=True))
        proj_lora = proj_lora.to(device=device, dtype=dtype).eval()

        # Dynamically load DeltaNetMergerV62 from training script
        v62_path = os.path.join(
            str(Path(__file__).resolve().parent.parent.parent),
            "scripts", "train_dual_proj_v6_2.py")
        if not os.path.exists(v62_path):
            v62_path = "/workspace/perception_models/scripts/train_dual_proj_v6_2.py"
        exec_globals = {
            "nn": torch.nn, "torch": torch, "F": torch.nn.functional,
            "DeltaNetMerger": DeltaNetMerger,
        }
        with open(v62_path) as f:
            source = f.read()
        match = _re.search(
            r'(class DeltaNetMergerV62\(DeltaNetMerger\):.*?)(?=\n# ════|\nclass (?!DeltaNetMergerV62))',
            source, _re.DOTALL)
        if match:
            exec(match.group(1), exec_globals)
        DeltaNetMergerV62 = exec_globals.get("DeltaNetMergerV62")
        if DeltaNetMergerV62 is None:
            raise RuntimeError("Could not load DeltaNetMergerV62 from train_dual_proj_v6_2.py")

        merger = DeltaNetMergerV62(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("merger_bottleneck", 128),
            max_alpha=cfg.get("max_alpha", 0.15),
            alpha_mode=cfg.get("alpha_mode", "frame_mlp"),
            use_enrich_norm=cfg.get("use_enrich_norm", False),
            num_alpha_groups=cfg.get("num_alpha_groups", 8),
        )
        merger_path = os.path.join(checkpoint_dir, "merger.pt")
        merger.load_state_dict(
            torch.load(merger_path, map_location="cpu", weights_only=True))
        merger = merger.to(device=device, dtype=dtype).eval()

        total_params = sum(p.numel() for p in proj_lora.parameters()) + \
                       sum(p.numel() for p in merger.parameters())
        eval_logger.info(
            f"Loaded Design A v6.2 (DeltaNet+GroupAlpha): proj_lora + merger = {total_params:,} params "
            f"(groups={cfg.get('num_alpha_groups', 8)})")
        return "deltanet", (proj_lora, merger), cfg

    elif module_type == "dpm_v4":
        # Design B v4: TemporalProjectorV4 (frame-wise gate) + FusionGate
        from core.vision_projector.dual_proj import TemporalProjectorV4, FusionGate

        temp_proj = TemporalProjectorV4(
            base_proj,
            lora_rank=cfg.get("lora_rank", 32),
            lora_alpha=cfg.get("lora_rank", 32) * 2.0,
            temporal_bottleneck=cfg.get("temporal_bottleneck", 64),
            temporal_gate_init=cfg.get("temporal_gate_init", -2.0),
            max_tb_gate=cfg.get("max_tb_gate", 0.5),
        )
        tp_path = os.path.join(checkpoint_dir, "temp_proj.pt")
        temp_proj.load_state_dict(
            torch.load(tp_path, map_location="cpu", weights_only=True))
        temp_proj = temp_proj.to(device=device, dtype=dtype).eval()

        fusion_gate = FusionGate(
            proj_dim=cfg.get("proj_dim", 2048),
            hidden=cfg.get("gate_hidden", 256),
            max_gate=cfg.get("max_gate", 0.15),
            num_gate_groups=cfg.get("num_gate_groups", 1),
        )
        fg_path = os.path.join(checkpoint_dir, "fusion_gate.pt")
        fusion_gate.load_state_dict(
            torch.load(fg_path, map_location="cpu", weights_only=True))
        fusion_gate = fusion_gate.to(device=device, dtype=dtype).eval()

        total_params = sum(p.numel() for p in temp_proj.parameters()) + \
                       sum(p.numel() for p in fusion_gate.parameters())
        eval_logger.info(
            f"Loaded Design B v4 (DPM): temp_proj_v4 + fusion_gate = {total_params:,} params "
            f"(gate_groups={cfg.get('num_gate_groups', 1)})")
        return "dpm", (temp_proj, fusion_gate), cfg

    elif module_type == "dpm_v6":
        # Design B v6: TemporalProjectorV4 + FusionGateV6 (tb_state input, random init)
        from core.vision_projector.dual_proj import TemporalProjectorV4
        # Import FusionGateV6 from training script (inline class)
        import importlib.util
        v6_path = os.path.join(
            str(Path(__file__).resolve().parent.parent.parent),
            "scripts", "train_dpm_v6.py")
        if not os.path.exists(v6_path):
            v6_path = "/workspace/perception_models/scripts/train_dpm_v6.py"
        spec = importlib.util.spec_from_file_location("train_dpm_v6", v6_path)
        v6_mod = importlib.util.module_from_spec(spec)
        # We only need FusionGateV6 class, not full module execution
        import types
        # Manually extract FusionGateV6 by exec'ing just the class
        exec_globals = {"nn": torch.nn, "torch": torch, "F": torch.nn.functional}
        with open(v6_path) as f:
            source = f.read()
        # Find and exec just the FusionGateV6 class
        import re as _re
        match = _re.search(r'(class FusionGateV6\(nn\.Module\):.*?)(?=\nclass |\n# ════)', source, _re.DOTALL)
        if match:
            exec(match.group(1), exec_globals)
        FusionGateV6 = exec_globals.get("FusionGateV6")
        if FusionGateV6 is None:
            raise RuntimeError("Could not load FusionGateV6 from train_dpm_v6.py")

        temp_proj = TemporalProjectorV4(
            base_proj,
            lora_rank=cfg.get("lora_rank", 32),
            lora_alpha=cfg.get("lora_rank", 32) * 2.0,
            temporal_bottleneck=cfg.get("temporal_bottleneck", 64),
            temporal_gate_init=cfg.get("temporal_gate_init", -2.0),
            max_tb_gate=cfg.get("max_tb_gate", 0.5),
        )
        tp_path = os.path.join(checkpoint_dir, "temp_proj.pt")
        temp_proj.load_state_dict(
            torch.load(tp_path, map_location="cpu", weights_only=True))
        temp_proj = temp_proj.to(device=device, dtype=dtype).eval()

        fusion_gate = FusionGateV6(
            proj_dim=cfg.get("proj_dim", 2048),
            tb_dim=cfg.get("tb_dim", cfg.get("temporal_bottleneck", 64)),
            hidden=cfg.get("gate_hidden", 256),
            max_gate=cfg.get("max_gate", 0.25),
            num_gate_groups=cfg.get("num_gate_groups", 1),
        )
        fg_path = os.path.join(checkpoint_dir, "fusion_gate.pt")
        fusion_gate.load_state_dict(
            torch.load(fg_path, map_location="cpu", weights_only=True))
        fusion_gate = fusion_gate.to(device=device, dtype=dtype).eval()

        total_params = sum(p.numel() for p in temp_proj.parameters()) + \
                       sum(p.numel() for p in fusion_gate.parameters())
        eval_logger.info(
            f"Loaded Design B v6 (DPM): temp_proj_v4 + FusionGateV6 = {total_params:,} params")
        return "dpm", (temp_proj, fusion_gate), cfg

    elif module_type == "dpm_v8":
        # Design B v8: TemporalProjectorV4 + FusionGateV6 + QuestionSuppressor
        from core.vision_projector.dual_proj import TemporalProjectorV4
        import importlib.util
        v6_path = os.path.join(
            str(Path(__file__).resolve().parent.parent.parent),
            "scripts", "train_dpm_v6.py")
        if not os.path.exists(v6_path):
            v6_path = "/workspace/perception_models/scripts/train_dpm_v6.py"
        exec_globals = {"nn": torch.nn, "torch": torch, "F": torch.nn.functional}
        with open(v6_path) as f:
            source = f.read()
        import re as _re
        match = _re.search(r'(class FusionGateV6\(nn\.Module\):.*?)(?=\nclass |\n# ════)', source, _re.DOTALL)
        if match:
            exec(match.group(1), exec_globals)
        FusionGateV6 = exec_globals.get("FusionGateV6")
        if FusionGateV6 is None:
            raise RuntimeError("Could not load FusionGateV6 from train_dpm_v6.py")

        temp_proj = TemporalProjectorV4(
            base_proj,
            lora_rank=cfg.get("lora_rank", 32),
            lora_alpha=cfg.get("lora_rank", 32) * 2.0,
            temporal_bottleneck=cfg.get("temporal_bottleneck", 64),
            temporal_gate_init=cfg.get("temporal_gate_init", -2.0),
            max_tb_gate=cfg.get("max_tb_gate", 0.5),
        )
        tp_path = os.path.join(checkpoint_dir, "temp_proj.pt")
        temp_proj.load_state_dict(
            torch.load(tp_path, map_location="cpu", weights_only=True))
        temp_proj = temp_proj.to(device=device, dtype=dtype).eval()

        fusion_gate = FusionGateV6(
            proj_dim=cfg.get("proj_dim", 2048),
            tb_dim=cfg.get("tb_dim", cfg.get("temporal_bottleneck", 64)),
            hidden=cfg.get("gate_hidden", 256),
            max_gate=cfg.get("max_gate", 0.25),
            num_gate_groups=cfg.get("num_gate_groups", 1),
        )
        fg_path = os.path.join(checkpoint_dir, "fusion_gate.pt")
        fusion_gate.load_state_dict(
            torch.load(fg_path, map_location="cpu", weights_only=True))
        fusion_gate = fusion_gate.to(device=device, dtype=dtype).eval()

        # QuestionSuppressor
        class QuestionSuppressor(torch.nn.Module):
            def __init__(self, embed_dim, hidden_dim=64):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
            def forward(self, q_embed):
                return torch.sigmoid(self.net(q_embed))

        embed_dim = base_model.model.get_input_embeddings().weight.shape[1]
        q_sup_hidden = cfg.get("q_suppressor_hidden", 64)
        q_suppressor = QuestionSuppressor(embed_dim=embed_dim, hidden_dim=q_sup_hidden)
        qs_path = os.path.join(checkpoint_dir, "q_suppressor.pt")
        if os.path.exists(qs_path):
            q_suppressor.load_state_dict(
                torch.load(qs_path, map_location="cpu", weights_only=True))
        q_suppressor = q_suppressor.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded Design B v8 (DPM): temp_proj_v4 + FusionGateV6 + QuestionSuppressor "
            f"(q_default={cfg.get('q_default', 'on')})")
        return "dpm_qsup", (temp_proj, fusion_gate, q_suppressor), cfg

    elif module_type in ("dpm_v10", "dpm_v10_1", "dpm_v10_2", "dpm_v10_3", "dpm_v10_4"):
        # Design B v10: DPM aggressive branch + dual_proj_v2-style safe branch + router
        from core.vision_projector.dual_proj import TemporalProjectorV4, ProjectorLoRA, DeltaNetMerger

        import re as _re
        v6_path = os.path.join(
            str(Path(__file__).resolve().parent.parent.parent),
            "scripts", "train_dpm_v6.py")
        if not os.path.exists(v6_path):
            v6_path = "/workspace/perception_models/scripts/train_dpm_v6.py"
        exec_globals = {"nn": torch.nn, "torch": torch, "F": torch.nn.functional}
        with open(v6_path) as f:
            source = f.read()
        match = _re.search(r'(class FusionGateV6\(nn\.Module\):.*?)(?=\nclass |\n# ════)', source, _re.DOTALL)
        if match:
            exec(match.group(1), exec_globals)
        FusionGateV6 = exec_globals.get("FusionGateV6")
        if FusionGateV6 is None:
            raise RuntimeError("Could not load FusionGateV6 from train_dpm_v6.py")

        temp_proj = TemporalProjectorV4(
            base_proj,
            lora_rank=cfg.get("lora_rank", 32),
            lora_alpha=cfg.get("lora_rank", 32) * 2.0,
            temporal_bottleneck=cfg.get("temporal_bottleneck", 64),
            temporal_gate_init=cfg.get("temporal_gate_init", -2.0),
            max_tb_gate=cfg.get("max_tb_gate", 0.5),
        )
        temp_proj.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "temp_proj.pt"), map_location="cpu", weights_only=True))
        temp_proj = temp_proj.to(device=device, dtype=dtype).eval()

        fusion_gate = FusionGateV6(
            proj_dim=cfg.get("proj_dim", 2048),
            tb_dim=cfg.get("tb_dim", cfg.get("temporal_bottleneck", 64)),
            hidden=cfg.get("gate_hidden", 256),
            max_gate=cfg.get("max_gate", 0.25),
            num_gate_groups=cfg.get("num_gate_groups", 1),
        )
        fusion_gate.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "fusion_gate.pt"), map_location="cpu", weights_only=True))
        fusion_gate = fusion_gate.to(device=device, dtype=dtype).eval()

        class QuestionSuppressor(torch.nn.Module):
            def __init__(self, embed_dim, hidden_dim=64):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
            def forward(self, q_embed):
                return torch.sigmoid(self.net(q_embed))

        class ExpertRouter(torch.nn.Module):
            def __init__(self, embed_dim, hidden_dim=128):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, 3),
                )
            def forward(self, q_embed):
                return torch.nn.functional.softmax(self.net(q_embed), dim=-1)

        embed_dim = base_model.model.get_input_embeddings().weight.shape[1]
        q_suppressor = QuestionSuppressor(
            embed_dim=embed_dim,
            hidden_dim=cfg.get("q_suppressor_hidden", 64),
        )
        q_suppressor.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "q_suppressor.pt"), map_location="cpu", weights_only=True))
        q_suppressor = q_suppressor.to(device=device, dtype=dtype).eval()

        safe_proj_lora = ProjectorLoRA(
            base_proj,
            rank=cfg.get("safe_proj_lora_rank", 32),
            alpha=cfg.get("safe_proj_lora_rank", 32) * 2.0,
        )
        safe_proj_lora.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "safe_proj_lora.pt"), map_location="cpu", weights_only=True))
        safe_proj_lora = safe_proj_lora.to(device=device, dtype=dtype).eval()

        safe_merger = DeltaNetMerger(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("safe_merger_bottleneck", 128),
            max_alpha=cfg.get("safe_max_alpha", 0.15),
            alpha_mode="scaled",
            use_enrich_norm=False,
        )
        safe_merger.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "safe_merger.pt"), map_location="cpu", weights_only=True))
        safe_merger = safe_merger.to(device=device, dtype=dtype).eval()

        router = ExpertRouter(
            embed_dim=cfg.get("embed_dim", embed_dim),
            hidden_dim=cfg.get("router_hidden", 128),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True))
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            "Loaded Design B v10 (DPM + safe branch + router)")
        return "dpm_router", (temp_proj, fusion_gate, q_suppressor, safe_proj_lora, safe_merger, router), cfg

    elif module_type in ("dpm_v11", "dpm_v12"):
        # Design B v11: DPM aggressive branch + minimal safe LoRA (no temporal) + router
        from core.vision_projector.dual_proj import TemporalProjectorV4

        import re as _re
        v6_path = os.path.join(
            str(Path(__file__).resolve().parent.parent.parent),
            "scripts", "train_dpm_v6.py")
        if not os.path.exists(v6_path):
            v6_path = "/workspace/perception_models/scripts/train_dpm_v6.py"
        exec_globals = {"nn": torch.nn, "torch": torch, "F": torch.nn.functional}
        with open(v6_path) as f:
            source = f.read()
        match = _re.search(r'(class FusionGateV6\(nn\.Module\):.*?)(?=\nclass |\n# ════)', source, _re.DOTALL)
        if match:
            exec(match.group(1), exec_globals)
        FusionGateV6 = exec_globals.get("FusionGateV6")
        if FusionGateV6 is None:
            raise RuntimeError("Could not load FusionGateV6 from train_dpm_v6.py")

        # Aggressive branch
        temp_proj = TemporalProjectorV4(
            base_proj,
            lora_rank=cfg.get("lora_rank", 32),
            lora_alpha=cfg.get("lora_rank", 32) * 2.0,
            temporal_bottleneck=cfg.get("temporal_bottleneck", 64),
            temporal_gate_init=cfg.get("temporal_gate_init", -2.0),
            max_tb_gate=cfg.get("max_tb_gate", 0.5),
        )
        temp_proj.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "temp_proj.pt"), map_location="cpu", weights_only=True))
        temp_proj = temp_proj.to(device=device, dtype=dtype).eval()

        fusion_gate = FusionGateV6(
            proj_dim=cfg.get("proj_dim", 2048),
            tb_dim=cfg.get("tb_dim", cfg.get("temporal_bottleneck", 64)),
            hidden=cfg.get("gate_hidden", 256),
            max_gate=cfg.get("max_gate", 0.25),
            num_gate_groups=cfg.get("num_gate_groups", 1),
        )
        fusion_gate.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "fusion_gate.pt"), map_location="cpu", weights_only=True))
        fusion_gate = fusion_gate.to(device=device, dtype=dtype).eval()

        # Safe branch: minimal LoRA (no temporal)
        class SafeProjectorLoRA(torch.nn.Module):
            def __init__(self, base_proj, rank=16, alpha=32.0):
                super().__init__()
                in_dim = base_proj.linear_2.in_features
                out_dim = base_proj.linear_2.out_features
                self.lora_A = torch.nn.Linear(in_dim, rank, bias=False)
                self.lora_B = torch.nn.Linear(rank, out_dim, bias=False)
                self.scale = alpha / rank
                self.base_proj = base_proj

            def forward(self, x):
                with torch.no_grad():
                    base_out = self.base_proj(x)
                h = torch.nn.functional.gelu(self.base_proj.linear_1(x))
                lora_corr = self.lora_B(self.lora_A(h)) * self.scale
                lora_pooled = self.base_proj.pooling(lora_corr)
                return base_out + lora_pooled

        safe_rank = cfg.get("safe_lora_rank", 16)
        safe_lora = SafeProjectorLoRA(base_proj, rank=safe_rank, alpha=safe_rank * 2.0)
        sl_data = torch.load(os.path.join(checkpoint_dir, "safe_lora.pt"), map_location="cpu", weights_only=True)
        safe_lora.lora_A.weight.data.copy_(sl_data["lora_A.weight"])
        safe_lora.lora_B.weight.data.copy_(sl_data["lora_B.weight"])
        safe_lora.scale = sl_data["scale"]
        safe_lora = safe_lora.to(device=device, dtype=dtype).eval()

        # Router
        class ExpertRouter(torch.nn.Module):
            def __init__(self, embed_dim, hidden_dim=128):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, 3),
                )
            def forward(self, q_embed):
                return torch.nn.functional.softmax(self.net(q_embed), dim=-1)

        embed_dim = base_model.model.get_input_embeddings().weight.shape[1]
        router = ExpertRouter(
            embed_dim=cfg.get("embed_dim", embed_dim),
            hidden_dim=cfg.get("router_hidden", 128),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True))
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded Design B v11 (DPM + minimal safe LoRA + router), safe_rank={safe_rank}")
        return "dpm_v11_router", (temp_proj, fusion_gate, safe_lora, router), cfg

    elif module_type in ("dpm_v13_stage1", "dpm_v13"):
        from core.vision_projector.dual_proj import TemporalProjectorV13

        temp_proj = TemporalProjectorV13(
            base_proj,
            temporal_bottleneck=cfg.get("temporal_bottleneck", 64),
        )
        temp_proj.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "temp_proj.pt"), map_location="cpu", weights_only=True))
        temp_proj = temp_proj.to(device=device, dtype=dtype).eval()

        if module_type == "dpm_v13_stage1":
            eval_logger.info("Loaded Design B v13 stage1 (pure aggressive DPM expert)")
            return "dpm_v13_stage1", (temp_proj,), cfg

        stage2_mode = cfg.get("stage2_mode", "alpha_mixer")
        if stage2_mode == "expert_router_v3":
            class TwoExpertRouter(torch.nn.Module):
                def __init__(self, embed_dim, hidden_dim=128, stats_dim=4):
                    super().__init__()
                    self.stats_dim = stats_dim
                    self.trunk = torch.nn.Sequential(
                        torch.nn.Linear(embed_dim + stats_dim, hidden_dim),
                        torch.nn.GELU(),
                    )
                    self.route_head = torch.nn.Linear(hidden_dim, 1)

                def forward(self, q_embed, video_stats=None):
                    if self.stats_dim > 0 and video_stats is not None:
                        x = torch.cat([q_embed, video_stats], dim=-1)
                    else:
                        x = q_embed
                    h = self.trunk(x)
                    return torch.sigmoid(self.route_head(h)).squeeze(-1)

            mixer = TwoExpertRouter(
                embed_dim=cfg.get("embed_dim", base_model.model.get_input_embeddings().weight.shape[1]),
                hidden_dim=cfg.get("mixer_hidden", 128),
                stats_dim=cfg.get("stats_dim", 4),
            )
        elif stage2_mode == "residual_router_v2":
            class ResidualRouter(torch.nn.Module):
                def __init__(self, embed_dim, hidden_dim=128, stats_dim=4):
                    super().__init__()
                    self.stats_dim = stats_dim
                    self.trunk = torch.nn.Sequential(
                        torch.nn.Linear(embed_dim + stats_dim, hidden_dim),
                        torch.nn.GELU(),
                    )
                    self.gate_head = torch.nn.Linear(hidden_dim, 1)
                    self.scale_head = torch.nn.Linear(hidden_dim, 1)

                def forward(self, q_embed, video_stats=None):
                    if self.stats_dim > 0 and video_stats is not None:
                        x = torch.cat([q_embed, video_stats], dim=-1)
                    else:
                        x = q_embed
                    h = self.trunk(x)
                    gate = torch.sigmoid(self.gate_head(h)).squeeze(-1)
                    scale = torch.sigmoid(self.scale_head(h)).squeeze(-1)
                    return gate, scale

            mixer = ResidualRouter(
                embed_dim=cfg.get("embed_dim", base_model.model.get_input_embeddings().weight.shape[1]),
                hidden_dim=cfg.get("mixer_hidden", 128),
                stats_dim=cfg.get("stats_dim", 4),
            )
        else:
            class QuestionMixer(torch.nn.Module):
                def __init__(self, embed_dim, hidden_dim=128, stats_dim=4):
                    super().__init__()
                    self.stats_dim = stats_dim
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(embed_dim + stats_dim, hidden_dim),
                        torch.nn.GELU(),
                        torch.nn.Linear(hidden_dim, 1),
                    )

                def forward(self, q_embed, video_stats=None):
                    if self.stats_dim > 0 and video_stats is not None:
                        x = torch.cat([q_embed, video_stats], dim=-1)
                    else:
                        x = q_embed
                    return torch.sigmoid(self.net(x)).squeeze(-1)

            mixer = QuestionMixer(
                embed_dim=cfg.get("embed_dim", base_model.model.get_input_embeddings().weight.shape[1]),
                hidden_dim=cfg.get("mixer_hidden", 128),
                stats_dim=cfg.get("stats_dim", 4),
            )
        mixer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "mixer.pt"), map_location="cpu", weights_only=True))
        mixer = mixer.to(device=device, dtype=dtype).eval()

        eval_logger.info(f"Loaded Design B v13 (frozen aggressive DPM expert + {stage2_mode})")
        return "dpm_temporal_mixer", (temp_proj, mixer), cfg

    elif module_type == "dpm_v1":
        # Design B: TemporalProjector + FusionGate
        from core.vision_projector.dual_proj import TemporalProjector, FusionGate

        temp_proj = TemporalProjector(
            base_proj,
            lora_rank=cfg.get("lora_rank", 32),
            lora_alpha=cfg.get("lora_rank", 32) * 2.0,
            temporal_bottleneck=cfg.get("temporal_bottleneck", 64),
            temporal_gate_init=cfg.get("temporal_gate_init", -1.0),
        )
        tp_path = os.path.join(checkpoint_dir, "temp_proj.pt")
        temp_proj.load_state_dict(
            torch.load(tp_path, map_location="cpu", weights_only=True))
        temp_proj = temp_proj.to(device=device, dtype=dtype).eval()

        fusion_gate = FusionGate(
            proj_dim=cfg.get("proj_dim", 2048),
            hidden=cfg.get("gate_hidden", 256),
            max_gate=cfg.get("max_gate", 0.15),
            num_gate_groups=cfg.get("num_gate_groups", 1),
        )
        fg_path = os.path.join(checkpoint_dir, "fusion_gate.pt")
        fusion_gate.load_state_dict(
            torch.load(fg_path, map_location="cpu", weights_only=True))
        fusion_gate = fusion_gate.to(device=device, dtype=dtype).eval()

        total_params = sum(p.numel() for p in temp_proj.parameters()) + \
                       sum(p.numel() for p in fusion_gate.parameters())
        eval_logger.info(
            f"Loaded Design B (DPM): temp_proj + fusion_gate = {total_params:,} params "
            f"(gate_groups={cfg.get('num_gate_groups', 1)})")
        return "dpm", (temp_proj, fusion_gate), cfg

    elif module_type in ("dual_proj_deltanet_v70", "dual_proj_deltanet_v71", "dual_proj_deltanet_v89_stage1"):
        # Design A v7.0/v7.1: ProjectorLoRA + DeltaNetMerger + QuestionGate
        # At inference time, same as dual_proj_deltanet plus question-gated enrichment
        from core.vision_projector.dual_proj import ProjectorLoRA, DeltaNetMerger

        proj_lora = ProjectorLoRA(
            base_proj,
            rank=cfg.get("proj_lora_rank", 32),
            alpha=cfg.get("proj_lora_rank", 32) * 2.0,
        )
        proj_lora_path = os.path.join(checkpoint_dir, "proj_lora.pt")
        proj_lora.load_state_dict(
            torch.load(proj_lora_path, map_location="cpu", weights_only=True))
        proj_lora = proj_lora.to(device=device, dtype=dtype).eval()

        merger = DeltaNetMerger(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("merger_bottleneck", 128),
            max_alpha=cfg.get("max_alpha", 0.15),
            alpha_mode=cfg.get("alpha_mode", "scaled"),
            use_enrich_norm=cfg.get("use_enrich_norm", False),
        )
        merger_path = os.path.join(checkpoint_dir, "merger.pt")
        merger.load_state_dict(
            torch.load(merger_path, map_location="cpu", weights_only=True))
        merger = merger.to(device=device, dtype=dtype).eval()

        # QuestionGate: simple MLP that modulates enrichment based on question embedding
        class QuestionGate(torch.nn.Module):
            def __init__(self, embed_dim, hidden_dim=64):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
            def forward(self, q_embed):
                return torch.sigmoid(self.net(q_embed))

        # Infer embed_dim from base_model's embedding layer
        embed_dim = base_model.model.get_input_embeddings().weight.shape[1]
        q_gate = QuestionGate(embed_dim=embed_dim)
        q_gate_path = os.path.join(checkpoint_dir, "q_gate.pt")
        q_gate.load_state_dict(
            torch.load(q_gate_path, map_location="cpu", weights_only=True))
        q_gate = q_gate.to(device=device, dtype=dtype).eval()

        total_params = sum(p.numel() for p in proj_lora.parameters()) + \
                       sum(p.numel() for p in merger.parameters()) + \
                       sum(p.numel() for p in q_gate.parameters())
        eval_logger.info(
            f"Loaded Design A {module_type} (DeltaNet+QGate): "
            f"proj_lora + merger + q_gate = {total_params:,} params")
        return "deltanet_qgate", (proj_lora, merger, q_gate), cfg

    elif module_type == "dual_proj_deltanet_v72":
        # Design A v7.2: HiddenTemporalBranch (no ProjectorLoRA, no DeltaNetMerger)
        # Operates in the hidden space between linear_1 and linear_2

        class HiddenTemporalBranch(torch.nn.Module):
            def __init__(self, hidden_dim=4096, bottleneck=128, max_gate=0.15):
                super().__init__()
                self.down = torch.nn.Linear(hidden_dim, bottleneck)
                self.beta_proj = torch.nn.Linear(bottleneck, bottleneck)
                self.q_proj = torch.nn.Linear(bottleneck, bottleneck)
                self.k_proj = torch.nn.Linear(bottleneck, bottleneck)
                self.up = torch.nn.Linear(bottleneck, hidden_dim)
                self.gate_head = torch.nn.Sequential(
                    torch.nn.Linear(bottleneck, 64),
                    torch.nn.GELU(),
                    torch.nn.Linear(64, 1),
                )
                self.max_gate = max_gate

            def forward(self, h):
                B, T, N, H = h.shape
                h_pool = h.mean(dim=2)  # (B, T, H)
                h_down = self.down(h_pool)  # (B, T, bottleneck)
                beta = torch.sigmoid(self.beta_proj(h_down))
                q = self.q_proj(h_down)
                k = self.k_proj(h_down)
                S = torch.zeros(B, h_down.shape[-1], h_down.shape[-1],
                                device=h.device, dtype=h.dtype)
                outputs = []
                for t in range(T):
                    S = beta[:, t, :].unsqueeze(-1) * S + \
                        k[:, t, :].unsqueeze(-1) * h_down[:, t, :].unsqueeze(1)
                    o_t = (q[:, t, :].unsqueeze(1) * S).sum(dim=-1)
                    outputs.append(o_t)
                recurrence = torch.stack(outputs, dim=1)  # (B, T, bottleneck)
                gate_logit = self.gate_head(recurrence).squeeze(-1)  # (B, T)
                gate = self.max_gate * torch.sigmoid(gate_logit)
                temporal_up = self.up(recurrence)  # (B, T, H)
                temporal_broadcast = temporal_up.unsqueeze(2).expand(B, T, N, H)
                gate_4d = gate.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
                return gate_4d * temporal_broadcast

        hidden_dim = cfg.get("hidden_dim", 4096)
        temporal_branch = HiddenTemporalBranch(
            hidden_dim=hidden_dim,
            bottleneck=cfg.get("hidden_bottleneck", cfg.get("temporal_bottleneck", 128)),
            max_gate=cfg.get("max_gate", 0.15),
        )
        tb_path = os.path.join(checkpoint_dir, "temporal_branch.pt")
        temporal_branch.load_state_dict(
            torch.load(tb_path, map_location="cpu", weights_only=True))
        temporal_branch = temporal_branch.to(device=device, dtype=dtype).eval()

        total_params = sum(p.numel() for p in temporal_branch.parameters())
        eval_logger.info(
            f"Loaded Design A v7.2 (HiddenTemporalBranch): {total_params:,} params "
            f"(bottleneck={cfg.get('hidden_bottleneck', 128)}, "
            f"max_gate={cfg.get('max_gate', 0.15)})")
        return "hidden_temporal", (temporal_branch,), cfg

    elif module_type in ("dual_proj_deltanet_v83", "dual_proj_deltanet_v85", "dual_proj_deltanet_v87", "dual_proj_deltanet_v88"):
        # Design A v8.3/v8.5/v8.7/v8.8: Two-Expert DeltaNet
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(checkpoint_dir).resolve().parent.parent.parent))
        from core.vision_projector.dual_proj import ProjectorLoRA, DeltaNetMerger

        # Expert S: late merger
        proj_lora = ProjectorLoRA(
            base_proj,
            rank=cfg.get("proj_lora_rank", 32),
            alpha=cfg.get("proj_lora_rank", 32) * 2.0,
        )
        merger = DeltaNetMerger(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("merger_bottleneck", 128),
            max_alpha=cfg.get("max_alpha", 0.15),
            alpha_mode=cfg.get("alpha_mode", "scaled"),
            use_enrich_norm=cfg.get("use_enrich_norm", False),
        )

        pl_path = os.path.join(checkpoint_dir, "proj_lora.pt")
        proj_lora.load_state_dict(
            torch.load(pl_path, map_location="cpu", weights_only=True))
        proj_lora = proj_lora.to(device=device, dtype=dtype).eval()

        mg_path = os.path.join(checkpoint_dir, "merger.pt")
        merger.load_state_dict(
            torch.load(mg_path, map_location="cpu", weights_only=True))
        merger = merger.to(device=device, dtype=dtype).eval()

        # Expert T: hidden-level temporal branch (same as v7.2)
        class HiddenTemporalBranch(torch.nn.Module):
            def __init__(self, hidden_dim=4096, bottleneck=128, max_gate=0.15):
                super().__init__()
                self.down = torch.nn.Linear(hidden_dim, bottleneck)
                self.beta_proj = torch.nn.Linear(bottleneck, bottleneck)
                self.q_proj = torch.nn.Linear(bottleneck, bottleneck)
                self.k_proj = torch.nn.Linear(bottleneck, bottleneck)
                self.up = torch.nn.Linear(bottleneck, hidden_dim)
                self.gate_head = torch.nn.Sequential(
                    torch.nn.Linear(bottleneck, 64),
                    torch.nn.GELU(),
                    torch.nn.Linear(64, 1),
                )
                self.max_gate = max_gate

            def forward(self, h):
                B, T, N, H = h.shape
                h_pool = h.mean(dim=2)
                h_down = self.down(h_pool)
                beta = torch.sigmoid(self.beta_proj(h_down))
                q = self.q_proj(h_down)
                k = self.k_proj(h_down)
                S = torch.zeros(B, h_down.shape[-1], h_down.shape[-1],
                                device=h.device, dtype=h.dtype)
                outputs = []
                for t in range(T):
                    S = beta[:, t, :].unsqueeze(-1) * S + \
                        k[:, t, :].unsqueeze(-1) * h_down[:, t, :].unsqueeze(1)
                    o_t = (q[:, t, :].unsqueeze(1) * S).sum(dim=-1)
                    outputs.append(o_t)
                recurrence = torch.stack(outputs, dim=1)
                gate_logit = self.gate_head(recurrence).squeeze(-1)
                gate = self.max_gate * torch.sigmoid(gate_logit)
                temporal_up = self.up(recurrence)
                temporal_broadcast = temporal_up.unsqueeze(2).expand(B, T, N, H)
                gate_4d = gate.unsqueeze(-1).unsqueeze(-1)
                return gate_4d * temporal_broadcast

        hidden_dim = cfg.get("hidden_dim", base_proj.linear_1.out_features)
        temporal_branch = HiddenTemporalBranch(
            hidden_dim=hidden_dim,
            bottleneck=cfg.get("hidden_bottleneck", 128),
            max_gate=cfg.get("max_gate_t", cfg.get("max_gate", 0.15)),
        )
        tb_path = os.path.join(checkpoint_dir, "temporal_branch.pt")
        temporal_branch.load_state_dict(
            torch.load(tb_path, map_location="cpu", weights_only=True))
        temporal_branch = temporal_branch.to(device=device, dtype=dtype).eval()

        # Router
        class ExpertRouter(torch.nn.Module):
            def __init__(self, embed_dim, hidden_dim=128, stats_dim=0):
                super().__init__()
                self.stats_dim = stats_dim
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim + stats_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, 3),
                )
            def forward(self, q_embed, video_stats=None):
                if self.stats_dim > 0:
                    if video_stats is None:
                        video_stats = torch.zeros(
                            q_embed.shape[0], self.stats_dim,
                            device=q_embed.device, dtype=q_embed.dtype)
                    q_embed = torch.cat([q_embed, video_stats], dim=-1)
                return torch.nn.functional.softmax(self.net(q_embed), dim=-1)

        router = ExpertRouter(
            embed_dim=cfg.get("embed_dim", 2048),
            hidden_dim=cfg.get("router_hidden", 128),
            stats_dim=cfg.get("router_stats_dim", 0),
        )
        rt_path = os.path.join(checkpoint_dir, "router.pt")
        router.load_state_dict(
            torch.load(rt_path, map_location="cpu", weights_only=True))
        router = router.to(device=device, dtype=dtype).eval()

        s_params = sum(p.numel() for p in proj_lora.parameters()) + \
                   sum(p.numel() for p in merger.parameters())
        t_params = sum(p.numel() for p in temporal_branch.parameters())
        r_params = sum(p.numel() for p in router.parameters())
        eval_logger.info(
            f"Loaded Design A {module_type} (Two-Expert DeltaNet): "
            f"Expert S={s_params:,}, Expert T={t_params:,}, Router={r_params:,}")
        return "two_expert", (proj_lora, merger, temporal_branch, router), cfg

    elif module_type == "dual_proj_deltanet_v89":
        # Design A v8.9: Frozen temporal expert + QuestionMixer
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(checkpoint_dir).resolve().parent.parent.parent))
        from core.vision_projector.dual_proj import ProjectorLoRA, DeltaNetMerger

        proj_lora = ProjectorLoRA(
            base_proj,
            rank=cfg.get("proj_lora_rank", 32),
            alpha=cfg.get("proj_lora_rank", 32) * 2.0,
        )
        merger = DeltaNetMerger(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("merger_bottleneck", 128),
            max_alpha=cfg.get("max_alpha", 0.15),
            alpha_mode=cfg.get("alpha_mode", "scaled"),
            use_enrich_norm=cfg.get("use_enrich_norm", False),
        )

        proj_lora.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "proj_lora.pt"),
                        map_location="cpu", weights_only=True))
        proj_lora = proj_lora.to(device=device, dtype=dtype).eval()

        merger.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "merger.pt"),
                        map_location="cpu", weights_only=True))
        merger = merger.to(device=device, dtype=dtype).eval()

        class QuestionMixer(torch.nn.Module):
            def __init__(self, embed_dim, hidden_dim=128, stats_dim=4):
                super().__init__()
                self.stats_dim = stats_dim
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim + stats_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
            def forward(self, q_embed, video_stats=None):
                if self.stats_dim > 0 and video_stats is not None:
                    x = torch.cat([q_embed, video_stats], dim=-1)
                else:
                    x = q_embed
                return torch.sigmoid(self.net(x)).squeeze(-1)

        mixer = QuestionMixer(
            embed_dim=cfg.get("embed_dim", 2048),
            hidden_dim=cfg.get("mixer_hidden", 128),
            stats_dim=cfg.get("stats_dim", 4),
        )
        mixer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "mixer.pt"),
                        map_location="cpu", weights_only=True))
        mixer = mixer.to(device=device, dtype=dtype).eval()

        m_params = sum(p.numel() for p in mixer.parameters())
        eval_logger.info(
            f"Loaded Design A v8.9 (Frozen Expert + Mixer): mixer={m_params:,}")
        return "temporal_mixer", (proj_lora, merger, mixer), cfg

    elif module_type == "dual_proj_deltanet_v11_stage1":
        from core.vision_projector.dual_proj import HiddenTemporalBranch

        temporal_branch = HiddenTemporalBranch(
            hidden_dim=cfg.get("hidden_dim", base_proj.linear_2.in_features),
            bottleneck=cfg.get("hidden_bottleneck", 128),
            max_gate=cfg.get("max_gate", 0.15),
        )
        temporal_branch.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "temporal_branch.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        temporal_branch = temporal_branch.to(device=device, dtype=dtype).eval()
        eval_logger.info("Loaded Design A v11 stage1 (safe hidden temporal corrector)")
        return "v11_stage1", (temporal_branch,), cfg

    elif module_type == "dual_proj_deltanet_v11":
        from core.vision_projector.dual_proj import HiddenTemporalBranch

        class BinaryExpertRouter(torch.nn.Module):
            def __init__(self, embed_dim, hidden_dim=128, stats_dim=4):
                super().__init__()
                self.stats_dim = stats_dim
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim + stats_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, 2),
                )

            def forward(self, q_embed, video_stats=None):
                if self.stats_dim > 0:
                    if video_stats is None:
                        video_stats = torch.zeros(q_embed.shape[0], self.stats_dim, device=q_embed.device, dtype=q_embed.dtype)
                    x = torch.cat([q_embed, video_stats], dim=-1)
                else:
                    x = q_embed
                return torch.nn.functional.softmax(self.net(x), dim=-1)

        temporal_branch = HiddenTemporalBranch(
            hidden_dim=cfg.get("hidden_dim", base_proj.linear_2.in_features),
            bottleneck=cfg.get("hidden_bottleneck", 128),
            max_gate=cfg.get("max_gate", 0.15),
        )
        temporal_branch.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "temporal_branch.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        temporal_branch = temporal_branch.to(device=device, dtype=dtype).eval()

        router = BinaryExpertRouter(
            embed_dim=base_model.model.get_input_embeddings().weight.shape[1],
            hidden_dim=cfg.get("router_hidden", 128),
            stats_dim=cfg.get("router_stats_dim", 4),
        )
        router.load_state_dict(torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True))
        router = router.to(device=device, dtype=dtype).eval()
        eval_logger.info("Loaded Design A v11 (safe router)")
        return "v11_router", (temporal_branch, router), cfg

    elif module_type == "dual_proj_deltanet_v12":
        from core.vision_projector.dual_proj import ProjectorLoRA, DeltaNetMerger, BinaryExpertRouter

        proj_lora = ProjectorLoRA(
            base_proj,
            rank=cfg.get("proj_lora_rank", 32),
            alpha=cfg.get("proj_lora_rank", 32) * 2.0,
        )
        proj_lora.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "proj_lora.pt"), map_location="cpu", weights_only=True)
        )
        proj_lora = proj_lora.to(device=device, dtype=dtype).eval()

        merger = DeltaNetMerger(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("merger_bottleneck", 128),
            max_alpha=cfg.get("max_alpha", 0.15),
            alpha_mode=cfg.get("alpha_mode", "scaled"),
            use_enrich_norm=cfg.get("use_enrich_norm", False),
        )
        merger.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "merger.pt"), map_location="cpu", weights_only=True)
        )
        merger = merger.to(device=device, dtype=dtype).eval()

        router = BinaryExpertRouter(
            embed_dim=base_model.model.get_input_embeddings().weight.shape[1],
            hidden_dim=cfg.get("router_hidden", 128),
            stats_dim=cfg.get("router_stats_dim", 4),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info("Loaded Design A v12 (base vs safe-temp router)")
        return "v12_router", (proj_lora, merger, router), cfg

    elif module_type == "dual_proj_tmem_v1_stage1":
        from core.vision_projector.dual_proj import FactorizedTemporalMemoryExpert

        expert = FactorizedTemporalMemoryExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", 128),
            corr_bottleneck=cfg.get("corr_bottleneck", 128),
            state_slots=cfg.get("state_slots", 16),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_motion_gate=cfg.get("max_motion_gate", 0.20),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            max_corr_gate=cfg.get("max_corr_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "factorized_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()
        eval_logger.info("Loaded TMem v1 stage1 (factorized temporal memories)")
        return "factorized_temporal_memory_stage1", (expert,), cfg

    elif module_type == "dual_proj_tmem_v1_readout":
        from core.vision_projector.dual_proj import FactorizedTemporalMemoryExpert, QueryConditionedMemoryReadout

        expert = FactorizedTemporalMemoryExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", 128),
            corr_bottleneck=cfg.get("corr_bottleneck", 128),
            state_slots=cfg.get("state_slots", 16),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_motion_gate=cfg.get("max_motion_gate", 0.20),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            max_corr_gate=cfg.get("max_corr_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "factorized_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        readout = QueryConditionedMemoryReadout(
            embed_dim=base_model.model.get_input_embeddings().weight.shape[1],
            hidden_dim=cfg.get("readout_hidden", 128),
            stats_dim=cfg.get("readout_stats_dim", 6),
            num_slots=4,
        )
        readout.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "readout.pt"), map_location="cpu", weights_only=True)
        )
        readout = readout.to(device=device, dtype=dtype).eval()
        eval_logger.info("Loaded TMem v1 stage2 (query-conditioned memory readout)")
        return "factorized_temporal_memory_readout", (expert, readout), cfg

    elif module_type == "dual_proj_deltanet_fx_v1_stage1":
        from core.vision_projector.dual_proj import FactorizedGatedDeltaExpert

        expert = FactorizedGatedDeltaExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", 128),
            struct_bottleneck=cfg.get("struct_bottleneck", 128),
            state_slots=cfg.get("state_slots", 16),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_motion_gate=cfg.get("max_motion_gate", 0.20),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            max_struct_gate=cfg.get("max_struct_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "factorized_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()
        eval_logger.info("Loaded FX v1 stage1 (factorized gated deltas)")
        return "factorized_gated_delta_stage1", (expert,), cfg

    elif module_type == "dual_proj_deltanet_fx_v1":
        from core.vision_projector.dual_proj import FactorizedGatedDeltaExpert, MultiExpertUsageRouter

        expert = FactorizedGatedDeltaExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", 128),
            struct_bottleneck=cfg.get("struct_bottleneck", 128),
            state_slots=cfg.get("state_slots", 16),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_motion_gate=cfg.get("max_motion_gate", 0.20),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            max_struct_gate=cfg.get("max_struct_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "factorized_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        router = MultiExpertUsageRouter(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=9,
            hidden_dim=cfg.get("router_hidden", 128),
            num_experts=4,
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()
        eval_logger.info("Loaded FX v1 stage2 (multi-expert usage policy)")
        return "factorized_gated_delta_router", (expert, router), cfg

    elif module_type in ("dual_proj_deltanet_v922_gate_mod", "dual_proj_deltanet_v923_gate_mod", "dual_proj_deltanet_v924_gate_mod"):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            TwoTowerExpertGateModulator,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for gate_mod checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert_path = os.path.join(checkpoint_dir, "diff_expert.pt")
        if not os.path.exists(expert_path):
            expert_path = os.path.join(cfg.get("stage1_ckpt", ""), "diff_expert.pt")
        expert.load_state_dict(
            torch.load(expert_path, map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        modulator = TwoTowerExpertGateModulator(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 9),
            num_slots=cfg.get("state_slots", 4),
            hidden_dim=cfg.get("mod_hidden", 128),
            max_local_bias=cfg.get("max_local_bias", 0.75),
            max_state_bias=cfg.get("max_state_bias", 0.75),
            max_slot_bias=cfg.get("max_slot_bias", 0.75),
        )
        modulator.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "modulator.pt"), map_location="cpu", weights_only=True)
        )
        modulator = modulator.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded question-conditioned gate modulation stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_gate_mod", (expert, modulator), cfg

    elif module_type in ("dual_proj_deltanet_v922_rescomp", "dual_proj_deltanet_v923_rescomp", "dual_proj_deltanet_v924_rescomp"):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            TwoTowerResidualComposer,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for rescomp checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert_path = os.path.join(checkpoint_dir, "diff_expert.pt")
        if not os.path.exists(expert_path):
            expert_path = os.path.join(cfg.get("stage1_ckpt", ""), "diff_expert.pt")
        expert.load_state_dict(
            torch.load(expert_path, map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        composer = TwoTowerResidualComposer(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 9),
            num_slots=cfg.get("state_slots", 4),
            hidden_dim=cfg.get("comp_hidden", 128),
            max_local_delta=cfg.get("max_local_delta", 0.75),
            max_state_delta=cfg.get("max_state_delta", 0.75),
            max_slot_bias=cfg.get("max_slot_bias", 0.75),
        )
        composer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "composer.pt"), map_location="cpu", weights_only=True)
        )
        composer = composer.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded temporal residual composer stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_rescomp", (expert, composer), cfg

    elif module_type in ("dual_proj_deltanet_v922_resdistill", "dual_proj_deltanet_v923_resdistill", "dual_proj_deltanet_v924_resdistill"):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            TwoTowerResidualUsageComposer,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for resdistill checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        composer = TwoTowerResidualUsageComposer(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 9),
            hidden_dim=cfg.get("comp_hidden", 128),
        )
        composer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "composer.pt"), map_location="cpu", weights_only=True)
        )
        composer = composer.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded temporal residual distill stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_resdistill", (expert, composer), cfg

    elif module_type in ("dual_proj_deltanet_v922_mix4", "dual_proj_deltanet_v923_mix4", "dual_proj_deltanet_v924_mix4"):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            TwoTowerResidualSimplexMixer,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for mix4 checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        mixer = TwoTowerResidualSimplexMixer(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 9),
            hidden_dim=cfg.get("mix_hidden", 128),
            num_candidates=4,
        )
        mixer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "mixer.pt"), map_location="cpu", weights_only=True)
        )
        mixer = mixer.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded temporal 4-way simplex mixer stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_mix4", (expert, mixer), cfg

    elif module_type in ("dual_proj_deltanet_v926_srouter3", "dual_proj_deltanet_v927_srouter3", "dual_proj_deltanet_v928_srouter3"):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            MultiExpertUsageRouter,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for srouter3 checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        router = MultiExpertUsageRouter(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 9),
            hidden_dim=cfg.get("router_hidden", 128),
            num_experts=3,
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded supervised simplex router stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_srouter3", (expert, router), cfg

    elif module_type in ("dual_proj_deltanet_v926_hrouter3", "dual_proj_deltanet_v927_hrouter3", "dual_proj_deltanet_v928_hrouter3"):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            HierarchicalResidualRouter,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for hrouter3 checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        router = HierarchicalResidualRouter(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 9),
            hidden_dim=cfg.get("router_hidden", 128),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded hierarchical router stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_hrouter3", (expert, router), cfg

    elif module_type in (
        "dual_proj_deltanet_v926_bosrouter3",
        "dual_proj_deltanet_v927_bosrouter3",
        "dual_proj_deltanet_v928_bosrouter3",
        "dual_proj_deltanet_udh_v1_bosrouter3",
    ):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            BranchOutputSimplexRouter,
            UnifiedDualHeadExpert,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1", "dual_proj_deltanet_udh_v1_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for bosrouter3 checkpoint: {stage1_module_type!r}")

        if stage1_module_type == "dual_proj_deltanet_udh_v1_stage1":
            expert = UnifiedDualHeadExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_local_gate=cfg.get("max_local_gate", 0.35),
                max_state_gate=cfg.get("max_state_gate", 0.35),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        else:
            expert = StateAwareDiffDeltaTemporalExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                state_slots=cfg.get("state_slots", 1),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                max_state_gate=cfg.get("max_state_gate", 0.15),
                pooling_ratio=cfg.get("pooling_ratio", 2),
                state_from_prepool=cfg.get("state_from_prepool", False),
                local_short_window=cfg.get("local_short_window", False),
                separate_local_state=cfg.get("separate_local_state", False),
            )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        router = BranchOutputSimplexRouter(
            token_dim=base_model.model.get_input_embeddings().weight.shape[1],
            branch_dim=cfg.get("proj_dim", 2048),
            hidden_dim=cfg.get("router_hidden", 128),
            num_heads=cfg.get("router_heads", 4),
            init_probs=cfg.get("init_probs", [0.50, 0.30, 0.20]),
            max_logit_delta=cfg.get("max_logit_delta", 1.0),
            head_init_std=cfg.get("head_init_std", 1e-3),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded branch-output simplex router stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_bosrouter3", (expert, router), cfg

    elif module_type == "dual_proj_deltanet_udh_v2_res2":
        from core.vision_projector.dual_proj import (
            SplitDualHeadExpert,
            TwoTowerDualResidualCalibrator,
            GatedAttentionCalibrator,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type != "dual_proj_deltanet_udh_v2_stage1":
            raise ValueError(f"udh_v2_res2 expects udh_v2 stage1, got {stage1_module_type!r}")

        expert = SplitDualHeadExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_local_gate=cfg.get("max_local_gate", 0.35),
            max_state_gate=cfg.get("max_state_gate", 0.35),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            ortho_proj=cfg.get("ortho_proj", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        q_dim = base_model.model.get_input_embeddings().weight.shape[1]
        cal_kwargs = dict(
            q_dim=q_dim,
            video_dim=cfg.get("video_dim", cfg.get("proj_dim", 2048)),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_heads=cfg.get("num_heads", 4),
            init_local=cfg.get("init_local", 1.0),
            init_state=cfg.get("init_state", 1.0),
            max_local_delta=cfg.get("max_local_delta", 0.35),
            max_state_delta=cfg.get("max_state_delta", 0.35),
        )
        calibrator_type = cfg.get("calibrator_type", "mlp")
        if calibrator_type == "gated_attn":
            calibrator = GatedAttentionCalibrator(**cal_kwargs)
        else:
            calibrator = TwoTowerDualResidualCalibrator(**cal_kwargs)
        calibrator.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "calibrator.pt"), map_location="cpu", weights_only=True)
        )
        calibrator = calibrator.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded UDH v2 dual residual calibrator ({calibrator_type}) stage2"
        )
        return "udh_v2_res2", (expert, calibrator), cfg

    elif module_type in (
        "dual_proj_deltanet_v926_dbrouter",
        "dual_proj_deltanet_v927_dbrouter",
        "dual_proj_deltanet_v928_dbrouter",
        "dual_proj_deltanet_udh_v2_dbrouter",
    ):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            SplitDualHeadExpert,
            DualBranchRouter,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type == "dual_proj_deltanet_udh_v2_stage1":
            expert = SplitDualHeadExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_local_gate=cfg.get("max_local_gate", 0.35),
                max_state_gate=cfg.get("max_state_gate", 0.35),
                pooling_ratio=cfg.get("pooling_ratio", 2),
                ortho_proj=cfg.get("ortho_proj", False),
            )
        else:
            expert = StateAwareDiffDeltaTemporalExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                state_slots=cfg.get("state_slots", 1),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                max_state_gate=cfg.get("max_state_gate", 0.15),
                pooling_ratio=cfg.get("pooling_ratio", 2),
                state_from_prepool=cfg.get("state_from_prepool", False),
                local_short_window=cfg.get("local_short_window", False),
                separate_local_state=cfg.get("separate_local_state", False),
            )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        q_dim = base_model.model.get_input_embeddings().weight.shape[1]
        router = DualBranchRouter(
            embed_dim=q_dim,
            hidden_dim=cfg.get("router_hidden", 128),
            stats_dim=cfg.get("router_stats_dim", 4),
            init_local=cfg.get("init_local", 0.5),
            init_state=cfg.get("init_state", 0.5),
            w_min=cfg.get("w_min", 0.0),
            w_max=cfg.get("w_max", 1.0),
            init_std=cfg.get("init_std", 0.01),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded dual-branch router stage2 ({stage1_module_type} -> {module_type})"
        )
        return "dbrouter", (expert, router), cfg

    elif module_type == "dual_proj_deltanet_udh_v2_gate_finetune":
        from core.vision_projector.dual_proj import SplitDualHeadExpert
        from scripts.train_gate_finetune_stage2 import ContextAwareGate

        expert = SplitDualHeadExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_local_gate=cfg.get("max_local_gate", 0.35),
            max_state_gate=cfg.get("max_state_gate", 0.35),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            ortho_proj=cfg.get("ortho_proj", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False)
        expert = expert.to(device=device, dtype=dtype).eval()

        bottleneck = cfg.get("expert_bottleneck", 128)
        proj_dim = cfg.get("proj_dim", 2048)
        local_gate_new = ContextAwareGate(bottleneck, proj_dim)
        state_gate_new = ContextAwareGate(bottleneck, proj_dim)
        gate_state = torch.load(os.path.join(checkpoint_dir, "gate_finetune.pt"), map_location="cpu", weights_only=True)
        local_gate_new.load_state_dict(gate_state["local_gate"])
        state_gate_new.load_state_dict(gate_state["state_gate"])
        local_gate_new = local_gate_new.to(device=device, dtype=dtype).eval()
        state_gate_new = state_gate_new.to(device=device, dtype=dtype).eval()

        eval_logger.info("Loaded gate-finetune stage2")
        return "gate_finetune", (expert, local_gate_new, state_gate_new), cfg

    elif module_type in ("udh_v2_lora_sft", "v926_lora_sft", "v927_lora_sft", "v928_lora_sft"):
        from core.vision_projector.dual_proj import (
            SplitDualHeadExpert,
            StateAwareDiffDeltaTemporalExpert,
        )
        from core.vision_projector.hier_module import apply_lora_to_llm

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type == "dual_proj_deltanet_udh_v2_stage1":
            expert = SplitDualHeadExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_local_gate=cfg.get("max_local_gate", 0.35),
                max_state_gate=cfg.get("max_state_gate", 0.35),
                pooling_ratio=cfg.get("pooling_ratio", 2),
                ortho_proj=cfg.get("ortho_proj", False),
            )
        else:
            expert = StateAwareDiffDeltaTemporalExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                state_slots=cfg.get("state_slots", 1),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                max_state_gate=cfg.get("max_state_gate", 0.15),
                pooling_ratio=cfg.get("pooling_ratio", 2),
                state_from_prepool=cfg.get("state_from_prepool", False),
                local_short_window=cfg.get("local_short_window", False),
                separate_local_state=cfg.get("separate_local_state", False),
            )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        # Apply LoRA to LLM
        lora_rank = cfg.get("lora_rank", 16)
        lora_alpha = cfg.get("lora_alpha", 32.0)
        lora_params, n_replaced = apply_lora_to_llm(
            base_model.model.language_model,
            rank=lora_rank, alpha=lora_alpha,
            target_modules=tuple(cfg.get("lora_target_modules", ["q_proj", "v_proj"])))
        # Move LoRA params to device first, then load weights
        for p in lora_params:
            p.data = p.data.to(device=device, dtype=dtype)
        lora_state = torch.load(os.path.join(checkpoint_dir, "lora.pt"), map_location="cpu", weights_only=True)
        for i, p in enumerate(lora_params):
            key = f"lora_{i}"
            if key in lora_state:
                p.data.copy_(lora_state[key].to(device=device, dtype=dtype))

        eval_logger.info(
            f"Loaded LoRA SFT stage2 ({stage1_module_type}): {n_replaced} modules, rank={lora_rank}"
        )
        return "lora_sft_expert", (expert,), cfg

    elif module_type in (
        "dual_proj_deltanet_v926_ctxsrouter3",
        "dual_proj_deltanet_v927_ctxsrouter3",
        "dual_proj_deltanet_v928_ctxsrouter3",
        "dual_proj_deltanet_v926_ctxsrouter3sup",
        "dual_proj_deltanet_v927_ctxsrouter3sup",
        "dual_proj_deltanet_v928_ctxsrouter3sup",
        "dual_proj_deltanet_v926_ctxsrouter3joint",
        "dual_proj_deltanet_v927_ctxsrouter3joint",
        "dual_proj_deltanet_v928_ctxsrouter3joint",
    ):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            RawContextSimplexRouter,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for ctxsrouter3 checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        router = RawContextSimplexRouter(
            token_dim=base_model.model.get_input_embeddings().weight.shape[1],
            hidden_dim=cfg.get("router_hidden", 128),
            num_heads=cfg.get("router_heads", 4),
            init_probs=cfg.get("init_probs", [0.20, 0.40, 0.40]),
            max_logit_delta=cfg.get("max_logit_delta", 1.0),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded raw-context simplex router stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_ctxsrouter3", (expert, router), cfg

    elif module_type in (
        "dual_proj_deltanet_v926_ctxhrouter3",
        "dual_proj_deltanet_v927_ctxhrouter3",
        "dual_proj_deltanet_v928_ctxhrouter3",
        "dual_proj_deltanet_v926_ctxhrouter3sup",
        "dual_proj_deltanet_v927_ctxhrouter3sup",
        "dual_proj_deltanet_v928_ctxhrouter3sup",
        "dual_proj_deltanet_v926_ctxhrouter3joint",
        "dual_proj_deltanet_v927_ctxhrouter3joint",
        "dual_proj_deltanet_v928_ctxhrouter3joint",
    ):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            RawContextHierarchicalRouter,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for ctxhrouter3 checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        router = RawContextHierarchicalRouter(
            token_dim=base_model.model.get_input_embeddings().weight.shape[1],
            hidden_dim=cfg.get("router_hidden", 128),
            num_heads=cfg.get("router_heads", 4),
            max_logit_delta=cfg.get("max_logit_delta", 1.0),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded raw-context hierarchical router stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_ctxhrouter3", (expert, router), cfg

    elif module_type in (
        "dual_proj_deltanet_v926_ctxres2",
        "dual_proj_deltanet_v927_ctxres2",
        "dual_proj_deltanet_v928_ctxres2",
    ):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            RawContextResidualScaler,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for ctxres2 checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        scaler = RawContextResidualScaler(
            token_dim=base_model.model.get_input_embeddings().weight.shape[1],
            hidden_dim=cfg.get("router_hidden", 128),
            num_heads=cfg.get("router_heads", 4),
            init_local=cfg.get("init_local", 1.0),
            init_state=cfg.get("init_state", 1.0),
            max_local_delta=cfg.get("max_local_delta", 0.35),
            max_state_delta=cfg.get("max_state_delta", 0.35),
            mean_preserve=cfg.get("mean_preserve_scales", True),
            center_momentum=cfg.get("center_momentum", 0.01),
            fuse_gain=cfg.get("fuse_gain", 40.0),
        )
        scaler.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "scaler.pt"), map_location="cpu", weights_only=True)
        )
        scaler = scaler.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded raw-context residual scaler stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_ctxres2", (expert, scaler), cfg

    elif module_type in (
        "dual_proj_deltanet_v926_lspolicy",
        "dual_proj_deltanet_v927_lspolicy",
        "dual_proj_deltanet_v928_lspolicy",
        "dual_proj_deltanet_v926_lspolicy_v2",
        "dual_proj_deltanet_v927_lspolicy_v2",
        "dual_proj_deltanet_v928_lspolicy_v2",
        "dual_proj_deltanet_v926_lspolicy_v3",
        "dual_proj_deltanet_v927_lspolicy_v3",
        "dual_proj_deltanet_v928_lspolicy_v3",
    ):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            RawContextResidualScaler,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for lspolicy checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        scaler = RawContextResidualScaler(
            token_dim=base_model.model.get_input_embeddings().weight.shape[1],
            hidden_dim=cfg.get("hidden_dim", 128),
            num_heads=cfg.get("num_heads", 4),
            init_local=cfg.get("init_local", 0.20),
            init_state=cfg.get("init_state", 0.20),
            max_local_delta=cfg.get("max_local_delta", 0.20),
            max_state_delta=cfg.get("max_state_delta", 0.20),
            mean_preserve=False,
            head_init_std=cfg.get("head_init_std", 0.02),
            fuse_gain=cfg.get("fuse_gain", 40.0),
        )
        scaler.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "policy_scaler.pt"), map_location="cpu", weights_only=True)
        )
        scaler = scaler.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded local/state dual policy stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_lspolicy", (expert, scaler), cfg

    elif module_type in ("dual_proj_deltanet_v922_mix3", "dual_proj_deltanet_v923_mix3", "dual_proj_deltanet_v924_mix3", "dual_proj_deltanet_v925_mix3", "dual_proj_deltanet_v926_mix3"):
        from core.vision_projector.dual_proj import (
            StateAwareDiffDeltaTemporalExpert,
            TwoTowerResidualSimplexMixer,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for mix3 checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        mixer = TwoTowerResidualSimplexMixer(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 9),
            hidden_dim=cfg.get("mix_hidden", 128),
            num_candidates=3,
            init_probs=cfg.get("mix_init_probs", [0.20, 0.40, 0.40]),
        )
        mixer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "mixer.pt"), map_location="cpu", weights_only=True)
        )
        mixer = mixer.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded temporal 3-way simplex mixer stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_mix3", (expert, mixer), cfg

    elif module_type in ("dual_proj_deltanet_v922_g2s", "dual_proj_deltanet_v923_g2s", "dual_proj_deltanet_v924_g2s", "dual_proj_deltanet_v925_g2s", "dual_proj_deltanet_v926_g2s"):
        from core.vision_projector.dual_proj import (
            GlobalResidualScaler,
            StateAwareDiffDeltaTemporalExpert,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for g2s checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        scaler = GlobalResidualScaler()
        scaler.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "scaler.pt"), map_location="cpu", weights_only=True)
        )
        scaler = scaler.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded global 2-scalar residual stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_g2s", (expert, scaler), cfg
    elif module_type in ("dual_proj_deltanet_v922_qg2s", "dual_proj_deltanet_v923_qg2s", "dual_proj_deltanet_v924_qg2s", "dual_proj_deltanet_v925_qg2s", "dual_proj_deltanet_v926_qg2s"):
        from core.vision_projector.dual_proj import (
            QueryResidualScaler,
            StateAwareDiffDeltaTemporalExpert,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
            raise ValueError(f"Unsupported stage1_module_type for qg2s checkpoint: {stage1_module_type!r}")

        expert = StateAwareDiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
            separate_local_state=cfg.get("separate_local_state", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        scaler = QueryResidualScaler(q_dim=cfg.get("proj_dim", 2048))
        scaler.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "scaler.pt"), map_location="cpu", weights_only=True)
        )
        scaler = scaler.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded question-conditioned 2-scalar residual stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_qg2s", (expert, scaler), cfg

    elif module_type in ("dual_proj_deltanet_v810_router_v2", "dual_proj_deltanet_v92_router_v2", "dual_proj_deltanet_v921_router_v2", "dual_proj_deltanet_v922_router_v2", "dual_proj_deltanet_v923_router_v2", "dual_proj_deltanet_v924_router_v2", "dual_proj_deltanet_v10_router_v2"):
        from core.vision_projector.dual_proj import (
            DiffDeltaTemporalExpert,
            EventFocusedTemporalExpert,
            StateAwareDiffDeltaTemporalExpert,
            TwoTowerBinaryExpertRouter,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type == "dual_proj_deltanet_v810_stage1":
            expert = DiffDeltaTemporalExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        elif stage1_module_type in ("dual_proj_deltanet_v92_stage1", "dual_proj_deltanet_v921_stage1", "dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v10_stage1"):
            expert_cls = EventFocusedTemporalExpert if stage1_module_type == "dual_proj_deltanet_v10_stage1" else StateAwareDiffDeltaTemporalExpert
            expert = expert_cls(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                state_slots=cfg.get("state_slots", 1),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                max_state_gate=cfg.get("max_state_gate", 0.15),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        else:
            raise ValueError(f"Unsupported stage1_module_type for router_v2 checkpoint: {stage1_module_type!r}")

        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        router = TwoTowerBinaryExpertRouter(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 9),
            hidden_dim=cfg.get("router_hidden", 128),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded slot-aware two-tower router stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_router_v2", (expert, router), cfg

    elif module_type in ("dual_proj_deltanet_v92_gattn", "dual_proj_deltanet_v921_gattn", "dual_proj_deltanet_v922_gattn", "dual_proj_deltanet_v923_gattn", "dual_proj_deltanet_v924_gattn", "dual_proj_deltanet_v10_gattn"):
        from core.vision_projector.dual_proj import (
            EventFocusedTemporalExpert,
            QueryConditionedSlotTemporalGate,
            StateAwareDiffDeltaTemporalExpert,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type in ("dual_proj_deltanet_v92_stage1", "dual_proj_deltanet_v921_stage1", "dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v10_stage1"):
            expert_cls = EventFocusedTemporalExpert if stage1_module_type == "dual_proj_deltanet_v10_stage1" else StateAwareDiffDeltaTemporalExpert
            expert = expert_cls(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                state_slots=cfg.get("state_slots", 1),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                max_state_gate=cfg.get("max_state_gate", 0.15),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        else:
            raise ValueError(f"Unsupported stage1_module_type for gattn checkpoint: {stage1_module_type!r}")

        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        gate_module = QueryConditionedSlotTemporalGate(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            evidence_dim=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)) + 3,
            hidden_dim=cfg.get("gate_hidden", 128),
            init_alpha=cfg.get("init_alpha", 0.75),
        )
        gate_module.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "gattn.pt"), map_location="cpu", weights_only=True)
        )
        gate_module = gate_module.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded slot/time gated-attention temporal expert stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_gattn", (expert, gate_module), cfg

    elif module_type in ("dual_proj_deltanet_v810_alpha_v2", "dual_proj_deltanet_v92_alpha_v2", "dual_proj_deltanet_v921_alpha_v2", "dual_proj_deltanet_v922_alpha_v2", "dual_proj_deltanet_v923_alpha_v2", "dual_proj_deltanet_v924_alpha_v2", "dual_proj_deltanet_v925_alpha_v2", "dual_proj_deltanet_v926_alpha_v2", "dual_proj_deltanet_v927_alpha_v2", "dual_proj_deltanet_v928_alpha_v2", "dual_proj_deltanet_v10_alpha_v2"):
        from core.vision_projector.dual_proj import (
            DiffDeltaTemporalExpert,
            EventFocusedTemporalExpert,
            StateAwareDiffDeltaTemporalExpert,
            TwoTowerResidualAlphaCalibrator,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type == "dual_proj_deltanet_v810_stage1":
            expert = DiffDeltaTemporalExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        elif stage1_module_type in ("dual_proj_deltanet_v92_stage1", "dual_proj_deltanet_v921_stage1", "dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1", "dual_proj_deltanet_v10_stage1"):
            expert_cls = EventFocusedTemporalExpert if stage1_module_type == "dual_proj_deltanet_v10_stage1" else StateAwareDiffDeltaTemporalExpert
            expert_kwargs = {
                "proj_dim": cfg.get("proj_dim", 2048),
                "bottleneck": cfg.get("expert_bottleneck", 128),
                "state_bottleneck": cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                "state_slots": cfg.get("state_slots", 1),
                "kernel_size": cfg.get("expert_kernel_size", 3),
                "max_enrich_gate": cfg.get("max_enrich_gate", 0.25),
                "max_state_gate": cfg.get("max_state_gate", 0.15),
                "pooling_ratio": cfg.get("pooling_ratio", 2),
            }
            if stage1_module_type != "dual_proj_deltanet_v10_stage1":
                expert_kwargs.update(
                    {
                        "state_from_prepool": cfg.get("state_from_prepool", False),
                        "local_short_window": cfg.get("local_short_window", False),
                        "separate_local_state": cfg.get("separate_local_state", False),
                    }
                )
            expert = expert_cls(**expert_kwargs)
        else:
            raise ValueError(f"Unsupported stage1_module_type for alpha_v2 checkpoint: {stage1_module_type!r}")

        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        alpha_calibrator = TwoTowerResidualAlphaCalibrator(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 9),
            hidden_dim=cfg.get("alpha_hidden", 128),
            alpha_prior=cfg.get("alpha_prior", 0.75),
            max_delta=cfg.get("alpha_max_delta", 0.20),
        )
        alpha_calibrator.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "alpha_calibrator.pt"), map_location="cpu", weights_only=True)
        )
        alpha_calibrator = alpha_calibrator.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded alpha-calibration v2 temporal expert stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_alpha_v2", (expert, alpha_calibrator), cfg

    elif module_type in (
        "dual_proj_deltanet_v810_policy",
        "dual_proj_deltanet_v92_policy",
        "dual_proj_deltanet_v921_policy",
        "dual_proj_deltanet_v922_policy",
        "dual_proj_deltanet_v923_policy",
        "dual_proj_deltanet_v924_policy",
        "dual_proj_deltanet_v925_policy",
        "dual_proj_deltanet_v926_policy",
        "dual_proj_deltanet_v927_policy",
        "dual_proj_deltanet_v928_policy",
        "dual_proj_deltanet_v10_policy",
    ):
        from core.vision_projector.dual_proj import (
            DiffDeltaTemporalExpert,
            EventFocusedTemporalExpert,
            StateAwareDiffDeltaTemporalExpert,
            TwoTowerResidualAlphaCalibrator,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type == "dual_proj_deltanet_v810_stage1":
            expert = DiffDeltaTemporalExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        elif stage1_module_type in ("dual_proj_deltanet_v92_stage1", "dual_proj_deltanet_v921_stage1", "dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1", "dual_proj_deltanet_v10_stage1"):
            expert_cls = EventFocusedTemporalExpert if stage1_module_type == "dual_proj_deltanet_v10_stage1" else StateAwareDiffDeltaTemporalExpert
            expert_kwargs = {
                "proj_dim": cfg.get("proj_dim", 2048),
                "bottleneck": cfg.get("expert_bottleneck", 128),
                "state_bottleneck": cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                "state_slots": cfg.get("state_slots", 1),
                "kernel_size": cfg.get("expert_kernel_size", 3),
                "max_enrich_gate": cfg.get("max_enrich_gate", 0.25),
                "max_state_gate": cfg.get("max_state_gate", 0.15),
                "pooling_ratio": cfg.get("pooling_ratio", 2),
            }
            if stage1_module_type != "dual_proj_deltanet_v10_stage1":
                expert_kwargs.update(
                    {
                        "state_from_prepool": cfg.get("state_from_prepool", False),
                        "local_short_window": cfg.get("local_short_window", False),
                        "separate_local_state": cfg.get("separate_local_state", False),
                    }
                )
            expert = expert_cls(**expert_kwargs)
        else:
            raise ValueError(f"Unsupported stage1_module_type for policy checkpoint: {stage1_module_type!r}")

        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        policy_calibrator = TwoTowerResidualAlphaCalibrator(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 6),
            hidden_dim=cfg.get("policy_hidden", 128),
            alpha_prior=cfg.get("policy_prior", 0.35),
            max_delta=cfg.get("policy_max_delta", 0.25),
        )
        policy_calibrator.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "policy_calibrator.pt"), map_location="cpu", weights_only=True)
        )
        policy_calibrator = policy_calibrator.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded intervention-policy temporal expert stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_policy", (expert, policy_calibrator), cfg

    elif module_type in ("dual_proj_deltanet_v810_alpha", "dual_proj_deltanet_v92_alpha", "dual_proj_deltanet_v921_alpha", "dual_proj_deltanet_v922_alpha", "dual_proj_deltanet_v923_alpha", "dual_proj_deltanet_v924_alpha", "dual_proj_deltanet_v925_alpha", "dual_proj_deltanet_v926_alpha", "dual_proj_deltanet_v927_alpha", "dual_proj_deltanet_v928_alpha", "dual_proj_deltanet_v10_alpha"):
        from core.vision_projector.dual_proj import (
            DiffDeltaTemporalExpert,
            EventFocusedTemporalExpert,
            StateAwareDiffDeltaTemporalExpert,
            TwoTowerTemporalAlphaCalibrator,
        )

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type == "dual_proj_deltanet_v810_stage1":
            expert = DiffDeltaTemporalExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        elif stage1_module_type in ("dual_proj_deltanet_v92_stage1", "dual_proj_deltanet_v921_stage1", "dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1", "dual_proj_deltanet_v10_stage1"):
            expert_cls = EventFocusedTemporalExpert if stage1_module_type == "dual_proj_deltanet_v10_stage1" else StateAwareDiffDeltaTemporalExpert
            expert_kwargs = {
                "proj_dim": cfg.get("proj_dim", 2048),
                "bottleneck": cfg.get("expert_bottleneck", 128),
                "state_bottleneck": cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                "state_slots": cfg.get("state_slots", 1),
                "kernel_size": cfg.get("expert_kernel_size", 3),
                "max_enrich_gate": cfg.get("max_enrich_gate", 0.25),
                "max_state_gate": cfg.get("max_state_gate", 0.15),
                "pooling_ratio": cfg.get("pooling_ratio", 2),
            }
            if stage1_module_type != "dual_proj_deltanet_v10_stage1":
                expert_kwargs.update(
                    {
                        "state_from_prepool": cfg.get("state_from_prepool", False),
                        "local_short_window": cfg.get("local_short_window", False),
                        "separate_local_state": cfg.get("separate_local_state", False),
                    }
                )
            expert = expert_cls(**expert_kwargs)
        else:
            raise ValueError(f"Unsupported stage1_module_type for alpha checkpoint: {stage1_module_type!r}")

        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        alpha_calibrator = TwoTowerTemporalAlphaCalibrator(
            q_dim=base_model.model.get_input_embeddings().weight.shape[1],
            video_dim=cfg.get("video_feat_dim", 6),
            hidden_dim=cfg.get("alpha_hidden", 128),
            init_alpha=cfg.get("init_alpha", 0.70),
        )
        alpha_calibrator.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "alpha_calibrator.pt"), map_location="cpu", weights_only=True)
        )
        alpha_calibrator = alpha_calibrator.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded alpha-calibration temporal expert stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_alpha", (expert, alpha_calibrator), cfg

    elif module_type in ("dual_proj_deltanet_v810_router", "dual_proj_deltanet_v92_router", "dual_proj_deltanet_v921_router", "dual_proj_deltanet_v922_router", "dual_proj_deltanet_v923_router", "dual_proj_deltanet_v924_router", "dual_proj_deltanet_v925_router", "dual_proj_deltanet_v926_router", "dual_proj_deltanet_v927_router", "dual_proj_deltanet_v928_router", "dual_proj_deltanet_v10_router", "dual_proj_deltanet_udh_v2_router"):
        from core.vision_projector.dual_proj import DiffDeltaTemporalExpert, EventFocusedTemporalExpert, StateAwareDiffDeltaTemporalExpert, SplitDualHeadExpert, BinaryExpertRouter

        stage1_module_type = cfg.get("stage1_module_type", "")
        if stage1_module_type == "dual_proj_deltanet_v810_stage1":
            expert = DiffDeltaTemporalExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        elif stage1_module_type == "dual_proj_deltanet_udh_v2_stage1":
            expert = SplitDualHeadExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_local_gate=cfg.get("max_local_gate", 0.35),
                max_state_gate=cfg.get("max_state_gate", 0.35),
                pooling_ratio=cfg.get("pooling_ratio", 2),
                ortho_proj=cfg.get("ortho_proj", False),
            )
        elif stage1_module_type in ("dual_proj_deltanet_v92_stage1", "dual_proj_deltanet_v921_stage1", "dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1", "dual_proj_deltanet_v10_stage1"):
            expert_cls = EventFocusedTemporalExpert if stage1_module_type == "dual_proj_deltanet_v10_stage1" else StateAwareDiffDeltaTemporalExpert
            expert = expert_cls(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                state_slots=cfg.get("state_slots", 1),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                max_state_gate=cfg.get("max_state_gate", 0.15),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        else:
            raise ValueError(f"Unsupported stage1_module_type for router checkpoint: {stage1_module_type!r}")

        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        router = BinaryExpertRouter(
            embed_dim=base_model.model.get_input_embeddings().weight.shape[1],
            hidden_dim=cfg.get("router_hidden", 128),
            stats_dim=cfg.get("router_stats_dim", 4),
        )
        router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "router.pt"), map_location="cpu", weights_only=True)
        )
        router = router.to(device=device, dtype=dtype).eval()

        eval_logger.info(
            f"Loaded router-only temporal expert stage2 ({stage1_module_type} -> {module_type})"
        )
        return "temporal_expert_router", (expert, router), cfg

    elif module_type in ("dual_proj_deltanet_v90_stage1", "dual_proj_deltanet_v91_stage1", "dual_proj_deltanet_v92_stage1", "dual_proj_deltanet_v921_stage1", "dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1", "dual_proj_deltanet_v93_stage1", "dual_proj_deltanet_v94_stage1", "dual_proj_deltanet_v10_stage1", "dual_proj_deltanet_udh_v1_stage1", "dual_proj_deltanet_udh_v2_stage1"):
        from core.vision_projector.dual_proj import StateAwareDiffDeltaTemporalExpert, EventFocusedTemporalExpert, UnifiedDualHeadExpert, SplitDualHeadExpert

        if module_type == "dual_proj_deltanet_udh_v2_stage1":
            expert = SplitDualHeadExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_local_gate=cfg.get("max_local_gate", 0.35),
                max_state_gate=cfg.get("max_state_gate", 0.35),
                pooling_ratio=cfg.get("pooling_ratio", 2),
                ortho_proj=cfg.get("ortho_proj", False),
            )
        elif module_type == "dual_proj_deltanet_udh_v1_stage1":
            expert = UnifiedDualHeadExpert(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_local_gate=cfg.get("max_local_gate", 0.35),
                max_state_gate=cfg.get("max_state_gate", 0.35),
                pooling_ratio=cfg.get("pooling_ratio", 2),
            )
        else:
            expert_cls = EventFocusedTemporalExpert if module_type == "dual_proj_deltanet_v10_stage1" else StateAwareDiffDeltaTemporalExpert
            expert = expert_cls(
                proj_dim=cfg.get("proj_dim", 2048),
                bottleneck=cfg.get("expert_bottleneck", 128),
                state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                state_slots=cfg.get("state_slots", 1),
                kernel_size=cfg.get("expert_kernel_size", 3),
                max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
                max_state_gate=cfg.get("max_state_gate", 0.15),
                pooling_ratio=cfg.get("pooling_ratio", 2),
                state_from_prepool=cfg.get("state_from_prepool", False),
                local_short_window=cfg.get("local_short_window", False),
                separate_local_state=cfg.get("separate_local_state", False),
            )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True), strict=False
        )
        expert = expert.to(device=device, dtype=dtype).eval()
        version_label = (
            "UDH v2" if module_type == "dual_proj_deltanet_udh_v2_stage1"
            else "UDH v1" if module_type == "dual_proj_deltanet_udh_v1_stage1"
            else
            "v10" if module_type == "dual_proj_deltanet_v10_stage1"
            else
            "v9.4" if module_type == "dual_proj_deltanet_v94_stage1"
            else "v9.3" if module_type == "dual_proj_deltanet_v93_stage1"
            else
            "v9.2.8" if module_type == "dual_proj_deltanet_v928_stage1" else
            "v9.2.7" if module_type == "dual_proj_deltanet_v927_stage1" else
            "v9.2.6" if module_type == "dual_proj_deltanet_v926_stage1" else
            "v9.2.5" if module_type == "dual_proj_deltanet_v925_stage1"
            else
            "v9.2.4" if module_type == "dual_proj_deltanet_v924_stage1"
            else "v9.2.3" if module_type == "dual_proj_deltanet_v923_stage1"
            else
            "v9.2.2" if module_type == "dual_proj_deltanet_v922_stage1"
            else
            "v9.2.1" if module_type == "dual_proj_deltanet_v921_stage1"
            else
            "v9.2" if module_type == "dual_proj_deltanet_v92_stage1"
            else "v9.1" if module_type == "dual_proj_deltanet_v91_stage1"
            else "v9.0"
        )
        expert_label = "split dual-head temporal expert" if module_type == "dual_proj_deltanet_udh_v2_stage1" else "unified dual-head temporal expert" if module_type == "dual_proj_deltanet_udh_v1_stage1" else "event-focused temporal specialist" if module_type == "dual_proj_deltanet_v10_stage1" else "state-aware diff delta expert"
        eval_logger.info(f"Loaded Design A {version_label} stage1 ({expert_label})")
        return "diff_delta_expert", (expert,), cfg

    elif module_type == "plm_udh_v2_stage2_tqrouter":
        # PLM Stage 2: SplitDualHeadExpert + TemporalQueryRouter
        from core.vision_projector.dual_proj import SplitDualHeadExpert
        from core.vision_projector.temporal_query_router import TemporalQueryRouter

        # Stage 1 config에서 expert 설정 가져옴
        s1_cfg = cfg.get("stage1_config", cfg)
        proj_dim = cfg.get("proj_dim", 2048)

        expert = SplitDualHeadExpert(
            proj_dim=proj_dim,
            bottleneck=s1_cfg.get("expert_bottleneck", 128),
            kernel_size=s1_cfg.get("expert_kernel_size", 3),
            max_local_gate=s1_cfg.get("max_local_gate", 0.35),
            max_state_gate=s1_cfg.get("max_state_gate", 0.35),
            pooling_ratio=s1_cfg.get("pooling_ratio", 2),
            ortho_proj=s1_cfg.get("ortho_proj", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True),
            strict=False,
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        tq_router = TemporalQueryRouter(
            feat_dim=proj_dim,
            hidden_dim=cfg.get("router_hidden", 256),
            init_bias=cfg.get("stage2_init_bias", 0.0),
        )
        tq_router.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "tq_router.pt"), map_location="cpu", weights_only=True)
        )
        tq_router = tq_router.to(device=device, dtype=dtype).eval()

        eval_logger.info(f"Loaded PLM UDH v2 Stage 2 (expert + TemporalQueryRouter)")
        return "plm_tq_router", (expert, tq_router), cfg

    elif module_type in ("dual_proj_deltanet_v90", "dual_proj_deltanet_v91", "dual_proj_deltanet_v92", "dual_proj_deltanet_v93", "dual_proj_deltanet_v94", "dual_proj_deltanet_v10"):
        from core.vision_projector.dual_proj import StateAwareDiffDeltaTemporalExpert, EventFocusedTemporalExpert

        expert_cls = EventFocusedTemporalExpert if module_type == "dual_proj_deltanet_v10" else StateAwareDiffDeltaTemporalExpert
        expert = expert_cls(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            state_bottleneck=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
            state_slots=cfg.get("state_slots", 1),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            max_state_gate=cfg.get("max_state_gate", 0.15),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True), strict=False
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        if cfg.get("stage2_mode", "") in ("temporal_delta_scale_v90", "temporal_delta_scale_v91", "temporal_delta_scale_v92", "temporal_delta_scale_v93", "temporal_delta_scale_v94", "temporal_delta_scale_v10"):
            class TemporalDeltaScale(torch.nn.Module):
                def __init__(self, proj_dim=2048, hidden_dim=128, init_alpha=0.15, max_offset=2.0):
                    super().__init__()
                    init_alpha = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
                    init_logit = math.log(init_alpha / (1.0 - init_alpha))
                    self.global_logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
                    self.max_offset = float(max_offset)
                    self.local_alpha_scale = float(local_alpha_scale)
                    self.hidden_dim = int(hidden_dim)
                    self.q_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.e_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.time_head = torch.nn.Linear(hidden_dim, 1)

                def forward(self, q_embed, enrich_pooled, return_offset=False):
                    with torch.autocast(device_type="cuda", enabled=False):
                        q_embed = q_embed.float()
                        enrich_pooled = enrich_pooled.float()
                        enrich_temporal = enrich_pooled.mean(dim=2)
                        q = self.q_proj(q_embed).unsqueeze(1)
                        e = self.e_proj(enrich_temporal)
                        h = torch.nn.functional.gelu(q + e)
                        offset_raw = torch.tanh(self.time_head(h).squeeze(-1)) * self.max_offset
                        offset = offset_raw - offset_raw.mean(dim=1, keepdim=True)
                        alpha = torch.sigmoid(self.global_logit + offset)
                    alpha = alpha.to(dtype=q_embed.dtype)
                    if return_offset:
                        return alpha, offset
                    return alpha

            class StateAwareTemporalDeltaScale(torch.nn.Module):
                def __init__(self, proj_dim=2048, hidden_dim=128, state_dim=256, init_alpha=0.10, max_offset=1.5, local_alpha_scale=0.5):
                    super().__init__()
                    init_alpha = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
                    init_logit = math.log(init_alpha / (1.0 - init_alpha))
                    self.global_logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
                    self.max_offset = float(max_offset)
                    self.local_alpha_scale = float(local_alpha_scale)
                    self.hidden_dim = int(hidden_dim)
                    self.state_dim = int(state_dim)
                    self.q_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.e_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.s_proj = torch.nn.Linear(self.state_dim, hidden_dim)
                    self.g_proj = torch.nn.Linear(1, hidden_dim)
                    self.mix = torch.nn.Linear(hidden_dim * 4, hidden_dim)
                    self.time_head = torch.nn.Linear(hidden_dim, 1)
                    self.state_norm = torch.nn.LayerNorm(self.state_dim)

                def forward(self, q_embed, enrich_pooled, state_h, event_gate, return_offset=False):
                    with torch.autocast(device_type="cuda", enabled=False):
                        q_embed = q_embed.float()
                        enrich_pooled = enrich_pooled.float()
                        state_h = state_h.float()
                        event_gate = event_gate.float()
                        enrich_temporal = enrich_pooled.mean(dim=2)
                        q = self.q_proj(q_embed).unsqueeze(1)
                        e = self.e_proj(enrich_temporal)
                        state_feat = self.state_norm(state_h)
                        s = self.s_proj(state_feat)
                        g = self.g_proj(event_gate)
                        h = torch.nn.functional.gelu(self.mix(torch.cat([q.expand_as(e), e, s, g], dim=-1)))
                        offset_raw = torch.tanh(self.time_head(h).squeeze(-1)) * self.max_offset
                        offset = offset_raw - offset_raw.mean(dim=1, keepdim=True)
                        alpha = torch.sigmoid(self.global_logit + offset)
                    alpha = alpha.to(dtype=q_embed.dtype)
                    if return_offset:
                        return alpha, offset
                    return alpha

            class TrustAwareTemporalDeltaScale(torch.nn.Module):
                def __init__(self, proj_dim=2048, hidden_dim=128, state_dim=256, init_alpha=0.10, max_offset=1.5, local_alpha_scale=0.5):
                    super().__init__()
                    init_alpha = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
                    init_logit = math.log(init_alpha / (1.0 - init_alpha))
                    self.global_logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
                    self.max_offset = float(max_offset)
                    self.local_alpha_scale = float(local_alpha_scale)
                    self.hidden_dim = int(hidden_dim)
                    self.state_dim = int(state_dim)
                    self.q_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.e_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.s_proj = torch.nn.Linear(self.state_dim, hidden_dim)
                    self.g_proj = torch.nn.Linear(1, hidden_dim)
                    self.mix = torch.nn.Linear(hidden_dim * 4, hidden_dim)
                    self.time_head = torch.nn.Linear(hidden_dim, 1)
                    self.state_norm = torch.nn.LayerNorm(self.state_dim)
                    self.gq_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.ge_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.gs_proj = torch.nn.Linear(self.state_dim, hidden_dim)
                    self.gg_proj = torch.nn.Linear(1, hidden_dim)
                    self.type_proj = torch.nn.Linear(3, hidden_dim)
                    self.span_proj = torch.nn.Linear(5, hidden_dim)
                    self.global_mix = torch.nn.Linear(hidden_dim * 6, hidden_dim)
                    self.trust_head = torch.nn.Linear(hidden_dim, 1)
                    self.type_bias = torch.nn.Parameter(torch.tensor([-2.00, -0.60, -1.20], dtype=torch.float32))

                def forward(self, q_embed, enrich_pooled, state_h, event_gate, q_type_feats, span_feats, return_offset=False):
                    with torch.autocast(device_type="cuda", enabled=False):
                        q_embed = q_embed.float()
                        enrich_pooled = enrich_pooled.float()
                        state_h = state_h.float()
                        event_gate = event_gate.float()
                        q_type_feats = q_type_feats.float()
                        span_feats = span_feats.float()
                        enrich_temporal = enrich_pooled.mean(dim=2)
                        q = self.q_proj(q_embed).unsqueeze(1)
                        e = self.e_proj(enrich_temporal)
                        state_feat = self.state_norm(state_h)
                        s = self.s_proj(state_feat)
                        g = self.g_proj(event_gate)
                        h = torch.nn.functional.gelu(self.mix(torch.cat([q.expand_as(e), e, s, g], dim=-1)))
                        offset_raw = torch.tanh(self.time_head(h).squeeze(-1)) * self.max_offset
                        offset = offset_raw - offset_raw.mean(dim=1, keepdim=True)
                        alpha_local = torch.sigmoid(self.global_logit + self.local_alpha_scale * offset)
                        e_summary = enrich_temporal.mean(dim=1)
                        s_summary = state_h.mean(dim=1)
                        g_summary = event_gate.mean(dim=1)
                        trust_h = torch.nn.functional.gelu(self.global_mix(torch.cat([
                            self.gq_proj(q_embed),
                            self.ge_proj(e_summary),
                            self.gs_proj(s_summary),
                            self.gg_proj(g_summary),
                            self.type_proj(q_type_feats),
                            self.span_proj(span_feats),
                        ], dim=-1)))
                        trust_logit = self.trust_head(trust_h).squeeze(-1) + (q_type_feats * self.type_bias.unsqueeze(0)).sum(dim=-1)
                        trust = torch.sigmoid(trust_logit)
                        alpha = trust.unsqueeze(1) * alpha_local
                    alpha = alpha.to(dtype=q_embed.dtype)
                    if return_offset:
                        return alpha, offset
                    return alpha

            mixer_cls = (
                TrustAwareTemporalDeltaScale if cfg.get("stage2_mode", "") == "temporal_delta_scale_v10"
                else StateAwareTemporalDeltaScale if cfg.get("stage2_mode", "") == "temporal_delta_scale_v94"
                else TemporalDeltaScale
            )
            mixer = mixer_cls(
                proj_dim=cfg.get("proj_dim", 2048),
                hidden_dim=cfg.get("mixer_hidden", 128),
                state_dim=cfg.get("state_bottleneck", cfg.get("expert_bottleneck", 128)),
                init_alpha=cfg.get("init_alpha", 0.10),
                max_offset=cfg.get("max_offset", 1.5),
                local_alpha_scale=cfg.get("local_alpha_scale", 0.5),
            )
            mixer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "mixer.pt"), map_location="cpu", weights_only=True)
            )
            mixer = mixer.to(device=device, dtype=torch.float32).eval()
            version_label = (
                "v10" if module_type == "dual_proj_deltanet_v10"
                else
                "v9.4" if module_type == "dual_proj_deltanet_v94"
                else "v9.3" if module_type == "dual_proj_deltanet_v93"
                else
                "v9.2" if module_type == "dual_proj_deltanet_v92"
                else "v9.1" if module_type == "dual_proj_deltanet_v91"
                else "v9.0"
            )
            eval_logger.info(f"Loaded Design A {version_label} ({cfg.get('mixer_type', 'temporal delta scale')})")
        else:
            raise ValueError(f"Unsupported v9/v10 stage2_mode: {cfg.get('stage2_mode', '')}")

        return "diff_delta_mixer", (expert, mixer), cfg

    elif module_type in ("dual_proj_deltanet_v810_stage1", "dual_proj_deltanet_v810"):
        from core.vision_projector.dual_proj import DiffDeltaTemporalExpert

        expert = DiffDeltaTemporalExpert(
            proj_dim=cfg.get("proj_dim", 2048),
            bottleneck=cfg.get("expert_bottleneck", 128),
            kernel_size=cfg.get("expert_kernel_size", 3),
            max_enrich_gate=cfg.get("max_enrich_gate", 0.25),
            pooling_ratio=cfg.get("pooling_ratio", 2),
            state_from_prepool=cfg.get("state_from_prepool", False),
            local_short_window=cfg.get("local_short_window", False),
        )
        expert.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "diff_expert.pt"), map_location="cpu", weights_only=True), strict=False
        )
        expert = expert.to(device=device, dtype=dtype).eval()

        if module_type == "dual_proj_deltanet_v810_stage1":
            eval_logger.info("Loaded Design A v8.10 stage1 (token-wise diff delta expert)")
            return "diff_delta_expert", (expert,), cfg

        if cfg.get("stage2_mode", "") == "global_delta_scale_v2":
            class GlobalDeltaScale(torch.nn.Module):
                def __init__(self, init_alpha=0.25):
                    super().__init__()
                    init_alpha = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
                    init_logit = math.log(init_alpha / (1.0 - init_alpha))
                    self.logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

                def forward(self, batch_size=None, dtype=None, device=None):
                    alpha = torch.sigmoid(self.logit)
                    if dtype is not None or device is not None:
                        alpha = alpha.to(dtype=dtype or alpha.dtype, device=device or alpha.device)
                    if batch_size is None:
                        return alpha
                    return alpha.expand(batch_size)

            mixer = GlobalDeltaScale(init_alpha=cfg.get("init_alpha", 0.25))
            mixer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "mixer.pt"), map_location="cpu", weights_only=True)
            )
            mixer = mixer.to(device=device, dtype=torch.float32).eval()
            eval_logger.info("Loaded Design A v8.10 (diff-delta expert + global delta scale)")
        elif cfg.get("stage2_mode", "") == "conditional_delta_scale_v3":
            class ConditionalDeltaScale(torch.nn.Module):
                def __init__(self, proj_dim=2048, hidden_dim=128, init_alpha=0.25, max_offset=1.5):
                    super().__init__()
                    init_alpha = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
                    init_logit = math.log(init_alpha / (1.0 - init_alpha))
                    self.global_logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
                    self.max_offset = float(max_offset)
                    self.local_alpha_scale = float(local_alpha_scale)
                    self.q_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.e_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.offset_head = torch.nn.Linear(hidden_dim, 1)

                def forward(self, q_embed, enrich_pooled, return_offset=False):
                    with torch.autocast(device_type="cuda", enabled=False):
                        q_embed = q_embed.float()
                        enrich_pooled = enrich_pooled.float()
                        enrich_summary = enrich_pooled.mean(dim=(1, 2))
                        h = torch.nn.functional.gelu(self.q_proj(q_embed) + self.e_proj(enrich_summary))
                        offset = torch.tanh(self.offset_head(h).squeeze(-1)) * self.max_offset
                        alpha = torch.sigmoid(self.global_logit + offset)
                    return alpha.to(dtype=q_embed.dtype)

            mixer = ConditionalDeltaScale(
                proj_dim=cfg.get("proj_dim", 2048),
                hidden_dim=cfg.get("mixer_hidden", 128),
                init_alpha=cfg.get("init_alpha", 0.25),
                max_offset=cfg.get("max_offset", 1.5),
            )
            mixer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "mixer.pt"), map_location="cpu", weights_only=True)
            )
            mixer = mixer.to(device=device, dtype=torch.float32).eval()
            eval_logger.info("Loaded Design A v8.10 (diff-delta expert + conditional delta scale)")
        elif cfg.get("stage2_mode", "") == "conditional_delta_scale_v4":
            class ConditionalDeltaScale(torch.nn.Module):
                def __init__(self, proj_dim=2048, hidden_dim=128, init_alpha=0.15, max_offset=2.0):
                    super().__init__()
                    init_alpha = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
                    init_logit = math.log(init_alpha / (1.0 - init_alpha))
                    self.global_logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
                    self.max_offset = float(max_offset)
                    self.local_alpha_scale = float(local_alpha_scale)
                    self.hidden_dim = int(hidden_dim)
                    self.q_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.e_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.offset_head = torch.nn.Linear(hidden_dim, 1)

                def forward(self, q_embed, enrich_pooled):
                    with torch.autocast(device_type="cuda", enabled=False):
                        q_embed = q_embed.float()
                        enrich_pooled = enrich_pooled.float()
                        enrich_temporal = enrich_pooled.mean(dim=2)
                        q = self.q_proj(q_embed)
                        e = self.e_proj(enrich_temporal)
                        scores = torch.einsum("bd,btd->bt", q, e) / math.sqrt(self.hidden_dim)
                        attn = torch.nn.functional.softmax(scores, dim=-1)
                        context = torch.einsum("bt,btd->bd", attn, e)
                        h = torch.nn.functional.gelu(q + context)
                        offset = torch.tanh(self.offset_head(h).squeeze(-1)) * self.max_offset
                        alpha = torch.sigmoid(self.global_logit + offset)
                    alpha = alpha.to(dtype=q_embed.dtype)
                    if return_offset:
                        return alpha, offset
                    return alpha

            mixer = ConditionalDeltaScale(
                proj_dim=cfg.get("proj_dim", 2048),
                hidden_dim=cfg.get("mixer_hidden", 128),
                init_alpha=cfg.get("init_alpha", 0.10),
                max_offset=cfg.get("max_offset", 1.5),
                local_alpha_scale=cfg.get("local_alpha_scale", 0.5),
            )
            mixer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "mixer.pt"), map_location="cpu", weights_only=True)
            )
            mixer = mixer.to(device=device, dtype=torch.float32).eval()
            eval_logger.info("Loaded Design A v8.10 (diff-delta expert + temporal-aware conditional delta scale)")
        elif cfg.get("stage2_mode", "") == "temporal_delta_scale_v6":
            class TemporalDeltaScale(torch.nn.Module):
                def __init__(self, proj_dim=2048, hidden_dim=128, init_alpha=0.15, max_offset=2.0):
                    super().__init__()
                    init_alpha = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
                    init_logit = math.log(init_alpha / (1.0 - init_alpha))
                    self.global_logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
                    self.max_offset = float(max_offset)
                    self.local_alpha_scale = float(local_alpha_scale)
                    self.hidden_dim = int(hidden_dim)
                    self.q_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.e_proj = torch.nn.Linear(proj_dim, hidden_dim)
                    self.time_head = torch.nn.Linear(hidden_dim, 1)

                def forward(self, q_embed, enrich_pooled, return_offset=False):
                    with torch.autocast(device_type="cuda", enabled=False):
                        q_embed = q_embed.float()
                        enrich_pooled = enrich_pooled.float()
                        enrich_temporal = enrich_pooled.mean(dim=2)
                        q = self.q_proj(q_embed).unsqueeze(1)
                        e = self.e_proj(enrich_temporal)
                        h = torch.nn.functional.gelu(q + e)
                        offset_raw = torch.tanh(self.time_head(h).squeeze(-1)) * self.max_offset
                        offset = offset_raw - offset_raw.mean(dim=1, keepdim=True)
                        alpha = torch.sigmoid(self.global_logit + offset)
                    alpha = alpha.to(dtype=q_embed.dtype)
                    if return_offset:
                        return alpha, offset
                    return alpha

            mixer = TemporalDeltaScale(
                proj_dim=cfg.get("proj_dim", 2048),
                hidden_dim=cfg.get("mixer_hidden", 128),
                init_alpha=cfg.get("init_alpha", 0.10),
                max_offset=cfg.get("max_offset", 1.5),
                local_alpha_scale=cfg.get("local_alpha_scale", 0.5),
            )
            mixer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "mixer.pt"), map_location="cpu", weights_only=True)
            )
            mixer = mixer.to(device=device, dtype=torch.float32).eval()
            eval_logger.info("Loaded Design A v8.10 (diff-delta expert + temporal delta scale)")
        else:
            class EnrichQueryMixer(torch.nn.Module):
                def __init__(self, proj_dim=2048, shared_dim=128):
                    super().__init__()
                    self.shared_dim = shared_dim
                    self.q_proj = torch.nn.Linear(proj_dim, shared_dim)
                    self.e_proj = torch.nn.Linear(proj_dim, shared_dim)
                    self.alpha_head = torch.nn.Linear(shared_dim, 1)
                def forward(self, q_embed, enrich_pooled):
                    e_temporal = enrich_pooled.mean(dim=2)
                    q = self.q_proj(q_embed)
                    e = self.e_proj(e_temporal)
                    attn = torch.einsum("bd,btd->bt", q, e) / (self.shared_dim ** 0.5)
                    attn = torch.nn.functional.softmax(attn, dim=-1)
                    context = torch.einsum("bt,btd->bd", attn, e)
                    interaction = q * context
                    return torch.sigmoid(self.alpha_head(interaction).squeeze(-1))

            mixer = EnrichQueryMixer(
                proj_dim=cfg.get("proj_dim", 2048),
                shared_dim=cfg.get("mixer_shared_dim", 128),
            )
            mixer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "mixer.pt"), map_location="cpu", weights_only=True)
            )
            mixer = mixer.to(device=device, dtype=dtype).eval()
            eval_logger.info("Loaded Design A v8.10 (diff-delta expert + enrich-query mixer)")

        return "diff_delta_mixer", (expert, mixer), cfg

    else:
        eval_logger.error(f"Unknown module_type: {module_type}")
        return None, None, None


def _load_llm_lora(checkpoint_dir, model):
    """Load LLM LoRA weights if present."""
    import sys
    from pathlib import Path
    for candidate in [
        str(Path(__file__).resolve().parent.parent.parent),
        "/workspace/perception_models",
    ]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from core.lora import apply_lora_to_model, load_lora_weights

    lora_path = os.path.join(checkpoint_dir, "lora.pt")
    if not os.path.exists(lora_path):
        return 0

    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    if not cfg.get("use_llm_lora", False):
        return 0

    llm_lora_rank = cfg.get("llm_lora_rank", 8)
    apply_lora_to_model(
        model.model.language_model,
        target_modules=["q_proj", "v_proj"],
        rank=llm_lora_rank,
        alpha=llm_lora_rank * 2.0,
        dropout=0.0,  # no dropout at eval
    )
    loaded = load_lora_weights(model.model.language_model, lora_path)
    eval_logger.info(f"Loaded LLM LoRA: {loaded} tensors (rank={llm_lora_rank})")
    return loaded


@register_model("plm_dual_proj")
class PLMWithDualProj(lmms):
    """PLM-1B + Dual Projector (Design A or B)."""

    def __init__(
        self,
        pretrained: str = "facebook/Perception-LM-1B",
        checkpoint_dir: str = "none",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        max_video_frames: Optional[Union[int, str]] = 32,
        progress_dir: str = "/tmp/plm_eval_progress",
        **kwargs,
    ) -> None:
        super().__init__()

        from transformers import AutoModelForImageTextToText, AutoProcessor

        accelerator = Accelerator()
        self._device = torch.device(f"cuda:{accelerator.local_process_index}")

        eval_logger.info(f"Loading PLM-1B from {pretrained}")
        self._hf_model = AutoModelForImageTextToText.from_pretrained(
            pretrained, dtype=torch.bfloat16, device_map={"": self._device})
        self._hf_processor = AutoProcessor.from_pretrained(pretrained)

        for p in self._hf_model.parameters():
            p.requires_grad = False
        self._hf_model.eval()

        vanilla_mode = str(checkpoint_dir).lower() in ("none", "null", "") or \
            not os.path.isdir(str(checkpoint_dir))

        self._design = None
        self._adapter = None
        self._q_gate = None
        self._temporal_branch = None
        self._q_suppressor = None
        self._num_frames_cfg = 32
        self._adapter_cfg = {}
        self._router_marker_ids = build_router_marker_ids(self._hf_processor.tokenizer)
        self._question_type_keyword_ids = build_question_type_keyword_ids(self._hf_processor.tokenizer)

        if vanilla_mode:
            eval_logger.info("=== VANILLA MODE ===")
        else:
            param_dtype = next(self._hf_model.parameters()).dtype

            # Load LLM LoRA first (modifies model in-place)
            _load_llm_lora(checkpoint_dir, self._hf_model)

            # Load adapter
            self._design, self._adapter, cfg = _load_dual_proj_adapter(
                checkpoint_dir, self._hf_model, self._device, param_dtype)
            self._checkpoint_dir = checkpoint_dir
            if cfg:
                self._adapter_cfg = cfg
                self._num_frames_cfg = cfg.get("num_frames", 32)

            # Store extra components for specific designs
            if self._design == "deltanet_qgate":
                self._q_gate = self._adapter[2]  # QuestionGate
            elif self._design == "hidden_temporal":
                self._temporal_branch = self._adapter[0]  # HiddenTemporalBranch
            elif self._design == "dpm_qsup":
                self._q_suppressor = self._adapter[2]  # QuestionSuppressor
            elif self._design == "dpm_router":
                self._q_suppressor = self._adapter[2]
                self._safe_proj_lora = self._adapter[3]
                self._safe_merger = self._adapter[4]
                self._expert_router = self._adapter[5]
            elif self._design == "dpm_v11_router":
                self._safe_lora = self._adapter[2]   # SafeProjectorLoRA
                self._expert_router = self._adapter[3]  # ExpertRouter
            elif self._design == "dpm_temporal_mixer":
                self._tm_temp_proj = self._adapter[0]
                self._tm_mixer = self._adapter[1]
            elif self._design == "two_expert":
                self._expert_s_lora = self._adapter[0]  # ProjectorLoRA
                self._expert_s_merger = self._adapter[1]  # DeltaNetMerger
                self._expert_t_branch = self._adapter[2]  # HiddenTemporalBranch
                self._expert_router = self._adapter[3]  # ExpertRouter
            elif self._design == "temporal_mixer":
                self._tm_proj_lora = self._adapter[0]   # frozen ProjectorLoRA
                self._tm_merger = self._adapter[1]       # frozen DeltaNetMerger
                self._tm_mixer = self._adapter[2]        # QuestionMixer
            elif self._design == "v11_stage1":
                self._v11_temporal_branch = self._adapter[0]
            elif self._design == "v11_router":
                self._v11_temporal_branch = self._adapter[0]
                self._v11_router = self._adapter[1]
            elif self._design == "v12_router":
                self._v12_proj_lora = self._adapter[0]
                self._v12_merger = self._adapter[1]
                self._v12_router = self._adapter[2]

            elif self._design == "temporal_expert_router":
                self._te_router_expert = self._adapter[0]
                self._te_router = self._adapter[1]
            elif self._design == "temporal_expert_alpha":
                self._te_alpha_expert = self._adapter[0]
                self._te_alpha = self._adapter[1]
            elif self._design == "temporal_expert_policy":
                self._te_policy_expert = self._adapter[0]
                self._te_policy = self._adapter[1]
            elif self._design == "temporal_expert_alpha_v2":
                self._te_alpha_v2_expert = self._adapter[0]
                self._te_alpha_v2 = self._adapter[1]
            elif self._design == "temporal_expert_gate_mod":
                self._te_gate_mod_expert = self._adapter[0]
                self._te_gate_mod = self._adapter[1]
            elif self._design == "temporal_expert_rescomp":
                self._te_rescomp_expert = self._adapter[0]
                self._te_rescomp = self._adapter[1]
            elif self._design == "temporal_expert_resdistill":
                self._te_resdistill_expert = self._adapter[0]
                self._te_resdistill = self._adapter[1]
            elif self._design == "temporal_expert_mix4":
                self._te_mix4_expert = self._adapter[0]
                self._te_mix4 = self._adapter[1]
            elif self._design == "temporal_expert_mix3":
                self._te_mix3_expert = self._adapter[0]
                self._te_mix3 = self._adapter[1]
            elif self._design == "temporal_expert_srouter3":
                self._te_srouter3_expert = self._adapter[0]
                self._te_srouter3 = self._adapter[1]
            elif self._design == "temporal_expert_hrouter3":
                self._te_hrouter3_expert = self._adapter[0]
                self._te_hrouter3 = self._adapter[1]
            elif self._design == "temporal_expert_ctxsrouter3":
                self._te_ctxsrouter3_expert = self._adapter[0]
                self._te_ctxsrouter3 = self._adapter[1]
            elif self._design == "temporal_expert_bosrouter3":
                self._te_bosrouter3_expert = self._adapter[0]
                self._te_bosrouter3 = self._adapter[1]
            elif self._design == "udh_v2_res2":
                self._udh_v2_expert = self._adapter[0]
                self._udh_v2_calibrator = self._adapter[1]
            elif self._design == "temporal_expert_ctxhrouter3":
                self._te_ctxhrouter3_expert = self._adapter[0]
                self._te_ctxhrouter3 = self._adapter[1]
            elif self._design == "temporal_expert_ctxres2":
                self._te_ctxres2_expert = self._adapter[0]
                self._te_ctxres2 = self._adapter[1]
            elif self._design == "temporal_expert_lspolicy":
                self._te_lspolicy_expert = self._adapter[0]
                self._te_lspolicy = self._adapter[1]
            elif self._design == "temporal_expert_g2s":
                self._te_g2s_expert = self._adapter[0]
                self._te_g2s = self._adapter[1]
            elif self._design == "temporal_expert_qg2s":
                self._te_qg2s_expert = self._adapter[0]
                self._te_qg2s = self._adapter[1]
            elif self._design == "temporal_expert_router_v2":
                self._te_router_v2_expert = self._adapter[0]
                self._te_router_v2 = self._adapter[1]
            elif self._design == "temporal_expert_gattn":
                self._te_gattn_expert = self._adapter[0]
                self._te_gattn = self._adapter[1]

            elif self._design == "factorized_temporal_memory_stage1":
                self._tmem_expert = self._adapter[0]
            elif self._design == "factorized_temporal_memory_readout":
                self._tmem_expert = self._adapter[0]
                self._tmem_readout = self._adapter[1]
            elif self._design == "factorized_gated_delta_stage1":
                self._fx_expert = self._adapter[0]
            elif self._design == "factorized_gated_delta_router":
                self._fx_expert = self._adapter[0]
                self._fx_router = self._adapter[1]

            elif self._design == "plm_tq_router":
                self._plm_tqr_expert = self._adapter[0]
                self._plm_tqr_router = self._adapter[1]

            elif self._design == "diff_delta_expert":
                self._dd_expert = self._adapter[0]
            elif self._design == "dbrouter":
                self._dbrouter_expert = self._adapter[0]
                self._dbrouter = self._adapter[1]
            elif self._design == "gate_finetune":
                self._gf_expert = self._adapter[0]
                self._gf_local_gate = self._adapter[1]
                self._gf_state_gate = self._adapter[2]
            elif self._design == "lora_sft_expert":
                self._dd_expert = self._adapter[0]  # reuse same forward as diff_delta_expert
            elif self._design == "diff_delta_mixer":
                self._dd_expert = self._adapter[0]
                self._dd_mixer = self._adapter[1]

        self.batch_size_per_gpu = int(batch_size)
        self._max_video_frames = int(max_video_frames) if max_video_frames else 32

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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _load_video_frames(self, video_path, max_frames=None):
        if not HAS_DECORD:
            raise RuntimeError("decord required")
        max_frames = max_frames or self._max_video_frames
        try:
            vr = decord.VideoReader(video_path, num_threads=1)
        except Exception as e:
            eval_logger.warning(f"Failed to read video {video_path}: {e}, using blank frames")
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * max_frames
        total = len(vr)
        if total <= max_frames:
            indices = list(range(total))
            while len(indices) < max_frames:
                indices.append(total - 1)
        else:
            indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()
        batch = vr.get_batch(indices)
        frames = batch.numpy() if isinstance(batch, torch.Tensor) else batch.asnumpy()
        return [frames[i] for i in range(frames.shape[0])]

    @staticmethod
    def _find_subsequence(seq, pattern, start, end):
        plen = len(pattern)
        if plen == 0 or end - start < plen:
            return None
        last = end - plen + 1
        for idx in range(start, last):
            if seq[idx:idx + plen] == pattern:
                return idx
        return None

    def _get_question_span(self, input_ids_row):
        seq = input_ids_row.tolist()
        valid_len = len(seq)
        while valid_len > 0 and seq[valid_len - 1] == PAD_ID:
            valid_len -= 1

        start = 0
        end = valid_len
        eot_positions = (input_ids_row == EOT_ID).nonzero(as_tuple=True)[0].tolist()
        if eot_positions:
            user_end = eot_positions[0]
            header_positions = (input_ids_row == END_HEADER_ID).nonzero(as_tuple=True)[0].tolist()
            header_positions = [pos for pos in header_positions if pos < user_end]
            if header_positions:
                start = header_positions[-1] + 1
            end = user_end

        if self._adapter_cfg.get("router_use_stem_only", False):
            for marker_ids in self._router_marker_ids:
                pos = self._find_subsequence(seq, marker_ids, start, end)
                if pos is not None:
                    end = min(end, pos)

        return start, end

    def _get_question_embedding(self, input_ids):
        """Extract mean question(-stem) embedding from input_ids."""
        if not self._adapter_cfg.get("router_use_stem_only", False):
            video_mask = (input_ids == VIDEO_TOKEN_ID)
            pad_mask = (input_ids == PAD_ID)
            text_mask = ~video_mask & ~pad_mask
            with torch.no_grad():
                all_embeds = self._hf_model.model.get_input_embeddings()(input_ids)
            B = input_ids.shape[0]
            q_embeds = []
            for i in range(B):
                mask_i = text_mask[i]
                if mask_i.any():
                    q_embeds.append(all_embeds[i][mask_i].mean(dim=0))
                else:
                    q_embeds.append(torch.zeros(
                        all_embeds.shape[-1], device=all_embeds.device, dtype=all_embeds.dtype))
            return torch.stack(q_embeds)

        with torch.no_grad():
            all_embeds = self._hf_model.model.get_input_embeddings()(input_ids)

        B = input_ids.shape[0]
        q_embeds = []
        for i in range(B):
            start, end = self._get_question_span(input_ids[i])
            tokens_i = input_ids[i, start:end]
            embeds_i = all_embeds[i, start:end]
            text_mask_i = (
                (tokens_i != VIDEO_TOKEN_ID)
                & (tokens_i != PAD_ID)
                & (tokens_i != END_HEADER_ID)
                & (tokens_i != EOT_ID)
            )
            if text_mask_i.any():
                q_embeds.append(embeds_i[text_mask_i].mean(dim=0))
            else:
                q_embeds.append(torch.zeros(
                    all_embeds.shape[-1], device=all_embeds.device, dtype=all_embeds.dtype))
        return torch.stack(q_embeds)

    def _get_question_type_features(self, input_ids):
        B = input_ids.shape[0]
        out = torch.zeros(B, 3, device=input_ids.device, dtype=torch.float32)
        for i in range(B):
            start, end = self._get_question_span(input_ids[i])
            seq = input_ids[i, start:end].tolist()
            for j, key in enumerate(("episodic", "sequence", "localization")):
                found = False
                for pattern in self._question_type_keyword_ids.get(key, []):
                    if self._find_subsequence(seq, pattern, 0, len(seq)) is not None:
                        found = True
                        break
                if found:
                    out[i, j] = 1.0
        return out

    @staticmethod
    def _get_span_features(event_gate):
        gate = event_gate.float().squeeze(-1)
        T = gate.shape[1]
        t_pos = torch.linspace(0.0, 1.0, T, device=gate.device, dtype=gate.dtype).unsqueeze(0)
        gate_sum = gate.sum(dim=1, keepdim=True).clamp(min=1e-6)
        mean = gate.mean(dim=1)
        peak = gate.max(dim=1).values
        coverage = (gate > 0.05).float().mean(dim=1)
        com = (gate * t_pos).sum(dim=1) / gate_sum.squeeze(1)
        spread = (gate * (t_pos - com.unsqueeze(1)).abs()).sum(dim=1) / gate_sum.squeeze(1)
        return torch.stack([mean, peak, coverage, com, spread], dim=-1)

    def _get_question_token_pack(self, input_ids):
        with torch.no_grad():
            all_embeds = self._hf_model.model.get_input_embeddings()(input_ids)

        B = input_ids.shape[0]
        seqs = []
        max_len = 1
        for i in range(B):
            start, end = self._get_question_span(input_ids[i])
            tokens_i = input_ids[i, start:end]
            embeds_i = all_embeds[i, start:end]
            text_mask_i = (
                (tokens_i != VIDEO_TOKEN_ID)
                & (tokens_i != PAD_ID)
                & (tokens_i != END_HEADER_ID)
                & (tokens_i != EOT_ID)
            )
            emb = embeds_i[text_mask_i]
            if emb.numel() == 0:
                emb = torch.zeros(
                    1,
                    all_embeds.shape[-1],
                    device=all_embeds.device,
                    dtype=all_embeds.dtype,
                )
            seqs.append(emb)
            max_len = max(max_len, emb.shape[0])

        packed = torch.zeros(
            B, max_len, all_embeds.shape[-1],
            device=all_embeds.device, dtype=all_embeds.dtype,
        )
        mask = torch.zeros(B, max_len, device=all_embeds.device, dtype=all_embeds.dtype)
        for i, emb in enumerate(seqs):
            packed[i, :emb.shape[0]] = emb
            mask[i, :emb.shape[0]] = 1
        return packed, mask

    def _get_user_text_token_pack(self, input_ids):
        with torch.no_grad():
            all_embeds = self._hf_model.model.get_input_embeddings()(input_ids)

        B = input_ids.shape[0]
        seqs = []
        max_len = 1
        for i in range(B):
            seq = input_ids[i]
            eot_positions = (seq == EOT_ID).nonzero(as_tuple=True)[0].tolist()
            if eot_positions:
                end = eot_positions[0]
                header_positions = (seq == END_HEADER_ID).nonzero(as_tuple=True)[0].tolist()
                header_positions = [pos for pos in header_positions if pos < end]
                start = header_positions[-1] + 1 if header_positions else 0
            else:
                start, end = 0, seq.shape[0]

            tokens_i = seq[start:end]
            embeds_i = all_embeds[i, start:end]
            text_mask_i = (
                (tokens_i != VIDEO_TOKEN_ID)
                & (tokens_i != PAD_ID)
                & (tokens_i != END_HEADER_ID)
                & (tokens_i != EOT_ID)
            )
            emb = embeds_i[text_mask_i]
            if emb.numel() == 0:
                emb = torch.zeros(
                    1,
                    all_embeds.shape[-1],
                    device=all_embeds.device,
                    dtype=all_embeds.dtype,
                )
            seqs.append(emb)
            max_len = max(max_len, emb.shape[0])

        packed = torch.zeros(
            B, max_len, all_embeds.shape[-1],
            device=all_embeds.device, dtype=all_embeds.dtype,
        )
        mask = torch.zeros(B, max_len, device=all_embeds.device, dtype=all_embeds.dtype)
        for i, emb in enumerate(seqs):
            packed[i, :emb.shape[0]] = emb
            mask[i, :emb.shape[0]] = 1
        return packed, mask

    def _get_question_texts(self, input_ids):
        texts = []
        for i in range(input_ids.shape[0]):
            start, end = self._get_question_span(input_ids[i])
            tokens_i = input_ids[i, start:end]
            text_mask_i = (
                (tokens_i != VIDEO_TOKEN_ID)
                & (tokens_i != PAD_ID)
                & (tokens_i != END_HEADER_ID)
                & (tokens_i != EOT_ID)
            )
            if text_mask_i.any():
                text = self._hf_processor.tokenizer.decode(
                    tokens_i[text_mask_i].tolist(), skip_special_tokens=True
                )
            else:
                text = ""
            texts.append(text.lower())
        return texts

    def _temporal_question_hint(self, input_ids):
        hints = []
        texts = self._get_question_texts(input_ids)
        temporal_keywords = [
            "what happens next",
            "what will happen",
            "going to do",
            "what is going to happen",
            "faster",
            "slower",
            "speed",
            "direction",
            "moving",
            "before",
            "after",
            "in order",
            "order",
        ]
        narrative_keywords = [
            "why did",
            "what did",
            "what was",
            "say",
            "said",
            "because",
            "reason",
        ]
        for text in texts:
            temporal_hit = any(k in text for k in temporal_keywords)
            narrative_hit = any(k in text for k in narrative_keywords)
            hints.append(temporal_hit and not narrative_hit)
        return torch.tensor(hints, device=input_ids.device, dtype=torch.bool)

    def _compute_router_video_stats(self, z_base, delta_safe, delta_main):
        base_pool = z_base.detach().mean(dim=2)
        delta_safe_det = delta_safe.detach()
        delta_main_det = delta_main.detach()
        if z_base.shape[1] > 1:
            base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
            main_temporal = (delta_main_det[:, 1:] - delta_main_det[:, :-1]).norm(dim=-1).mean(dim=(1, 2))
        else:
            base_motion = torch.zeros(z_base.shape[0], device=z_base.device, dtype=z_base.dtype)
            main_temporal = torch.zeros_like(base_motion)
        safe_mag = delta_safe_det.norm(dim=-1).mean(dim=(1, 2))
        main_mag = delta_main_det.norm(dim=-1).mean(dim=(1, 2))
        stats = torch.stack([base_motion, safe_mag, main_mag, main_temporal], dim=-1)
        return torch.log1p(stats.float()).to(device=z_base.device, dtype=z_base.dtype)

    def _encode_video(self, pixel_values_videos, input_ids=None):
        """Encode video with dual projector adapter.

        Args:
            pixel_values_videos: (B, T, C, H, W) video frames
            input_ids: (B, seq_len) token ids, needed for deltanet_qgate design
        """
        B, T = pixel_values_videos.shape[:2]
        frames_flat = pixel_values_videos.flatten(0, 1)
        proj = self._hf_model.model.multi_modal_projector

        with torch.no_grad():
            vision_out = self._hf_model.model.vision_tower(frames_flat)
            hidden = vision_out.last_hidden_state
            if self._hf_model.config.vision_use_cls_token:
                hidden = hidden[:, 1:, :]

            if self._adapter is None:
                # Vanilla mode
                projected = proj(hidden)

            elif self._design == "deltanet":
                proj_lora, merger = self._adapter
                _, N, C = hidden.shape

                # Base projector
                z_base_flat = proj(hidden)  # (BT, M, D)

                # Temporal projector (LoRA)
                z_temp_flat = proj_lora(hidden)  # (BT, M, D)

                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_base = z_base_flat.reshape(B, T, M, D)
                z_temp = z_temp_flat.reshape(B, T, M, D)

                merged = merger(z_base, z_temp)
                projected = merged.reshape(B * T, M, D)

            elif self._design == "deltanet_qgate":
                proj_lora, merger, q_gate = self._adapter
                _, N, C = hidden.shape

                # Base projector
                z_base_flat = proj(hidden)  # (BT, M, D)

                # Temporal projector (LoRA)
                z_temp_flat = proj_lora(hidden)  # (BT, M, D)

                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_base = z_base_flat.reshape(B, T, M, D)
                z_temp = z_temp_flat.reshape(B, T, M, D)

                # Merge via DeltaNet
                merged = merger(z_base, z_temp)  # (B, T, M, D)

                # Apply question gate to modulate enrichment
                enrichment = merged - z_base  # (B, T, M, D)
                q_embed = self._get_question_embedding(input_ids)  # (B, embed_dim)
                q_gate_val = self._q_gate(q_embed)  # (B, 1)
                # Broadcast: (B, 1, 1, 1) over (T, M, D)
                merged_gated = z_base + q_gate_val.unsqueeze(-1).unsqueeze(-1) * enrichment
                projected = merged_gated.reshape(B * T, M, D)

            elif self._design == "hidden_temporal":
                # v7.2: HiddenTemporalBranch operates between linear_1+GELU and linear_2
                # PLM projector order: linear_1 → GELU → linear_2 → pooling(256→64)
                _, N, C = hidden.shape

                # Forward through linear_1 + GELU
                h_base_flat = F.gelu(proj.linear_1(hidden))  # (BT, 256, 2048)
                N_out = h_base_flat.shape[1]
                hidden_dim = h_base_flat.shape[2]

                h_base = h_base_flat.reshape(B, T, N_out, hidden_dim)

                # Apply temporal branch
                h_temporal = self._temporal_branch(h_base)  # (B, T, 256, 2048)
                h_merged = h_base + h_temporal

                # Forward through linear_2 + pooling
                h_merged_flat = h_merged.reshape(B * T, N_out, hidden_dim)
                z_merged_prepool = proj.linear_2(h_merged_flat)  # (BT, 256, 2048)
                projected = proj.pooling(z_merged_prepool)        # (BT, 64, 2048)

            elif self._design == "two_expert":
                # v8.3: Two-Expert DeltaNet
                _, N, C = hidden.shape

                # z_base
                z_base_flat = proj(hidden)  # (BT, M, D)
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]

                # Expert S: ProjectorLoRA + DeltaNetMerger
                z_temp_s_flat = self._expert_s_lora(hidden)  # (BT, M, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                z_temp_s_4d = z_temp_s_flat.reshape(B, T, M, D)
                merged_s = self._expert_s_merger(z_base_4d, z_temp_s_4d)  # (B, T, M, D)
                delta_s = merged_s - z_base_4d

                # Expert T: HiddenTemporalBranch
                h_base_flat = F.gelu(proj.linear_1(hidden))  # (BT, 256, 2048)
                N_h = h_base_flat.shape[1]
                H_dim = h_base_flat.shape[2]
                h_base = h_base_flat.reshape(B, T, N_h, H_dim)
                h_temporal = self._expert_t_branch(h_base)
                h_merged = h_base + h_temporal
                h_merged_flat = h_merged.reshape(B * T, N_h, H_dim)
                z_temp_t_flat = proj.pooling(proj.linear_2(h_merged_flat))  # (BT, M, D)
                z_temp_t_4d = z_temp_t_flat.reshape(B, T, M, D)
                delta_t = z_temp_t_4d - z_base_4d

                # Router
                q_embed = self._get_question_embedding(input_ids)
                router_stats = None
                if self._adapter_cfg.get("router_stats_dim", 0) > 0:
                    router_stats = self._compute_router_video_stats(z_base_4d, delta_t, delta_s)
                weights = self._expert_router(q_embed, video_stats=router_stats)  # (B, 3)

                merge_mode = self._adapter_cfg.get("merge_mode", "weighted_sum")
                if merge_mode == "safe_main_correction":
                    w_safe = weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    w_main = weights[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    non_base = (w_safe + w_main).clamp(min=1e-6)
                    main_share = w_main / non_base
                    corr_main = delta_s - delta_t
                    corr_main = _apply_relative_delta_clamp(
                        corr_main,
                        delta_t,
                        float(self._adapter_cfg.get("main_corr_ratio", 0.35)),
                    )
                    z_anchor = z_base_4d + non_base * delta_t
                    z_final = z_anchor + non_base * main_share * corr_main
                else:
                    w_safe = weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                    w_temp = weights[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    z_final = z_base_4d + w_safe * delta_s + w_temp * delta_t

                # Eval-time selective clamp (optional, from adapter_config)
                clamp_mode = str(self._adapter_cfg.get("eval_clamp_mode", "none") or "none")
                if clamp_mode == "selective":
                    default_ratio = float(self._adapter_cfg.get("eval_clamp_ratio_default", 0.5) or 0.5)
                    relaxed_ratio = float(self._adapter_cfg.get("eval_clamp_ratio_relaxed", default_ratio) or default_ratio)
                    aggr_threshold = float(self._adapter_cfg.get("eval_relax_aggr_threshold", 0.45) or 0.45)
                    relax_margin = float(self._adapter_cfg.get("eval_relax_margin", 0.10) or 0.10)
                    use_hint = bool(self._adapter_cfg.get("eval_temporal_hint", True))
                    w_base_s = weights[:, 0]
                    w_safe_s = weights[:, 1]
                    w_temp_s = weights[:, 2]
                    relax_mask = (w_temp_s >= aggr_threshold) & ((w_temp_s - w_safe_s) >= relax_margin)
                    if use_hint:
                        relax_mask = relax_mask & self._temporal_question_hint(input_ids)
                    sample_ratio = torch.full((B, 1, 1, 1), default_ratio, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    relaxed = torch.full_like(sample_ratio, relaxed_ratio)
                    relax_mask = relax_mask.view(B, 1, 1, 1)
                    sample_ratio = torch.where(relax_mask, relaxed, sample_ratio)
                    z_final = _apply_delta_clamp(z_base_4d, z_final, sample_ratio)
                elif clamp_mode == "global":
                    clamp_ratio = float(self._adapter_cfg.get("eval_clamp_ratio", 0.0) or 0.0)
                    if clamp_ratio > 0:
                        z_final = _apply_delta_clamp(z_base_4d, z_final, clamp_ratio)

                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_mixer":
                # v8.9: Frozen temporal expert + QuestionMixer
                _, N, C = hidden.shape
                z_base_flat = proj(hidden)
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                z_temp_flat = self._tm_proj_lora(hidden)
                z_temp_4d = z_temp_flat.reshape(B, T, M, D)
                z_temporal = self._tm_merger(z_base_4d, z_temp_4d)
                delta = z_temporal - z_base_4d

                q_embed = self._get_question_embedding(input_ids)
                # Video stats
                base_pool = z_base_4d.mean(dim=2)
                delta_det = delta.detach()
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_diff = (delta_det[:, 1:] - delta_det[:, :-1]).norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=hidden.device, dtype=hidden.dtype)
                    temp_diff = torch.zeros_like(base_motion)
                delta_mag = delta_det.norm(dim=-1).mean(dim=(1, 2))
                delta_std = delta_det.norm(dim=-1).std(dim=1).mean(dim=1)
                stats = torch.log1p(torch.stack([base_motion, delta_mag, temp_diff, delta_std], dim=-1).float())
                stats = stats.to(device=hidden.device, dtype=hidden.dtype)

                alpha = self._tm_mixer(q_embed, stats)  # (B,)
                alpha_4d = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                z_final = z_base_4d + alpha_4d * delta
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "v11_stage1":
                temporal_branch = self._v11_temporal_branch
                _, N, C = hidden.shape
                h_base_flat = F.gelu(proj.linear_1(hidden))
                z_base_flat = proj.pooling(proj.linear_2(h_base_flat))
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                H_dim = h_base_flat.shape[-1]
                h_base = h_base_flat.reshape(B, T, N, H_dim)
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                h_temporal = temporal_branch(h_base)
                h_merged = h_base + h_temporal
                h_merged_flat = h_merged.reshape(B * T, N, H_dim)
                z_temp_flat = proj.pooling(proj.linear_2(h_merged_flat))
                z_temp_4d = z_temp_flat.reshape(B, T, M, D)
                z_temp_4d = _apply_delta_clamp(z_base_4d, z_temp_4d, float(self._adapter_cfg.get("eval_clamp_ratio", 0.30) or 0.30))
                projected = z_temp_4d.reshape(B * T, M, D)

            elif self._design == "v11_router":
                temporal_branch = self._v11_temporal_branch
                router = self._v11_router
                _, N, C = hidden.shape
                h_base_flat = F.gelu(proj.linear_1(hidden))
                z_base_flat = proj.pooling(proj.linear_2(h_base_flat))
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                H_dim = h_base_flat.shape[-1]
                h_base = h_base_flat.reshape(B, T, N, H_dim)
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                h_temporal = temporal_branch(h_base)
                h_merged = h_base + h_temporal
                h_merged_flat = h_merged.reshape(B * T, N, H_dim)
                z_temp_flat = proj.pooling(proj.linear_2(h_merged_flat))
                z_temp_4d = z_temp_flat.reshape(B, T, M, D)
                z_temp_4d = _apply_delta_clamp(z_base_4d, z_temp_4d, float(self._adapter_cfg.get("eval_clamp_ratio", 0.30) or 0.30))
                delta = z_temp_4d - z_base_4d
                q_embed = self._get_question_embedding(input_ids)
                gate_tensor = getattr(temporal_branch, '_last_gate_tensor', None)
                base_pool = z_base_4d.detach().mean(dim=2)
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_temporal = (delta[:, 1:] - delta[:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    temp_temporal = torch.zeros_like(base_motion)
                temp_mag = delta.detach().norm(dim=-1).mean(dim=(1, 2))
                if gate_tensor is not None:
                    gate_mean = gate_tensor.detach().mean(dim=1)
                else:
                    gate_mean = torch.zeros_like(base_motion)
                router_stats = torch.log1p(torch.stack([base_motion, temp_mag, temp_temporal, gate_mean], dim=-1).float()).to(device=z_base_4d.device, dtype=z_base_4d.dtype)
                weights = router(q_embed, video_stats=router_stats)
                w_temp = weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                z_final = z_base_4d + w_temp * delta
                z_final = _apply_delta_clamp(z_base_4d, z_final, float(self._adapter_cfg.get("eval_clamp_ratio", 0.30) or 0.30))
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "v12_router":
                proj_lora = self._v12_proj_lora
                merger = self._v12_merger
                router = self._v12_router
                _, N, C = hidden.shape

                z_base_flat = proj(hidden)
                z_temp_flat = proj_lora(hidden)

                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                z_temp_4d = z_temp_flat.reshape(B, T, M, D)
                z_safe_4d = merger(z_base_4d, z_temp_4d)
                delta = z_safe_4d - z_base_4d

                q_embed = self._get_question_embedding(input_ids)
                base_pool = z_base_4d.detach().mean(dim=2)
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_temporal = (delta[:, 1:] - delta[:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    temp_temporal = torch.zeros_like(base_motion)
                temp_mag = delta.detach().norm(dim=-1).mean(dim=(1, 2))
                temp_peak = delta.detach().norm(dim=-1).amax(dim=(1, 2))
                router_stats = torch.log1p(
                    torch.stack([base_motion, temp_mag, temp_temporal, temp_peak], dim=-1).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                weights = router(q_embed, video_stats=router_stats)
                w_temp = weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                z_final = z_base_4d + w_temp * delta
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_router":
                expert = self._te_router_expert
                router = self._te_router
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type == "dual_proj_deltanet_v810_stage1":
                    z_temp_4d, _, _ = expert(z_pre_pool_4d, z_base_4d)
                    aux = {}
                else:
                    z_temp_4d, _, _, aux = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="router")
                delta = z_temp_4d - z_base_4d

                q_embed = self._get_question_embedding(input_ids)
                base_pool = z_base_4d.detach().mean(dim=2)
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_temporal = (delta[:, 1:] - delta[:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    temp_temporal = torch.zeros_like(base_motion)
                temp_mag = delta.detach().norm(dim=-1).mean(dim=(1, 2))
                if "event_gate" in aux:
                    control = aux["event_gate"].detach().mean(dim=1).squeeze(-1)
                else:
                    control = delta.detach().norm(dim=-1).amax(dim=(1, 2))
                router_stats = torch.log1p(
                    torch.stack([base_motion, temp_mag, temp_temporal, control], dim=-1).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                weights = router(q_embed, video_stats=router_stats)
                w_temp = weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                z_final = z_base_4d + w_temp * delta
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_alpha":
                expert = self._te_alpha_expert
                alpha_calibrator = self._te_alpha
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type == "dual_proj_deltanet_v810_stage1":
                    z_temp_4d, _, _ = expert(z_pre_pool_4d, z_base_4d)
                    aux = {}
                else:
                    z_temp_4d, _, _, aux = expert(z_pre_pool_4d, z_base_4d, return_aux=True)
                delta = z_temp_4d - z_base_4d

                q_embed = self._get_question_embedding(input_ids)
                base_pool = z_base_4d.detach().mean(dim=2)
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_temporal = (delta[:, 1:] - delta[:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    temp_temporal = torch.zeros_like(base_motion)
                temp_mag = delta.detach().norm(dim=-1).mean(dim=(1, 2))
                delta_peak = delta.detach().norm(dim=-1).amax(dim=(1, 2))
                if "event_gate" in aux:
                    gate_mean = aux["event_gate"].detach().mean(dim=1).squeeze(-1)
                elif "state_gate" in aux:
                    gate_mean = aux["state_gate"].detach().mean(dim=1).squeeze(-1)
                else:
                    gate_mean = torch.zeros_like(base_motion)
                if "state_h" in aux and aux["state_h"].shape[1] > 1:
                    state_h = aux["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)
                video_feat = torch.log1p(
                    torch.stack([base_motion, temp_mag, temp_temporal, delta_peak, gate_mean, state_shift], dim=-1).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                alpha = alpha_calibrator(q_embed, video_feat)
                z_final = z_base_4d + alpha.unsqueeze(-1).unsqueeze(-1) * delta
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_policy":
                expert = self._te_policy_expert
                policy_calibrator = self._te_policy
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type == "dual_proj_deltanet_v810_stage1":
                    z_temp_4d, _, _ = expert(z_pre_pool_4d, z_base_4d)
                    aux = {}
                else:
                    z_temp_4d, _, _, aux = expert(z_pre_pool_4d, z_base_4d, return_aux=True)
                delta = z_temp_4d - z_base_4d

                q_embed = self._get_question_embedding(input_ids)
                base_pool = z_base_4d.detach().mean(dim=2)
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_temporal = (delta[:, 1:] - delta[:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    temp_temporal = torch.zeros_like(base_motion)
                temp_mag = delta.detach().norm(dim=-1).mean(dim=(1, 2))
                delta_peak = delta.detach().norm(dim=-1).amax(dim=(1, 2))
                if "event_gate" in aux:
                    gate_mean = aux["event_gate"].detach().mean(dim=1).squeeze(-1)
                elif "state_gate" in aux:
                    gate_mean = aux["state_gate"].detach().mean(dim=1).squeeze(-1)
                else:
                    gate_mean = torch.zeros_like(base_motion)
                if "state_h" in aux and aux["state_h"].shape[1] > 1:
                    state_h = aux["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)
                video_feat = torch.log1p(
                    torch.stack([base_motion, temp_mag, temp_temporal, delta_peak, gate_mean, state_shift], dim=-1).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                gate = policy_calibrator(q_embed, video_feat)
                z_final = z_base_4d + gate.unsqueeze(-1).unsqueeze(-1) * delta
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_alpha_v2":
                expert = self._te_alpha_v2_expert
                alpha_calibrator = self._te_alpha_v2
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type == "dual_proj_deltanet_v810_stage1":
                    z_temp_4d, _, _ = expert(z_pre_pool_4d, z_base_4d)
                    aux = {}
                else:
                    z_temp_4d, _, _, aux = expert(z_pre_pool_4d, z_base_4d, return_aux=True)
                delta = z_temp_4d - z_base_4d

                q_embed = self._get_question_embedding(input_ids)
                base_pool = z_base_4d.detach().mean(dim=2)
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_temporal = (delta[:, 1:] - delta[:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    temp_temporal = torch.zeros_like(base_motion)
                temp_mag = delta.detach().norm(dim=-1).mean(dim=(1, 2))
                delta_peak = delta.detach().norm(dim=-1).amax(dim=(1, 2))
                if "event_gate" in aux:
                    gate_mean = aux["event_gate"].detach().mean(dim=1).squeeze(-1)
                elif "state_gate" in aux:
                    gate_mean = aux["state_gate"].detach().mean(dim=1).squeeze(-1)
                else:
                    gate_mean = torch.zeros_like(base_motion)
                if "state_h" in aux and aux["state_h"].shape[1] > 1:
                    state_h = aux["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)
                if "state_h_slots" in aux and aux["state_h_slots"].shape[2] > 1:
                    state_h_slots = aux["state_h_slots"].detach()
                    if T > 1:
                        slot_delta = (state_h_slots[:, 1:] - state_h_slots[:, :-1]).norm(dim=-1)
                        slot_change_mean = slot_delta.mean(dim=(1, 2))
                        slot_change_max = slot_delta.amax(dim=(1, 2))
                    else:
                        slot_change_mean = torch.zeros_like(base_motion)
                        slot_change_max = torch.zeros_like(base_motion)
                    slot_dispersion = state_h_slots.norm(dim=-1).std(dim=2).mean(dim=1)
                else:
                    slot_change_mean = torch.zeros_like(base_motion)
                    slot_change_max = torch.zeros_like(base_motion)
                    slot_dispersion = torch.zeros_like(base_motion)
                video_feat = torch.log1p(
                    torch.stack(
                        [
                            base_motion,
                            temp_mag,
                            temp_temporal,
                            delta_peak,
                            gate_mean,
                            state_shift,
                            slot_change_mean,
                            slot_change_max,
                            slot_dispersion,
                        ],
                        dim=-1,
                    ).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                alpha = alpha_calibrator(q_embed, video_feat)
                z_final = z_base_4d + alpha.unsqueeze(-1).unsqueeze(-1) * delta
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_gate_mod":
                expert = self._te_gate_mod_expert
                modulator = self._te_gate_mod
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
                    raise ValueError(f"temporal_expert_gate_mod expects slot-aware v9.2.x stage1, got {stage1_module_type!r}")

                z_ref_4d, _, _, aux = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="router")
                delta = z_ref_4d - z_base_4d

                q_embed = self._get_question_embedding(input_ids)
                base_pool = z_base_4d.detach().mean(dim=2)
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_temporal = (delta[:, 1:] - delta[:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    temp_temporal = torch.zeros_like(base_motion)
                temp_mag = delta.detach().norm(dim=-1).mean(dim=(1, 2))
                delta_peak = delta.detach().norm(dim=-1).amax(dim=(1, 2))
                gate_mean = aux["event_gate"].detach().mean(dim=1).squeeze(-1)

                if aux["state_h"].shape[1] > 1:
                    state_h = aux["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)

                state_h_slots = aux["state_h_slots"].detach()
                if T > 1:
                    slot_delta = (state_h_slots[:, 1:] - state_h_slots[:, :-1]).norm(dim=-1)
                    slot_change_mean = slot_delta.mean(dim=(1, 2))
                    slot_change_max = slot_delta.amax(dim=(1, 2))
                else:
                    slot_change_mean = torch.zeros_like(base_motion)
                    slot_change_max = torch.zeros_like(base_motion)
                slot_dispersion = state_h_slots.norm(dim=-1).std(dim=2).mean(dim=1)

                video_feat = torch.log1p(
                    torch.stack(
                        [
                            base_motion,
                            temp_mag,
                            temp_temporal,
                            delta_peak,
                            gate_mean,
                            state_shift,
                            slot_change_mean,
                            slot_change_max,
                            slot_dispersion,
                        ],
                        dim=-1,
                    ).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                modulation = modulator(q_embed, video_feat)
                z_final, _, _, _ = expert(
                    z_pre_pool_4d,
                    z_base_4d,
                    return_aux=True,
                    return_aux_mode="router",
                    modulation=modulation,
                )
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_rescomp":
                expert = self._te_rescomp_expert
                composer = self._te_rescomp
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
                    raise ValueError(f"temporal_expert_rescomp expects slot-aware v9.2.x stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="full")
                base_pool = z_base_4d.detach().mean(dim=2)
                local_res = aux_ref["local_residual"].detach()
                state_res = aux_ref["state_residual"].detach()
                total_res = local_res + state_res

                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                local_mag = local_res.norm(dim=-1).mean(dim=(1, 2))
                state_mag = state_res.norm(dim=-1).mean(dim=(1, 2))
                total_mag = total_res.norm(dim=-1).mean(dim=(1, 2))
                gate_mean = aux_ref["event_gate"].detach().mean(dim=1).squeeze(-1)

                if aux_ref["state_h"].shape[1] > 1:
                    state_h = aux_ref["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)

                state_h_slots = aux_ref["state_h_slots"].detach()
                if T > 1:
                    slot_delta = (state_h_slots[:, 1:] - state_h_slots[:, :-1]).norm(dim=-1)
                    slot_change_mean = slot_delta.mean(dim=(1, 2))
                    slot_change_max = slot_delta.amax(dim=(1, 2))
                else:
                    slot_change_mean = torch.zeros_like(base_motion)
                    slot_change_max = torch.zeros_like(base_motion)
                slot_dispersion = state_h_slots.norm(dim=-1).std(dim=2).mean(dim=1)

                video_feat = torch.log1p(
                    torch.stack(
                        [
                            base_motion,
                            local_mag,
                            state_mag,
                            total_mag,
                            gate_mean,
                            state_shift,
                            slot_change_mean,
                            slot_change_max,
                            slot_dispersion,
                        ],
                        dim=-1,
                    ).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                q_embed = self._get_question_embedding(input_ids)
                compose = composer(q_embed, video_feat)
                aux_mod = expert.compose_residuals_from_cache(
                    z_base_4d,
                    aux_ref["local_h"],
                    aux_ref["state_h_slots"],
                    slot_bias=compose["slot_bias"],
                )
                a_local = compose["a_local"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                a_state = compose["a_state"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + a_local * aux_ref["local_residual"] + a_state * aux_mod["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_resdistill":
                expert = self._te_resdistill_expert
                composer = self._te_resdistill
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
                    raise ValueError(f"temporal_expert_resdistill expects slot-aware v9.2.x stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                base_pool = z_base_4d.detach().mean(dim=2)
                local_res = aux_ref["local_residual"].detach()
                state_res = aux_ref["state_residual"].detach()
                total_res = local_res + state_res

                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                local_mag = local_res.norm(dim=-1).mean(dim=(1, 2))
                state_mag = state_res.norm(dim=-1).mean(dim=(1, 2))
                total_mag = total_res.norm(dim=-1).mean(dim=(1, 2))
                gate_mean = aux_ref["event_gate"].detach().mean(dim=1).squeeze(-1)

                if aux_ref["state_h"].shape[1] > 1:
                    state_h = aux_ref["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)

                state_h_slots = aux_ref["state_h_slots"].detach()
                if T > 1:
                    slot_delta = (state_h_slots[:, 1:] - state_h_slots[:, :-1]).norm(dim=-1)
                    slot_change_mean = slot_delta.mean(dim=(1, 2))
                    slot_change_max = slot_delta.amax(dim=(1, 2))
                else:
                    slot_change_mean = torch.zeros_like(base_motion)
                    slot_change_max = torch.zeros_like(base_motion)
                slot_dispersion = state_h_slots.norm(dim=-1).std(dim=2).mean(dim=1)

                video_feat = torch.log1p(
                    torch.stack(
                        [
                            base_motion,
                            local_mag,
                            state_mag,
                            total_mag,
                            gate_mean,
                            state_shift,
                            slot_change_mean,
                            slot_change_max,
                            slot_dispersion,
                        ],
                        dim=-1,
                    ).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                q_embed = self._get_question_embedding(input_ids)
                compose = composer(q_embed, video_feat)
                u_local = compose["u_local"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                u_state = compose["u_state"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + u_local * aux_ref["local_residual"] + u_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_mix4":
                expert = self._te_mix4_expert
                mixer = self._te_mix4
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
                    raise ValueError(f"temporal_expert_mix4 expects slot-aware v9.2.x stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                base_pool = z_base_4d.detach().mean(dim=2)
                local_res = aux_ref["local_residual"].detach()
                state_res = aux_ref["state_residual"].detach()
                total_res = local_res + state_res

                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                local_mag = local_res.norm(dim=-1).mean(dim=(1, 2))
                state_mag = state_res.norm(dim=-1).mean(dim=(1, 2))
                total_mag = total_res.norm(dim=-1).mean(dim=(1, 2))
                gate_mean = aux_ref["event_gate"].detach().mean(dim=1).squeeze(-1)

                if aux_ref["state_h"].shape[1] > 1:
                    state_h = aux_ref["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)

                state_h_slots = aux_ref["state_h_slots"].detach()
                if T > 1:
                    slot_delta = (state_h_slots[:, 1:] - state_h_slots[:, :-1]).norm(dim=-1)
                    slot_change_mean = slot_delta.mean(dim=(1, 2))
                    slot_change_max = slot_delta.amax(dim=(1, 2))
                else:
                    slot_change_mean = torch.zeros_like(base_motion)
                    slot_change_max = torch.zeros_like(base_motion)
                slot_dispersion = state_h_slots.norm(dim=-1).std(dim=2).mean(dim=1)

                video_feat = torch.log1p(
                    torch.stack(
                        [
                            base_motion,
                            local_mag,
                            state_mag,
                            total_mag,
                            gate_mean,
                            state_shift,
                            slot_change_mean,
                            slot_change_max,
                            slot_dispersion,
                        ],
                        dim=-1,
                    ).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                q_embed = self._get_question_embedding(input_ids)
                mix = mixer(q_embed, video_feat)
                weights = mix["weights"]
                u_local = (weights[:, 1] + weights[:, 3]).to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                u_state = (weights[:, 2] + weights[:, 3]).to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + u_local * aux_ref["local_residual"] + u_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_mix3":
                expert = self._te_mix3_expert
                mixer = self._te_mix3
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
                    raise ValueError(f"temporal_expert_mix3 expects slot-aware v9.2.x stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                base_pool = z_base_4d.detach().mean(dim=2)
                local_res = aux_ref["local_residual"].detach()
                state_res = aux_ref["state_residual"].detach()

                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                local_mag = local_res.norm(dim=-1).mean(dim=(1, 2))
                state_mag = state_res.norm(dim=-1).mean(dim=(1, 2))
                gate_mean = aux_ref["event_gate"].detach().mean(dim=1).squeeze(-1)

                if aux_ref["state_h"].shape[1] > 1:
                    state_h = aux_ref["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)

                state_h_slots = aux_ref["state_h_slots"].detach()
                if T > 1:
                    slot_delta = (state_h_slots[:, 1:] - state_h_slots[:, :-1]).norm(dim=-1)
                    slot_change_mean = slot_delta.mean(dim=(1, 2))
                    slot_change_max = slot_delta.amax(dim=(1, 2))
                else:
                    slot_change_mean = torch.zeros_like(base_motion)
                    slot_change_max = torch.zeros_like(base_motion)
                slot_dispersion = state_h_slots.norm(dim=-1).std(dim=2).mean(dim=1)

                video_feat = torch.log1p(
                    torch.stack(
                        [
                            base_motion,
                            local_mag,
                            state_mag,
                            local_mag + state_mag,
                            gate_mean,
                            state_shift,
                            slot_change_mean,
                            slot_change_max,
                            slot_dispersion,
                        ],
                        dim=-1,
                    ).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                q_embed = self._get_question_embedding(input_ids)
                mix = mixer(q_embed, video_feat)
                weights = mix["weights"]
                u_local = weights[:, 1].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                u_state = weights[:, 2].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + u_local * aux_ref["local_residual"] + u_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_srouter3":
                expert = self._te_srouter3_expert
                router = self._te_srouter3
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
                    raise ValueError(f"temporal_expert_srouter3 expects v926/v927 stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                base_pool = z_base_4d.detach().mean(dim=2)
                local_res = aux_ref["local_residual"].detach()
                state_res = aux_ref["state_residual"].detach()

                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                local_mag = local_res.norm(dim=-1).mean(dim=(1, 2))
                state_mag = state_res.norm(dim=-1).mean(dim=(1, 2))
                gate_mean = aux_ref["event_gate"].detach().mean(dim=1).squeeze(-1)
                if aux_ref["state_h"].shape[1] > 1:
                    state_h = aux_ref["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)
                state_h_slots = aux_ref["state_h_slots"].detach()
                if T > 1:
                    slot_delta = (state_h_slots[:, 1:] - state_h_slots[:, :-1]).norm(dim=-1)
                    slot_change_mean = slot_delta.mean(dim=(1, 2))
                    slot_change_max = slot_delta.amax(dim=(1, 2))
                else:
                    slot_change_mean = torch.zeros_like(base_motion)
                    slot_change_max = torch.zeros_like(base_motion)
                slot_dispersion = state_h_slots.norm(dim=-1).std(dim=2).mean(dim=1)
                video_feat = torch.log1p(
                    torch.stack(
                        [
                            base_motion,
                            local_mag,
                            state_mag,
                            local_mag + state_mag,
                            gate_mean,
                            state_shift,
                            slot_change_mean,
                            slot_change_max,
                            slot_dispersion,
                        ],
                        dim=-1,
                    ).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)
                q_embed = self._get_question_embedding(input_ids)
                weights = router(q_embed, video_feat)
                u_local = weights[:, 1].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                u_state = weights[:, 2].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + u_local * aux_ref["local_residual"] + u_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_hrouter3":
                expert = self._te_hrouter3_expert
                router = self._te_hrouter3
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
                    raise ValueError(f"temporal_expert_hrouter3 expects v926/v927 stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                base_pool = z_base_4d.detach().mean(dim=2)
                local_res = aux_ref["local_residual"].detach()
                state_res = aux_ref["state_residual"].detach()

                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                local_mag = local_res.norm(dim=-1).mean(dim=(1, 2))
                state_mag = state_res.norm(dim=-1).mean(dim=(1, 2))
                gate_mean = aux_ref["event_gate"].detach().mean(dim=1).squeeze(-1)
                if aux_ref["state_h"].shape[1] > 1:
                    state_h = aux_ref["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)
                state_h_slots = aux_ref["state_h_slots"].detach()
                if T > 1:
                    slot_delta = (state_h_slots[:, 1:] - state_h_slots[:, :-1]).norm(dim=-1)
                    slot_change_mean = slot_delta.mean(dim=(1, 2))
                    slot_change_max = slot_delta.amax(dim=(1, 2))
                else:
                    slot_change_mean = torch.zeros_like(base_motion)
                    slot_change_max = torch.zeros_like(base_motion)
                slot_dispersion = state_h_slots.norm(dim=-1).std(dim=2).mean(dim=1)
                video_feat = torch.log1p(
                    torch.stack(
                        [
                            base_motion,
                            local_mag,
                            state_mag,
                            local_mag + state_mag,
                            gate_mean,
                            state_shift,
                            slot_change_mean,
                            slot_change_max,
                            slot_dispersion,
                        ],
                        dim=-1,
                    ).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)
                q_embed = self._get_question_embedding(input_ids)
                route = router(q_embed, video_feat)
                weights = route["weights"]
                u_local = weights[:, 1].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                u_state = weights[:, 2].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + u_local * aux_ref["local_residual"] + u_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_bosrouter3":
                expert = self._te_bosrouter3_expert
                router = self._te_bosrouter3
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1", "dual_proj_deltanet_udh_v1_stage1"):
                    raise ValueError(f"temporal_expert_bosrouter3 expects v926/v927/v928/udh_v1 stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                z_local = z_base_4d + aux_ref["local_residual"]
                z_state = z_base_4d + aux_ref["state_residual"]
                q_tokens, q_mask = self._get_user_text_token_pack(input_ids)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    route = router(q_tokens, q_mask, z_base_4d.detach(), z_local.detach(), z_state.detach())
                weights = route["weights"]
                w_base = weights[:, 0].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                w_local = weights[:, 1].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                w_state = weights[:, 2].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = w_base * z_base_4d + w_local * z_local + w_state * z_state
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "udh_v2_res2":
                expert = self._udh_v2_expert
                calibrator = self._udh_v2_calibrator

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_base_flat = proj.pooling(features)
                n_pre = features.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = features.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                local_res = aux_ref["local_residual"]
                state_res = aux_ref["state_residual"]

                # Calibrator: question + per-frame video + stats → a_local, a_state
                q_embed = self._get_question_embedding(input_ids)
                z_temp = z_base_4d + local_res + state_res
                v_frames = z_temp.mean(dim=2)  # (B, T, D)

                # Compute sample_stats
                local_mag = local_res.flatten(1).norm(dim=-1)
                state_mag = state_res.flatten(1).norm(dim=-1)
                enrich_mag = (z_temp - z_base_4d).flatten(1).norm(dim=-1)
                v_temporal_std = v_frames.std(dim=1).mean(dim=-1)
                qv_cos = F.cosine_similarity(q_embed, v_frames.mean(dim=1), dim=-1)
                local_state_ratio = local_mag / (state_mag + 1e-6)
                raw_stats = torch.stack([
                    local_mag, state_mag, enrich_mag,
                    v_temporal_std, qv_cos, local_state_ratio
                ], dim=-1)
                # Normalize if stats_norm.pt exists (for checkpoints trained with batch norm)
                stats_norm_path = os.path.join(self._checkpoint_dir, "stats_norm.pt")
                if os.path.exists(stats_norm_path):
                    norm = torch.load(stats_norm_path, map_location=raw_stats.device, weights_only=True)
                    s_mean = norm["mean"].to(raw_stats.dtype).unsqueeze(0)
                    s_std = norm["std"].to(raw_stats.dtype).unsqueeze(0).clamp(min=1e-6)
                    sample_stats = (raw_stats - s_mean) / s_std
                else:
                    sample_stats = raw_stats

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    scales = calibrator(q_embed, v_frames, sample_stats=sample_stats)
                a_local = scales["a_local"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                a_state = scales["a_state"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + a_local * local_res + a_state * state_res
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "dbrouter":
                expert = self._dbrouter_expert
                router = self._dbrouter

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_base_flat = proj.pooling(features)
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                z_pre_pool_4d = z_base_4d  # training uses pooled for both

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                local_res = aux_ref["local_residual"]
                state_res = aux_ref["state_residual"]

                # Router: question embedding + video stats → w_local, w_state
                q_embed = self._get_question_embedding(input_ids)
                # Compute stats (same as training)
                base_pool = z_base_4d.detach().mean(dim=2)
                if z_base_4d.shape[1] > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                local_mag = local_res.detach().norm(dim=-1).mean(dim=(1, 2))
                state_mag = state_res.detach().norm(dim=-1).mean(dim=(1, 2))
                total_mag = (local_res + state_res).detach().norm(dim=-1).mean(dim=(1, 2))
                stats = torch.log1p(torch.stack([base_motion, local_mag, state_mag, total_mag], dim=-1).float()).to(dtype=z_base_4d.dtype)

                oracle_tau = self._adapter_cfg.get("oracle_tau", 0.0)
                if oracle_tau > 0:
                    # ── Per-sample oracle routing (Method C) ──
                    # Compare QA losses to determine optimal per-sample weights
                    z_base_flat_o = z_base_4d.reshape(B, -1, D)
                    z_local_flat = (z_base_4d + local_res).reshape(B, -1, D)
                    z_state_flat = (z_base_4d + state_res).reshape(B, -1, D)

                    def _oracle_nll(z_flat):
                        ie = self.model.model.get_input_embeddings()(input_ids).detach()
                        zc = z_flat.to(ie.dtype)
                        _, vm = self.model.model.get_placeholder_mask(input_ids, inputs_embeds=ie, video_features=zc)
                        ie = ie.masked_scatter(vm, zc)
                        out = self.model.model.language_model(attention_mask=attention_mask, inputs_embeds=ie, return_dict=True)
                        h = out.last_hidden_state
                        # Per-sample loss (mean over valid tokens per sample)
                        losses = []
                        for si in range(B):
                            shift_h = h[si, :-1]
                            shift_y = labels[si, 1:] if labels is not None else input_ids[si, 1:]
                            valid = shift_y != -100
                            if valid.any():
                                logits_si = self.model.lm_head(shift_h[valid]).float()
                                losses.append(torch.nn.functional.cross_entropy(logits_si, shift_y[valid], reduction="mean"))
                            else:
                                losses.append(torch.tensor(0.0, device=h.device))
                        return torch.stack(losses)

                    with torch.no_grad():
                        nll_base = _oracle_nll(z_base_flat_o)
                        nll_local = _oracle_nll(z_local_flat)
                        nll_state = _oracle_nll(z_state_flat)
                        gain_l = nll_base - nll_local
                        gain_s = nll_base - nll_state
                        w_local = torch.sigmoid(gain_l / oracle_tau).to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                        w_state = torch.sigmoid(gain_s / oracle_tau).to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                else:
                    # ── Standard router ──
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        w = router(q_embed, video_stats=stats)
                    w_local = w[:, 0].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                    w_state = w[:, 1].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)

                z_final = z_base_4d + w_local * local_res + w_state * state_res
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "gate_finetune":
                expert = self._gf_expert
                local_gate_new = self._gf_local_gate
                state_gate_new = self._gf_state_gate

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_base_flat = proj.pooling(features)
                n_pre = features.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = features.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                # Run expert DeltaNet (frozen) to get hidden states
                _, local_res_orig, state_res_orig, aux = expert(
                    z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                local_h = aux.get("local_h")  # (B, T-1, 1024, 128)
                state_h = aux.get("state_h")  # (B, T, 128)

                s_post = M
                # Pool local_h and get up projections
                local_h_pooled = expert._pool_enrich(local_h)  # (B, T-1, 256, 128)
                local_up_val = expert.local_up(local_h_pooled)
                state_ctx = state_h[:, 1:]
                state_up_val = expert.state_up(state_ctx)

                z_base_for_gate = z_base_4d[:, 1:]  # (B, T-1, 256, D)

                # New context-aware gates
                local_gate_logits = local_gate_new(local_h_pooled, z_base_for_gate)
                local_gate_val = expert.max_local_gate * torch.sigmoid(local_gate_logits)
                local_enrich = local_gate_val * local_up_val

                state_gate_logits = state_gate_new(
                    state_ctx.unsqueeze(2).expand(-1, -1, s_post, -1),
                    z_base_for_gate)
                state_gate_val = expert.max_state_gate * torch.sigmoid(state_gate_logits)
                state_enrich = state_gate_val * state_up_val.unsqueeze(2).expand(-1, -1, s_post, -1)

                local_residual = torch.zeros_like(z_base_4d)
                local_residual[:, 1:] = local_enrich
                state_residual = torch.zeros_like(z_base_4d)
                state_residual[:, 1:] = state_enrich

                z_final = z_base_4d + local_residual + state_residual
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_ctxsrouter3":
                expert = self._te_ctxsrouter3_expert
                router = self._te_ctxsrouter3
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
                    raise ValueError(f"temporal_expert_ctxsrouter3 expects v926/v927/v928 stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                q_tokens, q_mask = self._get_user_text_token_pack(input_ids)
                v_tokens = z_base_4d.detach()
                route = router(q_tokens, q_mask, v_tokens)
                weights = route["weights"]
                u_local = weights[:, 1].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                u_state = weights[:, 2].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + u_local * aux_ref["local_residual"] + u_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_ctxhrouter3":
                expert = self._te_ctxhrouter3_expert
                router = self._te_ctxhrouter3
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
                    raise ValueError(f"temporal_expert_ctxhrouter3 expects v926/v927/v928 stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                q_tokens, q_mask = self._get_user_text_token_pack(input_ids)
                v_tokens = z_base_4d.detach()
                route = router(q_tokens, q_mask, v_tokens)
                weights = route["weights"]
                u_local = weights[:, 1].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                u_state = weights[:, 2].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + u_local * aux_ref["local_residual"] + u_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_ctxres2":
                expert = self._te_ctxres2_expert
                scaler = self._te_ctxres2
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
                    raise ValueError(f"temporal_expert_ctxres2 expects v926/v927/v928 stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                q_tokens, q_mask = self._get_user_text_token_pack(input_ids)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    scales = scaler(q_tokens, q_mask, z_base_4d.detach())
                a_local = scales["a_local"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                a_state = scales["a_state"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + a_local * aux_ref["local_residual"] + a_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_lspolicy":
                expert = self._te_lspolicy_expert
                scaler = self._te_lspolicy
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v926_stage1", "dual_proj_deltanet_v927_stage1", "dual_proj_deltanet_v928_stage1"):
                    raise ValueError(f"temporal_expert_lspolicy expects v926/v927/v928 stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                q_tokens, q_mask = self._get_user_text_token_pack(input_ids)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    scales = scaler(q_tokens, q_mask, z_base_4d.detach())
                a_local = scales["a_local"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                a_state = scales["a_state"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + a_local * aux_ref["local_residual"] + a_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_g2s":
                expert = self._te_g2s_expert
                scaler = self._te_g2s
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
                    raise ValueError(f"temporal_expert_g2s expects slot-aware v9.2.x stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                scales = scaler(batch_size=B, dtype=z_base_4d.dtype, device=z_base_4d.device)
                a_local = scales["a_local"].view(B, 1, 1, 1)
                a_state = scales["a_state"].view(B, 1, 1, 1)
                z_final = z_base_4d + a_local * aux_ref["local_residual"] + a_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)
            elif self._design == "temporal_expert_qg2s":
                expert = self._te_qg2s_expert
                scaler = self._te_qg2s
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type not in ("dual_proj_deltanet_v922_stage1", "dual_proj_deltanet_v923_stage1", "dual_proj_deltanet_v924_stage1", "dual_proj_deltanet_v925_stage1", "dual_proj_deltanet_v926_stage1"):
                    raise ValueError(f"temporal_expert_qg2s expects slot-aware v9.2.x stage1, got {stage1_module_type!r}")

                _, _, _, aux_ref = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="residual")
                q_embed = self._get_question_embedding(input_ids)
                scales = scaler(q_embed)
                a_local = scales["a_local"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                a_state = scales["a_state"].to(dtype=z_base_4d.dtype).view(B, 1, 1, 1)
                z_final = z_base_4d + a_local * aux_ref["local_residual"] + a_state * aux_ref["state_residual"]
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_router_v2":
                expert = self._te_router_v2_expert
                router = self._te_router_v2
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type == "dual_proj_deltanet_v810_stage1":
                    z_temp_4d, _, _ = expert(z_pre_pool_4d, z_base_4d)
                    aux = {}
                else:
                    z_temp_4d, _, _, aux = expert(z_pre_pool_4d, z_base_4d, return_aux=True, return_aux_mode="router")
                delta = z_temp_4d - z_base_4d

                q_embed = self._get_question_embedding(input_ids)
                base_pool = z_base_4d.detach().mean(dim=2)
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_temporal = (delta[:, 1:] - delta[:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    temp_temporal = torch.zeros_like(base_motion)
                temp_mag = delta.detach().norm(dim=-1).mean(dim=(1, 2))
                delta_peak = delta.detach().norm(dim=-1).amax(dim=(1, 2))
                if "event_gate" in aux:
                    gate_mean = aux["event_gate"].detach().mean(dim=1).squeeze(-1)
                elif "state_gate" in aux:
                    gate_mean = aux["state_gate"].detach().mean(dim=1).squeeze(-1)
                else:
                    gate_mean = torch.zeros_like(base_motion)
                if "state_h" in aux and aux["state_h"].shape[1] > 1:
                    state_h = aux["state_h"].detach()
                    state_shift = (state_h[:, 1:] - state_h[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    state_shift = torch.zeros_like(base_motion)
                if "state_h_slots" in aux and aux["state_h_slots"].shape[2] > 1:
                    state_h_slots = aux["state_h_slots"].detach()
                    if T > 1:
                        slot_delta = (state_h_slots[:, 1:] - state_h_slots[:, :-1]).norm(dim=-1)
                        slot_change_mean = slot_delta.mean(dim=(1, 2))
                        slot_change_max = slot_delta.amax(dim=(1, 2))
                    else:
                        slot_change_mean = torch.zeros_like(base_motion)
                        slot_change_max = torch.zeros_like(base_motion)
                    slot_dispersion = state_h_slots.norm(dim=-1).std(dim=2).mean(dim=1)
                else:
                    slot_change_mean = torch.zeros_like(base_motion)
                    slot_change_max = torch.zeros_like(base_motion)
                    slot_dispersion = torch.zeros_like(base_motion)
                video_feat = torch.log1p(
                    torch.stack(
                        [
                            base_motion,
                            temp_mag,
                            temp_temporal,
                            delta_peak,
                            gate_mean,
                            state_shift,
                            slot_change_mean,
                            slot_change_max,
                            slot_dispersion,
                        ],
                        dim=-1,
                    ).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                weights = router(q_embed, video_feat)
                w_temp = weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                z_final = z_base_4d + w_temp * delta
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "temporal_expert_gattn":
                expert = self._te_gattn_expert
                gate_module = self._te_gattn
                stage1_module_type = self._adapter_cfg.get("stage1_module_type", "")
                _, N, C = hidden.shape

                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)

                if stage1_module_type == "dual_proj_deltanet_v810_stage1":
                    raise ValueError("temporal_expert_gattn expects slot-aware stage1, not v810")
                z_temp_4d, _, _, aux = expert(z_pre_pool_4d, z_base_4d, return_aux=True)
                delta = z_temp_4d - z_base_4d

                if "state_h_slots" in aux:
                    state_h_slots = aux["state_h_slots"].detach()
                    k = state_h_slots.shape[2]
                elif "state_h" in aux:
                    state_h_slots = aux["state_h"].detach().unsqueeze(2)
                    k = 1
                else:
                    state_h_slots = torch.zeros(
                        B, T, 1, self._adapter_cfg.get("state_bottleneck", self._adapter_cfg.get("expert_bottleneck", 128)),
                        device=z_base_4d.device, dtype=z_base_4d.dtype,
                    )
                    k = 1

                if "state_slots" in aux:
                    state_slots = aux["state_slots"].detach()
                    k = state_slots.shape[2]
                else:
                    m_grid = int(math.sqrt(M))
                    k_grid = int(math.sqrt(k))
                    if m_grid * m_grid != M or k_grid * k_grid != k:
                        raise ValueError(f"Invalid token/slot grid for gattn eval: M={M}, K={k}")
                    x_2d = z_base_4d.detach().permute(0, 1, 3, 2).reshape(B * T, D, m_grid, m_grid)
                    x_2d = F.adaptive_avg_pool2d(x_2d, (k_grid, k_grid))
                    state_slots = x_2d.flatten(2).transpose(1, 2).reshape(B, T, k, D)

                k_grid = int(math.sqrt(k))
                if k_grid * k_grid != k:
                    raise ValueError(f"Expected square state slots for gattn eval, got K={k}")

                delta_2d = delta.detach().permute(0, 1, 3, 2).reshape(B * T, D, int(math.sqrt(M)), int(math.sqrt(M)))
                delta_2d = F.adaptive_avg_pool2d(delta_2d, (k_grid, k_grid))
                delta_slots = delta_2d.flatten(2).transpose(1, 2).reshape(B, T, k, D)
                slot_delta_mag = delta_slots.norm(dim=-1, keepdim=True)

                slot_state_change = torch.zeros(B, T, k, 1, device=z_base_4d.device, dtype=z_base_4d.dtype)
                if T > 1:
                    slot_state_change[:, 1:] = (state_slots[:, 1:] - state_slots[:, :-1]).norm(dim=-1, keepdim=True)

                if "event_gate" in aux:
                    event_gate = aux["event_gate"].detach()
                elif "state_gate" in aux:
                    event_gate = aux["state_gate"].detach()
                else:
                    event_gate = torch.zeros(B, T, 1, device=z_base_4d.device, dtype=z_base_4d.dtype)
                event_gate = event_gate.unsqueeze(2).expand(-1, -1, k, -1)

                evidence = torch.cat([state_h_slots, slot_delta_mag, slot_state_change, event_gate], dim=-1)
                evidence_tokens = evidence.reshape(B, T * k, evidence.shape[-1])

                q_embed = self._get_question_embedding(input_ids)
                gates, _, _ = gate_module(q_embed, evidence_tokens, return_parts=True)
                slot_gates = gates.reshape(B, T, k)
                gate_2d = slot_gates.reshape(B * T, 1, k_grid, k_grid)
                gate_2d = F.interpolate(gate_2d, size=(int(math.sqrt(M)), int(math.sqrt(M))), mode="nearest")
                gate_map = gate_2d.reshape(B, T, M, 1)

                z_final = z_base_4d + gate_map * delta
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "factorized_temporal_memory_stage1":
                expert = self._tmem_expert
                _, N, C = hidden.shape
                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                z_mix_4d = expert(z_pre_pool_4d, z_base_4d)
                projected = z_mix_4d.reshape(B * T, M, D)

            elif self._design == "factorized_temporal_memory_readout":
                expert = self._tmem_expert
                readout = self._tmem_readout
                _, N, C = hidden.shape
                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                _, branch, _ = expert(z_pre_pool_4d, z_base_4d, return_aux=True)

                q_embed = self._get_question_embedding(input_ids)
                base_pool = z_base_4d.detach().mean(dim=2)
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    motion_temporal = (branch["motion"][:, 1:] - branch["motion"][:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    motion_temporal = torch.zeros_like(base_motion)
                motion_mag = branch["motion"].detach().norm(dim=-1).mean(dim=(1, 2))
                state_mag = branch["state"].detach().norm(dim=-1).mean(dim=(1, 2))
                corr_mag = branch["corr"].detach().norm(dim=-1).mean(dim=(1, 2))
                corr_anchor = (z_base_4d - z_base_4d[:, :1]).detach().norm(dim=-1).mean(dim=(1, 2))
                stats = torch.log1p(
                    torch.stack([base_motion, motion_mag, state_mag, corr_mag, motion_temporal, corr_anchor], dim=-1).float()
                ).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                weights = readout(q_embed, video_stats=stats)
                candidates = torch.stack(
                    [
                        z_base_4d,
                        z_base_4d + branch["motion"],
                        z_base_4d + branch["state"],
                        z_base_4d + branch["corr"],
                    ],
                    dim=1,
                )
                z_final = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * candidates).sum(dim=1)
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "factorized_gated_delta_stage1":
                expert = self._fx_expert
                _, N, C = hidden.shape
                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                z_mix_4d = expert(z_pre_pool_4d, z_base_4d)
                projected = z_mix_4d.reshape(B * T, M, D)

            elif self._design == "factorized_gated_delta_router":
                expert = self._fx_expert
                router = self._fx_router
                _, N, C = hidden.shape
                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                _, branch, aux = expert(z_pre_pool_4d, z_base_4d, return_aux=True)

                q_embed = self._get_question_embedding(input_ids)
                base_pool = aux["base_pool"].detach()
                motion_summary = aux["motion_summary"].detach()
                state_summary = aux["state_summary"].detach()
                struct_summary = aux["struct_summary"].detach()
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    motion_temporal = (branch["motion"][:, 1:] - branch["motion"][:, :-1]).detach().norm(dim=-1).mean(dim=(1, 2))
                    struct_temporal = (struct_summary[:, 1:] - struct_summary[:, :-1]).norm(dim=-1).mean(dim=1)
                else:
                    base_motion = torch.zeros(B, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    motion_temporal = torch.zeros_like(base_motion)
                    struct_temporal = torch.zeros_like(base_motion)

                stats = torch.stack(
                    [
                        motion_summary.norm(dim=-1).mean(dim=1),
                        state_summary.norm(dim=-1).mean(dim=1),
                        struct_summary.norm(dim=-1).mean(dim=1),
                        branch["motion"].detach().norm(dim=-1).mean(dim=(1, 2)),
                        branch["state"].detach().norm(dim=-1).mean(dim=(1, 2)),
                        branch["structure"].detach().norm(dim=-1).mean(dim=(1, 2)),
                        base_motion,
                        motion_temporal,
                        struct_temporal,
                    ],
                    dim=-1,
                )
                stats = torch.log1p(stats.float()).to(device=z_base_4d.device, dtype=z_base_4d.dtype)

                weights = router(q_embed, stats)
                candidates = torch.stack(
                    [
                        z_base_4d,
                        z_base_4d + branch["motion"],
                        z_base_4d + branch["state"],
                        z_base_4d + branch["structure"],
                    ],
                    dim=1,
                )
                z_final = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * candidates).sum(dim=1)
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "plm_tq_router":
                # PLM Stage 2: TemporalQueryRouter로 프레임별 gate 적용
                _, N, C = hidden.shape
                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_base_flat = proj.pooling(features)
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                z_pre_pool_4d = z_base_4d  # pooling 후이므로 동일

                # Expert → local/state enrichment
                _, local_res, state_res, _ = self._plm_tqr_expert(z_pre_pool_4d, z_base_4d, return_aux=True)

                # 질문 embedding 추출
                q_embed = self._get_question_embedding(input_ids)  # (B, D_llm)

                # 프레임별 summary (spatial mean)
                z_base_summary = z_base_4d.mean(dim=2)    # (B, T, D)
                local_summary = local_res.mean(dim=2)      # (B, T, D)
                state_summary = state_res.mean(dim=2)      # (B, T, D)

                # Router → 프레임별 gate
                local_scale, state_scale = self._plm_tqr_router(
                    q_embed, z_base_summary, local_summary, state_summary,
                )
                # Enrichment normalize + scale
                base_norm = z_base_4d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                local_norm = local_res * (base_norm / local_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                state_norm = state_res * (base_norm / state_res.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                ls = local_scale.unsqueeze(-1)
                ss = state_scale.unsqueeze(-1)
                z_final = (1.0 - ls - ss) * z_base_4d + ls * local_norm + ss * state_norm
                projected = z_final.reshape(B * T, M, D)

            elif self._design in ("diff_delta_expert", "lora_sft_expert"):
                _, N, C = hidden.shape
                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                expert_out = self._dd_expert(z_pre_pool_4d, z_base_4d)
                z_temp = expert_out[0] if isinstance(expert_out, tuple) else expert_out
                projected = z_temp.reshape(B * T, M, D)

            elif self._design == "diff_delta_mixer":
                stage2_mode = self._adapter_cfg.get("stage2_mode", "")
                _, N, C = hidden.shape
                features = hidden.permute(1, 0, 2)
                features = proj.linear_1(features)
                features = proj.gelu(features)
                features = proj.linear_2(features)
                features = features.permute(1, 0, 2)
                z_pre_pool_flat = features
                z_base_flat = proj.pooling(features)
                n_pre = z_pre_pool_flat.shape[1]
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_pre_pool_4d = z_pre_pool_flat.reshape(B, T, n_pre, D)
                z_base_4d = z_base_flat.reshape(B, T, M, D)
                if stage2_mode in ("temporal_delta_scale_v94", "temporal_delta_scale_v10"):
                    z_temp, enrich_pooled, _, aux = self._dd_expert(z_pre_pool_4d, z_base_4d, return_aux=True)
                else:
                    z_temp, enrich_pooled, _ = self._dd_expert(z_pre_pool_4d, z_base_4d)
                delta = z_temp - z_base_4d
                if stage2_mode == "global_delta_scale_v2":
                    alpha = self._dd_mixer(batch_size=B, dtype=z_base_4d.dtype, device=z_base_4d.device)
                    alpha_4d = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    z_final = z_base_4d + alpha_4d * delta
                elif stage2_mode in ("conditional_delta_scale_v3", "conditional_delta_scale_v4", "conditional_delta_scale_v5"):
                    q_embed = self._get_question_embedding(input_ids)
                    alpha = self._dd_mixer(q_embed, enrich_pooled)
                    alpha_4d = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    z_final = z_base_4d + alpha_4d * delta
                elif stage2_mode == "temporal_delta_scale_v6":
                    q_embed = self._get_question_embedding(input_ids)
                    alpha_t = self._dd_mixer(q_embed, enrich_pooled)
                    alpha_full = torch.zeros(B, T, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    alpha_full[:, 1:] = alpha_t.to(dtype=z_base_4d.dtype)
                    alpha_4d = alpha_full.unsqueeze(-1).unsqueeze(-1)
                    z_final = z_base_4d + alpha_4d * delta
                elif stage2_mode in ("temporal_delta_scale_v90", "temporal_delta_scale_v91", "temporal_delta_scale_v92", "temporal_delta_scale_v93", "temporal_delta_scale_v94", "temporal_delta_scale_v10"):
                    q_embed = self._get_question_embedding(input_ids)
                    # v9/v10 stage2 predicts per-time alpha only for frames 1..T-1.
                    # Frame 0 has no temporal delta and stays at alpha=0.
                    if stage2_mode in ("temporal_delta_scale_v94", "temporal_delta_scale_v10"):
                        if stage2_mode == "temporal_delta_scale_v10":
                            q_type_feats = self._get_question_type_features(input_ids)
                            span_feats = self._get_span_features(aux["event_gate"][:, 1:])
                            alpha_t = self._dd_mixer(
                                q_embed,
                                enrich_pooled[:, 1:],
                                aux["state_h"][:, 1:],
                                aux["event_gate"][:, 1:],
                                q_type_feats,
                                span_feats,
                            )
                        else:
                            alpha_t = self._dd_mixer(
                                q_embed,
                                enrich_pooled[:, 1:],
                                aux["state_h"][:, 1:],
                                aux["event_gate"][:, 1:],
                            )
                    else:
                        alpha_t = self._dd_mixer(q_embed, enrich_pooled[:, 1:])
                    alpha_full = torch.zeros(B, T, device=z_base_4d.device, dtype=z_base_4d.dtype)
                    alpha_full[:, 1:] = alpha_t.to(dtype=z_base_4d.dtype)
                    alpha_4d = alpha_full.unsqueeze(-1).unsqueeze(-1)
                    z_final = z_base_4d + alpha_4d * delta
                else:
                    q_embed = self._get_question_embedding(input_ids)
                    route_prob = self._dd_mixer(q_embed, enrich_pooled)
                    alpha_4d = route_prob.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    z_final = z_base_4d + alpha_4d * delta
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "dpm_v13_stage1":
                temp_proj = self._adapter[0]
                z_temp_flat = temp_proj(hidden, T)
                projected = z_temp_flat

            elif self._design in ("dpm", "dpm_qsup", "dpm_router"):
                temp_proj, fusion_gate = self._adapter[0], self._adapter[1]
                _, N, C = hidden.shape

                # Base projector
                z_base_flat = proj(hidden)

                # Temporal projector
                z_temp_flat = temp_proj(hidden, T)

                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_base = z_base_flat.reshape(B, T, M, D)
                z_temp = z_temp_flat.reshape(B, T, M, D)

                # v6: pass temporal block state to FusionGateV6 if available
                tb_state = getattr(temp_proj.temporal_block, '_last_frame_summary_detached', None)
                if tb_state is None:
                    tb_state = getattr(temp_proj.temporal_block, '_last_frame_summary', None)
                if hasattr(fusion_gate, 'forward') and 'tb_state' in fusion_gate.forward.__code__.co_varnames:
                    fused = fusion_gate(z_base, z_temp, tb_state=tb_state)
                else:
                    fused = fusion_gate(z_base, z_temp)

                # v8+: apply question suppressor if available
                if self._design in ("dpm_qsup", "dpm_router") and hasattr(self, '_q_suppressor'):
                    q_embed = self._get_question_embedding(input_ids)
                    q_sup = self._q_suppressor(q_embed)  # (B, 1)
                    enrichment = fused - z_base
                    fused = z_base + q_sup.unsqueeze(-1).unsqueeze(-1) * enrichment

                if self._design == "dpm_router":
                    # Safe dual_proj_v2-style branch
                    z_temp_safe_flat = self._safe_proj_lora(hidden)
                    z_temp_safe = z_temp_safe_flat.reshape(B, T, M, D)
                    z_safe = self._safe_merger(z_base, z_temp_safe)
                    delta_safe = z_safe - z_base

                    delta_aggr = fused - z_base
                    q_embed = self._get_question_embedding(input_ids)
                    weights = self._expert_router(q_embed)  # (B, 3)
                    w_safe = weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    w_aggr = weights[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    fused = z_base + w_safe * delta_safe + w_aggr * delta_aggr

                    clamp_mode = str(self._adapter_cfg.get("eval_clamp_mode", "global") or "global")
                    if clamp_mode == "route_aware":
                        base_ratio = float(self._adapter_cfg.get("eval_clamp_ratio_base", 0.0) or 0.0)
                        aggr_bonus = float(self._adapter_cfg.get("eval_clamp_ratio_aggr_bonus", 0.0) or 0.0)
                        sample_ratio = base_ratio + aggr_bonus * w_aggr
                        fused = _apply_delta_clamp(z_base, fused, sample_ratio)
                    elif clamp_mode == "selective":
                        default_ratio = float(self._adapter_cfg.get("eval_clamp_ratio_default", 0.5) or 0.5)
                        relaxed_ratio = float(self._adapter_cfg.get("eval_clamp_ratio_relaxed", default_ratio) or default_ratio)
                        aggr_threshold = float(self._adapter_cfg.get("eval_relax_aggr_threshold", 0.45) or 0.45)
                        relax_margin = float(self._adapter_cfg.get("eval_relax_margin", 0.10) or 0.10)
                        use_temporal_hint = bool(self._adapter_cfg.get("eval_temporal_hint", True))
                        w_base = weights[:, 0]
                        w_safe_s = weights[:, 1]
                        w_aggr_s = weights[:, 2]
                        relax_mask = (w_aggr_s >= aggr_threshold) & ((w_aggr_s - w_safe_s) >= relax_margin)
                        if use_temporal_hint:
                            relax_mask = relax_mask & self._temporal_question_hint(input_ids)
                        sample_ratio = torch.full(
                            (B, 1, 1, 1),
                            default_ratio,
                            device=z_base.device,
                            dtype=z_base.dtype,
                        )
                        relaxed = torch.full_like(sample_ratio, relaxed_ratio)
                        relax_mask = relax_mask.view(B, 1, 1, 1)
                        sample_ratio = torch.where(relax_mask, relaxed, sample_ratio)
                        fused = _apply_delta_clamp(z_base, fused, sample_ratio)
                    else:
                        clamp_ratio = float(self._adapter_cfg.get("eval_clamp_ratio", 0.0) or 0.0)
                        if clamp_mode != "none" and clamp_ratio > 0:
                            fused = _apply_delta_clamp(z_base, fused, clamp_ratio)

                projected = fused.reshape(B * T, M, D)

            elif self._design == "dpm_v11_router":
                temp_proj, fusion_gate = self._adapter[0], self._adapter[1]
                _, N, C = hidden.shape

                # Base + Aggressive branch (same as dpm)
                z_base_flat = proj(hidden)
                z_temp_flat = temp_proj(hidden, T)
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_base = z_base_flat.reshape(B, T, M, D)
                z_temp = z_temp_flat.reshape(B, T, M, D)

                tb_state = getattr(temp_proj.temporal_block, '_last_frame_summary_detached', None)
                if tb_state is None:
                    tb_state = getattr(temp_proj.temporal_block, '_last_frame_summary', None)
                if hasattr(fusion_gate, 'forward') and 'tb_state' in fusion_gate.forward.__code__.co_varnames:
                    fused_aggr = fusion_gate(z_base, z_temp, tb_state=tb_state)
                else:
                    fused_aggr = fusion_gate(z_base, z_temp)
                delta_aggr = fused_aggr - z_base

                # Safe branch (minimal LoRA, no temporal)
                z_safe_flat = self._safe_lora(hidden)  # (BT, M, D)
                z_safe = z_safe_flat.reshape(B, T, M, D)
                delta_safe = z_safe - z_base

                # Router
                q_embed = self._get_question_embedding(input_ids)
                weights = self._expert_router(q_embed)  # (B, 3) [base, aggr, safe]
                w_aggr = weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                w_safe = weights[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                z_final = z_base + w_aggr * delta_aggr + w_safe * delta_safe
                projected = z_final.reshape(B * T, M, D)

            elif self._design == "dpm_temporal_mixer":
                temp_proj, mixer = self._adapter
                _, N, C = hidden.shape
                z_base_flat = proj(hidden)
                z_temp_flat = temp_proj(hidden, T)
                M, D = z_base_flat.shape[1], z_base_flat.shape[2]
                z_base = z_base_flat.reshape(B, T, M, D)
                z_temp = z_temp_flat.reshape(B, T, M, D)

                z_aggr = z_temp
                delta = z_aggr - z_base

                q_embed = self._get_question_embedding(input_ids)
                base_pool = z_base.mean(dim=2)
                delta_det = delta.detach()
                if T > 1:
                    base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
                    temp_diff = (delta_det[:, 1:] - delta_det[:, :-1]).norm(dim=-1).mean(dim=(1, 2))
                else:
                    base_motion = torch.zeros(B, device=hidden.device, dtype=hidden.dtype)
                    temp_diff = torch.zeros_like(base_motion)
                delta_mag = delta_det.norm(dim=-1).mean(dim=(1, 2))
                delta_std = delta_det.norm(dim=-1).std(dim=1).mean(dim=1)
                stats = torch.log1p(torch.stack([base_motion, delta_mag, temp_diff, delta_std], dim=-1).float())
                stats = stats.to(device=hidden.device, dtype=hidden.dtype)

                stage2_mode = self._adapter_cfg.get("stage2_mode", "alpha_mixer")
                if stage2_mode == "expert_router_v3":
                    route = mixer(q_embed, stats)
                    if not self._adapter_cfg.get("soft_route", False):
                        route = (route >= 0.5).to(route.dtype)
                    route_4d = route.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    z_final = (1.0 - route_4d) * z_base + route_4d * z_aggr
                elif stage2_mode == "residual_router_v2":
                    delta = _apply_delta_clamp(
                        z_base, z_base + delta,
                        float(self._adapter_cfg.get("residual_max_ratio", 0.35) or 0.35),
                    ) - z_base
                    gate, scale = mixer(q_embed, stats)
                    gate_4d = gate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    scale_4d = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    z_final = z_base + gate_4d * scale_4d * delta
                else:
                    alpha = mixer(q_embed, stats)
                    alpha_4d = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    z_final = z_base + alpha_4d * delta
                projected = z_final.reshape(B * T, M, D)

        M = projected.shape[1]
        D = projected.shape[2]
        return projected.reshape(B, T * M, D)

    @torch.no_grad()
    def _generate_greedy_with_features(self, input_ids, video_features, max_new_tokens=256):
        """Generate using pre-computed video features, returning tokens + log probs."""
        inputs_embeds = self._hf_model.model.get_input_embeddings()(input_ids)
        if video_features is not None:
            vf = video_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            _, special_video_mask = self._hf_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=vf)
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, vf)

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        generated_tokens = []
        token_logprobs = []
        past_key_values = None
        cur_embeds = inputs_embeds
        cur_mask = attention_mask

        for _ in range(max_new_tokens):
            outputs = self._hf_model.model.language_model(
                inputs_embeds=cur_embeds, attention_mask=cur_mask,
                past_key_values=past_key_values, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
            logits = self._hf_model.lm_head(outputs.last_hidden_state[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            token_id = next_token.item()
            if token_id == self._hf_processor.tokenizer.eos_token_id:
                break
            # Log prob of chosen token
            log_probs = torch.nn.functional.log_softmax(logits[0, 0].float(), dim=-1)
            token_logprobs.append(log_probs[token_id].item())
            generated_tokens.append(token_id)
            cur_embeds = self._hf_model.model.get_input_embeddings()(next_token)
            cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=torch.long, device=cur_mask.device)], dim=1)

        text = self._hf_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        avg_logprob = sum(token_logprobs) / max(len(token_logprobs), 1)
        return text, avg_logprob

    def _prepare_embeds(self, input_ids, video_features):
        """Build inputs_embeds with video features inserted at placeholder positions."""
        inputs_embeds = self._hf_model.model.get_input_embeddings()(input_ids)
        if video_features is not None:
            vf = video_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            _, mask = self._hf_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=vf)
            inputs_embeds = inputs_embeds.masked_scatter(mask, vf)
        return inputs_embeds

    def _get_base_features(self, pixel_values_videos):
        """Compute vanilla (no-adapter) video features."""
        proj = self._hf_model.model.multi_modal_projector
        B, T = pixel_values_videos.shape[:2]
        frames_flat = pixel_values_videos.flatten(0, 1)
        with torch.no_grad():
            vision_out = self._hf_model.model.vision_tower(frames_flat)
            hidden = vision_out.last_hidden_state
            if self._hf_model.config.vision_use_cls_token:
                hidden = hidden[:, 1:, :]
            return proj(hidden)

    def _generate_contrastive(self, input_ids, feat_temp, feat_base, alpha, max_new_tokens=256):
        """Adaptive Contrastive Decoding.

        At each token step, α is determined by KL divergence between
        z_temp and z_base logit distributions:

          α_t = clamp(KL(p_temp || p_base), 0, 1)
          logits_final = (1 + α_t) * logits_temp - α_t * logits_base

        When KL is large (z_temp makes a different prediction due to temporal info),
        the contrastive effect is strong — amplifying temporal contribution.
        When KL ≈ 0 (both agree), logits_final ≈ logits_temp — no interference.

        If alpha > 0 is passed explicitly, uses fixed alpha instead of adaptive.
        """
        embeds_temp = self._prepare_embeds(input_ids, feat_temp)
        embeds_base = self._prepare_embeds(input_ids, feat_base)
        adaptive = (alpha <= 0)  # alpha=0 or negative → use adaptive KL-based alpha

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        generated_tokens = []
        past_kv_temp = None
        past_kv_base = None
        cur_embeds_temp = embeds_temp
        cur_embeds_base = embeds_base
        cur_mask = attention_mask

        for _ in range(max_new_tokens):
            out_temp = self._hf_model.model.language_model(
                inputs_embeds=cur_embeds_temp, attention_mask=cur_mask,
                past_key_values=past_kv_temp, use_cache=True, return_dict=True)
            out_base = self._hf_model.model.language_model(
                inputs_embeds=cur_embeds_base, attention_mask=cur_mask,
                past_key_values=past_kv_base, use_cache=True, return_dict=True)

            past_kv_temp = out_temp.past_key_values
            past_kv_base = out_base.past_key_values

            logits_temp = self._hf_model.lm_head(out_temp.last_hidden_state[:, -1:, :]).float()
            logits_base = self._hf_model.lm_head(out_base.last_hidden_state[:, -1:, :]).float()

            if adaptive:
                # Adaptive α: KL(p_temp || p_base) measures how much temporal info
                # changes the prediction. Larger KL → stronger contrastive effect.
                p_temp = torch.nn.functional.softmax(logits_temp, dim=-1)
                p_base = torch.nn.functional.softmax(logits_base, dim=-1)
                kl = torch.nn.functional.kl_div(
                    p_base.log().clamp(min=-100), p_temp,
                    reduction='batchmean', log_target=False)
                alpha_t = kl.clamp(0, 1).item()
            else:
                alpha_t = alpha

            logits_final = (1.0 + alpha_t) * logits_temp - alpha_t * logits_base

            next_token = logits_final.argmax(dim=-1)
            token_id = next_token.item()

            if token_id == self._hf_processor.tokenizer.eos_token_id:
                break
            generated_tokens.append(token_id)

            cur_embeds_temp = self._hf_model.model.get_input_embeddings()(next_token)
            cur_embeds_base = cur_embeds_temp
            cur_mask = torch.cat([
                cur_mask, torch.ones((1, 1), dtype=torch.long, device=cur_mask.device)
            ], dim=1)

        return self._hf_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _generate_greedy(self, input_ids, pixel_values_videos, max_new_tokens=256):
        cfg = self._adapter_cfg or {}
        contrastive_alpha = cfg.get("contrastive_alpha", 0.0)
        contrastive_adaptive = cfg.get("contrastive_adaptive", False)
        dual_select = cfg.get("dual_select", False)

        if pixel_values_videos is not None and self._adapter is not None:
            if contrastive_adaptive or contrastive_alpha > 0:
                # ── Contrastive Decoding (VCD-style) ──
                with torch.no_grad():
                    feat_temp = self._encode_video(pixel_values_videos, input_ids=input_ids)
                    feat_base = self._get_base_features(pixel_values_videos)
                # alpha=0 triggers adaptive KL-based mode
                ca = 0.0 if contrastive_adaptive else contrastive_alpha
                return self._generate_contrastive(
                    input_ids, feat_temp, feat_base, ca, max_new_tokens)

            elif dual_select:
                # ── Dual-inference selection ──
                with torch.no_grad():
                    feat_temp = self._encode_video(pixel_values_videos, input_ids=input_ids)
                    feat_base = self._get_base_features(pixel_values_videos)

                gen_temp, logp_temp = self._generate_greedy_with_features(
                    input_ids, feat_temp, max_new_tokens)
                gen_base, logp_base = self._generate_greedy_with_features(
                    input_ids, feat_base, max_new_tokens)

                if gen_temp.strip() == gen_base.strip():
                    return gen_temp
                return gen_temp if logp_temp >= logp_base else gen_base

        # ── Standard single-pass generation ──
        if pixel_values_videos is not None:
            video_features = self._encode_video(pixel_values_videos, input_ids=input_ids)
        else:
            video_features = None
        text, _ = self._generate_greedy_with_features(
            input_ids, video_features, max_new_tokens)
        return text

    def _log_progress(self, task_name, response, ground_truth, current, total):
        gt_letter = ""
        if ground_truth and "candidates" in ground_truth:
            gt_text = ground_truth.get("answer", "")
            candidates = ground_truth["candidates"]
            for i, c in enumerate(candidates):
                if c.strip() == gt_text.strip():
                    gt_letter = chr(65 + i)
                    break
        is_correct = _mcq_acc_match(response, gt_letter) if gt_letter else False
        entry = {
            "type": "sample",
            "current": current,
            "total": total,
            "task": task_name,
            "response": response[:200],
            "ground_truth": str(ground_truth.get("answer", "") if isinstance(ground_truth, dict) else ground_truth)[:200],
            "correct": is_correct,
            "timestamp": time.time(),
            "rank": self._rank,
        }
        with open(self._progress_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

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
                    desc="Model Responding")
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
                        device=self._device, dtype=torch.bfloat16)

                gen = self._generate_greedy(
                    input_ids, pixel_values_videos,
                    max_new_tokens=max_new_tokens)

                if gen.endswith("."):
                    gen = gen[:-1]

                res.append(gen)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), gen)
                pbar.update(1)

                task_name = task
                doc = self.task_dict[task][split][doc_id[i]]
                self._log_progress(task_name, gen, doc,
                                   len(res), len(requests))

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError
