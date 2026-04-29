# Copyright (c) Meta Platforms, Inc. and affiliates.
# LoRA (Low-Rank Adaptation) module for efficient LLM fine-tuning.
#
# Implements LoRA adapters that wrap existing nn.Linear modules,
# adding trainable low-rank matrices while keeping base weights frozen.

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    LoRA wrapper around an existing nn.Linear.

    Adds low-rank adaptation: y = W_orig(x) + (B @ A)(x) * scaling
    where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, scaling = alpha / rank.

    The original weight W_orig is frozen; only A and B are trained.
    B is zero-initialized so the adapter starts as a no-op.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Keep original frozen weight (not as a sub-module to avoid param duplication)
        self.original = original

        # Low-rank adapter matrices (same device as original)
        self.lora_A = nn.Linear(
            self.in_features, rank, bias=False,
            dtype=original.weight.dtype,
            device=original.weight.device,
        )
        self.lora_B = nn.Linear(
            rank, self.out_features, bias=False,
            dtype=original.weight.dtype,
            device=original.weight.device,
        )

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros (starts as no-op)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward (frozen)
        result = self.original(x)
        # LoRA path
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return result + lora_out * self.scaling

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into original linear for inference."""
        merged = nn.Linear(
            self.in_features, self.out_features,
            bias=self.original.bias is not None,
            dtype=self.original.weight.dtype,
        )
        # W_merged = W_orig + scaling * B @ A
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        merged.weight.data = self.original.weight.data + delta.to(self.original.weight.dtype)
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data.clone()
        return merged


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> int:
    """
    Replace target nn.Linear modules with LoRALinear wrappers.

    Args:
        model: The model to modify (in-place).
        target_modules: List of module attribute names to wrap, e.g. ["wq", "wv"].
        rank: LoRA rank.
        alpha: LoRA alpha (scaling = alpha / rank).
        dropout: Dropout probability for LoRA path.

    Returns:
        Number of LoRA adapters applied.
    """
    count = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                original = getattr(module, target)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(
                        original, rank=rank, alpha=alpha, dropout=dropout,
                    )
                    setattr(module, target, lora_layer)
                    count += 1
                    logger.debug(f"Applied LoRA to {name}.{target}: "
                                 f"{original.in_features}→{rank}→{original.out_features}")

    logger.info(f"Applied {count} LoRA adapters (rank={rank}, alpha={alpha}, "
                f"dropout={dropout}) to targets: {target_modules}")
    return count


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return only LoRA adapter parameters (lora_A, lora_B)."""
    params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            params.append(param)
    return params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract LoRA weights from model state dict."""
    lora_sd = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_sd[name] = param.data.cpu()
    return lora_sd


def save_lora_weights(
    model: nn.Module,
    path: str,
    extra_metadata: Optional[Dict] = None,
):
    """Save LoRA adapter weights to a file."""
    lora_sd = get_lora_state_dict(model)

    output = {
        "lora_state_dict": lora_sd,
        "num_adapters": len(lora_sd) // 2,
    }
    if extra_metadata:
        output.update(extra_metadata)

    torch.save(output, path)
    logger.info(f"Saved {len(lora_sd)} LoRA tensors to {path}")


def load_lora_weights(model: nn.Module, path: str) -> int:
    """
    Load LoRA adapter weights into model.

    Args:
        model: Model with LoRA layers already applied.
        path: Path to saved LoRA weights.

    Returns:
        Number of loaded tensors.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    lora_sd = checkpoint["lora_state_dict"]

    model_sd = dict(model.named_parameters())
    loaded = 0
    for key, value in lora_sd.items():
        if key in model_sd:
            model_sd[key].data.copy_(value.to(model_sd[key].dtype))
            loaded += 1
        else:
            logger.warning(f"LoRA key not found in model: {key}")

    logger.info(f"Loaded {loaded}/{len(lora_sd)} LoRA tensors from {path}")
    return loaded


def merge_lora_into_model(model: nn.Module) -> int:
    """
    Merge all LoRA adapters into base weights for inference.
    Replaces LoRALinear modules with plain nn.Linear.

    Returns:
        Number of merged adapters.
    """
    count = 0
    # Collect all (parent, attr_name, lora_module) tuples first to avoid
    # modifying the module tree during iteration.
    replacements = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                replacements.append((module, child_name, child))

    for parent, attr_name, lora_module in replacements:
        merged = lora_module.merge_weights()
        setattr(parent, attr_name, merged)
        count += 1
        logger.debug(f"Merged LoRA in {attr_name}")

    logger.info(f"Merged {count} LoRA adapters into base weights")
    return count


def count_lora_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters related to LoRA.

    Returns:
        (lora_trainable, total_trainable) parameter counts.
    """
    lora_params = 0
    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
            if "lora_A" in name or "lora_B" in name:
                lora_params += param.numel()
    return lora_params, total_trainable
