import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch.distributed as dist

from core.distributed import DistributedArgs, get_is_master

logger = logging.getLogger()


def _build_default_deepspeed_config(
    distributed_args: DistributedArgs,
    micro_batch_size: int,
    grad_acc_steps: int,
    grad_clip: float,
) -> Dict[str, Any]:
    zero_opt: Dict[str, Any] = {
        "stage": int(distributed_args.deepspeed_zero_stage),
        "contiguous_gradients": True,
        "overlap_comm": True,
        "allgather_partitions": True,
        "reduce_scatter": True,
    }

    if distributed_args.deepspeed_offload_optimizer != "none":
        zero_opt["offload_optimizer"] = {
            "device": distributed_args.deepspeed_offload_optimizer,
            "pin_memory": distributed_args.deepspeed_offload_optimizer == "cpu",
        }

    if distributed_args.deepspeed_offload_param != "none":
        zero_opt["offload_param"] = {
            "device": distributed_args.deepspeed_offload_param,
            "pin_memory": distributed_args.deepspeed_offload_param == "cpu",
        }

    if int(distributed_args.deepspeed_zero_stage) >= 3:
        zero_opt.update(
            {
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
            }
        )

    model_dtype = distributed_args.model_dtype.lower()
    return {
        "train_micro_batch_size_per_gpu": int(micro_batch_size),
        "gradient_accumulation_steps": int(grad_acc_steps),
        "gradient_clipping": float(grad_clip),
        "steps_per_print": int(distributed_args.deepspeed_steps_per_print),
        "zero_optimization": zero_opt,
        "bf16": {"enabled": model_dtype == "bf16"},
        "fp16": {"enabled": model_dtype == "fp16"},
    }


def resolve_deepspeed_config(
    distributed_args: DistributedArgs,
    micro_batch_size: int,
    grad_acc_steps: int,
    grad_clip: float,
) -> Dict[str, Any]:
    if distributed_args.deepspeed_config is None:
        return _build_default_deepspeed_config(
            distributed_args=distributed_args,
            micro_batch_size=micro_batch_size,
            grad_acc_steps=grad_acc_steps,
            grad_clip=grad_clip,
        )

    config_path = Path(distributed_args.deepspeed_config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"DeepSpeed config not found: {distributed_args.deepspeed_config}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    config.setdefault("train_micro_batch_size_per_gpu", int(micro_batch_size))
    config.setdefault("gradient_accumulation_steps", int(grad_acc_steps))
    config.setdefault("gradient_clipping", float(grad_clip))
    config.setdefault("steps_per_print", int(distributed_args.deepspeed_steps_per_print))
    return config


def initialize_deepspeed_engine(
    model,
    optimizer,
    scheduler,
    distributed_args: DistributedArgs,
    micro_batch_size: int,
    grad_acc_steps: int,
    grad_clip: float,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    try:
        import deepspeed
    except ImportError as e:
        raise ImportError(
            "DeepSpeed backend requested but `deepspeed` is not installed. "
            "Install it with `pip install deepspeed`."
        ) from e

    ds_config = resolve_deepspeed_config(
        distributed_args=distributed_args,
        micro_batch_size=micro_batch_size,
        grad_acc_steps=grad_acc_steps,
        grad_clip=grad_clip,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=trainable_params,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
        dist_init_required=False,
    )
    return engine, optimizer, scheduler, ds_config


def save_deepspeed_checkpoint(engine, checkpoint_path: str, train_state) -> None:
    path = Path(checkpoint_path)
    if get_is_master():
        path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    tag = f"{train_state.step:010d}"
    client_state = {"train_state": train_state.state_dict()}
    engine.save_checkpoint(str(path), tag=tag, client_state=client_state)


def load_deepspeed_checkpoint(
    engine,
    checkpoint_path: Optional[str],
    train_state,
) -> Optional[str]:
    if checkpoint_path is None:
        return None

    path = Path(checkpoint_path)
    if not path.exists():
        return None

    load_path, client_state = engine.load_checkpoint(str(path))
    if load_path is not None and client_state is not None:
        state_dict = client_state.get("train_state")
        if state_dict is not None:
            train_state.load_state_dict(state_dict)
            logger.info(f"DeepSpeed train_state restored from {load_path}")
    return load_path


def get_optimizer_lr(optimizer=None, engine=None) -> float:
    if engine is not None:
        try:
            lrs = engine.get_lr()
            if isinstance(lrs, (list, tuple)) and len(lrs) > 0:
                return float(lrs[0])
        except Exception:
            pass

        if hasattr(engine, "optimizer"):
            optimizer = engine.optimizer

    if optimizer is not None and hasattr(optimizer, "param_groups"):
        if len(optimizer.param_groups) > 0:
            return float(optimizer.param_groups[0]["lr"])
    return 0.0


def get_deepspeed_grad_norm(engine) -> float:
    for name in ("get_global_grad_norm", "get_grad_norm"):
        fn = getattr(engine, name, None)
        if callable(fn):
            try:
                grad_norm = fn()
                if isinstance(grad_norm, (list, tuple)):
                    if len(grad_norm) > 0:
                        return float(grad_norm[0])
                    return -1.0
                return float(grad_norm)
            except Exception:
                continue
    return -1.0
