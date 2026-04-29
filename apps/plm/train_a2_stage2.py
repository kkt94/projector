# Copyright (c) Meta Platforms, Inc. and affiliates.
# A2/A3 Stage 2: LoRA SFT Training Script
#
# Stage 1 trained projector-only (LLM frozen).
# Stage 2 fine-tunes LLM via LoRA + continues projector training.
#
# Usage:
#   torchrun --nproc-per-node 8 -m apps.plm.train_a2_stage2 \
#       config=apps/plm/configs/stage_2_a2/a3_v1_qwen7b_sft.yaml

import gc
import logging
import os
import sys
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import torch
import torch.distributed
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.distributed._tensor import DTensor
from torch.optim import lr_scheduler

# A2-specific components
from apps.plm.transformer_a2 import (
    A2LMTransformer,
    A2TransformerArgs,
    build_fsdp_grouping_plan,
    compute_num_visual_tokens,
    get_no_recompute_ops,
    tp_parallelize,
)

# Shared components
from apps.plm.dataset_conf import dataset_config as DATASET_CONFIGS
from apps.plm.tokenizer import build_tokenizer
from apps.plm.train import (
    TrainState,
    every_n_steps,
)

# Reuse utilities from CRT/A2 training
from apps.plm.train_crt import (
    ensure_model_downloaded,
    load_huggingface_checkpoint,
    set_preemption_flag,
)

# LoRA
from core.lora import (
    apply_lora_to_model,
    count_lora_params,
    get_lora_state_dict,
    save_lora_weights,
)

from core.args import dataclass_from_dict, dump_config
from core.checkpoint import (
    CheckpointArgs,
    CheckpointManager,
)
from core.data.dataloader import DataloadArgs, get_dataloader
from core.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    check_model_value_range,
    clean_env,
    get_device_mesh,
    get_is_master,
    get_world_size,
    init_signal_handler,
    is_deepspeed_enabled,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
)
from core.deepspeed_utils import (
    get_deepspeed_grad_norm,
    get_optimizer_lr,
    initialize_deepspeed_engine,
    load_deepspeed_checkpoint,
    save_deepspeed_checkpoint,
)
from core.logger import init_logger
from core.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
    log_model_params,
)
from core.optim import OptimArgs, build_optimizer
from core.profiling import ProfilerArgs, maybe_run_profiler

logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger()

preemption_flag = dict(flag=False)


def _apply_liger_kernel():
    """Monkey-patch RMSNorm and cross_entropy with Liger Kernel fused ops."""
    try:
        from liger_kernel.transformers import LigerRMSNorm
        from liger_kernel.transformers.cross_entropy import (
            LigerCrossEntropyFunction,
        )

        import core.transformer as _transformer_mod

        # Replace RMSNorm with Liger's fused version
        _transformer_mod.RMSNorm = LigerRMSNorm
        logger.info("Liger Kernel: RMSNorm → LigerRMSNorm")

        # Replace cross_entropy with Liger's fused version
        _orig_ce = _transformer_mod.cross_entropy

        def _liger_cross_entropy(pred, target, **kwargs):
            logits_flat = pred.flatten(end_dim=-2).float()
            target_flat = target.flatten(end_dim=-1)
            return LigerCrossEntropyFunction.apply(
                logits_flat, target_flat
            )

        _transformer_mod.cross_entropy = _liger_cross_entropy
        logger.info("Liger Kernel: cross_entropy → LigerCrossEntropyFunction")

    except ImportError:
        logger.warning("liger-kernel not installed, skipping optimizations")


def _is_oom_error(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cublas_status_alloc_failed" in msg
    )


def _recover_from_oom(model, optimizer=None, use_deepspeed: bool = False) -> None:
    try:
        if use_deepspeed and hasattr(model, "zero_grad"):
            model.zero_grad()
        elif optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        elif hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)
    except Exception:
        pass

    torch.cuda.empty_cache()
    gc.collect()


def _should_skip_batch_low_memory(
    min_free_gb: float = 1.5,
) -> bool:
    """FSDP-safe pre-flight memory check.

    All ranks check free GPU memory BEFORE forward pass.
    If ANY rank is below threshold, ALL ranks skip the batch.
    This prevents desync that happens when one rank OOMs mid-collective.
    """
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    low_mem = torch.tensor(
        [1.0 if free_gb < min_free_gb else 0.0], device="cuda"
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(low_mem, op=torch.distributed.ReduceOp.MAX)
    return low_mem.item() > 0


@dataclass
class Stage2TrainArgs:
    """Training arguments for Stage 2 LoRA SFT."""
    name: str = "a3_stage2_sft"
    dump_dir: str = ""

    seed: int = 42
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None
    steps: int = 20000
    max_iterations: Optional[int] = None
    oom_skip_batches: bool = True
    oom_max_consecutive: int = 20
    # Optional early stopping on smoothed training loss.
    # Disabled when patience is None or <= 0.
    early_stop_patience: Optional[int] = None  # number of optimizer steps without improvement
    early_stop_min_delta: float = 0.0          # minimum decrease to count as improvement
    early_stop_warmup_steps: int = 0           # do not evaluate early-stop before this step
    early_stop_window: int = 100               # moving-average window for loss smoothing

    # Stage 1 bridge weights path
    bridge_weights_path: Optional[str] = None

    data: DataloadArgs = field(default_factory=DataloadArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: A2TransformerArgs = field(default_factory=A2TransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


def _load_bridge_weights(model: A2LMTransformer, bridge_path: str):
    """Load Stage 1 projector weights into the model."""
    logger.info(f"Loading Stage 1 bridge weights from {bridge_path}")
    ckpt = torch.load(bridge_path, map_location="cpu", weights_only=False)
    bridge_sd = ckpt["bridge_state_dict"]

    # Convert DTensor (from FSDP checkpoint) to plain tensor
    for key in list(bridge_sd.keys()):
        v = bridge_sd[key]
        if hasattr(v, '_local_tensor'):
            bridge_sd[key] = v._local_tensor.detach().clone()

    projector = model.vision_projector
    proj_sd = projector.state_dict()

    loaded, skipped = 0, 0
    for key, value in bridge_sd.items():
        if key in proj_sd:
            if proj_sd[key].shape == value.shape:
                proj_sd[key].copy_(value.to(proj_sd[key].dtype))
                loaded += 1
            else:
                logger.warning(
                    f"Shape mismatch for {key}: "
                    f"model={proj_sd[key].shape}, ckpt={value.shape}"
                )
                skipped += 1
        else:
            logger.warning(f"Key not found in projector: {key}")
            skipped += 1

    total = loaded + skipped
    load_ratio = loaded / total if total > 0 else 0.0
    logger.info(
        f"Loaded {loaded}/{total} bridge weight tensors "
        f"(bridge_type={ckpt.get('bridge_type', 'unknown')}, "
        f"trained_steps={ckpt.get('steps', '?')})"
    )

    # Fail fast on severe mismatch (commonly caused by sharded/local bridge export).
    if load_ratio < 0.9:
        raise RuntimeError(
            f"Bridge load mismatch is too large: loaded {loaded}/{total} ({load_ratio:.1%}). "
            "This usually means the stage1 bridge was exported as sharded local tensors. "
            "Re-export stage1 bridge using full_state_dict and retry stage2."
        )


def train_stage2(args: Stage2TrainArgs):
    """Main training loop for Stage 2 LoRA SFT."""

    with ExitStack() as context_stack:
        # ── Setup distributed ───────────────────────────────────────────
        use_deepspeed = is_deepspeed_enabled(args.distributed)
        world_mesh = None
        if use_deepspeed:
            dp_degree = get_world_size()
            dp_rank = (
                torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            )
            logger.info(
                f"DeepSpeed backend enabled (world_size={dp_degree}, rank={dp_rank})"
            )
        else:
            world_mesh = get_device_mesh(args.distributed)
            dp_mesh = world_mesh["dp_replicate"]
            dp_degree = dp_mesh.size()
            dp_rank = dp_mesh.get_local_rank()
            if args.distributed.dp_shard > 1:
                dp_rank = dp_rank * dp_degree + world_mesh["dp_shard"].get_local_rank()
                dp_degree *= world_mesh["dp_shard"].size()

        # ── Ensure models are downloaded ────────────────────────────────
        if get_is_master():
            if args.data.tokenizer_name == "qwen":
                args.data.tokenizer_path = ensure_model_downloaded(args.data.tokenizer_path)
            if args.checkpoint.init_ckpt_path and not args.checkpoint.is_consolidated_model:
                args.checkpoint.init_ckpt_path = ensure_model_downloaded(args.checkpoint.init_ckpt_path)
            vlm_base_path = getattr(args.model, "vlm_base_path", None)
            if vlm_base_path:
                args.model.vlm_base_path = ensure_model_downloaded(vlm_base_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # ── Build tokenizer ─────────────────────────────────────────────
        num_visual_tokens = compute_num_visual_tokens(args.model)
        logger.info(f"Visual tokens per video ({args.model.projector_type}): {num_visual_tokens}")

        tokenizer = build_tokenizer(
            name=args.data.tokenizer_name,
            path=args.data.tokenizer_path,
            seq_len=args.model.max_seqlen,
            conversation_format=args.data.conversation_format,
            num_visual_tokens=num_visual_tokens,
        )

        # ── Apply Liger Kernel optimizations ─────────────────────────────
        _apply_liger_kernel()

        # ── Build model ─────────────────────────────────────────────────
        logger.info("Building model...")
        model = A2LMTransformer(args.model)
        model_param_count = get_num_params(model)

        # Initialize V-JEPA2 grid encoder + projector
        device = f"cuda:{torch.cuda.current_device()}"
        model.init_vision_model(device=device)

        # ── Load LLM checkpoint (base weights) ──────────────────────────
        vlm_base_path = getattr(args.model, "vlm_base_path", None)
        vlm_load_llm = getattr(args.model, "vlm_load_llm", True)

        if vlm_base_path and vlm_load_llm:
            logger.info(f"Loading base LLM from VLM: {vlm_base_path}")
            model.rope_embeddings.reset_parameters()
            load_huggingface_checkpoint(
                model, vlm_base_path, strip_language_model_prefix=True
            )
        elif args.checkpoint.init_ckpt_path:
            logger.info(f"Loading base LLM from {args.checkpoint.init_ckpt_path}")
            model.rope_embeddings.reset_parameters()
            if args.checkpoint.is_consolidated_model:
                from core.checkpoint import load_consolidated_checkpoint
                load_consolidated_checkpoint(
                    model=model,
                    consolidated_path=args.checkpoint.init_ckpt_path,
                    vision_model_path=None,
                )
            else:
                load_huggingface_checkpoint(model, args.checkpoint.init_ckpt_path)

        # ── Load Stage 1 bridge weights (projector) ─────────────────────
        if args.bridge_weights_path:
            _load_bridge_weights(model, args.bridge_weights_path)
        else:
            logger.warning(
                "No bridge_weights_path specified! "
                "Projector will use random initialization."
            )

        # ── Apply LoRA to LLM ───────────────────────────────────────────
        lora_rank = args.model.lora_rank
        if lora_rank > 0:
            lora_targets = [t.strip() for t in args.model.lora_targets.split(",")]
            num_adapters = apply_lora_to_model(
                model,
                target_modules=lora_targets,
                rank=lora_rank,
                alpha=args.model.lora_alpha,
                dropout=args.model.lora_dropout,
            )
            lora_params, _ = count_lora_params(model)
            logger.info(f"LoRA: {num_adapters} adapters, {lora_params:,} params")
        else:
            logger.warning("LoRA disabled (lora_rank=0). Only projector will be trained.")

        # ── Parallelize model ───────────────────────────────────────────
        if use_deepspeed:
            model = model.cuda()
        else:
            model = parallelize_model(
                model,
                world_mesh,
                args.model,
                args.distributed,
                fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
                tp_parallelize=tp_parallelize,
                no_recompute_ops=get_no_recompute_ops(),
            )

        # ── Freeze/unfreeze logic ───────────────────────────────────────
        # model.train() sets: vision_model frozen, projector trainable,
        # LLM frozen (freeze_language_model=True)
        model.train()

        # Re-enable LoRA params that were frozen by model.train()
        if lora_rank > 0:
            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = True

        check_model_value_range(model, range=45.0, std=5.0)

        # ── Log parameter stats ─────────────────────────────────────────
        logger.info(f"Model size: {model_param_count:,} total parameters")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info(
            f"Trainable: {trainable_params:,} | Frozen: {frozen_params:,} "
            f"({trainable_params / (trainable_params + frozen_params) * 100:.2f}% trainable)"
        )

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )

        # ── Calculate effective steps ───────────────────────────────────
        effective_steps = args.steps
        if args.max_iterations:
            world_size = dp_degree
            samples_per_step = args.data.batch_size * world_size

            dataset_size = 0
            pe_video_train_count = None
            for ds_name in args.data.datamix.split(","):
                ds_key = ds_name.split(":")[0].strip()
                if ds_key in DATASET_CONFIGS:
                    ann_path = DATASET_CONFIGS[ds_key].annotation
                    if ann_path and os.path.isfile(ann_path):
                        with open(ann_path, "r") as f:
                            ds_lines = sum(1 for _ in f)
                        dataset_size += ds_lines
                        if ds_key == "pe_video_train":
                            pe_video_train_count = ds_lines
            if pe_video_train_count is not None and pe_video_train_count < 500000:
                logger.warning(
                    "pe_video_train has %d samples. This is much smaller than 1M-scale "
                    "pretraining and can reduce baseline parity.",
                    pe_video_train_count,
                )
            if dataset_size == 0:
                dataset_size = 100000
                logger.warning("Could not count dataset size, using fallback=100000")

            steps_per_iteration = dataset_size // samples_per_step
            effective_steps = min(args.steps, steps_per_iteration * args.max_iterations)
            logger.info(
                f"Dataset size={dataset_size}, samples_per_step={samples_per_step}, "
                f"max_iterations={args.max_iterations}, "
                f"effective_steps={effective_steps}"
            )

        # ── Build optimizer ─────────────────────────────────────────────
        optimizer, scheduler = build_optimizer(model, args.optim, effective_steps)

        # ── Build dataloader ────────────────────────────────────────────
        dataloader = get_dataloader(
            args.data,
            dp_rank=dp_rank,
            dp_world_size=dp_degree,
            dataset_configs=DATASET_CONFIGS,
            tokenizer=tokenizer,
        )

        # ── Training state ──────────────────────────────────────────────
        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state={},
            scheduler=scheduler,
        )

        deepspeed_engine = None
        checkpoint = None
        if use_deepspeed:
            deepspeed_engine, optimizer, scheduler, ds_config = initialize_deepspeed_engine(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                distributed_args=args.distributed,
                micro_batch_size=args.data.batch_size,
                grad_acc_steps=args.grad_acc_steps,
                grad_clip=args.optim.clip,
            )
            model = deepspeed_engine
            train_state.scheduler = scheduler

            loaded_ckpt = load_deepspeed_checkpoint(
                deepspeed_engine, args.checkpoint.path, train_state
            )
            if loaded_ckpt is not None:
                logger.info(f"Loaded DeepSpeed checkpoint: {loaded_ckpt}")
            logger.info(
                f"DeepSpeed initialized (ZeRO stage {ds_config.get('zero_optimization', {}).get('stage', 'n/a')})"
            )
        else:
            checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
            checkpoint.load(model, optimizer, train_state, world_mesh)

        gc.disable()
        log_model_params(model.module if use_deepspeed else model)

        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        data_iter = iter(dataloader)
        profile_target = model.module if use_deepspeed else model
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, profile_target, args.profiling)
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()

        run_name = f"Stage2-LoRA-{args.model.projector_type.upper()}"
        logger.info(
            f"Starting {run_name}: {effective_steps} steps, "
            f"lr={args.optim.lr}, bs={args.data.batch_size}, "
            f"lora_rank={lora_rank}"
        )

        # Optional early stopping state.
        early_stop_enabled = (
            args.early_stop_patience is not None and args.early_stop_patience > 0
        )
        loss_ma_window = max(1, int(args.early_stop_window))
        loss_window = deque(maxlen=loss_ma_window)
        best_smoothed_loss = float("inf")
        no_improve_steps = 0
        if early_stop_enabled:
            logger.info(
                "Early stopping enabled: "
                f"patience={args.early_stop_patience}, "
                f"min_delta={args.early_stop_min_delta}, "
                f"warmup_steps={args.early_stop_warmup_steps}, "
                f"window={loss_ma_window}"
            )

        # ── Training loop ───────────────────────────────────────────────
        is_master = get_is_master()
        pbar = tqdm(
            total=effective_steps,
            initial=train_state.step,
            desc=run_name,
            disable=not is_master,
            ncols=100,
        )
        oom_total = 0
        oom_consecutive = 0

        while train_state.step < effective_steps:

            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            curr_lr = get_optimizer_lr(optimizer, deepspeed_engine)
            data_load_start = timer()
            try:
                batch = next(data_iter)
            except StopIteration:
                logger.info("Dataloader exhausted, restarting for next epoch...")
                data_iter = iter(dataloader)
                batch = next(data_iter)
            data_load_time = round(timer() - data_load_start, 4)
            train_state.data_loader_state = dataloader.state_dict()

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                gc.collect()

            # FSDP-safe pre-flight: skip batch if ANY rank is low on memory.
            # This runs BEFORE forward so all ranks are still in sync.
            if _should_skip_batch_low_memory(min_free_gb=1.5):
                oom_total += 1
                oom_consecutive += 1
                logger.warning(
                    "Low GPU memory detected across ranks. Skipping batch "
                    "(global_step=%d, total_skips=%d, consecutive_skips=%d).",
                    train_state.step, oom_total, oom_consecutive,
                )
                torch.cuda.empty_cache()
                gc.collect()
                train_state.acc_step = 0
                if oom_consecutive >= args.oom_max_consecutive:
                    raise RuntimeError(
                        f"Exceeded oom_max_consecutive={args.oom_max_consecutive}. "
                        "Lower batch_size or max_video_frames."
                    )
                continue

            try:
                # Get batch data
                input_ids = batch.x.cuda()
                labels = batch.y.cuda()
                mask = batch.mask.cuda()

                videos = getattr(batch, 'videos', None)
                images = getattr(batch, 'images', None)

                if videos is not None:
                    videos = videos.cuda()
                    video_pos_index = batch.video_pos_index.cuda()
                elif images is not None:
                    images = images.cuda()
                    video_pos_index = batch.image_pos_index.cuda()
                else:
                    video_pos_index = None

                num_chunks = getattr(batch, 'num_image_chunks', [1])
                media_type = getattr(batch, 'media_type', ['video'])

                nwords_since_last_log += input_ids.numel()

                bsz, seqlen = labels.shape

                # Forward
                start_timer = torch.cuda.Event(enable_timing=True)
                end_timer = torch.cuda.Event(enable_timing=True)
                start_timer.record()

                loss = model(
                    input_ids,
                    labels,
                    videos=videos,
                    images=images,
                    video_pos_index=video_pos_index,
                    image_pos_index=video_pos_index,
                    num_chunks=num_chunks,
                    media_type=media_type,
                    loss_mask=mask,
                    attn_impl=args.model.attn_impl,
                )

                # NOTE: Projector reg loss (KD) is already added inside
                # transformer_a2.py forward(). Do NOT add it again here.

                loss = loss / args.grad_acc_steps
                if use_deepspeed:
                    model.backward(loss)
                else:
                    loss.backward()
                loss = loss.detach() * args.grad_acc_steps

                # Optimizer step
                grad_norm = -1.0
                if train_state.acc_step == 0:
                    if use_deepspeed:
                        model.step()
                        grad_norm = get_deepspeed_grad_norm(model)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=args.optim.clip, foreach=True
                        )
                        grad_norm = (
                            grad_norm.full_tensor()
                            if isinstance(grad_norm, DTensor)
                            else grad_norm
                        ).item()

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    train_state.step += 1
                    curr_lr = get_optimizer_lr(optimizer, deepspeed_engine)

                    loss_val = loss.item()
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f'{loss_val:.3f}',
                        'lr': f'{curr_lr:.1e}',
                        'gn': f'{grad_norm:.1f}',
                    })

                    # Early stopping check (master decides; then broadcast).
                    if early_stop_enabled:
                        loss_window.append(loss_val)
                        smooth_loss = sum(loss_window) / len(loss_window)

                        if train_state.step >= args.early_stop_warmup_steps:
                            if (best_smoothed_loss - smooth_loss) > args.early_stop_min_delta:
                                best_smoothed_loss = smooth_loss
                                no_improve_steps = 0
                            else:
                                no_improve_steps += 1

                        should_stop = False
                        if is_master and no_improve_steps >= args.early_stop_patience:
                            logger.info(
                                "Early stopping triggered at step "
                                f"{train_state.step}: smooth_loss={smooth_loss:.4f}, "
                                f"best={best_smoothed_loss:.4f}, "
                                f"no_improve_steps={no_improve_steps}"
                            )
                            should_stop = True

                        if torch.distributed.is_initialized():
                            stop_tensor = torch.tensor(
                                [1 if should_stop else 0], device=input_ids.device
                            )
                            torch.distributed.broadcast(stop_tensor, src=0)
                            should_stop = bool(stop_tensor.item())

                        if should_stop:
                            break

                end_timer.record()
                torch.cuda.synchronize()
                time_delta = start_timer.elapsed_time(end_timer) / 1000
                oom_consecutive = 0

            except RuntimeError as exc:
                if not args.oom_skip_batches or not _is_oom_error(exc):
                    raise
                oom_total += 1
                oom_consecutive += 1
                logger.warning(
                    "CUDA OOM detected. Skipping batch and recovering "
                    "(global_step=%d, acc_step=%d, total_oom=%d, consecutive_oom=%d).",
                    train_state.step,
                    train_state.acc_step,
                    oom_total,
                    oom_consecutive,
                )
                _recover_from_oom(model, optimizer=optimizer, use_deepspeed=use_deepspeed)
                train_state.acc_step = 0
                if oom_consecutive >= args.oom_max_consecutive:
                    raise RuntimeError(
                        f"Exceeded oom_max_consecutive={args.oom_max_consecutive}. "
                        "Lower batch_size or max_video_frames."
                    ) from exc
                continue

            # Logging
            if train_state.acc_step == 0 and every_n_steps(
                train_state, args.logging.freq, acc_step=0
            ):
                loss_val = loss.item()
                time_delta_log = timer() - time_last_log
                wps = nwords_since_last_log / time_delta_log

                nwords_since_last_log = 0
                time_last_log = timer()

                metrics = {
                    "global_step": train_state.step,
                    "step": train_state.step,
                    "loss": loss_val,
                    "lr": curr_lr,
                    "grad_norm": grad_norm,
                    "wps": wps,
                    "gpu_mem_gib": gpu_memory_monitor.get_peak_stats(),
                }
                metric_logger.log(metrics)

            # Checkpoint
            if train_state.acc_step == 0 and every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ):
                # Save lightweight stage2 weights FIRST (projector + LoRA only)
                try:
                    _save_stage2_weights(
                        model, args, train_state, use_deepspeed=use_deepspeed
                    )
                except Exception as e:
                    logger.warning(f"Failed to save stage2 weights: {e}")

                # Free memory before heavy FSDP checkpoint
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    if use_deepspeed:
                        save_deepspeed_checkpoint(
                            model, args.checkpoint.path, train_state
                        )
                    else:
                        checkpoint.save(model, optimizer, train_state, args, world_mesh)
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {e}")
                    logger.warning("Stage2 weights were saved — training continues.")

            if torch_profiler:
                torch_profiler.step()

        pbar.close()

        # ── Save lightweight weights (projector + LoRA) ──────────────────
        try:
            _save_stage2_weights(model, args, train_state, use_deepspeed=use_deepspeed)
        except Exception as e:
            logger.warning(f"Failed to save stage2 weights: {e}")

        # ── Final FSDP checkpoint ────────────────────────────────────────
        gc.collect()
        torch.cuda.empty_cache()
        try:
            if use_deepspeed:
                save_deepspeed_checkpoint(model, args.checkpoint.path, train_state)
            else:
                checkpoint.save(model, optimizer, train_state, args, world_mesh)
        except Exception as e:
            logger.warning(f"Final checkpoint save failed: {e}")

        logger.info(f"{run_name} Training completed! Final step: {train_state.step}")


def _save_stage2_weights(
    model,
    args: Stage2TrainArgs,
    train_state: TrainState,
    use_deepspeed: bool = False,
):
    """Save LoRA + projector weights for eval."""
    if use_deepspeed:
        if not get_is_master():
            return
        model_ref = model.module if hasattr(model, "module") else model
        full_sd = model_ref.state_dict()
    else:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        full_sd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )

    if not get_is_master():
        return

    # Extract vision_projector weights
    proj_prefix = "vision_projector."
    proj_state = {}
    for k, v in full_sd.items():
        if k.startswith(proj_prefix):
            proj_state[k[len(proj_prefix):]] = v.detach().cpu()

    # Extract LoRA weights
    lora_state = {}
    for k, v in full_sd.items():
        if "lora_A" in k or "lora_B" in k:
            lora_state[k] = v.detach().cpu()

    vision_dim = args.model.vision_model.get("width", 1024) if isinstance(args.model.vision_model, dict) else 1024
    output = {
        "bridge_type": args.model.projector_type,
        "bridge_state_dict": proj_state,
        "lora_state_dict": lora_state,
        "lora_rank": args.model.lora_rank,
        "lora_alpha": args.model.lora_alpha,
        "lora_targets": args.model.lora_targets,
        # A2 params
        "a2_bottleneck_dim": args.model.a2_bottleneck_dim,
        "a2_state_spatial": args.model.a2_state_spatial,
        "a2_delta_temporal": args.model.a2_delta_temporal,
        "a2_delta_spatial": args.model.a2_delta_spatial,
        "a2_use_locality_block": args.model.a2_use_locality_block,
        "a2_use_future_stream": args.model.a2_use_future_stream,
        "a2_future_temporal": args.model.a2_future_temporal,
        "a2_future_spatial": args.model.a2_future_spatial,
        "a2_align_use_convex": args.model.a2_align_use_convex,
        "a2_align_codebook_size": args.model.a2_align_codebook_size,
        "a2_align_mix_ratio": args.model.a2_align_mix_ratio,
        "a2_post_ln_scale_init": args.model.a2_post_ln_scale_init,
        # A3 params
        "a3_bottleneck_dim": args.model.a3_bottleneck_dim,
        "a3_state_spatial": args.model.a3_state_spatial,
        "a3_delta_temporal": args.model.a3_delta_temporal,
        "a3_delta_spatial": args.model.a3_delta_spatial,
        "a3_reg_weight": args.model.a3_reg_weight,
        "a3_reg_smooth_weight": args.model.a3_reg_smooth_weight,
        "a3_reg_ortho_weight": args.model.a3_reg_ortho_weight,
        "a3_adaptive_gate": args.model.a3_adaptive_gate,
        "a3_use_locality_block": args.model.a3_use_locality_block,
        "a3_delta_anchor_blend": args.model.a3_delta_anchor_blend,
        "a3_use_future_stream": args.model.a3_use_future_stream,
        "a3_future_temporal": args.model.a3_future_temporal,
        "a3_future_spatial": args.model.a3_future_spatial,
        "a3_align_use_convex": args.model.a3_align_use_convex,
        "a3_align_codebook_size": args.model.a3_align_codebook_size,
        "a3_align_mix_ratio": args.model.a3_align_mix_ratio,
        "a3_post_ln_scale_init": args.model.a3_post_ln_scale_init,
        # A4 params
        "a4_bottleneck_dim": args.model.a4_bottleneck_dim,
        "a4_num_static_tokens": args.model.a4_num_static_tokens,
        "a4_num_event_tokens": args.model.a4_num_event_tokens,
        "a4_num_global_tokens": args.model.a4_num_global_tokens,
        "a4_token_order": args.model.a4_token_order,
        "a4_reg_weight": args.model.a4_reg_weight,
        "a4_reg_smooth_weight": args.model.a4_reg_smooth_weight,
        "a4_reg_sparse_weight": args.model.a4_reg_sparse_weight,
        # BMR params
        "bmr_bottleneck_dim": args.model.bmr_bottleneck_dim,
        "bmr_total_tokens": args.model.bmr_total_tokens,
        "bmr_num_anchor_tokens": args.model.bmr_num_anchor_tokens,
        "bmr_num_recap_tokens": args.model.bmr_num_recap_tokens,
        "bmr_event_min_tokens": args.model.bmr_event_min_tokens,
        "bmr_event_max_tokens": args.model.bmr_event_max_tokens,
        "bmr_event_init_ratio": args.model.bmr_event_init_ratio,
        "bmr_router_hidden_dim": args.model.bmr_router_hidden_dim,
        "bmr_camera_suppress_init": args.model.bmr_camera_suppress_init,
        "bmr_reg_weight": args.model.bmr_reg_weight,
        "bmr_reg_smooth_weight": args.model.bmr_reg_smooth_weight,
        "bmr_reg_sparse_weight": args.model.bmr_reg_sparse_weight,
        "bmr_reg_ortho_weight": args.model.bmr_reg_ortho_weight,
        # BMRv2 params
        "bmr_v2_num_tokens": args.model.bmr_v2_num_tokens,
        "bmr_v2_lora_rank": args.model.bmr_v2_lora_rank,
        "bmr_v2_lora_alpha": args.model.bmr_v2_lora_alpha,
        "bmr_v2_til_kernel": args.model.bmr_v2_til_kernel,
        "bmr_v2_gate_mode": args.model.bmr_v2_gate_mode,
        "bmr_v2_kd_weight": args.model.bmr_v2_kd_weight,
        # TEP params
        "tep_bottleneck": getattr(args.model, "tep_bottleneck", 128),
        "tep_num_layers": getattr(args.model, "tep_num_layers", 3),
        "tep_num_heads": getattr(args.model, "tep_num_heads", 8),
        "tep_max_frames": getattr(args.model, "tep_max_frames", 32),
        "tep_use_frame_pe": getattr(args.model, "tep_use_frame_pe", True),
        "tep_num_tokens": getattr(args.model, "tep_num_tokens", 3136),
        # TDI params
        "tdi_num_tokens": getattr(args.model, "tdi_num_tokens", 3151),
        # VLM info
        "vlm_base_path": getattr(args.model, "vlm_base_path", None),
        "_tokenizer_name": getattr(args.data, "tokenizer_name", "plmchat"),
        "_conversation_format": getattr(args.data, "conversation_format", "plm_sft"),
        "_is_consolidated": getattr(args.checkpoint, "is_consolidated_model", False),
        "max_seqlen": args.model.max_seqlen,
        # Common
        "vision_dim": vision_dim,
        "llm_dim": args.model.dim,
        "steps": train_state.step,
        "stage": 2,
    }

    save_path = Path(args.dump_dir) / f"stage2_weights_step{train_state.step}.pt"
    torch.save(output, save_path)
    logger.info(f"Stage 2 weights saved to: {save_path}")
    logger.info(f"  Projector keys: {len(proj_state)}")
    logger.info(f"  LoRA keys: {len(lora_state)}")


def main():
    """Entry point for Stage 2 LoRA SFT training."""
    if len(sys.argv) < 2:
        print("Usage: torchrun -m apps.plm.train_a2_stage2 config=<config_path>")
        sys.exit(1)

    cfg = OmegaConf.from_cli()
    if "config" in cfg:
        file_cfg = OmegaConf.load(cfg.config)
        cfg = OmegaConf.merge(file_cfg, cfg)
        del cfg["config"]

    args = dataclass_from_dict(Stage2TrainArgs, OmegaConf.to_container(cfg), strict=False)
    if args.logging.wandb is not None and args.logging.wandb.name is None:
        args.logging.wandb.name = args.name

    # Setup environment
    setup_env(args.env)
    init_signal_handler(set_preemption_flag)

    os.makedirs(args.dump_dir, exist_ok=True)
    init_logger(Path(args.dump_dir) / "train.log", level=args.logging.level)

    setup_torch_distributed(args.distributed)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    if args.checkpoint.path is None:
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    if get_is_master():
        os.makedirs(args.dump_dir, exist_ok=True)
        dump_config(asdict(args), Path(args.dump_dir) / "config.yaml")

    train_stage2(args)

    clean_env()


if __name__ == "__main__":
    main()
