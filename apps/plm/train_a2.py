# Copyright (c) Meta Platforms, Inc. and affiliates.
# A2 (Dynamics-aware Projector) Training Script — Stage 1
#
# Usage:
#   torchrun --nproc-per-node 8 -m apps.plm.train_a2 \
#       config=apps/plm/configs/stage_1_a2/a2_v1_1b.yaml
#
# This script trains the Dynamics-aware A2 projector while keeping
# V-JEPA2 encoder and LLM frozen.

import gc
import logging
import math
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, Optional

import torch
import torch.distributed
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful
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

# Reuse utilities from CRT training
from apps.plm.train_crt import (
    ensure_model_downloaded,
    load_huggingface_checkpoint,
    set_preemption_flag,
)

from core.args import dataclass_from_dict, dump_config, flatten_dict
from core.checkpoint import (
    CheckpointArgs,
    CheckpointManager,
    load_consolidated_checkpoint,
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


@dataclass
class A2TrainArgs:
    """Training arguments for A2 model."""
    name: str = "a2_stage1"
    dump_dir: str = ""

    seed: int = 42
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None
    steps: int = 30000
    max_iterations: Optional[int] = None
    oom_skip_batches: bool = True
    oom_max_consecutive: int = 20

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


def train_a2(args: A2TrainArgs):
    """Main training loop for A2/A3/MLP vision projector."""

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

        # ── Build model ─────────────────────────────────────────────────
        logger.info("Building model...")
        model = A2LMTransformer(args.model)
        model_param_count = get_num_params(model)

        # Initialize V-JEPA2 grid encoder + A2 projector
        device = f"cuda:{torch.cuda.current_device()}"
        model.init_vision_model(device=device)

        # ── Load LLM checkpoint ─────────────────────────────────────────
        vlm_base_path = getattr(args.model, "vlm_base_path", None)
        vlm_load_llm = getattr(args.model, "vlm_load_llm", True)

        if vlm_base_path and vlm_load_llm:
            # Load LLM weights from VLM checkpoint (strip language_model. prefix)
            logger.info(f"Loading LLM from VLM: {vlm_base_path}")
            model.rope_embeddings.reset_parameters()
            load_huggingface_checkpoint(
                model, vlm_base_path, strip_language_model_prefix=True
            )
        elif args.checkpoint.init_ckpt_path:
            logger.info(f"Loading LLM from {args.checkpoint.init_ckpt_path}")
            model.rope_embeddings.reset_parameters()
            if args.checkpoint.is_consolidated_model:
                load_consolidated_checkpoint(
                    model=model,
                    consolidated_path=args.checkpoint.init_ckpt_path,
                    vision_model_path=None,
                )
            else:
                load_huggingface_checkpoint(model, args.checkpoint.init_ckpt_path)
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()

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
        model.train()
        check_model_value_range(model, range=45.0, std=5.0)

        logger.info(f"Model size: {model_param_count:,} total parameters")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Trainable parameters: {trainable_params:,} "
            f"({args.model.projector_type.upper()} projector)"
        )

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )

        # ── Calculate effective steps (before optimizer for correct LR schedule) ──
        effective_steps = args.steps
        if args.max_iterations:
            world_size = dp_degree
            samples_per_step = args.data.batch_size * world_size

            # Count actual dataset size from JSONL
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
                dataset_size = 100000  # fallback
                logger.warning("Could not count dataset size, using fallback=100000")

            steps_per_iteration = dataset_size // samples_per_step
            effective_steps = min(args.steps, steps_per_iteration * args.max_iterations)
            logger.info(
                f"Dataset size={dataset_size}, samples_per_step={samples_per_step}, "
                f"max_iterations={args.max_iterations}, "
                f"steps_per_iteration={steps_per_iteration}, "
                f"effective_steps={effective_steps}"
            )

        # ── Build optimizer (use effective_steps for correct LR schedule) ─
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

        run_name = args.model.projector_type.upper()
        logger.info(
            f"Starting {run_name} Training: {effective_steps} steps, "
            f"lr={args.optim.lr}, bs={args.data.batch_size}"
        )

        # ── Training loop ───────────────────────────────────────────────
        is_master = get_is_master()
        pbar = tqdm(
            total=effective_steps,
            initial=train_state.step,
            desc=f"{run_name} Training",
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

                # Add projector regularization loss (e.g. BMRv2 KD loss)
                proj = getattr(model, "vision_projector", None)
                if proj is not None and hasattr(proj, "get_reg_loss"):
                    reg_loss = proj.get_reg_loss()
                    if reg_loss.item() > 0:
                        loss = loss + reg_loss

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
                if use_deepspeed:
                    save_deepspeed_checkpoint(model, args.checkpoint.path, train_state)
                else:
                    checkpoint.save(model, optimizer, train_state, args, world_mesh)

            if torch_profiler:
                torch_profiler.step()

        pbar.close()

        # ── Final checkpoint ────────────────────────────────────────────
        if use_deepspeed:
            save_deepspeed_checkpoint(model, args.checkpoint.path, train_state)
        else:
            checkpoint.save(model, optimizer, train_state, args, world_mesh)

        # ── Save A2 projector weights separately ────────────────────────
        try:
            _save_a2_bridge_weights(model, args, train_state, use_deepspeed=use_deepspeed)
        except Exception as e:
            logger.warning(f"Failed to save bridge weights: {e}")
            import traceback
            traceback.print_exc()

        logger.info(f"{run_name} Training completed! Final step: {train_state.step}")


def _save_a2_bridge_weights(
    model,
    args: A2TrainArgs,
    train_state: TrainState,
    use_deepspeed: bool = False,
):
    """Extract A2 projector weights from model and save for eval."""
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

    # Extract vision_projector keys
    prefix = "vision_projector."
    clean_state = {}
    for k, v in full_sd.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            clean_state[new_key] = v.detach().cpu()

    if not clean_state:
        logger.warning("No vision_projector keys found in model state dict")
        return

    vision_dim = args.model.vision_model.get("width", 1024) if isinstance(args.model.vision_model, dict) else 1024
    bridge_output = {
        "bridge_type": args.model.projector_type,
        "bridge_state_dict": clean_state,
        "is_full_state_dict": True,
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
        # V-JEPA2 config
        "vjepa2_config": {
            "encoder_name": getattr(args.model.vjepa2_config, "encoder_name", "vit_huge"),
            "num_frames": getattr(args.model.vjepa2_config, "num_frames", 64),
            "native_preproc": getattr(args.model.vjepa2_config, "native_preproc", True),
            "input_is_plm_normalized": getattr(args.model.vjepa2_config, "input_is_plm_normalized", True),
        },
        # Common
        "vision_dim": vision_dim,
        "llm_dim": args.model.dim,
        "steps": train_state.step,
    }

    bridge_path = Path(args.dump_dir) / f"a2_bridge_weights_step{train_state.step}.pt"
    torch.save(bridge_output, bridge_path)
    logger.info(f"A2 bridge weights saved to: {bridge_path}")

    for k, v in clean_state.items():
        logger.info(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")


def main():
    """Entry point for A2 training."""
    if len(sys.argv) < 2:
        print("Usage: torchrun -m apps.plm.train_a2 config=<config_path>")
        sys.exit(1)

    cfg = OmegaConf.from_cli()
    if "config" in cfg:
        file_cfg = OmegaConf.load(cfg.config)
        cfg = OmegaConf.merge(file_cfg, cfg)
        del cfg["config"]

    args = dataclass_from_dict(A2TrainArgs, OmegaConf.to_container(cfg), strict=False)
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

    train_a2(args)

    clean_env()


if __name__ == "__main__":
    main()
