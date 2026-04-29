# Copyright (c) Meta Platforms, Inc. and affiliates.
# CRT (Causal Rollout Tokenization) Training Script
#
# Usage:
#   torchrun --nproc-per-node 8 -m apps.plm.train_crt \
#       config=apps/plm/configs/stage_1_crt/crt_v2_qwen7b.yaml
#
# This script trains the CRT vision projector while keeping
# V-JEPA2 encoder and LLM frozen.

import gc
import logging
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

# Import CRT-specific components
from apps.plm.transformer_crt import (
    CRTLMTransformer,
    CRTTransformerArgs,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    tp_parallelize,
)

# Import shared components from original train.py
from apps.plm.dataset_conf import dataset_config as DATASET_CONFIGS
from apps.plm.tokenizer import build_tokenizer
from apps.plm.train import (
    TrainArgs as BaseTrainArgs,
    TrainState,
    every_n_steps,
    validate_train_args,
)

from core.args import dataclass_from_dict, dump_config, flatten_dict
from core.checkpoint import (
    CheckpointArgs,
    CheckpointManager,
    load_consolidated_checkpoint,
    load_from_checkpoint,
)


def ensure_model_downloaded(model_path: str) -> str:
    """
    Ensure the model is downloaded. If path doesn't exist, try to download from HuggingFace.

    Supports patterns like:
      - ./models/Qwen2.5-1.5B -> downloads Qwen/Qwen2.5-1.5B
      - ./models/Llama-3.2-1B -> already exists (Meta format)
      - Qwen/Qwen2.5-1.5B -> downloads directly

    Returns the path to the downloaded model.
    """
    from pathlib import Path

    model_path = Path(model_path)

    # If path exists and has weight files, return as-is
    if model_path.exists():
        has_weights = (
            list(model_path.glob("*.safetensors")) or
            list(model_path.glob("pytorch_model*.bin")) or
            list(model_path.glob("consolidated*.pth")) or
            (model_path / "original").exists()
        )
        if has_weights:
            logger.info(f"Model found at {model_path}")
            return str(model_path)

    # Try to infer HuggingFace model name from path
    # e.g., ./models/Qwen2.5-1.5B -> Qwen/Qwen2.5-1.5B
    # e.g., DAMO-NLP-SG/VideoLLaMA3-2B -> used directly as repo ID
    original_str = str(model_path)
    model_name = model_path.name

    # If the original string looks like a HF repo ID (org/model), use it directly
    if "/" in original_str and not original_str.startswith((".", "/")):
        hf_repo = original_str
    else:
        # Common HuggingFace model name patterns
        hf_name_map = {
            "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
            "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
            "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
            "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
            "Qwen2-1.5B": "Qwen/Qwen2-1.5B",
            "Qwen2-7B": "Qwen/Qwen2-7B",
            "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
            "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
        }
        hf_repo = hf_name_map.get(model_name, f"Qwen/{model_name}")

    logger.info(f"Model not found at {model_path}, downloading from {hf_repo}...")

    try:
        from huggingface_hub import snapshot_download

        # Create parent directory
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download model
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded to {model_path}")
        return str(model_path)

    except Exception as e:
        logger.error(f"Failed to download model {hf_repo}: {e}")
        raise ValueError(f"Model not found at {model_path} and failed to download from {hf_repo}")


def load_huggingface_checkpoint(
    model,
    hf_model_path: str,
    strip_language_model_prefix: bool = False,
):
    """
    Load HuggingFace model weights (e.g., Qwen) into PLM model.

    Args:
        model: PLM model instance.
        hf_model_path: Path to HuggingFace model directory.
        strip_language_model_prefix: If True, strip 'language_model.' prefix from
            VLM checkpoint keys before mapping. This allows loading LLM weights
            from a full VLM checkpoint (e.g., LLaVA, InternVL) that wraps the
            LLM under a 'language_model.' namespace.
    """
    from pathlib import Path

    # Ensure model is downloaded
    hf_model_path = ensure_model_downloaded(hf_model_path)

    hf_path = Path(hf_model_path)
    logger.info(f"Loading HuggingFace model from {hf_path}")

    # Find model weights files
    weight_files = list(hf_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = list(hf_path.glob("pytorch_model*.bin"))

    if not weight_files:
        raise ValueError(f"No weight files found in {hf_path}")

    # Load weights
    hf_state_dict = {}
    for wf in weight_files:
        logger.info(f"  Loading {wf.name}...")
        if wf.suffix == ".safetensors":
            from safetensors.torch import load_file
            hf_state_dict.update(load_file(str(wf)))
        else:
            hf_state_dict.update(torch.load(wf, map_location="cpu", weights_only=False))

    # If loading from a VLM checkpoint, strip language_model. prefix and skip
    # vision/projector keys so only LLM weights are mapped.
    if strip_language_model_prefix:
        _VLM_SKIP_PREFIXES = (
            "vision_tower.", "vision_model.", "visual.",
            "multi_modal_projector.", "mlp1.",
            # VideoLLaMA3-style prefixes
            "model.vision_encoder.", "model.mm_projector.",
        )
        stripped = {}
        for k, v in hf_state_dict.items():
            if any(k.startswith(p) for p in _VLM_SKIP_PREFIXES):
                continue
            if k.startswith("language_model."):
                stripped[k[len("language_model."):]] = v
            else:
                stripped[k] = v
        logger.info(
            f"Stripped language_model prefix: {len(hf_state_dict)} -> {len(stripped)} keys"
        )
        hf_state_dict = stripped

    # Map HuggingFace keys to PLM keys
    key_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    # Layer mappings for Qwen/Llama style models
    layer_key_map = {
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        # Biases
        "self_attn.q_proj.bias": "attention.wq.bias",
        "self_attn.k_proj.bias": "attention.wk.bias",
        "self_attn.v_proj.bias": "attention.wv.bias",
        "self_attn.o_proj.bias": "attention.wo.bias",
        "mlp.gate_proj.bias": "feed_forward.w1.bias",
        "mlp.up_proj.bias": "feed_forward.w3.bias",
        "mlp.down_proj.bias": "feed_forward.w2.bias",
    }

    plm_state_dict = {}

    for hf_key, tensor in hf_state_dict.items():
        plm_key = None

        if hf_key in key_map:
            plm_key = key_map[hf_key]
        elif hf_key.startswith("model.layers."):
            parts = hf_key.split(".")
            layer_idx = parts[2]
            rest = ".".join(parts[3:])
            if rest in layer_key_map:
                plm_key = f"layers.{layer_idx}.{layer_key_map[rest]}"

        if plm_key:
            plm_state_dict[plm_key] = tensor

    logger.info(f"Mapped {len(plm_state_dict)} keys from HuggingFace to PLM format")

    # Load into model
    model_state = model.state_dict()
    loaded_count = 0
    for key, value in plm_state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_count += 1
            else:
                logger.warning(f"Shape mismatch for {key}: model={model_state[key].shape}, ckpt={value.shape}")

    model.load_state_dict(model_state, strict=False)
    logger.info(f"Loaded {loaded_count} tensors into model")
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
    requeue_slurm_job,
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

# Preemption handling
preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


@dataclass
class CRTTrainArgs:
    """Training arguments for CRT model."""
    name: str = "crt_v2"
    dump_dir: str = ""

    seed: int = 42
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None
    steps: int = 30000
    max_iterations: Optional[int] = None  # Stop after N iterations (epochs) over dataset

    data: DataloadArgs = field(default_factory=DataloadArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: CRTTransformerArgs = field(default_factory=CRTTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


def train_crt(args: CRTTrainArgs):
    """Main training loop for CRT vision projector."""

    with ExitStack() as context_stack:
        # Setup distributed
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

        # Ensure models are downloaded (only on rank 0 to avoid race conditions)
        if get_is_master():
            if args.data.tokenizer_name == "qwen":
                args.data.tokenizer_path = ensure_model_downloaded(args.data.tokenizer_path)
            if args.checkpoint.init_ckpt_path and not args.checkpoint.is_consolidated_model:
                args.checkpoint.init_ckpt_path = ensure_model_downloaded(args.checkpoint.init_ckpt_path)

        # Sync all ranks after download
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Build tokenizer (name first, then path)
        tokenizer = build_tokenizer(
            name=args.data.tokenizer_name,
            path=args.data.tokenizer_path,
            seq_len=args.model.max_seqlen,
            conversation_format=args.data.conversation_format,
        )

        # Build CRT model
        logger.info("Building CRT model...")
        model = CRTLMTransformer(args.model)
        model_param_count = get_num_params(model)

        # Initialize vision model (V-JEPA2)
        device = f"cuda:{torch.cuda.current_device()}"
        model.init_vision_model(device=device)

        # Load LLM checkpoint
        if args.checkpoint.init_ckpt_path:
            logger.info(f"Loading LLM from {args.checkpoint.init_ckpt_path}")
            model.rope_embeddings.reset_parameters()
            if args.checkpoint.is_consolidated_model:
                # Load Meta consolidated format (Llama original)
                load_consolidated_checkpoint(
                    model=model,
                    consolidated_path=args.checkpoint.init_ckpt_path,
                    vision_model_path=None,  # We use V-JEPA2, not PE
                )
            else:
                # Load HuggingFace format (Qwen, Llama-HF, etc.)
                load_huggingface_checkpoint(model, args.checkpoint.init_ckpt_path)
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()

        # Parallelize model
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

        # Log trainable params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,} (CRT projector)")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )

        # Build optimizer (only for trainable params)
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)

        # Build dataloader
        dataloader = get_dataloader(
            args.data,
            dp_rank=dp_rank,
            dp_world_size=dp_degree,
            dataset_configs=DATASET_CONFIGS,
            tokenizer=tokenizer,
        )

        # Initialize training state (data_loader_state will be set after first iteration)
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

        # Calculate effective steps limit based on max_iterations
        effective_steps = args.steps
        if args.max_iterations:
            # Calculate steps per iteration: dataset_size / (batch_size * world_size)
            # For dummy_video: 10000 / (batch_size * 8 GPUs)
            # We estimate based on typical iteration boundaries seen in logs
            world_size = dp_degree
            samples_per_step = args.data.batch_size * world_size
            # Estimate dataset size (for dummy_video it's 10000)
            estimated_dataset_size = 10000  # TODO: get actual dataset size
            steps_per_iteration = estimated_dataset_size // samples_per_step
            effective_steps = min(args.steps, steps_per_iteration * args.max_iterations)
            logger.info(f"max_iterations={args.max_iterations}, steps_per_iteration≈{steps_per_iteration}, effective_steps={effective_steps}")

        logger.info(f"Starting CRT Training: {effective_steps} steps, lr={args.optim.lr}, bs={args.data.batch_size}")

        # tqdm progress bar (only on rank 0)
        is_master = get_is_master()
        pbar = tqdm(
            total=effective_steps,
            initial=train_state.step,
            desc="Training",
            disable=not is_master,
            ncols=100,
        )

        while train_state.step < effective_steps:

            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            curr_lr = get_optimizer_lr(optimizer, deepspeed_engine)
            data_load_start = timer()
            batch = next(data_iter)
            data_load_time = round(timer() - data_load_start, 4)
            train_state.data_loader_state = dataloader.state_dict()

            # Log first batch arrival (removed verbose logging)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                gc.collect()

            # Get batch data
            input_ids = batch.x.cuda()
            labels = batch.y.cuda()
            mask = batch.mask.cuda()

            # Handle video or image input
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

                # Update progress bar
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

            # Logging (less frequent, to file/metrics only)
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

            # Checkpoint (save every N steps)
            if train_state.acc_step == 0 and every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ):
                if use_deepspeed:
                    save_deepspeed_checkpoint(model, args.checkpoint.path, train_state)
                else:
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        world_mesh,
                    )

            if torch_profiler:
                torch_profiler.step()

        # Close progress bar
        pbar.close()

        # Final checkpoint
        if use_deepspeed:
            save_deepspeed_checkpoint(model, args.checkpoint.path, train_state)
        else:
            checkpoint.save(
                model,
                optimizer,
                train_state,
                args,
                world_mesh,
            )

        # Save bridge weights separately for eval_benchmark.py
        # Use DCP to load the just-saved checkpoint and extract vision_projector weights
        if get_is_master():
            try:
                if use_deepspeed:
                    _save_crt_bridge_weights_from_model(model, args, train_state)
                else:
                    import torch.distributed.checkpoint as dcp
                    import math

                    # Create dummy state dict with correct shapes for vision_projector
                    vision_dim = args.model.vision_model.get("width", 1024)
                    llm_dim = args.model.dim

                    # Build the expected structure
                    import torch.nn as nn

                    class DummyProj(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.projector = nn.Sequential(
                                nn.Linear(vision_dim, llm_dim),
                                nn.GELU(),
                                nn.Linear(llm_dim, llm_dim),
                            )
                            self.type_cause = nn.Parameter(torch.zeros(llm_dim))
                            self.type_pred = nn.Parameter(torch.zeros(llm_dim))
                            self.type_inno = nn.Parameter(torch.zeros(llm_dim))
                            self.ln = nn.LayerNorm(llm_dim)
                            self.post_ln_scale = nn.Parameter(
                                torch.tensor([1.0 / math.sqrt(llm_dim)])
                            )

                    dummy = DummyProj()
                    load_state = {
                        "model": {
                            f"vision_projector.{k}": torch.zeros_like(v)
                            for k, v in dummy.state_dict().items()
                        }
                    }

                    # Load from checkpoint
                    ckpt_path = Path(args.checkpoint.path) / f"{train_state.step:010d}"
                    logger.info(f"Loading vision_projector from: {ckpt_path}")

                    dcp.load(
                        state_dict=load_state,
                        storage_reader=dcp.FileSystemReader(str(ckpt_path)),
                    )

                    # Extract and rename for CausalRolloutBridgeV2
                    clean_state = {}
                    prefix = "vision_projector."

                    for k, v in load_state["model"].items():
                        if k.startswith(prefix):
                            new_key = k[len(prefix):]
                            if new_key.startswith("projector."):
                                new_key = "proj." + new_key[len("projector."):]
                            if new_key == "post_ln_scale" and v.numel() == 1:
                                v = v.squeeze()
                            clean_state[new_key] = v.cpu()

                    if clean_state:
                        bridge_output = {
                            "bridge_type": "crt_v2",
                            "bridge_state_dict": clean_state,
                            "num_segments": args.model.num_segments,
                            "rollout_steps": args.model.rollout_steps,
                            "spatial_grid": args.model.spatial_grid,
                            "steps": train_state.step,
                        }

                        bridge_path = (
                            Path(args.dump_dir)
                            / f"bridge_weights_step{train_state.step}.pt"
                        )
                        torch.save(bridge_output, bridge_path)
                        logger.info(f"Bridge weights saved to: {bridge_path}")

                        for k, v in clean_state.items():
                            logger.info(
                                f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}"
                            )
                    else:
                        logger.warning("No vision_projector keys extracted")
            except Exception as e:
                logger.warning(f"Failed to save bridge weights: {e}")
                import traceback
                traceback.print_exc()

        logger.info(f"Training completed! Final step: {train_state.step}")


def _save_crt_bridge_weights_from_model(model, args: CRTTrainArgs, train_state: TrainState):
    model_ref = model.module if hasattr(model, "module") else model
    state_dict = model_ref.state_dict()

    clean_state = {}
    prefix = "vision_projector."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            if new_key.startswith("projector."):
                new_key = "proj." + new_key[len("projector."):]
            if new_key == "post_ln_scale" and v.numel() == 1:
                v = v.squeeze()
            clean_state[new_key] = v.detach().cpu()

    if not clean_state:
        logger.warning("No vision_projector keys extracted from DeepSpeed model state")
        return

    bridge_output = {
        "bridge_type": "crt_v2",
        "bridge_state_dict": clean_state,
        "num_segments": args.model.num_segments,
        "rollout_steps": args.model.rollout_steps,
        "spatial_grid": args.model.spatial_grid,
        "steps": train_state.step,
    }

    bridge_path = Path(args.dump_dir) / f"bridge_weights_step{train_state.step}.pt"
    torch.save(bridge_output, bridge_path)
    logger.info(f"Bridge weights saved to: {bridge_path}")


def main():
    """Entry point for CRT training."""
    # Parse config first to get env args
    if len(sys.argv) < 2:
        print("Usage: torchrun -m apps.plm.train_crt config=<config_path>")
        sys.exit(1)

    cfg = OmegaConf.from_cli()
    if "config" in cfg:
        file_cfg = OmegaConf.load(cfg.config)
        cfg = OmegaConf.merge(file_cfg, cfg)
        del cfg["config"]

    args = dataclass_from_dict(CRTTrainArgs, OmegaConf.to_container(cfg), strict=False)
    if args.logging.wandb is not None and args.logging.wandb.name is None:
        args.logging.wandb.name = args.name

    # Setup environment with proper args
    setup_env(args.env)
    init_signal_handler(set_preemption_flag)

    # Create dump_dir before logging
    os.makedirs(args.dump_dir, exist_ok=True)
    init_logger(Path(args.dump_dir) / "train.log", level=args.logging.level)

    # Setup distributed
    setup_torch_distributed(args.distributed)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Validate args
    if args.checkpoint.path is None:
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    # Dump config
    if get_is_master():
        os.makedirs(args.dump_dir, exist_ok=True)
        dump_config(asdict(args), Path(args.dump_dir) / "config.yaml")

    # Train
    train_crt(args)

    clean_env()


if __name__ == "__main__":
    main()
