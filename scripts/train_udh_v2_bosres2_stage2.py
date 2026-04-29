"""
Train UDH v2 Stage 2 — Branch-Output Residual Scaler.

Uses the same inference form:
  z_final = z_base + a_local(x) * local_residual + a_state(x) * state_residual

Key design:
  - Target from z_base vs z_temp (combined), not z_local/z_state individually
  - Local/state split guided by branch norm ratio
  - QA gradient does NOT flow through scaler (target-only MSE)
  - Expert: SplitDualHeadExpert (local DeltaNet 1024pos + state DeltaNet 1pos)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.train_dual_proj as v2
from core.vision_projector.dual_proj import RawContextResidualScaler, SplitDualHeadExpert
from scripts.train_temporal_expert_router_stage2 import (
    TRAIN_DATASETS,
    TOKENS_PER_FRAME,
    build_dataset,
    collate_fn,
    log_rank0,
)


SUPPORTED_STAGE1_TYPES = {
    "dual_proj_deltanet_udh_v2_stage1": "dual_proj_deltanet_udh_v2_bosres2",
}

MCQ_BENCHLIKE_DATASETS = [
    {"jsonl": "datasets/llava_nextqa_qa.jsonl", "video_root": "datasets/llava_video_plm/videos"},
    {"jsonl": "datasets/llava_perceptiontest_qa.jsonl", "video_root": "datasets/llava_video_plm/videos"},
    {"jsonl": "datasets/multitask/nextgqa/train_plm.jsonl", "video_root": "datasets/multitask/nextgqa/videos"},
    {"jsonl": "datasets/multitask/egoschema/train_plm.jsonl", "video_root": "datasets/multitask/egoschema/videos"},
]


def resolve_dataset_configs(preset):
    if preset == "full":
        return list(TRAIN_DATASETS)
    if preset == "mcq_benchlike":
        return list(MCQ_BENCHLIKE_DATASETS)
    raise ValueError(f"Unknown dataset_preset: {preset!r}")


class PLMWithUDHResidualScaler(nn.Module):
    def __init__(
        self,
        model,
        expert,
        scaler,
        alpha_tau=0.20,
        alpha_margin=0.02,
        policy_lambda=0.20,
        policy_warmup_steps=50,
        ema_momentum=0.95,
    ):
        super().__init__()
        self.model = model
        self.expert = expert
        self.scaler = scaler
        self.alpha_tau = float(alpha_tau)
        self.alpha_margin = float(alpha_margin)
        self.policy_lambda = float(policy_lambda)
        self.policy_warmup_steps = int(policy_warmup_steps)
        self.ema_momentum = float(ema_momentum)
        self.register_buffer("_ema_base", torch.tensor(0.0), persistent=False)
        self.register_buffer("_ema_gap", torch.tensor(0.0), persistent=False)
        self.register_buffer("_ema_ready", torch.tensor(False), persistent=False)

        self._last_qa_loss = 0.0
        self._last_policy_loss = 0.0
        self._last_a_local = 0.0
        self._last_a_state = 0.0
        self._last_a_local_std = 0.0
        self._last_a_state_std = 0.0
        self._last_target_alpha = 0.0
        self._last_target_local = 0.0
        self._last_target_state = 0.0
        self._last_need = 0.0
        self._last_benefit = 0.0
        self._last_local_ratio = 0.0
        self._last_delta_norm = 0.0
        self._last_grad_norm = 0.0

    def _get_vision_features(self, pixel_values_videos, num_frames):
        bsz = pixel_values_videos.shape[0]
        frames_flat = pixel_values_videos.flatten(0, 1)
        with torch.no_grad():
            vision_out = self.model.model.vision_tower(frames_flat)
            features = vision_out.last_hidden_state[:, 1:, :].float()
        _, n, c = features.shape
        return features.reshape(bsz, num_frames, n, c)

    def _build_stage1_inputs(self, pixel_values_videos, num_frames):
        vision_4d = self._get_vision_features(pixel_values_videos, num_frames)
        bsz, t, n, c = vision_4d.shape
        proj = self.model.model.multi_modal_projector
        flat = vision_4d.reshape(bsz * t, n, c)
        with torch.no_grad():
            features = flat.permute(1, 0, 2)
            features = proj.linear_1(features)
            features = proj.gelu(features)
            features = proj.linear_2(features)
            features = features.permute(1, 0, 2)
            z_pre_pool_flat = features
            z_base_flat = proj.pooling(features)
            n_pre = z_pre_pool_flat.shape[1]
            m, d = z_base_flat.shape[1], z_base_flat.shape[2]
            z_pre_pool = z_pre_pool_flat.reshape(bsz, t, n_pre, d)
            z_base = z_base_flat.reshape(bsz, t, m, d)
        return z_pre_pool, z_base

    def _get_user_text_pack(self, input_ids):
        with torch.no_grad():
            all_embeds = self.model.model.get_input_embeddings()(input_ids)
        bsz = input_ids.shape[0]
        seqs = []
        max_len = 1
        for i in range(bsz):
            seq = input_ids[i]
            eot_positions = (seq == v2.EOT_ID).nonzero(as_tuple=True)[0].tolist()
            if eot_positions:
                end = eot_positions[0]
                header_positions = (seq == v2.END_HEADER_ID).nonzero(as_tuple=True)[0].tolist()
                header_positions = [pos for pos in header_positions if pos < end]
                start = header_positions[-1] + 1 if header_positions else 0
            else:
                start, end = 0, seq.shape[0]
            tokens_i = seq[start:end]
            embeds_i = all_embeds[i, start:end]
            text_mask_i = (
                (tokens_i != v2.VIDEO_TOKEN_ID) & (tokens_i != v2.PAD_ID)
                & (tokens_i != v2.END_HEADER_ID) & (tokens_i != v2.EOT_ID)
            )
            emb = embeds_i[text_mask_i]
            if emb.numel() == 0:
                emb = torch.zeros(1, all_embeds.shape[-1], device=all_embeds.device, dtype=all_embeds.dtype)
            seqs.append(emb)
            max_len = max(max_len, emb.shape[0])
        packed = torch.zeros(bsz, max_len, all_embeds.shape[-1], device=all_embeds.device, dtype=all_embeds.dtype)
        mask = torch.zeros(bsz, max_len, device=all_embeds.device, dtype=torch.bool)
        for i, emb in enumerate(seqs):
            packed[i, :emb.shape[0]] = emb
            mask[i, :emb.shape[0]] = True
        return packed, mask

    def _qa_loss(self, projected_flat, input_ids, attention_mask, labels):
        with torch.no_grad():
            inputs_embeds = self.model.model.get_input_embeddings()(input_ids).detach()
        projected_flat = projected_flat.to(inputs_embeds.dtype)
        _, special_video_mask = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=projected_flat)
        inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, projected_flat)
        outputs = self.model.model.language_model(attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=True)
        qa_loss = torch.tensor(0.0, device=input_ids.device)
        if labels is not None:
            hidden_states = outputs.last_hidden_state
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            total_tokens = 0
            chunk_seq = 1024
            for s in range(0, shift_hidden.shape[1], chunk_seq):
                h_chunk = shift_hidden[:, s:s + chunk_seq, :].reshape(-1, shift_hidden.shape[-1])
                y_chunk = shift_labels[:, s:s + chunk_seq].reshape(-1)
                valid = y_chunk != -100
                if valid.any():
                    logits_chunk = self.model.lm_head(h_chunk[valid]).float()
                    qa_loss = qa_loss + F.cross_entropy(logits_chunk, y_chunk[valid], reduction="sum")
                    total_tokens += valid.sum().item()
            if total_tokens > 0:
                qa_loss = qa_loss / total_tokens
        return qa_loss

    def _qa_loss_per_sample(self, z_projected, input_ids, attention_mask, labels):
        losses = []
        for i in range(z_projected.shape[0]):
            loss_i = self._qa_loss(
                z_projected[i].reshape(z_projected.shape[1] * z_projected.shape[2], z_projected.shape[3]).unsqueeze(0),
                input_ids[i:i + 1], attention_mask[i:i + 1],
                labels[i:i + 1] if labels is not None else None,
            )
            losses.append(loss_i)
        return torch.stack(losses)

    def forward(self, input_ids, attention_mask, pixel_values_videos, labels=None, num_frames=32, global_step=0, **kwargs):
        del kwargs
        z_pre_pool, z_base = self._build_stage1_inputs(pixel_values_videos, num_frames)
        with torch.no_grad():
            z_temp, local_res, state_res, aux = self.expert(z_pre_pool, z_base, return_aux=True, return_aux_mode="residual")

        # Scaler predicts a_local, a_state from input context only
        q_tokens, q_mask = self._get_user_text_pack(input_ids)
        scales = self.scaler(q_tokens, q_mask, z_base.detach())
        a_local = scales["a_local"]
        a_state = scales["a_state"]

        # Build target from z_base vs z_temp (combined), NOT z_local/z_state individually
        with torch.no_grad():
            local_res_det = aux["local_residual"].detach()
            state_res_det = aux["state_residual"].detach()

            qa_base_ps = self._qa_loss_per_sample(z_base, input_ids, attention_mask, labels)
            qa_temp_ps = self._qa_loss_per_sample(z_temp.detach(), input_ids, attention_mask, labels)
            gap = qa_base_ps - qa_temp_ps

            # EMA for centering
            base_mean = qa_base_ps.mean()
            gap_mean = gap.mean()
            if not bool(self._ema_ready.item()):
                self._ema_base.copy_(base_mean)
                self._ema_gap.copy_(gap_mean)
                self._ema_ready.fill_(True)
            else:
                m = self.ema_momentum
                self._ema_base.mul_(m).add_(base_mean, alpha=1.0 - m)
                self._ema_gap.mul_(m).add_(gap_mean, alpha=1.0 - m)

            # Global temporal alpha target: how much of z_temp to use
            need = torch.sigmoid((qa_base_ps - self._ema_base) / self.alpha_tau)
            benefit = torch.sigmoid((gap - self._ema_gap - self.alpha_margin) / self.alpha_tau)
            alpha_target = (need * benefit).clamp(0.0, 1.0)  # (B,) ∈ [0, 1]

            # Split alpha into local/state proportional to their contribution norms
            local_norm = local_res_det.norm(dim=-1).mean(dim=(1, 2))  # (B,)
            state_norm = state_res_det.norm(dim=-1).mean(dim=(1, 2))  # (B,)
            total_norm = (local_norm + state_norm).clamp(min=1e-6)
            local_ratio = local_norm / total_norm  # (B,) ∈ [0, 1]

            # Target: scale each branch proportionally within the alpha budget
            # a_local_target ≈ alpha * local_ratio * 2 (since init=1.0, range around 1.0)
            # Map: alpha=0 → both targets=0 (no temporal), alpha=1 → targets proportional to ratio
            target_local = self.scaler.init_local * (1.0 - alpha_target) + (self.scaler.init_local + self.scaler.max_local_delta) * alpha_target * local_ratio * 2.0
            target_state = self.scaler.init_state * (1.0 - alpha_target) + (self.scaler.init_state + self.scaler.max_state_delta) * alpha_target * (1.0 - local_ratio) * 2.0
            target_local = target_local.clamp(self.scaler.init_local - self.scaler.max_local_delta,
                                               self.scaler.init_local + self.scaler.max_local_delta)
            target_state = target_state.clamp(self.scaler.init_state - self.scaler.max_state_delta,
                                               self.scaler.init_state + self.scaler.max_state_delta)

        # Policy loss: MSE to target (no QA gradient through scaler)
        policy_loss = torch.tensor(0.0, device=a_local.device)
        if self.policy_lambda > 0 and global_step >= self.policy_warmup_steps:
            policy_loss = 0.5 * (
                F.mse_loss(a_local.float(), target_local.detach().float())
                + F.mse_loss(a_state.float(), target_state.detach().float())
            )

        # QA for logging only (detached scaler outputs)
        z_mix = z_base + a_local.detach().to(z_base.dtype).view(-1, 1, 1, 1) * local_res_det + a_state.detach().to(z_base.dtype).view(-1, 1, 1, 1) * state_res_det
        qa_mix = self._qa_loss(
            z_mix.reshape(z_mix.shape[0], z_mix.shape[1] * z_mix.shape[2], z_mix.shape[3]),
            input_ids, attention_mask, labels,
        )

        total_loss = self.policy_lambda * policy_loss

        self._last_qa_loss = qa_mix.item()
        self._last_policy_loss = policy_loss.item()
        self._last_a_local = a_local.detach().mean().item()
        self._last_a_state = a_state.detach().mean().item()
        self._last_a_local_std = a_local.detach().std(unbiased=False).item()
        self._last_a_state_std = a_state.detach().std(unbiased=False).item()
        self._last_target_alpha = alpha_target.mean().item()
        self._last_target_local = target_local.mean().item()
        self._last_target_state = target_state.mean().item()
        self._last_need = need.mean().item()
        self._last_benefit = benefit.mean().item()
        self._last_local_ratio = local_ratio.mean().item()
        self._last_delta_norm = (local_res_det.norm(dim=-1).mean() + state_res_det.norm(dim=-1).mean()).item()
        return total_loss


def save_checkpoint(output_dir, step, expert, scaler, optimizer, scheduler, adapter_config):
    ckpt_dir = os.path.join(output_dir, f"ckpt_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    exp = expert._orig_mod if hasattr(expert, "_orig_mod") else expert
    sc = scaler._orig_mod if hasattr(scaler, "_orig_mod") else scaler
    torch.save(exp.state_dict(), os.path.join(ckpt_dir, "diff_expert.pt"))
    sc_bf16_sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sc.state_dict().items()}
    torch.save(sc_bf16_sd, os.path.join(ckpt_dir, "scaler.pt"))
    with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, os.path.join(ckpt_dir, "training_state.pt"))


def _build_udh_expert(stage1_cfg, device):
    expert = SplitDualHeadExpert(
        proj_dim=stage1_cfg.get("proj_dim", 2048),
        bottleneck=stage1_cfg.get("expert_bottleneck", 128),
        kernel_size=stage1_cfg.get("expert_kernel_size", 3),
        max_local_gate=stage1_cfg.get("max_local_gate", 0.35),
        max_state_gate=stage1_cfg.get("max_state_gate", 0.35),
        pooling_ratio=stage1_cfg.get("pooling_ratio", 2),
    )
    return expert.to(device=device, dtype=torch.bfloat16)


def train(args):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda:0")

    with open(os.path.join(args.stage1_ckpt, "adapter_config.json")) as f:
        stage1_cfg = json.load(f)
    stage1_module_type = stage1_cfg.get("module_type", "")
    if stage1_module_type not in SUPPORTED_STAGE1_TYPES:
        raise ValueError(f"Unsupported stage1 module_type {stage1_module_type!r}")

    os.makedirs(args.output_dir, exist_ok=True)
    num_frames = args.num_frames or stage1_cfg.get("num_frames", 32)
    max_length = num_frames * TOKENS_PER_FRAME + 512
    num_opt_steps = args.max_steps // args.grad_accum
    warmup_steps = min(50, num_opt_steps // 5)
    eff_batch = args.batch_size * args.grad_accum * world_size
    dataset_configs = resolve_dataset_configs(args.dataset_preset)

    log_rank0(rank, "=== UDH v2 Stage 2: Branch-Output Residual Scaler ===")
    log_rank0(rank, f"  stage1_ckpt={args.stage1_ckpt}")
    log_rank0(rank, f"  dataset_preset={args.dataset_preset}")
    log_rank0(rank, f"  batch={args.batch_size}x{args.grad_accum}x{world_size}={eff_batch}")

    processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B")
    base_model = AutoModelForImageTextToText.from_pretrained(
        "facebook/Perception-LM-1B", torch_dtype=torch.bfloat16, device_map={"": device}
    )
    base_model.model.language_model.gradient_checkpointing_enable()
    base_model.model.language_model.config.use_cache = False
    base_model.model.language_model.train()
    for p in base_model.parameters():
        p.requires_grad = False

    expert = _build_udh_expert(stage1_cfg, device)
    expert.load_state_dict(
        torch.load(os.path.join(args.stage1_ckpt, "diff_expert.pt"), map_location="cpu", weights_only=True),
        strict=False,
    )
    expert.eval()
    for p in expert.parameters():
        p.requires_grad = False

    scaler = RawContextResidualScaler(
        token_dim=base_model.model.get_input_embeddings().weight.shape[1],
        hidden_dim=args.router_hidden,
        num_heads=args.router_heads,
        init_local=args.init_local,
        init_state=args.init_state,
        max_local_delta=args.max_local_delta,
        max_state_delta=args.max_state_delta,
        mean_preserve=False,
        head_init_std=args.head_init_std,
        fuse_gain=args.fuse_gain,
    ).to(device=device, dtype=torch.float32)

    wrapped = PLMWithUDHResidualScaler(
        base_model, expert, scaler,
        alpha_tau=args.alpha_tau,
        alpha_margin=args.alpha_margin,
        policy_lambda=args.policy_lambda,
        policy_warmup_steps=args.policy_warmup_steps,
        ema_momentum=args.ema_momentum,
    ).to(device=device)

    dataset = build_dataset(dataset_configs, processor, num_frames, max_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True, persistent_workers=True)

    trainable_params = [p for p in scaler.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": trainable_params, "lr": args.adapter_lr}], weight_decay=0.01)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, num_opt_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    adapter_config = {
        "module_type": SUPPORTED_STAGE1_TYPES[stage1_module_type],
        "stage1_module_type": stage1_module_type,
        "stage1_ckpt": args.stage1_ckpt,
        "stage2_style": "udh_v2_bosres2",
        "router_hidden": args.router_hidden,
        "init_local": args.init_local,
        "init_state": args.init_state,
        "max_local_delta": args.max_local_delta,
        "max_state_delta": args.max_state_delta,
        "head_init_std": args.head_init_std,
        "fuse_gain": args.fuse_gain,
        "alpha_tau": args.alpha_tau,
        "alpha_margin": args.alpha_margin,
        "policy_lambda": args.policy_lambda,
        "num_frames": num_frames,
        "proj_dim": stage1_cfg.get("proj_dim", 2048),
        "expert_bottleneck": stage1_cfg.get("expert_bottleneck", 128),
        "expert_kernel_size": stage1_cfg.get("expert_kernel_size", 3),
        "max_local_gate": stage1_cfg.get("max_local_gate", 0.35),
        "max_state_gate": stage1_cfg.get("max_state_gate", 0.35),
        "pooling_ratio": stage1_cfg.get("pooling_ratio", 2),
    }

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(999999):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values_videos = batch["pixel_values_videos"].to(device=device, dtype=torch.bfloat16)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = wrapped(
                    input_ids=input_ids, attention_mask=attention_mask,
                    pixel_values_videos=pixel_values_videos, labels=labels,
                    num_frames=num_frames, global_step=global_step,
                )
            (loss / args.grad_accum).backward()

            if (global_step + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                wrapped._last_grad_norm = float(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if rank == 0 and global_step % args.log_interval == 0:
                m = wrapped
                log_rank0(rank,
                    f"[Step {global_step}/{args.max_steps}] "
                    f"qa={m._last_qa_loss:.4f} pol={m._last_policy_loss:.4f} "
                    f"a_l={m._last_a_local:.3f} a_s={m._last_a_state:.3f} "
                    f"a_l_std={m._last_a_local_std:.3f} a_s_std={m._last_a_state_std:.3f} "
                    f"tgt_a={m._last_target_alpha:.3f} tgt_l={m._last_target_local:.3f} tgt_s={m._last_target_state:.3f} "
                    f"need={m._last_need:.3f} benefit={m._last_benefit:.3f} l_ratio={m._last_local_ratio:.3f} "
                    f"delta_n={m._last_delta_norm:.3f} g_norm={m._last_grad_norm:.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if rank == 0 and global_step % args.save_interval == 0:
                save_checkpoint(args.output_dir, global_step, expert, scaler, optimizer, scheduler, adapter_config)

            if global_step >= args.max_steps:
                if rank == 0:
                    save_checkpoint(args.output_dir, global_step, expert, scaler, optimizer, scheduler, adapter_config)
                if dist.is_initialized():
                    dist.destroy_process_group()
                return


def parse_args():
    p = argparse.ArgumentParser(description="Train UDH v1 Branch-Output Residual Scaler Stage 2")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_frames", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=2400)
    p.add_argument("--save_interval", type=int, default=400)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--adapter_lr", type=float, default=3e-4)
    p.add_argument("--router_hidden", type=int, default=128)
    p.add_argument("--router_heads", type=int, default=4)
    p.add_argument("--init_local", type=float, default=1.0)
    p.add_argument("--init_state", type=float, default=1.0)
    p.add_argument("--max_local_delta", type=float, default=0.35)
    p.add_argument("--max_state_delta", type=float, default=0.35)
    p.add_argument("--head_init_std", type=float, default=0.02)
    p.add_argument("--fuse_gain", type=float, default=40.0)
    p.add_argument("--alpha_tau", type=float, default=0.20)
    p.add_argument("--alpha_margin", type=float, default=0.02)
    p.add_argument("--policy_lambda", type=float, default=0.20)
    p.add_argument("--policy_warmup_steps", type=int, default=50)
    p.add_argument("--ema_momentum", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dataset_preset", type=str, default="mcq_benchlike")
    p.add_argument("--no_liger", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
