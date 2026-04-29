"""
Train UDH v2 Stage 2 — Target-Supervised Dual Residual Calibrator.

  z_final = z_base + a_local(q,v) * local_residual + a_state(q,v) * state_residual

Training approach (target-supervised, NOT direct QA gradient):
  1. Per-sample: compute L_base, L_local, L_state via frozen LLM forward
  2. Build soft targets from QA loss gaps:
       target_local = sigmoid((L_base - L_local) / tau)
       target_state = sigmoid((L_base - L_state) / tau)
  3. Train calibrator with MSE(pred, target) — strong, direct gradient

Two calibrator architectures (--calibrator_type):
  - mlp: Cross-attention MLP (TwoTowerDualResidualCalibrator)
  - gated_attn: Gated Attention (GatedAttentionCalibrator, NeurIPS 2025 style)

Key improvements over previous LS Policy:
  - init=1.0, range [0.65, 1.35] — matches Stage 1 training condition
  - Direct gap→sigmoid target (no Need×Benefit multiplication)
  - No EMA centering — targets are absolute, not relative
  - 2× LLM forward (base + local, base + state share base) not 3×
"""

import argparse
import json
import math
import os
import sys
import time
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
from core.vision_projector.dual_proj import TwoTowerDualResidualCalibrator, GatedAttentionCalibrator, SplitDualHeadExpert
from scripts.train_temporal_expert_router_stage2 import (
    TRAIN_DATASETS,
    TOKENS_PER_FRAME,
    build_dataset,
    build_router_marker_ids,
    collate_fn,
    log_rank0,
)


SUPPORTED_STAGE1_TYPES = {
    "dual_proj_deltanet_udh_v2_stage1": "dual_proj_deltanet_udh_v2_res2",
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


class PLMWithUDHv2TargetCalibrator(nn.Module):
    def __init__(
        self,
        model,
        expert,
        calibrator,
        router_marker_ids=None,
        tau=0.30,
        target_bias=0.0,
    ):
        super().__init__()
        self.model = model
        self.expert = expert
        self.calibrator = calibrator
        self.router_marker_ids = router_marker_ids or []
        self.tau = float(tau)
        self.target_bias = float(target_bias)
        self._last_grad = 0.0

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
            z_base_flat = proj.pooling(features)
            n_pre = features.shape[1]
            m, d = z_base_flat.shape[1], z_base_flat.shape[2]
            z_pre_pool = features.reshape(bsz, t, n_pre, d)
            z_base = z_base_flat.reshape(bsz, t, m, d)
        return z_pre_pool, z_base

    @staticmethod
    def _find_subsequence(seq, pattern, start, end):
        plen = len(pattern)
        if plen == 0 or end - start < plen:
            return None
        for idx in range(start, end - plen + 1):
            if seq[idx:idx + plen] == pattern:
                return idx
        return None

    def _get_question_embedding(self, input_ids):
        with torch.no_grad():
            all_embeds = self.model.model.get_input_embeddings()(input_ids)
        q_embeds = []
        for i in range(input_ids.shape[0]):
            seq = input_ids[i].tolist()
            valid_len = len(seq)
            while valid_len > 0 and seq[valid_len - 1] == v2.PAD_ID:
                valid_len -= 1
            start, end = 0, valid_len
            eot_positions = (input_ids[i] == v2.EOT_ID).nonzero(as_tuple=True)[0].tolist()
            if eot_positions:
                user_end = eot_positions[0]
                header_positions = (input_ids[i] == v2.END_HEADER_ID).nonzero(as_tuple=True)[0].tolist()
                header_positions = [pos for pos in header_positions if pos < user_end]
                if header_positions:
                    start = header_positions[-1] + 1
                end = user_end
            for marker_ids in self.router_marker_ids:
                pos = self._find_subsequence(seq, marker_ids, start, end)
                if pos is not None:
                    end = min(end, pos)
            tokens_i = input_ids[i, start:end]
            embeds_i = all_embeds[i, start:end]
            text_mask = (
                (tokens_i != v2.VIDEO_TOKEN_ID) & (tokens_i != v2.PAD_ID)
                & (tokens_i != v2.END_HEADER_ID) & (tokens_i != v2.EOT_ID)
            )
            if text_mask.any():
                q_embeds.append(embeds_i[text_mask].mean(dim=0))
            else:
                q_embeds.append(torch.zeros(all_embeds.shape[-1], device=all_embeds.device, dtype=all_embeds.dtype))
        return torch.stack(q_embeds)

    def _qa_loss_per_sample(self, z_4d, input_ids, attention_mask, labels):
        """Compute per-sample QA loss (no gradient)."""
        B = z_4d.shape[0]
        losses = []
        for i in range(B):
            z_flat = z_4d[i].reshape(-1, z_4d.shape[-1]).unsqueeze(0)  # (1, T*M, D)
            with torch.no_grad():
                inputs_embeds = self.model.model.get_input_embeddings()(input_ids[i:i+1]).detach()
            z_flat = z_flat.to(inputs_embeds.dtype)
            _, special_video_mask = self.model.model.get_placeholder_mask(
                input_ids[i:i+1], inputs_embeds=inputs_embeds, video_features=z_flat)
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, z_flat)
            outputs = self.model.model.language_model(
                attention_mask=attention_mask[i:i+1], inputs_embeds=inputs_embeds, return_dict=True)
            hidden = outputs.last_hidden_state
            shift_h = hidden[..., :-1, :].contiguous().reshape(-1, hidden.shape[-1])
            shift_y = labels[i:i+1, 1:].contiguous().reshape(-1)
            valid = shift_y != -100
            if valid.any():
                logits = self.model.lm_head(shift_h[valid]).float()
                loss_i = F.cross_entropy(logits, shift_y[valid], reduction="mean")
            else:
                loss_i = torch.tensor(0.0, device=z_4d.device)
            losses.append(loss_i)
        return torch.stack(losses)

    def forward(self, input_ids, attention_mask, pixel_values_videos, labels=None, num_frames=32, **kwargs):
        z_pre_pool, z_base = self._build_stage1_inputs(pixel_values_videos, num_frames)

        with torch.no_grad():
            z_temp, _, _, aux = self.expert(z_pre_pool, z_base, return_aux=True, return_aux_mode="residual")
            local_res = aux["local_residual"].detach()
            state_res = aux["state_residual"].detach()

            # ---- Build per-sample targets (3× LLM forward, all no_grad) ----
            z_local = z_base + local_res   # what z looks like with only local expert
            z_state = z_base + state_res   # what z looks like with only state expert

            qa_base = self._qa_loss_per_sample(z_base, input_ids, attention_mask, labels)
            qa_local = self._qa_loss_per_sample(z_local, input_ids, attention_mask, labels)
            qa_state = self._qa_loss_per_sample(z_state, input_ids, attention_mask, labels)

            # gap > 0 means expert helps (base loss higher than expert loss)
            gap_local = qa_base - qa_local   # (B,)
            gap_state = qa_base - qa_state   # (B,)

            # Soft target: sigmoid maps gap to [0, 1]
            # bias > 0 shifts targets higher (more aggressive expert usage)
            target_local = torch.sigmoid((gap_local + self.target_bias) / self.tau)
            target_state = torch.sigmoid((gap_state + self.target_bias) / self.tau)

            # Scale targets to calibrator range [a_min, a_max]
            cal = self.calibrator._orig_mod if hasattr(self.calibrator, "_orig_mod") else self.calibrator
            target_local_scaled = cal.a_local_min + (cal.a_local_max - cal.a_local_min) * target_local
            target_state_scaled = cal.a_state_min + (cal.a_state_max - cal.a_state_min) * target_state

        # ---- Calibrator prediction (with gradient) ----
        q_embed = self._get_question_embedding(input_ids)
        v_frames = z_temp.detach().mean(dim=2)  # (B, T, D)

        # Handcrafted per-sample statistics (high variance by construction)
        with torch.no_grad():
            B = z_base.shape[0]
            local_mag = local_res.flatten(1).norm(dim=-1)          # (B,) local residual magnitude
            state_mag = state_res.flatten(1).norm(dim=-1)          # (B,) state residual magnitude
            enrich_mag = (z_temp - z_base).detach().flatten(1).norm(dim=-1)  # (B,) total enrichment
            v_temporal_std = v_frames.std(dim=1).mean(dim=-1)      # (B,) temporal variation
            qv_cos = F.cosine_similarity(q_embed, v_frames.mean(dim=1), dim=-1)  # (B,) q-v alignment
            local_state_ratio = local_mag / (state_mag + 1e-6)     # (B,) local vs state balance
            sample_stats = torch.stack([
                local_mag, state_mag, enrich_mag,
                v_temporal_std, qv_cos, local_state_ratio
            ], dim=-1).detach()  # (B, 6) — raw values, no normalization

        scales = self.calibrator(q_embed, v_frames, sample_stats=sample_stats)
        a_local = scales["a_local"]
        a_state = scales["a_state"]

        # MSE supervision — direct, strong gradient to calibrator
        local_loss = F.mse_loss(a_local, target_local_scaled.detach())
        state_loss = F.mse_loss(a_state, target_state_scaled.detach())
        total_loss = 0.5 * (local_loss + state_loss)

        # ---- Monitoring: compute mixed QA for logging (no gradient) ----
        with torch.no_grad():
            z_mix = z_base + a_local.detach().to(z_base.dtype).view(-1, 1, 1, 1) * local_res \
                           + a_state.detach().to(z_base.dtype).view(-1, 1, 1, 1) * state_res
            z_mix_flat = z_mix.reshape(z_mix.shape[0], -1, z_mix.shape[-1])
            inputs_embeds = self.model.model.get_input_embeddings()(input_ids).detach()
            z_mix_flat_cast = z_mix_flat.to(inputs_embeds.dtype)
            _, mask = self.model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=z_mix_flat_cast)
            inputs_embeds = inputs_embeds.masked_scatter(mask, z_mix_flat_cast)
            out = self.model.model.language_model(attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=True)
            h = out.last_hidden_state[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            qa_mix_loss = 0.0
            total_tok = 0
            for s in range(0, h.shape[1], 1024):
                hc = h[:, s:s+1024].reshape(-1, h.shape[-1])
                yc = y[:, s:s+1024].reshape(-1)
                v = yc != -100
                if v.any():
                    qa_mix_loss += F.cross_entropy(self.model.lm_head(hc[v]).float(), yc[v], reduction="sum").item()
                    total_tok += v.sum().item()
            qa_mix_loss = qa_mix_loss / max(total_tok, 1)

        self._last_loss = total_loss.item()
        self._last_qa = qa_mix_loss
        self._last_l_l = local_loss.item()
        self._last_s_l = state_loss.item()
        self._last_a_l = a_local.detach().mean().item()
        self._last_a_s = a_state.detach().mean().item()
        self._last_a_l_std = a_local.detach().std(unbiased=False).item()
        self._last_a_s_std = a_state.detach().std(unbiased=False).item()
        self._last_a_l_min = a_local.detach().amin().item()
        self._last_a_l_max = a_local.detach().amax().item()
        self._last_a_s_min = a_state.detach().amin().item()
        self._last_a_s_max = a_state.detach().amax().item()
        self._last_tgt_l = target_local_scaled.mean().item()
        self._last_tgt_s = target_state_scaled.mean().item()
        self._last_tgt_l_std = target_local_scaled.std(unbiased=False).item()
        self._last_tgt_s_std = target_state_scaled.std(unbiased=False).item()
        self._last_gap_l = gap_local.mean().item()
        self._last_gap_s = gap_state.mean().item()
        self._last_qa_base = qa_base.mean().item()
        return total_loss


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


def save_checkpoint(output_dir, step, expert, calibrator, optimizer, scheduler, adapter_config):
    ckpt_dir = os.path.join(output_dir, f"ckpt_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    exp = expert._orig_mod if hasattr(expert, "_orig_mod") else expert
    cal = calibrator._orig_mod if hasattr(calibrator, "_orig_mod") else calibrator
    torch.save(exp.state_dict(), os.path.join(ckpt_dir, "diff_expert.pt"))
    cal_state_bf16 = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in cal.state_dict().items()}
    torch.save(cal_state_bf16, os.path.join(ckpt_dir, "calibrator.pt"))
    with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, os.path.join(ckpt_dir, "training_state.pt"))


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

    log_rank0(rank, f"=== UDH v2 Stage 2: Target-Supervised Calibrator ({args.calibrator_type}) ===")
    log_rank0(rank, f"  stage1_ckpt={args.stage1_ckpt}")
    log_rank0(rank, f"  dataset_preset={args.dataset_preset}")
    log_rank0(rank, f"  batch={args.batch_size}x{args.grad_accum}x{world_size}={eff_batch}")
    log_rank0(rank, f"  tau={args.tau}, target_bias={args.target_bias}")

    if not args.no_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_mllama
            apply_liger_kernel_to_mllama()
            log_rank0(rank, "[Liger] Applied kernel patches")
        except Exception:
            pass

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

    proj_dim = stage1_cfg.get("proj_dim", 2048)
    q_dim = base_model.model.get_input_embeddings().weight.shape[1]
    cal_kwargs = dict(
        q_dim=q_dim, video_dim=proj_dim, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, init_local=args.init_local, init_state=args.init_state,
        max_local_delta=args.max_local_delta, max_state_delta=args.max_state_delta,
    )
    if args.calibrator_type == "gated_attn":
        calibrator = GatedAttentionCalibrator(**cal_kwargs)
        log_rank0(rank, "  calibrator=GatedAttentionCalibrator (Gated Attention)")
    else:
        calibrator = TwoTowerDualResidualCalibrator(**cal_kwargs)
        log_rank0(rank, "  calibrator=TwoTowerDualResidualCalibrator (Cross-Attention MLP)")
    calibrator = calibrator.to(device=device, dtype=torch.float32)

    router_marker_ids = build_router_marker_ids(processor.tokenizer)
    wrapped = PLMWithUDHv2TargetCalibrator(
        base_model, expert, calibrator,
        router_marker_ids=router_marker_ids,
        tau=args.tau,
        target_bias=args.target_bias,
    ).to(device=device)

    dataset = build_dataset(dataset_configs, processor, num_frames, max_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True, persistent_workers=True)

    # Split params: stats path gets 10x higher LR (primary routing path)
    stats_params = [p for n, p in calibrator.named_parameters() if 'stats' in n]
    deep_params = [p for n, p in calibrator.named_parameters() if 'stats' not in n]
    trainable_params = stats_params + deep_params
    optimizer = torch.optim.AdamW([
        {"params": stats_params, "lr": args.adapter_lr * 10, "name": "stats"},
        {"params": deep_params, "lr": args.adapter_lr, "name": "deep"},
    ], weight_decay=0.01)
    log_rank0(rank, f"  stats_params={sum(p.numel() for p in stats_params)} deep_params={sum(p.numel() for p in deep_params)}")

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, num_opt_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    adapter_config = {
        "module_type": SUPPORTED_STAGE1_TYPES[stage1_module_type],
        "stage1_module_type": stage1_module_type,
        "stage1_ckpt": args.stage1_ckpt,
        "stage2_style": "udh_v2_target_supervised_calibrator",
        "calibrator_type": args.calibrator_type,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "video_dim": proj_dim,
        "init_local": args.init_local,
        "init_state": args.init_state,
        "max_local_delta": args.max_local_delta,
        "max_state_delta": args.max_state_delta,
        "tau": args.tau,
        "target_bias": args.target_bias,
        "num_frames": num_frames,
        "proj_dim": stage1_cfg.get("proj_dim", 2048),
        "expert_bottleneck": stage1_cfg.get("expert_bottleneck", 128),
        "expert_kernel_size": stage1_cfg.get("expert_kernel_size", 3),
        "max_local_gate": stage1_cfg.get("max_local_gate", 0.35),
        "max_state_gate": stage1_cfg.get("max_state_gate", 0.35),
        "pooling_ratio": stage1_cfg.get("pooling_ratio", 2),
    }

    t_start = time.time()
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
                loss = wrapped(input_ids=input_ids, attention_mask=attention_mask,
                               pixel_values_videos=pixel_values_videos, labels=labels,
                               num_frames=num_frames)
            (loss.float() / args.grad_accum).backward()

            if (global_step + 1) % args.grad_accum == 0:
                # Sync gradients across GPUs
                if world_size > 1:
                    grads = [p.grad for p in trainable_params if p.grad is not None]
                    if grads:
                        flat = torch.cat([g.flatten() for g in grads])
                        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
                        offset = 0
                        for g in grads:
                            numel = g.numel()
                            g.copy_(flat[offset:offset + numel].view_as(g))
                            offset += numel
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                wrapped._last_grad = float(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if rank == 0 and global_step % args.log_interval == 0:
                m = wrapped
                elapsed = time.time() - t_start
                speed = elapsed / args.log_interval
                log_rank0(rank,
                    f"[Step {global_step}/{args.max_steps}] "
                    f"loss={m._last_loss:.4f} qa={m._last_qa:.4f} "
                    f"l_l={m._last_l_l:.4f} s_l={m._last_s_l:.4f} "
                    f"a_l={m._last_a_l:.3f} a_s={m._last_a_s:.3f} "
                    f"a_l_std={m._last_a_l_std:.3f} a_s_std={m._last_a_s_std:.3f} "
                    f"a_l_rng={m._last_a_l_min:.3f}-{m._last_a_l_max:.3f} "
                    f"a_s_rng={m._last_a_s_min:.3f}-{m._last_a_s_max:.3f} "
                    f"tgt_l={m._last_tgt_l:.3f}±{m._last_tgt_l_std:.3f} "
                    f"tgt_s={m._last_tgt_s:.3f}±{m._last_tgt_s_std:.3f} "
                    f"gap_l={m._last_gap_l:.3f} gap_s={m._last_gap_s:.3f} "
                    f"qa_base={m._last_qa_base:.3f} "
                    f"g={m._last_grad:.3f} lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"spd={speed:.1f}s"
                )
                t_start = time.time()

            if rank == 0 and global_step % args.save_interval == 0:
                save_checkpoint(args.output_dir, global_step, expert, calibrator, optimizer, scheduler, adapter_config)

            if global_step >= args.max_steps:
                if rank == 0:
                    save_checkpoint(args.output_dir, global_step, expert, calibrator, optimizer, scheduler, adapter_config)
                if dist.is_initialized():
                    dist.destroy_process_group()
                return


def parse_args():
    p = argparse.ArgumentParser(description="Train UDH v2 Target-Supervised Calibrator Stage 2")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_frames", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=2400)
    p.add_argument("--save_interval", type=int, default=400)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--adapter_lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--calibrator_type", type=str, default="gated_attn", choices=["mlp", "gated_attn"],
                   help="Calibrator architecture: mlp (cross-attention MLP) or gated_attn (Gated Attention)")
    p.add_argument("--init_local", type=float, default=1.0)
    p.add_argument("--init_state", type=float, default=1.0)
    p.add_argument("--max_local_delta", type=float, default=0.35)
    p.add_argument("--max_state_delta", type=float, default=0.35)
    p.add_argument("--tau", type=float, default=0.03,
                   help="Temperature for gap→target sigmoid. Must match gap scale (~0.02). Lower=sharper")
    p.add_argument("--target_bias", type=float, default=0.005,
                   help="Positive bias shifts targets toward 'use expert'. Scale relative to gap (~0.02)")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dataset_preset", type=str, default="mcq_benchlike")
    p.add_argument("--no_liger", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
