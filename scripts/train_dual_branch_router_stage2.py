"""
Stage 2 — Dual Branch Router for v928 / UDH v2 experts.

Routes local and state residuals independently:
  z_final = z_base + w_local * local_res + w_state * state_res

Key design (from successful router_combo experiments):
  - QA gradient flows directly through w_local/w_state → router
  - KL supervision as auxiliary signal
  - LLM frozen → MCQ format preserved
  - log1p(stats) as router input
  - Full dataset for diversity

Compatible with both v928 (StateAwareDiffDeltaTemporalExpert)
and UDH v2 (SplitDualHeadExpert).
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.train_dual_proj as v2
from core.vision_projector.dual_proj import DualBranchRouter, GatedAttentionDualRouter
from scripts.train_temporal_expert_router_stage2 import (
    TOKENS_PER_FRAME,
    TRAIN_DATASETS,
    _build_frozen_expert,
    build_dataset,
    collate_fn,
    log_rank0,
)

SUPPORTED_STAGE1_TYPES = {
    "dual_proj_deltanet_v926_stage1": "dual_proj_deltanet_v926_dbrouter",
    "dual_proj_deltanet_v927_stage1": "dual_proj_deltanet_v927_dbrouter",
    "dual_proj_deltanet_v928_stage1": "dual_proj_deltanet_v928_dbrouter",
    "dual_proj_deltanet_udh_v2_stage1": "dual_proj_deltanet_udh_v2_dbrouter",
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


def resolve_stage1_module_type(stage1_cfg):
    module_type = stage1_cfg.get("module_type", "")
    if module_type in SUPPORTED_STAGE1_TYPES:
        return module_type

    fallback = stage1_cfg.get("stage1_module_type", "")
    if fallback in SUPPORTED_STAGE1_TYPES:
        return fallback

    raise ValueError(
        f"Unsupported stage1 module_type {module_type!r}"
        + (f" (stage1_module_type={fallback!r})" if fallback else "")
    )


class PLMWithDualBranchRouter(torch.nn.Module):
    def __init__(
        self,
        model,
        expert,
        router,
        stage1_module_type,
        align_lambda=0.03,
        router_lambda=0.10,
        router_tau=0.5,
        router_margin=0.02,
        router_warmup_steps=50,
        diversity_lambda=0.0,
        oracle_mix=False,
    ):
        super().__init__()
        self.model = model
        self.expert = expert
        self.router = router
        self.stage1_module_type = stage1_module_type
        self.align_lambda = align_lambda
        self.router_lambda = router_lambda
        self.router_tau = router_tau
        self.router_margin = router_margin
        self.router_warmup_steps = router_warmup_steps
        self.diversity_lambda = float(diversity_lambda)
        self.oracle_mix = bool(oracle_mix)
        self._last_grad = 0.0
        self._last_diversity = 0.0

    def _get_vision_features(self, pixel_values_videos, num_frames):
        bsz = pixel_values_videos.shape[0]
        frames_flat = pixel_values_videos.flatten(0, 1)
        with torch.no_grad():
            vision_out = self.model.model.vision_tower(frames_flat)
            features = vision_out.last_hidden_state[:, 1:, :].float()
        _, n, c = features.shape
        return features.reshape(bsz, num_frames, n, c)

    def _build_stage1_features(self, pixel_values_videos, num_frames):
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

            z_temp, _, _, aux = self.expert(z_pre_pool, z_base, return_aux=True, return_aux_mode="residual")
            local_res = aux["local_residual"].detach()
            state_res = aux["state_residual"].detach()

        return z_base, local_res, state_res, z_temp.detach()

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

    def _compute_router_stats(self, z_base, local_res, state_res):
        """4 stats matching router_combo format + local/state separation."""
        base_pool = z_base.detach().mean(dim=2)
        if z_base.shape[1] > 1:
            base_motion = (base_pool[:, 1:] - base_pool[:, :-1]).norm(dim=-1).mean(dim=1)
        else:
            base_motion = torch.zeros(z_base.shape[0], device=z_base.device, dtype=z_base.dtype)
        local_mag = local_res.detach().norm(dim=-1).mean(dim=(1, 2))
        state_mag = state_res.detach().norm(dim=-1).mean(dim=(1, 2))
        total_mag = (local_res + state_res).detach().norm(dim=-1).mean(dim=(1, 2))
        stats = torch.stack([base_motion, local_mag, state_mag, total_mag], dim=-1)
        return torch.log1p(stats.float()).to(device=z_base.device, dtype=z_base.dtype)

    def _qa_loss(self, projected_flat, input_ids, attention_mask, labels):
        with torch.no_grad():
            inputs_embeds = self.model.model.get_input_embeddings()(input_ids).detach()
        projected_flat = projected_flat.to(inputs_embeds.dtype)
        _, special_video_mask = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=projected_flat)
        inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, projected_flat)
        outputs = self.model.model.language_model(
            attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=True)
        qa_loss = torch.tensor(0.0, device=input_ids.device)
        if labels is not None:
            hidden_states = outputs.last_hidden_state
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            total_tokens = 0
            for s in range(0, shift_hidden.shape[1], 1024):
                h_c = shift_hidden[:, s:s + 1024, :].reshape(-1, shift_hidden.shape[-1])
                y_c = shift_labels[:, s:s + 1024].reshape(-1)
                valid = y_c != -100
                if valid.any():
                    qa_loss = qa_loss + F.cross_entropy(
                        self.model.lm_head(h_c[valid]).float(), y_c[valid], reduction="sum")
                    total_tokens += valid.sum().item()
            if total_tokens > 0:
                qa_loss = qa_loss / total_tokens
        return qa_loss

    def _qa_loss_per_sample(self, z_4d, input_ids, attention_mask, labels):
        bsz = z_4d.shape[0]
        z_flat = z_4d.reshape(bsz, -1, z_4d.shape[-1])
        with torch.inference_mode():
            inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
            z_cast = z_flat.to(inputs_embeds.dtype)
            _, mask = self.model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=z_cast)
            inputs_embeds = inputs_embeds.masked_scatter(mask, z_cast)
            outputs = self.model.model.language_model(
                attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=True)
            per_sample = torch.zeros(bsz, device=input_ids.device)
            if labels is not None:
                h = outputs.last_hidden_state[..., :-1, :].contiguous()
                y = labels[..., 1:].contiguous()
                for b in range(bsz):
                    total_tok = 0
                    loss_b = 0.0
                    for s in range(0, h.shape[1], 1024):
                        hc = h[b:b+1, s:s+1024].reshape(-1, h.shape[-1])
                        yc = y[b:b+1, s:s+1024].reshape(-1)
                        v = yc != -100
                        if v.any():
                            loss_b += F.cross_entropy(
                                self.model.lm_head(hc[v]).float(), yc[v], reduction="sum").item()
                            total_tok += v.sum().item()
                    if total_tok > 0:
                        per_sample[b] = loss_b / total_tok
        return per_sample

    @staticmethod
    def _alignment_loss(z_final, z_base):
        diff = (z_final - z_base).norm(dim=-1)
        base_norm = z_base.detach().norm(dim=-1).clamp(min=1e-6)
        return (diff / base_norm).mean()

    def forward(self, input_ids, attention_mask, pixel_values_videos, labels=None, num_frames=32, global_step=0, **kwargs):
        z_base, local_res, state_res, z_temp = self._build_stage1_features(pixel_values_videos, num_frames)

        # Router prediction
        q_embed = self._get_question_embedding(input_ids)
        stats = self._compute_router_stats(z_base, local_res, state_res)
        # Pass v_frames for GatedAttentionDualRouter (ignored by DualBranchRouter)
        v_frames = z_temp.mean(dim=2)  # (B, T, D) spatial mean
        w, logits = self.router(q_embed, video_stats=stats, v_frames=v_frames, return_logits=True)  # (B, 2)
        w_local = w[:, 0]
        w_state = w[:, 1]
        logit_local = logits[:, 0]
        logit_state = logits[:, 1]

        # Oracle-guided mixing (Method A):
        # Use per-sample oracle w for z_final (better QA signal),
        # but train router to predict it via BCE.
        if self.oracle_mix and global_step >= self.router_warmup_steps:
            with torch.no_grad():
                qa_base_ps = self._qa_loss_per_sample(z_base, input_ids, attention_mask, labels)
                qa_local_ps = self._qa_loss_per_sample(z_base + local_res, input_ids, attention_mask, labels)
                qa_state_ps = self._qa_loss_per_sample(z_base + state_res, input_ids, attention_mask, labels)
                gain_local = qa_base_ps - qa_local_ps - self.router_margin
                gain_state = qa_base_ps - qa_state_ps - self.router_margin
                oracle_w_l = torch.sigmoid(gain_local / self.router_tau).to(z_base.dtype)
                oracle_w_s = torch.sigmoid(gain_state / self.router_tau).to(z_base.dtype)
            z_final = z_base + oracle_w_l.view(-1, 1, 1, 1) * local_res + oracle_w_s.view(-1, 1, 1, 1) * state_res
            self._last_oracle_w_l = oracle_w_l.detach().mean().item()
            self._last_oracle_w_s = oracle_w_s.detach().mean().item()
            self._last_oracle_w_l_std = oracle_w_l.detach().std(unbiased=False).item()
            self._last_oracle_w_s_std = oracle_w_s.detach().std(unbiased=False).item()
        else:
            # Standard: mix with router-predicted weights
            z_final = z_base + w_local.view(-1, 1, 1, 1) * local_res + w_state.view(-1, 1, 1, 1) * state_res
            self._last_oracle_w_l = 0.0
            self._last_oracle_w_s = 0.0
            self._last_oracle_w_l_std = 0.0
            self._last_oracle_w_s_std = 0.0

        # QA loss (main training signal)
        qa_loss = self._qa_loss(
            z_final.reshape(z_final.shape[0], -1, z_final.shape[-1]),
            input_ids, attention_mask, labels)

        # Alignment
        align_loss = self._alignment_loss(z_final, z_base) if self.align_lambda > 0 else torch.tensor(0.0, device=qa_loss.device)

        # KL supervision target (per-branch, independent)
        router_loss = torch.tensor(0.0, device=qa_loss.device)
        if self.router_lambda > 0 and global_step >= self.router_warmup_steps:
            if self.oracle_mix:
                # oracle_mix already computed gain_local/gain_state — reuse via oracle_w
                target_local = oracle_w_l.detach()
                target_state = oracle_w_s.detach()
            else:
                with torch.no_grad():
                    qa_base_ps = self._qa_loss_per_sample(z_base, input_ids, attention_mask, labels)
                    qa_local_ps = self._qa_loss_per_sample(z_base + local_res, input_ids, attention_mask, labels)
                    qa_state_ps = self._qa_loss_per_sample(z_base + state_res, input_ids, attention_mask, labels)
                    gain_local = (qa_base_ps - qa_local_ps - self.router_margin)
                    gain_state = (qa_base_ps - qa_state_ps - self.router_margin)
                    target_local = torch.sigmoid(gain_local / self.router_tau)
                    target_state = torch.sigmoid(gain_state / self.router_tau)

            # BCE with logits: autocast-safe + numerically stable
            router_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(logit_local, target_local.detach())
                + F.binary_cross_entropy_with_logits(logit_state, target_state.detach())
            )

        # Diversity loss: encourage per-sample spread in router weights
        # Uses range (max - min) instead of variance for stronger gradient
        # when weights are clustered, plus a variance floor penalty.
        diversity_loss = torch.tensor(0.0, device=qa_loss.device)
        if self.diversity_lambda > 0 and global_step >= self.router_warmup_steps:
            w_range = (w.max(dim=0).values - w.min(dim=0).values).mean()  # mean range across branches
            w_std = w.std(dim=0).mean()
            min_std = 0.01  # target minimum std
            std_floor_penalty = F.relu(min_std - w_std)  # penalty when std < 0.01
            diversity_loss = -w_range + 5.0 * std_floor_penalty  # maximize range + floor penalty

        total_loss = (qa_loss
                      + self.align_lambda * align_loss
                      + self.router_lambda * router_loss
                      + self.diversity_lambda * diversity_loss)

        # Logging
        self._last_qa = qa_loss.item()
        self._last_align = align_loss.item()
        self._last_router = router_loss.item()
        self._last_diversity = diversity_loss.item()
        self._last_w_l = w_local.detach().mean().item()
        self._last_w_s = w_state.detach().mean().item()
        self._last_w_l_std = w_local.detach().std(unbiased=False).item()
        self._last_w_s_std = w_state.detach().std(unbiased=False).item()
        if self.router_lambda > 0 and global_step >= self.router_warmup_steps:
            self._last_tgt_l = target_local.mean().item()
            self._last_tgt_s = target_state.mean().item()
        else:
            self._last_tgt_l = 0.0
            self._last_tgt_s = 0.0
        return total_loss


def save_checkpoint(output_dir, step, expert, router, optimizer, scheduler, adapter_config):
    ckpt_dir = os.path.join(output_dir, f"ckpt_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    exp = expert._orig_mod if hasattr(expert, "_orig_mod") else expert
    rt = router._orig_mod if hasattr(router, "_orig_mod") else router
    torch.save(exp.state_dict(), os.path.join(ckpt_dir, "diff_expert.pt"))
    torch.save(rt.state_dict(), os.path.join(ckpt_dir, "router.pt"))
    with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
               os.path.join(ckpt_dir, "training_state.pt"))


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
    stage1_module_type = resolve_stage1_module_type(stage1_cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    num_frames = args.num_frames or stage1_cfg.get("num_frames", 32)
    max_length = num_frames * TOKENS_PER_FRAME + 512
    num_opt_steps = args.max_steps // args.grad_accum
    warmup_steps = min(50, num_opt_steps // 5)
    eff_batch = args.batch_size * args.grad_accum * world_size
    dataset_configs = resolve_dataset_configs(args.dataset_preset)

    log_rank0(rank, f"=== Dual Branch Router Stage 2 ({stage1_module_type}) ===")
    log_rank0(rank, f"  stage1_ckpt={args.stage1_ckpt}")
    log_rank0(rank, f"  source_module_type={stage1_cfg.get('module_type', '')}")
    log_rank0(rank, f"  dataset_preset={args.dataset_preset}")
    log_rank0(rank, f"  batch={args.batch_size}x{args.grad_accum}x{world_size}={eff_batch}")

    if not args.no_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_mllama
            apply_liger_kernel_to_mllama()
            log_rank0(rank, "[Liger] Applied kernel patches")
        except Exception:
            pass

    processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B")
    base_model = AutoModelForImageTextToText.from_pretrained(
        "facebook/Perception-LM-1B", torch_dtype=torch.bfloat16, device_map={"": device})
    base_model.model.language_model.gradient_checkpointing_enable()
    base_model.model.language_model.config.use_cache = False
    base_model.model.language_model.train()
    for p in base_model.parameters():
        p.requires_grad = False

    expert = _build_frozen_expert(stage1_cfg, device)
    expert.load_state_dict(
        torch.load(os.path.join(args.stage1_ckpt, "diff_expert.pt"), map_location="cpu", weights_only=True),
        strict=False)
    expert.eval()
    for p in expert.parameters():
        p.requires_grad = False

    proj_dim = stage1_cfg.get("proj_dim", 2048)
    q_dim = base_model.model.get_input_embeddings().weight.shape[1]
    if args.router_type == "gated_attn":
        router = GatedAttentionDualRouter(
            q_dim=q_dim, video_dim=proj_dim, hidden_dim=args.router_hidden,
            num_heads=4, stats_dim=args.router_stats_dim,
            init_local=args.init_local, init_state=args.init_state,
        )
        log_rank0(rank, "  router=GatedAttentionDualRouter")
    else:
        router = DualBranchRouter(
            embed_dim=q_dim, hidden_dim=args.router_hidden,
            stats_dim=args.router_stats_dim,
            init_local=args.init_local, init_state=args.init_state,
            w_min=args.w_min, w_max=args.w_max, init_std=args.init_std,
        )
        log_rank0(rank, f"  router=DualBranchRouter (MLP, w=[{args.w_min},{args.w_max}], init_std={args.init_std})")
    router = router.to(device=device, dtype=torch.bfloat16)

    wrapped = PLMWithDualBranchRouter(
        base_model, expert, router, stage1_module_type,
        align_lambda=args.align_lambda,
        router_lambda=args.router_lambda,
        router_tau=args.router_tau,
        router_margin=args.router_margin,
        router_warmup_steps=args.router_warmup_steps,
        diversity_lambda=args.diversity_lambda,
        oracle_mix=args.oracle_mix,
    ).to(device=device)

    dataset = build_dataset(dataset_configs, processor, num_frames, max_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True, persistent_workers=True)

    trainable_params = list(router.parameters())
    optimizer = torch.optim.AdamW([{"params": trainable_params, "lr": args.adapter_lr}], weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, num_opt_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    adapter_config = {
        "module_type": SUPPORTED_STAGE1_TYPES[stage1_module_type],
        "stage1_module_type": stage1_module_type,
        "stage1_ckpt": args.stage1_ckpt,
        "stage2_style": "dual_branch_router",
        "router_type": args.router_type,
        "router_hidden": args.router_hidden,
        "router_stats_dim": args.router_stats_dim,
        "init_local": args.init_local,
        "init_state": args.init_state,
        "w_min": args.w_min,
        "w_max": args.w_max,
        "init_std": args.init_std,
        "router_tau": args.router_tau,
        "router_margin": args.router_margin,
        "router_lambda": args.router_lambda,
        "router_warmup_steps": args.router_warmup_steps,
        "align_lambda": args.align_lambda,
        "diversity_lambda": args.diversity_lambda,
        "oracle_mix": args.oracle_mix,
        "num_frames": num_frames,
        "use_llm_lora": False,
        "proj_dim": stage1_cfg.get("proj_dim", 2048),
        "pooling_ratio": stage1_cfg.get("pooling_ratio", 2),
        # Pass through stage1 expert config
        **{k: v for k, v in stage1_cfg.items() if k not in ("module_type",)},
    }

    t_start = time.time()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(999999):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= args.max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values_videos = batch["pixel_values_videos"].to(device=device, dtype=torch.bfloat16)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = wrapped(input_ids=input_ids, attention_mask=attention_mask,
                               pixel_values_videos=pixel_values_videos, labels=labels,
                               num_frames=num_frames, global_step=global_step)
            (loss / args.grad_accum).backward()

            if (global_step + 1) % args.grad_accum == 0 or global_step == args.max_steps - 1:
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
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                wrapped._last_grad = float(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if rank == 0 and global_step % args.log_interval == 0:
                m = wrapped
                elapsed = time.time() - t_start
                speed = elapsed / args.log_interval
                oracle_str = ""
                if hasattr(m, '_last_oracle_w_l') and m._last_oracle_w_l > 0:
                    oracle_str = (f"ow_l={m._last_oracle_w_l:.3f} ow_s={m._last_oracle_w_s:.3f} "
                                  f"ow_l_std={m._last_oracle_w_l_std:.3f} ow_s_std={m._last_oracle_w_s_std:.3f} ")
                log_rank0(rank,
                    f"[Step {global_step}/{args.max_steps}] "
                    f"qa={m._last_qa:.4f} align={m._last_align:.4f} rtr={m._last_router:.4f} "
                    f"div={m._last_diversity:.4f} "
                    f"w_l={m._last_w_l:.3f} w_s={m._last_w_s:.3f} "
                    f"w_l_std={m._last_w_l_std:.3f} w_s_std={m._last_w_s_std:.3f} "
                    f"{oracle_str}"
                    f"tgt_l={m._last_tgt_l:.3f} tgt_s={m._last_tgt_s:.3f} "
                    f"g={m._last_grad:.3f} lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"spd={speed:.1f}s"
                )
                t_start = time.time()

            if global_step % args.save_interval == 0 or global_step == args.max_steps:
                if rank == 0:
                    save_checkpoint(args.output_dir, global_step, expert, router, optimizer, scheduler, adapter_config)
                if dist.is_initialized():
                    dist.barrier()

        if global_step >= args.max_steps:
            break

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Dual Branch Router Stage 2")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_frames", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--adapter_lr", type=float, default=1e-3)
    p.add_argument("--router_type", type=str, default="mlp", choices=["mlp", "gated_attn"],
                   help="Router architecture: mlp (DualBranchRouter) or gated_attn (GatedAttentionDualRouter)")
    p.add_argument("--router_hidden", type=int, default=128)
    p.add_argument("--router_stats_dim", type=int, default=4)
    p.add_argument("--init_local", type=float, default=0.5)
    p.add_argument("--init_state", type=float, default=0.5)
    p.add_argument("--w_min", type=float, default=0.0, help="Lower bound for router weights")
    p.add_argument("--w_max", type=float, default=1.0, help="Upper bound for router weights (>1.0 allows amplification)")
    p.add_argument("--init_std", type=float, default=0.01, help="Init std for router output layer (higher = more initial diversity)")
    p.add_argument("--router_tau", type=float, default=0.5)
    p.add_argument("--router_margin", type=float, default=0.02)
    p.add_argument("--router_lambda", type=float, default=0.10)
    p.add_argument("--router_warmup_steps", type=int, default=50)
    p.add_argument("--align_lambda", type=float, default=0.03)
    p.add_argument("--diversity_lambda", type=float, default=0.0, help="Weight for diversity loss (encourages per-sample w variance)")
    p.add_argument("--oracle_mix", action="store_true", help="Use oracle w for z_final during training (Method A)")
    p.add_argument("--dataset_preset", type=str, default="full")
    p.add_argument("--no_liger", action="store_true")
    train(p.parse_args())
