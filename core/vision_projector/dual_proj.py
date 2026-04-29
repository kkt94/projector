"""
Dual Projector architectures for temporal video understanding.

Design A: DeltaNet Dual Projector
  - P_base (frozen) + P_temporal (P_base + LoRA) merged via DeltaNet recurrence
  - Safe init: starts identical to vanilla PLM-1B

Design B: DPM (Dual Projector Mixture)
  - P_base (frozen) + P_temp (clone + temporal block + LoRA on linear_2)
  - Gated interpolation: z = z_base + g * (z_temp - z_base)
  - DeltaNet inside temporal projector for memory

v3 changes:
  - Scaled sigmoid (no hard clamp) for all gates
  - Frame-wise alpha MLP option for DeltaNetMerger
  - RMSNorm on enrichment option
  - Channel-group gate for FusionGate
  - TemporalBlock gate init raised to -1.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
#  Design A: DeltaNet Dual Projector
# ════════════════════════════════════════════════════════════════

class ProjectorLoRA(nn.Module):
    """LoRA adapter for a frozen MLP projector.

    Wraps linear_1 and linear_2 with LoRA. GELU and pooling are shared.
    At init (B=0), output is identical to base projector.
    """

    def __init__(self, base_projector, rank=32, alpha=64.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        # Reference to frozen base (NOT registered as submodule to avoid param duplication)
        self._base = [base_projector]  # wrap in list to hide from nn.Module

        # LoRA for linear_1: (1024 -> 2048)
        in1 = base_projector.linear_1.in_features
        out1 = base_projector.linear_1.out_features
        self.lora_A1 = nn.Linear(in1, rank, bias=False)
        self.lora_B1 = nn.Linear(rank, out1, bias=False)

        # LoRA for linear_2: (2048 -> 2048)
        in2 = base_projector.linear_2.in_features
        out2 = base_projector.linear_2.out_features
        self.lora_A2 = nn.Linear(in2, rank, bias=False)
        self.lora_B2 = nn.Linear(rank, out2, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: A with kaiming, B with zeros -> starts as identity to base
        nn.init.kaiming_uniform_(self.lora_A1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B1.weight)
        nn.init.kaiming_uniform_(self.lora_A2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B2.weight)

    @property
    def base(self):
        return self._base[0]

    def forward(self, x):
        """
        x: (BT, N, C) in NLD format expected by projector
        Returns: (BT, M, D) after pooling (M=256, D=2048)
        """
        proj = self.base

        # NLD format: permute for linear layers
        # x: (BT, N, C) -> (N, BT, C)
        x_nld = x.permute(1, 0, 2)

        # linear_1 + LoRA_1
        with torch.no_grad():
            h_base = proj.linear_1(x_nld)
        h_lora = self.lora_B1(self.lora_A1(self.dropout(x_nld))) * self.scaling
        h = h_base + h_lora

        # GELU (shared, no params)
        h = proj.gelu(h)

        # linear_2 + LoRA_2
        out_base = proj.linear_2(h)
        out_lora = self.lora_B2(self.lora_A2(self.dropout(h))) * self.scaling
        out = out_base + out_lora

        # Back to (BT, N, D)
        out = out.permute(1, 0, 2)

        # Pooling (shared, no trainable params)
        out = proj.pooling(out)

        return out


class DeltaNetMerger(nn.Module):
    """Merges base and temporal projector outputs using DeltaNet recurrence.

    v3: scaled sigmoid (no hard clamp), optional frame-wise alpha, optional RMSNorm.
    """

    def __init__(self, proj_dim=2048, bottleneck=128, alpha_init=-2.0, max_alpha=0.15,
                 alpha_mode="scaled", use_enrich_norm=False):
        super().__init__()
        self.proj_dim = proj_dim
        self.bottleneck = bottleneck
        self.max_alpha = max_alpha
        self.alpha_mode = alpha_mode
        self.use_enrich_norm = use_enrich_norm

        # Project down: 2048 -> 128
        self.proj_down = nn.Linear(proj_dim, bottleneck)

        # DeltaNet parameters
        self.q_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.k_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.v_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.beta_proj = nn.Linear(bottleneck, bottleneck, bias=False)

        # Project up: 128 -> 2048
        self.proj_up = nn.Linear(bottleneck, proj_dim)

        # Alpha gate
        if alpha_mode == "frame_mlp":
            # Frame-wise alpha from DeltaNet output features
            self.alpha_head = nn.Sequential(
                nn.Linear(bottleneck, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            nn.init.zeros_(self.alpha_head[2].weight)
            self.alpha_head[2].bias.data.fill_(-3.0)  # max_alpha * sigmoid(-3) ≈ 0.007
        else:
            # Scalar alpha (scaled sigmoid: max_alpha * sigmoid(raw))
            self.alpha_raw = nn.Parameter(torch.tensor(alpha_init))

        # Optional enrichment normalization
        if use_enrich_norm:
            self.enrich_norm = nn.LayerNorm(proj_dim)

        # Init
        nn.init.xavier_uniform_(self.proj_up.weight, gain=0.01)
        nn.init.zeros_(self.proj_up.bias)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.beta_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)

    def gate_parameters(self):
        """Return gate-specific parameters for separate optimizer group."""
        if self.alpha_mode == "frame_mlp":
            return list(self.alpha_head.parameters())
        else:
            return [self.alpha_raw]

    def non_gate_parameters(self):
        """Return non-gate parameters."""
        gate_ids = {id(p) for p in self.gate_parameters()}
        return [p for p in self.parameters() if id(p) not in gate_ids]

    def forward(self, z_base, z_temp, return_extras=False):
        """
        z_base: (B, T, M, D) -- frozen base projector output
        z_temp: (B, T, M, D) -- temporal projector output
        return_extras: if True, return (result, enrichment, alpha_tensor) for v4 losses
        Returns: (B, T, M, D) -- enriched output (or tuple if return_extras)
        """
        B, T, M, D = z_base.shape

        # Compute difference
        diff = z_temp - z_base

        # Project down
        x = self.proj_down(diff)  # (B, T, M, bn)

        # DeltaNet recurrence along T, per spatial position
        x = x.permute(0, 2, 1, 3).reshape(B * M, T, self.bottleneck)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x))

        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)

        outputs = []
        S = torch.zeros(B * M, self.bottleneck, self.bottleneck,
                        device=x.device, dtype=x.dtype)

        for t in range(T):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]
            beta_t = beta[:, t, :]

            erase = torch.einsum('bi,bj->bij', beta_t, k_t)
            write = torch.einsum('bi,bj->bij', beta_t * v_t, k_t)
            S = S * (1.0 - erase) + write

            o_t = torch.einsum('bij,bj->bi', S, q_t)
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)  # (B*M, T, bn)
        output = output.reshape(B, M, T, self.bottleneck).permute(0, 2, 1, 3)  # (B, T, M, bn)

        # Store only spatially pooled recurrence for external use.
        # Keeping the full (B, T, M, bottleneck) tensor alive inflates memory,
        # especially when v5 runs multiple perturbation passes for discrimination.
        output_pool = output.mean(dim=2)  # (B, T, bottleneck)
        self._last_recurrence_pool = output_pool

        # Project up
        enrichment = self.proj_up(output)  # (B, T, M, D)

        if self.use_enrich_norm:
            enrichment = self.enrich_norm(enrichment)

        # Compute alpha (scaled sigmoid, no hard clamp)
        if self.alpha_mode == "frame_mlp":
            alpha = self.max_alpha * torch.sigmoid(self.alpha_head(output_pool))  # (B, T, 1)
            alpha_val = alpha.detach().mean().item()
            alpha_tensor = alpha.unsqueeze(2)  # (B, T, 1, 1)
        else:
            alpha = self.max_alpha * torch.sigmoid(self.alpha_raw)
            alpha_val = alpha.item()
            alpha_tensor = alpha

        result = z_base + alpha_tensor * enrichment

        # Logging cache
        self._last_alpha = alpha_val
        self._last_enrichment_norm = enrichment.detach().norm(dim=-1).mean().item()
        self._last_eff_scale = (alpha_tensor * enrichment).detach().norm(dim=-1).mean().item()
        self._last_diff_norm = diff.detach().norm(dim=-1).mean().item()

        if return_extras:
            return result, enrichment, alpha_tensor
        return result


class BinaryExpertRouter(nn.Module):
    """2-way router for base vs temporal expert selection.

    Used in Design A v12 Stage 2.
    """

    def __init__(self, embed_dim, hidden_dim=128, stats_dim=4):
        super().__init__()
        self.stats_dim = stats_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim + stats_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        with torch.no_grad():
            self.net[2].bias.data.copy_(torch.tensor([1.0, 0.0]))
            nn.init.normal_(self.net[2].weight, std=0.01)

    def forward(self, q_embed, video_stats=None):
        if self.stats_dim > 0:
            if video_stats is None:
                video_stats = torch.zeros(
                    q_embed.shape[0], self.stats_dim, device=q_embed.device, dtype=q_embed.dtype
                )
            x = torch.cat([q_embed, video_stats], dim=-1)
        else:
            x = q_embed
        return F.softmax(self.net(x), dim=-1)


class DualBranchRouter(nn.Module):
    """Independent local/state router for v928 and UDH v2 experts.

    Predicts w_local, w_state independently:
      z_final = z_base + w_local * local_res + w_state * state_res

    QA gradient flows directly through w_local/w_state to the router.

    Weight range: [w_min, w_max] via shifted-scaled sigmoid:
      w = w_min + (w_max - w_min) * sigmoid(logit)

    This ensures:
      - w_min > 0 prevents the router from completely removing expert residuals
      - w_max > 1 allows amplification beyond original magnitude
      - init_std > 0 seeds per-sample diversity from the start
    """

    def __init__(self, embed_dim, hidden_dim=128, stats_dim=4,
                 init_local=0.5, init_state=0.5,
                 w_min=0.0, w_max=1.0, init_std=0.01):
        super().__init__()
        self.stats_dim = stats_dim
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + stats_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # [local_logit, state_logit]
        )
        # Init logits so w_min + (w_max - w_min) * sigmoid(logit) → init values
        _range = self.w_max - self.w_min
        _init_l = max(1e-4, min((init_local - self.w_min) / _range, 1 - 1e-4))
        _init_s = max(1e-4, min((init_state - self.w_min) / _range, 1 - 1e-4))
        init_l_logit = math.log(_init_l / (1 - _init_l))
        init_s_logit = math.log(_init_s / (1 - _init_s))
        with torch.no_grad():
            nn.init.normal_(self.net[2].weight, std=float(init_std))
            self.net[2].bias.data.copy_(torch.tensor([init_l_logit, init_s_logit]))

    def forward(self, q_embed, video_stats=None, v_frames=None, return_logits=False):
        if self.stats_dim > 0:
            if video_stats is None:
                video_stats = torch.zeros(
                    q_embed.shape[0], self.stats_dim, device=q_embed.device, dtype=q_embed.dtype
                )
            x = torch.cat([q_embed, video_stats], dim=-1)
        else:
            x = q_embed
        logits = self.net(x)  # (B, 2)
        weights = self.w_min + (self.w_max - self.w_min) * torch.sigmoid(logits)  # (B, 2) ∈ [w_min, w_max]
        if return_logits:
            return weights, logits
        return weights


class GatedAttentionDualRouter(nn.Module):
    """Gated-Attention router for independent local/state routing.

    Combines the Gated Attention mechanism (NeurIPS 2025) with the
    dbrouter training paradigm (QA gradient + BCE supervision).

    Architecture:
      1. q→proj→norm, v→proj→norm
      2. Multi-head attention: q attends over v frames
      3. Gated Attention: gate = sigmoid(v_mean @ W_gate)
         output = attended * gate
      4. [q, gated_out, q*gated_out] → trunk → heads → logits
      5. sigmoid(logits) → [w_local, w_state] ∈ [0,1]

    Same interface as DualBranchRouter: forward(q_embed, video_stats, v_frames)
    """

    def __init__(self, q_dim, video_dim, hidden_dim=128, num_heads=4,
                 stats_dim=4, init_local=0.5, init_state=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.stats_dim = stats_dim
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        init_l_logit = math.log(max(init_local, 1e-4) / max(1 - init_local, 1e-4))
        init_s_logit = math.log(max(init_state, 1e-4) / max(1 - init_state, 1e-4))

        # Input projections
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)

        # Attention Q/K/V
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Gated Attention gate
        self.W_gate = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.W_gate.bias)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Decision trunk: [q, gated_out, q*gated_out, stats] → 2 logits
        trunk_input_dim = hidden_dim * 3 + stats_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # [local_logit, state_logit]
        )
        # Init: small weight so logit ≈ bias at start
        with torch.no_grad():
            nn.init.normal_(self.trunk[2].weight, std=0.01)
            self.trunk[2].bias.data.copy_(torch.tensor([init_l_logit, init_s_logit]))

    def forward(self, q_embed, video_stats=None, v_frames=None, return_logits=False):
        """
        Args:
            q_embed:     (B, q_dim) — question embedding
            video_stats: (B, stats_dim) — log1p stats (same as DualBranchRouter)
            v_frames:    (B, T, video_dim) — per-frame video features (optional)
            return_logits: if True, return (weights, logits)
        """
        B = q_embed.shape[0]

        if video_stats is None:
            video_stats = torch.zeros(B, self.stats_dim, device=q_embed.device, dtype=q_embed.dtype)

        # If no v_frames, fallback to stats-only (like DualBranchRouter)
        if v_frames is None:
            # Simple linear path without attention
            q_h = self.q_norm(self.q_proj(q_embed))
            h = torch.cat([q_h, q_h, q_h * q_h, video_stats], dim=-1)
            logits = self.trunk(h)
            weights = torch.sigmoid(logits)
            return (weights, logits) if return_logits else weights

        q_h = self.q_norm(self.q_proj(q_embed))
        v_h = self.v_norm(self.v_proj(v_frames))
        T = v_frames.shape[1]

        # Multi-head attention: q attends over v
        Q = self.W_q(q_h).unsqueeze(1).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(v_h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(v_h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, self.hidden_dim).squeeze(1)

        # Gated Attention
        v_mean = v_h.mean(dim=1)
        gate = torch.sigmoid(self.W_gate(v_mean))
        gated_out = self.out_proj(attn_out * gate)

        # Decision: [q, gated_video, q*gated_video, stats] → logits
        h = torch.cat([q_h, gated_out, q_h * gated_out, video_stats], dim=-1)
        logits = self.trunk(h)
        weights = torch.sigmoid(logits)

        if return_logits:
            return weights, logits
        return weights


class TwoTowerBinaryExpertRouter(nn.Module):
    """Balanced two-tower router for base vs temporal expert selection."""

    def __init__(self, q_dim, video_dim, hidden_dim=128):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        with torch.no_grad():
            self.out[-1].bias.data.copy_(torch.tensor([1.0, 0.0]))
            nn.init.normal_(self.out[-1].weight, std=0.01)

    def forward(self, q_embed, video_feat):
        q_h = self.q_proj(q_embed)
        v_h = self.v_proj(video_feat)
        h = torch.cat([q_h, v_h, q_h * v_h], dim=-1)
        return F.softmax(self.out(h), dim=-1)


class TwoTowerTemporalAlphaCalibrator(nn.Module):
    """Question/video-conditioned scalar calibrator for temporal residual use.

    Instead of choosing between base and temporal variants, this module predicts
    a single alpha in [0, 1] and forms:

      z_final = z_base + alpha * (z_temp - z_base)

    The design is intentionally balanced: question and video features are first
    projected into the same hidden dimension before interaction.
    """

    def __init__(self, q_dim, video_dim, hidden_dim=128, init_alpha=0.70):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        init_alpha = float(min(max(init_alpha, 1e-3), 1.0 - 1e-3))
        init_logit = math.log(init_alpha / (1.0 - init_alpha))
        with torch.no_grad():
            nn.init.zeros_(self.out[-1].weight)
            self.out[-1].bias.fill_(init_logit)

    def forward(self, q_embed, video_feat):
        q_h = self.q_proj(q_embed)
        v_h = self.v_proj(video_feat)
        h = torch.cat([q_h, v_h, q_h * v_h], dim=-1)
        return torch.sigmoid(self.out(h))


class TwoTowerResidualAlphaCalibrator(nn.Module):
    """Balanced two-tower calibrator with bounded residual alpha updates.

    Instead of predicting alpha directly, this module learns a small correction
    around a prior alpha:

      alpha = clamp(alpha_prior + max_delta * tanh(f(q, v)), 0, 1)

    This keeps the stage2 problem focused on calibration rather than collapsing
    to a near-constant temp-only solution.
    """

    def __init__(self, q_dim, video_dim, hidden_dim=128, alpha_prior=0.75, max_delta=0.20):
        super().__init__()
        self.alpha_prior = float(alpha_prior)
        self.max_delta = float(max_delta)
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)

    def forward(self, q_embed, video_feat, return_delta=False):
        q_h = self.q_proj(q_embed)
        v_h = self.v_proj(video_feat)
        h = torch.cat([q_h, v_h, q_h * v_h], dim=-1)
        delta = self.max_delta * torch.tanh(self.out(h))
        alpha = torch.clamp(self.alpha_prior + delta, min=0.0, max=1.0)
        if return_delta:
            return alpha, delta
        return alpha


class TwoTowerDualResidualCalibrator(nn.Module):
    """Cross-attention calibrator that predicts independent a_local and a_state.

    The question embedding attends over video frame representations to build
    a temporal-aware context, then predicts per-branch residual scales.

    This lets the calibrator learn "which parts of the video matter for this
    question" before deciding local/state balance.
    """

    def __init__(self, q_dim, video_dim, hidden_dim=128, num_heads=4,
                 init_local=1.0, init_state=1.0,
                 max_local_delta=0.35, max_state_delta=0.35):
        super().__init__()
        # Sigmoid parameterization: a = a_min + (a_max - a_min) * sigmoid(logit)
        # Always has non-zero gradient → no dead-gradient at boundaries
        self.a_local_min = max(0.0, float(init_local) - float(max_local_delta))
        self.a_local_max = float(init_local) + float(max_local_delta)
        self.a_state_min = max(0.0, float(init_state) - float(max_state_delta))
        self.a_state_max = float(init_state) + float(max_state_delta)

        # Compute init logit so sigmoid(logit) = (init - min) / (max - min)
        def _init_logit(init_val, lo, hi):
            t = (float(init_val) - lo) / (hi - lo)
            t = max(1e-4, min(1.0 - 1e-4, t))
            return math.log(t / (1.0 - t))

        self._init_local_logit = _init_logit(init_local, self.a_local_min, self.a_local_max)
        self._init_state_logit = _init_logit(init_state, self.a_state_min, self.a_state_max)

        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention: question attends over video frames
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)

        NUM_STATS = 6

        # === PRIMARY: Stats MLP (6 → 16 → 2 logits) ===
        self.stats_mlp = nn.Sequential(
            nn.Linear(NUM_STATS, 16),
            nn.GELU(),
        )
        self.stats_local_head = nn.Linear(16, 1)
        self.stats_state_head = nn.Linear(16, 1)

        # === AUXILIARY: Deep path (cross-attn), scaled down ===
        self.deep_scale = 0.1
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.deep_local_head = nn.Linear(hidden_dim, 1)
        self.deep_state_head = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            nn.init.zeros_(self.deep_local_head.weight)
            nn.init.zeros_(self.deep_local_head.bias)
            nn.init.zeros_(self.deep_state_head.weight)
            nn.init.zeros_(self.deep_state_head.bias)
            # Stats MLP first layer: small init to handle raw stats (magnitudes ~0.3-15)
            # std=0.02 → hidden std ≈ 0.02*√6*10 ≈ 0.5 → GELU → ~0.5
            nn.init.normal_(self.stats_mlp[0].weight, std=0.02)
            nn.init.zeros_(self.stats_mlp[0].bias)
            # Stats heads: non-zero init so gradient flows to MLP from step 1
            nn.init.normal_(self.stats_local_head.weight, std=0.1)
            nn.init.zeros_(self.stats_local_head.bias)
            nn.init.normal_(self.stats_state_head.weight, std=0.1)
            nn.init.zeros_(self.stats_state_head.bias)

    def forward(self, q_embed, v_frames, sample_stats=None):
        """
        Args:
            q_embed: (B, q_dim) — mean-pooled question embedding
            v_frames: (B, T, video_dim) — per-frame video representations
            sample_stats: (B, 6) — batch-normalized per-sample statistics
        """
        with torch.amp.autocast("cuda", enabled=False):
            _dtype = self.q_proj.weight.dtype
            q_embed = q_embed.to(_dtype)
            v_frames = v_frames.to(_dtype)

            # === PRIMARY: Stats path ===
            if sample_stats is not None:
                stats = sample_stats.to(_dtype)
                stats_h = self.stats_mlp(stats)
                stats_local = self.stats_local_head(stats_h).squeeze(-1)
                stats_state = self.stats_state_head(stats_h).squeeze(-1)
            else:
                B = q_embed.shape[0]
                stats_local = torch.zeros(B, device=q_embed.device, dtype=_dtype)
                stats_state = torch.zeros(B, device=q_embed.device, dtype=_dtype)

            # === AUXILIARY: Deep path ===
            q_h = self.q_norm(self.q_proj(q_embed)).unsqueeze(1)
            v_h = self.v_norm(self.v_proj(v_frames))
            attended, _ = self.cross_attn(q_h, v_h, v_h, need_weights=False)
            attended = attended.squeeze(1)
            q_flat = q_h.squeeze(1)
            h = self.trunk(torch.cat([q_flat, attended, q_flat * attended], dim=-1))
            deep_local = self.deep_local_head(h).squeeze(-1) * self.deep_scale
            deep_state = self.deep_state_head(h).squeeze(-1) * self.deep_scale

            local_logit = stats_local + deep_local + self._init_local_logit
            state_logit = stats_state + deep_state + self._init_state_logit
            a_local = self.a_local_min + (self.a_local_max - self.a_local_min) * torch.sigmoid(local_logit)
            a_state = self.a_state_min + (self.a_state_max - self.a_state_min) * torch.sigmoid(state_logit)
        return {
            "a_local": a_local,
            "a_state": a_state,
        }


class GatedAttentionCalibrator(nn.Module):
    """Gated-Attention calibrator (NeurIPS 2025 Best Paper style).

    Instead of cross-attention, uses the Gated Attention mechanism:
        attn_out = softmax(Q K^T / sqrt(d)) V
        gated    = attn_out * sigmoid(X @ W_gate)          ← input-dependent gate

    The gate lets the model suppress or amplify attended features *before*
    they are used to predict a_local / a_state.  This decouples "what to
    read from the video" (attention) from "how much to trust what was read"
    (gate), giving the network a more expressive calibration path with
    minimal extra parameters (just W_gate + b_gate per head pair).

    Architecture:
        1. Project q→H, v→H
        2. Standard dot-product attention: q attends over v frames
        3. Gated Attention: gate = sigmoid(v_mean @ W_gate)
           output = attended * gate                       (element-wise)
        4. Trunk MLP → local_head / state_head → sigmoid-bounded [a_min, a_max]
    """

    def __init__(self, q_dim, video_dim, hidden_dim=128, num_heads=4,
                 init_local=1.0, init_state=1.0,
                 max_local_delta=0.35, max_state_delta=0.35):
        super().__init__()
        # Sigmoid parameterization: a = a_min + (a_max - a_min) * sigmoid(logit)
        self.a_local_min = max(0.0, float(init_local) - float(max_local_delta))
        self.a_local_max = float(init_local) + float(max_local_delta)
        self.a_state_min = max(0.0, float(init_state) - float(max_state_delta))
        self.a_state_max = float(init_state) + float(max_state_delta)

        def _init_logit(init_val, lo, hi):
            t = (float(init_val) - lo) / (hi - lo)
            t = max(1e-4, min(1.0 - 1e-4, t))
            return math.log(t / (1.0 - t))

        self._init_local_logit = _init_logit(init_local, self.a_local_min, self.a_local_max)
        self._init_state_logit = _init_logit(init_state, self.a_state_min, self.a_state_max)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        # Input projections
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)

        # Attention Q/K/V projections (within hidden_dim space)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Gated Attention gate: sigmoid(X @ W_gate)
        # X = video representation (input to attention, NOT output)
        self.W_gate = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.W_gate.bias)

        # Output projection after gated attention
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        NUM_STATS = 6  # number of handcrafted per-sample statistics

        # === PRIMARY: Stats MLP (6 → 16 → 2 logits) ===
        # This is the main routing path — small, direct, high per-sample variance
        self.stats_mlp = nn.Sequential(
            nn.Linear(NUM_STATS, 16),
            nn.GELU(),
        )
        self.stats_local_head = nn.Linear(16, 1)
        self.stats_state_head = nn.Linear(16, 1)

        # === AUXILIARY: Deep path (q + v attention) → 2 logits, scaled down ===
        # Provides nuanced q-v interaction but contributes 10% initially
        self.deep_scale = 0.1  # shrink deep contribution
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.deep_local_head = nn.Linear(hidden_dim, 1)
        self.deep_state_head = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            # Deep path: zero-init (starts silent)
            nn.init.zeros_(self.deep_local_head.weight)
            nn.init.zeros_(self.deep_local_head.bias)
            nn.init.zeros_(self.deep_state_head.weight)
            nn.init.zeros_(self.deep_state_head.bias)
            # Stats MLP first layer: small init to handle raw stats (magnitudes ~0.3-15)
            nn.init.normal_(self.stats_mlp[0].weight, std=0.02)
            nn.init.zeros_(self.stats_mlp[0].bias)
            # Stats heads: non-zero init so gradient flows to MLP from step 1
            nn.init.normal_(self.stats_local_head.weight, std=0.1)
            nn.init.zeros_(self.stats_local_head.bias)
            nn.init.normal_(self.stats_state_head.weight, std=0.1)
            nn.init.zeros_(self.stats_state_head.bias)

    def forward(self, q_embed, v_frames, sample_stats=None):
        """
        Args:
            q_embed:  (B, q_dim)        — mean-pooled question embedding
            v_frames: (B, T, video_dim) — per-frame video representations
            sample_stats: (B, 6) — batch-normalized per-sample statistics
        Returns:
            dict with a_local, a_state
        """
        with torch.amp.autocast("cuda", enabled=False):
            _dtype = self.q_proj.weight.dtype
            q_embed = q_embed.to(_dtype)
            v_frames = v_frames.to(_dtype)
            B, T, _ = v_frames.shape

            # === PRIMARY: Stats path (2 layers, 6→16→1) ===
            if sample_stats is not None:
                stats = sample_stats.to(_dtype)
                stats_h = self.stats_mlp(stats)  # (B, 16) — high per-sample variance
                stats_local = self.stats_local_head(stats_h).squeeze(-1)  # (B,)
                stats_state = self.stats_state_head(stats_h).squeeze(-1)  # (B,)
            else:
                stats_local = torch.zeros(B, device=q_embed.device, dtype=_dtype)
                stats_state = torch.zeros(B, device=q_embed.device, dtype=_dtype)

            # === AUXILIARY: Deep path (gated attention over video) ===
            q_h = self.q_norm(self.q_proj(q_embed))
            v_h = self.v_norm(self.v_proj(v_frames))

            Q = self.W_q(q_h).unsqueeze(1)
            K = self.W_k(v_h)
            V = self.W_v(v_h)

            Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out = torch.matmul(attn_weights, V)
            attn_out = attn_out.transpose(1, 2).reshape(B, 1, self.hidden_dim).squeeze(1)

            v_mean = v_h.mean(dim=1)
            gate = torch.sigmoid(self.W_gate(v_mean))
            gated_out = self.out_proj(attn_out * gate)

            h = self.trunk(torch.cat([q_h, gated_out, q_h * gated_out], dim=-1))
            deep_local = self.deep_local_head(h).squeeze(-1) * self.deep_scale
            deep_state = self.deep_state_head(h).squeeze(-1) * self.deep_scale

            # === Combine: stats (primary) + deep (auxiliary) + init ===
            local_logit = stats_local + deep_local + self._init_local_logit
            state_logit = stats_state + deep_state + self._init_state_logit

            a_local = self.a_local_min + (self.a_local_max - self.a_local_min) * torch.sigmoid(local_logit)
            a_state = self.a_state_min + (self.a_state_max - self.a_state_min) * torch.sigmoid(state_logit)
        return {
            "a_local": a_local,
            "a_state": a_state,
        }


class QueryConditionedSlotTemporalGate(nn.Module):
    """Question-conditioned gated attention over temporal slot evidence.

    The module produces:
      - attention over time/slot evidence tokens
      - a global usage scalar
      - per-token gates derived from attention mass

    This is designed for stage2 usage control when stage1 already provides
    temporal state slots (e.g. a-v922 / a-v923).
    """

    def __init__(self, q_dim, evidence_dim, hidden_dim=128, init_alpha=0.75):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.e_proj = nn.Linear(evidence_dim, hidden_dim)
        self.score_bias = nn.Linear(hidden_dim, 1)
        self.global_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        init_alpha = float(min(max(init_alpha, 1e-3), 1.0 - 1e-3))
        init_logit = math.log(init_alpha / (1.0 - init_alpha))
        with torch.no_grad():
            nn.init.zeros_(self.global_head[-1].weight)
            self.global_head[-1].bias.fill_(init_logit)
            nn.init.zeros_(self.score_bias.weight)
            nn.init.zeros_(self.score_bias.bias)

    def forward(self, q_embed, evidence, return_parts=False):
        # q_embed: (B, Dq), evidence: (B, L, De)
        q_h = self.q_proj(q_embed)
        e_h = self.e_proj(evidence)
        logits = (q_h.unsqueeze(1) * e_h).sum(dim=-1) / math.sqrt(e_h.shape[-1])
        logits = logits + self.score_bias(e_h).squeeze(-1)
        attn = F.softmax(logits, dim=-1)
        context = torch.einsum("bl,bld->bd", attn, e_h)
        global_alpha = torch.sigmoid(
            self.global_head(torch.cat([q_h, context, q_h * context], dim=-1))
        )
        num_tokens = float(max(1, evidence.shape[1]))
        gates = torch.clamp(global_alpha * num_tokens * attn, min=0.0, max=1.0)
        if return_parts:
            return gates, global_alpha, attn
        return gates


class TwoTowerExpertGateModulator(nn.Module):
    """Question/video-conditioned modulation over frozen v9.2-style expert internals.

    The module predicts small bounded logit biases for:
      - local temporal gate
      - state gate
      - slot readout logits

    This keeps stage2 focused on *how to use* the expert rather than deciding
    between base/temp endpoints after the fact.
    """

    def __init__(
        self,
        q_dim,
        video_dim,
        num_slots,
        hidden_dim=128,
        max_local_bias=0.75,
        max_state_bias=0.75,
        max_slot_bias=0.75,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.max_local_bias = float(max_local_bias)
        self.max_state_bias = float(max_state_bias)
        self.max_slot_bias = float(max_slot_bias)

        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.trunk = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
        )
        self.local_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Linear(hidden_dim, 1)
        self.slot_head = nn.Linear(hidden_dim, self.num_slots)

        with torch.no_grad():
            nn.init.zeros_(self.local_head.weight)
            nn.init.zeros_(self.local_head.bias)
            nn.init.zeros_(self.state_head.weight)
            nn.init.zeros_(self.state_head.bias)
            nn.init.zeros_(self.slot_head.weight)
            nn.init.zeros_(self.slot_head.bias)

    def forward(self, q_embed, video_feat, return_parts=False):
        target_dtype = self.q_proj.weight.dtype
        if q_embed.dtype != target_dtype:
            q_embed = q_embed.to(dtype=target_dtype)
        if video_feat.dtype != target_dtype:
            video_feat = video_feat.to(dtype=target_dtype)
        q_h = self.q_proj(q_embed)
        v_h = self.v_proj(video_feat)
        h = self.trunk(torch.cat([q_h, v_h, q_h * v_h], dim=-1))

        local_bias = self.max_local_bias * torch.tanh(self.local_head(h)).squeeze(-1)
        state_bias = self.max_state_bias * torch.tanh(self.state_head(h)).squeeze(-1)
        slot_bias = self.max_slot_bias * torch.tanh(self.slot_head(h))
        slot_bias = slot_bias - slot_bias.mean(dim=-1, keepdim=True)

        modulation = {
            "local_bias": local_bias,
            "state_bias": state_bias,
            "slot_bias": slot_bias,
        }
        if return_parts:
            return modulation, h
        return modulation


class TwoTowerResidualComposer(nn.Module):
    """Question/video-conditioned residual composer over frozen v9.2-style experts.

    The module predicts:
      - a_local: multiplier for local residual
      - a_state: multiplier for state-conditioned residual
      - slot_bias: additive bias over slot readout logits

    It starts as near no-op: a_local ~= 1, a_state ~= 1, slot_bias ~= 0.
    """

    def __init__(
        self,
        q_dim,
        video_dim,
        num_slots,
        hidden_dim=128,
        max_local_delta=0.75,
        max_state_delta=0.75,
        max_slot_bias=0.75,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.max_local_delta = float(max_local_delta)
        self.max_state_delta = float(max_state_delta)
        self.max_slot_bias = float(max_slot_bias)

        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.trunk = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
        )
        self.local_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Linear(hidden_dim, 1)
        self.slot_head = nn.Linear(hidden_dim, self.num_slots)

        with torch.no_grad():
            nn.init.zeros_(self.local_head.weight)
            nn.init.zeros_(self.local_head.bias)
            nn.init.zeros_(self.state_head.weight)
            nn.init.zeros_(self.state_head.bias)
            nn.init.zeros_(self.slot_head.weight)
            nn.init.zeros_(self.slot_head.bias)

    def forward(self, q_embed, video_feat, return_parts=False):
        target_dtype = self.q_proj.weight.dtype
        if q_embed.dtype != target_dtype:
            q_embed = q_embed.to(dtype=target_dtype)
        if video_feat.dtype != target_dtype:
            video_feat = video_feat.to(dtype=target_dtype)

        q_h = self.q_proj(q_embed)
        v_h = self.v_proj(video_feat)
        h = self.trunk(torch.cat([q_h, v_h, q_h * v_h], dim=-1))

        a_local = 1.0 + self.max_local_delta * torch.tanh(self.local_head(h)).squeeze(-1)
        a_state = 1.0 + self.max_state_delta * torch.tanh(self.state_head(h)).squeeze(-1)
        slot_bias = self.max_slot_bias * torch.tanh(self.slot_head(h))
        slot_bias = slot_bias - slot_bias.mean(dim=-1, keepdim=True)

        compose = {
            "a_local": a_local,
            "a_state": a_state,
            "slot_bias": slot_bias,
        }
        if return_parts:
            return compose, h
        return compose


class TwoTowerResidualUsageComposer(nn.Module):
    """Question/video-conditioned usage composer over local/state residuals.

    Outputs bounded usage coefficients in [0, 1] for:
      - local residual
      - state residual

    This is intended for lightweight stage2 distillation where a frozen model
    acts as a scorer and only this small controller is trained.
    """

    def __init__(self, q_dim, video_dim, hidden_dim=128):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.trunk = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
        )
        self.local_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            nn.init.zeros_(self.local_head.weight)
            nn.init.zeros_(self.local_head.bias)
            nn.init.zeros_(self.state_head.weight)
            nn.init.zeros_(self.state_head.bias)

    def forward(self, q_embed, video_feat, return_parts=False):
        target_dtype = self.q_proj.weight.dtype
        if q_embed.dtype != target_dtype:
            q_embed = q_embed.to(dtype=target_dtype)
        if video_feat.dtype != target_dtype:
            video_feat = video_feat.to(dtype=target_dtype)

        q_h = self.q_proj(q_embed)
        v_h = self.v_proj(video_feat)
        h = self.trunk(torch.cat([q_h, v_h, q_h * v_h], dim=-1))

        u_local = torch.sigmoid(self.local_head(h)).squeeze(-1)
        u_state = torch.sigmoid(self.state_head(h)).squeeze(-1)
        compose = {
            "u_local": u_local,
            "u_state": u_state,
        }
        if return_parts:
            return compose, h
        return compose


class TwoTowerResidualSimplexMixer(nn.Module):
    """Question/video-conditioned simplex mixer over stage1 candidates.

    The mixer supports configurable initial candidate priors so stage2 can start
    from an explicit base/local/state or base/local/state/full mixture rather
    than an arbitrary "last candidate wins" bias.
    """

    def __init__(self, q_dim, video_dim, hidden_dim=128, num_candidates=4, init_probs=None):
        super().__init__()
        self.num_candidates = int(num_candidates)
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.trunk = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden_dim, self.num_candidates)

        with torch.no_grad():
            nn.init.zeros_(self.out.weight)
            if init_probs is None:
                bias = torch.zeros(self.num_candidates)
                if self.num_candidates >= 4:
                    bias[3] = 1.0
                else:
                    bias[-1] = 1.0
            else:
                init_probs = torch.as_tensor(init_probs, dtype=torch.float32)
                if init_probs.numel() != self.num_candidates:
                    raise ValueError(
                        f"init_probs length {init_probs.numel()} does not match "
                        f"num_candidates={self.num_candidates}"
                    )
                init_probs = init_probs / init_probs.sum().clamp(min=1e-6)
                bias = torch.log(init_probs.clamp(min=1e-6))
            self.out.bias.copy_(bias)

    def forward(self, q_embed, video_feat, return_parts=False):
        target_dtype = self.q_proj.weight.dtype
        if q_embed.dtype != target_dtype:
            q_embed = q_embed.to(dtype=target_dtype)
        if video_feat.dtype != target_dtype:
            video_feat = video_feat.to(dtype=target_dtype)

        q_h = self.q_proj(q_embed)
        v_h = self.v_proj(video_feat)
        h = self.trunk(torch.cat([q_h, v_h, q_h * v_h], dim=-1))
        logits = self.out(h)
        weights = F.softmax(logits, dim=-1)
        if return_parts:
            return {"weights": weights, "logits": logits}, h
        return {"weights": weights, "logits": logits}


class GlobalResidualScaler(nn.Module):
    """Global 2-scalar residual scaler over local/state residuals.

    This is the simplest possible stage2:
      z_final = z_base + a_local * local_residual + a_state * state_residual

    Both scalars are shared across all samples and initialized to 1.0.
    """

    def __init__(self, init_local=1.0, init_state=1.0):
        super().__init__()
        self.logit_local = nn.Parameter(torch.tensor(float(init_local)).atanh() if abs(init_local) < 1 else torch.tensor(3.0 if init_local > 0 else -3.0))
        self.logit_state = nn.Parameter(torch.tensor(float(init_state)).atanh() if abs(init_state) < 1 else torch.tensor(3.0 if init_state > 0 else -3.0))

    def forward(self, batch_size=None, dtype=None, device=None):
        # Map unconstrained parameters to positive usage around 1.0
        # 0 -> 1, large positive -> ~2, large negative -> ~0
        a_local = 1.0 + torch.tanh(self.logit_local)
        a_state = 1.0 + torch.tanh(self.logit_state)
        if batch_size is None:
            return {"a_local": a_local, "a_state": a_state}
        a_local = a_local.to(device=device, dtype=dtype).expand(batch_size)
        a_state = a_state.to(device=device, dtype=dtype).expand(batch_size)
        return {"a_local": a_local, "a_state": a_state}


class QueryResidualScaler(nn.Module):
    """Question-conditioned 2-scalar residual scaler.

    This is the smallest question-aware stage2:
      z_final = z_base + a_local(q) * local_residual + a_state(q) * state_residual
    """

    def __init__(self, q_dim, hidden_dim=128, max_local_delta=0.75, max_state_delta=0.75):
        super().__init__()
        self.max_local_delta = float(max_local_delta)
        self.max_state_delta = float(max_state_delta)
        self.net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.local_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            nn.init.zeros_(self.local_head.weight)
            nn.init.zeros_(self.local_head.bias)
            nn.init.zeros_(self.state_head.weight)
            nn.init.zeros_(self.state_head.bias)

    def forward(self, q_embed):
        target_dtype = self.net[0].weight.dtype
        if q_embed.dtype != target_dtype:
            q_embed = q_embed.to(dtype=target_dtype)
        h = self.net(q_embed)
        a_local = 1.0 + self.max_local_delta * torch.tanh(self.local_head(h)).squeeze(-1)
        a_state = 1.0 + self.max_state_delta * torch.tanh(self.state_head(h)).squeeze(-1)
        return {"a_local": a_local, "a_state": a_state}


class QueryConditionedMemoryReadout(nn.Module):
    """Query-conditioned readout over base/motion/state/correspondence memories."""

    def __init__(self, embed_dim, hidden_dim=128, stats_dim=6, num_slots=4):
        super().__init__()
        self.stats_dim = stats_dim
        self.num_slots = num_slots
        self.net = nn.Sequential(
            nn.Linear(embed_dim + stats_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_slots),
        )
        with torch.no_grad():
            nn.init.zeros_(self.net[2].weight)
            bias = torch.zeros(num_slots)
            bias[0] = 1.0
            self.net[2].bias.copy_(bias)

    def forward(self, q_embed, video_stats=None):
        if self.stats_dim > 0:
            if video_stats is None:
                video_stats = torch.zeros(
                    q_embed.shape[0], self.stats_dim, device=q_embed.device, dtype=q_embed.dtype
                )
            x = torch.cat([q_embed, video_stats], dim=-1)
        else:
            x = q_embed
        return F.softmax(self.net(x), dim=-1)


class MultiExpertUsageRouter(nn.Module):
    """Balanced two-tower router for base/motion/state/structure usage."""

    def __init__(self, q_dim, video_dim, hidden_dim=128, num_experts=4):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )
        with torch.no_grad():
            nn.init.zeros_(self.out[-1].weight)
            bias = torch.zeros(num_experts)
            bias[0] = 1.0
            self.out[-1].bias.copy_(bias)

    def forward(self, q_embed, video_feat):
        q_h = self.q_proj(q_embed)
        v_h = self.v_proj(video_feat)
        h = torch.cat([q_h, v_h, q_h * v_h], dim=-1)
        return F.softmax(self.out(h), dim=-1)


class HierarchicalResidualRouter(nn.Module):
    """Hierarchical router: base vs temporal, then local vs state inside temporal."""

    def __init__(self, q_dim, video_dim, hidden_dim=128):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)
        self.trunk = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
        )
        self.temp_head = nn.Linear(hidden_dim, 1)
        self.branch_head = nn.Linear(hidden_dim, 2)
        with torch.no_grad():
            nn.init.zeros_(self.temp_head.weight)
            self.temp_head.bias.fill_(0.0)
            nn.init.zeros_(self.branch_head.weight)
            self.branch_head.bias.copy_(torch.tensor([0.0, 0.0]))

    def forward(self, q_embed, video_feat):
        q_h = self.q_proj(q_embed)
        v_h = self.v_proj(video_feat)
        h = torch.cat([q_h, v_h, q_h * v_h], dim=-1)
        h = self.trunk(h)
        p_temp = torch.sigmoid(self.temp_head(h)).squeeze(-1)
        p_ls = F.softmax(self.branch_head(h), dim=-1)
        w_base = 1.0 - p_temp
        w_local = p_temp * p_ls[:, 0]
        w_state = p_temp * p_ls[:, 1]
        weights = torch.stack([w_base, w_local, w_state], dim=-1)
        return {
            "weights": weights,
            "p_temp": p_temp,
            "p_local": p_ls[:, 0],
            "p_state": p_ls[:, 1],
        }


class _GatedDeltaSequenceEncoder(nn.Module):
    """Lightweight gated DeltaNet-style sequence encoder."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, mask=None):
        # x: (B, L, H)
        B, L, H = x.shape
        orig_dtype = x.dtype
        x_f = x.float()
        q = self.q_proj(x_f)
        k = F.elu(self.k_proj(x_f)) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)
        v = self.v_proj(x_f)
        beta = torch.sigmoid(self.beta_proj(x_f))

        S = torch.zeros(B, H, H, device=x.device, dtype=torch.float32)
        outputs = []
        for t in range(L):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]
            beta_t = beta[:, t, :]
            erase = torch.einsum("bi,bj->bij", beta_t, k_t).clamp(0.0, 1.0)
            write = torch.einsum("bi,bj->bij", beta_t * v_t, k_t)
            S_new = S * (1.0 - erase) + write
            S_new = torch.nan_to_num(S_new, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
            o_t = torch.einsum("bij,bj->bi", S_new, q_t).clamp(-1e3, 1e3)
            if mask is not None:
                m_t = mask[:, t].to(dtype=torch.float32).view(B, 1)
                S = S_new * m_t.view(B, 1, 1) + S * (1.0 - m_t).view(B, 1, 1)
                o_t = o_t * m_t
            else:
                S = S_new
            outputs.append(o_t)
        out = torch.stack(outputs, dim=1)
        out = torch.tanh(out / 10.0)
        out = self.out_norm(out.to(dtype=orig_dtype))
        return 0.25 * self.out_proj(out)


class RawContextRouterEncoder(nn.Module):
    """Gated-DeltaNet router encoder over raw question/video token sequences.

    Question tokens are read as a sequence controller.
    Video tokens are read as token-position trajectories across time.
    The question controller then gates spatial token trajectories to build a
    routing state without hand-crafted scalar summaries.
    """

    def __init__(self, token_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        del num_heads
        self.q_proj = nn.Linear(token_dim, hidden_dim)
        self.v_proj = nn.Linear(token_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.q_delta = _GatedDeltaSequenceEncoder(hidden_dim)
        self.v_delta = _GatedDeltaSequenceEncoder(hidden_dim)
        self.token_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        with torch.no_grad():
            nn.init.zeros_(self.token_gate[-1].weight)
            nn.init.zeros_(self.token_gate[-1].bias)

    def forward(self, q_tokens, q_mask, v_tokens):
        target_dtype = self.q_proj.weight.dtype
        if q_tokens.dtype != target_dtype:
            q_tokens = q_tokens.to(dtype=target_dtype)
        if v_tokens.dtype != target_dtype:
            v_tokens = v_tokens.to(dtype=target_dtype)

        q_h = self.q_norm(self.q_proj(q_tokens))
        q_seq = self.q_delta(q_h, mask=q_mask)
        if q_mask is not None:
            q_mask_f = q_mask.to(dtype=q_seq.dtype).unsqueeze(-1)
            q_ctrl = (q_seq * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1.0)
        else:
            q_ctrl = q_seq.mean(dim=1)

        if v_tokens.dim() == 3:
            B, L, D = v_tokens.shape
            v_h = self.v_norm(self.v_proj(v_tokens))
            v_ctrl = v_h.mean(dim=1)
            fused = self.fuse(torch.cat([q_ctrl, v_ctrl, q_ctrl * v_ctrl, (q_ctrl - v_ctrl).abs()], dim=-1))
            return {"text": q_ctrl, "video": v_ctrl, "fused": fused}

        # Expected raw video tokens: (B, T, M, D)
        B, T, M, D = v_tokens.shape
        v_h = self.v_norm(self.v_proj(v_tokens))
        v_seq = v_h.permute(0, 2, 1, 3).reshape(B * M, T, -1)
        v_seq = self.v_delta(v_seq)  # (B*M, T, H)
        v_last = v_seq[:, -1, :].reshape(B, M, -1)

        q_expand = q_ctrl.unsqueeze(1).expand(B, M, -1)
        gate_logits = self.token_gate(torch.cat([q_expand, v_last, q_expand * v_last], dim=-1)).squeeze(-1)
        gate = F.softmax(gate_logits, dim=-1)
        v_ctrl = torch.einsum("bm,bmh->bh", gate, v_last)
        token_var = v_last.std(dim=1)
        fused = self.fuse(torch.cat([q_ctrl, v_ctrl, q_ctrl * v_ctrl, token_var], dim=-1))
        return {
            "text": q_ctrl,
            "video": v_ctrl,
            "fused": fused,
            "token_gate": gate,
        }


class RawContextSimplexRouter(nn.Module):
    """3-way simplex router over base/local/state from raw token context."""

    def __init__(self, token_dim, hidden_dim=128, num_heads=4, init_probs=None, max_logit_delta=1.0):
        super().__init__()
        self.encoder = RawContextRouterEncoder(token_dim=token_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        self.out = nn.Linear(hidden_dim, 3)
        self.max_logit_delta = float(max_logit_delta)
        with torch.no_grad():
            nn.init.zeros_(self.out.weight)
            if init_probs is None:
                init_probs = torch.tensor([0.20, 0.40, 0.40], dtype=torch.float32)
            else:
                init_probs = torch.as_tensor(init_probs, dtype=torch.float32)
            init_probs = init_probs / init_probs.sum().clamp(min=1e-6)
            prior_logits = torch.log(init_probs.clamp(min=1e-6))
            self.register_buffer("prior_logits", prior_logits, persistent=False)
            self.out.bias.zero_()

    def forward(self, q_tokens, q_mask, v_tokens):
        enc = self.encoder(q_tokens, q_mask, v_tokens)
        raw_logits = self.out(enc["fused"])
        logits = self.prior_logits.to(raw_logits.device, raw_logits.dtype) + self.max_logit_delta * torch.tanh(raw_logits)
        weights = F.softmax(logits, dim=-1)
        return {
            "weights": weights,
            "logits": logits,
        }


class BranchOutputSimplexRouter(nn.Module):
    """3-way simplex router over base/local/state branch outputs.

    Unlike raw-context routers that only observe question tokens plus base video
    tokens, this router directly reads the stage1 branch outputs:

      - z_base
      - z_local
      - z_state

    Each branch is projected to a small routing space and processed only along
    the time axis with a gated DeltaNet encoder. This keeps stage2 lightweight
    while still letting the router compare temporal behavior across branches.
    """

    def __init__(
        self,
        token_dim,
        branch_dim,
        hidden_dim=128,
        num_heads=4,
        init_probs=None,
        max_logit_delta=1.0,
        head_init_std=1e-3,
    ):
        super().__init__()
        del num_heads
        self.hidden_dim = int(hidden_dim)
        self.max_logit_delta = float(max_logit_delta)
        self.head_init_std = float(head_init_std)

        self.q_proj = nn.Linear(token_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.q_delta = _GatedDeltaSequenceEncoder(hidden_dim)

        self.base_proj = nn.Linear(branch_dim, hidden_dim)
        self.local_proj = nn.Linear(branch_dim, hidden_dim)
        self.state_proj = nn.Linear(branch_dim, hidden_dim)
        self.base_norm = nn.LayerNorm(hidden_dim)
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.base_delta = _GatedDeltaSequenceEncoder(hidden_dim)
        self.local_delta = _GatedDeltaSequenceEncoder(hidden_dim)
        self.state_delta = _GatedDeltaSequenceEncoder(hidden_dim)

        self.token_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.branch_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        with torch.no_grad():
            nn.init.normal_(self.token_gate[-1].weight, mean=0.0, std=self.head_init_std)
            nn.init.zeros_(self.token_gate[-1].bias)
            nn.init.normal_(self.score_head[-1].weight, mean=0.0, std=self.head_init_std)
            nn.init.zeros_(self.score_head[-1].bias)
            if init_probs is None:
                init_probs = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float32)
            else:
                init_probs = torch.as_tensor(init_probs, dtype=torch.float32)
            init_probs = init_probs / init_probs.sum().clamp(min=1e-6)
            prior_logits = torch.log(init_probs.clamp(min=1e-6))
            self.register_buffer("prior_logits", prior_logits, persistent=False)

    def _encode_question(self, q_tokens, q_mask):
        target_dtype = self.q_proj.weight.dtype
        if q_tokens.dtype != target_dtype:
            q_tokens = q_tokens.to(dtype=target_dtype)
        q_h = self.q_norm(self.q_proj(q_tokens))
        q_seq = self.q_delta(q_h, mask=q_mask)
        if q_mask is not None:
            q_mask_f = q_mask.to(dtype=q_seq.dtype).unsqueeze(-1)
            q_ctrl = (q_seq * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1.0)
        else:
            q_ctrl = q_seq.mean(dim=1)
        return q_ctrl

    def _encode_branch(self, branch_tokens, q_ctrl, proj, norm, delta):
        target_dtype = proj.weight.dtype
        if branch_tokens.dtype != target_dtype:
            branch_tokens = branch_tokens.to(dtype=target_dtype)

        bsz, t, m, _ = branch_tokens.shape
        x = norm(proj(branch_tokens))
        x_seq = x.permute(0, 2, 1, 3).reshape(bsz * m, t, self.hidden_dim)
        x_seq = delta(x_seq)
        x_last = x_seq[:, -1, :].reshape(bsz, m, self.hidden_dim)

        q_expand = q_ctrl.unsqueeze(1).expand(bsz, m, self.hidden_dim)
        gate_logits = self.token_gate(torch.cat([q_expand, x_last, q_expand * x_last], dim=-1)).squeeze(-1)
        gate = F.softmax(gate_logits, dim=-1)
        summary = torch.einsum("bm,bmh->bh", gate, x_last)
        token_var = x_last.std(dim=1)
        branch_feat = self.branch_fuse(torch.cat([q_ctrl, summary, q_ctrl * summary, token_var], dim=-1))
        score = self.score_head(torch.cat([q_ctrl, branch_feat, q_ctrl * branch_feat, (q_ctrl - branch_feat).abs()], dim=-1)).squeeze(-1)
        return branch_feat, score, gate

    def forward(self, q_tokens, q_mask, z_base, z_local, z_state):
        q_ctrl = self._encode_question(q_tokens, q_mask)
        base_feat, base_score, base_gate = self._encode_branch(z_base, q_ctrl, self.base_proj, self.base_norm, self.base_delta)
        local_feat, local_score, local_gate = self._encode_branch(z_local, q_ctrl, self.local_proj, self.local_norm, self.local_delta)
        state_feat, state_score, state_gate = self._encode_branch(z_state, q_ctrl, self.state_proj, self.state_norm, self.state_delta)

        raw_scores = torch.stack([base_score, local_score, state_score], dim=-1)
        logits = self.prior_logits.to(raw_scores.device, raw_scores.dtype) + self.max_logit_delta * torch.tanh(raw_scores)
        weights = F.softmax(logits, dim=-1)
        return {
            "weights": weights,
            "logits": logits,
            "raw_scores": raw_scores,
            "base_gate": base_gate,
            "local_gate": local_gate,
            "state_gate": state_gate,
            "base_feat": base_feat,
            "local_feat": local_feat,
            "state_feat": state_feat,
        }


class RawContextResidualScaler(nn.Module):
    """Raw-context 2-scalar residual controller.

    Unlike a simplex router, this does not choose between base/local/state.
    Base remains the anchor and the controller only adjusts how much of the
    frozen stage1 local/state residuals are used:

      z_final = z_base + a_local(x) * local_residual + a_state(x) * state_residual
    """

    def __init__(
        self,
        token_dim,
        hidden_dim=128,
        num_heads=4,
        init_local=1.0,
        init_state=1.0,
        max_local_delta=0.35,
        max_state_delta=0.35,
        mean_preserve=False,
        center_momentum=0.01,
        head_init_std=0.02,
        fuse_gain=40.0,
    ):
        super().__init__()
        self.encoder = RawContextRouterEncoder(token_dim=token_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        self.local_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Linear(hidden_dim, 1)
        self.max_local_delta = float(max_local_delta)
        self.max_state_delta = float(max_state_delta)
        self.init_local = float(init_local)
        self.init_state = float(init_state)
        self.mean_preserve = bool(mean_preserve)
        self.center_momentum = float(center_momentum)
        self.head_init_std = float(head_init_std)
        self.fuse_gain = float(fuse_gain)
        self.register_buffer("local_center", torch.zeros(()), persistent=True)
        self.register_buffer("state_center", torch.zeros(()), persistent=True)
        with torch.no_grad():
            nn.init.normal_(self.local_head.weight, mean=0.0, std=self.head_init_std)
            nn.init.zeros_(self.local_head.bias)
            nn.init.normal_(self.state_head.weight, mean=0.0, std=self.head_init_std)
            nn.init.zeros_(self.state_head.bias)

    def _center_raw(self, raw, center_buffer):
        if not self.mean_preserve:
            return raw
        if self.training and raw.numel() > 1:
            batch_center = raw.detach().float().mean()
            # Sync running center across DDP ranks so eval uses a consistent value.
            import torch.distributed as _dist
            if _dist.is_initialized():
                _dist.all_reduce(batch_center, op=_dist.ReduceOp.SUM)
                batch_center = batch_center / _dist.get_world_size()
            center_buffer.mul_(1.0 - self.center_momentum).add_(batch_center.to(center_buffer.device), alpha=self.center_momentum)
            return raw - raw.mean()
        return raw - center_buffer.to(device=raw.device, dtype=raw.dtype)

    def forward(self, q_tokens, q_mask, v_tokens):
        enc = self.encoder(q_tokens, q_mask, v_tokens)
        h = enc["fused"] * self.fuse_gain
        local_raw = torch.tanh(self.local_head(h)).squeeze(-1)
        state_raw = torch.tanh(self.state_head(h)).squeeze(-1)
        local_raw = self._center_raw(local_raw, self.local_center).clamp(min=-1.0, max=1.0)
        state_raw = self._center_raw(state_raw, self.state_center).clamp(min=-1.0, max=1.0)
        local_delta = self.max_local_delta * local_raw
        state_delta = self.max_state_delta * state_raw
        a_local = self.init_local + local_delta
        a_state = self.init_state + state_delta
        return {
            "a_local": a_local,
            "a_state": a_state,
            "local_delta": local_delta,
            "state_delta": state_delta,
        }


class RawContextHierarchicalRouter(nn.Module):
    """Hierarchical raw-context router: base vs temporal, then local vs state."""

    def __init__(self, token_dim, hidden_dim=128, num_heads=4, max_logit_delta=1.0):
        super().__init__()
        self.encoder = RawContextRouterEncoder(token_dim=token_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        self.temp_head = nn.Linear(hidden_dim, 1)
        self.branch_head = nn.Linear(hidden_dim, 2)
        self.max_logit_delta = float(max_logit_delta)
        with torch.no_grad():
            nn.init.zeros_(self.temp_head.weight)
            self.temp_head.bias.fill_(0.0)
            nn.init.zeros_(self.branch_head.weight)
            self.branch_head.bias.zero_()
            self.register_buffer("prior_temp_logit", torch.tensor(float(torch.logit(torch.tensor(0.8)))), persistent=False)
            self.register_buffer("prior_branch_logits", torch.tensor([0.0, 0.0], dtype=torch.float32), persistent=False)

    def forward(self, q_tokens, q_mask, v_tokens):
        enc = self.encoder(q_tokens, q_mask, v_tokens)
        h = enc["fused"]
        temp_logit = self.prior_temp_logit.to(h.device, h.dtype) + self.max_logit_delta * torch.tanh(self.temp_head(h)).squeeze(-1)
        p_temp = torch.sigmoid(temp_logit)
        branch_logits = self.prior_branch_logits.to(h.device, h.dtype) + self.max_logit_delta * torch.tanh(self.branch_head(h))
        p_ls = F.softmax(branch_logits, dim=-1)
        w_base = 1.0 - p_temp
        w_local = p_temp * p_ls[:, 0]
        w_state = p_temp * p_ls[:, 1]
        weights = torch.stack([w_base, w_local, w_state], dim=-1)
        return {
            "weights": weights,
            "p_temp": p_temp,
            "p_local": p_ls[:, 0],
            "p_state": p_ls[:, 1],
        }


class FramewiseRawContextRouterEncoder(nn.Module):
    """Frame-wise raw-context encoder using gated DeltaNet sequence processing."""

    def __init__(self, token_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        del num_heads
        self.q_proj = nn.Linear(token_dim, hidden_dim)
        self.v_proj = nn.Linear(token_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.q_delta = _GatedDeltaSequenceEncoder(hidden_dim)
        self.v_delta = _GatedDeltaSequenceEncoder(hidden_dim)
        self.token_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        with torch.no_grad():
            nn.init.zeros_(self.token_gate[-1].weight)
            nn.init.zeros_(self.token_gate[-1].bias)

    def forward(self, q_tokens, q_mask, v_tokens):
        target_dtype = self.q_proj.weight.dtype
        if q_tokens.dtype != target_dtype:
            q_tokens = q_tokens.to(dtype=target_dtype)
        if v_tokens.dtype != target_dtype:
            v_tokens = v_tokens.to(dtype=target_dtype)

        q_h = self.q_norm(self.q_proj(q_tokens))
        q_seq = self.q_delta(q_h, mask=q_mask)
        if q_mask is not None:
            q_mask_f = q_mask.to(dtype=q_seq.dtype).unsqueeze(-1)
            q_ctrl = (q_seq * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1.0)
        else:
            q_ctrl = q_seq.mean(dim=1)

        if v_tokens.dim() != 4:
            raise ValueError(f"FramewiseRawContextRouterEncoder expects video tokens (B,T,M,D), got {tuple(v_tokens.shape)}")

        B, T, M, D = v_tokens.shape
        v_h = self.v_norm(self.v_proj(v_tokens))
        v_seq = v_h.permute(0, 2, 1, 3).reshape(B * M, T, -1)
        v_seq = self.v_delta(v_seq).reshape(B, M, T, -1)

        q_expand = q_ctrl[:, None, None, :].expand(B, M, T, -1)
        v_time = v_seq
        gate_logits = self.token_gate(torch.cat([q_expand, v_time, q_expand * v_time], dim=-1)).squeeze(-1)  # (B,M,T)
        gate = F.softmax(gate_logits, dim=1)
        v_ctrl = torch.einsum("bmt,bmth->bth", gate, v_time)
        token_var = v_time.std(dim=1)
        q_time = q_ctrl[:, None, :].expand(B, T, -1)
        fused = self.fuse(torch.cat([q_time, v_ctrl, q_time * v_ctrl, token_var], dim=-1))
        return {
            "text": q_ctrl,
            "video": v_ctrl,
            "fused": fused,
            "token_gate": gate,
        }


class FramewiseRawContextSimplexRouter(nn.Module):
    """Frame-wise 3-way simplex router over base/local/state."""

    def __init__(self, token_dim, hidden_dim=128, num_heads=4, init_probs=None, max_logit_delta=0.5):
        super().__init__()
        self.encoder = FramewiseRawContextRouterEncoder(token_dim=token_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        self.out = nn.Linear(hidden_dim, 3)
        self.max_logit_delta = float(max_logit_delta)
        with torch.no_grad():
            nn.init.zeros_(self.out.weight)
            if init_probs is None:
                init_probs = torch.tensor([0.20, 0.40, 0.40], dtype=torch.float32)
            else:
                init_probs = torch.as_tensor(init_probs, dtype=torch.float32)
            init_probs = init_probs / init_probs.sum().clamp(min=1e-6)
            self.register_buffer("prior_logits", torch.log(init_probs.clamp(min=1e-6)), persistent=False)
            self.out.bias.zero_()

    def forward(self, q_tokens, q_mask, v_tokens):
        enc = self.encoder(q_tokens, q_mask, v_tokens)
        raw_logits = self.out(enc["fused"])  # (B,T,3)
        logits = self.prior_logits.to(raw_logits.device, raw_logits.dtype).view(1, 1, 3) + self.max_logit_delta * torch.tanh(raw_logits)
        weights = F.softmax(logits, dim=-1)
        return {"weights": weights, "logits": logits}


class FramewiseRawContextHierarchicalRouter(nn.Module):
    """Frame-wise hierarchical router: base vs temporal, then local vs state."""

    def __init__(self, token_dim, hidden_dim=128, num_heads=4, max_logit_delta=0.5):
        super().__init__()
        self.encoder = FramewiseRawContextRouterEncoder(token_dim=token_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        self.temp_head = nn.Linear(hidden_dim, 1)
        self.branch_head = nn.Linear(hidden_dim, 2)
        self.max_logit_delta = float(max_logit_delta)
        with torch.no_grad():
            nn.init.zeros_(self.temp_head.weight)
            self.temp_head.bias.zero_()
            nn.init.zeros_(self.branch_head.weight)
            self.branch_head.bias.zero_()
            self.register_buffer("prior_temp_logit", torch.tensor(float(torch.logit(torch.tensor(0.8)))), persistent=False)
            self.register_buffer("prior_branch_logits", torch.tensor([0.0, 0.0], dtype=torch.float32), persistent=False)

    def forward(self, q_tokens, q_mask, v_tokens):
        enc = self.encoder(q_tokens, q_mask, v_tokens)
        h = enc["fused"]  # (B,T,H)
        temp_logit = self.prior_temp_logit.to(h.device, h.dtype).view(1, 1) + self.max_logit_delta * torch.tanh(self.temp_head(h)).squeeze(-1)
        p_temp = torch.sigmoid(temp_logit)
        branch_logits = self.prior_branch_logits.to(h.device, h.dtype).view(1, 1, 2) + self.max_logit_delta * torch.tanh(self.branch_head(h))
        p_ls = F.softmax(branch_logits, dim=-1)
        w_base = 1.0 - p_temp
        w_local = p_temp * p_ls[..., 0]
        w_state = p_temp * p_ls[..., 1]
        weights = torch.stack([w_base, w_local, w_state], dim=-1)
        return {
            "weights": weights,
            "p_temp": p_temp,
            "p_local": p_ls[..., 0],
            "p_state": p_ls[..., 1],
        }


class HiddenTemporalBranch(nn.Module):
    """Conservative hidden-space temporal corrector.

    The branch mean-pools spatial tokens, runs a light temporal recurrence in
    hidden space, then broadcasts a bounded residual back to all spatial tokens.
    This is intentionally safe: zero-init on the up projection makes the branch
    start as a near no-op around the frozen base projector.
    """

    def __init__(self, hidden_dim=2048, bottleneck=128, max_gate=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck = bottleneck
        self.max_gate = max_gate

        self.down = nn.Linear(hidden_dim, bottleneck)
        self.beta_proj = nn.Linear(bottleneck, bottleneck)
        self.q_proj = nn.Linear(bottleneck, bottleneck)
        self.k_proj = nn.Linear(bottleneck, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_dim)

        self.gate_head = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        with torch.no_grad():
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
            self.gate_head[2].bias.data.fill_(-2.0)

        self._last_gate_mean = 0.0
        self._last_gate_std = 0.0
        self._last_recurrence_norm = 0.0
        self._last_recurrence = None
        self._last_gate_tensor = None

    def forward(self, h):
        if h.dim() != 4:
            raise ValueError(f"HiddenTemporalBranch expects 4D input (B,T,N,H), got {tuple(h.shape)}")

        B, T, N, H = h.shape
        h_pool = h.mean(dim=2)
        h_down = self.down(h_pool)

        beta = torch.sigmoid(self.beta_proj(h_down))
        q = self.q_proj(h_down)
        k = self.k_proj(h_down)

        bn = h_down.shape[-1]
        S = torch.zeros(B, bn, bn, device=h.device, dtype=h.dtype)
        outputs = []
        for t in range(T):
            S = beta[:, t, :].unsqueeze(-1) * S + k[:, t, :].unsqueeze(-1) * h_down[:, t, :].unsqueeze(1)
            o_t = (q[:, t, :].unsqueeze(1) * S).sum(dim=-1)
            outputs.append(o_t)
        recurrence = torch.stack(outputs, dim=1)

        gate_logit = self.gate_head(recurrence).squeeze(-1)
        gate = self.max_gate * torch.sigmoid(gate_logit)

        temporal_up = self.up(recurrence)
        temporal_broadcast = temporal_up.unsqueeze(2).expand(B, T, N, H)
        output = gate.unsqueeze(-1).unsqueeze(-1) * temporal_broadcast

        self._last_gate_mean = gate.detach().mean().item()
        self._last_gate_std = gate.detach().std().item()
        self._last_recurrence_norm = recurrence.detach().norm(dim=-1).mean().item()
        self._last_recurrence = recurrence
        self._last_gate_tensor = gate
        return output


# ════════════════════════════════════════════════════════════════
#  Design B: DPM (Dual Projector Mixture)
# ════════════════════════════════════════════════════════════════

class TemporalBlock(nn.Module):
    """Small temporal processing block inserted between linear_1+GELU and linear_2.

    v3: gate_init raised to -1.0 (sigmoid ~0.27), raw sigmoid (no max clamp).
    """

    def __init__(self, hidden_dim=2048, bottleneck=64, gate_init=-1.0):
        super().__init__()
        self.bottleneck = bottleneck

        self.down = nn.Linear(hidden_dim, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_dim)

        self.q_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.k_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.v_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.beta_proj = nn.Linear(bottleneck, bottleneck, bias=False)

        # Gate: raw sigmoid (no max clamp), higher init for gradient flow
        self.gate_raw = nn.Parameter(torch.tensor(gate_init))

        nn.init.xavier_uniform_(self.up.weight, gain=0.01)
        nn.init.zeros_(self.up.bias)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.beta_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)

    def forward(self, h, T):
        """
        h: (BT, N, H) hidden features after linear_1 + GELU
        T: number of frames
        Returns: (BT, N, H) temporally enhanced features
        """
        BT, N, H = h.shape
        B = BT // T

        h4d = h.reshape(B, T, N, H)
        h_seq = h4d.permute(0, 2, 1, 3).reshape(B * N, T, H)

        x = self.down(h_seq)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x))

        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)

        outputs = []
        S = torch.zeros(B * N, self.bottleneck, self.bottleneck,
                        device=x.device, dtype=x.dtype)

        for t in range(T):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]
            beta_t = beta[:, t, :]

            erase = torch.einsum('bi,bj->bij', beta_t, k_t)
            write = torch.einsum('bi,bj->bij', beta_t * v_t, k_t)
            S = S * (1.0 - erase) + write

            o_t = torch.einsum('bij,bj->bi', S, q_t)
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)

        temporal = self.up(output)
        temporal = temporal.reshape(B, N, T, H).permute(0, 2, 1, 3)
        temporal = temporal.reshape(BT, N, H)

        # Raw sigmoid gate (no max clamp)
        gate = torch.sigmoid(self.gate_raw)
        result = h + gate * temporal

        self._last_gate = gate.item()
        self._last_temporal_norm = temporal.detach().norm(dim=-1).mean().item()

        return result


class TemporalProjector(nn.Module):
    """Temporal projector: clone of base + LoRA on linear_2 + temporal block.

    Architecture:
      linear_1 (frozen clone) -> GELU -> TemporalBlock -> linear_2 (frozen clone + LoRA) -> pooling
    """

    def __init__(self, base_projector, lora_rank=32, lora_alpha=64.0,
                 temporal_bottleneck=64, temporal_gate_init=-1.0):
        super().__init__()

        self.register_buffer('linear_1_weight', base_projector.linear_1.weight.data.clone())
        self.register_buffer('linear_1_bias', base_projector.linear_1.bias.data.clone())
        self.register_buffer('linear_2_weight', base_projector.linear_2.weight.data.clone())
        self.register_buffer('linear_2_bias', base_projector.linear_2.bias.data.clone())

        self.gelu = nn.GELU()
        self.pooling_ratio = base_projector.pooling.pooling_ratio

        hidden_dim = base_projector.linear_2.in_features
        self.temporal_block = TemporalBlock(hidden_dim, temporal_bottleneck,
                                           gate_init=temporal_gate_init)

        in2 = base_projector.linear_2.in_features
        out2 = base_projector.linear_2.out_features
        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank
        self.lora_A2 = nn.Linear(in2, lora_rank, bias=False)
        self.lora_B2 = nn.Linear(lora_rank, out2, bias=False)

        nn.init.kaiming_uniform_(self.lora_A2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B2.weight)

    def _pooling(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        shape = (h // self.pooling_ratio, h // self.pooling_ratio)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = F.adaptive_avg_pool2d(x, shape)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward(self, x, T):
        x_nld = x.permute(1, 0, 2)
        h = F.linear(x_nld, self.linear_1_weight, self.linear_1_bias)
        h = self.gelu(h)
        h = h.permute(1, 0, 2)

        h = self.temporal_block(h, T)

        h_nld = h.permute(1, 0, 2)
        out = F.linear(h_nld, self.linear_2_weight, self.linear_2_bias)
        out_lora = self.lora_B2(self.lora_A2(h_nld)) * self.lora_scaling
        out = out + out_lora
        out = out.permute(1, 0, 2)

        out = self._pooling(out)
        return out


class FusionGate(nn.Module):
    """Frame-wise gated interpolation for merging z_base and z_temp.

    v3: scaled sigmoid (no hard clamp), optional channel-group gate.
    """

    def __init__(self, proj_dim=2048, hidden=256, init_bias=-3.0, max_gate=0.15,
                 num_gate_groups=1):
        super().__init__()
        self.max_gate = max_gate
        self.num_gate_groups = num_gate_groups
        self.proj_dim = proj_dim

        self.gate_net = nn.Sequential(
            nn.Linear(proj_dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_gate_groups, bias=True),
        )
        nn.init.zeros_(self.gate_net[2].weight)
        self.gate_net[2].bias.data.fill_(init_bias)

    def gate_parameters(self):
        """Return gate-specific parameters for separate optimizer group."""
        return list(self.gate_net.parameters())

    def non_gate_parameters(self):
        """Return non-gate parameters (none for FusionGate, it's all gate)."""
        return []

    def forward(self, z_base, z_temp, return_gate=False):
        B, T, M, D = z_base.shape

        base_pool = z_base.mean(dim=2)
        temp_pool = z_temp.mean(dim=2)
        diff_pool = (z_temp - z_base).abs().mean(dim=2)

        gate_input = torch.cat([base_pool, temp_pool, diff_pool], dim=-1)
        # Scaled sigmoid: g in [0, max_gate), no hard clamp
        g = self.max_gate * torch.sigmoid(self.gate_net(gate_input))  # (B, T, G)

        if self.num_gate_groups == 1:
            g = g.unsqueeze(2)  # (B, T, 1, 1)
        else:
            g = g.unsqueeze(2)  # (B, T, 1, G)
            g = g.repeat_interleave(D // self.num_gate_groups, dim=-1)  # (B, T, 1, D)

        result = z_base + g * (z_temp - z_base)

        self._last_gate_mean = g.detach().mean().item()
        self._last_gate_max = g.detach().max().item()
        self._last_gate_std = g.detach().std().item()

        if return_gate:
            return result, g
        return result


# ════════════════════════════════════════════════════════════════
#  Design B v4: DPM with frame-wise gate + perturbation losses
# ════════════════════════════════════════════════════════════════

class TemporalBlockV4(nn.Module):
    """TemporalBlock with frame-wise gate (v4).

    Key change: global scalar gate → frame-wise gate MLP.
    Gate takes per-frame DeltaNet output summary and produces per-frame gate values.
    Scaled sigmoid with max_gate to prevent eff_scale blowup.
    """

    def __init__(self, hidden_dim=2048, bottleneck=64, gate_init=-2.0, max_gate=0.5):
        super().__init__()
        self.bottleneck = bottleneck
        self.max_gate = max_gate

        self.down = nn.Linear(hidden_dim, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_dim)

        self.q_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.k_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.v_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.beta_proj = nn.Linear(bottleneck, bottleneck, bias=False)

        # Frame-wise gate: DeltaNet output summary → per-frame scalar
        self.frame_gate = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.frame_gate[2].weight)
        self.frame_gate[2].bias.data.fill_(gate_init)

        nn.init.xavier_uniform_(self.up.weight, gain=0.01)
        nn.init.zeros_(self.up.bias)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.beta_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)

    def gate_parameters(self):
        """Return frame gate MLP parameters for separate optimizer group."""
        return list(self.frame_gate.parameters())

    def forward(self, h, T):
        """
        h: (BT, N, H) hidden features after linear_1 + GELU
        T: number of frames
        Returns: (BT, N, H) temporally enhanced features
        """
        BT, N, H = h.shape
        B = BT // T

        h4d = h.reshape(B, T, N, H)
        h_seq = h4d.permute(0, 2, 1, 3).reshape(B * N, T, H)

        x = self.down(h_seq)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x))

        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)

        outputs = []
        S = torch.zeros(B * N, self.bottleneck, self.bottleneck,
                        device=x.device, dtype=x.dtype)

        for t in range(T):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]
            beta_t = beta[:, t, :]

            erase = torch.einsum('bi,bj->bij', beta_t, k_t)
            write = torch.einsum('bi,bj->bij', beta_t * v_t, k_t)
            S = S * (1.0 - erase) + write

            o_t = torch.einsum('bij,bj->bi', S, q_t)
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)  # (B*N, T, bottleneck)

        temporal = self.up(output)  # (B*N, T, H)
        temporal = temporal.reshape(B, N, T, H).permute(0, 2, 1, 3)  # (B, T, N, H)
        temporal = temporal.reshape(BT, N, H)

        # Frame-wise gate from DeltaNet output summary
        output_4d = output.reshape(B, N, T, self.bottleneck)
        frame_summary = output_4d.mean(dim=1)  # (B, T, bottleneck)
        gate = self.max_gate * torch.sigmoid(self.frame_gate(frame_summary))  # (B, T, 1)

        # Broadcast to (BT, N, 1)
        gate_bt = gate.unsqueeze(2).expand(B, T, N, 1).reshape(BT, N, 1)
        result = h + gate_bt * temporal

        # Store frame summary: gradient-flowing for FusionGateV6 gate input,
        # detached copy for logging only
        self._last_frame_summary = frame_summary  # (B, T, bottleneck) — WITH gradient
        self._last_frame_summary_detached = frame_summary.detach()  # for logging

        # Store frame gate tensor for external use (e.g. usefulness gate loss)
        self._last_frame_gate = gate  # (B, T, 1) — keep gradient for L_gate

        # Logging: mean, std, max
        self._last_gate_mean = gate.detach().mean().item()
        self._last_gate_std = gate.detach().std().item()
        self._last_gate_max = gate.detach().max().item()
        self._last_temporal_norm = temporal.detach().norm(dim=-1).mean().item()

        return result


class TemporalProjectorV4(nn.Module):
    """Temporal projector v4: uses TemporalBlockV4 with frame-wise gate.

    Architecture:
      linear_1 (frozen clone) -> GELU -> TemporalBlockV4 -> linear_2 (frozen clone + LoRA) -> pooling
    """

    def __init__(self, base_projector, lora_rank=32, lora_alpha=64.0,
                 temporal_bottleneck=64, temporal_gate_init=-2.0, max_tb_gate=0.5):
        super().__init__()

        self.register_buffer('linear_1_weight', base_projector.linear_1.weight.data.clone())
        self.register_buffer('linear_1_bias', base_projector.linear_1.bias.data.clone())
        self.register_buffer('linear_2_weight', base_projector.linear_2.weight.data.clone())
        self.register_buffer('linear_2_bias', base_projector.linear_2.bias.data.clone())

        self.gelu = nn.GELU()
        self.pooling_ratio = base_projector.pooling.pooling_ratio

        hidden_dim = base_projector.linear_2.in_features
        self.temporal_block = TemporalBlockV4(
            hidden_dim, temporal_bottleneck,
            gate_init=temporal_gate_init,
            max_gate=max_tb_gate,
        )

        in2 = base_projector.linear_2.in_features
        out2 = base_projector.linear_2.out_features
        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank
        self.lora_A2 = nn.Linear(in2, lora_rank, bias=False)
        self.lora_B2 = nn.Linear(lora_rank, out2, bias=False)

        nn.init.kaiming_uniform_(self.lora_A2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B2.weight)

    def _pooling(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        shape = (h // self.pooling_ratio, h // self.pooling_ratio)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = F.adaptive_avg_pool2d(x, shape)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward(self, x, T):
        x_nld = x.permute(1, 0, 2)
        h = F.linear(x_nld, self.linear_1_weight, self.linear_1_bias)
        h = self.gelu(h)
        h = h.permute(1, 0, 2)

        h = self.temporal_block(h, T)

        h_nld = h.permute(1, 0, 2)
        out = F.linear(h_nld, self.linear_2_weight, self.linear_2_bias)
        out_lora = self.lora_B2(self.lora_A2(h_nld)) * self.lora_scaling
        out = out + out_lora
        out = out.permute(1, 0, 2)

        out = self._pooling(out)
        return out


class TemporalProjectorV13(nn.Module):
    """Trainable projector clone with pre-pool temporal recurrence.

    Layout:
      linear_1 -> GELU -> linear_2 -> temporal recurrence -> pooling

    The temporal module sees full pre-pool spatial tokens so it can model
    token-local motion/state changes before adaptive pooling compresses them.
    """

    def __init__(self, base_projector, temporal_bottleneck=128):
        super().__init__()

        in1 = base_projector.linear_1.in_features
        out1 = base_projector.linear_1.out_features
        in2 = base_projector.linear_2.in_features
        out2 = base_projector.linear_2.out_features

        self.linear_1 = nn.Linear(in1, out1, bias=True)
        self.linear_2 = nn.Linear(in2, out2, bias=True)
        self.linear_1.load_state_dict(base_projector.linear_1.state_dict())
        self.linear_2.load_state_dict(base_projector.linear_2.state_dict())

        self.gelu = nn.GELU()
        self.pooling_ratio = base_projector.pooling.pooling_ratio
        self.proj_dim = out2
        self.temporal_bottleneck = temporal_bottleneck

        self.diff_norm = nn.LayerNorm(self.proj_dim)
        self.temporal_down = nn.Linear(self.proj_dim, temporal_bottleneck)
        self.q_proj = nn.Linear(temporal_bottleneck, temporal_bottleneck, bias=False)
        self.k_proj = nn.Linear(temporal_bottleneck, temporal_bottleneck, bias=False)
        self.v_proj = nn.Linear(temporal_bottleneck, temporal_bottleneck, bias=False)
        self.beta_proj = nn.Linear(temporal_bottleneck, temporal_bottleneck, bias=False)
        self.temporal_up = nn.Linear(temporal_bottleneck, self.proj_dim)

        nn.init.xavier_uniform_(self.temporal_up.weight, gain=0.01)
        nn.init.zeros_(self.temporal_up.bias)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.beta_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)

        self._last_diff_norm = 0.0
        self._last_temporal_norm = 0.0
        self._last_projector_delta_norm = 0.0

    def _pooling(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        shape = (h // self.pooling_ratio, h // self.pooling_ratio)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = F.adaptive_avg_pool2d(x, shape)
        x = x.flatten(2).transpose(1, 2)
        return x

    def _apply_temporal(self, z_tokens, T):
        BT, M, D = z_tokens.shape
        B = BT // T
        z_4d = z_tokens.reshape(B, T, M, D)

        if T <= 1:
            self._last_diff_norm = 0.0
            self._last_temporal_norm = 0.0
            self._last_projector_delta_norm = 0.0
            return z_tokens

        diff = self.diff_norm(z_4d[:, 1:] - z_4d[:, :-1])
        x = self.temporal_down(diff)
        x = x.permute(0, 2, 1, 3).reshape(B * M, T - 1, self.temporal_bottleneck)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x))

        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)

        outputs = []
        S = torch.zeros(
            B * M,
            self.temporal_bottleneck,
            self.temporal_bottleneck,
            device=x.device,
            dtype=x.dtype,
        )

        for t in range(T - 1):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]
            beta_t = beta[:, t, :]

            erase = torch.einsum("bi,bj->bij", beta_t, k_t)
            write = torch.einsum("bi,bj->bij", beta_t * v_t, k_t)
            S = S * (1.0 - erase) + write

            o_t = torch.einsum("bij,bj->bi", S, q_t)
            outputs.append(o_t)

        temporal = torch.stack(outputs, dim=1)
        temporal = temporal.reshape(B, M, T - 1, self.temporal_bottleneck).permute(0, 2, 1, 3)
        enrich = self.temporal_up(temporal)

        z_out = z_4d.clone()
        z_out[:, 1:] = z_out[:, 1:] + enrich

        self._last_diff_norm = diff.detach().norm(dim=-1).mean().item()
        self._last_temporal_norm = enrich.detach().norm(dim=-1).mean().item()
        self._last_projector_delta_norm = (z_out - z_4d).detach().norm(dim=-1).mean().item()

        return z_out.reshape(BT, M, D)

    def forward(self, x, T):
        x_nld = x.permute(1, 0, 2)
        h = self.linear_1(x_nld)
        h = self.gelu(h)
        out = self.linear_2(h)
        out = out.permute(1, 0, 2)
        out = self._apply_temporal(out, T)
        out = self._pooling(out)
        return out


class DiffDeltaTemporalExpert(nn.Module):
    """Pre-pool diff expert shared by DeltaNet v8.10 train/eval.

    The expert operates on full-resolution pre-pool projector tokens, then
    pools only the temporal enrichment to match the pooled base projector
    output. This preserves spatially-local temporal change during recurrence.
    """

    def __init__(self, proj_dim=2048, bottleneck=128, kernel_size=3,
                 max_enrich_gate=0.25, pooling_ratio=2):
        super().__init__()
        self.proj_dim = proj_dim
        self.bottleneck = bottleneck
        self.max_enrich_gate = max_enrich_gate
        self.pooling_ratio = pooling_ratio

        self.diff_norm = nn.LayerNorm(proj_dim)
        self.down = nn.Linear(proj_dim, bottleneck)
        self.temporal_conv = nn.Conv1d(
            bottleneck, bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=bottleneck,
        )
        self.hidden_norm = nn.LayerNorm(bottleneck)
        self.output_norm = nn.LayerNorm(bottleneck)
        self.fuse_norm = nn.LayerNorm(bottleneck)
        self.fuse_norm = nn.LayerNorm(bottleneck)

        self.q_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.k_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.v_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.beta_proj = nn.Linear(bottleneck, bottleneck, bias=False)

        self.up = nn.Linear(bottleneck, proj_dim)
        self.out_gate = nn.Linear(bottleneck, 1)
        nn.init.xavier_uniform_(self.up.weight, gain=0.01)
        nn.init.zeros_(self.up.bias)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.beta_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        nn.init.zeros_(self.out_gate.weight)
        self.out_gate.bias.data.fill_(-2.0)

    def _pool_spatial(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        out_h = h // self.pooling_ratio
        x = x.permute(0, 2, 1).reshape(b, c, h, h)
        x = F.adaptive_avg_pool2d(x, (out_h, out_h))
        return x.flatten(2).transpose(1, 2)

    def _pool_enrich(self, enrich):
        b, tm1, s_pre, d = enrich.shape
        x = enrich.reshape(b * tm1, s_pre, d)
        x = self._pool_spatial(x)
        s_post = x.shape[1]
        return x.reshape(b, tm1, s_post, d)

    def _pool_hidden(self, h):
        b, tm1, s_pre, d = h.shape
        x = h.reshape(b * tm1, s_pre, d)
        x = self._pool_spatial(x)
        s_post = x.shape[1]
        return x.reshape(b, tm1, s_post, d)

    def forward(self, z_pre_pool, z_base_pooled, return_aux=False, modulation=None, return_aux_mode=None):
        b, t, s_pre, d = z_pre_pool.shape
        aux_mode = return_aux_mode or "full"
        s_post = z_base_pooled.shape[2]

        if t <= 1:
            zero = torch.zeros_like(z_base_pooled)
            h_full = torch.zeros(
                b, t, s_post, self.bottleneck,
                device=z_base_pooled.device, dtype=z_base_pooled.dtype,
            )
            self._last_diff_norm = 0.0
            self._last_hidden_norm = 0.0
            self._last_gate_mean = 0.0
            self._last_gate_std = 0.0
            self._last_enrich_norm = 0.0
            if return_aux:
                aux = {
                    "state_h": torch.zeros(
                        b, t, self.bottleneck,
                        device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                    ),
                    "state": z_base_pooled.mean(dim=2),
                    "state_d1": torch.zeros(
                        b, t, d,
                        device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                    ),
                    "state_d0": torch.zeros(
                        b, t, d,
                        device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                    ),
                }
                return z_base_pooled, zero, h_full, aux
            return z_base_pooled, zero, h_full

        diff = self.diff_norm(z_pre_pool[:, 1:] - z_pre_pool[:, :-1])
        x = self.down(diff)

        x_tok = x.permute(0, 2, 3, 1).reshape(b * s_pre, self.bottleneck, t - 1)
        x_tok = self.temporal_conv(x_tok)
        x_tok = x_tok.reshape(b, s_pre, self.bottleneck, t - 1).permute(0, 3, 1, 2)
        x = self.hidden_norm(x_tok)

        seq = x.permute(0, 2, 1, 3).reshape(b * s_pre, t - 1, self.bottleneck)
        q = self.q_proj(seq)
        k = self.k_proj(seq)
        v = self.v_proj(seq)
        beta = torch.sigmoid(self.beta_proj(seq))

        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)

        state = torch.zeros(
            b * s_pre, self.bottleneck, self.bottleneck,
            device=z_pre_pool.device, dtype=z_pre_pool.dtype,
        )
        outputs = []
        for i in range(t - 1):
            q_t = q[:, i, :]
            k_t = k[:, i, :]
            v_t = v[:, i, :]
            beta_t = beta[:, i, :]

            erase = torch.einsum("bi,bj->bij", beta_t, k_t)
            write = torch.einsum("bi,bj->bij", beta_t * v_t, k_t)
            state = state * (1.0 - erase) + write
            y = torch.bmm(state, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(y)

        h = torch.stack(outputs, dim=1).reshape(b, s_pre, t - 1, self.bottleneck).permute(0, 2, 1, 3)
        h = self.output_norm(h)
        gate = self.max_enrich_gate * torch.sigmoid(self.out_gate(h))
        enrich_pre = gate * self.up(h)
        enrich_pooled = self._pool_enrich(enrich_pre)

        z_temp = z_base_pooled.clone()
        z_temp[:, 1:] = z_base_pooled[:, 1:] + enrich_pooled

        h_pooled = self._pool_hidden(h)
        h_full = torch.zeros(
            b, t, h_pooled.shape[2], self.bottleneck,
            device=z_base_pooled.device, dtype=z_base_pooled.dtype,
        )
        h_full[:, 1:] = h_pooled

        self._last_diff_norm = diff.detach().norm(dim=-1).mean().item()
        self._last_hidden_norm = h.detach().norm(dim=-1).mean().item()
        self._last_gate_mean = gate.detach().mean().item()
        self._last_gate_std = gate.detach().std().item()
        self._last_enrich_norm = enrich_pooled.detach().norm(dim=-1).mean().item()
        return z_temp, enrich_pooled, h_full


class StateAwareDiffDeltaTemporalExpert(nn.Module):
    """
    v9.0 stage1 expert: combine local pre-pool diffs with frame-level state memory.

    Local branch keeps the strong motion sensitivity of v8.10.
    State branch sees:
      - absolute pooled state s_t
      - short-range pooled delta s_t - s_{t-1}
      - anchor delta s_t - s_0
    and runs a stable DeltaNet recurrence over full T steps.
    """

    def __init__(
        self,
        proj_dim=2048,
        bottleneck=128,
        state_bottleneck=None,
        state_slots=1,
        kernel_size=3,
        max_enrich_gate=0.25,
        max_state_gate=0.15,
        pooling_ratio=2,
        state_from_prepool=False,
        local_short_window=False,
        separate_local_state=False,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.bottleneck = bottleneck
        self.state_bottleneck = int(state_bottleneck or bottleneck)
        self.state_slots = int(max(1, state_slots))
        self.state_slot_grid = int(math.sqrt(self.state_slots))
        if self.state_slot_grid * self.state_slot_grid != self.state_slots:
            raise ValueError(f"state_slots must be a perfect square, got {self.state_slots}")
        self.use_slot_state = (self.state_slots > 1) or (self.state_bottleneck != bottleneck)
        self.max_enrich_gate = max_enrich_gate
        self.max_state_gate = max_state_gate
        self.pooling_ratio = pooling_ratio
        self.state_from_prepool = bool(state_from_prepool)
        self.local_short_window = bool(local_short_window)
        self.separate_local_state = bool(separate_local_state)

        # Local token-wise diff branch.
        self.diff_norm = nn.LayerNorm(proj_dim)
        self.temporal_conv = nn.Conv1d(
            bottleneck, bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=bottleneck,
        )
        self.output_norm = nn.LayerNorm(bottleneck)

        if self.local_short_window:
            self.local_input_norm = nn.LayerNorm(proj_dim * 2)
            self.local_down = nn.Linear(proj_dim * 2, bottleneck)
        else:
            self.down = nn.Linear(proj_dim, bottleneck)
            self.hidden_norm = nn.LayerNorm(bottleneck)
            self.q_proj = nn.Linear(bottleneck, bottleneck, bias=False)
            self.k_proj = nn.Linear(bottleneck, bottleneck, bias=False)
            self.v_proj = nn.Linear(bottleneck, bottleneck, bias=False)
            self.beta_proj = nn.Linear(bottleneck, bottleneck, bias=False)

        self.up = nn.Linear(bottleneck, proj_dim)
        self.out_gate = nn.Linear(bottleneck, 1)

        # Frame-level state memory branch.
        self.state_input_norm = nn.LayerNorm(proj_dim * 3)
        self.state_down = nn.Linear(proj_dim * 3, bottleneck)
        self.state_slot_input_norm = nn.LayerNorm(proj_dim * 3)
        self.state_slot_down = nn.Linear(proj_dim * 3, self.state_bottleneck)
        self.state_temporal_conv = nn.Conv1d(
            self.state_bottleneck, self.state_bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=self.state_bottleneck,
        )
        self.state_hidden_norm = nn.LayerNorm(self.state_bottleneck)
        self.state_output_norm = nn.LayerNorm(self.state_bottleneck)

        self.state_q_proj = nn.Linear(self.state_bottleneck, self.state_bottleneck, bias=False)
        self.state_k_proj = nn.Linear(self.state_bottleneck, self.state_bottleneck, bias=False)
        self.state_v_proj = nn.Linear(self.state_bottleneck, self.state_bottleneck, bias=False)
        self.state_beta_proj = nn.Linear(self.state_bottleneck, self.state_bottleneck, bias=False)

        self.state_gamma = nn.Linear(self.state_bottleneck, bottleneck)
        self.state_beta = nn.Linear(self.state_bottleneck, bottleneck)
        self.state_gate = nn.Linear(self.state_bottleneck, 1)
        self.state_up = nn.Linear(self.state_bottleneck, proj_dim)
        self.state_readout = nn.Linear(self.state_bottleneck, 1, bias=False)
        self.fuse_norm = nn.LayerNorm(bottleneck)

        nn.init.xavier_uniform_(self.up.weight, gain=0.01)
        nn.init.zeros_(self.up.bias)
        nn.init.xavier_uniform_(self.state_up.weight, gain=0.01)
        nn.init.zeros_(self.state_up.bias)

        delta_projs = [self.state_q_proj, self.state_k_proj, self.state_v_proj, self.state_beta_proj]
        if not self.local_short_window:
            delta_projs.extend([self.q_proj, self.k_proj, self.v_proj, self.beta_proj])
        for m in delta_projs:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        nn.init.xavier_uniform_(self.state_slot_down.weight, gain=0.1)
        nn.init.zeros_(self.state_slot_down.bias)
        nn.init.xavier_uniform_(self.state_readout.weight, gain=0.1)
        if self.local_short_window:
            nn.init.xavier_uniform_(self.local_down.weight, gain=0.1)
            nn.init.zeros_(self.local_down.bias)

        if self.use_slot_state and not self.separate_local_state:
            nn.init.normal_(self.state_gamma.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.state_gamma.bias)
            nn.init.normal_(self.state_beta.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.state_beta.bias)
        else:
            nn.init.zeros_(self.state_gamma.weight)
            nn.init.zeros_(self.state_gamma.bias)
            nn.init.zeros_(self.state_beta.weight)
            nn.init.zeros_(self.state_beta.bias)
        nn.init.zeros_(self.out_gate.weight)
        self.out_gate.bias.data.fill_(-2.0)
        nn.init.zeros_(self.state_gate.weight)
        self.state_gate.bias.data.fill_(-1.0 if self.use_slot_state else -2.0)

    def _pool_spatial(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        out_h = h // self.pooling_ratio
        x = x.permute(0, 2, 1).reshape(b, c, h, h)
        x = F.adaptive_avg_pool2d(x, (out_h, out_h))
        return x.flatten(2).transpose(1, 2)

    def _pool_enrich(self, enrich):
        b, tm1, s_pre, d = enrich.shape
        x = enrich.reshape(b * tm1, s_pre, d)
        x = self._pool_spatial(x)
        s_post = x.shape[1]
        return x.reshape(b, tm1, s_post, d)

    def _pool_hidden(self, h):
        b, tm1, s_pre, d = h.shape
        x = h.reshape(b * tm1, s_pre, d)
        x = self._pool_spatial(x)
        s_post = x.shape[1]
        return x.reshape(b, tm1, s_post, d)

    def _pool_state_slots(self, x):
        # x: (B, T, S, D), where S is a spatial square token grid
        b, t, s_in, d = x.shape
        h = int(math.sqrt(s_in))
        if h * h != s_in:
            raise ValueError(f"state slot pooling expects square token grid, got {s_in}")
        x_2d = x.permute(0, 1, 3, 2).reshape(b * t, d, h, h)
        x_2d = F.adaptive_avg_pool2d(x_2d, (self.state_slot_grid, self.state_slot_grid))
        slots = x_2d.flatten(2).transpose(1, 2)
        return slots.reshape(b, t, self.state_slots, d)

    @staticmethod
    def _stable_delta_recurrence(seq, q_proj, k_proj, v_proj, beta_proj):
        # seq: (B, T, D)
        q = q_proj(seq)
        k = k_proj(seq)
        v = v_proj(seq)
        beta = torch.sigmoid(beta_proj(seq))

        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)

        b, t, d = seq.shape
        state = torch.zeros(b, d, d, device=seq.device, dtype=seq.dtype)
        outputs = []
        for i in range(t):
            q_t = q[:, i, :]
            k_t = k[:, i, :]
            v_t = v[:, i, :]
            beta_t = beta[:, i, :]

            erase = torch.einsum("bi,bj->bij", beta_t, k_t)
            write = torch.einsum("bi,bj->bij", beta_t * v_t, k_t)
            state = state * (1.0 - erase) + write
            y = torch.bmm(state, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(y)
        return torch.stack(outputs, dim=1)

    def compose_residuals_from_cache(self, z_base_pooled, local_h, state_h_slots, slot_bias=None):
        b, t, s_post, d = z_base_pooled.shape
        if t <= 1:
            zero = torch.zeros_like(z_base_pooled)
            state_h = torch.zeros(
                b, t, self.state_bottleneck,
                device=z_base_pooled.device, dtype=z_base_pooled.dtype,
            )
            return {
                "state_h": state_h,
                "state_h_slots": state_h_slots,
                "event_gate": torch.zeros(
                    b, t, 1,
                    device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                ),
                "z_local": z_base_pooled,
                "z_fused": z_base_pooled,
                "local_residual": zero,
                "state_residual": zero,
            }

        state_readout_logits = self.state_readout(state_h_slots).squeeze(-1)
        if slot_bias is not None:
            slot_bias = slot_bias.to(
                device=state_readout_logits.device,
                dtype=state_readout_logits.dtype,
            )
            if slot_bias.dim() == 2 and slot_bias.shape[-1] == self.state_slots:
                state_readout_logits = state_readout_logits + slot_bias.unsqueeze(1)
        state_readout = torch.softmax(state_readout_logits, dim=-1)
        state_h = torch.einsum("btk,btkd->btd", state_readout, state_h_slots)
        state_ctx = state_h[:, 1:]

        gamma = 0.1 * torch.tanh(self.state_gamma(state_ctx)).unsqueeze(2)
        beta = self.state_beta(state_ctx).unsqueeze(2)
        fused_h = self.fuse_norm((1.0 + gamma) * local_h + beta)

        local_gate_logits = self.out_gate(local_h)
        state_gate_logits = self.state_gate(state_ctx)
        local_gate = self.max_enrich_gate * torch.sigmoid(local_gate_logits)
        local_enrich_pre = local_gate * self.up(local_h)
        local_enrich_pooled = self._pool_enrich(local_enrich_pre)

        fused_gate = self.max_enrich_gate * torch.sigmoid(
            local_gate_logits + self.max_state_gate * state_gate_logits.unsqueeze(2)
        )
        enrich_pre = fused_gate * self.up(fused_h)
        enrich_pooled = self._pool_enrich(enrich_pre)
        event_gate = fused_gate.mean(dim=2)

        event_gate_full = torch.zeros(
            b, t, 1,
            device=z_base_pooled.device, dtype=z_base_pooled.dtype,
        )
        event_gate_full[:, 1:] = event_gate

        local_residual = torch.zeros_like(z_base_pooled)
        local_residual[:, 1:] = local_enrich_pooled
        fused_residual = torch.zeros_like(z_base_pooled)
        fused_residual[:, 1:] = enrich_pooled
        state_residual = fused_residual - local_residual
        z_local = z_base_pooled + local_residual
        z_fused = z_base_pooled + fused_residual

        return {
            "state_h": state_h,
            "state_h_slots": state_h_slots,
            "event_gate": event_gate_full,
            "z_local": z_local,
            "z_fused": z_fused,
            "local_residual": local_residual,
            "state_residual": state_residual,
        }

    def forward(self, z_pre_pool, z_base_pooled, return_aux=False, modulation=None, return_aux_mode=None):
        b, t, s_pre, d = z_pre_pool.shape
        aux_mode = return_aux_mode or "full"
        s_post = z_base_pooled.shape[2]

        if t <= 1:
            zero = torch.zeros_like(z_base_pooled)
            h_full = torch.zeros(
                b, t, s_post, self.bottleneck,
                device=z_base_pooled.device, dtype=z_base_pooled.dtype,
            )
            self._last_diff_norm = 0.0
            self._last_hidden_norm = 0.0
            self._last_gate_mean = 0.0
            self._last_gate_std = 0.0
            self._last_enrich_norm = 0.0
            state_mean = z_base_pooled.mean(dim=2)
            state_slots = state_mean.unsqueeze(2).expand(-1, -1, self.state_slots, -1).contiguous()
            state_h_slots = torch.zeros(
                b, t, self.state_slots, self.state_bottleneck,
                device=z_base_pooled.device, dtype=z_base_pooled.dtype,
            )
            if self.use_slot_state:
                aux_state_d1 = torch.zeros(
                    b, t, self.state_slots, d,
                    device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                )
                aux_state_d0 = torch.zeros(
                    b, t, self.state_slots, d,
                    device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                )
            else:
                aux_state_d1 = torch.zeros(
                    b, t, d,
                    device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                )
                aux_state_d0 = torch.zeros(
                    b, t, d,
                    device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                )
            if return_aux:
                aux = {
                    "state_h": torch.zeros(
                        b, t, self.state_bottleneck,
                        device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                    ),
                    "state_h_slots": state_h_slots,
                    "event_gate": torch.zeros(
                        b, t, 1,
                        device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                    ),
                }
                if aux_mode in ("residual", "full"):
                    aux.update(
                        {
                            "local_residual": zero,
                            "state_residual": zero,
                        }
                    )
                if aux_mode == "full":
                    aux.update(
                        {
                            "state": state_mean,
                            "state_slots": state_slots,
                            "state_d1": aux_state_d1,
                            "state_d0": aux_state_d0,
                            "z_state": z_base_pooled,
                            "z_local": z_base_pooled,
                            "z_fused": z_base_pooled,
                            "local_h": torch.zeros(
                                b, max(t - 1, 1), s_pre, self.bottleneck,
                                device=z_base_pooled.device, dtype=z_base_pooled.dtype,
                            ),
                        }
                    )
                return z_base_pooled, zero, h_full, aux
            return z_base_pooled, zero, h_full

        # Local branch: keep it explicitly short-horizon and spatially local.
        # When enabled, this avoids turning local into another long-memory branch.
        diff = self.diff_norm(z_pre_pool[:, 1:] - z_pre_pool[:, :-1])
        if self.local_short_window:
            diff2 = torch.zeros_like(diff)
            if t > 2:
                diff2[:, 1:] = z_pre_pool[:, 2:] - z_pre_pool[:, :-2]
            local_in = torch.cat([diff, diff2], dim=-1)
            x = self.local_down(self.local_input_norm(local_in))
            x_tok = x.permute(0, 2, 3, 1).reshape(b * s_pre, self.bottleneck, t - 1)
            x_tok = self.temporal_conv(x_tok)
            x_tok = x_tok.reshape(b, s_pre, self.bottleneck, t - 1).permute(0, 3, 1, 2)
            local_h = self.output_norm(x_tok)
        else:
            x = self.down(diff)
            x_tok = x.permute(0, 2, 3, 1).reshape(b * s_pre, self.bottleneck, t - 1)
            x_tok = self.temporal_conv(x_tok)
            x_tok = x_tok.reshape(b, s_pre, self.bottleneck, t - 1).permute(0, 3, 1, 2)
            x = self.hidden_norm(x_tok)

            seq = x.permute(0, 2, 1, 3).reshape(b * s_pre, t - 1, self.bottleneck)
            local_h = self._stable_delta_recurrence(seq, self.q_proj, self.k_proj, self.v_proj, self.beta_proj)
            local_h = local_h.reshape(b, s_pre, t - 1, self.bottleneck).permute(0, 2, 1, 3)
            local_h = self.output_norm(local_h)

        # State branch. In v9.2.5 we can optionally derive state slots directly
        # from pre-pool spatial tokens so state uses a coarser high-res view,
        # while local keeps tokenwise adjacent diffs.
        if self.use_slot_state:
            state_source = z_pre_pool if self.state_from_prepool else z_base_pooled
            state_slots = self._pool_state_slots(state_source)
            state_d1 = torch.zeros_like(state_slots)
            state_d1[:, 1:] = state_slots[:, 1:] - state_slots[:, :-1]
            state_d0 = state_slots - state_slots[:, :1]
            state_slot_in = torch.cat([state_slots, state_d1, state_d0], dim=-1)
            state_slot_in = self.state_slot_input_norm(state_slot_in)
            state_slot_x = self.state_slot_down(state_slot_in)  # (B, T, K, Ds)
            state_seq = state_slot_x.permute(0, 2, 1, 3).reshape(
                b * self.state_slots, t, self.state_bottleneck
            )
            state_seq = self.state_temporal_conv(state_seq.transpose(1, 2)).transpose(1, 2)
            state_seq = self.state_hidden_norm(state_seq)
            state_seq = self._stable_delta_recurrence(
                state_seq,
                self.state_q_proj,
                self.state_k_proj,
                self.state_v_proj,
                self.state_beta_proj,
            )
            state_h_slots = state_seq.reshape(b, self.state_slots, t, self.state_bottleneck).permute(0, 2, 1, 3)
            state_h_slots = self.state_output_norm(state_h_slots)
            state_readout_logits = self.state_readout(state_h_slots).squeeze(-1)
            if modulation is not None and "slot_bias" in modulation:
                slot_bias = modulation["slot_bias"].to(
                    device=state_readout_logits.device,
                    dtype=state_readout_logits.dtype,
                )
                if slot_bias.dim() == 2 and slot_bias.shape[-1] == self.state_slots:
                    state_readout_logits = state_readout_logits + slot_bias.unsqueeze(1)
            state_readout = torch.softmax(state_readout_logits, dim=-1)
            state_h = torch.einsum("btk,btkd->btd", state_readout, state_h_slots)
            state = state_slots.mean(dim=2)
        else:
            state_source = z_pre_pool if self.state_from_prepool else z_base_pooled
            state = state_source.mean(dim=2)  # (B, T, D)
            state_d1 = torch.zeros_like(state)
            state_d1[:, 1:] = state[:, 1:] - state[:, :-1]
            state_d0 = state - state[:, :1]
            state_in = torch.cat([state, state_d1, state_d0], dim=-1)
            state_in = self.state_input_norm(state_in)
            state_x = self.state_down(state_in)
            state_slots = state.unsqueeze(2)
            state_d1 = state_d1.unsqueeze(2)
            state_d0 = state_d0.unsqueeze(2)
            state_x = self.state_temporal_conv(state_x.transpose(1, 2)).transpose(1, 2)
            state_x = self.state_hidden_norm(state_x)

            state_h = self._stable_delta_recurrence(
                state_x,
                self.state_q_proj,
                self.state_k_proj,
                self.state_v_proj,
                self.state_beta_proj,
            )
            state_h = self.state_output_norm(state_h)
            state_h_slots = state_h.unsqueeze(2)
        state_ctx = state_h[:, 1:]  # align with local diff outputs on frames 1..T-1

        local_bias_4d = None
        state_bias_3d = None
        state_bias_4d = None
        local_h_mod = local_h
        if modulation is not None and "local_bias" in modulation:
            local_bias_4d = modulation["local_bias"].to(
                device=local_h.device,
                dtype=local_h.dtype,
            ).view(b, 1, 1, 1)
            local_scale = (1.0 + 0.5 * local_bias_4d).clamp(min=0.5, max=1.5)
            local_h_mod = local_h * local_scale
        local_gate_logits = self.out_gate(local_h_mod)
        state_gate_logits = self.state_gate(state_ctx)
        if local_bias_4d is not None:
            local_gate_logits = local_gate_logits + 1.5 * local_bias_4d
        local_gate = self.max_enrich_gate * torch.sigmoid(local_gate_logits)
        local_enrich_pre = local_gate * self.up(local_h_mod)
        local_enrich_pooled = self._pool_enrich(local_enrich_pre)
        if self.separate_local_state:
            if modulation is not None and "state_bias" in modulation:
                state_bias_3d = modulation["state_bias"].to(
                    device=state_ctx.device,
                    dtype=state_ctx.dtype,
                ).view(b, 1, 1)
                state_gate_logits = state_gate_logits + 1.5 * state_bias_3d
            state_gate = self.max_state_gate * torch.sigmoid(state_gate_logits)
            state_enrich_vec = self.state_up(state_ctx)
            state_enrich_pooled = state_gate.unsqueeze(2) * state_enrich_vec.unsqueeze(2).expand(-1, -1, s_post, -1)
            enrich_pooled = local_enrich_pooled + state_enrich_pooled
            event_gate = 0.5 * (local_gate.mean(dim=2) + state_gate)
            z_state = z_base_pooled.clone()
            z_state[:, 1:] = z_base_pooled[:, 1:] + state_enrich_pooled
            h_full = torch.zeros(
                b, t, s_post, self.bottleneck,
                device=z_base_pooled.device, dtype=z_base_pooled.dtype,
            )
            h_full[:, 1:] = self._pool_hidden(local_h_mod)
            self._last_hidden_norm = 0.5 * (
                local_h_mod.detach().norm(dim=-1).mean().item()
                + state_h_slots.detach().norm(dim=-1).mean().item()
            )
            self._last_gate_mean = local_gate.detach().mean().item()
            self._last_gate_std = state_gate.detach().mean().item()
        else:
            gamma = 0.1 * torch.tanh(self.state_gamma(state_ctx)).unsqueeze(2)
            beta = self.state_beta(state_ctx).unsqueeze(2)
            if modulation is not None and "state_bias" in modulation:
                state_bias_3d = modulation["state_bias"].to(
                    device=state_ctx.device,
                    dtype=state_ctx.dtype,
                ).view(b, 1, 1)
                state_bias_4d = state_bias_3d.unsqueeze(2)
                state_scale = (1.0 + 0.5 * state_bias_4d).clamp(min=0.5, max=1.5)
                gamma = gamma * state_scale
                beta = beta * state_scale
                state_gate_logits = state_gate_logits + 1.5 * state_bias_3d
            fused_h = self.fuse_norm((1.0 + gamma) * local_h_mod + beta)
            fused_gate = self.max_enrich_gate * torch.sigmoid(
                local_gate_logits + self.max_state_gate * state_gate_logits.unsqueeze(2)
            )
            enrich_pre = fused_gate * self.up(fused_h)
            enrich_pooled = self._pool_enrich(enrich_pre)
            event_gate = fused_gate.mean(dim=2)
            z_state = None
            fused_h_pooled = self._pool_hidden(fused_h)
            s_hidden = fused_h_pooled.shape[2]
            h_full = torch.zeros(
                b, t, s_hidden, self.bottleneck,
                device=z_base_pooled.device, dtype=z_base_pooled.dtype,
            )
            h_full[:, 1:] = fused_h_pooled
            if self.state_bottleneck == self.bottleneck:
                h_full[:, 0] = state_h[:, 0].unsqueeze(1).expand(-1, h_full.shape[2], -1)
            self._last_hidden_norm = 0.5 * (
                fused_h.detach().norm(dim=-1).mean().item()
                + state_h_slots.detach().norm(dim=-1).mean().item()
            )
            self._last_gate_mean = fused_gate.detach().mean().item()
            self._last_gate_std = fused_gate.detach().std().item()
        event_gate_full = torch.zeros(
            b, t, 1,
            device=z_base_pooled.device, dtype=z_base_pooled.dtype,
        )
        event_gate_full[:, 1:] = event_gate

        enrich_full = torch.zeros_like(z_base_pooled)
        enrich_full[:, 1:] = enrich_pooled
        local_enrich_full = torch.zeros_like(z_base_pooled)
        local_enrich_full[:, 1:] = local_enrich_pooled
        if self.separate_local_state:
            state_enrich_full = torch.zeros_like(z_base_pooled)
            state_enrich_full[:, 1:] = state_enrich_pooled
            state_residual_full = state_enrich_full
        else:
            state_residual_full = enrich_full - local_enrich_full
        z_local = z_base_pooled + local_enrich_full
        z_temp = z_base_pooled + enrich_full

        self._last_diff_norm = diff.detach().norm(dim=-1).mean().item()
        self._last_enrich_norm = enrich_full.detach().norm(dim=-1).mean().item()
        if return_aux:
            aux = {
                "state_h": state_h,
                "state_h_slots": state_h_slots,
                "event_gate": event_gate_full,
            }
            if aux_mode in ("residual", "full"):
                aux.update(
                    {
                        "local_residual": local_enrich_full,
                        "state_residual": state_residual_full,
                    }
                )
            if aux_mode == "full":
                aux_state_d1 = state_d1 if self.use_slot_state else state_d1.squeeze(2)
                aux_state_d0 = state_d0 if self.use_slot_state else state_d0.squeeze(2)
                aux.update(
                    {
                        "state": state,
                        "state_slots": state_slots,
                        "state_d1": aux_state_d1,
                        "state_d0": aux_state_d0,
                        "z_state": z_state if z_state is not None else (z_base_pooled + state_residual_full),
                        "z_local": z_local,
                        "z_fused": z_temp,
                        "local_h": local_h,
                    }
                )
            return z_temp, enrich_full, h_full, aux
        return z_temp, enrich_full, h_full


class EventFocusedTemporalExpert(nn.Module):
    """Temporal evidence specialist for v10.

    Local adjacent-frame diff branch is the main expert. A smaller state branch
    only estimates temporal evidence confidence and should not rewrite semantic
    content aggressively. The output stays residual:

      z_temp = z_base + evidence_gate * local_enrich
    """

    def __init__(
        self,
        proj_dim=2048,
        bottleneck=128,
        state_bottleneck=128,
        state_slots=4,
        kernel_size=3,
        max_enrich_gate=0.25,
        max_state_gate=0.25,
        pooling_ratio=2,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.bottleneck = bottleneck
        self.state_bottleneck = int(state_bottleneck)
        self.state_slots = int(max(1, state_slots))
        self.state_slot_grid = int(math.sqrt(self.state_slots))
        if self.state_slot_grid * self.state_slot_grid != self.state_slots:
            raise ValueError(f"state_slots must be a perfect square, got {self.state_slots}")
        self.max_enrich_gate = max_enrich_gate
        self.max_state_gate = max_state_gate
        self.pooling_ratio = pooling_ratio

        # Local temporal evidence branch.
        self.diff_norm = nn.LayerNorm(proj_dim)
        self.down = nn.Linear(proj_dim, bottleneck)
        self.temporal_conv = nn.Conv1d(
            bottleneck, bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=bottleneck,
        )
        self.hidden_norm = nn.LayerNorm(bottleneck)
        self.output_norm = nn.LayerNorm(bottleneck)
        self.q_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.k_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.v_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.beta_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, proj_dim)
        self.out_gate = nn.Linear(bottleneck, 1)

        # Lightweight evidence controller branch.
        self.state_slot_input_norm = nn.LayerNorm(proj_dim * 2)
        self.state_slot_down = nn.Linear(proj_dim * 2, self.state_bottleneck)
        self.state_temporal_conv = nn.Conv1d(
            self.state_bottleneck, self.state_bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=self.state_bottleneck,
        )
        self.state_hidden_norm = nn.LayerNorm(self.state_bottleneck)
        self.state_output_norm = nn.LayerNorm(self.state_bottleneck)
        self.state_q_proj = nn.Linear(self.state_bottleneck, self.state_bottleneck, bias=False)
        self.state_k_proj = nn.Linear(self.state_bottleneck, self.state_bottleneck, bias=False)
        self.state_v_proj = nn.Linear(self.state_bottleneck, self.state_bottleneck, bias=False)
        self.state_beta_proj = nn.Linear(self.state_bottleneck, self.state_bottleneck, bias=False)
        self.state_readout = nn.Linear(self.state_bottleneck, 1, bias=False)
        self.state_gate = nn.Linear(self.state_bottleneck, 1)

        nn.init.xavier_uniform_(self.up.weight, gain=0.01)
        nn.init.zeros_(self.up.bias)
        for m in [
            self.q_proj, self.k_proj, self.v_proj, self.beta_proj,
            self.state_q_proj, self.state_k_proj, self.state_v_proj, self.state_beta_proj,
        ]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        nn.init.xavier_uniform_(self.state_slot_down.weight, gain=0.1)
        nn.init.zeros_(self.state_slot_down.bias)
        nn.init.xavier_uniform_(self.state_readout.weight, gain=0.1)
        nn.init.zeros_(self.out_gate.weight)
        self.out_gate.bias.data.fill_(-2.0)
        nn.init.zeros_(self.state_gate.weight)
        self.state_gate.bias.data.fill_(-1.5)

    def _pool_spatial(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        out_h = h // self.pooling_ratio
        x = x.permute(0, 2, 1).reshape(b, c, h, h)
        x = F.adaptive_avg_pool2d(x, (out_h, out_h))
        return x.flatten(2).transpose(1, 2)

    def _pool_enrich(self, enrich):
        b, tm1, s_pre, d = enrich.shape
        x = enrich.reshape(b * tm1, s_pre, d)
        x = self._pool_spatial(x)
        s_post = x.shape[1]
        return x.reshape(b, tm1, s_post, d)

    def _pool_hidden(self, h):
        b, tm1, s_pre, d = h.shape
        x = h.reshape(b * tm1, s_pre, d)
        x = self._pool_spatial(x)
        s_post = x.shape[1]
        return x.reshape(b, tm1, s_post, d)

    def _pool_state_slots(self, x):
        b, t, s_post, d = x.shape
        h = int(math.sqrt(s_post))
        if h * h != s_post:
            raise ValueError(f"state slot pooling expects square token grid, got {s_post}")
        x_2d = x.permute(0, 1, 3, 2).reshape(b * t, d, h, h)
        x_2d = F.adaptive_avg_pool2d(x_2d, (self.state_slot_grid, self.state_slot_grid))
        slots = x_2d.flatten(2).transpose(1, 2)
        return slots.reshape(b, t, self.state_slots, d)

    @staticmethod
    def _stable_delta_recurrence(seq, q_proj, k_proj, v_proj, beta_proj):
        q = q_proj(seq)
        k = k_proj(seq)
        v = v_proj(seq)
        beta = torch.sigmoid(beta_proj(seq))
        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)
        b, t, d = seq.shape
        state = torch.zeros(b, d, d, device=seq.device, dtype=seq.dtype)
        outputs = []
        for i in range(t):
            q_t = q[:, i, :]
            k_t = k[:, i, :]
            v_t = v[:, i, :]
            beta_t = beta[:, i, :]
            erase = torch.einsum("bi,bj->bij", beta_t, k_t)
            write = torch.einsum("bi,bj->bij", beta_t * v_t, k_t)
            state = state * (1.0 - erase) + write
            y = torch.bmm(state, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(y)
        return torch.stack(outputs, dim=1)

    def forward(self, z_pre_pool, z_base_pooled, return_aux=False):
        b, t, s_pre, d = z_pre_pool.shape
        s_post = z_base_pooled.shape[2]
        if t <= 1:
            zero = torch.zeros_like(z_base_pooled)
            h_full = torch.zeros(b, t, s_post, self.bottleneck, device=z_base_pooled.device, dtype=z_base_pooled.dtype)
            self._last_diff_norm = 0.0
            self._last_hidden_norm = 0.0
            self._last_gate_mean = 0.0
            self._last_gate_std = 0.0
            self._last_enrich_norm = 0.0
            state_slots = self._pool_state_slots(z_base_pooled)
            aux = {
                "state_h": torch.zeros(b, t, self.state_bottleneck, device=z_base_pooled.device, dtype=z_base_pooled.dtype),
                "state_h_slots": torch.zeros(b, t, self.state_slots, self.state_bottleneck, device=z_base_pooled.device, dtype=z_base_pooled.dtype),
                "state_slots": state_slots,
                "event_gate": torch.zeros(b, t, 1, device=z_base_pooled.device, dtype=z_base_pooled.dtype),
                "diff_score": torch.zeros(b, t, 1, device=z_base_pooled.device, dtype=z_base_pooled.dtype),
            }
            if return_aux:
                return z_base_pooled, zero, h_full, aux
            return z_base_pooled, zero, h_full

        diff = self.diff_norm(z_pre_pool[:, 1:] - z_pre_pool[:, :-1])
        x = self.down(diff)
        x_tok = x.permute(0, 2, 3, 1).reshape(b * s_pre, self.bottleneck, t - 1)
        x_tok = self.temporal_conv(x_tok)
        x_tok = x_tok.reshape(b, s_pre, self.bottleneck, t - 1).permute(0, 3, 1, 2)
        x = self.hidden_norm(x_tok)

        seq = x.permute(0, 2, 1, 3).reshape(b * s_pre, t - 1, self.bottleneck)
        local_h = self._stable_delta_recurrence(seq, self.q_proj, self.k_proj, self.v_proj, self.beta_proj)
        local_h = local_h.reshape(b, s_pre, t - 1, self.bottleneck).permute(0, 2, 1, 3)
        local_h = self.output_norm(local_h)

        state_slots = self._pool_state_slots(z_base_pooled)
        state_d1 = torch.zeros_like(state_slots)
        state_d1[:, 1:] = state_slots[:, 1:] - state_slots[:, :-1]
        state_slot_in = torch.cat([state_slots, state_d1], dim=-1)
        state_slot_in = self.state_slot_input_norm(state_slot_in)
        state_slot_x = self.state_slot_down(state_slot_in)
        state_seq = state_slot_x.permute(0, 2, 1, 3).reshape(b * self.state_slots, t, self.state_bottleneck)
        state_seq = self.state_temporal_conv(state_seq.transpose(1, 2)).transpose(1, 2)
        state_seq = self.state_hidden_norm(state_seq)
        state_seq = self._stable_delta_recurrence(
            state_seq, self.state_q_proj, self.state_k_proj, self.state_v_proj, self.state_beta_proj
        )
        state_h_slots = state_seq.reshape(b, self.state_slots, t, self.state_bottleneck).permute(0, 2, 1, 3)
        state_h_slots = self.state_output_norm(state_h_slots)
        state_readout = torch.softmax(self.state_readout(state_h_slots).squeeze(-1), dim=-1)
        state_h = torch.einsum("btk,btkd->btd", state_readout, state_h_slots)

        state_ctx = state_h[:, 1:]
        diff_score = diff.detach().norm(dim=-1).mean(dim=2, keepdim=True)
        diff_score = diff_score / diff_score.mean(dim=1, keepdim=True).clamp(min=1e-6)
        evidence_gate = self.max_enrich_gate * torch.sigmoid(
            self.out_gate(local_h) + self.max_state_gate * self.state_gate(state_ctx).unsqueeze(2)
        )
        enrich_pre = evidence_gate * self.up(local_h)
        enrich_pooled = self._pool_enrich(enrich_pre)
        event_gate = evidence_gate.mean(dim=2)
        event_gate_full = torch.zeros(b, t, 1, device=z_base_pooled.device, dtype=z_base_pooled.dtype)
        event_gate_full[:, 1:] = event_gate
        diff_score_full = torch.zeros(b, t, 1, device=z_base_pooled.device, dtype=z_base_pooled.dtype)
        diff_score_full[:, 1:] = diff_score

        enrich_full = torch.zeros_like(z_base_pooled)
        enrich_full[:, 1:] = enrich_pooled
        z_temp = z_base_pooled + enrich_full

        h_full = torch.zeros(b, t, enrich_pooled.shape[2], self.bottleneck, device=z_base_pooled.device, dtype=z_base_pooled.dtype)
        h_full[:, 1:] = self._pool_hidden(local_h)

        self._last_diff_norm = diff.detach().norm(dim=-1).mean().item()
        self._last_hidden_norm = 0.5 * (
            local_h.detach().norm(dim=-1).mean().item() + state_h_slots.detach().norm(dim=-1).mean().item()
        )
        self._last_gate_mean = event_gate.detach().mean().item()
        self._last_gate_std = event_gate.detach().std().item()
        self._last_enrich_norm = enrich_full.detach().norm(dim=-1).mean().item()
        if return_aux:
            aux = {
                "state_h": state_h,
                "state_h_slots": state_h_slots,
                "state_slots": state_slots,
                "event_gate": event_gate_full,
                "diff_score": diff_score_full,
            }
            return z_temp, enrich_full, h_full, aux
        return z_temp, enrich_full, h_full


class FactorizedTemporalMemoryExpert(nn.Module):
    """Factorized temporal memory expert with motion/state/correspondence branches."""

    def __init__(
        self,
        proj_dim=2048,
        bottleneck=128,
        state_bottleneck=128,
        corr_bottleneck=128,
        state_slots=16,
        kernel_size=3,
        max_motion_gate=0.20,
        max_state_gate=0.15,
        max_corr_gate=0.15,
        pooling_ratio=2,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.bottleneck = bottleneck
        self.state_bottleneck = int(state_bottleneck)
        self.corr_bottleneck = int(corr_bottleneck)
        self.state_slots = int(max(1, state_slots))
        self.state_slot_grid = int(math.sqrt(self.state_slots))
        if self.state_slot_grid * self.state_slot_grid != self.state_slots:
            raise ValueError(f"state_slots must be a perfect square, got {self.state_slots}")
        self.max_motion_gate = float(max_motion_gate)
        self.max_state_gate = float(max_state_gate)
        self.max_corr_gate = float(max_corr_gate)
        self.pooling_ratio = int(pooling_ratio)

        self.motion_diff_norm = nn.LayerNorm(proj_dim)
        self.motion_down = nn.Linear(proj_dim, bottleneck)
        self.motion_temporal_conv = nn.Conv1d(
            bottleneck, bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=bottleneck,
        )
        self.motion_hidden_norm = nn.LayerNorm(bottleneck)
        self.motion_up = nn.Linear(bottleneck, proj_dim)
        self.motion_gate = nn.Linear(bottleneck, 1)

        self.state_input_norm = nn.LayerNorm(proj_dim)
        self.state_down = nn.Linear(proj_dim, self.state_bottleneck)
        self.state_temporal_conv = nn.Conv1d(
            self.state_bottleneck, self.state_bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=self.state_bottleneck,
        )
        self.state_hidden_norm = nn.LayerNorm(self.state_bottleneck)
        self.state_up = nn.Linear(self.state_bottleneck, proj_dim)
        self.state_gate = nn.Linear(self.state_bottleneck, 1)

        self.corr_input_norm = nn.LayerNorm(proj_dim * 2)
        self.corr_down = nn.Linear(proj_dim * 2, self.corr_bottleneck)
        self.corr_temporal_conv = nn.Conv1d(
            self.corr_bottleneck, self.corr_bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=self.corr_bottleneck,
        )
        self.corr_hidden_norm = nn.LayerNorm(self.corr_bottleneck)
        self.corr_up = nn.Linear(self.corr_bottleneck, proj_dim)
        self.corr_gate = nn.Linear(self.corr_bottleneck, 1)

        for up in (self.motion_up, self.state_up, self.corr_up):
            nn.init.xavier_uniform_(up.weight, gain=0.01)
            nn.init.zeros_(up.bias)
        for gate, bias in (
            (self.motion_gate, -2.0),
            (self.state_gate, -2.2),
            (self.corr_gate, -2.2),
        ):
            nn.init.zeros_(gate.weight)
            gate.bias.data.fill_(bias)

        self._last_motion_gate = 0.0
        self._last_state_gate = 0.0
        self._last_corr_gate = 0.0
        self._last_motion_norm = 0.0
        self._last_state_norm = 0.0
        self._last_corr_norm = 0.0
        self._last_total_norm = 0.0

    def _pool_pre_to_post(self, x):
        b, t, n_pre, d = x.shape
        if n_pre % (self.pooling_ratio ** 2) != 0:
            raise ValueError(f"Expected token count divisible by pooling square, got {n_pre}")
        n_post = n_pre // (self.pooling_ratio ** 2)
        return x.reshape(b, t, n_post, self.pooling_ratio ** 2, d).mean(dim=3)

    def _pool_tokens_to_slots(self, x):
        b, t, m, d = x.shape
        side = int(math.sqrt(m))
        if side * side != m:
            raise ValueError(f"Expected square token grid, got {m}")
        x_2d = x.permute(0, 1, 3, 2).reshape(b * t, d, side, side)
        x_2d = F.adaptive_avg_pool2d(x_2d, (self.state_slot_grid, self.state_slot_grid))
        slots = x_2d.flatten(2).transpose(1, 2)
        return slots.reshape(b, t, self.state_slots, d)

    def _expand_slots_to_tokens(self, slots, target_tokens):
        b, t, s, d = slots.shape
        side = int(math.sqrt(target_tokens))
        if side * side != target_tokens:
            raise ValueError(f"Expected square token grid, got {target_tokens}")
        x_2d = slots.permute(0, 1, 3, 2).reshape(b * t, d, self.state_slot_grid, self.state_slot_grid)
        x_2d = F.interpolate(x_2d, size=(side, side), mode="bilinear", align_corners=False)
        return x_2d.flatten(2).transpose(1, 2).reshape(b, t, target_tokens, d)

    @staticmethod
    def _apply_temporal_conv(x, conv, norm):
        b, t, m, d = x.shape
        x_conv = x.permute(0, 2, 3, 1).reshape(b * m, d, t)
        x_conv = conv(x_conv)
        x_conv = x_conv.reshape(b, m, d, t).permute(0, 3, 1, 2)
        return norm(x_conv)

    def forward(self, z_pre_pool, z_base, return_aux=False):
        b, t, _, _ = z_pre_pool.shape
        m = z_base.shape[2]
        zero = torch.zeros_like(z_base)
        base_pool = z_base.mean(dim=2)
        base_slots = self._pool_tokens_to_slots(z_base)
        if t <= 1:
            aux = {
                "motion_summary": torch.zeros(b, t, self.bottleneck, device=z_base.device, dtype=z_base.dtype),
                "state_summary": torch.zeros(b, t, self.state_bottleneck, device=z_base.device, dtype=z_base.dtype),
                "state_slot_summary": torch.zeros(b, t, self.state_slots, self.state_bottleneck, device=z_base.device, dtype=z_base.dtype),
                "corr_summary": torch.zeros(b, t, self.corr_bottleneck, device=z_base.device, dtype=z_base.dtype),
                "base_pool": base_pool,
                "base_slots": base_slots,
                "motion_gate": torch.zeros(b, t, 1, device=z_base.device, dtype=z_base.dtype),
                "state_gate": torch.zeros(b, t, 1, device=z_base.device, dtype=z_base.dtype),
                "corr_gate": torch.zeros(b, t, 1, device=z_base.device, dtype=z_base.dtype),
            }
            branch = {"motion": zero, "state": zero, "corr": zero}
            if return_aux:
                return z_base, branch, aux
            return z_base

        motion_diff = torch.zeros_like(z_pre_pool)
        motion_diff[:, 1:] = self.motion_diff_norm(z_pre_pool[:, 1:] - z_pre_pool[:, :-1])
        motion_pooled = self._pool_pre_to_post(motion_diff)
        motion_h = F.gelu(self.motion_down(motion_pooled))
        motion_h = self._apply_temporal_conv(motion_h, self.motion_temporal_conv, self.motion_hidden_norm)
        motion_summary = motion_h.mean(dim=2)
        motion_gate = self.max_motion_gate * torch.sigmoid(self.motion_gate(motion_summary))
        motion_delta = motion_gate.unsqueeze(2) * self.motion_up(motion_h)

        state_h_slots = F.gelu(self.state_down(self.state_input_norm(base_slots)))
        state_h_slots = self._apply_temporal_conv(state_h_slots, self.state_temporal_conv, self.state_hidden_norm)
        state_summary = state_h_slots.mean(dim=2)
        state_gate = self.max_state_gate * torch.sigmoid(self.state_gate(state_summary))
        state_delta_slots = state_gate.unsqueeze(2) * self.state_up(state_h_slots)
        state_delta = self._expand_slots_to_tokens(state_delta_slots, m)

        anchor = z_base[:, :1].expand(-1, t, -1, -1)
        prev = torch.cat([z_base[:, :1], z_base[:, :-1]], dim=1)
        corr_in = torch.cat([z_base - anchor, z_base - prev], dim=-1)
        corr_h = F.gelu(self.corr_down(self.corr_input_norm(corr_in)))
        corr_h = self._apply_temporal_conv(corr_h, self.corr_temporal_conv, self.corr_hidden_norm)
        corr_summary = corr_h.mean(dim=2)
        corr_gate = self.max_corr_gate * torch.sigmoid(self.corr_gate(corr_summary))
        corr_delta = corr_gate.unsqueeze(2) * self.corr_up(corr_h)

        z_mix = z_base + motion_delta + state_delta + corr_delta

        self._last_motion_gate = motion_gate.detach().mean().item()
        self._last_state_gate = state_gate.detach().mean().item()
        self._last_corr_gate = corr_gate.detach().mean().item()
        self._last_motion_norm = motion_delta.detach().norm(dim=-1).mean().item()
        self._last_state_norm = state_delta.detach().norm(dim=-1).mean().item()
        self._last_corr_norm = corr_delta.detach().norm(dim=-1).mean().item()
        self._last_total_norm = (motion_delta + state_delta + corr_delta).detach().norm(dim=-1).mean().item()

        if return_aux:
            branch = {"motion": motion_delta, "state": state_delta, "corr": corr_delta}
            aux = {
                "motion_summary": motion_summary,
                "state_summary": state_summary,
                "state_slot_summary": state_h_slots,
                "corr_summary": corr_summary,
                "base_pool": base_pool,
                "base_slots": base_slots,
                "motion_gate": motion_gate,
                "state_gate": state_gate,
                "corr_gate": corr_gate,
            }
            return z_mix, branch, aux
        return z_mix


class FactorizedGatedDeltaExpert(nn.Module):
    """Factorized bounded DeltaNet-style corrections for motion/state/structure."""

    def __init__(
        self,
        proj_dim=2048,
        bottleneck=128,
        state_bottleneck=128,
        struct_bottleneck=128,
        state_slots=16,
        kernel_size=3,
        max_motion_gate=0.20,
        max_state_gate=0.15,
        max_struct_gate=0.15,
        pooling_ratio=2,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.bottleneck = bottleneck
        self.state_bottleneck = int(state_bottleneck)
        self.struct_bottleneck = int(struct_bottleneck)
        self.state_slots = int(max(1, state_slots))
        self.state_slot_grid = int(math.sqrt(self.state_slots))
        if self.state_slot_grid * self.state_slot_grid != self.state_slots:
            raise ValueError(f"state_slots must be a perfect square, got {self.state_slots}")
        self.max_motion_gate = float(max_motion_gate)
        self.max_state_gate = float(max_state_gate)
        self.max_struct_gate = float(max_struct_gate)
        self.pooling_ratio = int(pooling_ratio)

        # Motion: local adjacent-frame token diffs.
        self.motion_diff_norm = nn.LayerNorm(proj_dim)
        self.motion_down = nn.Linear(proj_dim, bottleneck)
        self.motion_temporal_conv = nn.Conv1d(
            bottleneck, bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=bottleneck,
        )
        self.motion_hidden_norm = nn.LayerNorm(bottleneck)
        self.motion_up = nn.Linear(bottleneck, proj_dim)
        self.motion_gate = nn.Linear(bottleneck, 1)

        # State: coarse slot state evolution.
        self.state_input_norm = nn.LayerNorm(proj_dim)
        self.state_down = nn.Linear(proj_dim, self.state_bottleneck)
        self.state_temporal_conv = nn.Conv1d(
            self.state_bottleneck, self.state_bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=self.state_bottleneck,
        )
        self.state_hidden_norm = nn.LayerNorm(self.state_bottleneck)
        self.state_up = nn.Linear(self.state_bottleneck, proj_dim)
        self.state_gate = nn.Linear(self.state_bottleneck, 1)

        # Structure: long-horizon temporal organization on frame summaries.
        self.struct_input_norm = nn.LayerNorm(proj_dim * 3)
        self.struct_down = nn.Linear(proj_dim * 3, self.struct_bottleneck)
        self.struct_temporal_conv = nn.Conv1d(
            self.struct_bottleneck, self.struct_bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=self.struct_bottleneck,
        )
        self.struct_hidden_norm = nn.LayerNorm(self.struct_bottleneck)
        self.struct_up = nn.Linear(self.struct_bottleneck, proj_dim)
        self.struct_gate = nn.Linear(self.struct_bottleneck, 1)

        for up in (self.motion_up, self.state_up, self.struct_up):
            nn.init.xavier_uniform_(up.weight, gain=0.01)
            nn.init.zeros_(up.bias)
        for gate, bias in (
            (self.motion_gate, -2.0),
            (self.state_gate, -2.2),
            (self.struct_gate, -2.2),
        ):
            nn.init.zeros_(gate.weight)
            gate.bias.data.fill_(bias)

        self._last_motion_gate = 0.0
        self._last_state_gate = 0.0
        self._last_struct_gate = 0.0
        self._last_motion_norm = 0.0
        self._last_state_norm = 0.0
        self._last_struct_norm = 0.0
        self._last_total_norm = 0.0

    def _pool_pre_to_post(self, x):
        b, t, n_pre, d = x.shape
        if n_pre % (self.pooling_ratio ** 2) != 0:
            raise ValueError(f"Expected token count divisible by pooling square, got {n_pre}")
        n_post = n_pre // (self.pooling_ratio ** 2)
        return x.reshape(b, t, n_post, self.pooling_ratio ** 2, d).mean(dim=3)

    def _pool_tokens_to_slots(self, x):
        b, t, m, d = x.shape
        side = int(math.sqrt(m))
        if side * side != m:
            raise ValueError(f"Expected square token grid, got {m}")
        x_2d = x.permute(0, 1, 3, 2).reshape(b * t, d, side, side)
        x_2d = F.adaptive_avg_pool2d(x_2d, (self.state_slot_grid, self.state_slot_grid))
        slots = x_2d.flatten(2).transpose(1, 2)
        return slots.reshape(b, t, self.state_slots, d)

    def _expand_slots_to_tokens(self, slots, target_tokens):
        b, t, s, d = slots.shape
        side = int(math.sqrt(target_tokens))
        if side * side != target_tokens:
            raise ValueError(f"Expected square token grid, got {target_tokens}")
        x_2d = slots.permute(0, 1, 3, 2).reshape(b * t, d, self.state_slot_grid, self.state_slot_grid)
        x_2d = F.interpolate(x_2d, size=(side, side), mode="bilinear", align_corners=False)
        return x_2d.flatten(2).transpose(1, 2).reshape(b, t, target_tokens, d)

    @staticmethod
    def _apply_temporal_conv(x, conv, norm):
        b, t, m, d = x.shape
        x_conv = x.permute(0, 2, 3, 1).reshape(b * m, d, t)
        x_conv = conv(x_conv)
        x_conv = x_conv.reshape(b, m, d, t).permute(0, 3, 1, 2)
        return norm(x_conv)

    def forward(self, z_pre_pool, z_base, return_aux=False):
        b, t, _, _ = z_pre_pool.shape
        m = z_base.shape[2]
        zero = torch.zeros_like(z_base)
        base_pool = z_base.mean(dim=2)
        base_slots = self._pool_tokens_to_slots(z_base)
        if t <= 1:
            aux = {
                "motion_summary": torch.zeros(b, t, self.bottleneck, device=z_base.device, dtype=z_base.dtype),
                "state_summary": torch.zeros(b, t, self.state_bottleneck, device=z_base.device, dtype=z_base.dtype),
                "state_slot_summary": torch.zeros(b, t, self.state_slots, self.state_bottleneck, device=z_base.device, dtype=z_base.dtype),
                "struct_summary": torch.zeros(b, t, self.struct_bottleneck, device=z_base.device, dtype=z_base.dtype),
                "base_pool": base_pool,
                "base_slots": base_slots,
                "motion_gate": torch.zeros(b, t, 1, device=z_base.device, dtype=z_base.dtype),
                "state_gate": torch.zeros(b, t, 1, device=z_base.device, dtype=z_base.dtype),
                "struct_gate": torch.zeros(b, t, 1, device=z_base.device, dtype=z_base.dtype),
            }
            branch = {"motion": zero, "state": zero, "structure": zero}
            if return_aux:
                return z_base, branch, aux
            return z_base

        motion_diff = torch.zeros_like(z_pre_pool)
        motion_diff[:, 1:] = self.motion_diff_norm(z_pre_pool[:, 1:] - z_pre_pool[:, :-1])
        motion_pooled = self._pool_pre_to_post(motion_diff)
        motion_h = F.gelu(self.motion_down(motion_pooled))
        motion_h = self._apply_temporal_conv(motion_h, self.motion_temporal_conv, self.motion_hidden_norm)
        motion_summary = motion_h.mean(dim=2)
        motion_gate = self.max_motion_gate * torch.sigmoid(self.motion_gate(motion_summary))
        motion_delta = motion_gate.unsqueeze(2) * self.motion_up(motion_h)

        state_h_slots = F.gelu(self.state_down(self.state_input_norm(base_slots)))
        state_h_slots = self._apply_temporal_conv(state_h_slots, self.state_temporal_conv, self.state_hidden_norm)
        state_summary = state_h_slots.mean(dim=2)
        state_gate = self.max_state_gate * torch.sigmoid(self.state_gate(state_summary))
        state_delta_slots = state_gate.unsqueeze(2) * self.state_up(state_h_slots)
        state_delta = self._expand_slots_to_tokens(state_delta_slots, m)

        prev_pool = torch.cat([base_pool[:, :1], base_pool[:, :-1]], dim=1)
        anchor_pool = base_pool[:, :1].expand(-1, t, -1)
        struct_in = torch.cat([base_pool, base_pool - prev_pool, base_pool - anchor_pool], dim=-1)
        struct_h = F.gelu(self.struct_down(self.struct_input_norm(struct_in)))
        struct_h = self._apply_temporal_conv(struct_h.unsqueeze(2), self.struct_temporal_conv, self.struct_hidden_norm).squeeze(2)
        struct_gate = self.max_struct_gate * torch.sigmoid(self.struct_gate(struct_h))
        struct_delta = struct_gate.unsqueeze(2) * self.struct_up(struct_h).unsqueeze(2).expand(-1, -1, m, -1)

        z_mix = z_base + motion_delta + state_delta + struct_delta

        self._last_motion_gate = motion_gate.detach().mean().item()
        self._last_state_gate = state_gate.detach().mean().item()
        self._last_struct_gate = struct_gate.detach().mean().item()
        self._last_motion_norm = motion_delta.detach().norm(dim=-1).mean().item()
        self._last_state_norm = state_delta.detach().norm(dim=-1).mean().item()
        self._last_struct_norm = struct_delta.detach().norm(dim=-1).mean().item()
        self._last_total_norm = (motion_delta + state_delta + struct_delta).detach().norm(dim=-1).mean().item()

        if return_aux:
            branch = {"motion": motion_delta, "state": state_delta, "structure": struct_delta}
            aux = {
                "motion_summary": motion_summary,
                "state_summary": state_summary,
                "state_slot_summary": state_h_slots,
                "struct_summary": struct_h,
                "base_pool": base_pool,
                "base_slots": base_slots,
                "motion_gate": motion_gate,
                "state_gate": state_gate,
                "struct_gate": struct_gate,
            }
            return z_mix, branch, aux
        return z_mix


# ════════════════════════════════════════════════════════════════
#  UDH v1: Unified Dual Head Temporal Expert
# ════════════════════════════════════════════════════════════════

class UnifiedDualHeadExpert(nn.Module):
    """Single DeltaNet over pre-pool token diffs with dual output heads.

    One shared DeltaNet encoder produces hidden_h at full 1024-token resolution.
    Two independent heads read from the same hidden:
      - local head: per-token gate × up → pool(1024→256)
      - state head: mean(spatial) → gate × up → broadcast to 256

    This gives stage2 two independent knobs (a_local, a_state) while both
    aux losses (local_corr, state_mask) supervise the same DeltaNet.
    """

    def __init__(
        self,
        proj_dim=2048,
        bottleneck=128,
        kernel_size=3,
        max_local_gate=0.35,
        max_state_gate=0.35,
        pooling_ratio=2,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.bottleneck = bottleneck
        self.max_local_gate = float(max_local_gate)
        self.max_state_gate = float(max_state_gate)
        self.pooling_ratio = int(pooling_ratio)

        # Shared encoder: diff + base context → conv → DeltaNet
        self.diff_norm = nn.LayerNorm(proj_dim)
        self.down_diff = nn.Linear(proj_dim, bottleneck)
        self.down_base = nn.Linear(proj_dim, bottleneck)
        self.temporal_conv = nn.Conv1d(
            bottleneck, bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=bottleneck,
        )
        self.hidden_norm = nn.LayerNorm(bottleneck)
        self.output_norm = nn.LayerNorm(bottleneck)

        # DeltaNet projections
        self.q_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.k_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.v_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.beta_proj = nn.Linear(bottleneck, bottleneck, bias=False)

        # Local head: per-token enrichment
        self.local_up = nn.Linear(bottleneck, proj_dim)
        self.local_gate = nn.Linear(bottleneck, 1)

        # State head: spatial-mean enrichment
        self.state_up = nn.Linear(bottleneck, proj_dim)
        self.state_gate = nn.Linear(bottleneck, 1)

        # Init
        for up in (self.local_up, self.state_up):
            nn.init.xavier_uniform_(up.weight, gain=0.01)
            nn.init.zeros_(up.bias)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.beta_proj):
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
        nn.init.zeros_(self.local_gate.weight)
        self.local_gate.bias.data.fill_(-2.0)
        nn.init.zeros_(self.state_gate.weight)
        self.state_gate.bias.data.fill_(-2.0)

        self._last_diff_norm = 0.0
        self._last_hidden_norm = 0.0
        self._last_local_gate_mean = 0.0
        self._last_state_gate_mean = 0.0
        self._last_local_enrich_norm = 0.0
        self._last_state_enrich_norm = 0.0
        self._last_local_gate_std = 0.0
        self._last_local_gate_max = 0.0
        self._last_local_gate_sparse = 0.0
        self._last_state_gate_std = 0.0
        self._last_state_gate_max = 0.0
        self._last_state_gate_sparse = 0.0

    def _pool_spatial(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        out_h = h // self.pooling_ratio
        x = x.permute(0, 2, 1).reshape(b, c, h, h)
        x = F.adaptive_avg_pool2d(x, (out_h, out_h))
        return x.flatten(2).transpose(1, 2)

    def _pool_enrich(self, enrich):
        b, tm1, s_pre, d = enrich.shape
        x = enrich.reshape(b * tm1, s_pre, d)
        x = self._pool_spatial(x)
        s_post = x.shape[1]
        return x.reshape(b, tm1, s_post, d)

    def forward(self, z_pre_pool, z_base_pooled, return_aux=False, return_aux_mode=None):
        b, t, s_pre, d = z_pre_pool.shape
        s_post = z_base_pooled.shape[2]
        zero = torch.zeros_like(z_base_pooled)

        if t <= 1:
            self._last_diff_norm = 0.0
            self._last_hidden_norm = 0.0
            self._last_local_gate_mean = 0.0
            self._last_state_gate_mean = 0.0
            self._last_local_enrich_norm = 0.0
            self._last_state_enrich_norm = 0.0
            if return_aux:
                aux = {
                    "local_residual": zero,
                    "state_residual": zero,
                    "hidden_h": torch.zeros(b, t, s_pre, self.bottleneck, device=z_base_pooled.device, dtype=z_base_pooled.dtype),
                    "state_h": torch.zeros(b, t, self.bottleneck, device=z_base_pooled.device, dtype=z_base_pooled.dtype),
                    "state": z_base_pooled.mean(dim=2),
                }
                return z_base_pooled, zero, zero, aux
            return z_base_pooled

        # Shared encoder: diff + base context (chunked to reduce peak memory)
        pre_curr = z_pre_pool[:, 1:]
        diff = self.diff_norm(pre_curr - z_pre_pool[:, :-1])
        # Process down projections in temporal chunks to avoid (B,31,1024,2048) peak
        chunk_t = 4
        x_chunks = []
        for ci in range(0, t - 1, chunk_t):
            ce = min(ci + chunk_t, t - 1)
            x_chunks.append(self.down_diff(diff[:, ci:ce]) + self.down_base(pre_curr[:, ci:ce]))
        x = torch.cat(x_chunks, dim=1)
        _diff_norm_val = diff.detach().norm(dim=-1).mean().item()
        del diff, pre_curr

        # Temporal conv (per-token, along time axis)
        x_tok = x.permute(0, 2, 3, 1).reshape(b * s_pre, self.bottleneck, t - 1)
        x_tok = self.temporal_conv(x_tok)
        x_tok = x_tok.reshape(b, s_pre, self.bottleneck, t - 1).permute(0, 3, 1, 2)
        x = self.hidden_norm(x_tok)

        # DeltaNet recurrence (per spatial token, along time)
        seq = x.permute(0, 2, 1, 3).reshape(b * s_pre, t - 1, self.bottleneck)
        q = self.q_proj(seq)
        k = self.k_proj(seq)
        v = self.v_proj(seq)
        beta = torch.sigmoid(self.beta_proj(seq))

        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)

        S = torch.zeros(
            b * s_pre, self.bottleneck, self.bottleneck,
            device=z_pre_pool.device, dtype=z_pre_pool.dtype,
        )

        def _run_deltanet_segment(S_in, q_seg, k_seg, v_seg, beta_seg):
            """Run DeltaNet over a time segment. Wrapped for gradient checkpointing."""
            S_cur = S_in
            seg_outs = []
            for j in range(q_seg.shape[1]):
                q_j = q_seg[:, j, :]
                k_j = k_seg[:, j, :]
                v_j = v_seg[:, j, :]
                beta_j = beta_seg[:, j, :]
                erase = torch.einsum("bi,bj->bij", beta_j, k_j)
                write = torch.einsum("bi,bj->bij", beta_j * v_j, k_j)
                S_cur = S_cur * (1.0 - erase) + write
                y = torch.bmm(S_cur, q_j.unsqueeze(-1)).squeeze(-1)
                seg_outs.append(y)
            return S_cur, torch.stack(seg_outs, dim=1)

        # Run DeltaNet with gradient checkpointing in segments
        seg_size = 8
        all_outputs = []
        for seg_start in range(0, t - 1, seg_size):
            seg_end = min(seg_start + seg_size, t - 1)
            q_seg = q[:, seg_start:seg_end]
            k_seg = k[:, seg_start:seg_end]
            v_seg = v[:, seg_start:seg_end]
            beta_seg = beta[:, seg_start:seg_end]
            if self.training:
                S, seg_out = torch.utils.checkpoint.checkpoint(
                    _run_deltanet_segment, S, q_seg, k_seg, v_seg, beta_seg,
                    use_reentrant=False,
                )
            else:
                S, seg_out = _run_deltanet_segment(S, q_seg, k_seg, v_seg, beta_seg)
            all_outputs.append(seg_out)

        hidden = torch.cat(all_outputs, dim=1)
        hidden = hidden.reshape(b, s_pre, t - 1, self.bottleneck).permute(0, 2, 1, 3)
        hidden_h = self.output_norm(hidden)  # (B, T-1, 1024, 128)

        # Local head: pool first, then gate × up (saves 4x memory vs up-then-pool)
        hidden_h_pooled = self._pool_enrich(hidden_h)  # (B, T-1, 256, 128)
        local_gate_val = self.max_local_gate * torch.sigmoid(self.local_gate(hidden_h_pooled))
        local_enrich_pooled = local_gate_val * self.local_up(hidden_h_pooled)

        # State head: spatial-mean enrichment
        state_summary = hidden_h.mean(dim=2)  # (B, T-1, 128)
        state_gate_val = self.max_state_gate * torch.sigmoid(self.state_gate(state_summary))
        state_enrich_vec = self.state_up(state_summary)  # (B, T-1, 2048)
        state_enrich_pooled = state_gate_val.unsqueeze(2) * state_enrich_vec.unsqueeze(2).expand(-1, -1, s_post, -1)

        # Assemble residuals
        local_residual = torch.zeros_like(z_base_pooled)
        local_residual[:, 1:] = local_enrich_pooled
        state_residual = torch.zeros_like(z_base_pooled)
        state_residual[:, 1:] = state_enrich_pooled

        z_temp = z_base_pooled + local_residual + state_residual

        # Logging
        self._last_diff_norm = _diff_norm_val
        self._last_hidden_norm = hidden_h.detach().norm(dim=-1).mean().item()
        self._last_local_gate_mean = local_gate_val.detach().mean().item()
        self._last_state_gate_mean = state_gate_val.detach().mean().item()
        self._last_local_enrich_norm = local_enrich_pooled.detach().norm(dim=-1).mean().item()
        self._last_state_enrich_norm = state_enrich_pooled.detach().norm(dim=-1).mean().item()
        # Gate distribution stats
        lg_flat = local_gate_val.detach().reshape(-1)
        sg_flat = state_gate_val.detach().reshape(-1)
        self._last_local_gate_std = lg_flat.std().item()
        self._last_local_gate_max = lg_flat.max().item()
        self._last_local_gate_sparse = (lg_flat < 0.01).float().mean().item()
        self._last_state_gate_std = sg_flat.std().item()
        self._last_state_gate_max = sg_flat.max().item()
        self._last_state_gate_sparse = (sg_flat < 0.01).float().mean().item()

        if return_aux:
            base_pool = z_base_pooled.mean(dim=2)
            aux = {
                "local_residual": local_residual,
                "state_residual": state_residual,
                "hidden_h": hidden_h,
                "state_h": state_summary,
                "state": base_pool,
                "z_local": z_base_pooled + local_residual,
                "z_fused": z_temp,
            }
            return z_temp, local_residual, state_residual, aux
        return z_temp


# ════════════════════════════════════════════════════════════════
#  UDH v2: Split Dual Head Temporal Expert
# ════════════════════════════════════════════════════════════════

class SplitDualHeadExpert(nn.Module):
    """Two fully independent branches, each with aux-aligned structure.

    Local branch: per-token(1024) DeltaNet for long-range spatial correspondence.
      - Input: diff + base context
      - DeltaNet(1024 pos) with gradient checkpointing
      - Aux: L_local_corr needs per-token temporal memory

    State branch: spatial-mean DeltaNet(1 pos) for temporal trajectory.
      - Input: [state, Δ1, Δ0] of spatial mean
      - DeltaNet(1 pos) — very lightweight
      - Aux: L_state_mask needs long-range temporal interpolation

    No shared parameters. No gradient interference.
    """

    def __init__(
        self,
        proj_dim=2048,
        bottleneck=128,
        kernel_size=3,
        max_local_gate=0.35,
        max_state_gate=0.35,
        pooling_ratio=2,
        ortho_proj=False,
        skip_output_norm=False,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.bottleneck = bottleneck
        self.max_local_gate = float(max_local_gate)
        self.max_state_gate = float(max_state_gate)
        self.pooling_ratio = int(pooling_ratio)
        self.ortho_proj = bool(ortho_proj)
        self.skip_output_norm = bool(skip_output_norm)
        self.no_gate = False  # Stage 1 video-only: gate 없이 raw enrichment 출력

        # ── Local branch: per-token DeltaNet ──
        self.local_diff_norm = nn.LayerNorm(proj_dim)
        self.local_down_diff = nn.Linear(proj_dim, bottleneck)
        self.local_down_base = nn.Linear(proj_dim, bottleneck)
        self.local_conv = nn.Conv1d(
            bottleneck, bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=bottleneck,
        )
        self.local_hidden_norm = nn.LayerNorm(bottleneck)
        self.local_output_norm = nn.LayerNorm(bottleneck)
        self.local_q_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.local_k_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.local_v_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.local_beta_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.local_alpha_proj = nn.Linear(bottleneck, bottleneck)
        self.local_up = nn.Linear(bottleneck, proj_dim)
        self.local_gate = nn.Linear(bottleneck, 1)

        # ── State branch: spatial-mean → DeltaNet(1 pos) ──
        self.state_input_norm = nn.LayerNorm(proj_dim * 3)
        self.state_down = nn.Linear(proj_dim * 3, bottleneck)
        self.state_conv = nn.Conv1d(
            bottleneck, bottleneck, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=bottleneck,
        )
        self.state_hidden_norm = nn.LayerNorm(bottleneck)
        self.state_output_norm = nn.LayerNorm(bottleneck)
        self.state_q_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.state_k_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.state_v_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.state_beta_proj = nn.Linear(bottleneck, bottleneck, bias=False)
        self.state_alpha_proj = nn.Linear(bottleneck, bottleneck)
        self.state_up = nn.Linear(bottleneck, proj_dim)
        self.state_gate = nn.Linear(bottleneck, 1)

        # Init
        up_gain = 0.1 if self.skip_output_norm else 0.01
        for up in (self.local_up, self.state_up):
            nn.init.xavier_uniform_(up.weight, gain=up_gain)
            nn.init.zeros_(up.bias)
        for proj in (self.local_q_proj, self.local_k_proj, self.local_v_proj, self.local_beta_proj,
                     self.state_q_proj, self.state_k_proj, self.state_v_proj, self.state_beta_proj):
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
        # Alpha (retention gate): σ(2.0) ≈ 0.88 → ~12% decay per step
        for alpha_proj in (self.local_alpha_proj, self.state_alpha_proj):
            nn.init.zeros_(alpha_proj.weight)
            alpha_proj.bias.data.fill_(2.0)
        for gate, bias in ((self.local_gate, -2.0), (self.state_gate, -2.0)):
            nn.init.zeros_(gate.weight)
            gate.bias.data.fill_(bias)

        self._last_diff_norm = 0.0
        self._last_local_hidden_norm = 0.0
        self._last_state_hidden_norm = 0.0
        self._last_local_gate_mean = 0.0
        self._last_state_gate_mean = 0.0
        self._last_local_enrich_norm = 0.0
        self._last_state_enrich_norm = 0.0
        self._last_local_gate_std = 0.0
        self._last_local_gate_max = 0.0
        self._last_local_gate_sparse = 0.0
        self._last_state_gate_std = 0.0
        self._last_state_gate_max = 0.0
        self._last_state_gate_sparse = 0.0
        self._last_local_alpha_mean = 0.0
        self._last_local_alpha_std = 0.0
        self._last_local_alpha_min = 0.0
        self._last_state_alpha_mean = 0.0
        self._last_state_alpha_std = 0.0
        self._last_state_alpha_min = 0.0
        # Enrichment 분화 지표
        self._last_local_enrich_std = 0.0
        self._last_state_enrich_std = 0.0
        self._last_local_frame_std = 0.0
        self._last_state_frame_std = 0.0
        # Ortho projection stats
        self._last_ortho_ratio = 0.0
        self._last_para_ratio = 0.0

    def _project_ortho(self, residual, z_base):
        """Remove the component of residual parallel to z_base (per-token)."""
        # residual, z_base: (B, T, S, D)
        base_sq = (z_base * z_base).sum(dim=-1, keepdim=True).clamp(min=1e-12)
        proj_coeff = (residual * z_base).sum(dim=-1, keepdim=True) / base_sq
        parallel = proj_coeff * z_base
        ortho = residual - parallel
        # Stats (detach)
        with torch.no_grad():
            orig_norm = residual.norm(dim=-1).mean().item()
            if orig_norm > 1e-8:
                self._last_ortho_ratio = ortho.norm(dim=-1).mean().item() / orig_norm
                self._last_para_ratio = parallel.norm(dim=-1).mean().item() / orig_norm
            else:
                self._last_ortho_ratio = 0.0
                self._last_para_ratio = 0.0
        return ortho

    def _pool_spatial(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        out_h = h // self.pooling_ratio
        x = x.permute(0, 2, 1).reshape(b, c, h, h)
        x = F.adaptive_avg_pool2d(x, (out_h, out_h))
        return x.flatten(2).transpose(1, 2)

    def _pool_enrich(self, enrich):
        b, tm1, s_pre, d = enrich.shape
        x = enrich.reshape(b * tm1, s_pre, d)
        x = self._pool_spatial(x)
        s_post = x.shape[1]
        return x.reshape(b, tm1, s_post, d)

    @staticmethod
    def _run_deltanet_segment(S_in, q_seg, k_seg, v_seg, beta_seg, alpha_seg):
        S_cur = S_in
        seg_outs = []
        for j in range(q_seg.shape[1]):
            a = alpha_seg[:, j]
            b = beta_seg[:, j]
            k_j = k_seg[:, j]
            v_j = v_seg[:, j]
            # Gated DeltaNet v1: S_t = α(S - β⊗k) + β*v⊗k
            erase = torch.einsum("bi,bj->bij", b, k_j)
            write = torch.einsum("bi,bj->bij", b * v_j, k_j)
            S_cur = a.unsqueeze(-1) * (S_cur - erase) + write
            seg_outs.append(torch.bmm(S_cur, q_seg[:, j].unsqueeze(-1)).squeeze(-1))
        return S_cur, torch.stack(seg_outs, dim=1)

    def forward(self, z_pre_pool, z_base_pooled, return_aux=False, return_aux_mode=None):
        b, t, s_pre, d = z_pre_pool.shape
        s_post = z_base_pooled.shape[2]
        zero = torch.zeros_like(z_base_pooled)

        if t <= 1:
            self._last_diff_norm = 0.0
            self._last_local_hidden_norm = 0.0
            self._last_state_hidden_norm = 0.0
            self._last_local_gate_mean = 0.0
            self._last_state_gate_mean = 0.0
            self._last_local_enrich_norm = 0.0
            self._last_state_enrich_norm = 0.0
            self._last_local_gate_std = 0.0
            self._last_local_gate_max = 0.0
            self._last_local_gate_sparse = 0.0
            self._last_state_gate_std = 0.0
            self._last_state_gate_max = 0.0
            self._last_state_gate_sparse = 0.0
            self._last_local_alpha_mean = 0.0
            self._last_local_alpha_std = 0.0
            self._last_local_alpha_min = 0.0
            self._last_state_alpha_mean = 0.0
            self._last_state_alpha_std = 0.0
            self._last_state_alpha_min = 0.0
            if return_aux:
                aux = {
                    "local_residual": zero,
                    "state_residual": zero,
                    "local_h": torch.zeros(b, max(t - 1, 1), s_pre, self.bottleneck, device=z_base_pooled.device, dtype=z_base_pooled.dtype),
                    "state_h": torch.zeros(b, t, self.bottleneck, device=z_base_pooled.device, dtype=z_base_pooled.dtype),
                    "state": z_base_pooled.mean(dim=2),
                }
                return z_base_pooled, zero, zero, aux
            return z_base_pooled

        # ══════════════════════════════════════════
        # Local branch: per-token DeltaNet(1024 pos)
        # ══════════════════════════════════════════
        pre_curr = z_pre_pool[:, 1:]
        diff = self.local_diff_norm(pre_curr - z_pre_pool[:, :-1])
        chunk_t = 4
        local_x_chunks = []
        for ci in range(0, t - 1, chunk_t):
            ce = min(ci + chunk_t, t - 1)
            local_x_chunks.append(self.local_down_diff(diff[:, ci:ce]) + self.local_down_base(pre_curr[:, ci:ce]))
        local_x = torch.cat(local_x_chunks, dim=1)
        _diff_norm_val = diff.detach().norm(dim=-1).mean().item()
        del diff, pre_curr

        # Temporal conv
        lx = local_x.permute(0, 2, 3, 1).reshape(b * s_pre, self.bottleneck, t - 1)
        lx = self.local_conv(lx)
        lx = lx.reshape(b, s_pre, self.bottleneck, t - 1).permute(0, 3, 1, 2)
        lx = self.local_hidden_norm(lx)
        del local_x

        # DeltaNet recurrence (1024 pos, gradient checkpointing)
        seq = lx.permute(0, 2, 1, 3).reshape(b * s_pre, t - 1, self.bottleneck)
        del lx
        q = self.local_q_proj(seq)
        k = self.local_k_proj(seq)
        v = self.local_v_proj(seq)
        beta = torch.sigmoid(self.local_beta_proj(seq))
        alpha = torch.sigmoid(self.local_alpha_proj(seq))
        self._last_local_alpha_mean = alpha.detach().mean().item()
        self._last_local_alpha_std = alpha.detach().reshape(-1).std().item()
        self._last_local_alpha_min = alpha.detach().min().item()
        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)

        S = torch.zeros(b * s_pre, self.bottleneck, self.bottleneck, device=z_pre_pool.device, dtype=z_pre_pool.dtype)
        seg_size = 8
        all_local_outputs = []
        for seg_start in range(0, t - 1, seg_size):
            seg_end = min(seg_start + seg_size, t - 1)
            if self.training:
                S, seg_out = torch.utils.checkpoint.checkpoint(
                    self._run_deltanet_segment, S,
                    q[:, seg_start:seg_end], k[:, seg_start:seg_end],
                    v[:, seg_start:seg_end], beta[:, seg_start:seg_end],
                    alpha[:, seg_start:seg_end],
                    use_reentrant=False,
                )
            else:
                S, seg_out = self._run_deltanet_segment(
                    S, q[:, seg_start:seg_end], k[:, seg_start:seg_end],
                    v[:, seg_start:seg_end], beta[:, seg_start:seg_end],
                    alpha[:, seg_start:seg_end],
                )
            all_local_outputs.append(seg_out)
        del q, k, v, beta, alpha, S

        local_hidden = torch.cat(all_local_outputs, dim=1)
        local_hidden = local_hidden.reshape(b, s_pre, t - 1, self.bottleneck).permute(0, 2, 1, 3)
        local_h = local_hidden if self.skip_output_norm else self.local_output_norm(local_hidden)
        del local_hidden

        # Pool first, then gate × up
        local_h_pooled = self._pool_enrich(local_h)  # (B, T-1, 256, 128)
        if self.no_gate:
            local_gate_val = torch.ones(local_h_pooled.shape[:-1] + (1,), device=local_h_pooled.device, dtype=local_h_pooled.dtype)
            local_enrich_pooled = self.local_up(local_h_pooled)
        else:
            local_gate_val = self.max_local_gate * torch.sigmoid(self.local_gate(local_h_pooled))
            local_enrich_pooled = local_gate_val * self.local_up(local_h_pooled)

        # ══════════════════════════════════════════
        # State branch: spatial-mean → DeltaNet(1 pos)
        # ══════════════════════════════════════════
        base_pool = z_pre_pool.mean(dim=2)  # (B, T, 2048)
        state_d1 = torch.zeros_like(base_pool)
        state_d1[:, 1:] = base_pool[:, 1:] - base_pool[:, :-1]
        state_d0 = base_pool - base_pool[:, :1]
        state_in = self.state_input_norm(torch.cat([base_pool, state_d1, state_d0], dim=-1))
        state_x = self.state_down(state_in)

        sx = state_x.transpose(1, 2)
        sx = self.state_conv(sx).transpose(1, 2)
        state_x = self.state_hidden_norm(sx)

        # Gated DeltaNet recurrence (1 pos, T steps — no checkpoint needed)
        sq = self.state_q_proj(state_x)
        sk = self.state_k_proj(state_x)
        sv = self.state_v_proj(state_x)
        sbeta = torch.sigmoid(self.state_beta_proj(state_x))
        salpha = torch.sigmoid(self.state_alpha_proj(state_x))
        self._last_state_alpha_mean = salpha.detach().mean().item()
        self._last_state_alpha_std = salpha.detach().reshape(-1).std().item()
        self._last_state_alpha_min = salpha.detach().min().item()
        sk = F.elu(sk) + 1.0
        sk = sk / (sk.sum(dim=-1, keepdim=True) + 1e-6)

        sS = torch.zeros(b, self.bottleneck, self.bottleneck, device=z_pre_pool.device, dtype=z_pre_pool.dtype)
        state_outputs = []
        for i in range(t):
            a = salpha[:, i]
            b_val = sbeta[:, i]
            k_i = sk[:, i]
            v_i = sv[:, i]
            # Gated DeltaNet v1: S_t = α(S - β⊗k) + β*v⊗k
            erase = torch.einsum("bi,bj->bij", b_val, k_i)
            write = torch.einsum("bi,bj->bij", b_val * v_i, k_i)
            sS = a.unsqueeze(-1) * (sS - erase) + write
            state_outputs.append(torch.bmm(sS, sq[:, i].unsqueeze(-1)).squeeze(-1))

        state_h_raw = torch.stack(state_outputs, dim=1)  # (B, T, 128)
        state_h = state_h_raw if self.skip_output_norm else self.state_output_norm(state_h_raw)
        state_ctx = state_h[:, 1:]  # (B, T-1, 128)

        if self.no_gate:
            state_gate_val = torch.ones(state_ctx.shape[:-1] + (1,), device=state_ctx.device, dtype=state_ctx.dtype)
            state_enrich_vec = self.state_up(state_ctx)
            state_enrich_pooled = state_enrich_vec.unsqueeze(2).expand(-1, -1, s_post, -1)
        else:
            state_gate_val = self.max_state_gate * torch.sigmoid(self.state_gate(state_ctx))
            state_enrich_vec = self.state_up(state_ctx)
            state_enrich_pooled = state_gate_val.unsqueeze(2) * state_enrich_vec.unsqueeze(2).expand(-1, -1, s_post, -1)

        # ══════════════════════════════════════════
        # Assemble
        # ══════════════════════════════════════════
        local_residual = torch.zeros_like(z_base_pooled)
        local_residual[:, 1:] = local_enrich_pooled
        state_residual = torch.zeros_like(z_base_pooled)
        state_residual[:, 1:] = state_enrich_pooled

        total_residual = local_residual + state_residual
        if self.ortho_proj:
            total_residual = self._project_ortho(total_residual, z_base_pooled)

        z_temp = z_base_pooled + total_residual

        self._last_diff_norm = _diff_norm_val
        self._last_local_hidden_norm = local_h.detach().norm(dim=-1).mean().item()
        self._last_state_hidden_norm = state_h.detach().norm(dim=-1).mean().item()
        self._last_local_gate_mean = local_gate_val.detach().mean().item()
        self._last_state_gate_mean = state_gate_val.detach().mean().item()
        le_norms = local_enrich_pooled.detach().norm(dim=-1)  # (B, T-1, S)
        se_norms = state_enrich_pooled.detach().norm(dim=-1)  # (B, T-1, S)
        self._last_local_enrich_norm = le_norms.mean().item()
        self._last_state_enrich_norm = se_norms.mean().item()
        self._last_local_enrich_std = le_norms.std().item()
        self._last_state_enrich_std = se_norms.std().item()
        self._last_local_frame_std = le_norms.mean(dim=-1).std().item()
        self._last_state_frame_std = se_norms.mean(dim=-1).std().item()
        # Gate distribution stats
        lg_flat = local_gate_val.detach().reshape(-1)
        sg_flat = state_gate_val.detach().reshape(-1)
        self._last_local_gate_std = lg_flat.std().item()
        self._last_local_gate_max = lg_flat.max().item()
        self._last_local_gate_sparse = (lg_flat < 0.01).float().mean().item()
        self._last_state_gate_std = sg_flat.std().item()
        self._last_state_gate_max = sg_flat.max().item()
        self._last_state_gate_sparse = (sg_flat < 0.01).float().mean().item()

        if return_aux:
            aux = {
                "local_residual": local_residual,
                "state_residual": state_residual,
                "total_residual": total_residual,
                "local_h": local_h,
                "state_h": state_h,
                "state": z_base_pooled.mean(dim=2),
                "z_local": z_base_pooled + local_residual,
                "z_fused": z_temp,
            }
            return z_temp, local_residual, state_residual, aux
        return z_temp
