"""
Temporal Query Router: Stage 2 질문-조건부 프레임별 게이팅 모듈.

Stage 1에서 학습된 expert(frozen)의 local/state enrichment를
질문 의미에 따라 프레임별로 선택적 적용.

적용 공식:
  z_final = z_base + local_scale * local_enrich + state_scale * state_enrich

입력:
  - question_emb: LLM embed_tokens → mean pool → (B, D)
  - z_base_summary: z_base의 spatial mean → (B, T, D)
  - local_enrich_summary: local enrichment의 spatial mean → (B, T, D)
  - state_enrich_summary: state enrichment의 spatial mean → (B, T, D)

출력:
  - local_scale: (B, T, 1) ∈ [0, 1]
  - state_scale: (B, T, 1) ∈ [0, 1]
"""

import math
import torch
import torch.nn as nn


class TemporalQueryRouter(nn.Module):
    """질문 + 프레임별 feature로 local/state gate를 결정하는 라우터.

    구조:
      질문(D) + 프레임별 [z_base_mean, local_enrich_mean, state_enrich_mean](3*D)
      → hidden(256) → GELU → 2 gates (sigmoid)

    질문 embedding은 모든 프레임에 broadcast되고,
    각 프레임이 독립적으로 gate 값을 결정.
    """

    def __init__(self, feat_dim, hidden_dim=256, init_bias=0.0):
        """
        Args:
            feat_dim: 프레임 feature / 질문 embedding 차원 (예: 2048)
            hidden_dim: 중간 hidden 차원 (기본값 256)
            init_bias: 출력 레이어 bias 초기값.
                       0.0 → sigmoid(0)=0.5로 시작 (중립적 초기화)
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # 입력: 질문(D) + z_base(D) + local_enrich(D) + state_enrich(D) = 4*D
        input_dim = feat_dim * 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # [base_l, local, base_s, state] — 2-way softmax × 2
        )

        # 초기화: 작은 weight + bias → 초기에 sigmoid(bias) ≈ 0.5 (안정적 시작)
        self._init_weights(init_bias)

        # 로깅용 캐시
        self._last_local_scale_mean = 0.0
        self._last_local_scale_std = 0.0
        self._last_local_scale_min = 0.0
        self._last_local_scale_max = 0.0
        self._last_state_scale_mean = 0.0
        self._last_state_scale_std = 0.0
        self._last_state_scale_min = 0.0
        self._last_state_scale_max = 0.0

    def _init_weights(self, init_bias):
        """안정적 초기화: 출력 layer는 작은 weight + 지정된 bias."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 마지막 레이어: 특별히 작게 초기화
        # 출력 4개: [base_l, local, base_s, state]
        # softmax([base, enrich]) 에서 init_bias로 초기 scale 결정
        # init_bias=-2.2 → softmax([0, -2.2]) ≈ [0.9, 0.1] → scale≈0.1
        last_linear = self.net[-1]
        nn.init.normal_(last_linear.weight, std=0.01)
        # base_logit=0, enrich_logit=init_bias
        last_linear.bias.data[:] = 0.0
        last_linear.bias.data[1] = init_bias  # local enrich
        last_linear.bias.data[3] = init_bias  # state enrich

    def forward(self, question_emb, z_base_summary, local_enrich_summary, state_enrich_summary):
        """
        Args:
            question_emb: (B, D) 질문 embedding (mean-pooled)
            z_base_summary: (B, T, D) z_base의 프레임별 spatial mean
            local_enrich_summary: (B, T, D) local enrichment의 프레임별 spatial mean
            state_enrich_summary: (B, T, D) state enrichment의 프레임별 spatial mean

        Returns:
            local_scale: (B, T, 1) ∈ [0, 1]
            state_scale: (B, T, 1) ∈ [0, 1]
        """
        B, T, D = z_base_summary.shape

        # 질문을 모든 프레임에 broadcast: (B, D) → (B, T, D)
        q_expand = question_emb.unsqueeze(1).expand(B, T, D)

        # 프레임별 입력 결합: (B, T, 4*D)
        x = torch.cat([q_expand, z_base_summary, local_enrich_summary, state_enrich_summary], dim=-1)

        # MLP forward: (B, T, 4*D) → (B, T, 4)
        # [base_logit_local, local_logit, base_logit_state, state_logit]
        logits = self.net(x)

        # Local branch: softmax([base_logit, local_logit]) → local_scale
        local_weights = torch.softmax(logits[:, :, 0:2], dim=-1)
        local_scale = local_weights[:, :, 1:2]  # (B, T, 1) enrichment 비중

        # State branch: softmax([base_logit, state_logit]) → state_scale
        state_weights = torch.softmax(logits[:, :, 2:4], dim=-1)
        state_scale = state_weights[:, :, 1:2]  # (B, T, 1) enrichment 비중

        # 로깅 캐시 업데이트
        with torch.no_grad():
            self._last_local_scale_mean = local_scale.mean().item()
            self._last_local_scale_std = local_scale.std().item()
            self._last_local_scale_min = local_scale.min().item()
            self._last_local_scale_max = local_scale.max().item()
            self._last_state_scale_mean = state_scale.mean().item()
            self._last_state_scale_std = state_scale.std().item()
            self._last_state_scale_min = state_scale.min().item()
            self._last_state_scale_max = state_scale.max().item()

        return local_scale, state_scale
