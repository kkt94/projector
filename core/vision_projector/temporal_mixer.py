"""
Temporal Mixer: Pre-MLP temporal adapter for Perception-LM.

Injects temporal information into vision features BEFORE the frozen MLP
projector. Two sub-modules applied sequentially, each using a mechanism
that is fundamentally different from attention:

  ① Local Temporal Pattern (LTP) — CMR-Delta + Temporal Conv → FiLM
    First decomposes frame-to-frame changes into common-mode (camera
    ego-motion) and local residual (object motion) via parameter-free
    Common-Mode Rejection. Then a learned depthwise temporal conv
    detects shift-invariant patterns in the camera-compensated local
    motion, driving FiLM modulation of the original features.
    Unlike attention: conv is shift-invariant; CMR is global-local
    decomposition, not pairwise comparison.

  ② Global Temporal Context (GTC) — Set Statistics → Enhancement
    Computes aggregate statistics (mean, variance, range) over ALL
    frames at each position — a set function, not pairwise comparison.
    Each frame is then processed in context of this global descriptor,
    learning how to enhance features given the overall temporal profile.
    Unlike attention: uses set-level aggregation, not pairwise Q·K.

Both outputs are gated residuals (near-zero init), so at step 0 the
mixer is approximately an identity function.
For images (T=1), the mixer returns input unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMixer(nn.Module):
    def __init__(
        self,
        vision_dim: int = 1024,
        ltp_bottleneck: int = 128,
        gtc_bottleneck: int = 128,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        C = vision_dim
        d1 = ltp_bottleneck
        d2 = gtc_bottleneck

        # ── ① Local Temporal Pattern (Conv → FiLM) ──
        self.ltp_down = nn.Linear(C, d1, bias=False)
        self.ltp_conv = nn.Conv1d(
            d1, d1, kernel_size=3, padding=1, groups=d1, bias=False,
        )
        self.ltp_norm = nn.LayerNorm(d1)
        self.ltp_scale = nn.Linear(d1, C)
        self.ltp_shift = nn.Linear(d1, C)
        self.ltp_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid ≈ 0.12

        # ── ② Global Temporal Context (Set Stats → Enhancement) ──
        self.gtc_down = nn.Linear(C, d2, bias=False)
        self.gtc_mid = nn.Linear(4 * d2, d2)   # combines frame + 3 stats
        self.gtc_norm = nn.LayerNorm(d2)
        self.gtc_up = nn.Linear(d2, C)
        self.gtc_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid ≈ 0.12

        self._init_weights()

    def _init_weights(self):
        # ① LTP: xavier for down/conv, zero for scale/shift → FiLM identity
        nn.init.xavier_uniform_(self.ltp_down.weight)
        nn.init.zeros_(self.ltp_scale.weight)
        nn.init.zeros_(self.ltp_scale.bias)
        nn.init.zeros_(self.ltp_shift.weight)
        nn.init.zeros_(self.ltp_shift.bias)

        # ② GTC: xavier for down/mid, small random for up (avoids dead gate)
        nn.init.xavier_uniform_(self.gtc_down.weight)
        nn.init.xavier_uniform_(self.gtc_mid.weight)
        nn.init.zeros_(self.gtc_mid.bias)
        nn.init.xavier_uniform_(self.gtc_up.weight)
        nn.init.zeros_(self.gtc_up.bias)
        with torch.no_grad():
            self.gtc_up.weight.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) — vision patch features
               B=batch, T=frames, N=num_patches (1024), C=vision_dim (1024)

        Returns:
            (B, T, N, C) — same shape, with temporal info injected
        """
        B, T, N, C = x.shape

        if T <= 1:
            return x  # Image: identity pass-through

        x = self._local_temporal_pattern(x, B, T, N, C)
        x = self._global_temporal_context(x, B, T, N, C)

        return x

    def _local_temporal_pattern(self, x, B, T, N, C):
        """① CMR-Delta + temporal conv → FiLM modulation.

        Step 1 — Common-Mode Rejection (parameter-free):
          Δ_t = frame[t] - frame[t-1]         (total change)
          g_t = mean across spatial positions  (≈ camera ego-motion)
          δ_t = Δ_t - g_t                     (local object motion only)

        Step 2 — Temporal conv on δ (shift-invariant pattern detection):
          The conv kernel can learn patterns in camera-compensated motion:
            velocity, acceleration, onset/offset, etc.

        Step 3 — FiLM modulation of ORIGINAL features x.
        """
        # ── CMR-Delta: camera-compensated local motion ──
        # Frame diff (first frame zero-padded)
        delta = F.pad(x[:, 1:] - x[:, :-1], (0, 0, 0, 0, 1, 0))  # (B, T, N, C)

        # Common-mode = spatial mean of diffs (≈ camera/ego motion)
        common_mode = delta.mean(dim=2, keepdim=True)              # (B, T, 1, C)

        # Local-only motion (camera removed)
        local_delta = delta - common_mode                          # (B, T, N, C)

        # ── Temporal conv on local motion ──
        # Per-position temporal sequence: (B*N, T, C)
        d = local_delta.permute(0, 2, 1, 3).reshape(B * N, T, C)

        # Bottleneck + learned temporal conv
        h = self.ltp_down(d)                                 # (B*N, T, d1)
        h = h + self.ltp_conv(h.transpose(1, 2)).transpose(1, 2)  # residual conv
        h = self.ltp_norm(h)
        h = F.gelu(h)

        # Generate FiLM parameters
        scale = self.ltp_scale(h)                            # (B*N, T, C)
        shift = self.ltp_shift(h)                            # (B*N, T, C)

        # Reshape back
        scale = scale.reshape(B, N, T, C).permute(0, 2, 1, 3)
        shift = shift.reshape(B, N, T, C).permute(0, 2, 1, 3)

        # FiLM modulation of ORIGINAL features (not delta)
        # At init: scale=0, shift=0 → output = x (identity)
        g = torch.sigmoid(self.ltp_gate)
        return (1.0 + g * scale) * x + g * shift

    def _global_temporal_context(self, x, B, T, N, C):
        """② Set-statistic-based global temporal context → enhancement.

        Computes mean/variance/range over ALL frames at each position
        (set functions, not pairwise comparison). Each frame's features
        are then processed in context of this global descriptor to learn
        frame-specific enhancement.

        The model learns: "given this position's overall temporal profile,
        how should each frame's features be adjusted?"
        """
        # Per-position temporal sequence: (B*N, T, C)
        r = x.permute(0, 2, 1, 3).reshape(B * N, T, C)

        # Bottleneck
        h = self.gtc_down(r)                                 # (B*N, T, d2)

        # Global temporal statistics (SET functions over all T frames)
        t_mean = h.mean(dim=1, keepdim=True)                 # (B*N, 1, d2)
        t_var = h.var(dim=1, keepdim=True)                   # (B*N, 1, d2)
        t_range = (
            h.max(dim=1, keepdim=True).values
            - h.min(dim=1, keepdim=True).values
        )                                                    # (B*N, 1, d2)

        # Broadcast stats to all frames
        ctx = torch.cat([t_mean, t_var, t_range], dim=-1)    # (B*N, 1, 3*d2)
        ctx = ctx.expand(-1, T, -1)                          # (B*N, T, 3*d2)

        # Frame-in-context: each frame + its global temporal descriptor
        combined = torch.cat([h, ctx], dim=-1)               # (B*N, T, 4*d2)

        # Learn frame-specific enhancement from global context
        signal = self.gtc_mid(combined)                      # (B*N, T, d2)
        signal = self.gtc_norm(signal)
        signal = F.gelu(signal)
        signal = self.gtc_up(signal)                         # (B*N, T, C)

        # Reshape back
        signal = signal.reshape(B, N, T, C).permute(0, 2, 1, 3)

        # Gated residual
        return x + torch.sigmoid(self.gtc_gate) * signal

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"vision_dim={self.vision_dim}, "
            f"params={self.count_params():,}, "
            f"ltp_bottleneck={self.ltp_down.out_features}, "
            f"gtc_bottleneck={self.gtc_down.out_features}"
        )
