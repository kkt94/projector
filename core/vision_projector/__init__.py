# Copyright (c) Meta Platforms, Inc. and affiliates.

from core.vision_projector.base import BaseProjector
from core.vision_projector.mlp import MLPProjector

# CRT projector
try:
    from core.vision_projector.crt import CRTProjector
except ImportError:
    CRTProjector = None

# A2 (Dynamics-aware) projector
try:
    from core.vision_projector.a2 import DynamicsAwareA2Projector
except ImportError:
    DynamicsAwareA2Projector = None

# MLP baseline projector (for V-JEPA2 grid tokens)
try:
    from core.vision_projector.mlp_baseline import MLPBaselineProjector
except ImportError:
    MLPBaselineProjector = None

# A3 (Learned Anchor-State / Anchored-Delta) projector
try:
    from core.vision_projector.a3 import A3Projector
except ImportError:
    A3Projector = None

# A4 (single-pass static/event/global decomposition) projector
try:
    from core.vision_projector.a4 import A4Projector
except ImportError:
    A4Projector = None

# BMR (bootstrapped motion router) projector
try:
    from core.vision_projector.bmr import BMRProjector
except ImportError:
    BMRProjector = None

# BMRv2 (dual-MLP + temporal residual gating) projector
try:
    from core.vision_projector.bmr_v2 import BMRv2Projector
except ImportError:
    BMRv2Projector = None

__all__ = [
    "BaseProjector",
    "MLPProjector",
    "CRTProjector",
    "DynamicsAwareA2Projector",
    "MLPBaselineProjector",
    "A3Projector",
    "A4Projector",
    "BMRProjector",
    "BMRv2Projector",
]
