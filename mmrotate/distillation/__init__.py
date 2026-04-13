"""
Knowledge Distillation Module for MMRotate
"""

from .kd_loss import (
    SpatialProjectionAdapter,
    FeatureDistillationLoss,
    CosineSimLoss,
    AttentionTransferLoss,
    CrossResolutionKDLoss,
    build_spatial_adapters
)

__all__ = [
    'SpatialProjectionAdapter',
    'FeatureDistillationLoss',
    'CosineSimLoss',
    'AttentionTransferLoss',
    'CrossResolutionKDLoss',
    'build_spatial_adapters'
]
