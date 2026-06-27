"""
hqnn_forge.utils
================
Imbalance-robust loss functions and training helpers.

Exported symbols
----------------
FocalLoss               nn.Module implementing Focal Loss (Lin et al. 2017).
weighted_bce_loss       Functional helper: inverse-class-frequency weighted BCE.
compute_class_weights   Computes inverse-frequency class weights from a label tensor.
"""

from hqnn_forge.utils.imbalance import (
    FocalLoss,
    compute_class_weights,
    weighted_bce_loss,
)

__all__: list[str] = [
    "FocalLoss",
    "weighted_bce_loss",
    "compute_class_weights",
]
