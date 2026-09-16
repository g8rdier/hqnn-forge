"""
hqnn_forge.utils
================
Imbalance-robust loss functions and training helpers.

Exported symbols
----------------
FocalLoss               nn.Module implementing Focal Loss (Lin et al. 2017).
weighted_bce_loss       Functional helper: inverse-class-frequency weighted BCE.
compute_class_weights   Computes inverse-frequency class weights from a label tensor.
eval_mode               Context manager: eval mode for a block, submodule modes restored.
save_checkpoint         Write a classifier's class, constructor arguments and weights.
load_checkpoint         Rebuild a classifier from such a file.
"""

from hqnn_forge.utils.imbalance import (
    FocalLoss,
    compute_class_weights,
    weighted_bce_loss,
)
from hqnn_forge.utils.checkpoint import load_checkpoint, save_checkpoint
from hqnn_forge.utils.modes import eval_mode

__all__: list[str] = [
    "FocalLoss",
    "weighted_bce_loss",
    "compute_class_weights",
    "eval_mode",
    "load_checkpoint",
    "save_checkpoint",
]
