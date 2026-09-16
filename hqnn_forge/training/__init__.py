"""
hqnn_forge.training
===================
A minimal, model-agnostic train/validate loop with early stopping.

Exported symbols
----------------
train_model       Mini-batch training with per-epoch validation and early stopping.
TrainingHistory   Per-epoch record returned by train_model.
EpochRecord       One epoch of that record.
"""

from hqnn_forge.training.trainer import EpochRecord, TrainingHistory, train_model

__all__: list[str] = [
    "EpochRecord",
    "TrainingHistory",
    "train_model",
]
