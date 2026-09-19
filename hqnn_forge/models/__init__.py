"""
hqnn_forge.models
=================
Full hybrid quantum-classical architectures.

Exported symbols
----------------
BinaryClassifierBase      nn.Module base: predict_proba / predict / count_parameters shared by all.
HybridBinaryClassifier    Linear encoder → QuantumEncodingLayer → Linear head.
ParallelHybridClassifier  classical MLP branch ‖ QuantumEncodingLayer branch → Linear head.
MulticlassHybridClassifier  Linear encoder → QuantumEncodingLayer → n_classes heads (softmax / OvR).
"""

from hqnn_forge.models.base import BinaryClassifierBase
from hqnn_forge.models.hybrid_classifier import HybridBinaryClassifier
from hqnn_forge.models.multiclass_hybrid_classifier import MulticlassHybridClassifier
from hqnn_forge.models.parallel_hybrid_classifier import ParallelHybridClassifier

__all__: list[str] = [
    "BinaryClassifierBase",
    "HybridBinaryClassifier",
    "MulticlassHybridClassifier",
    "ParallelHybridClassifier",
]
