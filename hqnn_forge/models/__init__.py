"""
hqnn_forge.models
=================
Full hybrid quantum-classical architectures.

Exported symbols
----------------
HybridBinaryClassifier    nn.Module: Linear encoder → QuantumEncodingLayer → Linear head.
ParallelHybridClassifier  nn.Module: classical MLP branch ‖ QuantumEncodingLayer branch → Linear head.
"""

from hqnn_forge.models.hybrid_classifier import HybridBinaryClassifier
from hqnn_forge.models.parallel_hybrid_classifier import ParallelHybridClassifier

__all__: list[str] = [
    "HybridBinaryClassifier",
    "ParallelHybridClassifier",
]
