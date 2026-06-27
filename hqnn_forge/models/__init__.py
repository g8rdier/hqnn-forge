"""
hqnn_forge.models
=================
Full hybrid quantum-classical architectures.

Exported symbols
----------------
HybridBinaryClassifier    nn.Module: Linear encoder → QuantumEncodingLayer → Linear head.
"""

from hqnn_forge.models.hybrid_classifier import HybridBinaryClassifier

__all__: list[str] = [
    "HybridBinaryClassifier",
]
