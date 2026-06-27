"""
hqnn_forge.encoding
===================
Quantum feature-map modules for projecting classical tabular vectors into
an n-qubit Hilbert space.

Available encoders
------------------
AngleEmbeddingQNode     Raw PennyLane QNode (angle embedding + entangling VQC).
QuantumEncodingLayer    PyTorch nn.Module wrapper (TorchLayer) of the QNode above.
build_encoding_qnode    Factory that wires the QNode to a device and diff method.

See Also
--------
hqnn_forge.encoding.iqp_embedding  IQP-style encoding (placeholder / future work).
"""

from hqnn_forge.encoding.angle_embedding import (
    AngleEmbeddingQNode,
    QuantumEncodingLayer,
    build_encoding_qnode,
)

__all__: list[str] = [
    "AngleEmbeddingQNode",
    "QuantumEncodingLayer",
    "build_encoding_qnode",
]
