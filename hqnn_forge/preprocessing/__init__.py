"""
hqnn_forge.preprocessing
========================
Classical pre-processing pipeline — PCA dimensionality reduction and feature
standardisation — implemented in pure NumPy (no scikit-learn runtime dependency).

Exported symbols
----------------
PCANormalizer   Fits/applies PCA + per-feature standardisation; returns torch.Tensor.
"""

from hqnn_forge.preprocessing.pca_normalizer import PCANormalizer

__all__: list[str] = [
    "PCANormalizer",
]
