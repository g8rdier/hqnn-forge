"""
tests/test_preprocessing.py
=============================
Unit tests for hqnn_forge.preprocessing.PCANormalizer.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from hqnn_forge.preprocessing import PCANormalizer

N_SAMPLES = 100
N_FEATURES = 12
N_COMPONENTS = 4

@pytest.fixture
def fitted_pca() -> PCANormalizer:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    pca = PCANormalizer(n_components=N_COMPONENTS, scale_to_pi=True)
    pca.fit(X)
    return pca

@pytest.fixture
def training_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_SAMPLES, N_FEATURES))

class TestFitAttributes:
    def test_is_fitted(self, fitted_pca: PCANormalizer) -> None:
        assert fitted_pca.is_fitted_ is True

    def test_components_shape(self, fitted_pca: PCANormalizer) -> None:
        assert fitted_pca.components_.shape == (N_COMPONENTS, N_FEATURES)

    def test_explained_variance_sorted_descending(self, fitted_pca: PCANormalizer) -> None:
        ev = fitted_pca.explained_variance_
        for i in range(len(ev) - 1):
            assert ev[i] >= ev[i + 1]

class TestTransformOutput:
    def test_output_shape(self, fitted_pca: PCANormalizer, training_data: np.ndarray) -> None:
        result = fitted_pca.transform(training_data)
        assert result.shape == (N_SAMPLES, N_COMPONENTS)

    def test_output_dtype(self, fitted_pca: PCANormalizer, training_data: np.ndarray) -> None:
        result = fitted_pca.transform(training_data)
        assert result.dtype == torch.float32

class TestScaleToPi:
    def test_values_within_pi(self, fitted_pca: PCANormalizer, training_data: np.ndarray) -> None:
        result = fitted_pca.transform(training_data)
        assert result.min().item() >= -math.pi - 1e-6
        assert result.max().item() <= math.pi + 1e-6

class TestExplainedVarianceRatio:
    def test_sums_to_approximately_one(self, fitted_pca: PCANormalizer) -> None:
        ratio_sum = fitted_pca.explained_variance_ratio_.sum()
        assert 0.0 < ratio_sum <= 1.0 + 1e-6

class TestErrors:
    def test_transform_before_fit(self) -> None:
        pca = PCANormalizer(n_components=4)
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.transform(np.zeros((10, 12)))

    def test_too_few_features(self) -> None:
        pca = PCANormalizer(n_components=10)
        X_small = np.random.randn(50, 5)
        with pytest.raises(ValueError, match="n_features"):
            pca.fit(X_small)
