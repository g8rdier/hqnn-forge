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

@pytest.fixture
def held_out_data() -> np.ndarray:
    # Different seed and a mean offset, so held-out rows differ from the training mean.
    # Kept small so tanh stays out of saturation and assertions on scaled output bite.
    rng = np.random.default_rng(1)
    return rng.standard_normal((30, N_FEATURES)) + 1.0

class TestTransformUsesFitStatistics:
    """transform must depend only on statistics learned in fit, never on the batch."""

    def test_matches_manual_projection(self, fitted_pca: PCANormalizer, held_out_data: np.ndarray) -> None:
        z = ((held_out_data - fitted_pca.mean_) @ fitted_pca.components_.T) / fitted_pca.std_
        expected = torch.tensor(np.tanh(z) * np.pi, dtype=torch.float32)
        torch.testing.assert_close(fitted_pca.transform(held_out_data), expected)

    def test_single_row_matches_row_in_batch(self, fitted_pca: PCANormalizer, held_out_data: np.ndarray) -> None:
        batch = fitted_pca.transform(held_out_data)
        for i in range(len(held_out_data)):
            torch.testing.assert_close(fitted_pca.transform(held_out_data[i : i + 1])[0], batch[i])

    def test_held_out_batch_independent_of_split(self, fitted_pca: PCANormalizer, held_out_data: np.ndarray) -> None:
        whole = fitted_pca.transform(held_out_data)
        chunks = torch.cat([fitted_pca.transform(held_out_data[s : s + 7]) for s in range(0, 30, 7)])
        torch.testing.assert_close(chunks, whole)

    def test_single_non_mean_row_is_not_all_zeros(self, fitted_pca: PCANormalizer, held_out_data: np.ndarray) -> None:
        assert fitted_pca.transform(held_out_data[:1]).abs().max().item() > 0.1

    def test_train_test_shift_is_preserved(self, training_data: np.ndarray, held_out_data: np.ndarray) -> None:
        pca = PCANormalizer(n_components=N_COMPONENTS, scale_to_pi=False).fit(training_data)
        k = 2.0
        # Shift every row along the first principal axis; components are orthonormal,
        # so only the first standardised column should move, by exactly k / std_[0].
        delta = pca.transform(held_out_data + k * pca.components_[0]) - pca.transform(held_out_data)
        expected = torch.zeros_like(delta)
        expected[:, 0] = k / pca.std_[0]
        torch.testing.assert_close(delta, expected, atol=1e-5, rtol=0.0)

    def test_training_output_is_standardised(self, training_data: np.ndarray) -> None:
        z = PCANormalizer(n_components=N_COMPONENTS, scale_to_pi=False).fit_transform(training_data)
        torch.testing.assert_close(z.mean(dim=0), torch.zeros(N_COMPONENTS), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(z.std(dim=0), torch.ones(N_COMPONENTS), atol=1e-5, rtol=0.0)

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
