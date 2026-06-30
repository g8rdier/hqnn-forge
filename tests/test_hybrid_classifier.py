"""
tests/test_hybrid_classifier.py
================================
Unit tests for hqnn_forge.models.HybridBinaryClassifier.
"""
from __future__ import annotations

import pytest
import torch
from hqnn_forge.models import HybridBinaryClassifier

BATCH = 8
N_QUBITS = 4
N_LAYERS = 2
N_RAW_FEATURES = 12

@pytest.fixture(scope="module")
def classifier() -> HybridBinaryClassifier:
    return HybridBinaryClassifier(
        n_input_features=N_RAW_FEATURES,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        use_classical_encoder=True,
        device_name="default.qubit",
        diff_method="parameter-shift",
        init_strategy="restricted",
    )

@pytest.fixture
def random_raw_batch() -> torch.Tensor:
    return torch.randn(BATCH, N_RAW_FEATURES)


class TestForwardShape:
    def test_output_shape(self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor) -> None:
        assert classifier(random_raw_batch).shape == (BATCH, 1)

class TestPredictProba:
    def test_output_in_zero_one(self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert probs.min().item() >= 0.0 - 1e-6
        assert probs.max().item() <= 1.0 + 1e-6

class TestPredict:
    def test_returns_binary(self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor) -> None:
        preds = classifier.predict(random_raw_batch)
        for v in torch.unique(preds):
            assert v.item() in (0, 1)

class TestEncoderBypass:
    def test_mismatched_dims_raises(self) -> None:
        with pytest.raises(ValueError, match="n_input_features"):
            HybridBinaryClassifier(
                n_input_features=10, n_qubits=4,
                use_classical_encoder=False, device_name="default.qubit", diff_method="parameter-shift",
            )
