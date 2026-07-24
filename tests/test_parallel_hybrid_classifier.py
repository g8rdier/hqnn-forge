"""
tests/test_parallel_hybrid_classifier.py
=========================================
Unit tests for hqnn_forge.models.ParallelHybridClassifier.
"""

from __future__ import annotations

import pytest
import torch

from hqnn_forge.models import ParallelHybridClassifier


BATCH = 8
N_QUBITS = 4
N_LAYERS = 2
N_RAW_FEATURES = 12
CLASSICAL_HIDDEN_DIM = 6


@pytest.fixture(scope="module")
def classifier() -> ParallelHybridClassifier:
    """Small ParallelHybridClassifier for unit tests."""
    return ParallelHybridClassifier(
        n_input_features=N_RAW_FEATURES,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        classical_hidden_dim=CLASSICAL_HIDDEN_DIM,
        use_classical_encoder=True,
        device_name="default.qubit",
        diff_method="parameter-shift",
        init_strategy="restricted",
    )


@pytest.fixture
def random_raw_batch() -> torch.Tensor:
    """Random raw-feature batch, shape (BATCH, N_RAW_FEATURES)."""
    return torch.randn(BATCH, N_RAW_FEATURES)


class TestForwardShape:
    def test_output_shape(self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor) -> None:
        out = classifier(random_raw_batch)
        assert out.shape == (BATCH, 1)

    def test_single_sample(self, classifier: ParallelHybridClassifier) -> None:
        x = torch.randn(1, N_RAW_FEATURES)
        out = classifier(x)
        assert out.shape == (1, 1)


class TestPredictProba:
    def test_output_in_zero_one(self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert probs.min().item() >= 0.0 - 1e-6
        assert probs.max().item() <= 1.0 + 1e-6

    def test_output_shape(self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert probs.shape == (BATCH,)

    def test_no_grad(self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert not probs.requires_grad


class TestPredict:
    def test_returns_binary(self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor) -> None:
        preds = classifier.predict(random_raw_batch)
        unique_vals = torch.unique(preds)
        for v in unique_vals:
            assert v.item() in (0, 1)

    def test_output_shape(self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor) -> None:
        preds = classifier.predict(random_raw_batch)
        assert preds.shape == (BATCH,)
        assert preds.dtype == torch.long


class TestParameterCount:
    def test_positive_count(self, classifier: ParallelHybridClassifier) -> None:
        assert classifier.count_parameters() > 0

    def test_exceeds_serial_classifier(self, classifier: ParallelHybridClassifier) -> None:
        """Parallel topology adds an MLP branch, so it must have strictly more
        parameters than the equivalent serial HybridBinaryClassifier."""
        from hqnn_forge.models import HybridBinaryClassifier

        serial = HybridBinaryClassifier(
            n_input_features=N_RAW_FEATURES,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            use_classical_encoder=True,
            device_name="default.qubit",
            diff_method="parameter-shift",
            init_strategy="restricted",
        )
        assert classifier.count_parameters() > serial.count_parameters()


class TestGradientFlow:
    def test_gradients_reach_classical_branch(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        classifier.zero_grad()
        out = classifier(random_raw_batch)
        out.sum().backward()

        for param in classifier.classical_branch.parameters():
            assert param.grad is not None
            assert torch.any(param.grad != 0)

    def test_gradients_reach_quantum_branch(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        classifier.zero_grad()
        out = classifier(random_raw_batch)
        out.sum().backward()

        quantum_weights = classifier.quantum_layer.qlayer.weights
        assert quantum_weights.grad is not None
        assert torch.any(quantum_weights.grad != 0)

    def test_gradients_reach_head(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        classifier.zero_grad()
        out = classifier(random_raw_batch)
        out.sum().backward()

        assert classifier.head.weight.grad is not None
        assert torch.any(classifier.head.weight.grad != 0)


class TestEncoderBypass:
    def test_mismatched_dims_raises(self) -> None:
        with pytest.raises(ValueError, match="n_input_features"):
            ParallelHybridClassifier(
                n_input_features=10,
                n_qubits=4,
                use_classical_encoder=False,
                device_name="default.qubit",
                diff_method="parameter-shift",
            )

    def test_matching_dims_works(self) -> None:
        model = ParallelHybridClassifier(
            n_input_features=4,
            n_qubits=4,
            n_layers=1,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
        )
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 1)


class TestInitStrategies:
    def test_restricted_strategy(self) -> None:
        model = ParallelHybridClassifier(
            n_input_features=4, n_qubits=4, n_layers=2,
            use_classical_encoder=False, device_name="default.qubit",
            diff_method="parameter-shift", init_strategy="restricted",
        )
        assert model is not None

    def test_block_local_strategy(self) -> None:
        model = ParallelHybridClassifier(
            n_input_features=4, n_qubits=4, n_layers=2,
            use_classical_encoder=False, device_name="default.qubit",
            diff_method="parameter-shift", init_strategy="block_local",
        )
        assert model is not None


class TestEncodingTypes:
    def test_angle_encoding(self) -> None:
        model = ParallelHybridClassifier(
            n_input_features=4, n_qubits=4, n_layers=1,
            use_classical_encoder=False, device_name="default.qubit",
            diff_method="parameter-shift", encoding_type="angle",
        )
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 1)

    def test_iqp_encoding(self) -> None:
        model = ParallelHybridClassifier(
            n_input_features=4, n_qubits=4, n_layers=1,
            use_classical_encoder=False, device_name="default.qubit",
            diff_method="parameter-shift", encoding_type="iqp",
        )
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 1)

    def test_invalid_encoding(self) -> None:
        with pytest.raises(ValueError, match="Unsupported encoding_type"):
            ParallelHybridClassifier(
                n_input_features=4, n_qubits=4, n_layers=1,
                encoding_type="unknown_encoding"
            )
