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
    """Small HybridBinaryClassifier for unit tests."""
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
    """Random raw-feature batch, shape (BATCH, N_RAW_FEATURES)."""
    return torch.randn(BATCH, N_RAW_FEATURES)


class TestForwardShape:
    def test_output_shape(self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor) -> None:
        out = classifier(random_raw_batch)
        assert out.shape == (BATCH, 1)

    def test_single_sample(self, classifier: HybridBinaryClassifier) -> None:
        x = torch.randn(1, N_RAW_FEATURES)
        out = classifier(x)
        assert out.shape == (1, 1)


class TestPredictProba:
    def test_output_in_zero_one(self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert probs.min().item() >= 0.0 - 1e-6
        assert probs.max().item() <= 1.0 + 1e-6

    def test_output_shape(self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert probs.shape == (BATCH,)

    def test_no_grad(self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert not probs.requires_grad


class TestPredict:
    def test_returns_binary(self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor) -> None:
        preds = classifier.predict(random_raw_batch)
        unique_vals = torch.unique(preds)
        for v in unique_vals:
            assert v.item() in (0, 1)

    def test_output_shape(self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor) -> None:
        preds = classifier.predict(random_raw_batch)
        assert preds.shape == (BATCH,)
        assert preds.dtype == torch.long


def _dropout_classifier() -> HybridBinaryClassifier:
    """Classifier with active dropout, in train mode as left by construction."""
    torch.manual_seed(0)
    return HybridBinaryClassifier(
        n_input_features=N_RAW_FEATURES,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        dropout_p=0.5,
        device_name="default.qubit",
        diff_method="parameter-shift",
    )


class TestInferenceMode:
    """
    ``@torch.no_grad()`` does not disable ``nn.Dropout``, which checks
    ``self.training``.  ``predict_proba`` must switch to eval mode itself and
    put every submodule back in the mode it found it in.
    """

    def test_train_mode_matches_eval_forward(self, random_raw_batch: torch.Tensor) -> None:
        """Repeated calls in train mode give the dropout-free eval probabilities."""
        model = _dropout_classifier()
        model.eval()
        with torch.no_grad():
            expected = torch.sigmoid(model(random_raw_batch)).squeeze(-1)

        model.train()
        for _ in range(2):
            torch.testing.assert_close(model.predict_proba(random_raw_batch), expected)

    def test_restores_train_mode(self, random_raw_batch: torch.Tensor) -> None:
        model = _dropout_classifier()
        model.predict_proba(random_raw_batch)
        assert all(module.training for module in model.modules())

    def test_leaves_eval_mode(self, random_raw_batch: torch.Tensor) -> None:
        model = _dropout_classifier()
        model.eval()
        model.predict_proba(random_raw_batch)
        assert not any(module.training for module in model.modules())

    def test_preserves_mixed_submodule_modes(self, random_raw_batch: torch.Tensor) -> None:
        """A blanket ``self.train(was_training)`` restore would re-enable dropout here."""
        model = _dropout_classifier()
        model.dropout.eval()
        model.predict_proba(random_raw_batch)
        assert model.training
        assert model.quantum_layer.training
        assert not model.dropout.training

    def test_restores_mode_when_forward_raises(self) -> None:
        model = _dropout_classifier()
        with pytest.raises(RuntimeError):
            model.predict_proba(torch.randn(BATCH, N_RAW_FEATURES + 1))
        assert all(module.training for module in model.modules())


class TestParameterCount:
    def test_positive_count(self, classifier: HybridBinaryClassifier) -> None:
        assert classifier.count_parameters() > 0


def _circuit_input(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and return the tensor handed to the quantum layer."""
    captured: list[torch.Tensor] = []
    handle = model.quantum_layer.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    try:
        model(x)
    finally:
        handle.remove()
    return captured[0]


class TestEncoderBypass:
    def test_mismatched_dims_raises(self) -> None:
        with pytest.raises(ValueError, match="n_input_features"):
            HybridBinaryClassifier(
                n_input_features=10,
                n_qubits=4,
                use_classical_encoder=False,
                device_name="default.qubit",
                diff_method="parameter-shift",
            )

    def test_matching_dims_works(self) -> None:
        model = HybridBinaryClassifier(
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

    def test_bypassed_input_reaches_circuit_unscaled(self) -> None:
        """Bypassed input is already in (-π, π); a second π factor aliases angles mod 2π."""
        model = HybridBinaryClassifier(
            n_input_features=4,
            n_qubits=4,
            n_layers=1,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
        )
        x = torch.linspace(-3.0, 3.0, 8).reshape(2, 4)
        torch.testing.assert_close(_circuit_input(model, x), x)

    def test_encoder_output_scaled_by_pi(
        self, classifier: HybridBinaryClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        expected = classifier.classical_encoder(random_raw_batch).detach() * torch.pi
        torch.testing.assert_close(_circuit_input(classifier, random_raw_batch), expected)


class TestInitStrategies:
    def test_restricted_strategy(self) -> None:
        model = HybridBinaryClassifier(
            n_input_features=4, n_qubits=4, n_layers=2,
            use_classical_encoder=False, device_name="default.qubit",
            diff_method="parameter-shift", init_strategy="restricted",
        )
        assert model is not None

    def test_block_local_strategy(self) -> None:
        model = HybridBinaryClassifier(
            n_input_features=4, n_qubits=4, n_layers=2,
            use_classical_encoder=False, device_name="default.qubit",
            diff_method="parameter-shift", init_strategy="block_local",
        )
        assert model is not None


class TestEncodingTypes:
    def test_angle_encoding(self) -> None:
        model = HybridBinaryClassifier(
            n_input_features=4, n_qubits=4, n_layers=1,
            use_classical_encoder=False, device_name="default.qubit",
            diff_method="parameter-shift", encoding_type="angle",
        )
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 1)

    def test_iqp_encoding(self) -> None:
        model = HybridBinaryClassifier(
            n_input_features=4, n_qubits=4, n_layers=1,
            use_classical_encoder=False, device_name="default.qubit",
            diff_method="parameter-shift", encoding_type="iqp",
        )
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 1)

    def test_invalid_encoding(self) -> None:
        with pytest.raises(ValueError, match="Unsupported encoding_type"):
            HybridBinaryClassifier(
                n_input_features=4, n_qubits=4, n_layers=1,
                encoding_type="unknown_encoding"
            )
