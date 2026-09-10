"""
tests/test_parallel_hybrid_classifier.py
=========================================
Unit tests for hqnn_forge.models.ParallelHybridClassifier.
"""

from __future__ import annotations

import math

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

    def test_exact_count(self, classifier: ParallelHybridClassifier) -> None:
        """
        Pin the exact parameter count, block by block.  An inequality against
        the serial model passes for almost any MLP width, so it would not catch
        a mis-sized head or a wrongly wired branch — this does.
        """
        branch = (
            N_RAW_FEATURES * CLASSICAL_HIDDEN_DIM + CLASSICAL_HIDDEN_DIM       # Linear 1
            + CLASSICAL_HIDDEN_DIM * CLASSICAL_HIDDEN_DIM + CLASSICAL_HIDDEN_DIM  # Linear 2
        )
        encoder = N_RAW_FEATURES * N_QUBITS + N_QUBITS
        quantum = N_LAYERS * N_QUBITS * 3
        head    = (CLASSICAL_HIDDEN_DIM + N_QUBITS) * 1 + 1

        expected = branch + encoder + quantum + head
        assert expected == 207, "test constants drifted from the documented config"
        assert classifier.count_parameters() == expected

    def test_head_consumes_both_branches(self, classifier: ParallelHybridClassifier) -> None:
        """The head must be wired to the *concatenated* width, not one branch."""
        assert classifier.head.in_features == CLASSICAL_HIDDEN_DIM + N_QUBITS

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


# Wider/deeper than the shared fixture: each layer holds n_qubits * 3 weights,
# and at 4 qubits the per-layer std estimate is far too noisy to distinguish the
# two strategies without flaking (cf. #21).  At 16 qubits / 8 layers the
# first-to-last std ratio separates cleanly: measured over 2000 seeds,
# restricted stays below 1.55 and block_local above 1.73.
INIT_N_QUBITS = 16
INIT_N_LAYERS = 8
INIT_SEED = 0
DECAY_THRESHOLD = 1.5


def _build_for_init(strategy: str) -> ParallelHybridClassifier:
    torch.manual_seed(INIT_SEED)
    return ParallelHybridClassifier(
        n_input_features=INIT_N_QUBITS,
        n_qubits=INIT_N_QUBITS,
        n_layers=INIT_N_LAYERS,
        use_classical_encoder=False,
        device_name="default.qubit",
        diff_method="parameter-shift",
        init_strategy=strategy,
    )


def _per_layer_std(model: ParallelHybridClassifier) -> list[float]:
    w = model.quantum_layer.qlayer.weights.data
    return [w[i].std().item() for i in range(w.shape[0])]


class TestInitStrategies:
    def test_restricted_variance_is_flat_across_layers(self) -> None:
        """``restricted`` uses one shared sigma, so per-layer std must not decay."""
        stds = _per_layer_std(_build_for_init("restricted"))
        assert stds[0] / stds[-1] < DECAY_THRESHOLD

    def test_restricted_matches_documented_sigma(self) -> None:
        """sigma = pi / sqrt(n_qubits * n_layers), per the initializer docstring."""
        model = _build_for_init("restricted")
        expected = math.pi / math.sqrt(INIT_N_QUBITS * INIT_N_LAYERS)
        actual = model.quantum_layer.qlayer.weights.data.std().item()
        assert actual == pytest.approx(expected, rel=0.25)

    def test_block_local_variance_decays_with_depth(self) -> None:
        """``block_local`` uses sigma_l = pi / sqrt(n_qubits * (l + 1)) — decaying."""
        stds = _per_layer_std(_build_for_init("block_local"))
        assert stds[0] > stds[-1]
        assert stds[0] / stds[-1] > DECAY_THRESHOLD

    def test_strategies_produce_different_weights(self) -> None:
        """
        Wiring check: identical seeds, different strategy — the weights must
        differ.  If ``init_strategy`` were silently ignored these would match.
        """
        restricted = _build_for_init("restricted").quantum_layer.qlayer.weights.data
        block_local = _build_for_init("block_local").quantum_layer.qlayer.weights.data
        assert not torch.allclose(restricted, block_local)


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
