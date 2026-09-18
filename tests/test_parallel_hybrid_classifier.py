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
FIXTURE_SEED = 0


@pytest.fixture(scope="module")
def classifier() -> ParallelHybridClassifier:
    """Small ParallelHybridClassifier for unit tests."""
    torch.manual_seed(FIXTURE_SEED)
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
    """Seeded raw-feature batch, shape (BATCH, N_RAW_FEATURES).

    Seeded so the non-zero gradient assertions check the same draw every run.
    """
    generator = torch.Generator().manual_seed(FIXTURE_SEED)
    return torch.randn(BATCH, N_RAW_FEATURES, generator=generator)


class TestForwardShape:
    def test_output_shape(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        out = classifier(random_raw_batch)
        assert out.shape == (BATCH, 1)

    def test_single_sample(self, classifier: ParallelHybridClassifier) -> None:
        x = torch.randn(1, N_RAW_FEATURES)
        out = classifier(x)
        assert out.shape == (1, 1)


class TestPredictProba:
    def test_output_in_zero_one(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert probs.min().item() >= 0.0 - 1e-6
        assert probs.max().item() <= 1.0 + 1e-6

    def test_output_shape(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert probs.shape == (BATCH,)

    def test_no_grad(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert not probs.requires_grad


class TestPredict:
    def test_returns_binary(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        preds = classifier.predict(random_raw_batch)
        unique_vals = torch.unique(preds)
        for v in unique_vals:
            assert v.item() in (0, 1)

    def test_output_shape(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        preds = classifier.predict(random_raw_batch)
        assert preds.shape == (BATCH,)
        assert preds.dtype == torch.long


def _dropout_classifier() -> ParallelHybridClassifier:
    """Classifier with active dropout, in train mode as left by construction."""
    torch.manual_seed(FIXTURE_SEED)
    return ParallelHybridClassifier(
        n_input_features=N_RAW_FEATURES,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        classical_hidden_dim=CLASSICAL_HIDDEN_DIM,
        dropout_p=0.5,
        device_name="default.qubit",
        diff_method="parameter-shift",
    )


class TestInferenceMode:
    """
    ``@torch.no_grad()`` does not disable ``nn.Dropout``.  Mode handling itself
    is tested in ``tests/test_modes.py``; this checks ``predict_proba`` uses it.
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
        assert all(module.training for module in model.modules())


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
            N_RAW_FEATURES * CLASSICAL_HIDDEN_DIM
            + CLASSICAL_HIDDEN_DIM  # Linear 1
            + CLASSICAL_HIDDEN_DIM * CLASSICAL_HIDDEN_DIM
            + CLASSICAL_HIDDEN_DIM  # Linear 2
        )
        encoder = N_RAW_FEATURES * N_QUBITS + N_QUBITS
        quantum = N_LAYERS * N_QUBITS * 3
        head = (CLASSICAL_HIDDEN_DIM + N_QUBITS) * 1 + 1

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

    def test_gradients_reach_classical_encoder(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        """
        The encoder's only path to the loss runs back through the circuit, so
        this also checks that *input* gradients cross the quantum layer — not
        just gradients w.r.t. its weights.
        """
        classifier.zero_grad()
        out = classifier(random_raw_batch)
        out.sum().backward()

        encoder_params = list(classifier.classical_encoder.parameters())
        assert encoder_params, "fixture must build with use_classical_encoder=True"
        for param in encoder_params:
            assert param.grad is not None
            assert torch.any(param.grad != 0)

    def test_gradients_reach_head(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        classifier.zero_grad()
        out = classifier(random_raw_batch)
        out.sum().backward()

        assert classifier.head.weight.grad is not None
        assert torch.any(classifier.head.weight.grad != 0)


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

    def test_bypassed_input_reaches_circuit_unscaled(self) -> None:
        """Bypassed input is already in (-π, π); a second π factor aliases angles mod 2π."""
        model = ParallelHybridClassifier(
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
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        expected = classifier.classical_encoder(random_raw_batch).detach() * torch.pi
        torch.testing.assert_close(_circuit_input(classifier, random_raw_batch), expected)


# Wider/deeper than the shared fixture: each layer holds n_qubits * 3 weights,
# and at 4 qubits the per-layer std estimate is far too noisy to distinguish the
# two strategies without flaking (cf. #21).
#
# The strategies are told apart by the slope of log(std_l) against log(l + 1),
# fitted over all layers: 0 for restricted, -0.5 for block_local.  A first/last
# std ratio uses only two layers and its spread overlaps a fixed threshold (0.3%
# of seeds at 16 qubits / 8 layers).  At 16 qubits / 16 layers the slope has
# sd 0.034 under either strategy, so the +/-0.25 band sits ~7 sd out — no failures
# over 500 000 simulated draws — and, being half the gap between the targets, a
# model that ignores init_strategy cannot pass both tests.  The margin, not
# INIT_SEED, is what keeps these stable when __init__ changes re-roll the RNG.
INIT_N_QUBITS = 16
INIT_N_LAYERS = 16
INIT_SEED = 0
SLOPE_TOLERANCE = 0.25


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


def _log_std_slope(model: ParallelHybridClassifier) -> float:
    """Least-squares slope of log(per-layer std) against log(layer_index + 1)."""
    w = model.quantum_layer.qlayer.weights.data.double()
    y = torch.log(w.flatten(start_dim=1).std(dim=1))
    x = torch.log(torch.arange(1, w.shape[0] + 1, dtype=torch.float64))
    x, y = x - x.mean(), y - y.mean()
    return ((x * y).sum() / (x * x).sum()).item()


class TestInitStrategies:
    def test_restricted_variance_is_flat_across_layers(self) -> None:
        """``restricted`` uses one shared sigma, so log-std has zero slope in depth."""
        slope = _log_std_slope(_build_for_init("restricted"))
        assert abs(slope) < SLOPE_TOLERANCE

    def test_restricted_matches_documented_sigma(self) -> None:
        """sigma = pi / sqrt(n_qubits * n_layers), per the initializer docstring."""
        model = _build_for_init("restricted")
        expected = math.pi / math.sqrt(INIT_N_QUBITS * INIT_N_LAYERS)
        actual = model.quantum_layer.qlayer.weights.data.std().item()
        assert actual == pytest.approx(expected, rel=0.25)

    def test_block_local_variance_decays_with_depth(self) -> None:
        """
        ``block_local`` uses sigma_l = pi / sqrt(n_qubits * (l + 1)), i.e.
        log(sigma_l) = const - 0.5 * log(l + 1): slope -0.5 in depth.
        """
        slope = _log_std_slope(_build_for_init("block_local"))
        assert abs(slope + 0.5) < SLOPE_TOLERANCE

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
            n_input_features=4,
            n_qubits=4,
            n_layers=1,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
            encoding_type="angle",
        )
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 1)

    def test_iqp_encoding(self) -> None:
        model = ParallelHybridClassifier(
            n_input_features=4,
            n_qubits=4,
            n_layers=1,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
            encoding_type="iqp",
        )
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 1)

    def test_invalid_encoding(self) -> None:
        with pytest.raises(ValueError, match="Unsupported encoding_type"):
            ParallelHybridClassifier(
                n_input_features=4, n_qubits=4, n_layers=1, encoding_type="unknown_encoding"
            )
