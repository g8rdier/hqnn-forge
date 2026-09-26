"""
tests/test_noise_aware_training.py
==================================
Training-time depolarizing noise (``noise_level`` on the encoding layers and
classifiers): the noiseless default is untouched, train mode runs the noisy
circuit with a known effect, eval mode is noiseless, gradients flow through
the noise, and the post-hoc wrapper takes precedence.
"""

from __future__ import annotations

import math

import pytest
import torch

from hqnn_forge.diagnostics import gradient_variance
from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier
from hqnn_forge.noise import (
    apply_depolarizing_noise,
    noise_sweep,
    run_with_training_noise,
    training_noise_qnode,
)

N_QUBITS = 3
CPU = {"device_name": "default.qubit", "diff_method": "backprop"}
LAYERS = [QuantumEncodingLayer, IQPEncodingLayer]
MODELS = [HybridBinaryClassifier, ParallelHybridClassifier]


def _layer(cls: type = QuantumEncodingLayer, **kwargs: object) -> torch.nn.Module:
    torch.manual_seed(0)
    return cls(n_qubits=N_QUBITS, n_layers=2, **CPU, **kwargs)


def _pair(
    cls: type = QuantumEncodingLayer, **noise: object
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """A noisy layer and a noiseless one with identical weights."""
    noisy = _layer(cls, **noise)
    clean = _layer(cls)
    with torch.no_grad():
        clean.qlayer.weights.copy_(noisy.qlayer.weights)
    return noisy, clean


@pytest.fixture
def x() -> torch.Tensor:
    return (torch.rand(6, N_QUBITS, generator=torch.Generator().manual_seed(1)) * 2 - 1) * math.pi


# ---------------------------------------------------------------------------
# noise_level = 0 is the existing path, exactly
# ---------------------------------------------------------------------------


class TestNoiselessDefault:
    @pytest.mark.parametrize("cls", LAYERS)
    def test_zero_noise_is_bit_identical_in_both_modes(self, cls: type, x: torch.Tensor) -> None:
        explicit, default = _pair(cls, noise_level=0.0)
        assert explicit._training_noise_qnode is None
        for mode in (True, False):
            explicit.train(mode)
            default.train(mode)
            out_a, out_b = explicit(x), default(x)
            torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
        explicit.train()
        default.train()
        explicit(x).sum().backward()
        default(x).sum().backward()
        torch.testing.assert_close(
            explicit.qlayer.weights.grad, default.qlayer.weights.grad, rtol=0, atol=0
        )

    @pytest.mark.parametrize("cls", MODELS)
    def test_classifier_default_is_noiseless(self, cls: type) -> None:
        torch.manual_seed(0)
        model = cls(n_input_features=4, n_qubits=N_QUBITS, n_layers=1, **CPU)
        assert model.quantum_layer.noise_level == 0.0
        assert model.quantum_layer._training_noise_qnode is None


# ---------------------------------------------------------------------------
# Effect of the noise in train mode
# ---------------------------------------------------------------------------


class TestTrainingNoise:
    @pytest.mark.parametrize("cls", LAYERS)
    @pytest.mark.parametrize("p", [0.05, 0.3])
    def test_end_noise_damps_train_output_by_one_minus_four_thirds_p(
        self, cls: type, p: float, x: torch.Tensor
    ) -> None:
        noisy, clean = _pair(cls, noise_level=p, noise_position="end")
        noisy.train()
        with torch.no_grad():
            torch.testing.assert_close(noisy(x), (1 - 4 * p / 3) * clean(x), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("cls", LAYERS)
    def test_eval_mode_is_noiseless(self, cls: type, x: torch.Tensor) -> None:
        noisy, clean = _pair(cls, noise_level=0.3)
        noisy.eval()
        with torch.no_grad():
            torch.testing.assert_close(noisy(x), clean(x), rtol=0, atol=0)
        assert noisy.qlayer.qnode is not noisy._training_noise_qnode

    def test_gate_noise_changes_the_train_output(self, x: torch.Tensor) -> None:
        noisy, clean = _pair(noise_level=0.1)
        noisy.train()
        with torch.no_grad():
            assert not torch.allclose(noisy(x), clean(x), atol=1e-3)

    def test_gradients_flow_through_the_noisy_circuit(self, x: torch.Tensor) -> None:
        noisy, clean = _pair(noise_level=0.2, noise_position="end")
        noisy.train()
        clean.train()
        noisy(x).sum().backward()
        clean(x).sum().backward()
        assert noisy.qlayer.weights.grad is not None
        # End-position noise scales every ⟨Z⟩ by a constant, hence the gradient too.
        torch.testing.assert_close(
            noisy.qlayer.weights.grad,
            (1 - 4 * 0.2 / 3) * clean.qlayer.weights.grad,
            rtol=1e-4,
            atol=1e-6,
        )

    def test_qnode_is_restored_after_the_forward_pass(self, x: torch.Tensor) -> None:
        noisy = _layer(noise_level=0.1)
        original = noisy.qlayer.qnode
        noisy.train()
        noisy(x)
        assert noisy.qlayer.qnode is original

    def test_qnode_is_restored_when_the_forward_pass_raises(self) -> None:
        noisy = _layer(noise_level=0.1)
        original = noisy.qlayer.qnode
        noisy.train()
        with pytest.raises(Exception):  # noqa: B017 - any error from the bad input
            run_with_training_noise(
                noisy.qlayer, noisy._training_noise_qnode, torch.zeros(2, N_QUBITS + 1)
            )
        assert noisy.qlayer.qnode is original

    @pytest.mark.parametrize("cls", LAYERS)
    def test_extra_repr_mentions_the_noise(self, cls: type) -> None:
        assert "noise_level=0.1" in _layer(cls, noise_level=0.1).extra_repr()
        assert "noise_level" not in _layer(cls).extra_repr()


# ---------------------------------------------------------------------------
# Interaction with the post-hoc wrapper and the classifiers
# ---------------------------------------------------------------------------


class TestInteractions:
    def test_post_hoc_wrapper_wins_in_train_mode(self, x: torch.Tensor) -> None:
        """Inside apply_depolarizing_noise the sweep's channel is used, not the training one."""
        noisy, clean = _pair(noise_level=0.3, noise_position="end")
        noisy.train()
        with torch.no_grad(), apply_depolarizing_noise(noisy, 0.6, position="end"):
            torch.testing.assert_close(
                noisy(x), (1 - 4 * 0.6 / 3) * clean(x), rtol=1e-5, atol=1e-6
            )

    def test_zero_noise_wrapper_is_noiseless_in_train_mode(self, x: torch.Tensor) -> None:
        """p = 0 counts as the wrapper too: it suppresses the training channel."""
        noisy, clean = _pair(noise_level=0.3, noise_position="end")
        noisy.train()
        with torch.no_grad():
            with apply_depolarizing_noise(noisy, 0.0):
                torch.testing.assert_close(noisy(x), clean(x), rtol=0, atol=0)
            assert noisy.qlayer._hqnn_noise_original is None
            # Outside the block the training channel is back.
            torch.testing.assert_close(noisy(x), 0.6 * clean(x), rtol=1e-5, atol=1e-6)

    def test_train_mode_sweep_over_a_noisy_layer_is_monotone(self, x: torch.Tensor) -> None:
        noisy, clean = _pair(noise_level=0.3, noise_position="end")
        noisy.train()
        reference = clean(x).detach()
        for p in (0.0, 0.1, 0.4):
            with torch.no_grad(), apply_depolarizing_noise(noisy, p, position="end"):
                torch.testing.assert_close(
                    noisy(x), (1 - 4 * p / 3) * reference, rtol=1e-5, atol=1e-6
                )

    def test_zero_noise_wrapper_inside_an_open_block_keeps_that_block(
        self, x: torch.Tensor
    ) -> None:
        noisy, clean = _pair()
        with torch.no_grad(), apply_depolarizing_noise(noisy, 0.3, position="end"):
            with apply_depolarizing_noise(noisy, 0.0):
                torch.testing.assert_close(noisy(x), 0.6 * clean(x), rtol=1e-5, atol=1e-6)
            # The inner p = 0 block must not have disarmed the outer one.
            assert noisy.qlayer._hqnn_noise_original is not None
        assert noisy.qlayer._hqnn_noise_original is None

    @pytest.mark.parametrize("cls", LAYERS)
    def test_gradient_variance_measures_the_noiseless_circuit(self, cls: type) -> None:
        noisy, clean = _pair(cls, noise_level=0.3)
        noisy.train()
        a = gradient_variance(noisy, n_samples=4, generator=torch.Generator().manual_seed(0))
        b = gradient_variance(clean, n_samples=4, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(a.per_parameter, b.per_parameter, rtol=0, atol=0)
        assert noisy.training

    @pytest.mark.parametrize("cls", MODELS)
    def test_classifier_trains_and_sweeps(self, cls: type) -> None:
        torch.manual_seed(0)
        model = cls(n_input_features=4, n_qubits=N_QUBITS, n_layers=1, noise_level=0.1, **CPU)
        assert model.quantum_layer.noise_level == 0.1
        X = torch.randn(6, 4)
        y = torch.randint(0, 2, (6,)).float()
        model.train()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X).squeeze(-1), y)
        loss.backward()
        assert model.quantum_layer.qlayer.weights.grad is not None
        # predict_proba runs in eval mode: noiseless, so the sweep's p = 0 point
        # equals the plain prediction and larger p change it.
        points = noise_sweep(model, X, [0.0, 0.3])
        torch.testing.assert_close(points[0].probabilities, model.predict_proba(X), rtol=0, atol=0)
        assert not torch.allclose(points[1].probabilities, points[0].probabilities, atol=1e-4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("p", [-0.1, 0.8])
    def test_noise_level_range(self, p: float) -> None:
        with pytest.raises(ValueError, match="p must lie"):
            _layer(noise_level=p)

    def test_noise_position(self) -> None:
        with pytest.raises(ValueError, match="position"):
            _layer(noise_level=0.1, noise_position="middle")

    def test_classifier_validates_too(self) -> None:
        with pytest.raises(ValueError, match="p must lie"):
            HybridBinaryClassifier(n_input_features=4, n_qubits=N_QUBITS, noise_level=1.0, **CPU)

    def test_training_noise_qnode_rejects_zero(self) -> None:
        layer = _layer()
        with pytest.raises(ValueError, match="p > 0"):
            training_noise_qnode(layer.qlayer.qnode, N_QUBITS, 0.0)
