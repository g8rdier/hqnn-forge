"""
tests/test_inert_parameters.py
==============================
hqnn_forge.diagnostics.count_inert_parameters: the structural count of
trainable parameters that can never reach a measurement, checked against
autograd (an inert parameter has an exactly zero gradient for every input and
weight draw) and against hand-built tapes with a known answer.
"""

from __future__ import annotations

import math

import pennylane as qml
import pytest
import torch

from hqnn_forge.diagnostics import circuit_summary, count_inert_parameters
from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier

CPU = {"device_name": "default.qubit", "diff_method": "backprop"}


ZERO = 1e-6  # float32 backprop leaves ~1e-8 on dead entries; live ones are ~1e-2


def _zero_gradient_entries(layer: torch.nn.Module, n_draws: int = 4) -> int:
    """
    Weight entries whose gradient is zero (below ``ZERO``) for every one of
    ``n_draws`` random (input, weight) draws: the autograd view of "inert".
    """
    torch.manual_seed(0)
    weights = layer.qlayer.weights
    always_zero = torch.ones_like(weights, dtype=torch.bool)
    for _ in range(n_draws):
        with torch.no_grad():
            weights.uniform_(0, 2 * math.pi)
        x = torch.rand(3, layer.n_qubits) * 2 * math.pi - math.pi
        weights.grad = None
        (layer(x) * torch.rand(3, layer(x).shape[1])).sum().backward()
        always_zero &= weights.grad.abs() < ZERO
    return int(always_zero.sum())


class TestAgainstAutograd:
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_ring_last_layer_omega(self, n_layers: int) -> None:
        """Exactly n_qubits inert parameters: the ω of each last-layer Rot."""
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=n_layers, **CPU)  # type: ignore[arg-type]
        summary = circuit_summary(layer)
        assert summary.n_inert_params == 3
        assert summary.n_effective_params == 9 * n_layers - 3
        # At two or more layers the structural count is the whole story.
        if n_layers >= 2:
            assert _zero_gradient_entries(layer) == 3

    def test_structural_count_is_a_lower_bound_at_one_layer(self) -> None:
        """
        With one layer and RX embedding more entries are dead for the actual
        inputs (the φ of some Rots, #150), which the structural count does
        not claim; autograd finds at least as many zeros.
        """
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, **CPU)  # type: ignore[arg-type]
        assert _zero_gradient_entries(layer) >= circuit_summary(layer).n_inert_params == 3

    def test_iqp_layer(self) -> None:
        layer = IQPEncodingLayer(n_qubits=3, n_layers=2, **CPU)  # type: ignore[arg-type]
        assert circuit_summary(layer).n_inert_params == 3
        assert _zero_gradient_entries(layer) == 3

    @pytest.mark.parametrize("cls", [HybridBinaryClassifier, ParallelHybridClassifier])
    def test_default_eight_qubit_models_carry_eight_dead_weights(self, cls: type) -> None:
        model = cls(n_input_features=8, n_qubits=8, n_layers=2, **CPU)
        summary = circuit_summary(model)
        assert summary.n_trainable_params == 48
        assert summary.n_inert_params == 8
        assert summary.n_effective_params == 40
        assert "inert params     : 8" in str(summary)

    def test_inert_omegas_have_zero_gradient_through_the_classifier(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(n_input_features=5, n_qubits=3, n_layers=2, **CPU)
        x = torch.randn(4, 5)
        model(x).sum().backward()
        grad = model.quantum_layer.qlayer.weights.grad
        assert grad is not None
        assert grad[-1, :, 2].abs().max().item() < ZERO
        assert (grad[-1, :, :2].abs() > ZERO).all()


def _tape(ops_fn, measurements) -> qml.tape.QuantumScript:
    with qml.queuing.AnnotatedQueue() as q:
        ops_fn()
        for m in measurements:
            qml.apply(m)
    return qml.tape.QuantumScript.from_queue(q)


def _p(value: float) -> torch.Tensor:
    return torch.tensor(value, requires_grad=True)


class TestHandBuiltTapes:
    def test_rz_before_z_readout_is_inert_but_rx_is_not(self) -> None:
        tape = _tape(lambda: (qml.RZ(_p(0.3), 0), qml.RX(_p(0.4), 0)), [qml.expval(qml.PauliZ(0))])
        # RZ is applied first, RX after it: the RZ acts on Z content already
        # turned into X/Y by the later RX, so it is live; only a trailing RZ would be inert.
        assert count_inert_parameters(tape) == 0
        tape = _tape(lambda: (qml.RX(_p(0.4), 0), qml.RZ(_p(0.3), 0)), [qml.expval(qml.PauliZ(0))])
        assert count_inert_parameters(tape) == 1

    def test_gate_on_an_unmeasured_wire_is_inert(self) -> None:
        tape = _tape(
            lambda: (qml.RX(_p(0.1), 1), qml.Rot(_p(0.1), _p(0.2), _p(0.3), 1)),
            [qml.expval(qml.PauliZ(0))],
        )
        assert count_inert_parameters(tape) == 4

    def test_cnot_moves_z_content_to_the_control_and_xy_to_the_target(self) -> None:
        # Z_1 is measured; CNOT(0,1) makes it Z_0 Z_1, so an RX on wire 0
        # before the CNOT is live, while a trailing RZ on wire 0 stays inert.
        tape = _tape(
            lambda: (qml.RX(_p(0.1), 0), qml.CNOT([0, 1]), qml.RZ(_p(0.2), 0)),
            [qml.expval(qml.PauliZ(1))],
        )
        assert count_inert_parameters(tape) == 1
        # X/Y content on the control spreads to the target (Y_0 → Y_0 X_1), so
        # an RZ on the target before the CNOT is live once the control has
        # been rotated out of the Z basis after it: ⟨Z_0⟩ then contains
        # sin(b)·⟨Y_0 (cos(a) X_1 + sin(a) Y_1)⟩, which depends on a.
        tape = _tape(
            lambda: (qml.RZ(_p(0.1), 1), qml.CNOT([0, 1]), qml.RX(_p(0.2), 0)),
            [qml.expval(qml.PauliZ(0))],
        )
        assert count_inert_parameters(tape) == 0
        # Without the RX the control keeps Z content only and the RZ is inert.
        tape = _tape(
            lambda: (qml.RZ(_p(0.1), 1), qml.CNOT([0, 1])),
            [qml.expval(qml.PauliZ(0))],
        )
        assert count_inert_parameters(tape) == 1
        tape = _tape(
            lambda: (qml.RZ(_p(0.1), 1), qml.CNOT([0, 1]), qml.RX(_p(0.2), 1)),
            [qml.expval(qml.PauliZ(1))],
        )
        assert count_inert_parameters(tape) == 0  # X_1 → X_1 through the CNOT: the RZ is live

    def test_non_z_measurement_disables_the_shortcut(self) -> None:
        tape = _tape(lambda: (qml.RZ(_p(0.3), 0),), [qml.expval(qml.PauliX(0))])
        assert count_inert_parameters(tape) == 0
        tape = _tape(lambda: (qml.RZ(_p(0.3), 0),), [qml.probs(wires=[0])])
        assert count_inert_parameters(tape) == 0

    def test_non_trainable_parameters_are_not_counted(self) -> None:
        tape = _tape(lambda: (qml.RZ(0.3, 0),), [qml.expval(qml.PauliZ(0))])
        assert count_inert_parameters(tape) == 0
