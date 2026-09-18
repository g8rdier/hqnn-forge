"""
tests/test_brickwork_entangler.py
=================================
The ``"brickwork"`` entangler (#161): its CNOT pattern, the local light cone
that motivates it (a single-qubit readout depends on its own feature after
one layer, which the cascaded ring cannot do, #150), and the slower decay
of gradient variance with qubit count compared with the ring (#122).
"""

from __future__ import annotations

import math

import pennylane as qml
import pytest
import torch

from hqnn_forge.diagnostics import gradient_variance
from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier

CPU = {"device_name": "default.qubit", "diff_method": "backprop"}


def _layer(n_qubits: int, n_layers: int, entangler: str) -> QuantumEncodingLayer:
    torch.manual_seed(0)
    return QuantumEncodingLayer(
        n_qubits=n_qubits,
        n_layers=n_layers,
        entangler=entangler,  # type: ignore[arg-type]
        **CPU,  # type: ignore[arg-type]
    )


def _cnot_pairs(layer: torch.nn.Module) -> list[list[int]]:
    tape = qml.workflow.construct_tape(layer.qlayer.qnode, level=0)(
        torch.zeros(layer.n_qubits), torch.zeros(layer.n_layers, layer.n_qubits, 3)
    )
    return [op.wires.tolist() for op in tape.operations if op.name == "CNOT"]


def _first_harmonic_of_z0_in_x0(layer: torch.nn.Module, n_points: int = 16) -> float:
    """|c_1| of ⟨Z_0⟩ as a function of feature 0, other features fixed, random weights."""
    torch.manual_seed(0)
    with torch.no_grad():
        layer.qlayer.weights.uniform_(0, 2 * math.pi)
    torch.manual_seed(1)
    fixed = torch.rand(1, layer.n_qubits) * 2 * math.pi - math.pi
    grid = torch.arange(n_points, dtype=torch.float32) * 2 * math.pi / n_points - math.pi
    x = fixed.repeat(n_points, 1)
    x[:, 0] = grid
    with torch.no_grad():
        values = layer(x)[:, 0].to(torch.float64)
    return float(torch.fft.rfft(values).abs()[1] / n_points)


class TestStructure:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_even_then_odd_nearest_neighbour_pairs_without_wraparound(self, n_qubits: int) -> None:
        pairs = _cnot_pairs(_layer(n_qubits, 1, "brickwork"))
        even = [[q, q + 1] for q in range(0, n_qubits - 1, 2)]
        odd = [[q, q + 1] for q in range(1, n_qubits - 1, 2)]
        assert pairs == even + odd
        assert len(pairs) == n_qubits - 1
        assert [n_qubits - 1, 0] not in pairs

    def test_cnots_precede_rot_in_every_layer(self) -> None:
        layer = _layer(4, 2, "brickwork")
        tape = qml.workflow.construct_tape(layer.qlayer.qnode, level=0)(
            torch.zeros(4), torch.zeros(2, 4, 3)
        )
        names = [op.name for op in tape.operations]
        assert names == ["AngleEmbedding"] + (["CNOT"] * 3 + ["Rot"] * 4) * 2

    def test_same_parameter_count_as_ring(self) -> None:
        assert (
            _layer(5, 3, "brickwork").qlayer.weights.shape
            == _layer(5, 3, "ring").qlayer.weights.shape
            == (3, 5, 3)
        )

    def test_iqp_layer_and_classifier_accept_it(self) -> None:
        iqp = IQPEncodingLayer(n_qubits=3, n_layers=1, entangler="brickwork", **CPU)  # type: ignore[arg-type]
        assert iqp(torch.rand(2, 3)).shape == (2, 3)
        model = HybridBinaryClassifier(
            n_input_features=5,
            n_qubits=3,
            n_layers=1,
            entangler="brickwork",
            **CPU,  # type: ignore[arg-type]
        )
        assert _cnot_pairs(model.quantum_layer) == [[0, 1], [1, 2]]
        model(torch.randn(2, 5)).sum().backward()
        assert model.quantum_layer.qlayer.weights.grad is not None


class TestLocalLightCone:
    """The #150 property: readout 0 depends on feature 0 after one layer."""

    def test_brickwork_readout_sees_its_own_feature(self) -> None:
        assert _first_harmonic_of_z0_in_x0(_layer(4, 1, "brickwork")) > 1e-3

    @pytest.mark.parametrize("entangler", ["ring", "strongly_entangling"])
    def test_cascades_do_not(self, entangler: str) -> None:
        """Documents why the option exists: both cascades lose wire 0 from Z_0's image."""
        assert _first_harmonic_of_z0_in_x0(_layer(4, 1, entangler)) < 1e-6

    def test_brickwork_z0_light_cone_after_one_layer(self) -> None:
        """
        With the Rot angles at zero, one brickwork layer on 4 qubits is
        CNOT(0,1), CNOT(2,3), CNOT(1,2).  Z_0 is untouched by all three (wire
        0 is only ever a control), so ⟨Z_0⟩ = cos(x_0): it depends on x_0 and
        on nothing else, the smallest possible light cone.
        """
        layer = _layer(4, 1, "brickwork")
        with torch.no_grad():
            layer.qlayer.weights.zero_()
        torch.manual_seed(2)
        x = torch.rand(6, 4) * 2 * math.pi - math.pi
        with torch.no_grad():
            torch.testing.assert_close(layer(x)[:, 0], torch.cos(x[:, 0]), rtol=1e-5, atol=1e-6)
