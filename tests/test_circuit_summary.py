"""
tests/test_circuit_summary.py
==============================
hqnn_forge.diagnostics.circuit_summary against hand-computed gate counts.

Angle encoding, n qubits, L layers:
    RX n  ·  CNOT n·L  ·  Rot n·L        depth = 1 + L·(n + 1)
    (a CNOT ring on n wires has depth n: each gate shares a wire with the last)
IQP encoding, r repeats, then the same L entangling layers:
    H n·r  ·  RZ (n + C(n,2))·r  ·  CNOT 2·C(n,2)·r + n·L  ·  Rot n·L
Trainable parameters are 3·n·L for both.
"""

from __future__ import annotations

from math import comb

import pytest
import torch

from hqnn_forge.diagnostics import CircuitSummary, circuit_summary, draw_circuit
from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier

CPU = dict(device_name="default.qubit", diff_method="backprop")


def _lightning_available() -> bool:
    try:
        import pennylane as qml

        qml.device("lightning.qubit", wires=1)
        return True
    except Exception:  # noqa: BLE001
        return False


class TestAngleEncodingCounts:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_gate_counts_depth_and_params(self, n_qubits: int, n_layers: int) -> None:
        s = circuit_summary(QuantumEncodingLayer(n_qubits=n_qubits, n_layers=n_layers, **CPU))
        assert s.layer_type == "QuantumEncodingLayer"
        assert s.n_qubits == n_qubits
        assert dict(s.gate_counts) == {
            "CNOT": n_qubits * n_layers,
            "RX": n_qubits,
            "Rot": n_qubits * n_layers,
        }
        assert s.n_gates == n_qubits + 2 * n_qubits * n_layers
        assert s.n_two_qubit_gates == n_qubits * n_layers
        assert s.depth == 1 + n_layers * (n_qubits + 1)
        assert s.n_trainable_params == 3 * n_qubits * n_layers

    def test_rotation_axis_is_reported(self) -> None:
        s = circuit_summary(QuantumEncodingLayer(n_qubits=2, n_layers=1, rotation="Y", **CPU))
        assert s.gate_counts["RY"] == 2 and "RX" not in s.gate_counts


class TestIQPEncodingCounts:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    @pytest.mark.parametrize("n_repeats", [1, 2])
    def test_gate_counts_and_params(self, n_qubits: int, n_repeats: int) -> None:
        n_layers = 1
        pairs = comb(n_qubits, 2)
        s = circuit_summary(
            IQPEncodingLayer(n_qubits=n_qubits, n_layers=n_layers, n_repeats=n_repeats, **CPU)
        )
        assert s.layer_type == "IQPEncodingLayer"
        assert dict(s.gate_counts) == {
            "CNOT": 2 * pairs * n_repeats + n_qubits * n_layers,
            "Hadamard": n_qubits * n_repeats,
            "RZ": (n_qubits + pairs) * n_repeats,
            "Rot": n_qubits * n_layers,
        }
        assert s.n_two_qubit_gates == 2 * pairs * n_repeats + n_qubits * n_layers
        assert s.n_gates == sum(s.gate_counts.values())
        assert s.n_trainable_params == 3 * n_qubits * n_layers

    def test_costs_more_two_qubit_gates_than_angle(self) -> None:
        angle = circuit_summary(QuantumEncodingLayer(n_qubits=4, n_layers=2, **CPU))
        iqp = circuit_summary(IQPEncodingLayer(n_qubits=4, n_layers=2, **CPU))
        assert iqp.n_two_qubit_gates > angle.n_two_qubit_gates
        assert iqp.depth > angle.depth


class TestDeviceIndependence:
    @pytest.mark.skipif(not _lightning_available(), reason="pennylane-lightning not installed")
    @pytest.mark.parametrize("layer_cls", [QuantumEncodingLayer, IQPEncodingLayer])
    def test_same_counts_on_lightning_adjoint(self, layer_cls: type) -> None:
        """lightning's adjoint path rewrites Rot as RZ·RY·RZ; the summary must not."""
        cpu = circuit_summary(layer_cls(n_qubits=3, n_layers=2, **CPU))
        lightning = circuit_summary(
            layer_cls(n_qubits=3, n_layers=2, device_name="lightning.qubit", diff_method="adjoint")
        )
        assert lightning.device_name == "lightning.qubit" and lightning.diff_method == "adjoint"
        assert cpu.device_name == "default.qubit" and cpu.diff_method == "backprop"
        for attr in ("depth", "n_gates", "n_two_qubit_gates", "n_trainable_params"):
            assert getattr(lightning, attr) == getattr(cpu, attr), attr
        assert dict(lightning.gate_counts) == dict(cpu.gate_counts)


class TestModelPassthrough:
    @pytest.mark.parametrize("model_cls", [HybridBinaryClassifier, ParallelHybridClassifier])
    @pytest.mark.parametrize("encoding_type", ["angle", "iqp"])
    def test_model_summary_is_its_quantum_layer(self, model_cls: type, encoding_type: str) -> None:
        model = model_cls(n_input_features=6, n_qubits=3, n_layers=2, encoding_type=encoding_type, **CPU)
        assert circuit_summary(model) == circuit_summary(model.quantum_layer)

    def test_published_configuration(self) -> None:
        """The thesis SHNN: 8 qubits, 2 layers, angle encoding."""
        s = circuit_summary(HybridBinaryClassifier(n_input_features=8, n_qubits=8, n_layers=2, **CPU))
        assert s.n_qubits == 8 and s.n_trainable_params == 48
        assert s.gate_counts["CNOT"] == 16 and s.depth == 19

    def test_frozen_weights_are_not_counted(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=2, **CPU)
        layer.qlayer.weights.requires_grad_(False)
        assert circuit_summary(layer).n_trainable_params == 0

    def test_unsupported_target_raises(self) -> None:
        with pytest.raises(TypeError, match="expects an encoding layer.*got Linear"):
            circuit_summary(torch.nn.Linear(3, 1))


class TestPresentation:
    def test_str_lists_every_field(self) -> None:
        s = circuit_summary(QuantumEncodingLayer(n_qubits=3, n_layers=2, **CPU))
        text = str(s)
        assert text.splitlines()[0] == "Circuit summary: QuantumEncodingLayer on default.qubit (backprop)"
        for label, value in [("qubits", 3), ("trainable params", 18), ("depth", 9), ("gates", 15), ("two-qubit gates", 6), ("CNOT", 6), ("RX", 3), ("Rot", 6)]:
            assert any(line.strip().startswith(label) and line.rstrip().endswith(f": {value}") for line in text.splitlines()), label

    def test_to_dict_round_trips(self) -> None:
        s = circuit_summary(IQPEncodingLayer(n_qubits=3, n_layers=1, **CPU))
        d = s.to_dict()
        assert isinstance(d["gate_counts"], dict)
        assert CircuitSummary(**d) == s

    def test_is_immutable(self) -> None:
        s = circuit_summary(QuantumEncodingLayer(n_qubits=2, n_layers=1, **CPU))
        with pytest.raises(AttributeError):
            s.depth = 0  # type: ignore[misc]

    def test_draw_shows_the_logical_gates(self) -> None:
        drawing = draw_circuit(QuantumEncodingLayer(n_qubits=2, n_layers=1, **CPU))
        assert "RX(0.00)" in drawing and "Rot(" in drawing and "<Z>" in drawing
        assert drawing.count("\n") >= 1  # one line per wire

    def test_draw_accepts_a_model_and_decimals(self) -> None:
        model = HybridBinaryClassifier(n_input_features=2, n_qubits=2, n_layers=1, **CPU)
        drawing = draw_circuit(model, decimals=1)
        assert "RX(0.0)" in drawing
