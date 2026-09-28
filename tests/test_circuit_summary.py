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

import math
from math import comb
from types import MappingProxyType

import pennylane as qml
import pytest
import torch

from hqnn_forge.diagnostics import CircuitSummary, circuit_summary, draw_circuit
from hqnn_forge.diagnostics.circuit import _logical_tape
from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier

CPU = {"device_name": "default.qubit", "diff_method": "backprop"}


def _lightning_available() -> bool:
    try:
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


def _multirz_layer(n_wires: int, n_qubits: int = 4) -> torch.nn.Module:
    """RX embedding, one trainable ``MultiRZ`` on wires 0 … n_wires-1, ⟨Z_0⟩."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):  # type: ignore[no-untyped-def]
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.MultiRZ(weights[0], wires=range(n_wires))
        return [qml.expval(qml.PauliZ(0))]

    class MultiRZLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (1,)})

    return MultiRZLayer()


class TestMultiWireGateCost:
    """
    #170: a gate on k > 2 wires counts at its two-qubit cost.  MultiRZ on k
    wires is a CNOT ladder, 2(k-1) CNOTs around one RZ; on two wires it is a
    ZZ rotation and stays a single two-qubit gate.
    """

    @pytest.mark.parametrize("n_wires", [3, 4])
    def test_wide_multirz_counts_its_cnot_ladder(self, n_wires: int) -> None:
        s = circuit_summary(_multirz_layer(n_wires))
        assert s.n_two_qubit_gates == 2 * (n_wires - 1)
        assert s.gate_counts.get("CNOT") == 2 * (n_wires - 1)
        assert s.gate_counts.get("RZ") == 1
        assert "MultiRZ" not in s.gate_counts

    def test_two_wire_multirz_stays_one_gate(self) -> None:
        s = circuit_summary(_multirz_layer(2))
        assert s.n_two_qubit_gates == 1
        assert s.gate_counts.get("MultiRZ") == 1
        assert "CNOT" not in s.gate_counts

    @pytest.mark.parametrize("n_wires", [2, 3, 4])
    def test_the_decomposition_is_the_same_unitary(self, n_wires: int) -> None:
        """
        What is counted is the circuit that runs.  Compared as full unitaries:
        the layer's own ⟨Z_0⟩ commutes with a diagonal MultiRZ, so its output
        would not notice a dropped or mis-wired ladder.
        """
        layer = _multirz_layer(n_wires)
        x = torch.tensor([0.3, -1.1, 0.7, 2.0], dtype=torch.float64)
        with torch.no_grad():
            layer.qlayer.weights.copy_(torch.tensor([0.9]))
        qnode = layer.qlayer.qnode
        weights = dict(layer.qlayer.qnode_weights.items())
        written = qml.workflow.construct_tape(qnode, level="top")(x, **weights)
        counted = _logical_tape(layer.qlayer, 4, inputs=x)
        wires = list(range(4))
        u_written = qml.matrix(written, wire_order=wires)
        u_counted = qml.matrix(counted, wire_order=wires)
        # The CNOT-ladder decomposition is exact, global phase included.
        assert torch.allclose(torch.as_tensor(u_counted), torch.as_tensor(u_written), atol=1e-6)

    def test_wide_gate_without_decomposition_still_counts(self) -> None:
        """An opaque 3-wire gate survives decomposition; it counts once, not zero."""

        class Opaque(qml.operation.Operation):
            num_wires = 3
            num_params = 0

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):  # type: ignore[no-untyped-def]
            qml.AngleEmbedding(inputs, wires=range(3))
            qml.RY(weights[0], wires=0)
            qml.CNOT(wires=[0, 1])
            Opaque(wires=[0, 1, 2])
            return [qml.expval(qml.PauliZ(0))]

        class OpaqueLayer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_qubits = 3
                self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (1,)})

        with pytest.warns(UserWarning):
            s = circuit_summary(OpaqueLayer())
        assert s.gate_counts.get("Opaque") == 1
        assert s.n_two_qubit_gates == 2

    def test_wide_multirz_does_not_hide_inert_parameters(self) -> None:
        """
        Inert parameters are counted on the circuit as written, where a wide
        MultiRZ is one diagonal gate.  Through its CNOT ladder, ⟨Z_1⟩ would
        reach wire 2 and the RY there would stop counting as inert.
        """
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):  # type: ignore[no-untyped-def]
            qml.AngleEmbedding(inputs, wires=range(3))
            qml.RY(weights[1], wires=2)
            qml.MultiRZ(weights[0], wires=[0, 1, 2])
            return [qml.expval(qml.PauliZ(1))]

        class Layer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_qubits = 3
                self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (2,)})

        layer = Layer()
        torch.manual_seed(0)
        x = torch.rand(3, dtype=torch.float64) * 2 * math.pi
        weights = layer.qlayer.qnode_weights["weights"]
        with torch.no_grad():
            weights.copy_(torch.tensor([0.7, 1.3]))
        layer.qlayer(x).sum().backward()
        assert weights.grad is not None
        assert torch.all(weights.grad.abs() < 1e-12)  # both are inert
        assert circuit_summary(layer).n_inert_params == 2


class TestGraphDecomposition:
    """
    With PennyLane's graph-based decomposition enabled, ``decompose`` refuses
    a call without ``gate_set``; the diagnostics must still work, and count
    the same circuit whenever every gate but templates is in the gate set.
    """

    @pytest.fixture
    def graph_enabled(self):  # type: ignore[no-untyped-def]
        was_enabled = qml.decomposition.enabled_graph()
        qml.decomposition.enable_graph()
        try:
            yield
        finally:
            if not was_enabled:
                qml.decomposition.disable_graph()

    @pytest.mark.parametrize(
        "make",
        [
            lambda: QuantumEncodingLayer(n_qubits=3, n_layers=1, **CPU),
            lambda: IQPEncodingLayer(n_qubits=3, n_layers=1, **CPU),
            lambda: _multirz_layer(3),
        ],
        ids=["angle", "iqp", "multirz3"],
    )
    def test_same_summary_and_drawing(self, make, graph_enabled) -> None:  # type: ignore[no-untyped-def]
        torch.manual_seed(0)
        layer = make()
        with_graph = (circuit_summary(layer), draw_circuit(layer))
        qml.decomposition.disable_graph()
        without_graph = (circuit_summary(layer), draw_circuit(layer))
        assert with_graph == without_graph

    # Without GlobalPhase in its gate set, the graph finds no decomposition
    # for these ops and PennyLane falls back with a DecompositionWarning.
    @pytest.mark.filterwarnings("error::pennylane.exceptions.DecompositionWarning")
    @pytest.mark.parametrize("prepare", ["mottonen", "unitary"])
    def test_global_phase_is_not_counted(self, prepare: str, graph_enabled) -> None:  # type: ignore[no-untyped-def]
        """
        Graph decomposition of state preparation and ``QubitUnitary`` emits
        ``GlobalPhase``, which ``op.decomposition()`` does not; left in, a
        two-wire one counted as a two-qubit gate and added to the depth.
        """
        state = torch.tensor([0.1, 0.5, -0.3, 0.8], dtype=torch.float64)
        state = state / state.norm()
        u = torch.as_tensor(qml.matrix(qml.QFT(wires=[0, 1])), dtype=torch.complex128)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):  # type: ignore[no-untyped-def]
            if prepare == "mottonen":
                qml.MottonenStatePreparation(state, wires=[0, 1])
            else:
                qml.QubitUnitary(u, wires=[0, 1])
            qml.RY(weights[0], wires=0)
            return [qml.expval(qml.PauliZ(0))]

        class Layer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_qubits = 2
                self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (1,)})

        layer = Layer()
        with_graph = circuit_summary(layer)
        qml.decomposition.disable_graph()
        without_graph = circuit_summary(layer)
        assert "GlobalPhase" not in with_graph.gate_counts
        assert with_graph.n_two_qubit_gates == without_graph.n_two_qubit_gates
        if prepare == "mottonen":
            # Same rules for every other gate here, so the same circuit.
            assert with_graph == without_graph


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
        model = model_cls(
            n_input_features=6, n_qubits=3, n_layers=2, encoding_type=encoding_type, **CPU
        )
        assert circuit_summary(model) == circuit_summary(model.quantum_layer)

    def test_published_configuration(self) -> None:
        """The thesis SHNN: 8 qubits, 2 layers, angle encoding."""
        s = circuit_summary(
            HybridBinaryClassifier(n_input_features=8, n_qubits=8, n_layers=2, **CPU)
        )
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
        assert text.splitlines()[0] == (
            "Circuit summary: QuantumEncodingLayer on default.qubit (backprop)"
        )
        expected = [
            ("qubits", 3),
            ("trainable params", 18),
            ("inert params", 3),
            ("depth", 9),
            ("gates", 15),
            ("two-qubit gates", 6),
            ("CNOT", 6),
            ("RX", 3),
            ("Rot", 6),
        ]
        for label, value in expected:
            assert any(
                line.strip().startswith(label) and line.rstrip().endswith(f": {value}")
                for line in text.splitlines()
            ), label

    def test_str_values_are_aligned(self) -> None:
        text = str(circuit_summary(IQPEncodingLayer(n_qubits=3, n_layers=1, **CPU)))
        colons = {line.index(" : ") for line in text.splitlines()[1:]}
        assert len(colons) == 1, text

    def test_to_dict_round_trips(self) -> None:
        s = circuit_summary(IQPEncodingLayer(n_qubits=3, n_layers=1, **CPU))
        d = s.to_dict()
        assert isinstance(d["gate_counts"], dict)
        assert d.pop("n_effective_params") == s.n_effective_params
        assert CircuitSummary(**d) == s

    def test_to_dict_accepts_any_mapping(self) -> None:
        """gate_counts is annotated Mapping, so a non-dict one must convert."""
        s = CircuitSummary(
            layer_type="QuantumEncodingLayer",
            n_qubits=2,
            n_trainable_params=6,
            depth=5,
            n_gates=8,
            n_two_qubit_gates=2,
            gate_counts=MappingProxyType({"CNOT": 2, "RX": 2, "Rot": 2}),
        )
        d = s.to_dict()
        assert type(d["gate_counts"]) is dict
        assert d["gate_counts"] == {"CNOT": 2, "RX": 2, "Rot": 2}

    def test_is_immutable(self) -> None:
        s = circuit_summary(QuantumEncodingLayer(n_qubits=2, n_layers=1, **CPU))
        with pytest.raises(AttributeError):
            s.depth = 0  # type: ignore[misc]

    def test_draw_shows_the_logical_gates(self) -> None:
        drawing = draw_circuit(QuantumEncodingLayer(n_qubits=2, n_layers=1, **CPU))
        assert "RX(0.00)" in drawing and "Rot(" in drawing and "<Z>" in drawing
        assert drawing.count("\n") >= 1  # one line per wire

    def test_draw_uses_the_given_inputs(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=2, n_layers=1, **CPU)
        drawing = draw_circuit(layer, inputs=torch.tensor([0.5, 1.25]))
        assert "RX(0.50)" in drawing and "RX(1.25)" in drawing
        assert "RX(0.00)" not in drawing

    def test_draw_accepts_a_model_and_decimals(self) -> None:
        model = HybridBinaryClassifier(n_input_features=2, n_qubits=2, n_layers=1, **CPU)
        drawing = draw_circuit(model, decimals=1)
        assert "RX(0.0)" in drawing
