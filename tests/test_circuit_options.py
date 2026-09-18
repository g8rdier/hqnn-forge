"""
tests/test_circuit_options.py
=============================
The circuit options added for the published SHNN configuration (#131):
``entangler``, ``readout`` and ``rotation`` on the encoding layers, and
``embedding_rotation``, ``entangler``, ``readout``, ``encoder_activation``
and ``init_strategy="normal"`` on the classifiers.

The ``strongly_entangling`` block is pinned against a hand-written
Rot-then-CNOT circuit with the template's range rule ``r = ℓ mod (n-1) + 1``,
so the option is checked against the documented topology, not just against
the template it wraps.
"""

from __future__ import annotations

import math

import pennylane as qml
import pytest
import torch
import torch.nn as nn

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.angle_embedding import (
    apply_variational_layers,
    build_encoding_qnode,
    measure_z,
    readout_wires,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier

CPU = {"device_name": "default.qubit", "diff_method": "backprop"}
N_QUBITS = 4
N_LAYERS = 3  # > n-1 so the range rule wraps: ranges 1, 2, 3 for 4 qubits
BATCH = 5


def _angles(n: int = BATCH) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(n, N_QUBITS) * 2 * math.pi - math.pi


# ---------------------------------------------------------------------------
# Encoding layers
# ---------------------------------------------------------------------------


class TestEntangler:
    def test_strongly_entangling_matches_hand_written_rot_then_cnot(self) -> None:
        torch.manual_seed(0)
        weights = torch.randn(N_LAYERS, N_QUBITS, 3, dtype=torch.float64)
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev, interface="torch")
        def reference(x: torch.Tensor) -> list:
            qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="X")
            for layer in range(N_LAYERS):
                for q in range(N_QUBITS):
                    qml.Rot(
                        weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q
                    )
                r = layer % (N_QUBITS - 1) + 1
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + r) % N_QUBITS])
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        ours = build_encoding_qnode(
            n_qubits=N_QUBITS, n_layers=N_LAYERS, entangler="strongly_entangling", **CPU
        )
        for x in _angles(3).double():
            with torch.no_grad():
                torch.testing.assert_close(
                    torch.stack(ours(x, weights)), torch.stack(reference(x)), rtol=0, atol=1e-12
                )

    def test_ring_is_unchanged(self) -> None:
        """The default entangler is still CNOT ring then Rot, as documented."""
        qnode = build_encoding_qnode(n_qubits=N_QUBITS, n_layers=1, **CPU)
        tape = qml.workflow.construct_tape(qnode, level=0)(
            torch.zeros(N_QUBITS), torch.zeros(1, N_QUBITS, 3)
        )
        names = [op.name for op in tape.operations]
        assert names == ["AngleEmbedding"] + ["CNOT"] * N_QUBITS + ["Rot"] * N_QUBITS

    def test_two_entanglers_differ_but_share_the_parameter_count(self) -> None:
        torch.manual_seed(0)
        ring = QuantumEncodingLayer(n_qubits=N_QUBITS, n_layers=2, **CPU)
        sel = QuantumEncodingLayer(
            n_qubits=N_QUBITS, n_layers=2, entangler="strongly_entangling", **CPU
        )
        with torch.no_grad():
            sel.qlayer.weights.copy_(ring.qlayer.weights)
        assert ring.qlayer.weights.shape == sel.qlayer.weights.shape
        x = _angles()
        with torch.no_grad():
            assert not torch.allclose(ring(x), sel(x), atol=1e-3)
        assert "entangler='strongly_entangling'" in sel.extra_repr()
        assert "entangler" not in ring.extra_repr()

    def test_gradients_flow_through_the_template(self) -> None:
        layer = QuantumEncodingLayer(
            n_qubits=N_QUBITS, n_layers=2, entangler="strongly_entangling", **CPU
        )
        x = _angles().requires_grad_(True)
        layer(x).sum().backward()
        assert layer.qlayer.weights.grad.abs().sum().item() > 0
        assert x.grad.abs().sum().item() > 0

    def test_unknown_entangler_raises(self) -> None:
        with pytest.raises(ValueError, match="entangler"):
            QuantumEncodingLayer(n_qubits=N_QUBITS, n_layers=1, entangler="ladder", **CPU)
        with pytest.raises(ValueError, match="entangler"):
            apply_variational_layers(torch.zeros(1, 2, 3), 2, 1, "ladder")  # type: ignore[arg-type]


class TestReadout:
    @pytest.mark.parametrize("cls", [QuantumEncodingLayer, IQPEncodingLayer])
    def test_first_is_column_zero_of_all(self, cls: type) -> None:
        torch.manual_seed(0)
        every = cls(n_qubits=N_QUBITS, n_layers=2, **CPU)
        first = cls(n_qubits=N_QUBITS, n_layers=2, readout="first", **CPU)
        with torch.no_grad():
            first.qlayer.weights.copy_(every.qlayer.weights)
        x = _angles()
        with torch.no_grad():
            out_first = first(x)
            out_all = every(x)
        assert out_first.shape == (BATCH, 1)
        assert first.n_outputs == 1 and every.n_outputs == N_QUBITS
        torch.testing.assert_close(out_first[:, 0], out_all[:, 0], rtol=0, atol=1e-6)

    def test_readout_wires(self) -> None:
        assert readout_wires(4, "all") == [0, 1, 2, 3]
        assert readout_wires(4, "first") == [0]
        with pytest.raises(ValueError, match="readout"):
            readout_wires(4, "last")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="readout"):
            QuantumEncodingLayer(n_qubits=N_QUBITS, n_layers=1, readout="last", **CPU)

    def test_measure_z_returns_one_expectation_per_wire(self) -> None:
        with qml.queuing.AnnotatedQueue() as q:
            measurements = measure_z(3, "all")
        assert len(measurements) == 3 and len(q.queue) == 3
        assert [m.wires.tolist() for m in measurements] == [[0], [1], [2]]


class TestRotationPassThrough:
    def test_rotation_y_changes_the_embedding_gate(self) -> None:
        qnode = build_encoding_qnode(n_qubits=N_QUBITS, n_layers=1, rotation="Y", **CPU)
        tape = qml.workflow.construct_tape(qnode, level=0)(
            torch.zeros(N_QUBITS), torch.zeros(1, N_QUBITS, 3)
        )
        embedding = next(op for op in tape.operations if op.name == "AngleEmbedding")
        assert embedding.hyperparameters["rotation"] is qml.RY


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


CLASSIFIERS = [
    pytest.param(HybridBinaryClassifier, id="serial"),
    pytest.param(ParallelHybridClassifier, id="parallel"),
]


def _classifier(cls: type, **kwargs: object) -> nn.Module:
    torch.manual_seed(0)
    return cls(n_input_features=6, n_qubits=N_QUBITS, n_layers=2, **CPU, **kwargs)


class TestClassifierOptions:
    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_defaults_are_unchanged(self, cls: type) -> None:
        model = _classifier(cls)
        assert model.quantum_layer.entangler == "ring"
        assert model.quantum_layer.readout == "all"
        assert model.encoder_activation == "tanh"
        assert isinstance(model.classical_encoder[1], nn.Tanh)
        expected_in = N_QUBITS if cls is HybridBinaryClassifier else 16 + N_QUBITS
        assert model.head.in_features == expected_in

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_readout_first_narrows_the_head(self, cls: type) -> None:
        model = _classifier(cls, readout="first")
        expected_in = 1 if cls is HybridBinaryClassifier else 16 + 1
        assert model.head.in_features == expected_in
        assert model(torch.randn(BATCH, 6)).shape == (BATCH, 1)
        assert model.predict_proba(torch.randn(BATCH, 6)).shape == (BATCH,)

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_sigmoid_encoder_maps_into_zero_to_pi(self, cls: type) -> None:
        model = _classifier(cls, encoder_activation="sigmoid")
        assert isinstance(model.classical_encoder[1], nn.Sigmoid)
        x = torch.linspace(-50, 50, 6 * 10).reshape(10, 6)
        with torch.no_grad():
            angles = model.classical_encoder(x) * torch.pi
        assert 0.0 <= angles.min().item()
        assert angles.max().item() <= math.pi + 1e-6  # float32 π rounds up
        # And the quantum layer receives exactly those angles.
        seen: list[torch.Tensor] = []
        handle = model.quantum_layer.register_forward_pre_hook(lambda _m, inp: seen.append(inp[0]))
        with torch.no_grad():
            model(x)
        handle.remove()
        torch.testing.assert_close(seen[0], angles)

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_embedding_rotation_and_entangler_reach_the_quantum_layer(self, cls: type) -> None:
        model = _classifier(cls, embedding_rotation="Y", entangler="strongly_entangling")
        tape = qml.workflow.construct_tape(model.quantum_layer.qlayer.qnode, level=0)(
            torch.zeros(N_QUBITS), torch.zeros(2, N_QUBITS, 3)
        )
        names = [op.name for op in tape.operations]
        assert names == ["AngleEmbedding", "StronglyEntanglingLayers"]
        assert tape.operations[0].hyperparameters["rotation"] is qml.RY

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_normal_init(self, cls: type) -> None:
        torch.manual_seed(0)
        model = cls(
            n_input_features=16,
            n_qubits=16,
            n_layers=16,
            init_strategy="normal",
            init_std=0.05,
            **CPU,
        )
        weights = model.quantum_layer.qlayer.weights.detach()
        assert weights.std().item() == pytest.approx(0.05, rel=0.1)
        assert abs(weights.mean().item()) < 0.01

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_options_train(self, cls: type) -> None:
        model = _classifier(
            cls,
            embedding_rotation="Y",
            entangler="strongly_entangling",
            readout="first",
            encoder_activation="sigmoid",
            init_strategy="normal",
        )
        x = torch.randn(BATCH, 6)
        model(x).sum().backward()
        assert model.quantum_layer.qlayer.weights.grad.abs().sum().item() > 0
        assert model.classical_encoder[0].weight.grad.abs().sum().item() > 0
        assert model.head.weight.grad.abs().sum().item() > 0

    def test_iqp_rejects_embedding_rotation(self) -> None:
        with pytest.raises(ValueError, match="embedding_rotation"):
            _classifier(HybridBinaryClassifier, encoding_type="iqp", embedding_rotation="Y")

    def test_iqp_accepts_entangler_and_readout(self) -> None:
        model = _classifier(
            HybridBinaryClassifier,
            encoding_type="iqp",
            entangler="strongly_entangling",
            readout="first",
        )
        assert model.head.in_features == 1
        assert model(torch.randn(BATCH, 6)).shape == (BATCH, 1)

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_validation(self, cls: type) -> None:
        with pytest.raises(ValueError, match="encoder_activation"):
            _classifier(cls, encoder_activation="relu")
        with pytest.raises(ValueError, match="init_strategy"):
            _classifier(cls, init_strategy="xavier")


class TestPublishedPreset:
    def test_serial_preset_has_122_parameters(self) -> None:
        model = HybridBinaryClassifier.published_shnn(**CPU)
        assert model.count_parameters() == 122
        assert model.quantum_layer.qlayer.weights.numel() == 48
        assert model.head.in_features == 1
        assert model.encoder_activation == "sigmoid"
        assert model.quantum_layer.readout == "first"
        assert model.quantum_layer.entangler == "strongly_entangling"

    def test_overrides_apply(self) -> None:
        model = HybridBinaryClassifier.published_shnn(n_layers=3, **CPU)
        assert model.n_layers == 3
        assert model.count_parameters() == 122 + 8 * 3

    def test_parallel_preset_uses_the_same_quantum_branch(self) -> None:
        model = ParallelHybridClassifier.published_shnn(**CPU)
        assert model.quantum_layer.qlayer.weights.numel() == 48
        assert model.quantum_layer.readout == "first"
        assert model.head.in_features == model.classical_hidden_dim + 1
