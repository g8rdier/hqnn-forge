"""
tests/test_published_shnn_parity.py
====================================
Structural comparison with the SHNN published in the thesis and the
``hqnn-fraud-detection-benchmark`` repository.

What is checked, and how
------------------------
The reference model is rebuilt here from the benchmark's own source
(``src/models/quantum/shnn.py`` and ``vqc.py`` with the ``shnn`` block of
``configs/default.yaml``): ``Linear(8→8)`` + ``π·sigmoid``, ``AngleEmbedding``
with RY, ``StronglyEntanglingLayers`` (2 layers), readout ⟨Z_0⟩,
``Linear(1→1)``.  Its parameter count must equal the published 122, which
validates the rebuild.

Checked structurally (fast, no training):

* ``HybridBinaryClassifier.published_shnn()`` (#131) is the published model:
  122 parameters, identical circuit resources and CNOT pairs, and identical
  logits once the weights are copied across;
* the library's *default* ``HybridBinaryClassifier(8, 8, 2)`` shares qubits,
  layers, quantum parameter count and gate budget with it, and the defaults
  that differ (head width, embedding axis, entangler order and range, depth,
  encoder range) are pinned so a change to them shows up here.

Not checked, and why: the published MCC (0.5758 ± 0.0371) and MCC/kParam
(4.720) come from 5-fold CV on the 284,807-row Kaggle dataset with SMOTE and
100 epochs; reproducing them needs the dataset (not redistributable) and hours
of simulation, so they are out of scope for the test suite.  With the
structure now identical, a full run of ``published_shnn()`` on that data is
the remaining check of the reported numbers.
"""

from __future__ import annotations

import math

import pennylane as qml
import pytest
import torch
from torch import nn

from hqnn_forge.diagnostics import CircuitSummary, LOGICAL_GATE_SET, circuit_summary
from hqnn_forge.models import HybridBinaryClassifier

pytestmark = pytest.mark.reproducibility

N_QUBITS = 8
N_LAYERS = 2
PUBLISHED_PARAMS = 122  # thesis / benchmark README results table
PUBLISHED_QUANTUM_PARAMS = 48  # configs/default.yaml: "48 quantum params → total ~122 params"


class _PiSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.pi * torch.sigmoid(x)


def _published_circuit() -> qml.QNode:
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> qml.measurements.ExpectationMP:
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    return circuit


def _published_shnn() -> tuple[nn.Module, qml.QNode]:
    circuit = _published_circuit()
    vqc = qml.qnn.TorchLayer(circuit, {"weights": (N_LAYERS, N_QUBITS, 3)})
    pre = nn.Sequential(nn.Linear(N_QUBITS, N_QUBITS), _PiSigmoid())
    post = nn.Linear(1, 1)  # followed by a parameter-free Sigmoid
    return nn.ModuleDict({"pre": pre, "vqc": vqc, "post": post}), circuit


def _published_summary(circuit: qml.QNode) -> tuple[CircuitSummary, list[tuple[int, int]]]:
    tape = qml.workflow.construct_tape(circuit, level="top")(
        torch.zeros(N_QUBITS), torch.zeros(N_LAYERS, N_QUBITS, 3)
    )
    (tape,), _ = qml.transforms.decompose(tape, gate_set=LOGICAL_GATE_SET)
    res = tape.specs["resources"]
    summary = CircuitSummary(
        layer_type="published",
        n_qubits=N_QUBITS,
        n_trainable_params=PUBLISHED_QUANTUM_PARAMS,
        depth=int(res.depth),
        n_gates=int(res.num_gates),
        n_two_qubit_gates=sum(c for size, c in res.gate_sizes.items() if size >= 2),
        gate_counts=dict(sorted(res.gate_types.items())),
    )
    pairs = [tuple(op.wires.tolist()) for op in tape.operations if op.name == "CNOT"]
    return summary, pairs


def _ours() -> HybridBinaryClassifier:
    return HybridBinaryClassifier(
        n_input_features=N_QUBITS, n_qubits=N_QUBITS, n_layers=N_LAYERS,
        device_name="default.qubit", diff_method="backprop",
    )


@pytest.fixture(scope="module")
def published() -> tuple[nn.Module, CircuitSummary, list[tuple[int, int]]]:
    model, circuit = _published_shnn()
    summary, pairs = _published_summary(circuit)
    return model, summary, pairs


@pytest.fixture(scope="module")
def ours() -> tuple[HybridBinaryClassifier, CircuitSummary]:
    model = _ours()
    return model, circuit_summary(model)


class TestReferenceRebuild:
    def test_rebuild_has_the_published_parameter_count(self, published: tuple) -> None:
        model, _, _ = published
        assert sum(p.numel() for p in model.parameters()) == PUBLISHED_PARAMS

    def test_rebuild_quantum_parameters(self, published: tuple) -> None:
        model, _, _ = published
        assert sum(p.numel() for p in model["vqc"].parameters()) == PUBLISHED_QUANTUM_PARAMS


class TestPublishedConfigurationParity:
    """
    ``HybridBinaryClassifier.published_shnn()`` must *be* the published model:
    same parameter count, same circuit resources, same CNOT pairs, and the
    same logits when the weights are copied across.
    """

    @pytest.fixture(scope="class")
    def configured(self) -> tuple[HybridBinaryClassifier, CircuitSummary]:
        model = HybridBinaryClassifier.published_shnn(
            device_name="default.qubit", diff_method="backprop"
        )
        return model, circuit_summary(model)

    def test_parameter_counts(self, configured: tuple) -> None:
        model, summary = configured
        assert model.count_parameters() == PUBLISHED_PARAMS
        assert summary.n_trainable_params == PUBLISHED_QUANTUM_PARAMS
        assert sum(p.numel() for p in model.classical_encoder.parameters()) == 8 * 8 + 8
        assert sum(p.numel() for p in model.head.parameters()) == 2

    def test_circuit_resources_are_identical(self, published: tuple, configured: tuple) -> None:
        _, ref, _ = published
        _, summary = configured
        assert summary.n_qubits == ref.n_qubits
        assert summary.depth == ref.depth == 15
        assert summary.n_gates == ref.n_gates == 40
        assert summary.n_two_qubit_gates == ref.n_two_qubit_gates == 16
        assert dict(summary.gate_counts) == dict(ref.gate_counts)
        assert summary.gate_counts["RY"] == 8 and "RX" not in summary.gate_counts

    def test_cnot_pairs_are_identical(self, published: tuple, configured: tuple) -> None:
        _, _, ref_pairs = published
        model, _ = configured
        tape = qml.workflow.construct_tape(model.quantum_layer.qlayer.qnode, level="top")(
            torch.zeros(N_QUBITS), torch.zeros(N_LAYERS, N_QUBITS, 3)
        )
        (tape,), _ = qml.transforms.decompose(tape, gate_set=LOGICAL_GATE_SET)
        pairs = [tuple(op.wires.tolist()) for op in tape.operations if op.name == "CNOT"]
        assert pairs == ref_pairs

    def test_same_logits_with_the_same_weights(self, published: tuple, configured: tuple) -> None:
        """
        End-to-end numerical parity: copy the encoder, quantum and head
        weights from the rebuilt published model and compare logits.
        """
        reference, _, _ = published
        model, _ = configured
        with torch.no_grad():
            model.classical_encoder[0].weight.copy_(reference["pre"][0].weight)
            model.classical_encoder[0].bias.copy_(reference["pre"][0].bias)
            model.quantum_layer.qlayer.weights.copy_(reference["vqc"].weights)
            model.head.weight.copy_(reference["post"].weight)
            model.head.bias.copy_(reference["post"].bias)
        torch.manual_seed(0)
        x = torch.randn(6, N_QUBITS)
        with torch.no_grad():
            ours = model(x).squeeze(-1)
            angles = reference["pre"](x)
            theirs = reference["post"](reference["vqc"](angles).reshape(-1, 1)).squeeze(-1)
        torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-6)

    def test_quantum_init_is_normal_with_std_0_1(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier.published_shnn(
            device_name="default.qubit", diff_method="backprop"
        )
        weights = model.quantum_layer.qlayer.weights.detach()
        assert weights.std().item() == pytest.approx(0.1, rel=0.3)  # 48 draws
        assert abs(weights.mean().item()) < 0.1


class TestDefaultConfigurationIsAVariant:
    """
    The library's default ``HybridBinaryClassifier(8, 8, 2)`` shares the
    qubit count, layer count, quantum parameter count and gate budget with
    the published SHNN but is not that model.  Each assertion documents one
    default that differs, so a change to the defaults shows up here.
    """

    def test_qubits_layers_and_quantum_parameters(self, published: tuple, ours: tuple) -> None:
        _, ref, _ = published
        model, summary = ours
        assert summary.n_qubits == ref.n_qubits == N_QUBITS
        assert model.n_layers == N_LAYERS
        assert summary.n_trainable_params == ref.n_trainable_params == PUBLISHED_QUANTUM_PARAMS

    def test_gate_budget(self, published: tuple, ours: tuple) -> None:
        _, ref, _ = published
        _, summary = ours
        assert summary.n_gates == ref.n_gates == 40
        assert summary.n_two_qubit_gates == ref.n_two_qubit_gates == 16
        assert summary.gate_counts["Rot"] == ref.gate_counts["Rot"] == 16
        assert summary.gate_counts["CNOT"] == ref.gate_counts["CNOT"] == 16

    def test_total_parameters_differ_by_the_head(self, ours: tuple) -> None:
        model, _ = ours
        # Published head: Linear(1→1) = 2.  Default: Linear(8→1) = 9.
        assert model.count_parameters() == PUBLISHED_PARAMS - 2 + 9 == 129
        assert model.head.in_features == N_QUBITS  # readout="all"

    def test_embedding_axis(self, published: tuple, ours: tuple) -> None:
        _, ref, _ = published
        _, summary = ours
        assert ref.gate_counts.get("RY") == 8 and "RX" not in ref.gate_counts
        assert summary.gate_counts.get("RX") == 8 and "RY" not in summary.gate_counts

    def test_entangler_range_and_order(self, published: tuple, ours: tuple) -> None:
        _, ref, pairs = published
        _, summary = ours
        ring = [(i, (i + 1) % N_QUBITS) for i in range(N_QUBITS)]
        ring2 = [(i, (i + 2) % N_QUBITS) for i in range(N_QUBITS)]
        assert pairs == ring + ring2  # StronglyEntanglingLayers: range 1, then range 2
        assert ref.depth == 15 and summary.depth == 19  # Rot-then-CNOT vs CNOT-then-Rot

    def test_encoder_output_range(self, ours: tuple) -> None:
        model, _ = ours
        x = torch.linspace(-50, 50, 8 * 5).reshape(5, 8)
        with torch.no_grad():
            scaled = model.classical_encoder(x) * torch.pi
            published = _PiSigmoid()(x)
        assert scaled.min() < 0 <= published.min()  # (-π, π) vs (0, π)
