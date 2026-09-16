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

* the parts of the library's ``HybridBinaryClassifier(8, 8, 2)`` that match
  the published model -- qubits, layers, quantum parameter count, gate budget;
* the parts that do not, pinned explicitly (total parameters, embedding axis,
  entangler order and range, readout width, depth), so that closing the gap
  (#131) or drifting further both show up here.

Not checked, and why: the published MCC (0.5758 ± 0.0371) and MCC/kParam
(4.720) come from 5-fold CV on the 284,807-row Kaggle dataset with SMOTE and
100 epochs; reproducing them needs the dataset (not redistributable) and hours
of simulation, so they are out of scope for the test suite.  And since the
library's model differs from the published one (above), a numerical match
would not be expected until #131 is resolved.
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


class TestWhatMatches:
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

    def test_classical_encoder_size(self, ours: tuple) -> None:
        model, _ = ours
        assert sum(p.numel() for p in model.classical_encoder.parameters()) == 8 * 8 + 8


class TestKnownDifferences:
    """Each assertion documents one gap tracked in #131."""

    def test_total_parameters_differ_by_the_head(self, ours: tuple) -> None:
        model, _ = ours
        # Published head: Linear(1→1) = 2.  Ours: Linear(8→1) = 9.
        assert model.count_parameters() == PUBLISHED_PARAMS - 2 + 9 == 129

    def test_readout_width(self, ours: tuple) -> None:
        model, _ = ours
        assert model.head.in_features == N_QUBITS  # published reads out ⟨Z_0⟩ only

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
