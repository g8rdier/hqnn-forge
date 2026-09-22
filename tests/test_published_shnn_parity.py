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


def _logical_tape(circuit: qml.QNode, **weights: torch.Tensor) -> qml.tape.QuantumScript:
    """The tape ``circuit`` runs for one sample, decomposed to ``LOGICAL_GATE_SET``."""
    tape = qml.workflow.construct_tape(circuit, level="top")(torch.zeros(N_QUBITS), **weights)
    (decomposed,), _ = qml.transforms.decompose(tape, gate_set=LOGICAL_GATE_SET)
    return decomposed


def _cnot_pairs(tape: qml.tape.QuantumScript) -> list[tuple[int, int]]:
    return [tuple(op.wires.tolist()) for op in tape.operations if op.name == "CNOT"]


def _first_entangler_gate(tape: qml.tape.QuantumScript) -> str:
    """Name of the first gate after the ``N_QUBITS`` embedding rotations."""
    return str(tape.operations[N_QUBITS].name)


def _published_summary(
    circuit: qml.QNode, n_quantum_params: int
) -> tuple[CircuitSummary, qml.tape.QuantumScript]:
    tape = _logical_tape(circuit, weights=torch.zeros(N_LAYERS, N_QUBITS, 3))
    res = tape.specs["resources"]
    summary = CircuitSummary(
        layer_type="published",
        n_qubits=N_QUBITS,
        n_trainable_params=n_quantum_params,
        depth=int(res.depth),
        n_gates=int(res.num_gates),
        n_two_qubit_gates=sum(c for size, c in res.gate_sizes.items() if size >= 2),
        gate_counts=dict(sorted(res.gate_types.items())),
    )
    return summary, tape


def _our_tape(model: HybridBinaryClassifier) -> qml.tape.QuantumScript:
    qlayer = model.quantum_layer.qlayer
    weights = {name: param.detach() for name, param in qlayer.qnode_weights.items()}
    return _logical_tape(qlayer.qnode, **weights)


def _ours() -> HybridBinaryClassifier:
    return HybridBinaryClassifier(
        n_input_features=N_QUBITS, n_qubits=N_QUBITS, n_layers=N_LAYERS,
        device_name="default.qubit", diff_method="backprop",
    )


@pytest.fixture(scope="module")
def published() -> tuple[nn.Module, CircuitSummary, qml.tape.QuantumScript]:
    model, circuit = _published_shnn()
    # Counted from the rebuild rather than assumed: ``test_rebuild_quantum_parameters``
    # is what checks this number against the published table.
    n_quantum_params = sum(p.numel() for p in model["vqc"].parameters())
    summary, tape = _published_summary(circuit, n_quantum_params)
    return model, summary, tape


@pytest.fixture(scope="module")
def ours() -> tuple[HybridBinaryClassifier, CircuitSummary, qml.tape.QuantumScript]:
    torch.manual_seed(0)  # the encoder's random init feeds test_encoder_output_range
    model = _ours()
    return model, circuit_summary(model), _our_tape(model)


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
        model, summary, _ = ours
        assert summary.n_qubits == ref.n_qubits == N_QUBITS
        assert model.n_layers == N_LAYERS
        assert summary.n_trainable_params == ref.n_trainable_params == PUBLISHED_QUANTUM_PARAMS

    def test_gate_budget(self, published: tuple, ours: tuple) -> None:
        _, ref, _ = published
        _, summary, _ = ours
        assert summary.n_gates == ref.n_gates == 40
        assert summary.n_two_qubit_gates == ref.n_two_qubit_gates == 16
        assert summary.gate_counts["Rot"] == ref.gate_counts["Rot"] == 16
        assert summary.gate_counts["CNOT"] == ref.gate_counts["CNOT"] == 16

    def test_classical_encoder_size(self, ours: tuple) -> None:
        model, _, _ = ours
        assert sum(p.numel() for p in model.classical_encoder.parameters()) == 8 * 8 + 8


class TestKnownDifferences:
    """Each assertion documents one gap tracked in #131."""

    def test_total_parameters_differ_by_the_head(self, ours: tuple) -> None:
        model, _, _ = ours
        # Published head: Linear(1→1) = 2.  Ours: Linear(8→1) = 9.
        assert model.count_parameters() == PUBLISHED_PARAMS - 2 + 9 == 129

    def test_readout_width(self, ours: tuple) -> None:
        model, _, _ = ours
        assert model.head.in_features == N_QUBITS  # published reads out ⟨Z_0⟩ only

    def test_embedding_axis(self, published: tuple, ours: tuple) -> None:
        _, ref, _ = published
        _, summary, _ = ours
        assert ref.gate_counts.get("RY") == 8 and "RX" not in ref.gate_counts
        assert summary.gate_counts.get("RX") == 8 and "RY" not in summary.gate_counts

    def test_entangler_range(self, published: tuple, ours: tuple) -> None:
        _, ref, ref_tape = published
        _, summary, our_tape = ours
        ring = [(i, (i + 1) % N_QUBITS) for i in range(N_QUBITS)]
        ring2 = [(i, (i + 2) % N_QUBITS) for i in range(N_QUBITS)]
        # StronglyEntanglingLayers uses range l mod (n-1) + 1: range 1, then range 2.
        assert _cnot_pairs(ref_tape) == ring + ring2
        # Ours is a range-1 ring in every layer.
        assert _cnot_pairs(our_tape) == ring + ring
        # The range, not the gate order, is what costs the depth: a range-2 ring on
        # 8 qubits splits into two independent 4-cycles, while a range-1 ring
        # serialises around all 8.  Swapping Rot and CNOT leaves both numbers alone.
        assert ref.depth == 15 and summary.depth == 19

    def test_entangler_order(self, published: tuple, ours: tuple) -> None:
        _, _, ref_tape = published
        _, _, our_tape = ours
        assert _first_entangler_gate(ref_tape) == "Rot"  # Rot, then the CNOT ring
        assert _first_entangler_gate(our_tape) == "CNOT"  # CNOT ring, then Rot

    def test_encoder_output_range(self, ours: tuple) -> None:
        model, _, _ = ours
        x = torch.linspace(-50, 50, 8 * 5).reshape(5, 8)
        with torch.no_grad():
            scaled = model.classical_encoder(x) * torch.pi
            published = _PiSigmoid()(x)
        assert scaled.min() < 0 <= published.min()  # (-π, π) vs (0, π)
