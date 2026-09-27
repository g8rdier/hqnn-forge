"""
tests/test_tape_resources.py
============================
Gate counts and depth computed from the tape (#344), against PennyLane's own
``specs`` on the installed version.

``circuit_summary`` used to read ``tape.specs["resources"]``, whose fields
PennyLane 0.46 renames or drops (``num_gates``, ``gate_types``,
``gate_sizes``).  It now counts itself; these tests check that the counts are
the ones ``specs`` reported, using whichever spelling the installed PennyLane
has, so they also guard the next rename.
"""

from __future__ import annotations

from typing import Any

import pennylane as qml
import pytest

from hqnn_forge.diagnostics import circuit_summary
from hqnn_forge.diagnostics.circuit import _logical_tape, tape_resources
from hqnn_forge.encoding import DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}

LAYERS = [
    pytest.param(lambda: QuantumEncodingLayer(n_qubits=4, n_layers=2, **CPU), id="angle"),
    pytest.param(
        lambda: QuantumEncodingLayer(
            n_qubits=5, n_layers=3, entangler="strongly_entangling", readout="first", **CPU
        ),
        id="angle-strongly-first",
    ),
    pytest.param(lambda: IQPEncodingLayer(n_qubits=4, n_layers=2, n_repeats=2, **CPU), id="iqp"),
    # No AmplitudeEncodingLayer: on main, circuit_summary feeds every layer an
    # n_qubits-wide zero input, which the amplitude layer rejects (#218, fixed
    # by #279).
    pytest.param(
        lambda: DataReuploadingLayer(n_qubits=3, n_layers=3, trainable_input_scaling=True, **CPU),
        id="reuploading",
    ),
]


def _specs(tape: qml.tape.QuantumScript) -> dict[str, Any]:
    """PennyLane's resources under either spelling: 0.45 or 0.46 and later."""
    res = tape.specs["resources"]
    counts = getattr(res, "gate_types", None) or res.counts
    total = getattr(res, "num_gates", None)
    if total is None:
        total = res.total_quantum_operations
    return {"depth": res.depth, "n_gates": total, "gate_counts": dict(counts)}


@pytest.mark.parametrize("build", LAYERS)
def test_counts_match_pennylane_specs(build: Any) -> None:
    layer = build()
    tape = _logical_tape(layer.qlayer, layer.n_qubits)
    ours = tape_resources(tape)
    theirs = _specs(tape)
    assert ours.depth == theirs["depth"]
    assert ours.n_gates == theirs["n_gates"]
    assert ours.gate_counts == dict(sorted(theirs["gate_counts"].items()))
    gate_sizes = getattr(tape.specs["resources"], "gate_sizes", None)
    if gate_sizes is not None:  # dropped in PennyLane 0.46
        assert ours.n_two_qubit_gates == sum(c for size, c in gate_sizes.items() if size >= 2)


@pytest.mark.parametrize("build", LAYERS)
def test_the_summary_is_internally_consistent(build: Any) -> None:
    summary = circuit_summary(build())
    assert summary.n_gates == sum(summary.gate_counts.values())
    assert 0 < summary.n_two_qubit_gates <= summary.n_gates
    assert summary.depth <= summary.n_gates


def test_depth_by_hand() -> None:
    # H on 0 | CNOT 0-1 | RZ on 1, RX on 2 in parallel with the CNOT | CZ 1-2:
    # layers H(0),RX(2) -> CNOT(0,1) -> RZ(1) -> CZ(1,2): depth 4.
    ops = [
        qml.Hadamard(0),
        qml.RX(0.1, 2),
        qml.CNOT([0, 1]),
        qml.RZ(0.2, 1),
        qml.CZ([1, 2]),
        qml.Toffoli([0, 1, 2]),
    ]
    tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
    res = tape_resources(tape)
    assert res.depth == 5
    assert res.n_gates == 6
    assert res.n_two_qubit_gates == 3  # CNOT, CZ, and the three-qubit Toffoli once
    assert res.gate_counts == {"CNOT": 1, "CZ": 1, "Hadamard": 1, "RX": 1, "RZ": 1, "Toffoli": 1}


def test_empty_tape() -> None:
    res = tape_resources(qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))]))
    assert res == (0, 0, 0, {})
