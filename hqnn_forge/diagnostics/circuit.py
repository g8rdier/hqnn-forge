"""
hqnn_forge.diagnostics.circuit
==============================
Depth, gate counts and parameter counts for the library's quantum circuits.

The encoding layers build their QNodes internally, so answering "how many
CNOTs does this configuration use?" otherwise means reading the source.
``circuit_summary`` asks PennyLane instead: it constructs the tape the layer
would execute and reads the resources off it.

Counting convention
-------------------
Resources are counted on the *logical* circuit, i.e. after decomposing
templates (``AngleEmbedding`` → one ``RX`` per qubit) but **before** any
device-specific decomposition.  ``lightning.qubit``'s adjoint path, for
example, rewrites every ``Rot`` as ``RZ·RY·RZ``, which would make the same
model report different counts depending on the simulator it happens to run
on.  The gate set counted against is ``LOGICAL_GATE_SET``; everything is
decomposed until only those gates remain.  Two-qubit gates are counted
separately because they are what NISQ feasibility is usually judged by; a
gate on more than two wires counts once there, not at the number of CNOTs it
would compile to.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import pennylane as qml
import torch
import torch.nn as nn

#: Gate names a circuit is decomposed to before its resources are counted.
#: Every gate the library's circuits emit is in here, so the count is of the
#: circuit as written; a template such as ``AngleEmbedding`` is expanded.
LOGICAL_GATE_SET: frozenset[str] = frozenset(
    {"Hadamard", "RX", "RY", "RZ", "Rot", "PhaseShift", "CNOT", "CZ", "MultiRZ"}
)


@dataclass(frozen=True)
class CircuitSummary:
    """
    Resource summary of one quantum encoding layer's circuit.

    Attributes
    ----------
    layer_type:
        Class name of the summarised layer, e.g. ``"QuantumEncodingLayer"``.
    n_qubits:
        Wires the circuit acts on.
    n_trainable_params:
        Trainable entries in the layer's weight tensors (``requires_grad``).
    depth:
        Longest path of gates through the logical circuit.
    n_gates:
        Total gate count after decomposition to ``LOGICAL_GATE_SET``.
    n_two_qubit_gates:
        Gates acting on two or more wires (CNOT, CZ, MultiRZ).
    gate_counts:
        Count per gate name, sorted by name.  This field is a mapping, so the
        dataclass is frozen for immutability but is **not** hashable.
    device_name:
        PennyLane device the layer's QNode is bound to, after any fallback.
    diff_method:
        Differentiation method the QNode is configured with.
    """

    layer_type: str
    n_qubits: int
    n_trainable_params: int
    depth: int
    n_gates: int
    n_two_qubit_gates: int
    gate_counts: Mapping[str, int] = field(default_factory=dict)
    device_name: str = ""
    diff_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, for logging frameworks and JSON."""
        # Not dataclasses.asdict: it deep-copies, which raises for a
        # gate_counts that is a Mapping but not a dict (e.g. a mappingproxy).
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["gate_counts"] = dict(self.gate_counts)
        return d

    def __str__(self) -> str:
        labels = ("qubits", "trainable params", "depth", "gates", "two-qubit gates")
        # Gate names are indented two columns further, so they get two less padding
        width = max(
            max(len(label) for label in labels),
            max((len(name) + 2 for name in self.gate_counts), default=0),
        )
        lines = [
            f"Circuit summary: {self.layer_type} on {self.device_name} ({self.diff_method})",
            f"  {'qubits':<{width}} : {self.n_qubits}",
            f"  {'trainable params':<{width}} : {self.n_trainable_params}",
            f"  {'depth':<{width}} : {self.depth}",
            f"  {'gates':<{width}} : {self.n_gates}",
            f"  {'two-qubit gates':<{width}} : {self.n_two_qubit_gates}",
        ]
        lines += [f"    {name:<{width - 2}} : {count}" for name, count in self.gate_counts.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resolving what to inspect
# ---------------------------------------------------------------------------


def _resolve_layer(target: nn.Module) -> tuple[nn.Module, qml.qnn.TorchLayer, int]:
    """
    Return ``(layer, qlayer, n_qubits)`` for the encoding layer inside *target*.

    Accepts an encoding layer directly (anything with a ``qlayer`` TorchLayer
    and an integer ``n_qubits``), or a hybrid classifier exposing
    ``quantum_layer``.
    """
    layer = getattr(target, "quantum_layer", target)
    qlayer = getattr(layer, "qlayer", None)
    n_qubits = getattr(layer, "n_qubits", None)
    if (
        not isinstance(layer, nn.Module)
        or not isinstance(qlayer, qml.qnn.TorchLayer)
        or not isinstance(n_qubits, int)
    ):
        raise TypeError(
            f"circuit_summary expects an encoding layer (QuantumEncodingLayer, "
            f"IQPEncodingLayer) or a hybrid classifier with a quantum_layer attribute; "
            f"got {type(target).__name__}."
        )
    return layer, qlayer, n_qubits


def _logical_tape(
    qlayer: qml.qnn.TorchLayer, n_qubits: int, inputs: torch.Tensor | None = None
) -> qml.tape.QuantumScript:
    """The tape the layer executes for one sample, decomposed to LOGICAL_GATE_SET."""
    if inputs is None:
        inputs = torch.zeros(n_qubits, dtype=torch.float64)
    weights = {name: param.detach() for name, param in qlayer.qnode_weights.items()}
    # level="top": the circuit as written, before the QNode's own transforms
    # (batch expansion) and before the device rewrites gates it cannot run.
    tape = qml.workflow.construct_tape(qlayer.qnode, level="top")(inputs, **weights)
    (decomposed,), _ = qml.transforms.decompose(tape, gate_set=LOGICAL_GATE_SET)
    return decomposed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def circuit_summary(target: nn.Module) -> CircuitSummary:
    """
    Summarise the circuit of an encoding layer or of a hybrid classifier.

    Parameters
    ----------
    target:
        A ``QuantumEncodingLayer`` / ``IQPEncodingLayer``, or a classifier with
        a ``quantum_layer`` attribute (``HybridBinaryClassifier``,
        ``ParallelHybridClassifier``).

    Returns
    -------
    CircuitSummary
        Depth, gate counts and parameter count of the logical circuit.  The
        result does not depend on the device: the same layer configuration
        gives the same summary on ``default.qubit`` and ``lightning.qubit``.

    Examples
    --------
    >>> from hqnn_forge.encoding import QuantumEncodingLayer
    >>> from hqnn_forge.diagnostics import circuit_summary
    >>> summary = circuit_summary(QuantumEncodingLayer(n_qubits=4, n_layers=2))
    >>> summary.n_two_qubit_gates
    8
    >>> print(summary)  # doctest: +SKIP
    Circuit summary: QuantumEncodingLayer on lightning.qubit (adjoint)
      qubits           : 4
      ...
    """
    layer, qlayer, n_qubits = _resolve_layer(target)
    tape = _logical_tape(qlayer, n_qubits)
    resources = tape.specs["resources"]
    qnode = qlayer.qnode
    return CircuitSummary(
        layer_type=type(layer).__name__,
        n_qubits=n_qubits,
        n_trainable_params=sum(
            p.numel() for p in qlayer.qnode_weights.values() if p.requires_grad
        ),
        depth=int(resources.depth),
        n_gates=int(resources.num_gates),
        n_two_qubit_gates=sum(count for size, count in resources.gate_sizes.items() if size >= 2),
        gate_counts=dict(sorted(resources.gate_types.items())),
        device_name=str(qnode.device.name),
        diff_method=str(qnode.diff_method),
    )


def draw_circuit(target: nn.Module, inputs: torch.Tensor | None = None, decimals: int = 2) -> str:
    """
    Text drawing of the logical circuit for one sample, with the layer's
    current weights.  Suitable for ``print`` or a log line alongside a
    training run.

    Parameters
    ----------
    target:
        Same as for :func:`circuit_summary`.
    inputs:
        The one sample to draw the embedding angles for, shape ``(n_qubits,)``.
        Default: zeros, which draws every embedding rotation as ``RX(0.00)``.
        Pass a real sample to see the feature map it produces.
    decimals:
        Digits shown for gate parameters.  Default: 2.

    Notes
    -----
    Only the printed angles depend on ``inputs``; the gates and the wiring do
    not, which is why :func:`circuit_summary` does not take one.
    """
    _, qlayer, n_qubits = _resolve_layer(target)
    tape = _logical_tape(qlayer, n_qubits, inputs)
    return qml.drawer.tape_text(tape, decimals=decimals)
