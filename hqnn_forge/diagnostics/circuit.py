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
separately because they are what NISQ feasibility is usually judged by.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
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
        Count per gate name, sorted by name.
    device_name:
        PennyLane device the layer's QNode is bound to, after any fallback.
    diff_method:
        Differentiation method the QNode is configured with.
    n_inert_params:
        Trainable parameters that cannot affect any measurement for **any**
        input or weight values, found structurally by
        :func:`count_inert_parameters`.  The typical case is the ``ω`` of a
        ``Rot`` that is the last non-diagonal gate before a ``⟨Z⟩`` readout:
        ``Rot = RZ(ω)·RY(θ)·RZ(φ)`` and the final ``RZ`` commutes with ``Z``.
        ``n_trainable_params - n_inert_params`` is the count that can move the
        output.
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
    n_inert_params: int = 0

    @property
    def n_effective_params(self) -> int:
        """``n_trainable_params - n_inert_params``."""
        return self.n_trainable_params - self.n_inert_params

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, for logging frameworks and JSON."""
        d = asdict(self)
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
            f"  {'inert params':<{width}} : {self.n_inert_params}",
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
    qlayer: qml.qnn.TorchLayer, n_qubits: int, *, keep_grad: bool = False
) -> qml.tape.QuantumScript:
    """
    The tape the layer executes for one sample, decomposed to LOGICAL_GATE_SET.

    With ``keep_grad=True`` the weight tensors keep their ``requires_grad``
    flag, so the gate parameters that come from trainable weights can be told
    apart from the (non-trainable) inputs; used by :func:`count_inert_parameters`.
    """
    inputs = torch.zeros(n_qubits, dtype=torch.float64)
    weights = {
        name: (param if keep_grad else param.detach())
        for name, param in qlayer.qnode_weights.items()
    }
    # level="top": the circuit as written, before the QNode's own transforms
    # (batch expansion) and before the device rewrites gates it cannot run.
    tape = qml.workflow.construct_tape(qlayer.qnode, level="top")(inputs, **weights)
    (decomposed,), _ = qml.transforms.decompose(tape, gate_set=LOGICAL_GATE_SET)
    return decomposed


# ---------------------------------------------------------------------------
# Inert parameters: what can never reach a measurement
# ---------------------------------------------------------------------------

# Backward-propagated support of the measured observables on each wire:
# nothing measured touches the wire / only Z-type (diagonal) content / may
# carry X or Y.  Only the last one anticommutes with a Z rotation.
_NONE, _Z, _XY = 0, 1, 2

# Gates diagonal in the computational basis: they commute with every Z-string.
_DIAGONAL = frozenset(
    {"RZ", "PhaseShift", "MultiRZ", "IsingZZ", "CZ", "PauliZ", "S", "T", "Identity"}
)


def _requires_grad(value: Any) -> bool:
    try:
        return bool(qml.math.requires_grad(value))
    except Exception:  # noqa: BLE001 - plain floats and the like
        return False


def count_inert_parameters(tape: qml.tape.QuantumScript) -> int:
    """
    Number of trainable gate parameters that cannot affect any measurement of
    ``tape``, for any input or weight values.

    The measured observables are propagated backwards through the circuit in
    the Heisenberg picture, keeping per wire only whether the observable's
    content there is nothing, diagonal (``Z``-type) or possibly ``X``/``Y``:
    CNOT and CZ move ``Z`` content from target to control and ``X``/``Y``
    content from control to target, a non-diagonal single-qubit gate turns
    ``Z`` into ``X``/``Y``, and diagonal gates change nothing.  A trainable
    parameter is inert when its gate acts on wires with no content at all,
    or when the gate is diagonal (``RZ``, ``PhaseShift``, ``MultiRZ``) or the
    final ``RZ(ω)`` of a ``Rot`` and every wire it touches carries only
    diagonal content: the gate then commutes with everything measured.

    The propagation over-approximates the ``X``/``Y`` content, so the count
    is a lower bound on the parameters that are dead for structural reasons:
    everything it counts has an exactly zero gradient for every input, and
    parameters that are dead only for particular inputs or weights (say, a
    rotation of ``|0⟩`` about ``Z``) are not counted.

    Only ``expval`` of ``PauliZ`` products is treated as diagonal; any other
    measurement marks its wires as ``X``/``Y`` content, and a measurement
    without wires (``state``, ``probs`` over all wires) marks every wire.
    Gates outside :data:`LOGICAL_GATE_SET` are treated as fully mixing.
    """
    support: dict[Any, int] = dict.fromkeys(tape.wires, _NONE)
    for measurement in tape.measurements:
        obs = getattr(measurement, "obs", None)
        wires = list(measurement.wires) if len(measurement.wires) else list(tape.wires)
        diagonal = obs is not None and all(
            getattr(term, "name", "") == "PauliZ"
            for term in (obs.operands if hasattr(obs, "operands") else [obs])
        )
        for wire in wires:
            support[wire] = max(support[wire], _Z if diagonal else _XY)

    inert = 0
    for op in reversed(tape.operations):
        wires = list(op.wires)
        n_trainable = sum(1 for value in op.data if _requires_grad(value))
        if all(support[w] == _NONE for w in wires):
            inert += n_trainable  # nothing measured downstream ever sees this gate
            continue
        if op.name in _DIAGONAL:
            if all(support[w] != _XY for w in wires):
                inert += n_trainable  # commutes with every observable it meets
            elif len(wires) > 1:
                # X on one wire of a diagonal multi-qubit gate spreads Z to the others
                for w in wires:
                    support[w] = max(support[w], _Z)
        elif op.name == "CNOT":
            control, target = wires
            new_control, new_target = support[control], support[target]
            if support[target] != _NONE:
                new_control = max(new_control, _Z)  # Z_t → Z_c Z_t, Y_t → Z_c Y_t
            if support[control] == _XY:
                new_target = _XY  # X_c → X_c X_t
            support[control], support[target] = new_control, new_target
        elif op.name == "Rot":
            (wire,) = wires
            # Rot = RZ(ω)·RY(θ)·RZ(φ), ω applied last: it commutes with a
            # diagonal observable, so ω is inert whenever the wire carries no
            # X/Y content.  RY(θ) then mixes Z into X/Y for the earlier gates.
            if support[wire] != _XY and _requires_grad(op.data[2]):
                inert += 1
            support[wire] = _XY
        else:
            # Any other gate (RX, RY, Hadamard, ...): content on its wires may
            # become X/Y, and a multi-qubit gate may spread it across its wires.
            for w in wires:
                support[w] = _XY
    return inert


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
        Depth, gate counts, parameter count and inert-parameter count of the
        logical circuit.  The result does not depend on the device: the same
        layer configuration gives the same summary on ``default.qubit`` and
        ``lightning.qubit``.

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
    n_inert = count_inert_parameters(_logical_tape(qlayer, n_qubits, keep_grad=True))
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
        n_inert_params=n_inert,
    )


def draw_circuit(target: nn.Module, decimals: int = 2) -> str:
    """
    Text drawing of the logical circuit, one sample, with the layer's current
    weights.  Suitable for ``print`` or a log line alongside a training run.

    Parameters
    ----------
    target:
        Same as for :func:`circuit_summary`.
    decimals:
        Digits shown for gate parameters.  Default: 2.
    """
    _, qlayer, n_qubits = _resolve_layer(target)
    return qml.drawer.tape_text(_logical_tape(qlayer, n_qubits), decimals=decimals)
