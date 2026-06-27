"""
hqnn_forge.circuits.strongly_entangling
=========================================
Strongly-entangling ansatz primitive.

A single call applies one *block* of:

    1. CNOT entangling ring: CNOT(i → (i+1) % n_qubits) for all i.
    2. Per-qubit SU(2) block: qml.Rot(φ, θ, ω) for all i.

This primitive is designed to be called **inside a QNode context** (decorated
function).  It does *not* embed classical data; use ``qml.AngleEmbedding``
separately before calling this.

Usage inside a QNode
--------------------
::

    @qml.qnode(dev)
    def my_circuit(weights):
        qml.AngleEmbedding(features, wires=range(n_qubits))
        for l in range(n_layers):
            strongly_entangling_layer(weights[l], n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

References
----------
* Schuld et al. (2020) "Circuit-centric quantum classifiers", PRA 101, 032308.
"""

from __future__ import annotations

import pennylane as qml
import torch


def strongly_entangling_layer(
    weights: torch.Tensor,
    n_qubits: int,
) -> None:
    """
    Apply one strongly-entangling block to the current quantum circuit.

    Gate count per call:

    * n_qubits CNOT gates  (entangling ring)
    * n_qubits Rot gates   (3 parameters each → 3 * n_qubits trainable angles)

    Parameters
    ----------
    weights:
        Rotation angles tensor of shape ``(n_qubits, 3)``.
        Each row ``[φ_i, θ_i, ω_i]`` parameterises the Euler rotation
        ``Rz(ω_i) · Ry(θ_i) · Rz(φ_i)`` applied to qubit i.
    n_qubits:
        Number of qubits.  Must match ``weights.shape[0]``.

    Raises
    ------
    ValueError
        If ``weights.shape != (n_qubits, 3)``.
    """
    if weights.shape != (n_qubits, 3):
        raise ValueError(
            f"weights must have shape (n_qubits={n_qubits}, 3); "
            f"got {tuple(weights.shape)}."
        )

    # Entangling ring: CNOT(i → i+1 mod n)
    for qubit in range(n_qubits):
        qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])

    # SU(2) rotation block
    for qubit in range(n_qubits):
        qml.Rot(
            weights[qubit, 0],
            weights[qubit, 1],
            weights[qubit, 2],
            wires=qubit,
        )
