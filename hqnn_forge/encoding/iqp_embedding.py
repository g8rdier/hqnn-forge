"""
hqnn_forge.encoding.iqp_embedding
==================================
Quantum encoding module projecting classical tabular feature vectors
into an n-qubit Hilbert space via an Instantaneous Quantum Polynomial (IQP) embedding,
followed by a strongly-entangled variational ansatz.

Design Rationale
----------------
* **IQP Embedding** maps features into a highly entangled state using a diagonal
  Hamiltonian. It applies Hadamards, followed by RZ(x_i) and IsingZZ(x_i * x_j)
  entangling operations. This is known to be classically hard to simulate.
* **Strongly-Entangling Ansatz** — after embedding, L layers of a CNOT ring
  followed by per-qubit SU(2) Rot(φ, θ, ω) gates are applied.
* **Adjoint Differentiation** — the QNode is configured for the `adjoint` method.
* **Barren Plateau Avoidance** — use `restricted_normal_init_` on the returned layer.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import pennylane as qml
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DiffMethod   = Literal["adjoint", "parameter-shift", "backprop", "finite-diff"]
DeviceName   = Literal["lightning.qubit", "default.qubit"]


def _resolve_device(device_name: DeviceName, n_qubits: int) -> qml.Device:
    """Resolve and return a PennyLane device."""
    try:
        dev = qml.device(device_name, wires=n_qubits)
        logger.debug("Quantum device initialised: %s (%d qubits)", device_name, n_qubits)
        return dev
    except (qml.DeviceError, ImportError) as exc:
        fallback = "default.qubit"
        warnings.warn(
            f"Could not initialise '{device_name}' ({exc}).  "
            f"Falling back to '{fallback}'.",
            RuntimeWarning,
            stacklevel=3,
        )
        return qml.device(fallback, wires=n_qubits)


def _make_iqp_embedding_circuit(
    n_qubits: int,
    n_layers: int,
    n_repeats: int = 1,
) -> callable:
    """
    Factory returning the bare quantum function for the IQP embedding.
    """
    def circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[qml.measurements.ExpectationMP]:
        # ── 1. IQP embedding: H → RZ(x_i) → IsingZZ(x_i*x_j) ────────────
        qml.IQPEmbedding(
            features=inputs,
            wires=range(n_qubits),
            n_repeats=n_repeats,
            pattern=None, # full entanglement pattern
        )

        # ── 2 & 3. Strongly entangling layers ────────────────────────────
        for layer in range(n_layers):
            # 2a. CNOT entangling ring
            for qubit in range(n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])

            # 2b. Per-qubit SU(2) rotation block
            for qubit in range(n_qubits):
                qml.Rot(
                    weights[layer, qubit, 0],  # φ
                    weights[layer, qubit, 1],  # θ
                    weights[layer, qubit, 2],  # ω
                    wires=qubit,
                )

        # ── 3. Measurement: Pauli-Z expectation on every qubit ────────────
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def build_iqp_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    n_repeats: int = 1,
    device_name: DeviceName = "lightning.qubit",
    diff_method: DiffMethod = "adjoint",
) -> qml.QNode:
    """Build and return a PennyLane QNode for the IQP feature map."""
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2; got {n_qubits}.")

    device = _resolve_device(device_name, n_qubits)
    circuit_fn = _make_iqp_embedding_circuit(n_qubits, n_layers, n_repeats)

    qnode = qml.QNode(
        func=circuit_fn,
        device=device,
        diff_method=diff_method,
        interface="torch",
    )

    return qnode


class IQPEncodingLayer(nn.Module):
    """
    A PyTorch nn.Module wrapping the IQP-embedding QNode.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_repeats: int = 1,
        device_name: DeviceName = "lightning.qubit",
        diff_method: DiffMethod = "adjoint",
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_repeats = n_repeats

        qnode = build_iqp_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_repeats=n_repeats,
            device_name=device_name,
            diff_method=diff_method,
        )

        weight_shapes: dict[str, tuple[int, ...]] = {
            "weights": (n_layers, n_qubits, 3),
        }

        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of feature vectors."""
        if x.shape[-1] != self.n_qubits:
            raise ValueError(
                f"Input feature dimension {x.shape[-1]} does not match "
                f"n_qubits={self.n_qubits}."
            )
        return torch.stack([self.qlayer(sample) for sample in x])

    def extra_repr(self) -> str:
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_repeats={self.n_repeats}, "
            f"n_params={self.n_layers * self.n_qubits * 3}"
        )
