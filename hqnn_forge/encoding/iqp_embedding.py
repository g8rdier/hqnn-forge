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
from itertools import combinations
from typing import Literal

import pennylane as qml
import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import (
    Entangler,
    Readout,
    _expand_batch_dimension,
    apply_variational_layers,
    measure_z,
    readout_wires,
)

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
    entangler: Entangler = "ring",
    readout: Readout = "all",
) -> callable:
    """
    Factory returning the bare quantum function for the IQP embedding.

    ``entangler`` and ``readout`` are as in
    :func:`hqnn_forge.encoding.angle_embedding.apply_variational_layers` and
    :func:`~hqnn_forge.encoding.angle_embedding.measure_z`.
    """
    readout_wires(n_qubits, readout)  # validate early
    if entangler not in ("ring", "strongly_entangling"):
        raise ValueError(f"entangler must be 'ring' or 'strongly_entangling'; got {entangler!r}.")
    # All-to-all entangling pattern, the same as qml.IQPEmbedding(pattern=None)
    pairs = list(combinations(range(n_qubits), 2))

    def circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[qml.measurements.ExpectationMP]:
        # ── 1. IQP embedding: H → RZ(x_i) → exp(-i x_i x_j Z_i Z_j / 2) ─────
        # This is qml.IQPEmbedding's decomposition written out gate by gate,
        # with the two-qubit MultiRZ replaced by its exact CNOT·RZ·CNOT form.
        # Written out so that a batched ``inputs`` of shape (batch, n_qubits)
        # broadcasts through single-parameter gates only.  The QNode wrapper
        # (_expand_batch_dimension) already splits the batch into one tape per
        # sample for every method except backprop, so lightning.qubit's adjoint
        # path -- which mis-shapes results for a broadcasted MultiRZ -- never
        # sees a broadcasted tape here; this form is a safeguard in case the
        # circuit is ever executed broadcasted without that wrapper.
        # ``inputs[..., i]`` selects feature i for one sample or a batch alike.
        for _ in range(n_repeats):
            for qubit in range(n_qubits):
                qml.Hadamard(wires=qubit)
                qml.RZ(inputs[..., qubit], wires=qubit)
            for i, j in pairs:
                qml.CNOT(wires=[i, j])
                qml.RZ(inputs[..., i] * inputs[..., j], wires=j)
                qml.CNOT(wires=[i, j])

        # ── 2 & 3. Variational layers, then ⟨Z⟩ on the readout wires ─────
        apply_variational_layers(weights, n_qubits, n_layers, entangler)
        return measure_z(n_qubits, readout)

    return circuit


def build_iqp_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    n_repeats: int = 1,
    device_name: DeviceName = "lightning.qubit",
    diff_method: DiffMethod = "adjoint",
    entangler: Entangler = "ring",
    readout: Readout = "all",
) -> qml.QNode:
    """Build and return a PennyLane QNode for the IQP feature map."""
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2; got {n_qubits}.")

    device = _resolve_device(device_name, n_qubits)
    circuit_fn = _make_iqp_embedding_circuit(n_qubits, n_layers, n_repeats, entangler, readout)

    qnode = qml.QNode(
        func=circuit_fn,
        device=device,
        diff_method=diff_method,
        interface="torch",
    )
    # Batched inputs: see angle_embedding._expand_batch_dimension
    return _expand_batch_dimension(qnode, diff_method)


class IQPEncodingLayer(nn.Module):
    """
    A PyTorch nn.Module wrapping the IQP-embedding QNode.

    ``entangler`` and ``readout`` are the options of
    :class:`~hqnn_forge.encoding.QuantumEncodingLayer`; the output width is
    ``n_outputs`` (``n_qubits``, or 1 with ``readout="first"``).
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_repeats: int = 1,
        device_name: DeviceName = "lightning.qubit",
        diff_method: DiffMethod = "adjoint",
        entangler: Entangler = "ring",
        readout: Readout = "all",
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_repeats = n_repeats
        self.entangler = entangler
        self.readout = readout
        self.n_outputs = len(readout_wires(n_qubits, readout))

        qnode = build_iqp_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_repeats=n_repeats,
            device_name=device_name,
            diff_method=diff_method,
            entangler=entangler,
            readout=readout,
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
        # Whole batch in one call; see QuantumEncodingLayer.forward.
        return self.qlayer(x)

    def extra_repr(self) -> str:
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_repeats={self.n_repeats}, "
            f"n_params={self.n_layers * self.n_qubits * 3}"
        )
