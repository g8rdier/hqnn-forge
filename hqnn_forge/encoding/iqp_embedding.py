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

References
----------
* Havlíček et al. (2019) "Supervised learning with quantum-enhanced feature
  spaces", Nature 567, 209–212.  Introduces the IQP-type feature map
  (Hadamards, diagonal phases in the features and their pairwise products).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import combinations

import pennylane as qml
import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import (
    DeviceName,
    DiffMethod,
    Entangler,
    Readout,
    _build_training_noise,
    _expand_batch_dimension,
    _resolve_device,
    apply_variational_layers,
    check_inputs,
    measure_z,
    readout_wires,
    validate_circuit_options,
)
from hqnn_forge.noise import run_with_training_noise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases and device factory: shared with angle_embedding
# ---------------------------------------------------------------------------


def _make_iqp_embedding_circuit(
    n_qubits: int,
    n_layers: int,
    n_repeats: int = 1,
    entangler: Entangler = "ring",
    readout: Readout = "all",
) -> Callable[[torch.Tensor, torch.Tensor], list[qml.measurements.ExpectationMP]]:
    """
    Factory returning the bare quantum function for the IQP embedding.

    ``entangler`` and ``readout`` are as in
    :func:`hqnn_forge.encoding.angle_embedding.apply_variational_layers` and
    :func:`~hqnn_forge.encoding.angle_embedding.measure_z`.
    """
    validate_circuit_options(n_qubits, entangler, readout)
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
    ``noise_level`` / ``noise_position`` add training-time depolarizing
    noise exactly as in :class:`~hqnn_forge.encoding.QuantumEncodingLayer`.
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
        noise_level: float = 0.0,
        noise_position: str = "all",
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_repeats = n_repeats
        self.entangler = entangler
        self.readout = readout
        self.n_outputs = len(readout_wires(n_qubits, readout))
        self.noise_level = noise_level
        self.noise_position = noise_position

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
        # Training-time depolarizing noise; see QuantumEncodingLayer.
        self._training_noise_qnode = _build_training_noise(
            qnode, n_qubits, noise_level, noise_position
        )

    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check that ``x`` has ``n_qubits`` finite features; the values are used as given.

        This is the classical step ``forward`` applies before the QNode.  Every
        encoding layer has one, so tools which replay the circuit
        (:mod:`hqnn_forge.kernels`) validate and transform inputs exactly as
        ``forward`` does.
        """
        check_inputs(x, self.n_qubits)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of feature vectors."""
        # Whole batch in one call; see QuantumEncodingLayer.forward.
        x = self.prepare_inputs(x)
        if self.training and self._training_noise_qnode is not None:
            return run_with_training_noise(self.qlayer, self._training_noise_qnode, x)
        return self.qlayer(x)

    def extra_repr(self) -> str:
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_repeats={self.n_repeats}, "
            f"n_params={self.n_layers * self.n_qubits * 3}"
        )
