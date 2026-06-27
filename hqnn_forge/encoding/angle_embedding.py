"""
hqnn_forge.encoding.angle_embedding
====================================
Core quantum encoding module for projecting classical tabular feature vectors
into an n-qubit Hilbert space via angle (rotation) embedding followed by a
strongly-entangled variational ansatz.

Design Rationale
----------------
* **Angle Embedding** maps each input feature x_i ∈ ℝ to a Pauli-rotation angle
  (default: RX) on qubit i.  This keeps the encoding linear in feature values and
  avoids the exponential "Hilbert-space crowding" of more aggressive embeddings.

* **Strongly-Entangling Ansatz** — after embedding, L layers of a CNOT ring
  followed by per-qubit SU(2) Rot(φ, θ, ω) gates are applied.  This produces a
  high-expressibility ansatz while keeping the depth O(n * L).

* **Adjoint Differentiation** — the QNode is configured for the `adjoint` method
  on a `lightning.qubit` device.  Adjoint diff computes exact gradients in a
  single forward + backward pass and scales as O(p) in the number of parameters p,
  making it strictly superior to the parameter-shift rule for state-vector sims.

* **Barren Plateau Avoidance** — weights are *not* initialised here; callers should
  use `hqnn_forge.initializers.restricted_normal_init_` on the returned layer.

References
----------
* Schuld et al. (2020) "Circuit-centric quantum classifiers", PRA 101, 032308.
* Sim et al. (2019) "Expressibility and entangling capability of PQCs", Adv. Quantum
  Technol. 2, 1900070.
* Jones & Gacon (2021) "Efficient calculation of gradients in classical simulations
  of variational quantum algorithms" arXiv:2009.02823.
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
RotationAxis = Literal["X", "Y", "Z"]
DiffMethod   = Literal["adjoint", "parameter-shift", "backprop", "finite-diff"]
DeviceName   = Literal["lightning.qubit", "default.qubit"]


# ---------------------------------------------------------------------------
# Device factory — graceful fallback from lightning.qubit to default.qubit
# ---------------------------------------------------------------------------

def _resolve_device(device_name: DeviceName, n_qubits: int) -> qml.Device:
    """
    Attempt to create *device_name*; fall back to ``default.qubit`` when
    ``pennylane-lightning`` is not installed, emitting a warning in that case.

    Parameters
    ----------
    device_name:
        Preferred PennyLane device string, e.g. ``"lightning.qubit"``.
    n_qubits:
        Number of qubits to allocate.

    Returns
    -------
    qml.Device
        An initialised PennyLane device ready for QNode attachment.
    """
    try:
        dev = qml.device(device_name, wires=n_qubits)
        logger.debug("Quantum device initialised: %s (%d qubits)", device_name, n_qubits)
        return dev
    except (qml.DeviceError, ImportError) as exc:
        fallback = "default.qubit"
        warnings.warn(
            f"Could not initialise '{device_name}' ({exc}).  "
            f"Falling back to '{fallback}'.  Install pennylane-lightning for "
            f"adjoint differentiation support and significantly faster simulation.",
            RuntimeWarning,
            stacklevel=3,
        )
        return qml.device(fallback, wires=n_qubits)


# ---------------------------------------------------------------------------
# Raw QNode function
# ---------------------------------------------------------------------------

def _make_angle_embedding_circuit(
    n_qubits: int,
    n_layers: int,
    rotation: RotationAxis,
) -> callable:
    """
    Factory returning the *bare quantum function* (not yet a QNode) that
    implements the angle-embedding feature map + strongly-entangled ansatz.

    The returned function has the signature::

        circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[float]

    where

    * ``inputs``  — shape ``(n_qubits,)`` — the pre-processed feature vector.
    * ``weights`` — shape ``(n_layers, n_qubits, 3)`` — rotation angles per
                    layer, qubit, and Euler angle (φ, θ, ω) for ``qml.Rot``.

    Circuit structure (per layer ℓ = 0 … L-1)
    ------------------------------------------
    1. **Feature embedding** (applied before the first layer only):
       ``AngleEmbedding(inputs, wires, rotation=rotation)``
       → RX(x_i) on wire i, ∀ i ∈ {0, …, n_qubits-1}.

    2. **CNOT entangling ring**:
       CNOT(i → i+1 mod n) for i ∈ {0, …, n_qubits-1}.
       This creates a cyclic entanglement graph ensuring all-to-all reachability
       within a single layer and avoids "barren plateau–inducing" global 2-designs
       compared to random full entanglers.

    3. **Per-qubit SU(2) rotation block**:
       ``qml.Rot(φ, θ, ω, wires=i)`` applies Rz(ω)·Ry(θ)·Rz(φ), covering the
       full Bloch sphere.  This is the most expressive single-qubit gate.

    4. **Measurement**:
       Returns ``[qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]``.
       Each output is ∈ [-1, 1], giving an n_qubits–dimensional real vector
       suitable as input to a classical head.

    Parameters
    ----------
    n_qubits:
        Number of qubits (= number of input features).
    n_layers:
        Number of variational layers L.  Depth = O(n_qubits * n_layers).
    rotation:
        Pauli axis used by AngleEmbedding: ``"X"`` | ``"Y"`` | ``"Z"``.

    Returns
    -------
    callable
        A plain Python function suitable for ``@qml.qnode`` decoration.
    """
    def circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[qml.measurements.ExpectationMP]:
        # ── 1. Angle embedding: map x_i → RX(x_i)|0⟩ on wire i ──────────
        qml.AngleEmbedding(
            features=inputs,
            wires=range(n_qubits),
            rotation=rotation,
        )

        # ── 2 & 3. Strongly entangling layers ────────────────────────────
        for layer in range(n_layers):
            # 2a. CNOT entangling ring  (cyclic: last qubit → first qubit)
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


# ---------------------------------------------------------------------------
# Public QNode factory
# ---------------------------------------------------------------------------

def build_encoding_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation: RotationAxis = "X",
    device_name: DeviceName = "lightning.qubit",
    diff_method: DiffMethod = "adjoint",
) -> qml.QNode:
    """
    Build and return a PennyLane QNode for the angle-embedding feature map.

    The QNode is bound to a ``lightning.qubit`` device (or ``default.qubit``
    on fallback) and configured for the specified differentiation method.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  Must equal the dimensionality of the input feature
        vector after PCA reduction.  Default: 8.
    n_layers:
        Number of entangling + rotation layers in the VQC ansatz.
        More layers increase expressibility but deepen the circuit.  Default: 2.
    rotation:
        Pauli rotation axis for AngleEmbedding: ``"X"`` (default), ``"Y"``, or ``"Z"``.
    device_name:
        PennyLane device string.  ``"lightning.qubit"`` is strongly preferred for
        adjoint differentiation.  Falls back to ``"default.qubit"`` automatically.
    diff_method:
        Differentiation strategy:

        - ``"adjoint"``         — exact, O(p) memory; requires lightning device.
        - ``"parameter-shift"`` — exact, hardware-compatible, O(p) circuit evals.
        - ``"backprop"``        — auto-diff through simulator; requires default.qubit.
        - ``"finite-diff"``     — approximate; avoid for training.

    Returns
    -------
    qml.QNode
        A callable QNode with signature
        ``(inputs: Tensor, weights: Tensor) -> Tensor``
        where outputs are ⟨Z_i⟩ expectation values, shape ``(n_qubits,)``.

    Raises
    ------
    ValueError
        If ``n_qubits < 2`` (minimum for a meaningful entangling ring).

    Examples
    --------
    >>> qnode = build_encoding_qnode(n_qubits=8, n_layers=2)
    >>> x = torch.rand(8)
    >>> w = torch.zeros(2, 8, 3)
    >>> result = qnode(x, w)  # list of 8 expectation values
    """
    if n_qubits < 2:
        raise ValueError(
            f"n_qubits must be ≥ 2 for the CNOT entangling ring; got {n_qubits}."
        )

    device = _resolve_device(device_name, n_qubits)
    circuit_fn = _make_angle_embedding_circuit(n_qubits, n_layers, rotation)

    qnode = qml.QNode(
        func=circuit_fn,
        device=device,
        diff_method=diff_method,
        interface="torch",  # enables PyTorch autograd interop
    )

    logger.info(
        "QNode built | device=%s | qubits=%d | layers=%d | diff=%s | rotation=%s",
        device.name,
        n_qubits,
        n_layers,
        diff_method,
        rotation,
    )
    return qnode


# ---------------------------------------------------------------------------
# PyTorch nn.Module wrapper
# ---------------------------------------------------------------------------

class QuantumEncodingLayer(nn.Module):
    """
    A PyTorch ``nn.Module`` that wraps the angle-embedding QNode as a fully
    differentiable layer via ``pennylane.qnn.TorchLayer``.

    The layer owns the variational weights as ``nn.Parameter`` objects.  During
    the forward pass the classical ``inputs`` tensor is embedded into the quantum
    circuit and the Pauli-Z expectation values are returned as a real-valued
    tensor, enabling direct composition with classical ``nn.Linear`` layers.

    Weight Shapes
    -------------
    The internal ``TorchLayer`` registers one trainable parameter:

    +-----------+------------------------------------+
    | Name      | Shape                              |
    +===========+====================================+
    | ``weights``| ``(n_layers, n_qubits, 3)``       |
    +-----------+------------------------------------+

    **Important**: Call ``hqnn_forge.initializers.restricted_normal_init_``
    on ``layer.qlayer.weights`` immediately after construction to obtain
    barren-plateau-safe initial values (see :mod:`hqnn_forge.initializers`).

    Parameters
    ----------
    n_qubits:
        Number of qubits / input feature dimensions.  Default: 8.
    n_layers:
        Number of entangling + rotation blocks in the VQC ansatz.  Default: 2.
    rotation:
        Pauli axis for AngleEmbedding: ``"X"`` | ``"Y"`` | ``"Z"``.
    device_name:
        PennyLane device.  Falls back to ``default.qubit`` if
        ``pennylane-lightning`` is unavailable.
    diff_method:
        Gradient method.  Use ``"adjoint"`` with ``lightning.qubit`` for
        exact, efficient gradients during state-vector simulation.

    Attributes
    ----------
    n_qubits : int
    n_layers : int
    qlayer : pennylane.qnn.TorchLayer
        The underlying differentiable quantum layer.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.encoding import QuantumEncodingLayer
    >>> from hqnn_forge.initializers import restricted_normal_init_
    >>>
    >>> layer = QuantumEncodingLayer(n_qubits=8, n_layers=2)
    >>> restricted_normal_init_(layer.qlayer.weights, n_qubits=8, n_layers=2)
    >>>
    >>> x = torch.randn(4, 8)   # batch of 4 samples
    >>> out = layer(x)           # shape: (4, 8)
    >>> out.shape
    torch.Size([4, 8])
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation: RotationAxis = "X",
        device_name: DeviceName = "lightning.qubit",
        diff_method: DiffMethod = "adjoint",
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Build the QNode ─────────────────────────────────────────────────
        qnode = build_encoding_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation=rotation,
            device_name=device_name,
            diff_method=diff_method,
        )

        # Declare the trainable weight tensor shape for TorchLayer ─────────
        # Shape: (n_layers, n_qubits, 3)
        #   dim-0: layer index ℓ ∈ {0, …, n_layers-1}
        #   dim-1: qubit  index i ∈ {0, …, n_qubits-1}
        #   dim-2: Euler angles (φ, θ, ω) for qml.Rot
        weight_shapes: dict[str, tuple[int, ...]] = {
            "weights": (n_layers, n_qubits, 3),
        }

        # Wrap QNode as an nn.Module with registered Parameters ───────────
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of feature vectors into Pauli-Z expectation values.

        Parameters
        ----------
        x : torch.Tensor
            Classical input tensor of shape ``(batch_size, n_qubits)``.
            Values should be in ``[-π, π]`` for meaningful angle embedding
            (apply ``torch.tanh(x) * π`` or similar normalisation upstream).

        Returns
        -------
        torch.Tensor
            Quantum expectation values of shape ``(batch_size, n_qubits)``,
            with each element ∈ [-1, 1].

        Raises
        ------
        ValueError
            If the last dimension of ``x`` does not equal ``self.n_qubits``.
        """
        if x.shape[-1] != self.n_qubits:
            raise ValueError(
                f"Input feature dimension {x.shape[-1]} does not match "
                f"n_qubits={self.n_qubits}.  Apply PCA to reduce to {self.n_qubits} "
                f"features before passing to QuantumEncodingLayer."
            )

        # TorchLayer processes one sample at a time; vmap over batch dim.
        # torch.vmap is experimental — fall back to a list comprehension which
        # is safe and readable.  For high-throughput production use, replace
        # with torch.vmap once PennyLane adds full vmap support.
        return torch.stack([self.qlayer(sample) for sample in x])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_params={self.n_layers * self.n_qubits * 3}"
        )


# ---------------------------------------------------------------------------
# Re-export convenience alias
# ---------------------------------------------------------------------------
AngleEmbeddingQNode = build_encoding_qnode
"""Alias: ``build_encoding_qnode`` — returns the raw QNode without an nn.Module wrapper."""
