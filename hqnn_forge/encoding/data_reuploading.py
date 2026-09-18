"""
hqnn_forge.encoding.data_reuploading
====================================
Quantum encoding module that re-embeds the classical feature vector before
**every** variational layer instead of once at the start of the circuit
(data re-uploading, Pérez-Salinas et al. 2020).

Design Rationale
----------------
* **Why re-upload.**  A circuit that embeds ``x`` once and then applies a
  data-independent ansatz computes, as a function of each feature, a
  trigonometric polynomial of degree one: the ⟨Z_i⟩ outputs are of the form
  ``a + b·cos(x_j) + c·sin(x_j)`` in every ``x_j``.  Embedding the same
  features ``L`` times raises the accessible degree to ``L`` (Schuld, Sweke
  & Meyer 2021): the frequency spectrum of the model grows with the number
  of uploads, not with the number of qubits.  Re-uploading is therefore the
  way to increase a fixed-qubit-count circuit's expressivity without adding
  qubits.

* **Structure.**  Per layer ℓ = 0 … L-1:

  1. ``AngleEmbedding(x, rotation)``: one rotation per qubit, angle ``x_i``
     (or ``s_{ℓ,i}·x_i`` with trainable input scaling, see below);
  2. CNOT entangling ring ``CNOT(i → i+1 mod n)``;
  3. per-qubit ``Rot(φ, θ, ω)``.

  Steps 2 and 3 are exactly one layer of the angle encoder, so with
  ``n_layers=1`` this layer is :class:`~hqnn_forge.encoding.QuantumEncodingLayer`
  gate for gate.  The measurement is ⟨Z_i⟩ on every qubit as in the other
  encoders.

* **Trainable input scaling** (``trainable_input_scaling=True``) multiplies
  the features by a learned ``(n_layers, n_qubits)`` tensor before each
  upload, as in the original proposal, where the classical data enters
  through trainable weights.  This lets the model choose which frequencies
  to use per layer at the cost of ``n_layers·n_qubits`` extra parameters.
  It is off by default so that the parameter count matches the other
  encoders.

* **Cost.**  Each layer adds ``n_qubits`` single-qubit rotations and one unit
  of depth on top of the angle encoder's ``2·n_qubits`` gates per layer, so
  gate count goes from ``n + L·2n`` to ``L·3n`` and depth grows by ``L - 1``.
  Every upload also multiplies the input gradient's contribution: under
  ``backprop`` and ``parameter-shift`` this is handled by the framework, but
  a deeper circuit is a deeper circuit for adjoint and for hardware.

References
----------
* Pérez-Salinas et al. (2020) "Data re-uploading for a universal quantum
  classifier", Quantum 4, 226.
* Schuld, Sweke & Meyer (2021) "Effect of data encoding on the expressive
  power of variational quantum-machine-learning models", PRA 103, 032430.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pennylane as qml
import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import (
    DeviceName,
    DiffMethod,
    RotationAxis,
    _expand_batch_dimension,
    _resolve_device,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw QNode function
# ---------------------------------------------------------------------------


def _make_data_reuploading_circuit(
    n_qubits: int,
    n_layers: int,
    rotation: RotationAxis,
    trainable_input_scaling: bool,
) -> Callable[..., list[qml.measurements.ExpectationMP]]:
    """
    Factory returning the bare quantum function for the re-uploading circuit.

    The returned function has the signature::

        circuit(inputs, weights)                 # trainable_input_scaling=False
        circuit(inputs, weights, input_scaling)  # trainable_input_scaling=True

    with ``inputs`` of shape ``(n_qubits,)`` (or ``(batch, n_qubits)`` when
    broadcasted), ``weights`` of shape ``(n_layers, n_qubits, 3)`` and
    ``input_scaling`` of shape ``(n_layers, n_qubits)``.

    Circuit structure (per layer ℓ = 0 … L-1)
    ------------------------------------------
    1. ``AngleEmbedding(inputs · input_scaling[ℓ], rotation)`` — the upload.
    2. CNOT ring ``CNOT(i → i+1 mod n)``.
    3. ``Rot(weights[ℓ, i])`` on every qubit.

    Then ``[⟨Z_i⟩ for i in range(n_qubits)]``.
    """

    def _layer(inputs: torch.Tensor, weights: torch.Tensor, layer: int) -> None:
        qml.AngleEmbedding(features=inputs, wires=range(n_qubits), rotation=rotation)
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
        for qubit in range(n_qubits):
            qml.Rot(
                weights[layer, qubit, 0],
                weights[layer, qubit, 1],
                weights[layer, qubit, 2],
                wires=qubit,
            )

    def measure() -> list[qml.measurements.ExpectationMP]:
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    if trainable_input_scaling:

        def scaled_circuit(
            inputs: torch.Tensor,
            weights: torch.Tensor,
            input_scaling: torch.Tensor,
        ) -> list[qml.measurements.ExpectationMP]:
            for layer in range(n_layers):
                # ``inputs`` may carry a leading batch axis; scaling is per
                # (layer, qubit) and broadcasts over it.
                _layer(inputs * input_scaling[layer], weights, layer)
            return measure()

        return scaled_circuit

    def circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[qml.measurements.ExpectationMP]:
        for layer in range(n_layers):
            _layer(inputs, weights, layer)
        return measure()

    return circuit


# ---------------------------------------------------------------------------
# Public QNode factory
# ---------------------------------------------------------------------------


def build_data_reuploading_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation: RotationAxis = "X",
    device_name: DeviceName = "lightning.qubit",
    diff_method: DiffMethod = "adjoint",
    trainable_input_scaling: bool = False,
) -> qml.QNode:
    """
    Build and return a PennyLane QNode for the data re-uploading circuit.

    Parameters
    ----------
    n_qubits:
        Number of qubits (= number of input features).
    n_layers:
        Number of uploads, each followed by one entangling + rotation block.
    rotation:
        Pauli axis of the embedding rotations: ``"X"`` (default), ``"Y"``, ``"Z"``.
    device_name, diff_method:
        As for :func:`hqnn_forge.encoding.build_encoding_qnode`.
    trainable_input_scaling:
        If ``True`` the QNode takes a third argument ``input_scaling`` of shape
        ``(n_layers, n_qubits)`` that multiplies the features before each upload.

    Raises
    ------
    ValueError
        If ``n_qubits < 2`` or ``n_layers < 1``.
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2 for the CNOT entangling ring; got {n_qubits}.")
    if n_layers < 1:
        raise ValueError(f"n_layers must be ≥ 1 (one upload per layer); got {n_layers}.")

    device = _resolve_device(device_name, n_qubits)
    circuit_fn = _make_data_reuploading_circuit(
        n_qubits, n_layers, rotation, trainable_input_scaling
    )

    qnode = qml.QNode(
        func=circuit_fn,
        device=device,
        diff_method=diff_method,
        interface="torch",
    )
    qnode = _expand_batch_dimension(qnode, diff_method)

    logger.info(
        "Re-uploading QNode built | device=%s | qubits=%d | layers=%d | diff=%s | "
        "rotation=%s | trainable_input_scaling=%s",
        device.name,
        n_qubits,
        n_layers,
        diff_method,
        rotation,
        trainable_input_scaling,
    )
    return qnode


# ---------------------------------------------------------------------------
# PyTorch nn.Module wrapper
# ---------------------------------------------------------------------------


class DataReuploadingLayer(nn.Module):
    """
    A PyTorch ``nn.Module`` wrapping the data re-uploading QNode.

    API-consistent with :class:`~hqnn_forge.encoding.QuantumEncodingLayer`:
    same constructor arguments, same ``qlayer.weights`` of shape
    ``(n_layers, n_qubits, 3)``, same ``(batch, n_qubits)`` input and output.
    The difference is inside the circuit: the features are embedded before
    every variational layer, not only before the first.

    Expressivity versus depth
    -------------------------
    With one upload the model's dependence on each feature is a degree-one
    trigonometric polynomial, whatever ``n_layers`` is; with ``L`` uploads
    it is of degree ``L`` (Schuld, Sweke & Meyer 2021).  The price is
    ``n_qubits`` more single-qubit gates and one more unit of depth per
    layer than the single-upload encoder, and, on hardware, ``L`` times the
    data-loading cost.  For ``n_layers=1`` the two layers are identical.

    Parameters
    ----------
    n_qubits:
        Number of qubits / input features.  Default: 8.
    n_layers:
        Number of uploads, each followed by a CNOT ring and ``Rot`` block.
        Default: 2.
    rotation:
        Pauli axis of the embedding rotations.  Default: ``"X"``.
    device_name:
        PennyLane device.  Falls back to ``default.qubit`` if
        ``pennylane-lightning`` is unavailable.
    diff_method:
        Gradient method.  Default: ``"adjoint"``.
    trainable_input_scaling:
        Add a trainable ``qlayer.input_scaling`` of shape
        ``(n_layers, n_qubits)``, initialised to ones, that multiplies the
        features before each upload (Pérez-Salinas et al. 2020).  Default:
        ``False``, so the parameter count matches the other encoders.

    Attributes
    ----------
    n_qubits, n_layers : int
    trainable_input_scaling : bool
    qlayer : pennylane.qnn.TorchLayer
        Owns ``weights`` and, if enabled, ``input_scaling``.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.encoding import DataReuploadingLayer
    >>> layer = DataReuploadingLayer(n_qubits=4, n_layers=3)
    >>> layer(torch.rand(2, 4)).shape
    torch.Size([2, 4])
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation: RotationAxis = "X",
        device_name: DeviceName = "lightning.qubit",
        diff_method: DiffMethod = "adjoint",
        trainable_input_scaling: bool = False,
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.trainable_input_scaling = trainable_input_scaling

        qnode = build_data_reuploading_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation=rotation,
            device_name=device_name,
            diff_method=diff_method,
            trainable_input_scaling=trainable_input_scaling,
        )

        weight_shapes: dict[str, tuple[int, ...]] = {
            "weights": (n_layers, n_qubits, 3),
        }
        if trainable_input_scaling:
            weight_shapes["input_scaling"] = (n_layers, n_qubits)

        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        if trainable_input_scaling:
            # Start as the plain re-uploading circuit: every upload sees x.
            with torch.no_grad():
                self.qlayer.input_scaling.fill_(1.0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of feature vectors, re-uploading before every layer.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_qubits)``, values in ``[-π, π]`` as for
            the angle encoder.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, n_qubits)``, each element ∈ [-1, 1].

        Raises
        ------
        ValueError
            If the last dimension of ``x`` is not ``n_qubits``.
        """
        if x.shape[-1] != self.n_qubits:
            raise ValueError(
                f"Input feature dimension {x.shape[-1]} does not match n_qubits={self.n_qubits}."
            )
        # Whole batch in one call; see QuantumEncodingLayer.forward.
        return self.qlayer(x)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        n_params = self.n_layers * self.n_qubits * 3
        if self.trainable_input_scaling:
            n_params += self.n_layers * self.n_qubits
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"trainable_input_scaling={self.trainable_input_scaling}, "
            f"n_params={n_params}"
        )
