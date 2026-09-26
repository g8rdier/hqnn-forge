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
  The GPU-accelerated ``lightning.gpu`` and ``lightning.kokkos`` devices support
  the same method; ``_resolve_device`` falls back through ``lightning.qubit`` to
  ``default.qubit`` when a backend is not installed or has no usable hardware.

* **Barren Plateau Avoidance** — weights are *not* initialised here; callers should
  use `hqnn_forge.initializers.restricted_normal_init_` on the returned layer.

References
----------
* Schuld et al. (2020) "Circuit-centric quantum classifiers", PRA 101, 032308.
* Sim et al. (2019) "Expressibility and entangling capability of PQCs", Adv. Quantum
  Technol. 2, 1900070.
* Jones & Gacon (2020) "Efficient calculation of gradients in classical simulations
  of variational quantum algorithms" arXiv:2009.02823.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Literal, get_args

import pennylane as qml
import torch
import torch.nn as nn
from pennylane.exceptions import AllocationError, DeviceError

from hqnn_forge.noise import (
    Position,
    run_with_training_noise,
    training_noise_qnode,
    validate_noise,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RotationAxis = Literal["X", "Y", "Z"]
DiffMethod = Literal["adjoint", "parameter-shift", "backprop", "finite-diff"]
DeviceName = Literal["lightning.gpu", "lightning.kokkos", "lightning.qubit", "default.qubit"]
Entangler = Literal["ring", "strongly_entangling"]
Readout = Literal["all", "first"]

#: Devices tried, in order, after the requested one fails.  Each is a strict
#: subset of the previous one's requirements: ``lightning.qubit`` needs only
#: the ``pennylane-lightning`` wheel, ``default.qubit`` ships with PennyLane.
FALLBACK_CHAIN: tuple[str, ...] = ("lightning.qubit", "default.qubit")

#: What creating a device raises when its plugin or hardware is missing:
#: ``DeviceError`` for a device name no installed plugin registers,
#: ``ImportError`` / ``OSError`` when a plugin's compiled extension or a CUDA
#: library cannot be loaded, ``RuntimeError`` when the plugin loads but finds
#: no usable GPU.  A ``RuntimeError`` about memory is not one of these: see
#: :func:`_is_out_of_memory`.
_DEVICE_FAILURES: tuple[type[BaseException], ...] = (
    DeviceError,
    ImportError,
    OSError,
    RuntimeError,
)


# ---------------------------------------------------------------------------
# Variational block and readout, shared by every encoding circuit
# ---------------------------------------------------------------------------


def apply_variational_layers(
    weights: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    entangler: Entangler = "ring",
    layer_offset: int = 0,
) -> None:
    """
    Apply the ``n_layers`` variational blocks to the current circuit.

    ``entangler`` selects the block:

    * ``"ring"`` (the library's default): CNOT ring ``CNOT(i → i+1 mod n)``,
      then ``Rot(φ, θ, ω)`` on every qubit.
    * ``"strongly_entangling"``: ``qml.StronglyEntanglingLayers``, i.e.
      ``Rot`` on every qubit **then** a CNOT ring whose range grows with the
      layer index, ``r = ℓ mod (n-1) + 1``.  This is the block the published
      SHNN uses (Schuld et al. 2020, PennyLane template).

    Both take ``weights`` of shape ``(n_layers, n_qubits, 3)`` and use
    ``n_layers · n_qubits`` ``Rot`` and CNOT gates; they differ in gate order
    and, from the second layer on, in which qubits the CNOTs connect.

    ``layer_offset`` is the index of the first block within the whole ansatz,
    for circuits that interleave other gates between blocks and so apply them
    a few at a time: the ``"strongly_entangling"`` range of block ``ℓ`` is
    ``(layer_offset + ℓ) mod (n-1) + 1``, so applying the blocks one by one
    with offsets ``0 … L-1`` gives the same ranges as applying all ``L`` at
    once.  The ``"ring"`` block does not depend on the layer index.
    """
    if entangler == "strongly_entangling":
        # A single wire has no CNOT partner: leave the ranges to the template,
        # which uses 0 there instead of dividing by n - 1 = 0.
        ranges = (
            [(layer_offset + layer) % (n_qubits - 1) + 1 for layer in range(n_layers)]
            if n_qubits > 1
            else None
        )
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits), ranges=ranges)
        return
    if entangler != "ring":
        raise ValueError(f"entangler must be 'ring' or 'strongly_entangling'; got {entangler!r}.")
    for layer in range(n_layers):
        # CNOT entangling ring (cyclic: last qubit → first qubit)
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
        # Per-qubit SU(2) rotation block
        for qubit in range(n_qubits):
            qml.Rot(
                weights[layer, qubit, 0],  # φ
                weights[layer, qubit, 1],  # θ
                weights[layer, qubit, 2],  # ω
                wires=qubit,
            )


def validate_circuit_options(
    n_qubits: int,
    entangler: Entangler,
    readout: Readout,
    rotation: RotationAxis | None = None,
) -> None:
    """
    Raise ``ValueError`` for an ``entangler``, ``readout`` or (if given)
    ``rotation`` outside the allowed values.

    The encoding builders call this eagerly: ``qml.AngleEmbedding`` only
    rejects the axis when the circuit first runs, which is a forward pass away
    from the constructor that was given it -- and past get_config and a
    checkpoint.
    """
    readout_wires(n_qubits, readout)
    if entangler not in ("ring", "strongly_entangling"):
        raise ValueError(f"entangler must be 'ring' or 'strongly_entangling'; got {entangler!r}.")
    if rotation is not None and rotation not in ("X", "Y", "Z"):
        raise ValueError(f"rotation must be 'X', 'Y' or 'Z'; got {rotation!r}.")


def check_inputs(x: torch.Tensor, expected: int, name: str = "n_qubits", hint: str = "") -> None:
    """
    Raise ``ValueError`` unless ``x`` has ``expected`` features and is finite.

    The shared check of every encoding layer's ``prepare_inputs``.  A NaN or
    ±inf angle is simulated without error and gives NaN outputs, so it is
    refused here, where ``forward`` and :mod:`hqnn_forge.kernels` both see it.
    ``hint`` is appended to the width message.
    """
    if x.shape[-1] != expected:
        raise ValueError(
            f"Input feature dimension {x.shape[-1]} does not match {name}={expected}.{hint}"
        )
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Encoding layer inputs contain NaN or ±inf.")


def readout_wires(n_qubits: int, readout: Readout = "all") -> list[int]:
    """Wires measured in ⟨Z⟩: every qubit (``"all"``) or qubit 0 only (``"first"``)."""
    if readout == "all":
        return list(range(n_qubits))
    if readout == "first":
        return [0]
    raise ValueError(f"readout must be 'all' or 'first'; got {readout!r}.")


def measure_z(n_qubits: int, readout: Readout = "all") -> list[qml.measurements.ExpectationMP]:
    """``[⟨Z_i⟩ for i in readout_wires(...)]``: the circuit's return value."""
    return [qml.expval(qml.PauliZ(i)) for i in readout_wires(n_qubits, readout)]


# ---------------------------------------------------------------------------
# Device factory — graceful fallback down to default.qubit
# ---------------------------------------------------------------------------


def _is_out_of_memory(exc: BaseException) -> bool:
    """
    Whether a device-creation failure is the state vector not fitting.

    Every backend in the chain allocates the same ``2**n_qubits`` amplitudes,
    so falling back cannot help and would only move the allocation from GPU
    memory to host memory, where it can get the process killed instead of
    raising.  PennyLane raises :class:`AllocationError`; the lightning plugins
    raise a bare ``RuntimeError`` naming memory.
    """
    return isinstance(exc, AllocationError) or (
        isinstance(exc, RuntimeError) and "memory" in str(exc).lower()
    )


def _resolve_device(device_name: DeviceName, n_qubits: int) -> qml.devices.Device:
    """
    Create *device_name*, falling back along :data:`FALLBACK_CHAIN` when a
    backend is not installed or has no usable hardware, with one
    ``RuntimeWarning`` per failed step.

    The chain is ``requested → lightning.qubit → default.qubit``; entries at
    or before the requested device are skipped, so ``lightning.qubit`` falls
    straight to ``default.qubit`` and ``default.qubit`` has no fallback.

    A name outside :data:`DeviceName` is refused before anything is tried: a
    typo such as ``"default.qbit"`` would otherwise fail like a missing plugin
    and quietly run on another simulator.

    Parameters
    ----------
    device_name:
        Preferred PennyLane device string.  ``"lightning.gpu"`` (cuQuantum,
        NVIDIA) and ``"lightning.kokkos"`` (Kokkos: OpenMP on the PyPI wheel,
        CUDA/HIP when built from source) are the accelerated backends; see
        the README for their prerequisites.
    n_qubits:
        Number of qubits to allocate.

    Returns
    -------
    qml.devices.Device
        An initialised PennyLane device ready for QNode attachment.

    Raises
    ------
    ValueError
        If *device_name* is not one of :data:`DeviceName`.
    The backend's own exception if the state vector does not fit in memory
    (see :func:`_is_out_of_memory`), or if every step of the chain fails,
    which can only happen if PennyLane itself is broken (``default.qubit``
    has no dependencies).
    """
    if device_name not in get_args(DeviceName):
        raise ValueError(
            f"device_name must be one of {', '.join(map(repr, get_args(DeviceName)))}; "
            f"got {device_name!r}."
        )
    start = FALLBACK_CHAIN.index(device_name) + 1 if device_name in FALLBACK_CHAIN else 0
    candidates = [device_name, *FALLBACK_CHAIN[start:]]
    for attempt, name in enumerate(candidates):
        try:
            dev = qml.device(name, wires=n_qubits)
        except _DEVICE_FAILURES as exc:
            if attempt == len(candidates) - 1 or _is_out_of_memory(exc):
                raise
            fallback = candidates[attempt + 1]
            hint = (
                "  Install pennylane-lightning for adjoint differentiation support and "
                "significantly faster simulation."
                if fallback == "default.qubit"
                else ""
            )
            warnings.warn(
                f"Could not initialise '{name}' ({type(exc).__name__}: {exc}).  "
                f"Falling back to '{fallback}'.{hint}",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        if attempt:
            logger.info("Quantum device fell back from %s to %s", device_name, name)
        logger.debug("Quantum device initialised: %s (%d qubits)", name, n_qubits)
        return dev
    raise AssertionError("unreachable: the fallback chain always ends in a raise or a return")


# ---------------------------------------------------------------------------
# Raw QNode function
# ---------------------------------------------------------------------------


def _make_angle_embedding_circuit(
    n_qubits: int,
    n_layers: int,
    rotation: RotationAxis,
    entangler: Entangler = "ring",
    readout: Readout = "all",
) -> Callable[[torch.Tensor, torch.Tensor], list[qml.measurements.ExpectationMP]]:
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
    entangler:
        ``"ring"`` (steps 2 and 3 above) or ``"strongly_entangling"``
        (``qml.StronglyEntanglingLayers``: Rot first, then a CNOT ring of
        range ``ℓ mod (n-1) + 1``).  See :func:`apply_variational_layers`.
    readout:
        ``"all"`` (step 4 above) or ``"first"`` (``[⟨Z_0⟩]`` only, as in the
        published SHNN).

    Returns
    -------
    callable
        A plain Python function suitable for ``@qml.qnode`` decoration.
    """
    validate_circuit_options(n_qubits, entangler, readout, rotation)

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

        # ── 2 & 3. Variational layers ────────────────────────────────────
        apply_variational_layers(weights, n_qubits, n_layers, entangler)

        # ── 4. Measurement: Pauli-Z expectation on the readout wires ─────
        return measure_z(n_qubits, readout)

    return circuit


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def _expand_batch_dimension(qnode: qml.QNode, diff_method: str) -> qml.QNode:
    """
    Make *qnode* accept a batched ``inputs`` tensor of shape ``(batch, n_qubits)``
    under every supported differentiation method.

    A 2-D ``inputs`` reaches the circuit as a *broadcasted* tape: one tape whose
    embedding gates carry a batch of angles.  How that is executed depends on
    ``diff_method``:

    * ``"backprop"`` differentiates through the simulator, which handles the
      batch natively as one vectorised state-vector evolution.  This is the fast
      path and the tape is left broadcasted.
    * Every other method (``"adjoint"``, ``"parameter-shift"``,
      ``"finite-diff"``) is a gradient *transform* on the tape, and the
      parameter-shift and finite-difference transforms refuse a broadcasted
      tape when the gradient with respect to the broadcasted parameters is
      requested -- which is exactly the case when a classical encoder upstream
      needs input gradients.  ``lightning.qubit``'s adjoint path also
      mis-shapes results for some broadcasted two-qubit rotations.  For these
      the tape is split into one tape per sample *before* the gradient
      transform sees it, so each tape is unbroadcasted and the whole batch is
      still handed to the device as a single list of tapes.

    Either way the QNode's signature and results are unchanged: it returns
    ``n_qubits`` expectation values, each of shape ``(batch,)``.
    """
    if diff_method == "backprop":
        return qnode
    return qml.transforms.broadcast_expand(qnode)


# ---------------------------------------------------------------------------
# Public QNode factory
# ---------------------------------------------------------------------------


def build_encoding_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation: RotationAxis = "X",
    device_name: DeviceName = "lightning.qubit",
    diff_method: DiffMethod = "adjoint",
    entangler: Entangler = "ring",
    readout: Readout = "all",
) -> qml.QNode:
    """
    Build and return a PennyLane QNode for the angle-embedding feature map.

    The QNode is bound to the device :func:`_resolve_device` returns for
    *device_name* and configured for the specified differentiation method.

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
        adjoint differentiation.  An unavailable backend falls back along
        ``lightning.qubit → default.qubit`` with a warning per step.
    diff_method:
        Differentiation strategy:

        - ``"adjoint"``         — exact, O(p) memory; requires lightning device.
        - ``"parameter-shift"`` — exact, hardware-compatible, O(p) circuit evals.
        - ``"backprop"``        — auto-diff through simulator; requires default.qubit.
        - ``"finite-diff"``     — approximate; avoid for training.
    entangler:
        ``"ring"`` (default) or ``"strongly_entangling"``; see
        :func:`apply_variational_layers`.
    readout:
        ``"all"`` (default): ⟨Z_i⟩ on every qubit.  ``"first"``: ⟨Z_0⟩ only.

    Returns
    -------
    qml.QNode
        A callable QNode with signature
        ``(inputs: Tensor, weights: Tensor) -> Tensor``
        where outputs are ⟨Z_i⟩ expectation values, shape ``(n_qubits,)``
        (or ``(1,)`` with ``readout="first"``).

    Raises
    ------
    ValueError
        If ``n_qubits < 2`` (minimum for a meaningful entangling ring), or if
        ``rotation``, ``entangler`` or ``readout`` is not one of the values
        above -- all checked here, before the circuit first runs.

    Examples
    --------
    >>> qnode = build_encoding_qnode(n_qubits=8, n_layers=2)
    >>> x = torch.rand(8)
    >>> w = torch.zeros(2, 8, 3)
    >>> result = qnode(x, w)  # list of 8 expectation values
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2 for the CNOT entangling ring; got {n_qubits}.")

    device = _resolve_device(device_name, n_qubits)
    circuit_fn = _make_angle_embedding_circuit(n_qubits, n_layers, rotation, entangler, readout)

    qnode = qml.QNode(
        func=circuit_fn,
        device=device,
        diff_method=diff_method,
        interface="torch",  # enables PyTorch autograd interop
    )
    qnode = _expand_batch_dimension(qnode, diff_method)

    logger.info(
        "QNode built | device=%s | qubits=%d | layers=%d | diff=%s | rotation=%s | "
        "entangler=%s | readout=%s",
        device.name,
        n_qubits,
        n_layers,
        diff_method,
        rotation,
        entangler,
        readout,
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
    small-angle initial values (see :mod:`hqnn_forge.initializers` for what
    that heuristic does and does not guarantee).

    Parameters
    ----------
    n_qubits:
        Number of qubits / input feature dimensions.  Default: 8.
    n_layers:
        Number of entangling + rotation blocks in the VQC ansatz.  Default: 2.
    rotation:
        Pauli axis for AngleEmbedding: ``"X"`` | ``"Y"`` | ``"Z"``.
    device_name:
        PennyLane device, one of :data:`DeviceName`.  An unavailable backend
        falls back along ``lightning.qubit → default.qubit`` with a warning
        per step.
    diff_method:
        Gradient method.  Use ``"adjoint"`` with ``lightning.qubit`` for
        exact, efficient gradients during state-vector simulation.
    entangler:
        ``"ring"`` (default) or ``"strongly_entangling"``; see
        :func:`apply_variational_layers`.  Same parameter count either way.
    readout:
        ``"all"`` (default): the layer returns ``(batch, n_qubits)``.
        ``"first"``: ⟨Z_0⟩ only, ``(batch, 1)``, the published SHNN readout.
    noise_level:
        Depolarizing probability applied to the circuit in **train mode**,
        in ``[0, 0.75]``; ``0`` (default) is the plain noiseless layer.  With
        ``noise_level > 0`` the train-mode forward pass runs the circuit on
        ``default.mixed`` with a ``DepolarizingChannel`` inserted, so
        gradients are computed through the noisy circuit (noise-aware
        training); eval mode is always noiseless, like dropout.  Backprop
        keeps a ``batch × 4^n`` density matrix per operation, so this is
        practical up to about 6 qubits.  See :mod:`hqnn_forge.noise`.
    noise_position:
        ``"all"`` (after every gate, default) or ``"end"`` (before
        measurement), as in :func:`hqnn_forge.noise.apply_depolarizing_noise`.

    Attributes
    ----------
    n_qubits : int
    n_layers : int
    n_outputs : int
        Width of the output: ``n_qubits`` or 1.
    entangler : str
    readout : str
    noise_level : float
    noise_position : str
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
        entangler: Entangler = "ring",
        readout: Readout = "all",
        noise_level: float = 0.0,
        noise_position: Position = "all",
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        self.readout = readout
        self.n_outputs = len(readout_wires(n_qubits, readout))
        self.noise_level = noise_level
        self.noise_position = noise_position

        # Build the QNode ─────────────────────────────────────────────────
        qnode = build_encoding_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation=rotation,
            device_name=device_name,
            diff_method=diff_method,
            entangler=entangler,
            readout=readout,
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

        # Training-time depolarizing noise (see hqnn_forge.noise) ─────────
        self._training_noise_qnode = _build_training_noise(
            qnode, n_qubits, noise_level, noise_position
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check that ``x`` has ``n_qubits`` finite features; the angles are used as given.

        This is the classical step ``forward`` applies before the QNode.  Every
        encoding layer has one, so tools which replay the circuit
        (:mod:`hqnn_forge.kernels`) validate and transform inputs exactly as
        ``forward`` does.
        """
        check_inputs(
            x,
            self.n_qubits,
            hint=f"  Apply PCA to reduce to {self.n_qubits} features before passing "
            f"to QuantumEncodingLayer.",
        )
        return x

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
            Quantum expectation values of shape ``(batch_size, n_outputs)``
            (``n_qubits``, or 1 with ``readout="first"``), each ∈ [-1, 1].

        Raises
        ------
        ValueError
            If the last dimension of ``x`` does not equal ``self.n_qubits``.
        """
        x = self.prepare_inputs(x)

        # TorchLayer hands the whole batch to the QNode in one call and reshapes
        # the result to (batch, n_qubits).  Whether the batch is executed as one
        # broadcasted tape or split into one tape per sample is decided in
        # build_encoding_qnode (see _expand_batch_dimension); the outputs and
        # gradients are the same either way.
        if self.training and self._training_noise_qnode is not None:
            return run_with_training_noise(self.qlayer, self._training_noise_qnode, x)
        return self.qlayer(x)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        options = ""
        if self.entangler != "ring":
            options += f", entangler={self.entangler!r}"
        if self.readout != "all":
            options += f", readout={self.readout!r}"
        if self.noise_level:
            options += f", noise_level={self.noise_level}"
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_params={self.n_layers * self.n_qubits * 3}{options}"
        )


def _build_training_noise(
    qnode: qml.QNode, n_qubits: int, noise_level: float, noise_position: Position
) -> qml.QNode | None:
    """
    The train-mode QNode for ``noise_level > 0``, or ``None`` for the
    noiseless default.  Shared by every encoding layer; validation happens
    here so a bad ``noise_level`` fails at construction.
    """
    validate_noise(noise_level, noise_position)
    if noise_level == 0.0:
        return None
    return training_noise_qnode(qnode, n_qubits, noise_level, noise_position)


# ---------------------------------------------------------------------------
# Re-export convenience alias
# ---------------------------------------------------------------------------
AngleEmbeddingQNode = build_encoding_qnode
"""Alias: ``build_encoding_qnode`` — returns the raw QNode without an nn.Module wrapper."""
