"""
hqnn_forge.encoding._common
===========================
QNode plumbing every encoding layer shares.

The type aliases of the circuit options, the variational block and its weight
shape, input checks and readout, the device factory with its fallback chain,
and the batch expansion that makes a QNode accept ``(batch, n_features)``
inputs under every differentiation method.  They lived in
``angle_embedding.py`` until #306, which is why that module still re-exports
them; import them from here in new code.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal, get_args

import pennylane as qml
import torch
from pennylane.exceptions import AllocationError, DeviceError

from hqnn_forge.circuits import hardware_efficient_layer, strongly_entangling_layer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RotationAxis = Literal["X", "Y", "Z"]
DiffMethod = Literal["adjoint", "parameter-shift", "backprop", "finite-diff"]
DeviceName = Literal["lightning.gpu", "lightning.kokkos", "lightning.qubit", "default.qubit"]
Entangler = Literal["ring", "strongly_entangling", "hardware_efficient"]
ENTANGLERS: tuple[str, ...] = ("ring", "strongly_entangling", "hardware_efficient")
_ENTANGLER_CHOICES = "'ring', 'strongly_entangling' or 'hardware_efficient'"
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
#: :func:`is_out_of_memory`.
DEVICE_FAILURES: tuple[type[BaseException], ...] = (
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

    * ``"hardware_efficient"``: a nearest-neighbour ``CZ(i, i+1)`` ladder,
      then ``RY(θ)`` on every qubit (Kandala et al. 2017):
      :func:`hqnn_forge.circuits.hardware_efficient_layer`.  One angle per
      qubit per layer and ``n − 1`` two-qubit gates per layer, against three
      and ``n``: a third of the parameters, with CZ native on many devices.

    ``"ring"`` and ``"strongly_entangling"`` take ``weights`` of shape
    ``(n_layers, n_qubits, 3)`` and use ``n_layers · n_qubits`` ``Rot`` and
    CNOT gates; they differ in gate order and, from the second layer on, in
    which qubits the CNOTs connect.  ``"hardware_efficient"`` takes
    ``(n_layers, n_qubits)``.  :func:`variational_weight_shape` gives the shape
    for each.  The ``"ring"`` block is
    :func:`hqnn_forge.circuits.strongly_entangling_layer` applied per layer.

    ``layer_offset`` is the index of the first block within the whole ansatz,
    for circuits that interleave other gates between blocks and so apply them
    a few at a time: the ``"strongly_entangling"`` range of block ``ℓ`` is
    ``(layer_offset + ℓ) mod (n-1) + 1``, so applying the blocks one by one
    with offsets ``0 … L-1`` gives the same ranges as applying all ``L`` at
    once.  The ``"ring"`` and ``"hardware_efficient"`` blocks do not depend on
    the layer index.
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
    if entangler == "hardware_efficient":
        for layer in range(n_layers):
            hardware_efficient_layer(weights[layer], n_qubits)
        return
    if entangler != "ring":
        raise ValueError(f"entangler must be {_ENTANGLER_CHOICES}; got {entangler!r}.")
    for layer in range(n_layers):
        # CNOT ring (last qubit → first), then Rot(φ, θ, ω) on every qubit
        strongly_entangling_layer(weights[layer], n_qubits)


def variational_weight_shape(
    entangler: Entangler, n_qubits: int, n_layers: int
) -> tuple[int, ...]:
    """
    Shape of the ``weights`` tensor :func:`apply_variational_layers` reads for ``entangler``.

    The one place an encoder's variational weight shape is defined: every
    encoding layer registers its ``weights`` with this shape.  Dim 0 is always
    the layer index, which :func:`~hqnn_forge.initializers.block_local_init_`
    and the diagnostics' ``n_layers`` fallback rely on: ``(n_layers, n_qubits,
    3)`` for the ``Rot`` blocks, ``(n_layers, n_qubits)`` for the ``RY`` of
    ``"hardware_efficient"``.

    Raises
    ------
    ValueError
        For an unknown ``entangler``.
    """
    if entangler not in ENTANGLERS:
        raise ValueError(f"entangler must be {_ENTANGLER_CHOICES}; got {entangler!r}.")
    if entangler == "hardware_efficient":
        return (n_layers, n_qubits)
    return (n_layers, n_qubits, 3)


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
    if entangler not in ENTANGLERS:
        raise ValueError(f"entangler must be {_ENTANGLER_CHOICES}; got {entangler!r}.")
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


def is_out_of_memory(exc: BaseException) -> bool:
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


def resolve_device(device_name: DeviceName, n_qubits: int) -> qml.devices.Device:
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
    (see :func:`is_out_of_memory`), or if every step of the chain fails,
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
        except DEVICE_FAILURES as exc:
            if attempt == len(candidates) - 1 or is_out_of_memory(exc):
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
# Batching
# ---------------------------------------------------------------------------


def expand_batch_dimension(qnode: qml.QNode, diff_method: str) -> qml.QNode:
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
