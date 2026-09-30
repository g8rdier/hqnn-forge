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

* **Initialisation** — weights are *not* initialised here; callers should use
  `hqnn_forge.initializers.restricted_normal_init_` on the returned layer.  Note
  what that buys for this circuit (measured, see `hqnn_forge.initializers`):
  more initial gradient variance only for inputs near zero, and no escape from
  its exponential decay with qubit count.  The cascaded CNOT ring puts every
  qubit in the backward light cone of each ⟨Z_i⟩ within two layers (of ⟨Z_0⟩
  and ⟨Z_{n-1}⟩ within one), so at the default depth the per-qubit readouts
  are global costs in the sense of Cerezo et al. (2021).  The escape is the
  entangler: with ``entangler="brickwork"`` the readouts stay local at
  shallow depth and the total gradient variance does not fall from 4 to 8
  qubits (see :func:`apply_variational_layers`).

References
----------
* Schuld et al. (2020) "Circuit-centric quantum classifiers", PRA 101, 032308.
* Sim et al. (2019) "Expressibility and entangling capability of PQCs", Adv. Quantum
  Technol. 2, 1900070.
* Jones & Gacon (2020) "Efficient calculation of gradients in classical simulations
  of variational quantum algorithms" arXiv:2009.02823.
"""

from __future__ import annotations

import inspect
import logging
import os
import warnings
from collections.abc import Callable
from typing import Literal, assert_never, get_args

import pennylane as qml
import torch
import torch.nn as nn
from pennylane.exceptions import AllocationError, DeviceError

from hqnn_forge.circuits import hardware_efficient_layer, strongly_entangling_layer
from hqnn_forge.noise import NoiseMethod, Position, TrainingNoiseMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RotationAxis = Literal["X", "Y", "Z"]
DiffMethod = Literal["adjoint", "parameter-shift", "backprop", "finite-diff"]
DeviceName = Literal["lightning.gpu", "lightning.kokkos", "lightning.qubit", "default.qubit"]
Entangler = Literal["ring", "strongly_entangling", "brickwork", "hardware_efficient"]
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
    * ``"brickwork"``: nearest-neighbour CNOTs on the even pairs
      ``(0,1), (2,3), …``, then on the odd pairs ``(1,2), (3,4), …``, with
      no wrap-around, then ``Rot`` on every qubit.  Unlike the two cascades
      above, which carry a readout across the whole register at shallow
      depth (⟨Z_0⟩ ↦ Z_1⋯Z_{n-1} through one ring), each layer widens the
      backward light cone of a single-qubit readout by at most two qubits
      on each side, so the ⟨Z_i⟩ readouts stay local costs in the sense of
      Cerezo et al. (2021) while ``n_layers`` is small against ``n_qubits``;
      see :mod:`hqnn_forge.initializers.restricted_variance` for the
      measured gradient variance.
    * ``"hardware_efficient"``: a nearest-neighbour ``CZ(i, i+1)`` ladder,
      then ``RY(θ)`` on every qubit (Kandala et al. 2017):
      :func:`hqnn_forge.circuits.hardware_efficient_layer`.  One angle per
      qubit per layer and ``n − 1`` two-qubit gates per layer: a third of the
      parameters of the ``Rot`` blocks, with CZ native on many devices.

    ``"ring"``, ``"strongly_entangling"`` and ``"brickwork"`` take ``weights``
    of shape ``(n_layers, n_qubits, 3)`` and use ``n_layers · n_qubits``
    ``Rot`` gates.  The ring and ``"strongly_entangling"`` use ``n_qubits``
    CNOTs per layer and differ in gate order and, from the second layer on, in
    which qubits the CNOTs connect; ``"brickwork"`` uses ``n_qubits - 1``.
    ``"hardware_efficient"`` takes ``(n_layers, n_qubits)``.
    :func:`variational_weight_shape` gives the shape for each.  The ``"ring"``
    block is :func:`hqnn_forge.circuits.strongly_entangling_layer` applied per
    layer.

    ``layer_offset`` is the index of the first block within the whole ansatz,
    for circuits that interleave other gates between blocks and so apply them
    a few at a time: the ``"strongly_entangling"`` range of block ``ℓ`` is
    ``(layer_offset + ℓ) mod (n-1) + 1``, so applying the blocks one by one
    with offsets ``0 … L-1`` gives the same ranges as applying all ``L`` at
    once.  The ``"ring"``, ``"brickwork"`` and ``"hardware_efficient"``
    blocks do not depend on the layer index.
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
    _check_entangler(entangler)
    for layer in range(n_layers):
        if entangler == "ring":
            # CNOT ring (last qubit → first), then Rot(φ, θ, ω) on every qubit
            strongly_entangling_layer(weights[layer], n_qubits)
        elif entangler == "hardware_efficient":
            # CZ ladder, then RY(θ) on every qubit
            hardware_efficient_layer(weights[layer], n_qubits)
        elif entangler == "brickwork":
            # Brickwork: even nearest-neighbour pairs, then odd ones
            for start in (0, 1):
                for qubit in range(start, n_qubits - 1, 2):
                    qml.CNOT(wires=[qubit, qubit + 1])
            # Per-qubit SU(2) rotation block
            for qubit in range(n_qubits):
                qml.Rot(
                    weights[layer, qubit, 0],  # φ
                    weights[layer, qubit, 1],  # θ
                    weights[layer, qubit, 2],  # ω
                    wires=qubit,
                )
        else:
            assert_never(entangler)


def variational_weight_shape(
    entangler: Entangler, n_qubits: int, n_layers: int
) -> tuple[int, ...]:
    """
    Shape of the ``weights`` tensor :func:`apply_variational_layers` reads for ``entangler``.

    The one place an encoder's variational weight shape is defined: every
    encoding layer registers its ``weights`` with this shape.  Dim 0 is always
    the layer index, which :func:`~hqnn_forge.initializers.block_local_init_`
    and the diagnostics' ``n_layers`` fallback rely on: ``(n_layers, n_qubits,
    3)`` for the ``Rot`` blocks (``"ring"``, ``"strongly_entangling"``,
    ``"brickwork"``), ``(n_layers, n_qubits)`` for the ``RY`` of
    ``"hardware_efficient"``.

    Raises
    ------
    ValueError
        For an unknown ``entangler``.
    """
    _check_entangler(entangler)
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
    _check_entangler(entangler)
    if rotation is not None and rotation not in ("X", "Y", "Z"):
        raise ValueError(f"rotation must be 'X', 'Y' or 'Z'; got {rotation!r}.")


def _check_entangler(entangler: str) -> None:
    if entangler not in get_args(Entangler):
        raise ValueError(
            f"entangler must be one of {', '.join(map(repr, get_args(Entangler)))}; "
            f"got {entangler!r}."
        )


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


#: Backends that failed to initialise in this process, with the failure.  A
#: failed plugin import is not cached by Python, and a CUDA library load or a
#: GPU probe is slow, so every layer built with the same device_name would
#: otherwise repeat them -- and warn again.  Out-of-memory failures are never
#: recorded: they depend on n_qubits and are raised, not fallen back from.
_FAILED_BACKENDS: dict[str, BaseException] = {}

#: The hqnn_forge package directory, for attributing warnings to user code.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reset_device_fallback() -> None:
    """
    Forget the backends that failed to initialise, so the next layer tries
    them again -- e.g. after installing a plugin in a running session.
    """
    _FAILED_BACKENDS.clear()


def _stacklevel_outside_package() -> int:
    """
    ``stacklevel`` that attributes a warning issued by the caller of this
    function to the first frame outside ``hqnn_forge``: the user's own call,
    however deep the layer or classifier constructors that led here.
    (``warnings.warn(skip_file_prefixes=...)`` does this from Python 3.12 on;
    the floor is 3.11.)
    """
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None  # the function that warns
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        _PACKAGE_DIR + os.sep
    ):
        frame = frame.f_back
        level += 1
    return level


def _resolve_device(device_name: DeviceName, n_qubits: int) -> qml.devices.Device:
    """
    Create *device_name*, falling back along :data:`FALLBACK_CHAIN` when a
    backend is not installed or has no usable hardware.

    A backend that fails is remembered for the rest of the process: later
    layers skip it without trying again, and its ``RuntimeWarning`` is issued
    once, not once per layer (:func:`reset_device_fallback` forgets them).
    The warning is attributed to the first frame outside ``hqnn_forge`` --
    the user's call -- whichever layer or classifier constructor led here.

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
    last = len(candidates) - 1
    for attempt, name in enumerate(candidates):
        if attempt < last and name in _FAILED_BACKENDS:
            # Already failed and warned about in this process: go straight on.
            logger.debug("Skipping %s, which failed before: %r", name, _FAILED_BACKENDS[name])
            continue
        try:
            dev = qml.device(name, wires=n_qubits)
        except _DEVICE_FAILURES as exc:
            if attempt == last or _is_out_of_memory(exc):
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
                stacklevel=_stacklevel_outside_package(),
            )
            # Recorded only once warned: if a warnings-as-errors filter turns
            # the warning into an exception, the next build must try (and
            # raise) again rather than fall back silently.
            _FAILED_BACKENDS[name] = exc
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

        circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[ExpectationMP]

    where

    * ``inputs``  — shape ``(n_qubits,)`` — the pre-processed feature vector.
    * ``weights`` — shape :func:`variational_weight_shape`: ``(n_layers,
                    n_qubits, 3)``, the ``qml.Rot`` angles (φ, θ, ω) per layer
                    and qubit, or ``(n_layers, n_qubits)``, the ``RY`` angles,
                    for ``entangler="hardware_efficient"``.

    Called inside a QNode it records one ``qml.expval(PauliZ)`` measurement per
    readout wire; the QNode turns them into the expectation values.

    Circuit structure (per layer ℓ = 0 … L-1)
    ------------------------------------------
    1. **Feature embedding** (applied before the first layer only):
       ``AngleEmbedding(inputs, wires, rotation=rotation)``
       → RX(x_i) on wire i, ∀ i ∈ {0, …, n_qubits-1}.

    2. **CNOT entangling ring**:
       CNOT(i → i+1 mod n) for i ∈ {0, …, n_qubits-1}, applied as a cascade.
       This creates a cyclic entanglement graph with all-to-all reachability
       within a single layer.  The flip side is the backward light cone of
       the readouts: through one ring it covers qubits {0, …, i+1} for ⟨Z_i⟩
       with 0 < i < n-1, and all n qubits for ⟨Z_0⟩ (Z_0 ↦ Z_1⋯Z_{n-1} in the
       Heisenberg picture) and ⟨Z_{n-1}⟩; through two rings it covers all n
       qubits for every i.  From 2 layers on, the ⟨Z_i⟩ readouts therefore do
       not enjoy the local-cost gradient bounds of Cerezo et al. (2021); see
       `hqnn_forge.initializers` for what is measured instead.

       The same conjugation decides which *features* a readout sees.  After
       one layer with the default ``rotation="X"``, the X and Y terms the
       ``Rot`` mixes in land, for i < n-1, on operators with an ``X`` on some
       wire, whose expectation in the RX-embedded product state is 0, so only
       the Z image survives:

           ⟨Z_0⟩ = c_0(w) · cos x_1 ⋯ cos x_{n-1}      (no x_0)
           ⟨Z_i⟩ = c_i(w) · cos x_0 ⋯ cos x_i          (0 < i < n-1)

       For i = n-1 the wrap-around CNOT(n-1, 0) carries Y_{n-1} to
       ∝ Y_0 Y_1 Z_2 ⋯ Z_{n-2} Y_{n-1}, and ⟨Y⟩ = -sin x, so ⟨Z_{n-1}⟩ =
       a(w) · cos x_0 ⋯ cos x_{n-1} + b(w) · sin x_0 sin x_1 cos x_2 ⋯
       cos x_{n-2} sin x_{n-1}.

       Readout 0 is blind to its own feature, and carries a product of n-1
       cosines, which is small for inputs spread over (-π, π).  With
       ``rotation="Y"`` the X terms survive (⟨X⟩ = sin x) and ⟨Z_0⟩ does see
       x_0; with ``entangler="strongly_entangling"`` the image of Z_0 leaves
       wire 0 before its ``Rot`` is reached, so ⟨Z_0⟩ ignores x_0 under either
       rotation.  For both cascades, from two layers on every readout sees
       every feature under RX or RY.  (Under ``rotation="Z"`` no readout sees
       any feature at any depth: RZ on |0⟩ is only a phase, which is why
       :func:`build_encoding_qnode` refuses it, #212.)

       Under ``readout="all"`` the blind spot costs nothing, since readouts
       1 … n-1 together cover x_0.  Under ``readout="first"`` use
       ``n_layers >= 2`` with either cascade (#150).  ``entangler="brickwork"``
       is *not* a remedy there: its CNOT(0, 1) has wire 0 as control, so Z_0
       keeps its own wire and each ⟨Z_i⟩ sees x_i after one layer, but the
       same narrow light cone leaves ⟨Z_0⟩ seeing only x_0 (RX) or x_0, x_1
       (RY), and at 5 qubits still missing x_2 … x_4 (RX) or x_4 (RY) after
       two layers.

    3. **Per-qubit SU(2) rotation block**:
       ``qml.Rot(φ, θ, ω, wires=i)`` applies Rz(ω)·Ry(θ)·Rz(φ), covering the
       full Bloch sphere.  This is the most expressive single-qubit gate.
       In the **last** layer the trailing Rz(ω) commutes with the ⟨Z⟩
       readouts, so with ``readout="all"`` those ``n_qubits`` angles can
       never change the output.  With ``readout="first"`` far more is dead:
       for this ring ansatz the last layer's ``Rot`` on every wire but 0
       acts after anything that reaches ⟨Z_0⟩, and so do some earlier ω.
       At 4 qubits and 2 layers that is 4 of 24 weights under ``"all"`` and
       12 under ``"first"``; ``circuit_summary`` reports the count as
       ``n_inert_params``.  With ``entangler="strongly_entangling"`` the
       last-layer ``Rot`` comes before that layer's CNOTs, which carry Z_0
       onto other wires (at 4 qubits, to wire 2, whose last-layer ``Rot`` is
       then live while wire 0's is dead), and the count under ``"first"`` is
       only a lower bound: 8 of the 12 weights autograd finds dead at 4
       qubits and 2 layers.  With ``entangler="brickwork"`` the narrow light
       cone of ⟨Z_0⟩ leaves more dead under ``"first"``: 16 of 24 at 4
       qubits and 2 layers (the last-layer ``Rot`` off wire 0, wire 0's ω,
       and layer 0's ``Rot`` on wires 2 and 3), 17 by autograd.  The weight
       tensor keeps its ``(n_layers, n_qubits, 3)`` shape for every
       ``Rot`` entangler; ``"hardware_efficient"`` has one ``RY`` angle per
       qubit, shape ``(n_layers, n_qubits)``.

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
        Pauli axis used by AngleEmbedding: ``"X"`` | ``"Y"``.
        :func:`build_encoding_qnode` refuses ``"Z"``, a phase on ``|0⟩``.
    entangler:
        ``"ring"`` (steps 2 and 3 above), ``"strongly_entangling"``
        (``qml.StronglyEntanglingLayers``: Rot first, then a CNOT ring of
        range ``ℓ mod (n-1) + 1``), ``"brickwork"`` (nearest-neighbour
        CNOT pairs, no wrap-around, then Rot) or ``"hardware_efficient"`` (a
        CZ ladder, then ``RY``; ``weights`` of shape ``(n_layers, n_qubits)``).
        See :func:`apply_variational_layers`.
        :func:`apply_variational_layers`.
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
        Pauli rotation axis for AngleEmbedding: ``"X"`` (default) or ``"Y"``.
        ``"Z"`` raises: a single ``RZ`` on ``|0⟩`` is only a phase, so the
        layer would not depend on its inputs.
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
        ``"ring"`` (default), ``"strongly_entangling"``, ``"brickwork"`` or
        ``"hardware_efficient"``; see :func:`apply_variational_layers`.
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
        above, or if ``rotation="Z"`` -- all checked here, before the circuit
        first runs.

    Examples
    --------
    >>> qnode = build_encoding_qnode(n_qubits=8, n_layers=2)
    >>> x = torch.rand(8)
    >>> w = torch.zeros(2, 8, 3)
    >>> result = qnode(x, w)  # list of 8 expectation values
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2 for the CNOT entangling ring; got {n_qubits}.")
    if rotation == "Z":
        # The inputs are embedded once, on |0…0⟩, where RZ(x) only multiplies
        # each wire by a phase: the state entering the ansatz is the same for
        # every x, so the layer would be a constant.  DataReuploadingLayer
        # can use "Z" from its second upload on.
        raise ValueError(
            'rotation="Z" would make the layer ignore its inputs: a single RZ embedding '
            "acts on |0…0⟩, where it is only a global phase, so every input gives the same "
            'state and the input gradients are zero.  Use "X" or "Y", or '
            'DataReuploadingLayer(rotation="Z", n_layers >= 2).'
        )

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


class QuantumEncodingLayer(TrainingNoiseMixin, nn.Module):
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

    ``(n_layers, n_qubits)`` for ``entangler="hardware_efficient"``; see
    :func:`variational_weight_shape`.

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
        At 1, ⟨Z_0⟩ does not see feature 0 under the default ring and RX,
        so ``readout="first"`` wants 2 or more; see step 2 of
        :func:`_make_angle_embedding_circuit`.
    rotation:
        Pauli axis for AngleEmbedding: ``"X"`` | ``"Y"``; ``"Z"`` raises, see
        :func:`build_encoding_qnode`.
    device_name:
        PennyLane device, one of :data:`DeviceName`.  An unavailable backend
        falls back along ``lightning.qubit → default.qubit`` with a warning
        per step.
    diff_method:
        Gradient method.  Use ``"adjoint"`` with ``lightning.qubit`` for
        exact, efficient gradients during state-vector simulation.
    entangler:
        ``"ring"`` (default), ``"strongly_entangling"``, ``"brickwork"`` (the
        same parameter count) or ``"hardware_efficient"`` (a third of it: one
        ``RY`` angle per qubit per layer); see :func:`apply_variational_layers`.
    readout:
        ``"all"`` (default): the layer returns ``(batch, n_qubits)``.
        ``"first"``: ⟨Z_0⟩ only, ``(batch, 1)``, the published SHNN readout.
    noise_level:
        Depolarizing probability applied to the circuit in **train mode**,
        in ``[0, 0.75]``; ``0`` (default) is the plain noiseless layer.  With
        ``noise_level > 0`` the train-mode forward pass runs the circuit on
        ``default.mixed`` with a ``DepolarizingChannel`` inserted, so
        gradients are computed through the noisy circuit (noise-aware
        training); eval mode is always noiseless, like dropout.  With the
        default ``noise_method``, backprop keeps a ``batch × 4^n`` density
        matrix per operation, so this is practical up to about 6 qubits.  See
        :mod:`hqnn_forge.noise`.
    noise_position:
        ``"all"`` (after every gate, default) or ``"end"`` (before
        measurement), as in :func:`hqnn_forge.noise.apply_depolarizing_noise`.
    noise_method:
        ``"density"`` (default): the exact channel on ``default.mixed``.
        ``"trajectories"``: Pauli-trajectory sampling on this layer's own
        device and ``diff_method``, at pure-state memory; the train-mode
        output is then random, and equal to the ``"density"`` output on
        average.  See :mod:`hqnn_forge.noise`.
    noise_trajectories:
        Draws averaged per sample with ``noise_method="trajectories"``.
        Default 1; must be 1 for ``"density"``.

    Attributes
    ----------
    n_qubits : int
    n_features : int
        Width of the input, one feature per qubit: ``n_qubits``.
    n_layers : int
    n_outputs : int
        Width of the output: ``n_qubits`` or 1.
    entangler : str
    readout : str
    noise_level : float
    noise_position : str
    noise_method : str
    noise_trajectories : int
    qlayer : pennylane.qnn.TorchLayer
        The underlying differentiable quantum layer.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.encoding import QuantumEncodingLayer
    >>> from hqnn_forge.initializers import restricted_normal_init_
    >>>
    >>> layer = QuantumEncodingLayer(n_qubits=8, n_layers=2)
    >>> _ = restricted_normal_init_(layer.qlayer.weights, n_qubits=8, n_layers=2)
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
        noise_method: NoiseMethod = "density",
        noise_trajectories: int = 1,
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_features = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        self.readout = readout
        self.n_outputs = len(readout_wires(n_qubits, readout))

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
        # Shape: variational_weight_shape(entangler, ...)
        #   dim-0: layer index ℓ ∈ {0, …, n_layers-1}
        #   dim-1: qubit  index i ∈ {0, …, n_qubits-1}
        #   dim-2: Euler angles (φ, θ, ω) for qml.Rot; absent for
        #          "hardware_efficient", whose RY takes one angle
        weight_shapes: dict[str, tuple[int, ...]] = {
            "weights": variational_weight_shape(entangler, n_qubits, n_layers),
        }

        # Wrap QNode as an nn.Module with registered Parameters ───────────
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Training-time depolarizing noise (see hqnn_forge.noise) ─────────
        self._init_training_noise(
            qnode, n_qubits, noise_level, noise_position, noise_method, noise_trajectories
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
        return self._run_circuit(x)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        options = ""
        if self.entangler != "ring":
            options += f", entangler={self.entangler!r}"
        if self.readout != "all":
            options += f", readout={self.readout!r}"
        options += self._noise_repr()
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_params={sum(p.numel() for p in self.parameters())}{options}"
        )


# ---------------------------------------------------------------------------
# Re-export convenience alias
# ---------------------------------------------------------------------------
AngleEmbeddingQNode = build_encoding_qnode
"""Alias: ``build_encoding_qnode`` — returns the raw QNode without an nn.Module wrapper."""
