"""
hqnn_forge.models.hybrid_classifier
======================================
Full Hybrid Quantum-Classical Binary Classifier.

Architecture
------------
::

    Input (batch, n_input_features)
         │
         ▼
    [Classical encoder]  nn.Linear(n_input_features → n_qubits) + Tanh
         │
         ▼
    [Quantum encoding]   QuantumEncodingLayer(n_qubits, n_layers)
         │                  ├─ AngleEmbedding (RX)
         │                  └─ Strongly-entangling VQC (CNOT ring + Rot)
         │                  → ⟨Z_i⟩, shape (batch, n_qubits)
         ▼
    [Classical head]     nn.Linear(n_qubits → 1)
         │
         ▼
    Raw logit (batch, 1)   ← apply sigmoid for probability

Design Notes
------------
* The classical encoder projects arbitrary-width input to ``n_qubits`` dims
  and applies ``tanh`` to soft-clip values into (-1, 1), which ``forward`` then
  scales by π into (-π, π).  If ``PCANormalizer`` with ``scale_to_pi=True`` is
  used upstream, set ``use_classical_encoder=False`` to skip both steps: bypassed
  input reaches the circuit unscaled, so it must already lie in (-π, π).

* The quantum layer is initialised with ``restricted_normal_init_`` immediately
  after construction.  With the ``tanh(·)·π`` inputs this model feeds the
  circuit, that gives no measurable gain in initial gradient variance over
  uniform init, and it is not barren-plateau immunity: see
  :mod:`hqnn_forge.initializers` for the measured behaviour.

* The model exposes ``predict_proba(x)`` for inference (applies sigmoid).

Parameters
----------
n_input_features:
    Dimensionality of the raw / PCA-reduced input.
n_qubits:
    Number of qubits.  Must equal the output size of the classical encoder.
n_layers:
    Number of variational layers in the quantum circuit.
use_classical_encoder:
    If ``True`` (default), prepend a ``Linear + Tanh`` to project input to
    ``n_qubits`` dims.  Set ``False`` if input is already n_qubits-dim and
    already in (-π, π); it is then passed to the circuit unscaled.
device_name:
    PennyLane device name.  Default ``"auto"``: ``default.qubit`` up to
    12 qubits, ``lightning.qubit`` above (see
    :func:`~hqnn_forge.encoding.resolve_backend`).  The four simulators
    fall back along ``lightning.qubit → default.qubit`` with a warning
    when a backend cannot be initialised; any other name (a plugin or
    hardware) is used as given.
diff_method:
    ``"auto"`` (default) picks by device: ``"backprop"`` on
    ``default.qubit``, ``"adjoint"`` on lightning, ``"parameter-shift"``
    with ``shots`` or on any other device.  Or one of ``"adjoint"``,
    ``"parameter-shift"``, ``"backprop"``, ``"finite-diff"``.
init_strategy:
    ``"restricted"`` (default) — global restricted-normal init.
    ``"block_local"``           — per-layer decreasing variance.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import (
    DeviceName,
    DiffMethod,
    Entangler,
    Readout,
    RotationAxis,
)
from hqnn_forge.models._trunk import DEFAULT_ENCODER_ACTIVATION, DEFAULT_INIT_STD, QuantumTrunk
from hqnn_forge.models.base import BinaryClassifierBase
from hqnn_forge.noise import NoiseMethod, Position

#: Constructor arguments of the SHNN published in the thesis (see
#: ``HybridBinaryClassifier.published_shnn``).
_PUBLISHED_SHNN: dict[str, object] = {
    "n_input_features": 8,
    "n_qubits": 8,
    "n_layers": 2,
    "embedding_rotation": "Y",
    "entangler": "strongly_entangling",
    "readout": "first",
    "encoder_activation": "sigmoid",
    "init_strategy": "normal",
    "init_std": 0.1,
}


class HybridBinaryClassifier(QuantumTrunk, BinaryClassifierBase):
    """
    Hybrid quantum-classical binary classifier.

    See module docstring for architecture overview.

    Parameters
    ----------
    n_input_features:
        Number of raw (or PCA-reduced) input features.  Default: 8.
    n_qubits:
        Number of qubits in the quantum encoding layer.  Default: 8.
    n_layers:
        VQC ansatz layers.  Default: 2.
    use_classical_encoder:
        Prepend ``Linear(n_input_features → n_qubits) + Tanh``.  Default: True.
        If ``False``, input must already lie in (-π, π); it is not rescaled.
    dropout_p:
        Dropout probability applied after the quantum layer.  Default: 0.0.
    device_name:
        PennyLane device name.  Default ``"auto"``: ``default.qubit`` up to
        12 qubits, ``lightning.qubit`` above (see
        :func:`~hqnn_forge.encoding.resolve_backend`).  The four simulators
        fall back along ``lightning.qubit → default.qubit`` with a warning
        when a backend cannot be initialised; any other name (a plugin or
        hardware) is used as given.
    diff_method:
        ``"auto"`` (default) picks by device: ``"backprop"`` on
        ``default.qubit``, ``"adjoint"`` on lightning, ``"parameter-shift"``
        with ``shots`` or on any other device.  Or one of ``"adjoint"``,
        ``"parameter-shift"``, ``"backprop"``, ``"finite-diff"``.
    init_strategy:
        ``"restricted"`` (default), ``"block_local"``, or ``"normal"``
        (``N(0, init_std²)``, the published SHNN's init).
    encoding_type:
        The quantum embedding.  Default: ``"angle"``.

        * ``"angle"``: one rotation per feature
          (:class:`~hqnn_forge.encoding.QuantumEncodingLayer`).
        * ``"iqp"``: Hadamards, ``RZ(x_i)`` and pairwise ``x_i x_j`` phases
          (:class:`~hqnn_forge.encoding.iqp_embedding.IQPEncodingLayer`).
        * ``"reuploading"``: the angle embedding repeated before every
          variational layer (:class:`~hqnn_forge.encoding.DataReuploadingLayer`),
          optionally with ``trainable_input_scaling``.
        * ``"amplitude"``: the features as the ``2**n_qubits`` amplitudes of
          the state (:class:`~hqnn_forge.encoding.AmplitudeEncodingLayer`).  The
          classical encoder then maps to ``2**n_qubits`` features, and the
          ``·π`` scaling is irrelevant because the layer normalises.  Its
          input gradient is only correct under backprop, so with a classical
          encoder it requires ``diff_method="backprop"`` (on
          ``default.qubit``) and raises otherwise.  Without one, 1 to
          ``2**n_qubits`` raw features are zero-padded.
    embedding_rotation:
        Pauli axis of the angle embedding, ``"X"`` (default), ``"Y"`` or ``"Z"``.
        Angle and re-uploading encodings only.
    entangler:
        ``"ring"`` (default: CNOT ring then ``Rot``), ``"strongly_entangling"``
        (``qml.StronglyEntanglingLayers``: ``Rot`` then a CNOT ring of growing
        range) or ``"hardware_efficient"`` (a CZ ladder then ``RY``: a third of
        the circuit parameters).  See
        :func:`hqnn_forge.encoding.angle_embedding.apply_variational_layers`.
    readout:
        ``"all"`` (default): the head reads every ⟨Z_i⟩.  ``"first"``: ⟨Z_0⟩
        only, so the head is ``Linear(1 → 1)``.
    encoder_activation:
        ``"tanh"`` (default): encoder output ``tanh(·)·π`` in (-π, π).
        ``"sigmoid"``: ``π·sigmoid(·)`` in (0, π).  Requires
        ``use_classical_encoder=True``; there is no activation without an
        encoder, so a non-default value raises rather than being ignored.
    init_std:
        Standard deviation for ``init_strategy="normal"``.  Default: 0.1.
        Raises under the other strategies, which derive their own sigma.

    The published SHNN (thesis / ``hqnn-fraud-detection-benchmark``) is
    ``embedding_rotation="Y"``, ``entangler="strongly_entangling"``,
    ``readout="first"``, ``encoder_activation="sigmoid"``,
    ``init_strategy="normal"``; see :meth:`published_shnn`.
    noise_level:
        Training-time depolarizing probability for the quantum layer, in
        ``[0, 0.75]``.  Default: ``0.0`` (noiseless).  Applied in train mode
        only.  With the default ``noise_method`` it runs on ``default.mixed``
        with backprop, whose memory grows as ``batch × 4^n_qubits`` per
        operation: practical up to about 6 qubits.  See :mod:`hqnn_forge.noise`.
    noise_position:
        ``"all"`` (default) or ``"end"``; where the channel is inserted.
    noise_method:
        ``"density"`` (default, exact) or ``"trajectories"`` (Pauli-trajectory
        sampling on the layer's own device, at pure-state memory; equal to
        ``"density"`` on average).  See
        :class:`~hqnn_forge.encoding.QuantumEncodingLayer`.
    noise_trajectories:
        Draws averaged per sample with ``"trajectories"``.  Default: 1.
    trainable_input_scaling:
        With ``encoding_type="reuploading"`` only: a trainable per-upload
        scale on the features, initialised to 1.  Default: ``False``.
    shots:
        ``None`` (default): exact expectation values.  An ``int``: each circuit
        is sampled that many times, as on hardware, so predictions carry shot
        noise.  Requires ``diff_method="parameter-shift"`` (or
        ``"finite-diff"``); ``adjoint`` and ``backprop`` need the exact state.
        :func:`hqnn_forge.noise.apply_shots` evaluates a model with a finite
        shot count without rebuilding it.

    Attributes
    ----------
    classical_encoder : nn.Sequential or nn.Identity
    quantum_layer     : QuantumEncodingLayer
    dropout           : nn.Dropout
    head              : nn.Linear

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.models import HybridBinaryClassifier
    >>> model = HybridBinaryClassifier(n_input_features=30, n_qubits=8, n_layers=2)
    >>> x = torch.randn(4, 30)
    >>> logits = model(x)           # shape (4, 1)
    >>> probs  = model.predict_proba(x)  # shape (4,), values in [0, 1]
    """

    def __init__(
        self,
        n_input_features: int = 8,
        n_qubits: int = 8,
        n_layers: int = 2,
        *,
        use_classical_encoder: bool = True,
        dropout_p: float = 0.0,
        device_name: DeviceName = "auto",
        diff_method: DiffMethod = "auto",
        init_strategy: str = "restricted",
        encoding_type: str = "angle",
        embedding_rotation: RotationAxis = "X",
        entangler: Entangler = "ring",
        readout: Readout = "all",
        encoder_activation: str = DEFAULT_ENCODER_ACTIVATION,
        init_std: float = DEFAULT_INIT_STD,
        noise_level: float = 0.0,
        noise_position: Position = "all",
        noise_method: NoiseMethod = "density",
        noise_trajectories: int = 1,
        trainable_input_scaling: bool = False,
        shots: int | None = None,
    ) -> None:
        super().__init__()
        self._config = dict(
            n_input_features=n_input_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            use_classical_encoder=use_classical_encoder,
            dropout_p=dropout_p,
            device_name=device_name,
            diff_method=diff_method,
            init_strategy=init_strategy,
            encoding_type=encoding_type,
            embedding_rotation=embedding_rotation,
            entangler=entangler,
            readout=readout,
            encoder_activation=encoder_activation,
            init_std=init_std,
            noise_level=noise_level,
            noise_position=noise_position,
            noise_method=noise_method,
            noise_trajectories=noise_trajectories,
            trainable_input_scaling=trainable_input_scaling,
            shots=shots,
        )

        n_readouts = self._build_trunk(
            n_input_features=n_input_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            use_classical_encoder=use_classical_encoder,
            dropout_p=dropout_p,
            device_name=device_name,
            diff_method=diff_method,
            init_strategy=init_strategy,
            encoding_type=encoding_type,
            embedding_rotation=embedding_rotation,
            entangler=entangler,
            readout=readout,
            encoder_activation=encoder_activation,
            init_std=init_std,
            noise_level=noise_level,
            noise_position=noise_position,
            noise_method=noise_method,
            noise_trajectories=noise_trajectories,
            trainable_input_scaling=trainable_input_scaling,
            shots=shots,
        )

        # ── Classical head ────────────────────────────────────────────────
        self.head = nn.Linear(n_readouts, 1)

        # ── Small-angle restricted-variance initialisation ─────────────────
        self._initialise_weights()

    # ------------------------------------------------------------------
    @classmethod
    def published_shnn(cls, **overrides: object) -> HybridBinaryClassifier:
        """
        The SHNN configuration published in the thesis and in
        ``hqnn-fraud-detection-benchmark`` (``configs/default.yaml``, ``shnn``):
        8 qubits, 2 layers, ``Linear(8→8)`` + ``π·sigmoid``, RY angle embedding,
        ``StronglyEntanglingLayers``, ⟨Z_0⟩ readout, ``Linear(1→1)`` head,
        ``N(0, 0.1²)`` quantum init.  122 trainable parameters, 48 quantum.

        Parameters
        ----------
        **overrides:
            Constructor arguments to change, e.g. ``device_name`` or
            ``diff_method``; the structural options above can be overridden
            too, at which point the model is no longer the published one.
        """
        options: dict[str, object] = {**_PUBLISHED_SHNN, **overrides}
        return cls(**options)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    def _initialise_weights(self) -> None:
        """Xavier on the encoder and head; ``init_strategy`` on the quantum weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self._initialise_quantum_weights()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical encode → quantum encode → classification head.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.

        Returns
        -------
        torch.Tensor
            Raw logits, shape ``(batch_size, 1)``.  Apply ``torch.sigmoid``
            for probabilities, or pass directly to ``FocalLoss``.
        """
        x = self._quantum_features(x)  # (B, n_outputs), values ∈ [-1, 1]
        x = self.dropout(x)
        return self.head(x)  # (B, 1)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n_input_features={self.n_input_features}, "
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"total_params={self.count_parameters()}"
        )
