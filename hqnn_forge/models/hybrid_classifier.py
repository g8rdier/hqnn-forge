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
    PennyLane device.
diff_method:
    Gradient computation method.
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
    QuantumEncodingLayer,
    Readout,
    RotationAxis,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers.restricted_variance import (
    block_local_init_,
    restricted_normal_init_,
)
from hqnn_forge.models.base import BinaryClassifierBase, custom_encoder
from hqnn_forge.noise import Position
from hqnn_forge.utils.rng import as_seed, seeded_rng

#: Constructor arguments of the SHNN published in the thesis (see
#: ``HybridBinaryClassifier.published_shnn``).
#: Constructor defaults that mark an option as "not asked for".  Both options
#: below are inert outside the configuration that uses them, so a non-default
#: value there is a mistake worth naming rather than a setting to record and
#: ignore.
_DEFAULT_ENCODER_ACTIVATION = "tanh"
_DEFAULT_INIT_STD = 0.1

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


class HybridBinaryClassifier(BinaryClassifierBase):
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
        VQC ansatz layers.  Default: 2.  At 1, with angle encoding and
        ``readout="first"``, ⟨Z_0⟩ misses the first encoded angle under the
        default ring and RX embedding, and most of them under
        ``entangler="brickwork"``; use 2 or more with a ring.  See step 2 of
        :func:`hqnn_forge.encoding.angle_embedding._make_angle_embedding_circuit`.
    use_classical_encoder:
        Prepend ``Linear(n_input_features → n_qubits) + Tanh``.  Default: True.
        If ``False``, input must already lie in (-π, π); it is not rescaled.
    dropout_p:
        Dropout probability applied after the quantum layer.  Default: 0.0.
    device_name:
        PennyLane device string, one of ``"lightning.gpu"``, ``"lightning.kokkos"``,
        ``"lightning.qubit"`` or ``"default.qubit"``; any other name raises
        ``ValueError``.  Default: ``"lightning.qubit"``.  A backend that cannot be
        initialised falls back along ``lightning.qubit → default.qubit`` with a warning.
    diff_method:
        Gradient method: ``"adjoint"``, ``"parameter-shift"``, ``"backprop"`` or
        ``"finite-diff"``.  Default: ``"adjoint"``.
    init_strategy:
        ``"restricted"`` (default), ``"block_local"``, or ``"normal"``
        (``N(0, init_std²)``, the published SHNN's init).
    encoding_type:
        Type of quantum embedding to use: ``"angle"`` or ``"iqp"``. Default: ``"angle"``.
    embedding_rotation:
        Pauli axis of the angle embedding, ``"X"`` (default), ``"Y"`` or ``"Z"``.
        Angle encoding only.
    entangler:
        ``"ring"`` (default: CNOT ring then ``Rot``), ``"strongly_entangling"``
        (``qml.StronglyEntanglingLayers``: ``Rot`` then a CNOT ring of growing
        range) or ``"brickwork"`` (nearest-neighbour CNOT pairs, so each ⟨Z_i⟩
        keeps a local light cone at shallow depth).  See
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
        only, on ``default.mixed`` with backprop, whose memory grows as
        ``batch × 4^n_qubits`` per operation: practical up to about 6 qubits.
        See :mod:`hqnn_forge.noise`.
    noise_position:
        ``"all"`` (default) or ``"end"``; where the channel is inserted.
    init_seed:
        Seed for weight initialisation.  ``None`` (default) draws the initial
        weights from the global torch RNG; an int draws them from a private RNG
        seeded with it, so the same seed gives the same weights and the global
        RNG is left exactly as it was.
    classical_encoder:
        Your own module in place of the built-in ``Linear(n_input_features →
        n_qubits)``, trained together with the quantum layer: a small MLP,
        or a CNN or sequence model that reshapes the flat
        ``(batch, n_input_features)`` input itself.  It must return
        ``(batch, n_qubits)``, which is checked here with one forward pass.
        The model owns the angle range: it applies ``encoder_activation`` and
        the factor π on top of the module, exactly as for the built-in
        encoder, so the module should output unbounded features and not end
        in ``Tanh`` or ``Sigmoid`` (that warns).  The module is used as given
        and never re-initialised, so pretrained weights are kept.  Requires
        ``use_classical_encoder=True``.  ``save_checkpoint`` refuses a model
        with a custom encoder; save its ``state_dict`` instead.  Default:
        ``None``, the built-in encoder.

    Attributes
    ----------
    classical_encoder : nn.Sequential or nn.Identity
        ``Sequential(Linear, activation)``, ``Sequential(custom module,
        activation)`` with a custom ``classical_encoder``, or ``Identity``.
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
        device_name: DeviceName = "lightning.qubit",
        diff_method: DiffMethod = "adjoint",
        init_strategy: str = "restricted",
        encoding_type: str = "angle",
        embedding_rotation: RotationAxis = "X",
        entangler: Entangler = "ring",
        readout: Readout = "all",
        encoder_activation: str = "tanh",
        init_std: float = 0.1,
        noise_level: float = 0.0,
        noise_position: Position = "all",
        init_seed: int | None = None,
        classical_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        init_seed = as_seed(init_seed)
        # Building the layers draws from the global RNG (nn.Linear and
        # TorchLayer defaults), all of it overwritten by _initialise_weights.
        # With init_seed the whole build runs inside seeded_rng, so the
        # caller's stream is exactly where it was afterwards -- also when a
        # check below raises after some layers were built.
        with seeded_rng(init_seed) as reseed:
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
                init_seed=init_seed,
                classical_encoder=classical_encoder,
            )

            if encoder_activation not in ("tanh", "sigmoid"):
                raise ValueError(
                    f"encoder_activation must be 'tanh' or 'sigmoid'; got {encoder_activation!r}."
                )
            if init_strategy not in ("restricted", "block_local", "normal"):
                raise ValueError(
                    f"init_strategy must be 'restricted', 'block_local' or 'normal'; "
                    f"got {init_strategy!r}."
                )
            if init_std <= 0.0:
                raise ValueError(f"init_std must be > 0; got {init_std}.")
            if init_strategy != "normal" and init_std != _DEFAULT_INIT_STD:
                raise ValueError(
                    f"init_std applies to init_strategy='normal' only; "
                    f"'{init_strategy}' derives its own sigma from the circuit size, so "
                    f"init_std={init_std} would be recorded in the config and ignored."
                )

            self.n_input_features = n_input_features
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.init_strategy = init_strategy
            self.use_classical_encoder = use_classical_encoder
            self.encoder_activation = encoder_activation
            self.init_std = init_std

            # ── Classical encoder ─────────────────────────────────────────────
            if classical_encoder is not None:
                if not use_classical_encoder:
                    raise ValueError(
                        "classical_encoder replaces the built-in encoder and needs "
                        "use_classical_encoder=True; use_classical_encoder=False feeds the "
                        "input to the circuit directly, with no encoder at all."
                    )
                self.classical_encoder: nn.Module = custom_encoder(
                    classical_encoder, n_input_features, n_qubits, encoder_activation
                )
            elif use_classical_encoder:
                self.classical_encoder = nn.Sequential(
                    nn.Linear(n_input_features, n_qubits),
                    nn.Tanh() if encoder_activation == "tanh" else nn.Sigmoid(),
                )
            else:
                if n_input_features != n_qubits:
                    raise ValueError(
                        f"When use_classical_encoder=False, n_input_features "
                        f"({n_input_features}) must equal n_qubits ({n_qubits})."
                    )
                if encoder_activation != _DEFAULT_ENCODER_ACTIVATION:
                    raise ValueError(
                        f"encoder_activation applies with use_classical_encoder=True only; "
                        f"without the encoder the features enter the circuit as given, so "
                        f"{encoder_activation!r} would be recorded in the config and ignored."
                    )
                self.classical_encoder = nn.Identity()

            # ── Quantum encoding layer ────────────────────────────────────────
            if encoding_type == "angle":
                self.quantum_layer: QuantumEncodingLayer | IQPEncodingLayer = QuantumEncodingLayer(
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    rotation=embedding_rotation,
                    device_name=device_name,
                    diff_method=diff_method,
                    entangler=entangler,
                    readout=readout,
                    noise_level=noise_level,
                    noise_position=noise_position,
                )
            elif encoding_type == "iqp":
                if embedding_rotation != "X":
                    raise ValueError(
                        "embedding_rotation applies to encoding_type='angle' only; IQP embedding "
                        "has no rotation axis."
                    )
                self.quantum_layer = IQPEncodingLayer(
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    n_repeats=1,
                    device_name=device_name,
                    diff_method=diff_method,
                    entangler=entangler,
                    readout=readout,
                    noise_level=noise_level,
                    noise_position=noise_position,
                )
            else:
                raise ValueError(f"Unsupported encoding_type: {encoding_type}")
            n_readouts = self.quantum_layer.n_outputs

            # ── Dropout ───────────────────────────────────────────────────────
            self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

            # ── Classical head ────────────────────────────────────────────────
            self.head = nn.Linear(n_readouts, 1)

            # ── Small-angle restricted-variance initialisation ─────────────────
            # A custom encoder is left as given (it may be pretrained), so its
            # submodules are excluded from the classical init below.
            self._custom_encoder_ids: frozenset[int] = frozenset(
                map(id, classical_encoder.modules()) if classical_encoder is not None else ()
            )
            reseed()
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

        **Live parameters: 102 of the 122.**  With the ⟨Z_0⟩ readout, 20 of the
        48 quantum weights can never move the output, for any input or weight
        values: the whole last-layer ``Rot`` on wires 0, 1, 3, 5 and 7, the
        last-layer ``ω`` on wires 2, 4 and 6, and the first-layer ``ω`` on
        wires 0 and 1 (measured with autograd and pinned in
        ``tests/test_published_shnn_parity.py``).  ``circuit_summary`` reports
        16 inert, a lower bound: backwards through the last layer's range-2
        CNOTs, ``Z_0`` becomes ``Z_2 Z_4 Z_6``, a cancellation a per-wire
        analysis cannot see.

        The dead weights are **kept** (#234): the ``(2, 8, 3)`` weight shape is
        the published one, so the model and its checkpoints stay
        reproducible, and both counts are reported instead.  Parameter
        efficiency figures use the total, as the published results do (MCC
        0.5758 over 0.122 kParam = 4.72); over the 102 live parameters the same
        MCC is 5.65 per kParam.  :func:`hqnn_forge.evaluation.parameter_efficiency`
        takes an integer count for that.

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
        """Apply restricted-variance init to quantum weights; Xavier to classical."""
        # Classical encoder: Xavier uniform (standard for linear + Tanh)
        for module in self.modules():
            if isinstance(module, nn.Linear) and id(module) not in self._custom_encoder_ids:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Quantum weights: small-angle restricted-variance init
        weights = self.quantum_layer.qlayer.weights  # shape (n_layers, n_qubits, 3)
        if self.init_strategy == "block_local":
            block_local_init_(weights.data, n_qubits=self.n_qubits)
        elif self.init_strategy == "normal":
            # The published SHNN's init: N(0, init_std²), independent of size.
            with torch.no_grad():
                weights.normal_(mean=0.0, std=self.init_std)
        else:
            restricted_normal_init_(weights.data, n_qubits=self.n_qubits, n_layers=self.n_layers)

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
        # Classical projection + activation
        x = self.classical_encoder(x)  # (B, n_qubits)

        # Tanh output (-1, 1) → (-π, π), or sigmoid output (0, 1) → (0, π).
        # Bypassed input is already in (-π, π); scaling it again would alias
        # angles mod 2π.
        if self.use_classical_encoder:
            x = x * torch.pi

        # Quantum feature map
        x = self.quantum_layer(x)  # (B, n_qubits), values ∈ [-1, 1]

        # Regularisation
        x = self.dropout(x)

        # Classification head
        return self.head(x)  # (B, 1)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n_input_features={self.n_input_features}, "
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"total_params={self.count_parameters()}"
        )
