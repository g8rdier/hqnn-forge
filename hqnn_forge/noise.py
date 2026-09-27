"""
hqnn_forge.noise
================
Post-hoc depolarizing noise for evaluating a trained model's NISQ robustness.

A model trained on a noiseless simulator will run on hardware whose gates
depolarize.  :func:`apply_depolarizing_noise` re-executes the model's quantum
layer on ``default.mixed`` with a ``DepolarizingChannel`` of probability ``p``
inserted into the circuit, for the duration of a ``with`` block and without
touching the weights; :func:`noise_sweep` repeats that over a range of ``p``
and collects predictions (and a score, if asked).

Channel
-------
``qml.DepolarizingChannel(p)`` maps ρ → (1−p)ρ + p/3 (XρX + YρY + ZρZ), so on
its own it scales every Pauli expectation by ``1 − 4p/3``; ``p = 3/4`` is fully
depolarizing, and ``p`` is restricted to [0, 3/4].

``position`` chooses where channels go, as in ``qml.noise.insert``:

* ``"all"`` (default) -- after every gate, on every wire the gate acts on: a
  simple gate-noise model whose effect grows with circuit size.
* ``"end"`` -- once on every wire before measurement: readout-style noise that
  damps each ⟨Z_i⟩ by exactly ``1 − 4p/3``.

``p = 0`` replaces nothing: the original QNode stays in place, so the output
is bit-identical to the noiseless model rather than merely close to it.  It
still counts as an active wrapper, so a layer's training-time noise (below) is
suppressed inside it just as for ``p > 0``, but it does not count as a
replaced QNode: a ``p > 0`` block may be opened inside it.

Training-time noise
-------------------
The encoding layers and hybrid classifiers also take ``noise_level`` and
``noise_position`` at construction.  With ``noise_level > 0`` the layer runs
the same noisy QNode (built by :func:`training_noise_qnode`) whenever it is
in **train mode**, so gradients are computed through the noisy circuit, and
the noiseless QNode in eval mode, like dropout.  Evaluation under noise is
then done with :func:`apply_depolarizing_noise` / :func:`noise_sweep`, which
take precedence over the training-time channel if both are active at once.
``noise_level=0`` (the default) leaves the layer exactly as before.

Two methods, chosen with ``noise_method``:

``"density"`` (default)
    The exact channel: mixed-state simulation on ``default.mixed``,
    differentiated with backprop.  It costs ``O(4^n)`` memory, and training
    keeps a ``batch × 4^n`` complex density matrix for every gate and every
    inserted channel for the backward pass.  At 8 qubits, batch 64 and
    ``position="all"``, one step measured +2.7 GB peak and 50 s, so this is
    practical up to about 6 qubits.
``"trajectories"``
    Pauli-trajectory (Monte Carlo) sampling on the layer's own device and
    differentiation method.  At every channel site each sample independently
    gets ``I`` with probability ``1 − p``, or ``X``, ``Y`` or ``Z`` with ``p/3``
    each.  That mixture *is* the depolarizing channel, so the output
    averaged over draws equals the ``"density"`` output, and so does the
    gradient: each step's loss gradient is an unbiased estimate of the one
    ``"density"`` computes, at pure-state cost.  The same step measured
    +34 MB and 0.8 s on ``lightning.qubit`` with adjoint (+18 MB and 0.3 s
    noiseless).  The price is gradient variance, as with dropout, which a
    fresh draw every forward pass resembles.  ``noise_trajectories = k``
    averages ``k`` draws per sample, at ``k`` times the cost, to reduce it.

    The Pauli at a site is applied as ``RZ(π·z)`` then ``RX(π·x)`` with bits
    ``(x, z)``: ``(0, 0)`` is ``I``, ``(1, 0)`` is ``X``, ``(0, 1)`` is ``Z``
    and ``(1, 1)`` is ``Y`` up to a global phase.  Every sample therefore
    runs the same gate sequence with different angles, so parameter
    broadcasting and the per-sample batch split of the adjoint path apply
    unchanged.  The sites are placed by ``qml.noise.insert`` itself, so they
    are exactly the sites the ``"density"`` path puts channels on.  The
    draws use torch's global RNG, like dropout, so ``torch.manual_seed``
    makes them reproducible.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Literal, NamedTuple

import pennylane as qml
import torch
from torch import nn

Position = Literal["all", "end"]
NoiseMethod = Literal["density", "trajectories"]
MAX_P = 0.75


def _resolve_qlayer(target: nn.Module) -> tuple[qml.qnn.TorchLayer, int]:
    layer = getattr(target, "quantum_layer", target)
    qlayer = getattr(layer, "qlayer", None)
    n_qubits = getattr(layer, "n_qubits", None)
    if (
        not isinstance(layer, nn.Module)
        or not isinstance(qlayer, qml.qnn.TorchLayer)
        or not isinstance(n_qubits, int)
    ):
        raise TypeError(
            f"apply_depolarizing_noise expects an encoding layer or a hybrid classifier "
            f"with a quantum_layer attribute; got {type(target).__name__}."
        )
    return qlayer, n_qubits


def validate_noise(
    p: float, position: str, *, p_name: str = "p", position_name: str = "position"
) -> None:
    """
    Raise ``ValueError`` unless ``0 <= p <= MAX_P`` and ``position`` is known.

    ``p_name`` / ``position_name`` are the argument names the messages use, so
    a caller that exposes them under other names (``noise_level``) is quoted.
    """
    if not 0.0 <= p <= MAX_P:
        raise ValueError(f"{p_name} must lie in [0, {MAX_P}]; got {p}.")
    if position not in ("all", "end"):
        raise ValueError(f"{position_name} must be 'all' or 'end'; got {position!r}.")


def _noisy_qnode(qnode: qml.QNode, n_qubits: int, p: float, position: Position) -> qml.QNode:
    device = qml.device("default.mixed", wires=n_qubits)
    base = qml.QNode(qnode.func, device, diff_method="backprop", interface="torch")
    return qml.noise.insert(base, qml.DepolarizingChannel, p, position=position)


def training_noise_qnode(
    qnode: qml.QNode, n_qubits: int, p: float, position: Position = "all"
) -> qml.QNode:
    """
    The noisy counterpart of ``qnode`` an encoding layer runs in train mode.

    Same construction as :func:`apply_depolarizing_noise` uses: the layer's
    circuit function on ``default.mixed`` with ``DepolarizingChannel(p)``
    inserted at ``position``, differentiated with backprop.  The layer's own
    ``device_name`` and ``diff_method`` apply to its noiseless path only;
    mixed-state simulation costs ``O(4^n)`` memory per sample, and training
    through it keeps one such state per operation for backprop (see the module
    docstring).

    Raises
    ------
    ValueError
        If ``p`` is outside ``(0, 0.75]`` or ``position`` is unknown.
    """
    validate_noise(p, position)
    if p == 0.0:
        raise ValueError("training_noise_qnode needs p > 0; p = 0 is the noiseless QNode itself.")
    return _noisy_qnode(qnode, n_qubits, p, position)


@qml.transform
def _pauli_trajectories(
    tape: qml.tape.QuantumScript, p: float, position: Position
) -> tuple[qml.tape.QuantumScriptBatch, Callable[..., object]]:
    """
    ``tape`` with one randomly drawn Pauli per sample at every channel site.

    The draw happens here, when the tape is built, so every forward pass
    draws afresh; the backward pass differentiates the tape that ran.
    """
    shape = () if tape.batch_size is None else (tape.batch_size,)
    n_draws = 1 if tape.batch_size is None else tape.batch_size
    weights = torch.tensor([1.0 - p, p / 3, p / 3, p / 3], dtype=torch.float64)

    def pauli_error(wires: object) -> None:
        # 0 = I, 1 = X, 2 = Y, 3 = Z; Y = i·X·Z, so Y sets both bits.
        u = torch.multinomial(weights, n_draws, replacement=True).reshape(shape)
        qml.RZ(torch.pi * ((u == 2) | (u == 3)).to(torch.float64), wires=wires)
        qml.RX(torch.pi * ((u == 1) | (u == 2)).to(torch.float64), wires=wires)

    return qml.noise.insert(tape, pauli_error, (), position=position)


def trajectory_noise_qnode(qnode: qml.QNode, p: float, position: Position = "all") -> qml.QNode:
    """
    ``qnode`` with depolarizing noise sampled as Pauli trajectories.

    Runs on ``qnode``'s own device and differentiation method; each call draws
    a fresh error pattern per sample (see the module docstring).  The average
    over draws equals :func:`training_noise_qnode`'s output.

    Raises
    ------
    ValueError
        If ``p`` is outside ``(0, 0.75]`` or ``position`` is unknown.
    """
    validate_noise(p, position)
    if p == 0.0:
        raise ValueError(
            "trajectory_noise_qnode needs p > 0; p = 0 is the noiseless QNode itself."
        )
    return _pauli_trajectories(qnode, p=p, position=position)


def run_with_training_noise(
    qlayer: qml.qnn.TorchLayer,
    noisy_qnode: qml.QNode,
    x: torch.Tensor,
    n_trajectories: int = 1,
) -> torch.Tensor:
    """
    Evaluate ``qlayer`` on ``x`` with ``noisy_qnode`` in place of its QNode.

    ``n_trajectories > 1`` (the ``"trajectories"`` method only) runs each
    sample that many times and returns the mean, each run with its own draw.

    If :func:`apply_depolarizing_noise` is active on the layer, its channel
    (none at ``p = 0``) is kept and the training-time one is not applied: the
    post-hoc wrapper is the evaluation instrument and wins.  The original
    QNode is restored afterwards, including when the forward pass raises.
    """
    if getattr(qlayer, "_hqnn_noise_depth", 0) > 0:
        return qlayer(x)
    original = qlayer.qnode
    qlayer.qnode = noisy_qnode
    try:
        if n_trajectories == 1:
            return qlayer(x)
        # Sample-major repeat: rows k·i … k·i + k − 1 are sample i's draws.
        batched = x.ndim > 1
        repeated = (
            x.repeat_interleave(n_trajectories, dim=0)
            if batched
            else x.expand(n_trajectories, *x.shape)
        )
        out = qlayer(repeated)
        out = out.reshape(-1, n_trajectories, *out.shape[1:]).mean(dim=1)
        return out if batched else out[0]
    finally:
        qlayer.qnode = original


def validate_noise_method(method: str, n_trajectories: object) -> None:
    """
    Raise ``ValueError`` unless ``method`` is known and ``n_trajectories`` fits it.

    ``n_trajectories`` must be a positive ``int`` (not ``bool``), and 1 for
    ``"density"``, which is exact and has nothing to average.
    """
    if method not in ("density", "trajectories"):
        raise ValueError(f"noise_method must be 'density' or 'trajectories'; got {method!r}.")
    if (
        isinstance(n_trajectories, bool)
        or not isinstance(n_trajectories, int)
        or n_trajectories < 1
    ):
        raise ValueError(f"noise_trajectories must be a positive int; got {n_trajectories!r}.")
    if method == "density" and n_trajectories != 1:
        raise ValueError(
            f"noise_trajectories={n_trajectories} needs noise_method='trajectories'; the "
            f"density method is exact and has nothing to average."
        )


MAX_TRAINING_NOISE_QUBITS = 6
"""Above this many qubits, a layer built with ``noise_method="density"`` warns."""


class TrainingNoiseMixin:
    """
    Training-time noise for an encoding layer: construction, dispatch and repr.

    The one implementation behind every layer's ``noise_level``,
    ``noise_position``, ``noise_method`` and ``noise_trajectories``.  A layer
    calls :meth:`_init_training_noise` once its QNode exists, runs its circuit
    through :meth:`_run_circuit` (train mode with noise: the noisy QNode, else
    ``qlayer`` itself) and appends :meth:`_noise_repr` to ``extra_repr``.
    Everything else -- validation at construction, the memory warning, the
    QNode swap and its restore, the precedence of
    :func:`apply_depolarizing_noise` -- happens here.
    """

    qlayer: qml.qnn.TorchLayer
    training: bool
    noise_level: float
    noise_position: Position
    noise_method: NoiseMethod
    noise_trajectories: int
    _training_noise_qnode: qml.QNode | None

    def _init_training_noise(
        self,
        qnode: qml.QNode,
        n_qubits: int,
        noise_level: float,
        noise_position: Position,
        noise_method: NoiseMethod,
        noise_trajectories: int,
    ) -> None:
        """
        Validate the options and build the train-mode QNode (``None`` without noise).

        Raises ``ValueError`` for an option out of range, whatever
        ``noise_level`` is, so a bad value does not wait for the day the noise
        is switched on.  Warns when the density method is asked for past
        :data:`MAX_TRAINING_NOISE_QUBITS`.
        """
        validate_noise(
            noise_level, noise_position, p_name="noise_level", position_name="noise_position"
        )
        validate_noise_method(noise_method, noise_trajectories)
        self.noise_level = noise_level
        self.noise_position = noise_position
        self.noise_method = noise_method
        self.noise_trajectories = noise_trajectories
        self._training_noise_qnode = None
        if noise_level == 0.0:
            return
        if noise_method == "trajectories":
            self._training_noise_qnode = trajectory_noise_qnode(qnode, noise_level, noise_position)
            return
        if n_qubits > MAX_TRAINING_NOISE_QUBITS:
            warnings.warn(
                f"noise_level > 0 trains on default.mixed, which keeps a batch × 4^n density "
                f"matrix per operation for backprop; at n_qubits={n_qubits} (practical limit "
                f"about {MAX_TRAINING_NOISE_QUBITS}) a training step may run out of memory.  "
                f"noise_method='trajectories' samples the same noise at pure-state cost; "
                f"see hqnn_forge.noise.",
                RuntimeWarning,
                stacklevel=3,
            )
        self._training_noise_qnode = training_noise_qnode(
            qnode, n_qubits, noise_level, noise_position
        )

    def _run_circuit(self, x: torch.Tensor) -> torch.Tensor:
        """``qlayer(x)``, through the noisy QNode in train mode when there is one."""
        if self.training and self._training_noise_qnode is not None:
            return run_with_training_noise(
                self.qlayer, self._training_noise_qnode, x, self.noise_trajectories
            )
        return self.qlayer(x)  # type: ignore[no-any-return]

    def _noise_repr(self) -> str:
        """The ``extra_repr`` fragment for the training noise; empty without it."""
        if not self.noise_level:
            return ""
        text = f", noise_level={self.noise_level}, noise_position={self.noise_position!r}"
        if self.noise_method != "density":
            text += (
                f", noise_method={self.noise_method!r}, "
                f"noise_trajectories={self.noise_trajectories}"
            )
        return text


@contextmanager
def apply_depolarizing_noise(
    model: nn.Module,
    p: float,
    *,
    position: Position = "all",
) -> Iterator[nn.Module]:
    """
    Run ``model``'s quantum layer with depolarizing noise inside the block.

    Parameters
    ----------
    model:
        A hybrid classifier or an encoding layer.
    p:
        Depolarizing probability per channel, in [0, 0.75].
    position:
        ``"all"`` or ``"end"``; see the module docstring.

    Yields
    ------
    nn.Module
        ``model`` itself, for convenience.

    Raises
    ------
    TypeError
        If ``model`` has no quantum layer.
    ValueError
        If ``p`` or ``position`` is out of range.
    RuntimeError
        If the layer is already running a replaced QNode (nested use).

    Notes
    -----
    The original QNode is restored on exit, including when the block raises.
    Gradients flow through the noisy circuit, so the block can also be used to
    fine-tune under noise.
    """
    validate_noise(p, position)
    qlayer, n_qubits = _resolve_qlayer(model)
    # Two separate markers.  _hqnn_noise_depth counts open blocks of any p and
    # is what tells run_with_training_noise to skip a layer's train-mode
    # channel.  _hqnn_noise_original is set only while a p > 0 block has
    # replaced the QNode, and is the nesting guard.  p = 0 touches only the
    # depth, so it neither raises inside a p > 0 block (whose channel stays in
    # charge) nor blocks a p > 0 block opened inside it.
    if p > 0.0 and getattr(qlayer, "_hqnn_noise_original", None) is not None:
        raise RuntimeError("apply_depolarizing_noise cannot be nested on the same layer.")
    original = qlayer.qnode
    # Build the replacement before touching the layer. default.mixed refuses
    # more than 23 wires, and a failure here has to leave the layer as it was:
    # arming the guard first would leave it armed with no block to disarm it,
    # and every later call on that layer would raise "cannot be nested".
    noisy = _noisy_qnode(original, n_qubits, p, position) if p > 0.0 else None
    qlayer._hqnn_noise_depth = getattr(qlayer, "_hqnn_noise_depth", 0) + 1
    if noisy is not None:
        qlayer._hqnn_noise_original = original
        qlayer.qnode = noisy
    try:
        yield model
    finally:
        if noisy is not None:
            qlayer.qnode = original
            qlayer._hqnn_noise_original = None
        qlayer._hqnn_noise_depth -= 1


class NoiseSweepPoint(NamedTuple):
    """One noise level of a sweep."""

    p: float
    probabilities: torch.Tensor
    score: float | None


def noise_sweep(
    model: nn.Module,
    X: torch.Tensor,
    ps: Iterable[float],
    *,
    position: Position = "all",
    y: torch.Tensor | None = None,
    score_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
) -> list[NoiseSweepPoint]:
    """
    Predict ``X`` at each noise level in ``ps``, without retraining.

    Parameters
    ----------
    model:
        A classifier with ``predict_proba`` (the hybrid classifiers).
    X:
        Inputs to evaluate.
    ps:
        Depolarizing probabilities, each in [0, 0.75]; consumed once.
    position:
        Passed to :func:`apply_depolarizing_noise`.
    y, score_fn:
        If both are given, ``score_fn(y, probabilities)`` is recorded per level,
        e.g. ``lambda y, p: find_optimal_threshold(y, p).score``.

    Returns
    -------
    list[NoiseSweepPoint]
        In the order of ``ps``.

    Raises
    ------
    ValueError
        If only one of ``y`` and ``score_fn`` is given, or if any level is out
        of range -- checked before the first evaluation, not as the sweep
        reaches it.
    TypeError
        If ``model`` has no ``predict_proba``.
    """
    if (y is None) != (score_fn is None):
        raise ValueError("pass both y and score_fn, or neither.")
    predict = getattr(model, "predict_proba", None)
    if not callable(predict):
        raise TypeError(
            f"noise_sweep needs a model with predict_proba; got {type(model).__name__}."
        )
    # Materialised and range-checked up front: the levels may arrive as a
    # generator, and a bad one at the end would otherwise be found only after
    # every earlier (O(4^n)) evaluation had already been paid for.
    levels = [float(p) for p in ps]
    invalid = [p for p in levels if not 0.0 <= p <= MAX_P]
    if invalid:
        raise ValueError(f"every p must lie in [0, {MAX_P}]; got {invalid}.")
    points = []
    for p in levels:
        with apply_depolarizing_noise(model, p, position=position):
            probs = predict(X)
        score = float(score_fn(y, probs)) if score_fn is not None and y is not None else None
        points.append(NoiseSweepPoint(float(p), probs, score))
    return points
