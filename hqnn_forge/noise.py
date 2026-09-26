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
suppressed inside it just as for ``p > 0``.

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

Mixed-state simulation costs ``O(4^n)`` memory and is differentiated with
backprop; it is intended for the library's qubit counts (≤ ~10).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Literal, NamedTuple

import pennylane as qml
import torch
from torch import nn

Position = Literal["all", "end"]
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


def validate_noise(p: float, position: str) -> None:
    """Raise ``ValueError`` unless ``0 <= p <= MAX_P`` and ``position`` is known."""
    if not 0.0 <= p <= MAX_P:
        raise ValueError(f"p must lie in [0, {MAX_P}]; got {p}.")
    if position not in ("all", "end"):
        raise ValueError(f"position must be 'all' or 'end'; got {position!r}.")


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
    mixed-state simulation costs ``O(4^n)`` memory.

    Raises
    ------
    ValueError
        If ``p`` is outside ``(0, 0.75]`` or ``position`` is unknown.
    """
    validate_noise(p, position)
    if p == 0.0:
        raise ValueError("training_noise_qnode needs p > 0; p = 0 is the noiseless QNode itself.")
    return _noisy_qnode(qnode, n_qubits, p, position)


def run_with_training_noise(
    qlayer: qml.qnn.TorchLayer, noisy_qnode: qml.QNode, x: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate ``qlayer`` on ``x`` with ``noisy_qnode`` in place of its QNode.

    If :func:`apply_depolarizing_noise` is active on the layer, its channel
    (none at ``p = 0``) is kept and the training-time one is not applied: the
    post-hoc wrapper is the evaluation instrument and wins.  The original
    QNode is restored afterwards, including when the forward pass raises.
    """
    if getattr(qlayer, "_hqnn_noise_original", None) is not None:
        return qlayer(x)
    original = qlayer.qnode
    qlayer.qnode = noisy_qnode
    try:
        return qlayer(x)
    finally:
        qlayer.qnode = original


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
    armed = getattr(qlayer, "_hqnn_noise_original", None) is not None
    original = qlayer.qnode
    if p == 0.0:
        # Replaces no QNode, so the output is bit-identical to the noiseless
        # model.  Checked before the nesting guard: inside a block that is
        # already open it must not raise, and leaves that block's channel in
        # charge.  Otherwise it still arms the guard, which is what tells
        # run_with_training_noise to skip a layer's train-mode channel.
        if armed:
            yield model
            return
        qlayer._hqnn_noise_original = original
        try:
            yield model
        finally:
            qlayer._hqnn_noise_original = None
        return
    if armed:
        raise RuntimeError("apply_depolarizing_noise cannot be nested on the same layer.")

    # Build the replacement before touching the layer. default.mixed refuses
    # more than 23 wires, and a failure here has to leave the layer as it was:
    # arming the guard first would leave it armed with no block to disarm it,
    # and every later call on that layer would raise "cannot be nested".
    noisy = _noisy_qnode(original, n_qubits, p, position)
    qlayer._hqnn_noise_original = original
    qlayer.qnode = noisy
    try:
        yield model
    finally:
        qlayer.qnode = original
        qlayer._hqnn_noise_original = None


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
