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

``p = 0`` is a no-op: the original QNode stays in place, so the output is
bit-identical to the noiseless model rather than merely close to it.

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
    if not isinstance(qlayer, qml.qnn.TorchLayer) or not isinstance(n_qubits, int):
        raise TypeError(
            f"apply_depolarizing_noise expects an encoding layer or a hybrid classifier "
            f"with a quantum_layer attribute; got {type(target).__name__}."
        )
    return qlayer, n_qubits


def _noisy_qnode(qnode: qml.QNode, n_qubits: int, p: float, position: Position) -> qml.QNode:
    device = qml.device("default.mixed", wires=n_qubits)
    base = qml.QNode(qnode.func, device, diff_method="backprop", interface="torch")
    return qml.noise.insert(base, qml.DepolarizingChannel, p, position=position)


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
    if not 0.0 <= p <= MAX_P:
        raise ValueError(f"p must lie in [0, {MAX_P}]; got {p}.")
    if position not in ("all", "end"):
        raise ValueError(f"position must be 'all' or 'end'; got {position!r}.")
    qlayer, n_qubits = _resolve_qlayer(model)
    if getattr(qlayer, "_hqnn_noise_original", None) is not None:
        raise RuntimeError("apply_depolarizing_noise cannot be nested on the same layer.")
    if p == 0.0:
        yield model
        return

    original = qlayer.qnode
    qlayer._hqnn_noise_original = original
    qlayer.qnode = _noisy_qnode(original, n_qubits, p, position)
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
        Depolarizing probabilities, each in [0, 0.75].
    position:
        Passed to :func:`apply_depolarizing_noise`.
    y, score_fn:
        If both are given, ``score_fn(y, probabilities)`` is recorded per level,
        e.g. ``lambda y, p: find_optimal_threshold(y, p).score``.

    Returns
    -------
    list[NoiseSweepPoint]
        In the order of ``ps``.
    """
    if (y is None) != (score_fn is None):
        raise ValueError("pass both y and score_fn, or neither.")
    predict = getattr(model, "predict_proba", None)
    if not callable(predict):
        raise TypeError(f"noise_sweep needs a model with predict_proba; got {type(model).__name__}.")
    points = []
    for p in ps:
        with apply_depolarizing_noise(model, p, position=position):
            probs = predict(X)
        score = float(score_fn(y, probs)) if score_fn is not None and y is not None else None
        points.append(NoiseSweepPoint(float(p), probs, score))
    return points
