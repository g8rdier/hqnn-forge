"""
hqnn_forge.diagnostics.gradients
================================
Empirical gradient variance of an encoding layer, to see barren plateaus.

A barren plateau shows up as the variance of the cost gradient over random
parameter draws shrinking exponentially with the number of qubits.
:func:`gradient_variance` estimates that variance for one layer
configuration and one initialisation scheme; :func:`gradient_variance_sweep`
repeats it over qubit and layer counts so the trend is visible, and
:func:`format_sweep` prints the result as a table.

Estimator
---------
For each of ``n_samples`` draws the layer's weights are re-initialised with
``init``, an input is drawn uniformly from ``[-input_scale, input_scale]^n``,
the cost is evaluated for that single sample and its gradient with respect to
every quantum weight is recorded.  The sample variance is taken per weight;
``total_variance`` is its sum over weights (the variance of the gradient
vector, which does not shrink just because a larger circuit has more
parameters) and ``mean_variance`` its mean.  The default cost is ⟨Z_0⟩, a
local cost in the sense of Cerezo et al. (2021).

What the library's own circuits show
------------------------------------
Measured with this tool on ``QuantumEncodingLayer`` (2 layers, 100–200
samples, ``default.qubit``):

* ``total_variance`` falls by roughly 5x from 2 to 6 qubits under uniform
  init, even with the local cost at 2 layers.  The CNOT ring is a cascade, so
  the backward light cone of Z_0 covers every qubit within one layer and the
  "local" cost behaves like a global one.
* With inputs spread over (-π, π) -- what both classifiers produce, and what
  ``PCANormalizer(scale_to_pi=True)`` produces -- the restricted-variance init
  gives the same gradient variance as uniform init: the angle embedding
  already randomises the state.  Only with inputs near 0 does the restricted
  init retain more variance (about 1.3–1.8x at 8 qubits).

The layer's weights are restored when the estimate finishes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from hqnn_forge.initializers import block_local_init_, restricted_normal_init_

InitName = Literal["uniform", "restricted", "block_local"]
InitFn = Callable[[torch.Tensor], Any]
CostFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GradientVarianceResult:
    """
    Gradient-variance estimate for one layer configuration.

    Attributes
    ----------
    layer_type, n_qubits, n_layers:
        What was measured.
    init:
        Name of the initialisation scheme (``"custom"`` for a callable).
    input_scale:
        Inputs were drawn from ``[-input_scale, input_scale]``.
    n_samples:
        Number of random draws.
    total_variance:
        Sum over weights of the per-weight gradient variance.
    mean_variance:
        Mean over weights of the per-weight gradient variance.
    per_parameter:
        Per-weight variance, same shape as the layer's weight tensor.
    """

    layer_type: str
    n_qubits: int
    n_layers: int
    init: str
    input_scale: float
    n_samples: int
    total_variance: float
    mean_variance: float
    per_parameter: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        """Scalar fields only, for logging."""
        return {
            "layer_type": self.layer_type,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "init": self.init,
            "input_scale": self.input_scale,
            "n_samples": self.n_samples,
            "total_variance": self.total_variance,
            "mean_variance": self.mean_variance,
        }


def _resolve_weights(target: nn.Module) -> tuple[nn.Module, torch.Tensor, int, int]:
    """``(layer, weights, n_qubits, n_layers)`` for the encoding layer inside *target*."""
    layer = getattr(target, "quantum_layer", target)
    weights = getattr(getattr(layer, "qlayer", None), "weights", None)
    n_qubits = getattr(layer, "n_qubits", None)
    if (
        not isinstance(layer, nn.Module)
        or not isinstance(weights, torch.Tensor)
        or not isinstance(n_qubits, int)
    ):
        raise TypeError(
            f"gradient_variance expects an encoding layer or a hybrid classifier "
            f"with a quantum_layer attribute; got {type(target).__name__}."
        )
    n_layers = getattr(layer, "n_layers", None)
    return layer, weights, n_qubits, n_layers if isinstance(n_layers, int) else int(weights.shape[0])


def _make_init(init: InitName | InitFn, n_qubits: int, n_layers: int, generator: torch.Generator) -> tuple[str, InitFn]:
    if callable(init):
        return "custom", init
    if init == "uniform":
        return init, lambda w: w.copy_(torch.rand(w.shape, generator=generator, dtype=w.dtype) * 2 * math.pi)
    if init == "restricted":
        def restricted(w: torch.Tensor) -> None:
            with _seeded(generator):
                restricted_normal_init_(w, n_qubits=n_qubits, n_layers=n_layers)
        return init, restricted
    if init == "block_local":
        def block_local(w: torch.Tensor) -> None:
            with _seeded(generator):
                block_local_init_(w, n_qubits=n_qubits)
        return init, block_local
    raise ValueError(
        f"unknown init {init!r}; choose 'uniform', 'restricted', 'block_local' or pass a callable."
    )


class _seeded:
    """Run the library initialisers (which use the global RNG) from ``generator``."""

    def __init__(self, generator: torch.Generator) -> None:
        self.generator = generator

    def __enter__(self) -> None:
        self.saved = torch.get_rng_state()
        seed = int(torch.randint(0, 2**62, (1,), generator=self.generator))
        torch.manual_seed(seed)

    def __exit__(self, *exc: object) -> None:
        torch.set_rng_state(self.saved)


def _local_z0(outputs: torch.Tensor) -> torch.Tensor:
    return outputs[..., 0].sum()


def gradient_variance(
    target: nn.Module,
    n_samples: int = 100,
    *,
    init: InitName | InitFn = "uniform",
    input_scale: float = math.pi,
    cost_fn: CostFn | None = None,
    generator: torch.Generator | None = None,
) -> GradientVarianceResult:
    """
    Estimate the variance of the cost gradient over random weight draws.

    Parameters
    ----------
    target:
        An encoding layer, or a hybrid classifier (its ``quantum_layer`` is
        used and inputs are fed to it directly, bypassing the classical
        encoder).
    n_samples:
        Random draws.  At least 2.  Default: 100.
    init:
        ``"uniform"`` over [0, 2π) (the barren-plateau reference),
        ``"restricted"``, ``"block_local"``, or a callable that fills the
        weight tensor in place.
    input_scale:
        Inputs are uniform in ``[-input_scale, input_scale]``.  ``π`` matches
        what the classifiers feed the circuit; ``0`` feeds zeros.
    cost_fn:
        Maps the layer output of shape ``(1, n_qubits)`` to a scalar.
        Default: ⟨Z_0⟩.
    generator:
        Source of randomness for weights and inputs.  The global RNG state is
        left untouched either way.

    Returns
    -------
    GradientVarianceResult
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2 to estimate a variance; got {n_samples}.")
    if input_scale < 0:
        raise ValueError(f"input_scale must be >= 0; got {input_scale}.")
    layer, weights, n_qubits, n_layers = _resolve_weights(target)
    gen = generator if generator is not None else torch.Generator().manual_seed(0)
    init_name, init_fn = _make_init(init, n_qubits, n_layers, gen)
    cost = cost_fn if cost_fn is not None else _local_z0

    original = weights.detach().clone()
    original_grad = weights.grad
    grads = torch.empty((n_samples, *weights.shape), dtype=torch.float64)
    try:
        for s in range(n_samples):
            with torch.no_grad():
                init_fn(weights)
            x = (torch.rand(1, n_qubits, generator=gen) * 2 - 1) * input_scale
            weights.grad = None
            value = cost(layer(x))
            if value.ndim != 0:
                raise ValueError(f"cost_fn must return a scalar; got shape {tuple(value.shape)}.")
            (grad,) = torch.autograd.grad(value, weights)
            grads[s] = grad.detach().to(torch.float64)
    finally:
        with torch.no_grad():
            weights.copy_(original)
        weights.grad = original_grad

    per_parameter = grads.var(dim=0)
    return GradientVarianceResult(
        layer_type=type(layer).__name__,
        n_qubits=n_qubits,
        n_layers=n_layers,
        init=init_name,
        input_scale=float(input_scale),
        n_samples=n_samples,
        total_variance=float(per_parameter.sum()),
        mean_variance=float(per_parameter.mean()),
        per_parameter=per_parameter,
    )


def gradient_variance_sweep(
    build: Callable[[int, int], nn.Module],
    qubit_counts: Iterable[int],
    layer_counts: Iterable[int] = (2,),
    **kwargs: Any,
) -> list[GradientVarianceResult]:
    """
    :func:`gradient_variance` for every (n_qubits, n_layers) combination.

    Parameters
    ----------
    build:
        ``build(n_qubits, n_layers)`` returns a fresh layer or model, e.g.
        ``lambda q, l: QuantumEncodingLayer(q, l, device_name="default.qubit",
        diff_method="backprop")``.
    qubit_counts, layer_counts:
        Grid to sweep.
    **kwargs:
        Passed to :func:`gradient_variance` (``init``, ``n_samples``, ...).
        A ``generator`` is shared across the sweep.

    Returns
    -------
    list[GradientVarianceResult]
        One per combination, qubits in the outer loop.
    """
    layer_counts = list(layer_counts)
    return [
        gradient_variance(build(q, l), **kwargs)
        for q in qubit_counts
        for l in layer_counts
    ]


def format_sweep(results: Sequence[GradientVarianceResult]) -> str:
    """
    Plain-text table of a sweep, with the ratio to the previous row of the
    same init and layer count so exponential decay is readable at a glance.
    """
    header = f"{'init':<12} {'qubits':>6} {'layers':>6} {'total var':>12} {'mean var':>12} {'ratio':>7}"
    lines = [header, "-" * len(header)]
    previous: dict[tuple[str, int], float] = {}
    for r in results:
        key = (r.init, r.n_layers)
        prev = previous.get(key)
        ratio = f"{r.total_variance / prev:7.3f}" if prev else f"{'':>7}"
        previous[key] = r.total_variance
        lines.append(
            f"{r.init:<12} {r.n_qubits:>6} {r.n_layers:>6} "
            f"{r.total_variance:>12.4e} {r.mean_variance:>12.4e} {ratio}"
        )
    return "\n".join(lines)
