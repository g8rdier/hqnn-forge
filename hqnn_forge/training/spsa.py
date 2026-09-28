"""
hqnn_forge.training.spsa
========================
Simultaneous perturbation stochastic approximation (Spall 1992) as a
``torch.optim.Optimizer``, for training on shot-based devices and hardware.

With finite shots (#314) the only gradient methods left are the shift and
difference rules, which cost two circuit evaluations *per trainable
parameter* per sample: 96 per sample per step at the library defaults.  SPSA
estimates the whole gradient from two evaluations of the loss, whatever the
number of parameters.  Each step draws a random direction ``Δ`` with
independent ``±1`` entries and sets::

    ĝ_k = (L(θ + c_k Δ) − L(θ − c_k Δ)) / (2 c_k) · Δ        (Δ_i = 1 / Δ_i)
    θ   ← θ − a_k ĝ_k

with Spall's gain sequences ``a_k = lr / (k + 1 + stability)^alpha`` and
``c_k = perturbation / (k + 1)^gamma``.  Because ``E[Δ_i Δ_j] = δ_ij``, the
estimate is unbiased up to the finite-difference error in ``c_k`` (exact for a
quadratic loss), at the price of variance; the shrinking gains average it out.

When it pays
------------
SPSA's gradient is noisy, so it needs many more steps than Adam with exact
gradients: on a 2-qubit, 1-layer classifier (19 parameters), Adam reached a
loss of 0.35 in 60 steps and SPSA 0.52 in 500 and 0.49 in 1000.  What SPSA
saves is circuit evaluations per step.  Under ``parameter-shift`` that small
model costs 17 executions per sample per step (the forward pass plus two per
circuit weight and per circuit input the encoder needs a gradient for), so
60 Adam steps cost as much as about 500 SPSA steps, and Adam still wins.
The count grows with the circuit: at the library defaults (8 qubits,
2 layers: 48 weights, 8 inputs) it is 113 per sample per step, against SPSA's
2 whatever the size.  Use SPSA where each circuit evaluation is expensive and
the circuit has many parameters: on hardware and shot-based simulators, not on
small exact simulations.

Common random numbers
---------------------
Both loss evaluations of a step start from the same torch RNG state, so a
dropout mask or a trajectory-noise draw (``noise_method="trajectories"``) is the
same on both sides of the difference and cancels, instead of adding its own
variance to ``ĝ``.  Shot sampling uses each device's own NumPy generator,
which torch's state does not cover.  Pass ``model=`` and both evaluations
also start from the same state of every device generator behind the model,
so the two sides draw their samples from the same random numbers: for nearby
parameters the samples then move together and much of the shot noise cancels
in the difference.  On a two-qubit circuit at 100 shots this cut the variance
of ``L(θ + cΔ) − L(θ − cΔ)`` about sevenfold, on ``default.qubit`` and
``lightning.qubit`` alike.  It relies on PennyLane's simulators keeping their
generator as a ``numpy.random.Generator`` in the device's ``_rng`` attribute
(``default.qubit``, ``default.mixed`` and the lightning devices do); a device
without one, such as hardware, is left unsynchronised.

Usage
-----
The loss is evaluated by a closure, without ``backward``::

    opt = SPSA(model.parameters(), lr=0.2, perturbation=0.1, model=model)
    def closure():
        return loss_fn(model(x), y)
    opt.step(closure)

:func:`~hqnn_forge.training.train_model` recognises the optimiser and builds the
closure itself.  Optimise *all* parameters with SPSA: mixing it with a
gradient optimiser for the classical encoder does not save the circuit
evaluations, because the encoder's gradient runs through the circuit's input
gradient, which costs the same shift evaluations.

References
----------
* Spall (1992) "Multivariate stochastic approximation using a simultaneous
  perturbation gradient approximation", IEEE Trans. Autom. Control 37, 332.
* Spall (1998) "Implementation of the simultaneous perturbation algorithm for
  stochastic optimization", IEEE Trans. Aerosp. Electron. Syst. 34, 817
  (the gain exponents 0.602 and 0.101).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = ["SPSA", "device_generators"]

Closure = Callable[[], torch.Tensor | float]


def device_generators(model: nn.Module) -> list[np.random.Generator]:
    """
    The NumPy generators of the devices behind ``model``'s quantum layers.

    Each distinct generator once, in module order: the layer's QNode and its
    train-mode noise QNode share a device, and so a generator.  Devices that
    keep none (see *Common random numbers*) are skipped.
    """
    found: dict[int, np.random.Generator] = {}
    for module in model.modules():
        qlayer = getattr(module, "qlayer", None)
        for qnode in (
            getattr(qlayer, "qnode", None),
            getattr(module, "_training_noise_qnode", None),
        ):
            rng = getattr(getattr(qnode, "device", None), "_rng", None)
            if isinstance(rng, np.random.Generator):
                found.setdefault(id(rng), rng)
    return list(found.values())


class SPSA(torch.optim.Optimizer):
    """
    SPSA: two loss evaluations per step, independent of the parameter count.

    Parameters
    ----------
    params:
        Parameters or parameter groups, as for any ``torch.optim.Optimizer``.
        A group may override ``lr`` and ``perturbation``.
    lr:
        ``a``, the numerator of the step-size sequence.  Default: 0.1.  Unlike
        Adam's, it multiplies the raw gradient estimate, so the right value
        scales with the loss's gradient: for the BCE-trained 4-qubit model of
        ``examples/hardware_workflow.py`` (gradients of about 0.05), lr 0.1 to
        0.4 barely moved it in 30 epochs and lr 1 to 8 all trained it.
    perturbation:
        ``c``, the numerator of the perturbation-size sequence.  A value near
        the standard deviation of the loss's noise is Spall's guideline.
        Default: 0.1.
    alpha, gamma:
        Decay exponents of the two sequences; the defaults 0.602 and 0.101 are
        Spall's practical choices (asymptotically optimal: 1 and 1/6).
    stability:
        ``A`` in ``a_k``; about 10 % of the expected number of steps damps the
        first, largest steps.  Default: 0.
    generator:
        Source of the ``±1`` directions.  Default: a fresh generator seeded
        with 0, so runs are reproducible.
    model:
        The model the parameters belong to.  When given, the shot sampling of
        its devices is synchronised between the two evaluations of a step
        (see *Common random numbers*).  Default ``None``: only the torch RNG
        is.  Seed the devices (``seed=`` on the model) as well for runs that
        repeat exactly.

    Attributes
    ----------
    gradient_free : bool
        True: :func:`~hqnn_forge.training.train_model` calls ``step(closure)``
        without ``backward``.
    """

    gradient_free = True

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.1,
        perturbation: float = 0.1,
        *,
        alpha: float = 0.602,
        gamma: float = 0.101,
        stability: float = 0.0,
        generator: torch.Generator | None = None,
        model: nn.Module | None = None,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0; got {lr}.")
        if perturbation <= 0:
            raise ValueError(f"perturbation must be > 0; got {perturbation}.")
        if alpha <= 0 or gamma <= 0 or stability < 0:
            raise ValueError(
                f"alpha and gamma must be > 0 and stability ≥ 0; got {alpha}, {gamma}, {stability}."
            )
        defaults = {
            "lr": lr,
            "perturbation": perturbation,
            "alpha": alpha,
            "gamma": gamma,
            "stability": stability,
        }
        super().__init__(params, defaults)
        self.generator = generator if generator is not None else torch.Generator().manual_seed(0)
        self.model = model
        self.k = 0

    def _gains(self, group: dict[str, Any]) -> tuple[float, float]:
        a = group["lr"] / (self.k + 1 + group["stability"]) ** group["alpha"]
        c = group["perturbation"] / (self.k + 1) ** group["gamma"]
        return a, c

    def _params(self) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        return [(p, g) for g in self.param_groups for p in g["params"] if p.requires_grad]

    @torch.no_grad()
    def _evaluate(self, closure: Closure) -> tuple[list[torch.Tensor], float, float]:
        """``(Δ per parameter, L(θ + cΔ), L(θ − cΔ))``, parameters restored."""
        params = self._params()
        deltas = [
            torch.randint(0, 2, p.shape, generator=self.generator).to(p.dtype).mul_(2).sub_(1)
            for p, _ in params
        ]
        steps = [self._gains(g)[1] * d for (_, g), d in zip(params, deltas, strict=True)]
        # Restored from a copy, not by subtracting the steps back: θ + s − s
        # is not θ in floating point.
        originals = [p.detach().clone() for p, _ in params]
        rng = torch.get_rng_state()
        # Looked up every step: apply_shots swaps a layer's QNode in and out.
        devices = device_generators(self.model) if self.model is not None else []
        device_states = [g.bit_generator.state for g in devices]
        for (p, _), s in zip(params, steps, strict=True):
            p.add_(s)
        plus = float(closure())
        torch.set_rng_state(rng)  # the same dropout / noise draws on both sides
        for g, state in zip(devices, device_states, strict=True):
            g.bit_generator.state = state  # and the same shot-sampling draws
        for (p, _), orig, s in zip(params, originals, steps, strict=True):
            p.copy_(orig - s)
        minus = float(closure())
        for (p, _), orig in zip(params, originals, strict=True):
            p.copy_(orig)
        return deltas, plus, minus

    def gradient_estimate(self, closure: Closure) -> list[torch.Tensor]:
        """
        One SPSA gradient estimate at the current parameters, without a step.

        In the order of the parameters that require grad, group by group.
        Uses two loss evaluations and one draw from ``generator``.
        """
        deltas, plus, minus = self._evaluate(closure)
        return [
            (plus - minus) / (2 * self._gains(g)[1]) * d
            for (_, g), d in zip(self._params(), deltas, strict=True)
        ]

    @torch.no_grad()
    def step(self, closure: Closure | None = None) -> float:  # type: ignore[override]
        """
        One SPSA step.  Returns the mean of the two loss evaluations, an
        estimate of the loss at the parameters the step started from.

        Raises
        ------
        ValueError
            Without a ``closure``: SPSA evaluates the loss itself.
        """
        if closure is None:
            raise ValueError("SPSA.step needs a closure that returns the loss.")
        deltas, plus, minus = self._evaluate(closure)
        for (p, g), d in zip(self._params(), deltas, strict=True):
            a, c = self._gains(g)
            p.sub_(a * (plus - minus) / (2 * c) * d)
        self.k += 1
        return (plus + minus) / 2
