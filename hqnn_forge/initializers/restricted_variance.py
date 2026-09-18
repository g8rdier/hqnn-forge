"""
hqnn_forge.initializers.restricted_variance
=============================================
Barren-plateau-aware weight initialisation for variational quantum circuits.

Theory
------
In variational circuits whose parameters are drawn uniformly at random (so
that the circuit approximates a 2-design), the gradient variance decays
exponentially in the number of qubits n:

    Var[∂L/∂θ] ∝ 2^{-n}   (global cost, McClean et al. 2018)

Two published results bound that decay.  Cerezo et al. (2021) show that for
*local* cost functions and shallow circuits (depth O(log n)) the decay is
only polynomial -- the per-qubit ⟨Z_i⟩ readouts used throughout this library
are local costs for that reason.  Zhang et al. (2022) show that drawing the
parameters from N(0, σ²) with σ² = O(1/L) instead of uniformly bounds the
gradient norm below by a polynomial in n and L, for deep circuits too.

The two initialisers here are **this library's own heuristics** in the
spirit of the second result; neither formula is taken from a paper:

    restricted_normal_init_:  σ   = scale / sqrt(n_qubits * n_layers)
    block_local_init_:        σ_ℓ = scale / sqrt(n_qubits * (ℓ + 1))

Shrinking σ with both width and depth keeps the initial parameters far from
the uniform-over-[0, 2π) regime that the 2-design argument needs.  With the
default ``scale = π`` and 8 qubits × 2 layers that gives σ = π/4 ≈ 0.79 rad:
a *small-angle* initialisation, not an identity one -- the initial circuit is
not close to the identity, and the σ are not derived to guarantee any
particular gradient variance.  Grant et al. (2019) is a different strategy
(identity blocks: parameters chosen so that consecutive blocks compose to
the identity) and is not implemented here; it is cited for contrast.

What the initialiser does for this library's circuits (measured)
----------------------------------------------------------------
The local-cost argument above does not apply to the library's default
circuit: its CNOT ring is a cascade, so the backward light cone of every
⟨Z_i⟩ spans all n qubits after one layer and the readouts behave as global
costs.  Measured with :func:`hqnn_forge.diagnostics.gradient_variance` on
``QuantumEncodingLayer`` (2 layers, cost ⟨Z_0⟩, ``default.qubit``, mean
per-weight gradient variance over 5 seeds × 300 draws of weights and inputs),
the ratio of restricted-init to uniform-init variance is:

    inputs uniform in   n=4    n=6    n=8
    {0}                 1.09   1.52   1.75
    ±π/4                1.00   1.20   1.53
    ±π                  0.97   0.97   1.00     (5-seed range at n=8: 0.80–1.12)

and the uniform-init variance itself at ±π falls 0.0153 → 0.00428 → 0.00166
from 4 to 8 qubits, about 3x per two qubits, with the restricted init
following the same curve (0.0149 → 0.0042 → 0.0017).

So:

* With inputs spread over (-π, π), which is what both classifiers feed the
  circuit (``tanh(·)·π``) and what ``PCANormalizer(scale_to_pi=True)``
  produces, the initialiser makes **no measurable difference**: the angle
  embedding already randomises the state, and shrinking the weight angles
  cannot bring it back near the identity.
* With inputs near zero it keeps a **constant factor** more gradient
  variance (1.5–1.8x at 8 qubits), and the factor grows with n.
* It does **not** change the exponential decay with qubit count under either
  input range; that is set by the circuit, not the initialisation.

The initialisers are kept as the default because they are harmless and
cheap, and because the ``scale`` argument gives a one-parameter handle on
the initial angle spread.  They should not be relied on for trainability at
larger qubit counts; a locality-preserving entangler is the lever for that
(see the brickwork entangler issue).  ``tests/test_gradient_variance.py``
pins the three statements above so a change that alters them is noticed.

Functions
---------
restricted_normal_init_     In-place; fills a tensor with restricted-normal values.
block_local_init_           Fills each block (layer slice) independently.

References
----------
* McClean et al. (2018) "Barren plateaus in quantum neural network training
  landscapes", Nature Communications 9, 4812.
* Cerezo et al. (2021) "Cost function dependent barren plateaus in shallow
  parametrized quantum circuits", Nature Communications 12, 1791.
* Grant et al. (2019) "An initialization strategy for addressing barren
  plateaus in parametrized quantum circuits", Quantum 3, 214.
* Zhang et al. (2022) "Escaping from the barren plateau via Gaussian
  initializations in deep variational quantum circuits", NeurIPS 35.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# In-place initialiser: single call, shared σ across all parameters
# ---------------------------------------------------------------------------

def restricted_normal_init_(
    tensor: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    scale: float = math.pi,
) -> torch.Tensor:
    """
    Fill *tensor* **in-place** with values drawn from N(0, σ²) where

        σ = scale / sqrt(n_qubits * n_layers)

    A small-angle initialisation: σ shrinks with both width and depth so the
    initial parameters stay far from uniform over [0, 2π), the regime in which
    gradients vanish exponentially.  The formula is this library's heuristic
    (see the module docstring), not a published prescription, and it does not
    by itself guarantee O(1) gradient variance.

    Parameters
    ----------
    tensor:
        The weight tensor to initialise.  Typically the ``weights`` parameter
        of a ``QuantumEncodingLayer``, shape ``(n_layers, n_qubits, 3)``.
    n_qubits:
        Number of qubits in the circuit.
    n_layers:
        Number of variational layers.
    scale:
        Numerator of the standard deviation formula.  Default: π.
        Adjust downward (e.g. π/2) for deeper circuits if gradients still vanish.

    Returns
    -------
    torch.Tensor
        The initialised tensor (modified in-place and returned for chaining).

    Raises
    ------
    ValueError
        If ``n_qubits`` or ``n_layers`` is less than 1.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.initializers import restricted_normal_init_
    >>> w = torch.empty(2, 8, 3)  # (n_layers=2, n_qubits=8, 3 Euler angles)
    >>> restricted_normal_init_(w, n_qubits=8, n_layers=2)
    >>> w.std().item()  # ≈ π / sqrt(16) ≈ 0.785
    """
    if n_qubits < 1 or n_layers < 1:
        raise ValueError(
            f"n_qubits and n_layers must be ≥ 1; "
            f"got n_qubits={n_qubits}, n_layers={n_layers}."
        )

    std = scale / math.sqrt(n_qubits * n_layers)
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std)
    return tensor


# ---------------------------------------------------------------------------
# Block-local variant: each layer initialised with its own restricted σ
# ---------------------------------------------------------------------------

def block_local_init_(
    tensor: torch.Tensor,
    n_qubits: int,
    scale: float = math.pi,
) -> torch.Tensor:
    """
    Fill *tensor* **in-place** with per-block restricted-normal values.

    For each layer ℓ the standard deviation is computed using *only that
    layer's* depth contribution:

        σ_ℓ = scale / sqrt(n_qubits * (ℓ + 1))

    A per-layer variant of :func:`restricted_normal_init_` for deeper circuits:
    early layers keep a wider σ and only the later ones are narrowed, instead
    of narrowing every layer by the full depth.  The schedule is this library's
    heuristic; it is not the identity-block scheme of Grant et al. (2019).

    Parameters
    ----------
    tensor:
        Weight tensor of shape ``(n_layers, n_qubits, 3)`` or any shape
        where ``dim 0`` indexes layers.
    n_qubits:
        Number of qubits.
    scale:
        Numerator for std computation.  Default: π.

    Returns
    -------
    torch.Tensor
        The initialised tensor (in-place).

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.initializers import block_local_init_
    >>> w = torch.empty(4, 8, 3)  # 4-layer circuit
    >>> block_local_init_(w, n_qubits=8)
    >>> # Layer 0 has the largest variance; layer 3 the smallest.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be ≥ 1; got {n_qubits}.")

    n_layers: int = tensor.shape[0]
    with torch.no_grad():
        for layer_idx in range(n_layers):
            std = scale / math.sqrt(n_qubits * (layer_idx + 1))
            tensor[layer_idx].normal_(mean=0.0, std=std)
    return tensor


# ---------------------------------------------------------------------------
# Convenience: apply to all VQC parameters in an nn.Module
# ---------------------------------------------------------------------------

def apply_restricted_init(
    module: nn.Module,
    n_qubits: int,
    n_layers: int,
    *,
    block_local: bool = False,
    scale: float = math.pi,
) -> None:
    """
    Apply the small-angle initialisation to **all parameters** in *module*
    whose shape starts with ``(n_layers, ...)``.

    This is a convenience wrapper; for fine-grained control call
    :func:`restricted_normal_init_` or :func:`block_local_init_` directly.

    Parameters
    ----------
    module:
        The PyTorch module (e.g. a ``QuantumEncodingLayer`` or the full
        ``HybridBinaryClassifier``) whose quantum weight parameters should be
        initialised.
    n_qubits:
        Number of qubits.
    n_layers:
        Number of variational layers.
    block_local:
        If ``True``, use :func:`block_local_init_` (per-layer σ).
        If ``False`` (default), use :func:`restricted_normal_init_` (global σ).
    scale:
        Standard deviation numerator.  Default: π.
    """
    for name, param in module.named_parameters():
        if param.ndim >= 1 and param.shape[0] == n_layers:
            if block_local:
                block_local_init_(param.data, n_qubits=n_qubits, scale=scale)
            else:
                restricted_normal_init_(
                    param.data, n_qubits=n_qubits, n_layers=n_layers, scale=scale
                )
            # Do not update running stats / non-trainable buffers
