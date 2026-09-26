"""
hqnn_forge.initializers.restricted_variance
=============================================
Small-angle weight initialisation for variational quantum circuits.

Theory
------
In variational circuits that are deep and expressive enough to approximate a
2-design -- for which uniformly drawn parameters are one ingredient, not the
whole condition -- the gradient variance decays exponentially in the number of
qubits n:

    Var[∂L/∂θ] ∝ 2^{-n}   (global cost, McClean et al. 2018)

Two published results bound that decay.  Cerezo et al. (2021) show that for
*local* cost functions and shallow circuits (depth O(log n)) the decay is
only polynomial.  A single-qubit ⟨Z_i⟩ readout is a local observable, but
whether it is a local *cost* in their sense depends on how far the circuit
spreads it; for this library's default circuit it is not (measured below).  Zhang et al. (2022) show that drawing the
parameters from N(0, σ²) with σ² = O(1/L) instead of uniformly bounds the
gradient norm below by a polynomial in n and L, for deep circuits too.

The two initialisers here are **this library's own heuristics**; neither
formula is taken from a paper:

    restricted_normal_init_:  σ   = scale / sqrt(n_qubits * n_layers)
    block_local_init_:        σ_ℓ = scale / sqrt(n_qubits * (ℓ + 1))

Only the first shrinks with depth, and only it is loosely in the spirit of
Zhang et al.: σ² = scale²/(n L) carries their 1/L factor, with an extra 1/n
the paper does not ask for.  The schedule of ``block_local_init_`` depends on
the layer index alone, so σ_ℓ² is O(1) in the total depth L -- layer 0 of a
64-layer circuit gets the same σ = π/sqrt(8) ≈ 1.11 rad (at 8 qubits) as
layer 0 of a 2-layer one.  It orders the layers relative to each other; it is
not a depth-dependent variance bound, and not the O(1/L) result above.

What the σ are for is conditioning, not a plateau guarantee: small angles keep
the initial state near the encoded product state, where the single-qubit ⟨Z_i⟩
readouts are still informative.  Neither σ is derived to guarantee any
particular gradient variance, and two caveats are worth stating outright:

* "Small" only holds above a certain circuit size.  Angles drawn uniformly
  from [0, 2π) have standard deviation 2π/sqrt(12) ≈ 1.81 rad, so with the
  default ``scale = π`` this initialisation is narrower than that uniform
  draw only for ``n_qubits * n_layers ≥ 4``.  At the library defaults (8 qubits ×
  2 layers) σ = π/4 ≈ 0.79 rad, 43% of the uniform spread; at the smallest
  circuit the encoders accept (2 qubits, 1 layer) σ ≈ 2.22 rad, 22% *wider*
  than uniform.
* Leaving the uniform regime is not by itself what avoids a plateau.  The
  2-design argument needs depth and structure too.  Nor is the default
  circuit in the shallow local-cost regime of Cerezo et al.: its readouts
  behave as global costs, as the next section measures.

The initial circuit is therefore a *small-angle* one, not an identity one.
Grant et al. (2019) is a different strategy (identity blocks: parameters
chosen so that consecutive blocks compose to the identity) and is not
implemented here; it is cited for contrast.

What the initialiser does for this library's circuits (measured)
----------------------------------------------------------------
The local-cost argument above does not apply to the library's default
circuit.  Its CNOT ring is a cascade, CNOT(0,1), CNOT(1,2), …, CNOT(n-1,0),
so the backward light cone of ⟨Z_i⟩ through one ring covers qubits
{0, …, i+1} for 0 < i < n-1 and all n qubits for i = 0 and i = n-1; through
a second ring the closing CNOT(n-1,0) pulls in every qubit.  From 2 layers on
(the default) every ⟨Z_i⟩ therefore depends on every input, and the readouts
behave as global costs.  Measured with
:func:`hqnn_forge.diagnostics.gradient_variance` on ``QuantumEncodingLayer``
(2 layers, cost ⟨Z_0⟩, ``default.qubit``, mean
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
* With inputs near zero it keeps more gradient variance, by a factor that
  grows over the measured range: 1.09x at 4 qubits, 1.75x at 8 (5-seed range
  1.67–1.89).  Whether that growth continues past 8 qubits -- i.e. whether the
  initialiser slows the decay for near-zero inputs rather than shifting it --
  has not been measured.
* With inputs spread over (-π, π) it does **not** change the exponential
  decay with qubit count: both inits lose about 3x per two qubits.  That decay
  is set by the circuit, not the initialisation.

The initialisers are kept as the default because they are harmless and
cheap, and because the ``scale`` argument gives a one-parameter handle on
the initial angle spread.  They should not be relied on for trainability at
larger qubit counts; a locality-preserving entangler is the lever for that
(issue #161).  ``tests/test_gradient_variance.py`` pins the statements above
so a change that alters them is noticed.

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

    A small-angle initialisation: σ shrinks with both width and depth, so for
    circuits above a minimum size the initial parameters are narrower than a
    uniform draw over [0, 2π) (standard deviation ≈ 1.81 rad) -- with the
    default ``scale``, that means ``n_qubits * n_layers ≥ 4``; below that size
    this σ is the wider of the two.  The formula is this library's heuristic (see
    the module docstring), not a published prescription, and it does not by
    itself guarantee O(1) gradient variance.

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
        Smaller values narrow the initial angle spread; per the module
        docstring, that does not counter the decay of gradient variance with
        qubit count for inputs spread over (-π, π).

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
            f"n_qubits and n_layers must be ≥ 1; got n_qubits={n_qubits}, n_layers={n_layers}."
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
    of narrowing every layer by the full depth.  Because σ_ℓ depends on the
    layer index alone, it does **not** shrink with the total depth: layer 0 of
    a 64-layer circuit is initialised exactly as wide as layer 0 of a 2-layer
    one.  The schedule is this library's heuristic -- neither the identity-block
    scheme of Grant et al. (2019) nor the O(1/L) variance of Zhang et al.
    (2022).

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
