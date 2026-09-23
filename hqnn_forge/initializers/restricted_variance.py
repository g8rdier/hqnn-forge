"""
hqnn_forge.initializers.restricted_variance
=============================================
Barren-plateau-aware weight initialisation for variational quantum circuits.

Theory
------
In variational circuits that are deep and expressive enough to approximate a
2-design -- for which uniformly drawn parameters are one ingredient, not the
whole condition -- the gradient variance decays exponentially in the number of
qubits n:

    Var[∂L/∂θ] ∝ 2^{-n}   (global cost, McClean et al. 2018)

Two published results bound that decay.  Cerezo et al. (2021) show that for
*local* cost functions and shallow circuits (depth O(log n)) the decay is
only polynomial -- the per-qubit ⟨Z_i⟩ readouts used throughout this library
are local costs for that reason.  Zhang et al. (2022) show that drawing the
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
the initial state near the encoded product state, where the local ⟨Z_i⟩
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
  2-design argument needs depth and structure too, and at the defaults
  (8 qubits, 2 layers, local ⟨Z_i⟩ readouts) the circuit already sits in the
  shallow local-cost regime where Cerezo et al. predict polynomial decay.

The initial circuit is therefore a *small-angle* one, not an identity one.
Grant et al. (2019) is a different strategy (identity blocks: parameters
chosen so that consecutive blocks compose to the identity) and is not
implemented here; it is cited for contrast.

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
