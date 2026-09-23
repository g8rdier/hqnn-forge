"""
hqnn_forge.utils.ablation
=========================
Switch off a hybrid model's quantum layer to measure what it contributes.

The question "is the circuit carrying signal, or would the classical layers
around it do as well on their own?" is answered by comparing a model with its
quantum layer against the same model with that layer replaced by a constant.
:func:`disable_quantum_layer` does the replacement for the duration of a
``with`` block, without retraining and without touching the weights.

The replacement output is ``fill`` (default 0.0) for every qubit.  0 is the
midpoint of the ⟨Z⟩ range [-1, 1], i.e. the readout of a maximally
uninformative qubit, so the head sees "no quantum information" rather than an
out-of-distribution value.  The circuit is not executed at all inside the
block, so an ablated evaluation is also much faster.

Example
-------
::

    from hqnn_forge.evaluation import find_optimal_threshold
    from hqnn_forge.models import ParallelHybridClassifier
    from hqnn_forge.utils import disable_quantum_layer

    model = ParallelHybridClassifier(n_input_features=30, n_qubits=8, n_layers=2)
    # ... train model ...
    full = find_optimal_threshold(y_val, model.predict_proba(X_val)).score
    with disable_quantum_layer(model):
        ablated = find_optimal_threshold(y_val, model.predict_proba(X_val)).score
    print(f"MCC with circuit {full:.3f}, without {ablated:.3f}")

Which topology
--------------
Only a model with a classical path around the circuit gives a meaningful
comparison.  In :class:`~hqnn_forge.models.ParallelHybridClassifier` the
classical branch still reaches the head, so the ablated score is the score of
that branch on its own.  In :class:`~hqnn_forge.models.HybridBinaryClassifier`
the quantum layer is the only path from input to head, so ablating it leaves
``head(full((n_qubits,), fill))``: one probability for every sample, a constant
predictor scoring 0 MCC by construction.  That number restates the topology and
measures nothing.  A classical baseline for the serial model has to be a
separately trained classical model, not this context manager.

For a fair ablation study on the parallel topology, also train a model *from
scratch* with the layer disabled: a head trained alongside the circuit has
adapted to its outputs, so ablating after training measures dependence, not the
best achievable classical performance.  (Training a serial model from scratch
inside the block trains its head's bias and nothing else.)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def disable_quantum_layer(model: nn.Module, fill: float = 0.0) -> Iterator[nn.Module]:
    """
    Replace ``model.quantum_layer``'s output with the constant ``fill``.

    Inside the block the layer returns a tensor of shape
    ``(*batch_dims, n_outputs)`` filled with ``fill``, in the input's dtype and
    device, without running the circuit.  ``n_outputs`` is the layer's readout
    width -- ``n_qubits`` for ``readout="all"``, 1 for ``readout="first"`` -- so
    the head downstream sees the width it was built for.  The output therefore
    carries no gradient to the quantum weights or to anything upstream of it.  The
    original ``forward`` is restored on exit, including when the block raises.

    Training inside the block is allowed and trains only the parameters that
    still influence the loss.

    Parameters
    ----------
    model:
        A hybrid classifier with a ``quantum_layer`` attribute whose layer has
        ``n_qubits``.
    fill:
        Constant output per qubit.  Must lie in [-1, 1], the range of a Pauli-Z
        expectation.  Default: 0.0.

    Yields
    ------
    nn.Module
        The disabled quantum layer.

    Raises
    ------
    TypeError
        If ``model`` has no suitable ``quantum_layer``.
    ValueError
        If ``fill`` is outside [-1, 1], or, inside the block, if the layer is
        called with an input whose last dimension is not ``n_qubits``.
    RuntimeError
        If the layer is already disabled (nested use on the same model).
    """
    layer = getattr(model, "quantum_layer", None)
    n_qubits = getattr(layer, "n_qubits", None)
    if not isinstance(layer, nn.Module) or not isinstance(n_qubits, int):
        raise TypeError(
            f"disable_quantum_layer expects a model with a quantum_layer attribute; "
            f"got {type(model).__name__}."
        )
    # The readout decides how wide the layer's output is, and the head is built
    # for that width; filling n_qubits wide would break readout="first".
    n_outputs = getattr(layer, "n_outputs", None)
    if not isinstance(n_outputs, int):
        n_outputs = n_qubits
    if not -1.0 <= fill <= 1.0:
        raise ValueError(f"fill must lie in [-1, 1], the range of <Z>; got {fill}.")
    if "forward" in vars(layer):
        raise RuntimeError(
            f"{type(layer).__name__}.forward is already overridden on this instance; "
            f"disable_quantum_layer cannot be nested."
        )

    def constant_forward(x: torch.Tensor) -> torch.Tensor:
        # The encoding layers reject an input whose width is not n_qubits; the
        # replacement has to reject it too, or an ablated run silently returns
        # numbers for input the full model refuses.
        if x.shape[-1] != n_qubits:
            raise ValueError(
                f"Input feature dimension {x.shape[-1]} does not match n_qubits={n_qubits}."
            )
        return torch.full((*x.shape[:-1], n_outputs), fill, dtype=x.dtype, device=x.device)

    # nn.Module.__call__ dispatches to self.forward, so an instance attribute
    # shadows the class method for this layer only.
    layer.forward = constant_forward  # type: ignore[method-assign]
    try:
        yield layer
    finally:
        del layer.forward
