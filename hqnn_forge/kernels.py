"""
hqnn_forge.kernels
==================
Quantum kernel estimation from the library's encoding layers.

A quantum kernel is the fidelity between the states two inputs are encoded
into::

    k(x, x') = |⟨Φ(x) | Φ(x')⟩|²

Feeding the Gram matrix ``K[i, j] = k(x_i, x_j)`` to a classical SVM
(``sklearn.svm.SVC(kernel="precomputed")``) gives the quantum-kernel (QSVM)
approach to classification: the quantum device only evaluates the feature
map, and the optimisation is the SVM's convex problem, with a unique optimum
and no barren plateaus.  This is the complementary method to the trainable
VQCs in :mod:`hqnn_forge.models`.

Which circuit defines the kernel
--------------------------------
:func:`quantum_kernel_matrix` replays the circuit of an encoding layer
(``QuantumEncodingLayer``, ``IQPEncodingLayer``, ``AmplitudeEncodingLayer``,
``DataReuploadingLayer``) up to but not including its measurements, and
reads the state vector.  The layer's variational block is included as it
stands, with the layer's current weights.  For the single-upload encoders
this makes no difference: the ansatz is a data-independent unitary ``V`` and
``|⟨Φ(x)|V†V|Φ(x')⟩|² = |⟨Φ(x)|Φ(x')⟩|²``, so the kernel is that of the
embedding alone whatever the weights are.  For :class:`DataReuploadingLayer`
the weights sit between uploads and do shape the kernel; they are then part
of the kernel's definition (a "trainable kernel" in the sense of Hubregtsen
et al. 2022), and the matrix is that of the layer as currently parametrised.

Scaling: O(M²) against the VQC
------------------------------
A kernel matrix over ``M`` training points has ``M(M+1)/2`` distinct entries,
of which the ``M`` diagonal ones are 1 by construction.  On hardware the
other ``M(M-1)/2``, that is ``O(M²)``, are circuit evaluations, each an
overlap estimate with shot noise, before the SVM even starts, and every
prediction costs ``M`` more overlaps against the training set.  A VQC needs ``O(M)`` circuit
evaluations per epoch and one per prediction.  On a state-vector simulator
the picture is friendlier: ``M`` state vectors of size ``2^n`` and one
``M × M`` Gram product, which is what this module does.  Either way the
kernel approach stops being practical at the ``M`` where the VQC approach is
still routine, which is the trade-off discussed in the project's academic
context.

References
----------
* Havlíček et al. (2019) "Supervised learning with quantum-enhanced feature
  spaces", Nature 567, 209.
* Schuld & Killoran (2019) "Quantum machine learning in feature Hilbert
  spaces", PRL 122, 040504.
* Hubregtsen et al. (2022) "Training quantum embedding kernels on near-term
  quantum computers", PRA 106, 042431.
"""

from __future__ import annotations

from collections.abc import Callable

import pennylane as qml
import torch
from torch import nn

__all__ = ["encoded_states", "kernel_from_states", "quantum_kernel_matrix"]


PrepareInputs = Callable[[torch.Tensor], torch.Tensor]


def _resolve_layer(layer: nn.Module) -> tuple[qml.qnn.TorchLayer, int, PrepareInputs]:
    """``(qlayer, n_qubits, prepare_inputs)`` of an encoding layer, or raise ``TypeError``."""
    qlayer = getattr(layer, "qlayer", None)
    n_qubits = getattr(layer, "n_qubits", None)
    prepare = getattr(layer, "prepare_inputs", None)
    if (
        not isinstance(qlayer, qml.qnn.TorchLayer)
        or not isinstance(n_qubits, int)
        or not callable(prepare)
    ):
        raise TypeError(
            f"quantum_kernel_matrix expects an encoding layer with a qlayer TorchLayer, "
            f"an integer n_qubits and a prepare_inputs method (QuantumEncodingLayer, "
            f"IQPEncodingLayer, AmplitudeEncodingLayer, DataReuploadingLayer); "
            f"got {type(layer).__name__}."
        )
    return qlayer, n_qubits, prepare


def _prepare(X: torch.Tensor, prepare: PrepareInputs, name: str) -> torch.Tensor:
    """Check ``X`` is a non-empty 2-D tensor and apply the layer's ``prepare_inputs``."""
    if not isinstance(X, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(X).__name__}.")
    if X.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features); got {tuple(X.shape)}.")
    if X.shape[0] == 0:
        raise ValueError(f"{name} has no samples.")
    # The same validation and transform forward applies (width check, the
    # amplitude encoder's padding and normalisation).
    return prepare(X.detach().to(torch.float64))


def _simulate(prepared: torch.Tensor, qlayer: qml.qnn.TorchLayer, n_qubits: int) -> torch.Tensor:
    """State vectors for inputs that have already been through ``_prepare``."""
    # One tape for the whole batch from the layer's own QNode (level=0: the
    # circuit as written, before any batching or gradient transform), with the
    # measurements swapped for the state and run on a state-vector device,
    # which executes the broadcast tape as one vectorised pass.  The layer's
    # weights are used as they are, detached.
    weights = {name: p.detach().to(torch.float64) for name, p in qlayer.qnode_weights.items()}
    tape = qml.workflow.construct_tape(qlayer.qnode, level=0)(prepared, **weights)
    tape = tape.copy(measurements=[qml.state()])
    device = qml.device("default.qubit", wires=n_qubits)
    (result,) = qml.execute([tape], device, diff_method=None)
    states = torch.as_tensor(result).to(torch.complex128)
    return states.reshape(prepared.shape[0], 2**n_qubits)


def encoded_states(X: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """
    State vectors ``|Φ(x_i)⟩`` the layer prepares for each row of ``X``.

    ``X`` first goes through the layer's ``prepare_inputs``, the validation and
    classical transform ``forward`` applies before its QNode (a width check,
    and for the amplitude encoder padding and normalisation).  The layer's
    circuit is then replayed on ``default.qubit`` for the whole batch at once,
    with its measurements replaced by ``qml.state()``.

    Compute the states once and pass them to :func:`kernel_from_states` to
    reuse them, for example the training states at every prediction.

    Parameters
    ----------
    X:
        Inputs, shape ``(n_samples, n_features)``.
    layer:
        An encoding layer.

    Returns
    -------
    torch.Tensor
        Complex tensor of shape ``(n_samples, 2**n_qubits)``, one normalised
        state per row, ``complex128``.

    Raises
    ------
    TypeError
        If ``layer`` is not an encoding layer or ``X`` is not a tensor.
    ValueError
        If ``X`` is not a non-empty 2-D tensor, or ``prepare_inputs`` rejects
        it (wrong number of features, an all-zero amplitude vector).
    """
    qlayer, n_qubits, prepare = _resolve_layer(layer)
    return _simulate(_prepare(X, prepare, "X"), qlayer, n_qubits)


def kernel_from_states(
    states_x: torch.Tensor,
    states_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fidelity kernel ``K[i, j] = |⟨ψ_i|φ_j⟩|²`` from precomputed state vectors.

    Parameters
    ----------
    states_x:
        States of shape ``(n_x, dim)``, as returned by :func:`encoded_states`.
    states_y:
        Optional second set of states, shape ``(n_y, dim)``.  ``None``
        (default) gives the square Gram matrix of ``states_x`` with itself,
        made exactly symmetric.

    Returns
    -------
    torch.Tensor
        ``float64`` tensor of shape ``(n_x, n_y)`` (or ``(n_x, n_x)``).

    Examples
    --------
    >>> S_train = encoded_states(X_train, layer)
    >>> svm = SVC(kernel="precomputed").fit(kernel_from_states(S_train).numpy(), y_train)
    >>> K_test = kernel_from_states(encoded_states(X_test, layer), S_train)
    >>> y_pred = svm.predict(K_test.numpy())
    """
    symmetric = states_y is None
    if states_y is None:
        states_y = states_x
    if states_x.ndim != 2 or states_y.ndim != 2 or states_x.shape[1] != states_y.shape[1]:
        raise ValueError(
            f"states_x and states_y must be 2-D with the same state dimension; got "
            f"{tuple(states_x.shape)} and {tuple(states_y.shape)}."
        )
    kernel = (states_x @ states_y.conj().T).abs().pow(2).to(torch.float64)
    if symmetric:
        # Exact symmetry, not just up to rounding.  Nothing is clamped: the
        # diagonal is left as computed, so a state that is not normalised
        # shows up as K[i, i] != 1.
        kernel = 0.5 * (kernel + kernel.T)
    return kernel


def quantum_kernel_matrix(
    X: torch.Tensor,
    layer: nn.Module,
    Y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pairwise state-fidelity kernel ``K[i, j] = |⟨Φ(x_i)|Φ(y_j)⟩|²``.

    Parameters
    ----------
    X:
        Inputs, shape ``(n_samples_x, n_features)``.
    layer:
        The encoding layer whose circuit defines ``Φ``.  See the module
        docstring for the role of its variational weights.
    Y:
        Optional second set of inputs, shape ``(n_samples_y, n_features)``.
        ``None`` (default) computes the square Gram matrix of ``X`` with
        itself.  Pass the training inputs here to build the rectangular
        matrix an SVM needs at prediction time; to avoid simulating the
        training set again on every call, keep its :func:`encoded_states`
        and use :func:`kernel_from_states` instead.

    Returns
    -------
    torch.Tensor
        ``float64`` tensor of shape ``(n_samples_x, n_samples_y)`` (or
        ``(n_samples_x, n_samples_x)``), entries in ``[0, 1]`` up to rounding
        of order 1e-15.  The square matrix is symmetric, positive
        semi-definite, and has ones on the diagonal.

    Raises
    ------
    TypeError, ValueError
        As :func:`encoded_states`, for ``X`` and ``Y``.  Both are validated
        before anything is simulated.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from hqnn_forge.encoding import QuantumEncodingLayer
    >>> from hqnn_forge.kernels import quantum_kernel_matrix
    >>> layer = QuantumEncodingLayer(n_qubits=4, n_layers=1, device_name="default.qubit")
    >>> K_train = quantum_kernel_matrix(X_train, layer)
    >>> svm = SVC(kernel="precomputed").fit(K_train.numpy(), y_train)
    >>> K_test = quantum_kernel_matrix(X_test, layer, Y=X_train)
    >>> y_pred = svm.predict(K_test.numpy())

    Notes
    -----
    Each input set is one batched circuit replay, and the kernel one
    ``(n, 2^q) × (2^q, n)`` product, so the square case is O(n · 2^q) in
    memory and O(n² · 2^q) in time.  ``|G|²`` with ``G = S S†`` is the Schur
    product of a positive semi-definite matrix with its conjugate, hence
    positive semi-definite itself; small negative eigenvalues of order 1e-15
    are rounding.
    """
    qlayer, n_qubits, prepare = _resolve_layer(layer)
    # Validate both input sets before simulating either.
    prepared_x = _prepare(X, prepare, "X")
    prepared_y = None if Y is None else _prepare(Y, prepare, "Y")
    states_x = _simulate(prepared_x, qlayer, n_qubits)
    states_y = None if prepared_y is None else _simulate(prepared_y, qlayer, n_qubits)
    return kernel_from_states(states_x, states_y)
