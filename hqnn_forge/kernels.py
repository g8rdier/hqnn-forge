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
A kernel matrix over ``M`` training points has ``M(M+1)/2`` distinct entries.
On hardware that is ``O(M²)`` circuit evaluations, each an overlap estimate
with shot noise, before the SVM even starts, and every prediction costs
``M`` more overlaps against the training set.  A VQC needs ``O(M)`` circuit
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

import pennylane as qml
import torch
import torch.nn as nn

__all__ = ["encoded_states", "quantum_kernel_matrix"]


def _resolve_layer(layer: nn.Module) -> tuple[qml.qnn.TorchLayer, int]:
    """``(qlayer, n_qubits)`` of an encoding layer, or raise ``TypeError``."""
    qlayer = getattr(layer, "qlayer", None)
    n_qubits = getattr(layer, "n_qubits", None)
    if not isinstance(qlayer, qml.qnn.TorchLayer) or not isinstance(n_qubits, int):
        raise TypeError(
            f"quantum_kernel_matrix expects an encoding layer with a qlayer TorchLayer "
            f"and an integer n_qubits (QuantumEncodingLayer, IQPEncodingLayer, "
            f"AmplitudeEncodingLayer, DataReuploadingLayer); got {type(layer).__name__}."
        )
    return qlayer, n_qubits


def _check_inputs(X: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(X, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(X).__name__}.")
    if X.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features); got {tuple(X.shape)}.")
    if X.shape[0] == 0:
        raise ValueError(f"{name} has no samples.")
    return X.detach().to(torch.float64)


def encoded_states(X: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """
    State vectors ``|Φ(x_i)⟩`` the layer prepares for each row of ``X``.

    The layer's circuit is replayed on ``default.qubit`` with its measurements
    replaced by ``qml.state()``.  Any input preparation the layer performs in
    ``forward`` before its QNode (the amplitude encoder's padding and
    normalisation, exposed as ``prepare_inputs``) is applied first.

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
    """
    qlayer, n_qubits = _resolve_layer(layer)
    X = _check_inputs(X, "X")
    prepare = getattr(layer, "prepare_inputs", None)
    if callable(prepare):
        X = prepare(X)

    # Build one tape per sample from the layer's own QNode (level=0: the
    # circuit as written, before any batching or gradient transform), swap
    # the measurements for the state, and run them all on a state-vector
    # device.  The layer's weights are used as they are, detached.
    weights = {name: p.detach().to(torch.float64) for name, p in qlayer.qnode_weights.items()}
    build = qml.workflow.construct_tape(qlayer.qnode, level=0)
    tapes = [build(X[i], **weights).copy(measurements=[qml.state()]) for i in range(X.shape[0])]
    device = qml.device("default.qubit", wires=n_qubits)
    results = qml.execute(tapes, device, diff_method=None)
    states = torch.stack([torch.as_tensor(r) for r in results]).to(torch.complex128)
    return states.reshape(X.shape[0], 2**n_qubits)


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
        matrix an SVM needs at prediction time.

    Returns
    -------
    torch.Tensor
        ``float64`` tensor of shape ``(n_samples_x, n_samples_y)`` (or
        ``(n_samples_x, n_samples_x)``), entries in ``[0, 1]``.  The square
        matrix is symmetric, positive semi-definite, and has ones on the
        diagonal.

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
    The square case costs ``n_samples`` circuit replays and one
    ``(n, 2^q) × (2^q, n)`` product, so it is O(n · 2^q) in memory and
    O(n² · 2^q) in time.  ``|G|²`` with ``G = S S†`` is the Schur product of a
    positive semi-definite matrix with its conjugate, hence positive
    semi-definite itself; small negative eigenvalues of order 1e-15 are
    rounding.
    """
    states_x = encoded_states(X, layer)
    if Y is None:
        gram = states_x @ states_x.conj().T
    else:
        Y = _check_inputs(Y, "Y")
        if Y.shape[1] != X.shape[1]:
            raise ValueError(
                f"X and Y must have the same number of features; got {X.shape[1]} and {Y.shape[1]}."
            )
        states_y = encoded_states(Y, layer)
        gram = states_x @ states_y.conj().T
    kernel = gram.abs() ** 2
    if Y is None:
        # Exact symmetry, not just up to rounding; the diagonal is left as
        # computed so a state that is not normalised shows up as K[i, i] != 1.
        kernel = 0.5 * (kernel + kernel.T)
    return kernel.clamp_(0.0, 1.0)
