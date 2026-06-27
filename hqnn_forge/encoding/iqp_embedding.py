"""
hqnn_forge.encoding.iqp_embedding
===================================
Placeholder for Instantaneous Quantum Polynomial (IQP) feature encoding.

IQP circuits encode classical data as diagonal unitaries of the form

    U(x) = exp(i x_i Z_i) · exp(i x_i x_j Z_i Z_j)

producing a feature map that is believed to be classically hard to simulate
under plausible complexity-theoretic assumptions (Havlíček et al. 2019).

Status
------
This module is a **planned future addition**.  The current release focuses on
the angle-embedding approach (see :mod:`hqnn_forge.encoding.angle_embedding`),
which is better characterised and more stable for NISQ-era training.

References
----------
* Havlíček et al. (2019) "Supervised learning with quantum-enhanced feature
  spaces", Nature 567, 209–212.
* Schuld & Petruccione (2021) "Machine Learning with Quantum Computers",
  Springer.  Chapter 5.
"""

from __future__ import annotations

import warnings


class IQPEmbeddingLayer:  # noqa: D101  (placeholder; no public API yet)
    """Placeholder — IQP encoding is not yet implemented."""

    def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: ANN401
        warnings.warn(
            "IQPEmbeddingLayer is not yet implemented.  "
            "Use hqnn_forge.encoding.QuantumEncodingLayer (angle embedding) instead.",
            FutureWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "IQP embedding is planned for a future release.  "
            "Track progress at https://github.com/hqnn-forge/hqnn-forge/issues."
        )
