"""
hqnn_forge.preprocessing.pca_normalizer
==========================================
Classical pre-processing: PCA dimensionality reduction + per-feature
standardisation, implemented in **pure NumPy** (no scikit-learn runtime
dependency).

Workflow
--------
1. Fit on training data     → ``PCANormalizer.fit(X_train)``
2. Transform train/test     → ``PCANormalizer.transform(X)`` → ``torch.Tensor``
3. Pass to QuantumEncLayer  → features should lie in ``[-π, π]`` after scaling

Encoding Scaling
----------------
Raw PCA components are standardised (zero mean, unit variance) and then
optionally rescaled into ``[-π, π]`` via a tanh squeeze:

    x̂_i = tanh(x_std_i) * π

This keeps all features within the valid range for angle embedding while
preventing wrap-around aliasing for large outliers.

Notes
-----
* Eigendecomposition uses ``numpy.linalg.eigh`` (symmetric covariance matrix),
  which is numerically more stable than ``numpy.linalg.eig`` for this use case.
* Only the top ``n_components`` eigenvectors (by eigenvalue magnitude) are kept.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import torch


class PCANormalizer:
    """
    Fit/apply PCA + per-feature standardisation without scikit-learn.

    Parameters
    ----------
    n_components:
        Number of principal components to retain.  Default: 8.
    scale_to_pi:
        If ``True`` (default), rescale standardised components into ``[-π, π]``
        via ``tanh(x) * π`` before returning.  Ensures valid angle-embedding
        range without hard clipping.
    copy:
        If ``True`` (default), operate on copies of the input data.

    Attributes
    ----------
    mean_ : np.ndarray, shape (n_features,)
        Per-feature mean computed during ``fit``.
    components_ : np.ndarray, shape (n_components, n_features)
        Principal component matrix (rows = eigenvectors, sorted by descending
        explained variance).
    explained_variance_ : np.ndarray, shape (n_components,)
        Eigenvalues corresponding to retained components.
    std_ : np.ndarray, shape (n_components,)
        Per-component standard deviation (computed on training projections).
    is_fitted_ : bool
        ``True`` after ``fit`` has been called.

    Examples
    --------
    >>> import numpy as np
    >>> from hqnn_forge.preprocessing import PCANormalizer
    >>> rng = np.random.default_rng(42)
    >>> X_train = rng.standard_normal((1000, 30))  # 1000 samples, 30 raw features
    >>> pca = PCANormalizer(n_components=8)
    >>> pca.fit(X_train)
    >>> X_enc = pca.transform(X_train)  # torch.Tensor, shape (1000, 8)
    >>> X_enc.shape
    torch.Size([1000, 8])
    """

    def __init__(
        self,
        n_components: int = 8,
        *,
        scale_to_pi: bool = True,
        copy: bool = True,
    ) -> None:
        self.n_components = n_components
        self.scale_to_pi  = scale_to_pi
        self.copy         = copy

        # Populated by fit()
        self.mean_: Optional[npt.NDArray[np.float64]] = None
        self.components_: Optional[npt.NDArray[np.float64]] = None
        self.explained_variance_: Optional[npt.NDArray[np.float64]] = None
        self.std_: Optional[npt.NDArray[np.float64]] = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    def fit(self, X: npt.ArrayLike) -> "PCANormalizer":
        """
        Compute PCA basis and per-component statistics on training data.

        Parameters
        ----------
        X:
            Training data array-like of shape ``(n_samples, n_features)``.
            ``n_features`` must be ≥ ``n_components``.

        Returns
        -------
        self
            The fitted transformer (for method chaining).

        Raises
        ------
        ValueError
            If ``n_features < n_components``.
        """
        X_arr: npt.NDArray[np.float64] = np.array(X, dtype=np.float64)
        if self.copy:
            X_arr = X_arr.copy()

        n_samples, n_features = X_arr.shape
        if n_features < self.n_components:
            raise ValueError(
                f"n_features={n_features} < n_components={self.n_components}.  "
                f"Reduce n_components or provide higher-dimensional data."
            )

        # 1. Centre the data
        self.mean_ = X_arr.mean(axis=0)
        X_centered = X_arr - self.mean_

        # 2. Covariance matrix (unbiased estimator, ddof=1)
        cov = np.cov(X_centered, rowvar=False)  # shape (n_features, n_features)

        # 3. Eigendecomposition (eigh: exploit symmetry for stability + speed)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 4. Sort by descending eigenvalue and keep top-k components
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        self.explained_variance_ = eigenvalues[: self.n_components]
        # rows = components (shape: n_components × n_features)
        self.components_ = eigenvectors[:, : self.n_components].T

        # 5. Project training data → compute per-component std for standardisation
        projections = X_centered @ self.components_.T          # (n_samples, n_components)
        self.std_   = projections.std(axis=0, ddof=1) + 1e-8  # avoid div-by-zero

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def transform(self, X: npt.ArrayLike) -> torch.Tensor:
        """
        Project and standardise (and optionally scale to [-π, π]).

        Parameters
        ----------
        X:
            Data array-like of shape ``(n_samples, n_features)``.
            Must have the same ``n_features`` as the training data.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape ``(n_samples, n_components)``,
            dtype ``float32``, values in ``[-π, π]`` if ``scale_to_pi=True``.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        ValueError
            If input feature dimension doesn't match training data.
        """
        self._check_is_fitted()

        X_arr: npt.NDArray[np.float64] = np.array(X, dtype=np.float64)
        if self.copy:
            X_arr = X_arr.copy()

        if X_arr.shape[1] != self.mean_.shape[0]:  # type: ignore[union-attr]
            raise ValueError(
                f"Input has {X_arr.shape[1]} features but PCANormalizer was "
                f"fitted on {self.mean_.shape[0]} features."  # type: ignore[union-attr]
            )

        # Centre → project → standardise
        X_centered   = X_arr - self.mean_                            # type: ignore[operator]
        projections  = X_centered @ self.components_.T               # type: ignore[union-attr]
        standardised = (projections - projections.mean(axis=0)) / self.std_  # type: ignore[operator]

        if self.scale_to_pi:
            # Soft-clip to (-π, π) preserving relative magnitudes of outliers
            standardised = np.tanh(standardised) * np.pi

        return torch.tensor(standardised, dtype=torch.float32)

    # ------------------------------------------------------------------
    def fit_transform(self, X: npt.ArrayLike) -> torch.Tensor:
        """
        Fit and transform in a single call (convenience method).

        Parameters
        ----------
        X:
            Training data, shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Transformed tensor, shape ``(n_samples, n_components)``.
        """
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    @property
    def explained_variance_ratio_(self) -> npt.NDArray[np.float64]:
        """
        Fraction of total variance explained by each retained component.

        Returns
        -------
        np.ndarray, shape (n_components,)
        """
        self._check_is_fitted()
        total_var = self.explained_variance_.sum()  # type: ignore[union-attr]
        return self.explained_variance_ / total_var  # type: ignore[operator]

    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "PCANormalizer is not fitted.  Call .fit(X_train) first."
            )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"PCANormalizer("
            f"n_components={self.n_components}, "
            f"scale_to_pi={self.scale_to_pi}, "
            f"status={status})"
        )
