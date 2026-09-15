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
* Eigenvector signs are canonicalised so that the largest-magnitude entry of
  each component is positive.  ``eigh`` gives no guarantee about which of the
  two valid signs it returns, so without this the fitted basis -- and every
  encoded feature -- would differ between LAPACK builds and platforms.  The
  guarantee is conditional; ``PCANormalizer.components_`` states the conditions.
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
        Number of principal components to retain.  Must be ≥ 1; ``fit``
        raises ``ValueError`` otherwise.  Default: 8.
    scale_to_pi:
        If ``True`` (default), rescale standardised components into ``[-π, π]``
        via ``tanh(x) * π`` before returning.  Ensures valid angle-embedding
        range without hard clipping.

    Attributes
    ----------
    mean_ : np.ndarray, shape (n_features,)
        Per-feature mean computed during ``fit``.
    components_ : np.ndarray, shape (n_components, n_features)
        Principal component matrix (rows = eigenvectors, sorted by descending
        explained variance).  Each row's sign is canonicalised so its
        largest-magnitude entry is positive, making ``components_`` -- and
        therefore ``transform`` -- a deterministic function of the input data
        rather than of the LAPACK build.  This is part of the public contract,
        and holds whenever the eigenvalues are well separated.  Entries tied for
        largest magnitude are resolved by column order, so mirrored feature
        pairs are covered; near-degenerate eigenvalues, however, leave the basis
        itself build-dependent, which no sign convention can repair.
        The convention matches scikit-learn's ``svd_flip`` with
        ``u_based_decision=False``.  Note that ``sklearn.decomposition.PCA``
        uses the default ``u_based_decision=True``, which keys on the left
        singular vectors instead, so individual rows may differ in sign from it.
    explained_variance_ : np.ndarray, shape (n_components,)
        Eigenvalues corresponding to retained components.
    std_ : np.ndarray, shape (n_components,)
        Per-component standard deviation (computed on training projections).
    is_fitted_ : bool
        ``True`` after ``fit`` has been called.

    Notes
    -----
    Input is converted with ``np.asarray``, so ``float64`` input is used without
    copying and ``fit``/``transform`` may hold a view of the caller's buffer for
    the duration of the call.  Neither method writes into it, and no view of it
    is kept in the fitted attributes, so the caller is free to modify or discard
    the array afterwards.

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
    ) -> None:
        self.n_components = n_components
        self.scale_to_pi  = scale_to_pi

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
            ``n_features`` must be ≥ ``n_components`` and ``n_samples`` must
            be > ``n_components``.  Not copied when already ``float64``, and
            never modified.

        Returns
        -------
        self
            The fitted transformer (for method chaining).

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, if ``n_components < 1``, if
            ``n_features < n_components``, or if ``n_samples <= n_components``
            (centred data then has rank below ``n_components``, so some
            components have zero variance).
        """
        # asarray, not array: float64 input is used as-is rather than copied, so
        # X_arr may share memory with the caller.  Never write into it in place.
        X_arr: npt.NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        self._check_2d(X_arr)

        n_samples, n_features = X_arr.shape
        # Checked here rather than in __init__ because the attribute can be
        # reassigned afterwards.  Values <= 0 pass both shape checks below and
        # then silently slice off components from the end (-1 keeps all but one)
        if self.n_components < 1:
            raise ValueError(
                f"n_components={self.n_components} < 1.  Provide a positive "
                f"number of components to retain."
            )
        if n_features < self.n_components:
            raise ValueError(
                f"n_features={n_features} < n_components={self.n_components}.  "
                f"Reduce n_components or provide higher-dimensional data."
            )
        # Centred data has rank <= n_samples - 1, so fewer rows leave some
        # kept components with zero variance
        if n_samples <= self.n_components:
            raise ValueError(
                f"n_samples={n_samples} <= n_components={self.n_components}.  "
                f"Reduce n_components or provide at least "
                f"{self.n_components + 1} samples."
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
        components = eigenvectors[:, : self.n_components].T

        # 5. Canonicalise the sign of each component.  eigh returns eigenvectors
        # up to an arbitrary sign, so another LAPACK build may hand back a
        # component negated -- components_ and the encoded features would not be
        # reproducible across platforms.  Convention (matching scikit-learn's
        # svd_flip): the largest-magnitude entry of each component is positive.
        #
        # Which entry that is has to be decided with a tolerance, because exact
        # ties are structural rather than a coincidence of continuous data: if
        # two columns are exact mirrors (x_j == -x_i, as in a two-level one-hot,
        # a share/1-share pair, or a +/- sensor pair), then (e_i + e_j)/sqrt(2)
        # is an exact null eigenvector of the covariance, so every retained
        # component satisfies v_j == -v_i to the last ulp.  Where that pair
        # carries the largest entry, a strict argmax keys the whole row's sign
        # on ~1e-16 rounding noise.  Taking the first entry within a relative
        # tolerance of the maximum instead makes the choice a function of column
        # order alone; where the maximum is unique it selects what argmax would.
        # np.sign cannot return 0 here: the largest-magnitude entry of a
        # unit-norm vector is at least 1/sqrt(n_features).
        magnitudes = np.abs(components)
        tied       = magnitudes >= magnitudes.max(axis=1, keepdims=True) * (1 - 1e-12)
        leading    = tied.argmax(axis=1)
        signs      = np.sign(components[np.arange(components.shape[0]), leading])
        # Out-of-place: components is a non-contiguous view over the full
        # (n_features, n_features) eigenvector matrix, which this also drops
        self.components_ = components * signs[:, np.newaxis]

        # 6. Project training data → compute per-component std for standardisation
        projections = X_centered @ self.components_.T          # (n_samples, n_components)
        self.std_   = projections.std(axis=0, ddof=1) + 1e-8  # avoid div-by-zero

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def transform(self, X: npt.ArrayLike) -> torch.Tensor:
        """
        Project and standardise (and optionally scale to [-π, π]).

        Only statistics learned in ``fit`` (``mean_``, ``components_``,
        ``std_``) are used, so each row is encoded independently of the other
        rows in ``X``.  Standardised output has zero mean and unit variance on
        the training data; other data keeps its offset from the training
        distribution.

        Parameters
        ----------
        X:
            Data array-like of shape ``(n_samples, n_features)``.
            Must have the same ``n_features`` as the training data.
            Not copied when already ``float64``, and never modified.

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
            If ``X`` is not 2-D, or if its feature dimension doesn't match
            training data.
        """
        self._check_is_fitted()

        # asarray, not array: float64 input is used as-is rather than copied, so
        # X_arr may share memory with the caller.  Never write into it in place.
        X_arr: npt.NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        self._check_2d(X_arr)

        if X_arr.shape[1] != self.mean_.shape[0]:  # type: ignore[union-attr]
            raise ValueError(
                f"Input has {X_arr.shape[1]} features but PCANormalizer was "
                f"fitted on {self.mean_.shape[0]} features."  # type: ignore[union-attr]
            )

        # Centre → project → standardise, using training statistics only (no
        # per-batch mean), so each row's encoding is independent of the batch
        X_centered   = X_arr - self.mean_                            # type: ignore[operator]
        projections  = X_centered @ self.components_.T               # type: ignore[union-attr]
        standardised = projections / self.std_                       # type: ignore[operator]

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
    @staticmethod
    def _check_2d(X_arr: npt.NDArray[np.float64]) -> None:
        # 1-D input is ambiguous (one sample or one feature), so reject it
        # rather than guess a reshape
        if X_arr.ndim != 2:
            # The reshape hint only fits 1-D input; for higher ndim it would mislead
            hint = "  For a single sample, use X.reshape(1, -1)." if X_arr.ndim == 1 else ""
            raise ValueError(
                f"Expected 2-D input of shape (n_samples, n_features), got "
                f"shape {X_arr.shape}.{hint}"
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
