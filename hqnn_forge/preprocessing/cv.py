"""
hqnn_forge.preprocessing.cv
===========================
Stratified k-fold splits and SMOTE oversampling that cannot leak across folds.

Oversampling before splitting is the classic leak on imbalanced data: a
synthetic minority point interpolated from a sample that later lands in the
validation fold carries that sample's information into training, and the
validation score goes up for no real reason.  The helpers here make the safe
order the only easy one:

* :func:`stratified_kfold` splits indices so every fold keeps the class
  prevalence of the whole set.
* :func:`smote` oversamples the minority class of whatever arrays it is given.
* :func:`oversample_fold` applies :func:`smote` to the training indices of one
  fold only and returns the validation rows untouched;
  :func:`iter_folds` does that for every fold.

Everything is pure NumPy, like the rest of ``hqnn_forge.preprocessing``.

References
----------
* Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique",
  JAIR 16, 321–357.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.intp]
FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Stratified k-fold
# ---------------------------------------------------------------------------

def stratified_kfold(
    y: npt.ArrayLike,
    n_splits: int = 5,
    *,
    shuffle: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> list[tuple[IntArray, IntArray]]:
    """
    Split sample indices into ``n_splits`` folds with equal class prevalence.

    Each class's indices are (optionally) shuffled and dealt into
    ``n_splits`` chunks whose sizes differ by at most one; fold ``k``
    validates on chunk ``k`` of every class and trains on the rest.  Every
    sample is validated exactly once.

    Parameters
    ----------
    y:
        Class labels, shape ``(n_samples,)``.  Any hashable label values.
    n_splits:
        Number of folds, at least 2.  Default: 5.
    shuffle:
        Shuffle within each class before dealing.  Default: True.
    random_state:
        Seed or generator for the shuffle.

    Returns
    -------
    list of (train_idx, val_idx)
        Sorted index arrays, one pair per fold.

    Raises
    ------
    ValueError
        If ``n_splits < 2``, ``y`` is not 1-D, or a class has fewer members
        than ``n_splits`` (some fold would miss that class entirely).
    """
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError(f"y must be 1-D; got shape {labels.shape}.")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2; got {n_splits}.")
    rng = np.random.default_rng(random_state)

    classes, counts = np.unique(labels, return_counts=True)
    too_small = [(c, n) for c, n in zip(classes.tolist(), counts.tolist()) if n < n_splits]
    if too_small:
        raise ValueError(
            f"every class needs at least n_splits={n_splits} members so each fold "
            f"contains it; too small: "
            + ", ".join(f"class {c!r} has {n}" for c, n in too_small)
            + "."
        )

    val_chunks: list[list[IntArray]] = [[] for _ in range(n_splits)]
    for cls in classes:
        members = np.flatnonzero(labels == cls)
        if shuffle:
            members = rng.permutation(members)
        chunks = np.array_split(members, n_splits)
        for k, chunk in enumerate(chunks):
            val_chunks[k].append(chunk)

    all_idx = np.arange(labels.size)
    folds = []
    for k in range(n_splits):
        val_idx = np.sort(np.concatenate(val_chunks[k]))
        train_mask = np.ones(labels.size, dtype=bool)
        train_mask[val_idx] = False
        folds.append((all_idx[train_mask], val_idx))
    return folds


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------

class SmoteResult(NamedTuple):
    """
    Oversampled data.

    Attributes
    ----------
    X, y:
        The input rows first, in their original order, followed by the
        synthetic minority rows.
    sources:
        Shape ``(n_synthetic, 2)``: for each synthetic row, the input row
        indices of the sample it was interpolated from and of the neighbour
        it was interpolated towards.
    """

    X: FloatArray
    y: npt.NDArray[np.generic]
    sources: IntArray


def smote(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    k_neighbors: int = 5,
    sampling_ratio: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> SmoteResult:
    """
    Oversample the minority class of a binary problem with SMOTE.

    Each synthetic sample is ``x + u · (x_nn − x)``, where ``x`` is a minority
    sample drawn uniformly, ``x_nn`` one of its ``k_neighbors`` nearest
    minority neighbours (Euclidean, excluding itself), and ``u ~ U(0, 1)``.

    Parameters
    ----------
    X:
        Features, shape ``(n_samples, n_features)``.
    y:
        Binary labels, shape ``(n_samples,)``.  The less frequent value is the
        minority class.
    k_neighbors:
        Neighbours considered per minority sample.  Must be smaller than the
        minority count.  Default: 5.
    sampling_ratio:
        Target ``n_minority / n_majority`` after oversampling, in (0, 1].
        Default: 1.0 (balanced).  If the data already meets it, nothing is
        generated.
    random_state:
        Seed or generator.

    Returns
    -------
    SmoteResult

    Raises
    ------
    ValueError
        On shape mismatches, non-binary ``y``, an out-of-range
        ``sampling_ratio``, or too few minority samples for ``k_neighbors``.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X_arr.shape}.")
    if labels.shape != (X_arr.shape[0],):
        raise ValueError(
            f"y must have shape ({X_arr.shape[0]},) to match X; got {labels.shape}."
        )
    if not 0.0 < sampling_ratio <= 1.0:
        raise ValueError(f"sampling_ratio must lie in (0, 1]; got {sampling_ratio}.")
    if k_neighbors < 1:
        raise ValueError(f"k_neighbors must be >= 1; got {k_neighbors}.")

    classes, counts = np.unique(labels, return_counts=True)
    if classes.size != 2:
        raise ValueError(f"smote needs exactly two classes; got {classes.tolist()}.")
    minority = classes[np.argmin(counts)]
    n_min, n_maj = int(counts.min()), int(counts.max())
    if n_min <= k_neighbors:
        raise ValueError(
            f"minority class {minority!r} has {n_min} samples; k_neighbors={k_neighbors} "
            f"needs at least {k_neighbors + 1}."
        )

    rng = np.random.default_rng(random_state)
    n_new = max(0, int(np.ceil(sampling_ratio * n_maj)) - n_min)
    if n_new == 0:
        return SmoteResult(X_arr.copy(), labels.copy(), np.empty((0, 2), dtype=np.intp))

    min_idx = np.flatnonzero(labels == minority)
    M = X_arr[min_idx]
    # Pairwise squared distances within the minority class
    sq = np.sum(M**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * M @ M.T
    np.fill_diagonal(d2, np.inf)
    neighbours = np.argsort(d2, axis=1, kind="stable")[:, :k_neighbors]

    base = rng.integers(0, n_min, size=n_new)
    pick = neighbours[base, rng.integers(0, k_neighbors, size=n_new)]
    gap = rng.random((n_new, 1))
    synthetic = M[base] + gap * (M[pick] - M[base])

    sources = np.stack([min_idx[base], min_idx[pick]], axis=1).astype(np.intp)
    X_out = np.concatenate([X_arr, synthetic])
    y_out = np.concatenate([labels, np.full(n_new, minority, dtype=labels.dtype)])
    return SmoteResult(X_out, y_out, sources)


# ---------------------------------------------------------------------------
# Fold-safe composition
# ---------------------------------------------------------------------------

class Fold(NamedTuple):
    """One cross-validation fold with an oversampled training split."""

    X_train: FloatArray
    y_train: npt.NDArray[np.generic]
    X_val: FloatArray
    y_val: npt.NDArray[np.generic]
    train_idx: IntArray
    val_idx: IntArray
    sources: IntArray
    """Original-row indices (into the full ``X``) of each synthetic sample's pair."""


def oversample_fold(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    train_idx: npt.ArrayLike,
    val_idx: npt.ArrayLike,
    **smote_kwargs: object,
) -> Fold:
    """
    SMOTE the training rows of one fold; return the validation rows as-is.

    Only ``X[train_idx]`` is ever passed to :func:`smote`, so no synthetic
    sample can be derived from a validation row.

    Raises
    ------
    ValueError
        If ``train_idx`` and ``val_idx`` overlap.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    tr = np.asarray(train_idx, dtype=np.intp)
    va = np.asarray(val_idx, dtype=np.intp)
    if np.intersect1d(tr, va).size:
        raise ValueError("train_idx and val_idx overlap; a fold must not validate on training rows.")
    res = smote(X_arr[tr], labels[tr], **smote_kwargs)  # type: ignore[arg-type]
    return Fold(
        X_train=res.X,
        y_train=res.y,
        X_val=X_arr[va].copy(),
        y_val=labels[va].copy(),
        train_idx=tr,
        val_idx=va,
        sources=tr[res.sources],
    )


def iter_folds(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    n_splits: int = 5,
    *,
    shuffle: bool = True,
    random_state: int | None = None,
    oversample: bool = True,
    **smote_kwargs: object,
) -> Iterator[Fold]:
    """
    Stratified folds, each with its training split oversampled in isolation.

    Parameters
    ----------
    X, y:
        Full dataset.
    n_splits, shuffle, random_state:
        Passed to :func:`stratified_kfold`.  ``random_state`` also seeds SMOTE
        (a different stream per fold).
    oversample:
        Set ``False`` to get the same folds without SMOTE, for a baseline.
    **smote_kwargs:
        Passed to :func:`smote` (``k_neighbors``, ``sampling_ratio``).

    Yields
    ------
    Fold
    """
    labels = np.asarray(y)
    X_arr = np.asarray(X, dtype=np.float64)
    seeds = np.random.SeedSequence(random_state).spawn(n_splits)
    for (tr, va), seed in zip(
        stratified_kfold(labels, n_splits, shuffle=shuffle, random_state=random_state), seeds
    ):
        if oversample:
            yield oversample_fold(
                X_arr, labels, tr, va, random_state=np.random.default_rng(seed), **smote_kwargs
            )
        else:
            yield Fold(X_arr[tr].copy(), labels[tr].copy(), X_arr[va].copy(), labels[va].copy(),
                       tr, va, np.empty((0, 2), dtype=np.intp))
