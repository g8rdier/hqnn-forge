"""
tests/test_cv.py
================
Stratified k-fold and fold-safe SMOTE (hqnn_forge.preprocessing.cv).

The three properties the issue asks for:
(a) class prevalence is preserved per fold,
(b) validation partitions are never oversampled,
(c) no synthetic sample is derived from a row outside its training fold.
"""

from __future__ import annotations

import numpy as np
import pytest

from hqnn_forge.preprocessing import (
    Fold,
    iter_folds,
    oversample_fold,
    smote,
    stratified_kfold,
)


@pytest.fixture
def imbalanced() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, n_pos = 203, 23
    X = rng.standard_normal((n, 4))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    X[y == 1] += 2.0
    return X, y


class TestStratifiedKFold:
    @pytest.mark.parametrize("n_splits", [2, 3, 5, 7])
    def test_prevalence_preserved_per_fold(self, imbalanced: tuple, n_splits: int) -> None:
        _, y = imbalanced
        n_pos = int(y.sum())
        for train, val in stratified_kfold(y, n_splits, random_state=0):
            # Each class is dealt into near-equal chunks: counts differ by at most 1
            assert abs(int(y[val].sum()) - n_pos / n_splits) < 1
            assert abs(int((y[val] == 0).sum()) - (y.size - n_pos) / n_splits) < 1

    def test_partition_properties(self, imbalanced: tuple) -> None:
        _, y = imbalanced
        folds = stratified_kfold(y, 5, random_state=0)
        all_val = np.concatenate([val for _, val in folds])
        assert np.array_equal(np.sort(all_val), np.arange(y.size))  # each sample validated once
        for train, val in folds:
            assert np.intersect1d(train, val).size == 0
            assert train.size + val.size == y.size
            assert np.all(np.diff(train) > 0) and np.all(np.diff(val) > 0)

    def test_shuffle_and_seed(self, imbalanced: tuple) -> None:
        _, y = imbalanced
        a = stratified_kfold(y, 5, random_state=1)
        b = stratified_kfold(y, 5, random_state=1)
        c = stratified_kfold(y, 5, random_state=2)
        assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))
        assert not all(np.array_equal(x[1], z[1]) for x, z in zip(a, c))

    def test_no_shuffle_is_ordered(self) -> None:
        y = np.array([0, 0, 0, 0, 1, 1])
        folds = stratified_kfold(y, 2, shuffle=False)
        assert folds[0][1].tolist() == [0, 1, 4]
        assert folds[1][1].tolist() == [2, 3, 5]

    def test_multiclass_and_string_labels(self) -> None:
        y = np.array(["a"] * 6 + ["b"] * 9 + ["c"] * 3)
        for _, val in stratified_kfold(y, 3, random_state=0):
            assert sorted(np.unique(y[val], return_counts=True)[1].tolist()) == [1, 2, 3]

    def test_class_smaller_than_n_splits(self) -> None:
        with pytest.raises(ValueError, match=r"class 1 has 2"):
            stratified_kfold([0, 0, 0, 0, 1, 1], 3)

    @pytest.mark.parametrize("y, n_splits, match", [([0, 1], 1, "n_splits must be >= 2"), ([[0, 1]], 2, "must be 1-D")])
    def test_bad_arguments(self, y: list, n_splits: int, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            stratified_kfold(y, n_splits)


class TestSmote:
    def test_balances_and_keeps_originals_first(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        res = smote(X, y, random_state=0)
        assert np.array_equal(res.X[: len(X)], X) and np.array_equal(res.y[: len(y)], y)
        assert int((res.y == 1).sum()) == int((res.y == 0).sum())
        assert np.all(res.y[len(y):] == 1)
        assert res.sources.shape == (len(res.y) - len(y), 2)

    def test_synthetic_points_lie_between_their_sources(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        res = smote(X, y, random_state=0)
        synth = res.X[len(X):]
        a, b = X[res.sources[:, 0]], X[res.sources[:, 1]]
        assert np.all(y[res.sources] == 1)
        assert np.all(res.sources[:, 0] != res.sources[:, 1])
        direction = b - a
        u = np.sum((synth - a) * direction, axis=1) / np.sum(direction**2, axis=1)
        assert np.all((u >= 0) & (u <= 1))
        np.testing.assert_allclose(a + u[:, None] * direction, synth, atol=1e-12)

    def test_neighbours_are_among_the_k_nearest(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        k = 3
        res = smote(X, y, k_neighbors=k, random_state=0)
        M_idx = np.flatnonzero(y == 1)
        for src, nb in res.sources:
            d = np.linalg.norm(X[M_idx] - X[src], axis=1)
            d[M_idx == src] = np.inf
            nearest = set(M_idx[np.argsort(d, kind="stable")[:k]].tolist())
            assert nb in nearest

    def test_sampling_ratio(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        res = smote(X, y, sampling_ratio=0.5, random_state=0)
        assert int((res.y == 1).sum()) == int(np.ceil(0.5 * (y == 0).sum()))

    def test_already_balanced_generates_nothing(self) -> None:
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.array([0, 1] * 5)
        res = smote(X, y, k_neighbors=2)
        assert np.array_equal(res.X, X) and res.sources.shape == (0, 2)

    def test_minority_is_whichever_label_is_rarer(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 2))
        y = np.array([1] * 24 + [0] * 6)
        res = smote(X, y, k_neighbors=2, random_state=0)
        assert np.all(res.y[30:] == 0)

    def test_reproducible(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        a, b = smote(X, y, random_state=5), smote(X, y, random_state=5)
        assert np.array_equal(a.X, b.X)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(k_neighbors=30), "needs at least 31"),
            (dict(k_neighbors=0), "k_neighbors must be >= 1"),
            (dict(sampling_ratio=0.0), "sampling_ratio must lie"),
            (dict(sampling_ratio=1.5), "sampling_ratio must lie"),
        ],
    )
    def test_argument_errors(self, imbalanced: tuple, kwargs: dict, match: str) -> None:
        X, y = imbalanced
        with pytest.raises(ValueError, match=match):
            smote(X, y, **kwargs)

    def test_shape_and_class_errors(self) -> None:
        with pytest.raises(ValueError, match="X must be 2-D"):
            smote(np.zeros(4), np.zeros(4))
        with pytest.raises(ValueError, match="to match X"):
            smote(np.zeros((4, 2)), np.zeros(3))
        with pytest.raises(ValueError, match="exactly two classes"):
            smote(np.zeros((6, 2)), np.array([0, 1, 2, 0, 1, 2]))


class TestFoldSafety:
    def test_validation_rows_are_untouched(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        for fold in iter_folds(X, y, 5, random_state=0, k_neighbors=3):
            assert isinstance(fold, Fold)
            assert np.array_equal(fold.X_val, X[fold.val_idx])
            assert np.array_equal(fold.y_val, y[fold.val_idx])
            # prevalence of the validation split is the original one, not balanced
            assert fold.y_val.mean() < 0.2

    def test_training_split_is_oversampled(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        for fold in iter_folds(X, y, 5, random_state=0, k_neighbors=3):
            assert (fold.y_train == 1).sum() == (fold.y_train == 0).sum()
            assert np.array_equal(fold.X_train[: fold.train_idx.size], X[fold.train_idx])

    def test_no_synthetic_sample_comes_from_outside_its_training_fold(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        for fold in iter_folds(X, y, 5, random_state=0, k_neighbors=3):
            assert fold.sources.size > 0
            assert np.all(np.isin(fold.sources, fold.train_idx))
            assert not np.any(np.isin(fold.sources, fold.val_idx))
            # and the stored sources really are the pair each synthetic row was built from
            synth = fold.X_train[fold.train_idx.size:]
            a, b = X[fold.sources[:, 0]], X[fold.sources[:, 1]]
            u = np.sum((synth - a) * (b - a), axis=1) / np.sum((b - a) ** 2, axis=1)
            np.testing.assert_allclose(a + u[:, None] * (b - a), synth, atol=1e-12)

    def test_smote_on_full_data_would_leak(self, imbalanced: tuple) -> None:
        """The failure this module prevents, shown on the same data."""
        X, y = imbalanced
        leaked = smote(X, y, k_neighbors=3, random_state=0)
        _, val = stratified_kfold(y, 5, random_state=0)[0]
        assert np.any(np.isin(leaked.sources, val))

    def test_without_oversampling(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        for fold in iter_folds(X, y, 4, random_state=0, oversample=False):
            assert np.array_equal(fold.X_train, X[fold.train_idx])
            assert fold.sources.shape == (0, 2)

    def test_folds_are_reproducible_and_use_distinct_smote_streams(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        a = list(iter_folds(X, y, 3, random_state=7, k_neighbors=3))
        b = list(iter_folds(X, y, 3, random_state=7, k_neighbors=3))
        for fa, fb in zip(a, b):
            assert np.array_equal(fa.X_train, fb.X_train)
        gaps = [f.X_train[f.train_idx.size:][:3] for f in a]
        assert not np.allclose(gaps[0], gaps[1])

    def test_overlapping_indices_are_rejected(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        with pytest.raises(ValueError, match="overlap"):
            oversample_fold(X, y, np.arange(0, 150), np.arange(100, 203))
