"""
tests/test_sklearn_estimator.py
================================
hqnn_forge.sklearn.HybridClassifierEstimator against scikit-learn tooling.

Skipped where scikit-learn is not installed (optional dependency).

Everything is seeded (``random_state=0``), so the runs are deterministic.  The
accuracy bars (> 0.7 on a linearly separable task) test that the wrapper
trains the model, not how well a 2-qubit model optimises: at seed 0 the fits
score 0.94-0.99, but some other seeds land in a poor optimum (~0.55-0.6), so
the seed is part of the fixture rather than incidental.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.base import clone  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from hqnn_forge.sklearn import HybridClassifierEstimator  # noqa: E402
from hqnn_forge.training import train_model  # noqa: E402

FAST = dict(n_qubits=2, n_layers=1, device_name="default.qubit", diff_method="backprop",
            max_epochs=15, batch_size=16, lr=0.05, loss="bce", random_state=0)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


class TestParams:
    def test_get_set_params_round_trip(self) -> None:
        est = HybridClassifierEstimator(**FAST)
        params = est.get_params()
        assert params["n_qubits"] == 2 and params["model"] == "serial"
        est.set_params(n_layers=3, model="parallel")
        assert est.get_params()["n_layers"] == 3 and est.model == "parallel"
        cloned = clone(est)
        assert cloned.get_params() == est.get_params() and cloned is not est

    def test_init_stores_arguments_unchanged(self) -> None:
        est = HybridClassifierEstimator(threshold=0.3, patience=2)
        assert est.threshold == 0.3 and est.patience == 2
        assert not hasattr(est, "model_")


class TestFitPredict:
    @pytest.mark.parametrize("model", ["serial", "parallel"])
    def test_shapes_and_learning(self, data: tuple, model: str) -> None:
        X, y = data
        est = HybridClassifierEstimator(model=model, **FAST).fit(X, y)
        proba = est.predict_proba(X)
        assert proba.shape == (80, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        pred = est.predict(X)
        assert pred.shape == (80,) and set(pred) <= {0, 1}
        assert est.score(X, y) > 0.7
        assert est.n_features_in_ == 3 and list(est.classes_) == [0, 1]
        assert est.history_.n_epochs == 15

    def test_string_labels_and_positive_class(self, data: tuple) -> None:
        X, y = data
        labels = np.where(y == 1, "fraud", "legit")
        est = HybridClassifierEstimator(**FAST).fit(X, labels)
        assert list(est.classes_) == ["fraud", "legit"]
        # classes_[1] ("legit") is the positive column
        assert est.predict_proba(X).shape == (80, 2)
        assert set(est.predict(X)) <= {"fraud", "legit"}
        assert est.score(X, labels) > 0.7

    def test_reproducible_with_random_state(self, data: tuple) -> None:
        X, y = data
        a = HybridClassifierEstimator(**FAST).fit(X, y).predict_proba(X)
        b = HybridClassifierEstimator(**FAST).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(a, b)

    def test_validation_split_sets_optimal_threshold(self, data: tuple) -> None:
        X, y = data
        est = HybridClassifierEstimator(**{**FAST, "validation_fraction": 0.25, "patience": 5}).fit(X, y)
        assert est.history_.best_threshold is not None
        assert est.threshold_ == pytest.approx(est.history_.best_threshold)
        np.testing.assert_array_equal(
            est.predict(X), est.classes_[(est.predict_proba(X)[:, 1] >= est.threshold_).astype(int)]
        )

    def test_fixed_threshold_and_default_without_validation(self, data: tuple) -> None:
        X, y = data
        default = HybridClassifierEstimator(**FAST).fit(X, y)
        assert default.threshold_ == 0.5
        strict = clone(default).set_params(threshold=0.9).fit(X, y)
        assert strict.threshold_ == 0.9
        # Same seed, same weights: only the threshold differs
        np.testing.assert_array_equal(strict.predict_proba(X), default.predict_proba(X))
        assert strict.predict(X).sum() <= default.predict(X).sum()

    def test_numpy_scalar_threshold_accepted(self, data: tuple) -> None:
        # bool is rejected as a threshold, but a numpy float is a real number
        est = HybridClassifierEstimator(**{**FAST, "threshold": np.float32(0.3)}).fit(*data)
        assert est.threshold_ == pytest.approx(0.3, abs=1e-7)

    def test_patience_default_matches_train_model(self) -> None:
        # The wrapper's own default must not quietly disable the early stopping
        # that validation_fraction pays training samples for.
        assert (
            inspect.signature(HybridClassifierEstimator).parameters["patience"].default
            == inspect.signature(train_model).parameters["patience"].default
        )

    def test_focal_loss_default(self, data: tuple) -> None:
        X, y = data
        est = HybridClassifierEstimator(**{**FAST, "loss": "focal"}).fit(X, y)
        assert est.history_.train_loss[-1] < est.history_.train_loss[0]


class TestSklearnTooling:
    def test_cross_val_score(self, data: tuple) -> None:
        X, y = data
        scores = cross_val_score(HybridClassifierEstimator(**FAST), X, y, cv=3, scoring="matthews_corrcoef")
        assert scores.shape == (3,) and scores.mean() > 0.3

    def test_pipeline(self, data: tuple) -> None:
        X, y = data
        pipe = make_pipeline(StandardScaler(), HybridClassifierEstimator(**FAST)).fit(X * 50 + 7, y)
        assert pipe.score(X * 50 + 7, y) > 0.7

    def test_grid_search(self, data: tuple) -> None:
        X, y = data
        search = GridSearchCV(
            HybridClassifierEstimator(**{**FAST, "max_epochs": 3}),
            {"n_layers": [1, 2]}, cv=2, scoring="accuracy",
        ).fit(X, y)
        assert search.best_params_["n_layers"] in (1, 2)
        assert search.best_estimator_.model_.n_layers == search.best_params_["n_layers"]


class TestErrors:
    def test_not_fitted(self, data: tuple) -> None:
        X, _ = data
        with pytest.raises(NotFittedError):
            HybridClassifierEstimator(**FAST).predict(X)

    def test_feature_count_mismatch(self, data: tuple) -> None:
        X, y = data
        est = HybridClassifierEstimator(**{**FAST, "max_epochs": 1}).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            est.predict(X[:, :2])

    def test_multiclass_rejected(self, data: tuple) -> None:
        X, _ = data
        with pytest.raises(ValueError, match="binary classifier; got 3 classes"):
            HybridClassifierEstimator(**FAST).fit(X, np.arange(80) % 3)

    @pytest.mark.parametrize(
        "params, match",
        [
            (dict(model="deep"), "model must be 'serial' or 'parallel'"),
            (dict(loss="hinge"), "loss must be 'focal' or 'bce'"),
            (dict(validation_fraction=1.0), r"validation_fraction must lie in \[0, 1\)"),
            (dict(validation_fraction=0.001), "leaves no training or no validation samples"),
            (dict(threshold="best"), "threshold must be 'optimal' or a real number"),
            (dict(threshold=True), "threshold must be 'optimal' or a real number"),
            (dict(threshold=1.5), r"threshold must lie in \[0, 1\]"),
            (dict(threshold=-0.1), r"threshold must lie in \[0, 1\]"),
        ],
    )
    def test_bad_parameters_fail_in_fit(self, data: tuple, params: dict, match: str) -> None:
        X, y = data
        with pytest.raises(ValueError, match=match):
            HybridClassifierEstimator(**{**FAST, **params}).fit(X, y)

    def test_failed_refit_keeps_the_previous_fit(self, data: tuple) -> None:
        # The refit fails inside the validation split, after the new model is
        # built: the estimator must not be left holding that untrained model.
        X, y = data
        est = HybridClassifierEstimator(**FAST).fit(X, y)
        trained, history, predictions = est.model_, est.history_, est.predict(X)
        with pytest.raises(ValueError, match="leaves no training or no validation samples"):
            est.set_params(validation_fraction=0.001).fit(X, y)
        assert est.model_ is trained and est.history_ is history
        np.testing.assert_array_equal(est.predict(X), predictions)

    def test_nan_input_rejected(self, data: tuple) -> None:
        X, y = data
        X = X.copy()
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            HybridClassifierEstimator(**FAST).fit(X, y)
