"""
tests/test_benchmark.py
=======================
hqnn_forge.benchmark.run_benchmark (#199) on small synthetic data: both
models see identical folds and training inputs, nothing from a test fold
reaches training, the reported statistics are the library's own functions of
the per-fold scores, and a run is repeatable without touching the global RNG.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from hqnn_forge import benchmark
from hqnn_forge.benchmark import COLUMNS, BenchmarkResult, run_benchmark, write_csv
from hqnn_forge.evaluation import (
    parameter_efficiency,
    rank_biserial_correlation,
    wilcoxon_signed_rank,
)
from hqnn_forge.models import HybridBinaryClassifier

N_SPLITS = 4


def _hybrid(n_input_features: int) -> nn.Module:
    # 2 x 1 is too small for the restricted-variance init to restrict anything
    # (it warns), and the runner does not care which init the hybrid uses.
    return HybridBinaryClassifier(
        n_input_features,
        2,
        1,
        device_name="default.qubit",
        diff_method="backprop",
        init_strategy="normal",
    )


def _data(seed: int = 0, n: int = 160) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)) * [1.0, 5.0, 0.1, 50.0] + [0.0, 3.0, 0.0, -20.0]
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 1.0).astype(int)
    return X, y


def _run(datasets: dict[str, tuple[np.ndarray, np.ndarray]], **kwargs: Any) -> BenchmarkResult:
    options: dict[str, Any] = dict(
        n_splits=N_SPLITS, max_epochs=2, batch_size=32, smote_kwargs={"k_neighbors": 3}
    )
    options.update(kwargs)
    return run_benchmark(datasets, _hybrid, **options)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every train_model call's inputs and returned history, in call order."""
    calls: list[dict[str, Any]] = []
    real = benchmark.train_model

    def spy(model: nn.Module, loss_fn: Any, optimizer: Any, *tensors: Any, **kw: Any) -> Any:
        state = kw["generator"].get_state().clone()
        history = real(model, loss_fn, optimizer, *tensors, **kw)
        calls.append(
            {
                "model": type(model).__name__,
                "module": model,
                "tensors": [t.clone() for t in tensors],
                "generator": state,
                "options": {k: v for k, v in kw.items() if k != "generator"},
                "history": history,
            }
        )
        return history

    monkeypatch.setattr(benchmark, "train_model", spy)
    return calls


class TestSameFoldsForBoth:
    def test_identical_indices_and_training_inputs(self, captured: list[dict[str, Any]]) -> None:
        result = _run({"a": _data()})
        assert [c["model"] for c in captured] == [
            "HybridBinaryClassifier",
            "ClassicalBaseline",
        ] * N_SPLITS
        for k in range(N_SPLITS):
            hyb, ctl = captured[2 * k], captured[2 * k + 1]
            for got, want in zip(ctl["tensors"], hyb["tensors"]):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            assert torch.equal(ctl["generator"], hyb["generator"])  # same batch order
            assert ctl["options"] == hyb["options"]
            f_h, f_c = result.folds[2 * k], result.folds[2 * k + 1]
            assert (f_h.model, f_c.model) == ("hybrid", "control") and f_h.fold == f_c.fold == k
            for attr in ("train_idx", "val_idx", "test_idx"):
                np.testing.assert_array_equal(getattr(f_h, attr), getattr(f_c, attr))
            assert (f_h.init_seed, f_h.batch_seed, f_h.n_synthetic) == (
                f_c.init_seed,
                f_c.batch_seed,
                f_c.n_synthetic,
            )

    def test_both_are_built_under_the_recorded_init_seed(self) -> None:
        seen: list[int] = []

        def build(n_input_features: int) -> nn.Module:
            seen.append(torch.initial_seed())
            return _hybrid(n_input_features)

        result = run_benchmark(
            {"a": _data()}, build, n_splits=2, max_epochs=1, smote_kwargs={"k_neighbors": 3}
        )
        # The control is classical_baseline(build(...)), so build runs for both.
        assert seen == [f.init_seed for f in result.folds]
        assert seen[0] == seen[1] and seen[2] == seen[3] and seen[0] != seen[2]

    def test_the_control_is_the_matched_baseline(self) -> None:
        records = _run({"a": _data()}).records
        hybrid, control = records
        assert (hybrid["architecture"], control["architecture"]) == (
            "HybridBinaryClassifier",
            "ClassicalBaseline",
        )
        # classical_baseline's bound: within half a width step of the hybrid.
        n_in = 4
        assert abs(hybrid["n_parameters"] - control["n_parameters"]) <= (n_in + 2) / 2

    def test_folds_partition_the_rows(self) -> None:
        X, y = _data()
        result = _run({"a": (X, y)})
        tests = [f.test_idx for f in result.folds if f.model == "hybrid"]
        np.testing.assert_array_equal(np.sort(np.concatenate(tests)), np.arange(y.size))
        for f in result.folds:
            parts = [f.train_idx, f.val_idx, f.test_idx]
            assert sum(p.size for p in parts) == y.size
            np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(y.size))
            # Stratified: each part keeps positives.
            assert all(y[p].sum() > 0 for p in parts)

    def test_threshold_is_the_validation_threshold(self, captured: list[dict[str, Any]]) -> None:
        result = _run({"a": _data()})
        for fold, call in zip(result.folds, captured):
            assert fold.threshold == call["history"].best_threshold
            assert fold.epochs == call["history"].n_epochs
            assert call["options"]["monitor"] == "mcc"

    def test_reported_mcc_is_the_test_rows_at_that_threshold(
        self, captured: list[dict[str, Any]]
    ) -> None:
        # Recompute each fold's score from the trained model: standardise by
        # the training part (train + validation rows), predict the test rows,
        # cut at the recorded threshold and take the MCC from the confusion
        # counts.  At 0.5 instead, at least one fold scores differently, so
        # this also pins that the tuned threshold is the one applied.
        X, y = _data()
        result = _run({"a": (X, y)})
        differs_at_half = False
        for fold, call in zip(result.folds, captured):
            train_part = np.sort(np.concatenate([fold.train_idx, fold.val_idx]))
            X_fold = benchmark._standardise(X, train_part)
            prob = call["module"].predict_proba(
                torch.from_numpy(X_fold[fold.test_idx].astype(np.float32))
            )
            truth = y[fold.test_idx]

            def mcc(pred: np.ndarray, truth: np.ndarray = truth) -> float:
                tp = float(np.sum((pred == 1) & (truth == 1)))
                tn = float(np.sum((pred == 0) & (truth == 0)))
                fp = float(np.sum((pred == 1) & (truth == 0)))
                fn = float(np.sum((pred == 0) & (truth == 1)))
                den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                return 0.0 if den == 0 else (tp * tn - fp * fn) / den

            assert fold.mcc == pytest.approx(mcc((prob >= fold.threshold).numpy()), abs=1e-12)
            differs_at_half |= mcc((prob >= 0.5).numpy()) != pytest.approx(fold.mcc, abs=1e-12)
        assert differs_at_half


class TestNoLeakage:
    def test_test_rows_do_not_reach_training(self, captured: list[dict[str, Any]]) -> None:
        X, y = _data()
        _run({"a": (X, y)})
        first = [c["tensors"] for c in captured[:2]]
        test_rows = _run({"a": (X, y)}).folds[0].test_idx
        captured.clear()

        changed = X.copy()
        changed[test_rows] += 1e3  # would move any statistic computed over them
        _run({"a": (changed, y)})
        for before, after in zip(first, [c["tensors"] for c in captured[:2]]):
            # X_train, y_train, X_val, y_val: scaling, SMOTE and the
            # validation rows depend on the fold's training part only.
            for got, want in zip(after, before):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_scaling_uses_the_training_part_only(self) -> None:
        X = np.array([[0.0, 5.0], [2.0, 5.0], [100.0, 7.0]])
        scaled = benchmark._standardise(X, np.array([0, 1]))
        np.testing.assert_allclose(scaled[:2, 0], [-1.0, 1.0])
        np.testing.assert_allclose(scaled[2], [99.0, 2.0])  # constant column only centred

    def test_oversampling_adds_only_training_rows(self, captured: list[dict[str, Any]]) -> None:
        result = _run({"a": _data()})
        for fold, call in zip(result.folds, captured):
            X_train, y_train = call["tensors"][:2]
            assert fold.n_synthetic > 0
            assert X_train.shape[0] == fold.train_idx.size + fold.n_synthetic
            assert int(y_train.sum()) == int((y_train == 0).sum())  # balanced by SMOTE

    def test_without_oversampling(self, captured: list[dict[str, Any]]) -> None:
        result = _run({"a": _data()}, oversample=False)
        for fold, call in zip(result.folds, captured):
            assert fold.n_synthetic == 0
            assert call["tensors"][0].shape[0] == fold.train_idx.size


class TestReportedStatistics:
    def test_columns_and_values_follow_from_the_fold_scores(self) -> None:
        X, y = _data()
        result = _run({"first": (X, y), "second": _data(seed=1)})
        assert [(r["dataset"], r["model"]) for r in result.records] == [
            ("first", "hybrid"),
            ("first", "control"),
            ("second", "hybrid"),
            ("second", "control"),
        ]
        for record in result.records:
            assert tuple(record) == COLUMNS
            folds = [
                f
                for f in result.folds
                if f.dataset == record["dataset"] and f.model == record["model"]
            ]
            scores = np.array([f.mcc for f in folds])
            assert record["fold_mcc"] == tuple(scores)
            assert record["mcc_mean"] == pytest.approx(scores.mean(), abs=0)
            assert record["mcc_std"] == pytest.approx(scores.std(ddof=1), abs=0)
            assert record["mcc_per_kparam"] == parameter_efficiency(
                record["n_parameters"], record["mcc_mean"]
            )
            assert record["train_seconds"] == pytest.approx(sum(f.train_seconds for f in folds))
            assert record["n_folds"] == N_SPLITS
        first = result.records[0]
        assert (first["n_samples"], first["n_positives"]) == (y.size, int(y.sum()))

    def test_p_value_is_wilcoxon_on_the_per_fold_scores(self) -> None:
        result = _run({"a": _data()}, n_splits=6)
        hybrid, control = result.records
        test = wilcoxon_signed_rank(hybrid["fold_mcc"], control["fold_mcc"])
        for record in (hybrid, control):
            assert record["wilcoxon_p"] == test.p_value
            assert record["wilcoxon_min_p"] == test.min_p_value
            assert record["rank_biserial"] == rank_biserial_correlation(
                hybrid["fold_mcc"], control["fold_mcc"]
            )

    def test_all_folds_tied_gives_nan_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(benchmark, "_fit_and_score", lambda *a, **k: (0.3, 0.5, 0.01, 1))
        hybrid, control = _run({"a": _data()}).records
        assert math.isnan(hybrid["wilcoxon_p"]) and math.isnan(control["wilcoxon_min_p"])
        assert hybrid["rank_biserial"] == 0.0

    def test_five_folds_cannot_reach_five_percent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The limit the module docstring warns about: the hybrid wins every fold.
        scores = iter([0.9, 0.1] * 5)
        monkeypatch.setattr(
            benchmark, "_fit_and_score", lambda *a, **k: (next(scores), 0.5, 0.01, 1)
        )
        hybrid, _ = _run({"a": _data()}, n_splits=5).records
        assert hybrid["wilcoxon_p"] == hybrid["wilcoxon_min_p"] == 0.0625


class TestRepeatability:
    def test_same_seed_same_result(self) -> None:
        a, b = _run({"a": _data()}), _run({"a": _data()})
        strip = lambda r: {k: v for k, v in r.items() if k != "train_seconds"}
        assert [strip(r) for r in a.records] == [strip(r) for r in b.records]
        assert [(f.threshold, f.mcc, f.init_seed) for f in a.folds] == [
            (f.threshold, f.mcc, f.init_seed) for f in b.folds
        ]

    def test_another_seed_other_folds(self) -> None:
        a, b = _run({"a": _data()}), _run({"a": _data()}, random_state=1)
        assert not np.array_equal(a.folds[0].test_idx, b.folds[0].test_idx)

    def test_global_rng_is_neither_read_nor_moved(self) -> None:
        torch.manual_seed(7)
        first = _run({"a": _data()})
        after_first = torch.rand(3)
        torch.manual_seed(99)  # a different global state must not change the run
        second = _run({"a": _data()})
        assert [f.mcc for f in first.folds] == [f.mcc for f in second.folds]
        torch.manual_seed(7)
        torch.testing.assert_close(torch.rand(3), after_first, rtol=0, atol=0)


class TestInputsAndOutput:
    @pytest.mark.parametrize(
        ("datasets", "kwargs", "match"),
        [
            ({}, {}, "datasets is empty"),
            ({"a": _data()}, {"n_splits": 1}, "n_splits must be >= 2"),
            ({"a": _data()}, {"validation_folds": 1}, "validation_folds must be >= 2"),
            ({"a": (_data()[0], _data()[1] * 2)}, {}, "y must be binary"),
            ({"a": (_data()[0][:, 0], _data()[1])}, {}, r"X must be \(n_samples, n_features\)"),
        ],
        ids=["empty", "one-split", "one-validation-fold", "labels", "shape"],
    )
    def test_rejected(self, datasets: dict, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _run(datasets, **kwargs)

    def test_nan_rejected(self) -> None:
        X, y = _data()
        X[3, 1] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            _run({"a": (X, y)})

    def test_builder_must_return_a_classifier(self) -> None:
        build: Callable[[int], nn.Module] = lambda n: nn.Linear(n, 1)
        with pytest.raises(TypeError, match="hybrid must return a hybrid classifier"):
            run_benchmark({"a": _data()}, build, n_splits=2, max_epochs=1)

    def test_csv(self, tmp_path: Path) -> None:
        result = _run({"a": _data()})
        path = tmp_path / "benchmark.csv"
        write_csv(result.records, path)
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [tuple(r) for r in rows] == [COLUMNS, COLUMNS]
        for row, record in zip(rows, result.records):
            assert tuple(float(s) for s in row["fold_mcc"].split(";")) == record["fold_mcc"]
            assert float(row["mcc_mean"]) == record["mcc_mean"]
            assert row["model"] == record["model"]


# ---------------------------------------------------------------------------
# Several initialisation seeds per fold (#206)
# ---------------------------------------------------------------------------


class TestSeveralSeeds:
    def test_same_seeds_same_scores(self) -> None:
        a = _run({"a": _data()}, n_seeds=3)
        b = _run({"a": _data()}, n_seeds=3)
        assert [f.mcc for f in a.folds] == [f.mcc for f in b.folds]
        assert [f.init_seed for f in a.folds] == [f.init_seed for f in b.folds]

    def test_distinct_seeds_give_distinct_initial_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[int, torch.Tensor]] = []
        controls: list[torch.Tensor] = []
        real_baseline = benchmark.classical_baseline

        def baseline(model: nn.Module) -> Any:
            control = real_baseline(model)
            controls.append(
                torch.cat([p.detach().flatten().clone() for p in control.parameters()])
            )
            return control

        monkeypatch.setattr(benchmark, "classical_baseline", baseline)

        def build(n_input_features: int) -> nn.Module:
            model = _hybrid(n_input_features)
            assert isinstance(model, HybridBinaryClassifier)
            weights = model.quantum_layer.qlayer.weights
            assert isinstance(weights, torch.Tensor)
            seen.append((torch.initial_seed(), weights.detach().clone()))
            return model

        result = run_benchmark(
            {"a": _data()},
            build,
            n_splits=2,
            max_epochs=1,
            n_seeds=3,
            smote_kwargs={"k_neighbors": 3},
        )
        # Per fold: 3 hybrid builds, then 3 for the control, the same seeds.
        assert [s for s, _ in seen] == [f.init_seed for f in result.folds]
        for fold in range(2):
            hybrid = seen[6 * fold : 6 * fold + 3]
            control = seen[6 * fold + 3 : 6 * fold + 6]
            assert len({s for s, _ in hybrid}) == 3
            assert [s for s, _ in hybrid] == [s for s, _ in control]
            for i in range(3):
                for j in range(i + 1, 3):
                    assert not torch.equal(hybrid[i][1], hybrid[j][1])
                    control_i, control_j = controls[3 * fold + i], controls[3 * fold + j]
                    assert not torch.equal(control_i, control_j)
        assert len(controls) == 6
        assert [f.seed_index for f in result.folds[:6]] == [0, 1, 2, 0, 1, 2]

    def test_a_seeded_builder_is_refused(self) -> None:
        # A model's own init_seed overrides the runner's: every repeat would
        # start from the same weights and report a spread of zero.
        def build(n_input_features: int) -> nn.Module:
            return HybridBinaryClassifier(
                n_input_features,
                2,
                1,
                device_name="default.qubit",
                init_strategy="normal",
                init_seed=7,
            )

        with pytest.raises(ValueError, match="init_seed=None"):
            run_benchmark({"a": _data()}, build, n_splits=2, max_epochs=1, n_seeds=2)
        # One seed per fold is unaffected.
        run_benchmark(
            {"a": _data()}, build, n_splits=2, max_epochs=1, smote_kwargs={"k_neighbors": 3}
        )

    def test_one_seed_is_the_default_run(self) -> None:
        default, explicit = _run({"a": _data()}), _run({"a": _data()}, n_seeds=1)
        strip = lambda r: {k: v for k, v in r.items() if k != "train_seconds"}
        assert [strip(r) for r in default.records] == [strip(r) for r in explicit.records]
        assert default.records[0]["mcc_seed_std"] is None

    def test_fold_scores_are_seed_means_and_the_test_pairs_folds(self) -> None:
        result = _run({"a": _data()}, n_seeds=3)
        assert len(result.folds) == N_SPLITS * 2 * 3
        for record in result.records:
            per_fold = [
                [f.mcc for f in result.folds if f.model == record["model"] and f.fold == k]
                for k in range(N_SPLITS)
            ]
            assert all(len(scores) == 3 for scores in per_fold)
            assert record["fold_mcc"] == pytest.approx(tuple(np.mean(s) for s in per_fold))
            assert record["mcc_seed_std"] == pytest.approx(
                float(np.mean([np.std(s, ddof=1) for s in per_fold]))
            )
            assert record["n_seeds"] == 3
        hybrid, control = result.records
        try:
            p = wilcoxon_signed_rank(hybrid["fold_mcc"], control["fold_mcc"]).p_value
        except ValueError:
            p = math.nan
        assert (math.isnan(p) and math.isnan(hybrid["wilcoxon_p"])) or hybrid["wilcoxon_p"] == p

    def test_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_seeds must be >= 1"):
            _run({"a": _data()}, n_seeds=0)
