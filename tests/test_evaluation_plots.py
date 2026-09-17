"""
tests/test_evaluation_plots.py
===============================
hqnn_forge.evaluation.plots: each figure is built from the right numbers.

Skipped where matplotlib is not installed (it is an optional extra).
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from hqnn_forge.evaluation import plots  # noqa: E402

THESIS = {  # (MCC, params) from the benchmark README
    "SHNN": (0.5758, 122),
    "PHNN": (0.5688, 489),
    "SNN": (0.5633, 3201),
    "TabNet": (0.4824, 6176),
    "ResNet": (0.6933, 8897),
    "FT-T": (0.6934, 14869),
    "SAINT": (0.6975, 29357),
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestConfusionMatrix:
    Y_TRUE = [0, 0, 0, 0, 1, 1, 1]
    Y_PRED = [0, 0, 1, 0, 1, 0, 1]  # tn=3 fp=1 fn=1 tp=2

    def test_counts(self) -> None:
        cm = plots.confusion_matrix(self.Y_TRUE, self.Y_PRED)
        assert cm.tolist() == [[3, 1], [1, 2]]

    def test_matches_scikit_learn(self) -> None:
        sk = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        t, p = rng.integers(0, 2, 40), rng.integers(0, 2, 40)
        assert np.array_equal(plots.confusion_matrix(t, p), sk.confusion_matrix(t, p, labels=[0, 1]))

    def test_figure_shows_every_count(self) -> None:
        fig = plots.plot_confusion_matrix(self.Y_TRUE, self.Y_PRED, labels=("legit", "fraud"))
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert sorted(t.get_text() for t in ax.texts) == ["1", "1", "2", "3"]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["legit", "fraud"]
        assert ax.get_xlabel() == "Predicted" and ax.get_ylabel() == "True"

    def test_normalized_rows(self) -> None:
        fig = plots.plot_confusion_matrix(self.Y_TRUE, self.Y_PRED, normalize=True)
        texts = sorted(t.get_text() for t in fig.axes[0].texts)
        assert texts == ["0.25", "0.33", "0.67", "0.75"]

    def test_normalize_with_empty_row(self) -> None:
        fig = plots.plot_confusion_matrix([0, 0], [0, 1], normalize=True)
        assert sorted(t.get_text() for t in fig.axes[0].texts) == ["0.00", "0.00", "0.50", "0.50"]

    def test_draws_into_given_axes(self) -> None:
        fig, (a, b) = plt.subplots(1, 2)
        assert plots.plot_confusion_matrix([0, 1], [0, 1], ax=b) is fig
        assert len(b.texts) == 4 and len(a.texts) == 0

    @pytest.mark.parametrize("t, p, match", [([0, 1], [0], "differ in length"), ([0, 2], [0, 1], "binary 0/1")])
    def test_errors(self, t: list, p: list, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            plots.plot_confusion_matrix(t, p)


class TestFoldBoxplot:
    SCORES = {"SHNN": [0.55, 0.58, 0.60, 0.57, 0.59], "SNN": [0.56, 0.55, 0.57, 0.56, 0.58]}

    def test_one_box_per_model_in_order(self) -> None:
        fig = plots.plot_fold_metric_boxplot(self.SCORES, metric_name="MCC")
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["SHNN", "SNN"]
        assert ax.get_ylabel() == "MCC"
        medians = [line.get_ydata()[0] for line in ax.lines if line.get_linestyle() == "-" and len(line.get_xdata()) == 2 and line.get_xdata()[0] != line.get_xdata()[1]]
        assert 0.58 in medians and 0.56 in medians

    def test_points_overlaid(self) -> None:
        ax = plots.plot_fold_metric_boxplot(self.SCORES).axes[0]
        offsets = np.concatenate([c.get_offsets() for c in ax.collections])
        assert offsets.shape == (10, 2)
        assert sorted(offsets[:, 1].tolist()) == sorted(sum(self.SCORES.values(), []))

    def test_points_can_be_hidden(self) -> None:
        ax = plots.plot_fold_metric_boxplot(self.SCORES, show_points=False).axes[0]
        assert len(ax.collections) == 0

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plots.plot_fold_metric_boxplot({})
        with pytest.raises(ValueError, match=r"no scores for: \['B'\]"):
            plots.plot_fold_metric_boxplot({"A": [0.1], "B": []})


class TestEfficiencyFrontier:
    def test_pareto_frontier_on_thesis_numbers(self) -> None:
        # SHNN has the fewest parameters.  PHNN, SNN and TabNet score below it
        # with more parameters, so they are dominated.  ResNet, FT-T (0.6934 >
        # 0.6933) and SAINT each buy a higher score with more parameters.
        assert plots.pareto_frontier(THESIS) == ["SHNN", "ResNet", "FT-T", "SAINT"]

    def test_pareto_ties(self) -> None:
        models = {"a": (0.5, 10), "b": (0.5, 10), "c": (0.5, 20)}
        # a and b tie exactly: neither is strictly better, both stay; c is dominated
        assert plots.pareto_frontier(models) == ["a", "b"]

    def test_figure(self) -> None:
        fig = plots.plot_efficiency_frontier(THESIS)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert sorted(t.get_text() for t in ax.texts) == sorted(THESIS)
        (step,) = [line for line in ax.lines if line.get_label() == "Pareto frontier"]
        assert list(step.get_xdata()) == [122, 8897, 14869, 29357]
        points = np.concatenate([c.get_offsets() for c in ax.collections])
        assert points.shape == (len(THESIS), 2)

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plots.plot_efficiency_frontier({})
        with pytest.raises(ValueError, match=r"must be positive.*\['bad'\]"):
            plots.plot_efficiency_frontier({"bad": (0.5, 0)})

    def test_never_shows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*_: object, **__: object) -> None:
            raise AssertionError("plt.show() called")

        monkeypatch.setattr(plt, "show", fail)
        plots.plot_efficiency_frontier(THESIS)
        plots.plot_fold_metric_boxplot({"a": [0.1, 0.2]})
        plots.plot_confusion_matrix([0, 1], [1, 1])
