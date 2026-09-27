"""
Minimal pytest configuration for hqnn-forge.

Skips fail under ``HQNN_FORGE_FAIL_ON_SKIP=1``
----------------------------------------------
Tests that need an optional extra skip without it (``pytest.importorskip``,
``requires_lightning``), which is right for a local checkout but a blind spot
in CI: a missing extra removes tests and the job still goes green (#187).  CI
sets ``HQNN_FORGE_FAIL_ON_SKIP=1``, which turns every skip into a failure that
names the skip's reason, including a module skipped at import.  A test whose
skip is expected in CI, because it needs hardware the runners lack, carries
``@pytest.mark.may_skip``.  Expected failures (``xfail``) are unaffected.

``slow``: the fast local suite
------------------------------
A handful of tests account for most of the suite's run time (end-to-end
training, the gradient-variance physics checks, parameter-shift batching).
They carry ``@pytest.mark.slow``, so ``pytest -m "not slow"`` is the quick
edit-test loop; CI runs everything.  The rule: mark a test that takes
``SLOW_SECONDS`` or more on a laptop (``pytest --durations=30`` to find them).
Deselected tests never run, so they are not skips and the hooks below leave
them alone.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Generator

import pytest
import torch

FAIL_ON_SKIP_ENV = "HQNN_FORGE_FAIL_ON_SKIP"
#: A test taking this long or longer on a laptop is marked ``slow``.
SLOW_SECONDS = 1.2


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "reproducibility: checks against the published benchmark configuration "
        "(deselect with -m 'not reproducibility')",
    )
    config.addinivalue_line(
        "markers",
        f"slow: takes {SLOW_SECONDS} s or more; deselect with -m 'not slow' for a quick local run",
    )
    config.addinivalue_line(
        "markers",
        f"may_skip: the test may skip even under {FAIL_ON_SKIP_ENV}=1, e.g. "
        "because it needs hardware the CI runners do not have",
    )


def _fail_on_skip() -> bool:
    return os.environ.get(FAIL_ON_SKIP_ENV) == "1"


def _skip_failure(what: str, report: pytest.TestReport | pytest.CollectReport) -> str:
    """The failure text for a skip: what skipped, why, and how to allow it."""
    longrepr = report.longrepr
    # A skip's longrepr is (path, lineno, "Skipped: <reason>").
    reason = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
    return f"{what} skipped, and {FAIL_ON_SKIP_ENV}=1 makes a skip fail.\n{reason}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if (
        _fail_on_skip()
        and report.skipped
        and not hasattr(report, "wasxfail")
        and item.get_closest_marker("may_skip") is None
    ):
        report.outcome = "failed"
        report.longrepr = (
            _skip_failure(item.nodeid, report)
            + "\nMark the test `may_skip` if the skip is expected in CI."
        )
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    # A module-level importorskip skips the whole module at import, before a
    # marker could be read, so there is nothing to exempt it with.
    report = yield
    if _fail_on_skip() and report.skipped:
        report.outcome = "failed"
        report.longrepr = _skip_failure(collector.nodeid, report)
    return report


def _grad(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor.grad`` after a backward pass, narrowed from ``Tensor | None``."""
    assert tensor.grad is not None
    return tensor.grad


@pytest.fixture
def grad_of() -> Callable[[torch.Tensor], torch.Tensor]:
    """
    ``_grad`` as a fixture.  Test modules take it as an argument rather than
    importing ``conftest``, which only resolves under pytest's default
    ``prepend`` import mode.
    """
    return _grad
