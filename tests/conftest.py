"""
Minimal pytest configuration for hqnn-forge.

Skips fail under ``HQNN_FORGE_FAIL_ON_SKIP=1``
----------------------------------------------
Tests that need an optional extra skip without it (``pytest.importorskip``,
``requires_lightning``), which is right for a local checkout but a blind spot
in CI: a missing extra removes tests and the job still goes green (#187).  CI
sets ``HQNN_FORGE_FAIL_ON_SKIP=1``, which turns every skip into a failure that
names the skip's reason, including a module skipped at import.  A test whose
skip is expected in CI, because it needs hardware the runners lack or is
opt-in (``HQNN_FORGE_REPRODUCE=1`` and a dataset CI does not have), carries
``@pytest.mark.may_skip``.  Expected failures (``xfail``) are unaffected.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Generator

import pytest
import torch

FAIL_ON_SKIP_ENV = "HQNN_FORGE_FAIL_ON_SKIP"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "reproducibility: checks against the published benchmark configuration "
        "(deselect with -m 'not reproducibility')",
    )
    config.addinivalue_line(
        "markers", "slow: takes minutes to days; opt-in (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers",
        f"may_skip: the test may skip even under {FAIL_ON_SKIP_ENV}=1, e.g. "
        "because it needs hardware the CI runners do not have or is opt-in",
    )
    config.addinivalue_line(
        "markers",
        "requires_lightning: skip unless pennylane-lightning can create a lightning.qubit device",
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


@functools.cache
def _lightning_available() -> bool:
    import pennylane as qml

    try:
        qml.device("lightning.qubit", wires=1)
    except Exception:  # noqa: BLE001 - any failure means "not installed"
        return False
    return True


def pytest_runtest_setup(item: pytest.Item) -> None:
    # The one shared lightning check.  A marker rather than an importable
    # skipif object: test modules do not import conftest (see grad_of below),
    # and ``pytest.mark.requires_lightning`` works on functions, classes and
    # pytest.param alike.
    if item.get_closest_marker("requires_lightning") is not None and not _lightning_available():
        pytest.skip("pennylane-lightning not installed")


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


def _cnot_pairs(target: object) -> list[tuple[int, int]]:
    """
    CNOT ``(control, target)`` pairs, in circuit order, of a tape or of an
    encoding layer's circuit decomposed to ``LOGICAL_GATE_SET`` (zero inputs,
    the layer's current weights).  Order and direction are what a wire-pattern
    test pins; gate counts are invariant under a reversed ring or a permuted
    wire order (#183).
    """
    import pennylane as qml

    from hqnn_forge.diagnostics.circuit import _logical_tape

    if isinstance(target, qml.tape.QuantumScript):
        tape = target
    else:
        tape = _logical_tape(target)  # type: ignore[arg-type]
    return [(int(op.wires[0]), int(op.wires[1])) for op in tape.operations if op.name == "CNOT"]


@pytest.fixture
def cnot_pairs() -> Callable[[object], list[tuple[int, int]]]:
    """``_cnot_pairs`` as a fixture, for the same reason as ``grad_of``."""
    return _cnot_pairs
