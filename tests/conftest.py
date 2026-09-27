"""Minimal pytest configuration for hqnn-forge."""

from __future__ import annotations

import functools
from collections.abc import Callable

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "reproducibility: checks against the published benchmark configuration "
        "(deselect with -m 'not reproducibility')",
    )
    config.addinivalue_line(
        "markers",
        "requires_lightning: skip unless pennylane-lightning can create a lightning.qubit device",
    )


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
