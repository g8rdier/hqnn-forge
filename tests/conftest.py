"""Minimal pytest configuration for hqnn-forge."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "reproducibility: checks against the published benchmark configuration "
        "(deselect with -m 'not reproducibility')",
    )


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


@pytest.fixture(autouse=True)
def _fresh_device_fallback() -> Iterator[None]:
    """
    Forget backends that failed to initialise before and after every test.
    The library remembers them for the whole process, so a test that fakes a
    failing ``qml.device`` would otherwise leave that backend marked as
    failed for every test after it.
    """
    from hqnn_forge.encoding.angle_embedding import reset_device_fallback

    reset_device_fallback()
    yield
    reset_device_fallback()
