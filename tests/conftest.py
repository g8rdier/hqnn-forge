"""Minimal pytest configuration for hqnn-forge."""

from __future__ import annotations

from collections.abc import Callable

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
        tape = _logical_tape(target.qlayer, target.n_qubits)  # type: ignore[attr-defined]
    return [(int(op.wires[0]), int(op.wires[1])) for op in tape.operations if op.name == "CNOT"]


@pytest.fixture
def cnot_pairs() -> Callable[[object], list[tuple[int, int]]]:
    """``_cnot_pairs`` as a fixture, for the same reason as ``grad_of``."""
    return _cnot_pairs
