"""
hqnn_forge.utils.modes
======================
Temporarily switch a module to eval or train mode.

``torch.no_grad()`` only disables autograd.  Layers such as ``nn.Dropout`` read
``module.training`` instead, so inference code that can run mid-training has to
switch to eval mode itself and put every submodule's mode back afterwards.
Training code has the mirror problem: ``module.train()`` recurses, and would
unfreeze a submodule the caller put in eval mode on purpose, such as a
batch-norm layer whose statistics are frozen while a head is fine-tuned.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def eval_mode(module: nn.Module) -> Iterator[None]:
    """
    Put ``module`` and all its submodules in eval mode for the ``with`` block.

    On exit, including when the block raises, every submodule gets back the
    ``training`` flag it had on entry.  A single ``module.train(was_training)``
    would not do that: it overwrites submodules the caller had put in a
    different mode, such as a model in train mode with its dropout frozen.

    Parameters
    ----------
    module:
        Module to switch.

    Examples
    --------
    >>> with eval_mode(model):
    ...     logits = model(x)
    """
    modes = _modes(module)
    module.eval()
    try:
        yield
    finally:
        _restore(modes)


@contextmanager
def train_mode(module: nn.Module) -> Iterator[None]:
    """
    Train ``module`` for the ``with`` block without unfreezing what the caller froze.

    If any submodule is in train mode on entry, every submodule keeps the mode
    it has: a model in train mode with a batch-norm layer put in eval mode
    trains with that layer still frozen, and so does a model put in eval mode
    with only its head switched back to train mode.  Only if ``module`` and all
    its submodules are in eval mode -- a model fresh from ``load_checkpoint``
    or ``predict`` code, say -- is there no training configuration to respect,
    and every submodule is put in train mode.  On exit, including when the block raises, every submodule
    gets back the ``training`` flag it had on entry, as with :func:`eval_mode`.

    Parameters
    ----------
    module:
        Module to switch.

    Examples
    --------
    >>> with train_mode(model):
    ...     loss = loss_fn(model(x), y)
    """
    modes = _modes(module)
    if not any(training for _, training in modes):
        module.train()
    try:
        yield
    finally:
        _restore(modes)


def _modes(module: nn.Module) -> list[tuple[nn.Module, bool]]:
    return [(submodule, submodule.training) for submodule in module.modules()]


def _restore(modes: list[tuple[nn.Module, bool]]) -> None:
    # Restore through train() so overrides on submodules still run.
    for submodule, training in modes:
        submodule.train(training)
    # train() recurses, so a submodule registered under two parents can be
    # overwritten by the parent restored after it.  Set the flags last.
    for submodule, training in modes:
        submodule.training = training
