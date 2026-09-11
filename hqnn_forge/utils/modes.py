"""
hqnn_forge.utils.modes
======================
Temporarily switch a module to eval mode.

``torch.no_grad()`` only disables autograd.  Layers such as ``nn.Dropout`` read
``module.training`` instead, so inference code that can run mid-training has to
switch to eval mode itself and put every submodule's mode back afterwards.
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
    modes = [(submodule, submodule.training) for submodule in module.modules()]
    module.eval()
    try:
        yield
    finally:
        # Restore through train() so overrides on submodules still run.
        for submodule, training in modes:
            submodule.train(training)
        # train() recurses, so a submodule registered under two parents can be
        # overwritten by the parent restored after it.  Set the flags last.
        for submodule, training in modes:
            submodule.training = training
