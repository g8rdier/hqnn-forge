"""
hqnn_forge.models.base
======================
Shared inference and bookkeeping API for the binary classifiers.

Every hybrid classifier in this package is an ``nn.Module`` whose ``forward``
returns one raw logit per sample, shape ``(batch, 1)``.  Everything downstream
of that logit -- probabilities, thresholded labels, parameter counting -- is
the same for all of them and lives here once, so a fix applies to every model
rather than to whichever copy happened to be found (cf. #59, which had to be
fixed twice).

Subclasses implement ``__init__`` and ``forward`` only.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hqnn_forge.utils.modes import eval_mode


class BinaryClassifierBase(nn.Module):
    """
    Base class for hybrid binary classifiers.

    Contract for subclasses
    -----------------------
    ``forward(x)`` takes ``(batch, n_input_features)`` and returns raw logits of
    shape ``(batch, 1)``.  ``predict_proba`` and ``predict`` are derived from it
    and must not be overridden to keep the two models interchangeable.

    Methods
    -------
    predict_proba(x)
        Sigmoid of the logits, shape ``(batch,)``, computed in eval mode.
    predict(x, threshold=0.5)
        ``predict_proba(x) >= threshold`` as ``torch.long``.
    count_parameters(trainable_only=True)
        Total number of (trainable) parameters, quantum and classical.
    get_config()
        The constructor arguments, so ``type(model)(**model.get_config())``
        rebuilds an equivalent architecture.  Subclasses record them in
        ``self._config`` at the top of ``__init__``.
    """

    _config: dict[str, Any] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(x) -> logits of shape (batch, 1)."
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute positive-class probabilities (inference mode, no gradients).

        Runs in eval mode whatever mode the model is in, so dropout is off and
        repeated calls on the same input agree.  Every submodule's ``training``
        flag is restored afterwards, so calling this mid-training leaves the
        model exactly as it was.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.

        Returns
        -------
        torch.Tensor
            Probability of class 1, shape ``(batch_size,)``, values ∈ [0, 1].
        """
        # no_grad alone leaves nn.Dropout active: it checks self.training, not
        # grad mode.
        with eval_mode(self):
            logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary labels.  Runs in eval mode, like ``predict_proba``.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.
        threshold:
            Decision threshold.  Default: 0.5.
            For imbalanced datasets consider tuning via ROC/PR curves.

        Returns
        -------
        torch.Tensor
            Binary label tensor of shape ``(batch_size,)``, dtype ``torch.long``.
        """
        return (self.predict_proba(x) >= threshold).long()

    # ------------------------------------------------------------------
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total parameter count (quantum + classical)."""
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        """
        Constructor arguments of this model, as a fresh dict.

        ``type(model)(**model.get_config())`` builds a model with the same
        architecture (weights are re-initialised; load a ``state_dict`` for
        those).  Used by ``hqnn_forge.utils.checkpoint``.
        """
        if self._config is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not record its constructor arguments; "
                f"set self._config in __init__."
            )
        return dict(self._config)
