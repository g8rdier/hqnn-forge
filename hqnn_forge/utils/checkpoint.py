"""
hqnn_forge.utils.checkpoint
===========================
Save and reload a hybrid classifier together with its architecture.

``torch.save(model.state_dict())`` alone is not enough to get a model back:
the state dict holds weights, not ``n_qubits``, ``n_layers``,
``encoding_type`` or the other constructor arguments needed to build a module
those weights fit.  A checkpoint written here stores both, plus the class
name and the library version, and ``load_checkpoint`` rebuilds the model and
loads the weights in one call.

Format
------
A ``torch.save`` file containing a plain dict::

    {
        "format_version": 1,
        "hqnn_forge_version": "0.1.0",
        "class_name": "HybridBinaryClassifier",
        "config": {...constructor kwargs...},
        "state_dict": {...},
    }

Only primitives and tensors are stored, so the file loads with
``torch.load(weights_only=True)``: loading a checkpoint never executes code
from it, and only classes in ``hqnn_forge.models`` can be rebuilt.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import torch

import hqnn_forge

#: Bumped whenever the dict layout above changes incompatibly.
FORMAT_VERSION: int = 1

PathLike = str | os.PathLike[str]


def _registry() -> dict[str, type]:
    # Imported lazily: hqnn_forge.models imports hqnn_forge.utils, so a
    # module-level import here would be circular.
    from hqnn_forge import models

    return {name: getattr(models, name) for name in models.__all__ if name != "BinaryClassifierBase"}


def save_checkpoint(model: torch.nn.Module, path: PathLike) -> None:
    """
    Write ``model``'s class, constructor arguments and weights to ``path``.

    Parameters
    ----------
    model:
        A classifier from ``hqnn_forge.models``.
    path:
        Destination file; overwritten if it exists.

    Raises
    ------
    TypeError
        If ``model`` is not one of the library's classifiers.
    """
    class_name = type(model).__name__
    if _registry().get(class_name) is not type(model):
        raise TypeError(
            f"save_checkpoint supports the classifiers in hqnn_forge.models "
            f"({', '.join(sorted(_registry()))}); got {type(model).__module__}.{class_name}."
        )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "hqnn_forge_version": hqnn_forge.__version__,
        "class_name": class_name,
        "config": model.get_config(),  # type: ignore[operator]
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: PathLike,
    *,
    map_location: str | torch.device = "cpu",
    allow_version_mismatch: bool = False,
    **overrides: Any,
) -> torch.nn.Module:
    """
    Rebuild a classifier saved with :func:`save_checkpoint`.

    Parameters
    ----------
    path:
        Checkpoint file.
    map_location:
        Passed to ``torch.load``.  Default: ``"cpu"``.
    allow_version_mismatch:
        Load a checkpoint written by a different ``hqnn_forge`` version.  Off
        by default because weight layouts are not guaranteed stable across
        versions before 1.0.
    **overrides:
        Constructor arguments that replace the stored ones, typically
        ``device_name`` / ``diff_method`` to run on a different simulator.
        Architecture arguments can be overridden too, but the stored weights
        will then fail to load.

    Returns
    -------
    torch.nn.Module
        The rebuilt model with the saved weights, in eval mode.

    Raises
    ------
    ValueError
        On an unknown format version, a library version mismatch (unless
        allowed), an unknown class, or a config with missing or unexpected
        constructor fields.
    RuntimeError
        If the stored weights do not fit the rebuilt architecture.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(f"{os.fspath(path)!r} is not an hqnn_forge checkpoint.")

    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {payload['format_version']} is not supported; "
            f"this hqnn_forge reads format version {FORMAT_VERSION}."
        )

    saved_version = payload.get("hqnn_forge_version")
    if saved_version != hqnn_forge.__version__ and not allow_version_mismatch:
        raise ValueError(
            f"checkpoint was written by hqnn_forge {saved_version}, this is "
            f"{hqnn_forge.__version__}.  Pass allow_version_mismatch=True to load it anyway."
        )

    class_name = payload.get("class_name")
    registry = _registry()
    if class_name not in registry:
        raise ValueError(
            f"checkpoint holds unknown class {class_name!r}; expected one of {sorted(registry)}."
        )
    cls = registry[class_name]

    expected = _init_parameter_names(cls)
    unknown_overrides = sorted(set(overrides) - expected)
    if unknown_overrides:
        raise ValueError(
            f"unknown constructor arguments for {class_name}: {unknown_overrides}."
        )
    config = dict(payload.get("config") or {})
    config.update(overrides)

    missing = sorted(expected - set(config))
    unexpected = sorted(set(config) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(
            f"checkpoint config does not match {class_name}'s constructor: {'; '.join(details)}."
        )

    model = cls(**config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _init_parameter_names(cls: type) -> set[str]:
    """
    Keyword names of ``cls.__init__``.

    A checkpoint must carry every one of them, not only those without a
    default: get_config records them all, and a default that changed between
    versions would otherwise silently change the rebuilt model.
    """
    return {
        name
        for name, p in inspect.signature(cls).parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
