"""
hqnn_forge.data.credit_card
===========================
Loader for the Kaggle "Credit Card Fraud Detection" dataset, the benchmark the
library's design and the accompanying thesis target.

The CSV (``creditcard.csv``, ~150 MB) is not redistributable and is not
shipped with the package.  Download it once with the Kaggle CLI::

    kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

or call ``load_credit_card_fraud(download=True)``, which runs that command.

Schema: 31 columns -- ``Time``, ``V1`` … ``V28`` (PCA-anonymised), ``Amount``
and the label ``Class`` (1 = fraud).  The published file has 284,807 rows and
492 frauds; ``strict=True`` checks both.

Loading uses NumPy only, like the rest of the library.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
FILE_NAME = "creditcard.csv"
#: Environment variable naming the directory that holds ``creditcard.csv``.
DATA_DIR_ENV = "HQNN_FORGE_DATA"
DEFAULT_DIR = Path("data") / "raw"

FEATURE_NAMES: tuple[str, ...] = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
COLUMNS: tuple[str, ...] = (*FEATURE_NAMES, "Class")
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


class DatasetNotFoundError(FileNotFoundError):
    """The dataset file is missing and downloading was not requested."""


class CreditCardFraud(NamedTuple):
    """
    Attributes
    ----------
    X:
        Features, shape ``(n_samples, n_features)``, float64.
    y:
        Labels, shape ``(n_samples,)``, int64, 1 = fraud.
    feature_names:
        Column name for each column of ``X``.
    """

    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.int64]
    feature_names: tuple[str, ...]


def _download_command(directory: Path) -> list[str]:
    return ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(directory), "--unzip"]


def _resolve_path(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        p = Path(path)
        return p / FILE_NAME if p.is_dir() else p
    base = os.environ.get(DATA_DIR_ENV)
    return (Path(base) if base else DEFAULT_DIR) / FILE_NAME


def _download(target: Path) -> None:
    if shutil.which("kaggle") is None:
        raise DatasetNotFoundError(
            f"{target} does not exist and the Kaggle CLI is not installed.  "
            f"Install it (pip install kaggle), configure ~/.kaggle/kaggle.json, "
            f"then retry, or download manually:\n    {' '.join(_download_command(target.parent))}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        _download_command(target.parent), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    if not target.exists():
        raise RuntimeError(f"Kaggle download finished but {target} was not created.")


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline()
    return [name.strip().strip('"') for name in first.rstrip("\r\n").split(",")]


def load_credit_card_fraud(
    path: str | os.PathLike[str] | None = None,
    *,
    download: bool = False,
    drop_time: bool = False,
    strict: bool = False,
) -> CreditCardFraud:
    """
    Load ``creditcard.csv`` as NumPy arrays.

    Parameters
    ----------
    path:
        The CSV file, or a directory containing ``creditcard.csv``.  Default:
        ``$HQNN_FORGE_DATA/creditcard.csv`` if that variable is set, else
        ``data/raw/creditcard.csv`` relative to the working directory (the
        benchmark repository's layout).
    download:
        If the file is missing, fetch it with the Kaggle CLI into the file's
        directory.  Requires ``kaggle`` on ``PATH`` and configured credentials.
    drop_time:
        Drop the ``Time`` column (seconds since the first transaction), which
        the benchmark excludes as a leakage-prone ordering feature.
    strict:
        Also require the published row count (284,807) and fraud count (492).

    Returns
    -------
    CreditCardFraud

    Raises
    ------
    DatasetNotFoundError
        The file is missing and ``download`` is False.  The message contains
        the Kaggle command that fetches it.
    ValueError
        The file does not have the expected columns, has non-numeric values,
        labels other than 0/1, or (with ``strict``) the wrong row or fraud
        count.
    """
    csv = _resolve_path(path)
    if not csv.exists():
        if not download:
            raise DatasetNotFoundError(
                f"{csv} not found.  Download the Kaggle Credit Card Fraud dataset with:\n"
                f"    {' '.join(_download_command(csv.parent))}\n"
                f"or pass download=True, or point path (or ${DATA_DIR_ENV}) at the file."
            )
        _download(csv)

    header = _read_header(csv)
    if tuple(header) != COLUMNS:
        if len(header) != len(COLUMNS):
            detail = f"expected {len(COLUMNS)} columns, found {len(header)}"
        else:
            diffs = [f"{i}: {got!r} != {want!r}" for i, (got, want) in enumerate(zip(header, COLUMNS)) if got != want]
            detail = "column names differ at " + ", ".join(diffs[:5]) + (" …" if len(diffs) > 5 else "")
        raise ValueError(f"{csv} does not look like the Kaggle creditcard.csv: {detail}.")

    try:
        data = np.loadtxt(csv, delimiter=",", skiprows=1, quotechar='"', dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"{csv} contains a value that is not a number: {exc}") from exc
    if data.shape[1] != len(COLUMNS):
        raise ValueError(f"{csv}: rows have {data.shape[1]} values, expected {len(COLUMNS)}.")

    labels = data[:, -1]
    if not np.isin(labels, (0.0, 1.0)).all():
        bad = np.unique(labels[~np.isin(labels, (0.0, 1.0))])[:5]
        raise ValueError(f"{csv}: Class must be 0 or 1; found {bad.tolist()}.")
    y = labels.astype(np.int64)
    X = data[:, :-1]
    names = FEATURE_NAMES

    if strict:
        if X.shape[0] != EXPECTED_ROWS:
            raise ValueError(f"{csv}: expected {EXPECTED_ROWS} rows, found {X.shape[0]}.")
        if int(y.sum()) != EXPECTED_FRAUDS:
            raise ValueError(f"{csv}: expected {EXPECTED_FRAUDS} frauds, found {int(y.sum())}.")

    if drop_time:
        X = X[:, 1:]
        names = names[1:]
    return CreditCardFraud(np.ascontiguousarray(X), y, names)
