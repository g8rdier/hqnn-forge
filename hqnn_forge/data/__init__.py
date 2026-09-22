"""
hqnn_forge.data
===============
Dataset loaders.

Exported symbols
----------------
load_credit_card_fraud   Kaggle Credit Card Fraud Detection (mlg-ulb/creditcardfraud).
CreditCardFraud          (X, y, feature_names) returned by the loader.
DatasetNotFoundError     Raised when the CSV is absent and download=False.
DatasetDownloadError     Raised when download=True but fetching the CSV fails.
"""

from hqnn_forge.data.credit_card import (
    CreditCardFraud,
    DatasetDownloadError,
    DatasetNotFoundError,
    load_credit_card_fraud,
)

__all__: list[str] = [
    "CreditCardFraud",
    "DatasetDownloadError",
    "DatasetNotFoundError",
    "load_credit_card_fraud",
]
