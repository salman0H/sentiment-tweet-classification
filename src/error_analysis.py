"""Helpers for inspecting where a model goes wrong, used in the
"error analysis" section of the report.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from .data import LABEL_NAMES


def build_error_table(
    texts: List[str], y_true: List[int], y_pred: List[int], max_rows: int = 30
) -> pd.DataFrame:
    rows = [
        {
            "text": text,
            "true_label": LABEL_NAMES[t],
            "predicted_label": LABEL_NAMES[p],
        }
        for text, t, p in zip(texts, y_true, y_pred)
        if t != p
    ]
    df = pd.DataFrame(rows)
    return df.head(max_rows)


def confusion_summary(y_true: List[int], y_pred: List[int]) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES)
