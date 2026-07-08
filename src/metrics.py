"""Evaluation metrics required by the assignment: accuracy, precision,
recall and F1, all reported as weighted averages across the three
sentiment classes so the (imbalanced) label distribution is accounted for.
"""

from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from .data import LABEL_NAMES


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def full_report(y_true: List[int], y_pred: List[int]) -> str:
    return classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, zero_division=0
    )
