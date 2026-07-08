"""
Loading the raw corpus and building a single, fixed train/dev/test split
that every experiment in this project reuses.

The assignment requires all model/hyperparameter decisions to be made on
the development set only, with the test set touched exactly once for the
final report. The safest way to guarantee that in practice is to compute
the split a single time, persist the resulting indices to disk, and have
every script load those same indices instead of re-splitting the data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

LABEL_NAMES = ("negative", "neutral", "positive")


@dataclass
class SplitIndices:
    train: List[int]
    dev: List[int]
    test: List[int]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SplitIndices":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


def load_raw_corpus(text_path: Path, label_path: Path) -> Tuple[List[str], List[int]]:
    texts = text_path.read_text(encoding="utf-8").splitlines()
    labels = [int(line) for line in label_path.read_text(encoding="utf-8").splitlines()]

    if len(texts) != len(labels):
        raise ValueError(
            f"Text/label count mismatch: {len(texts)} texts vs {len(labels)} labels"
        )
    return texts, labels


def build_or_load_splits(
    n_samples: int,
    labels: List[int],
    split_path: Path,
    dev_fraction: float = 0.10,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Return cached splits if they exist, otherwise create and cache them.

    Splitting is stratified by label so that the class balance of the
    original corpus (roughly 16% negative / 45% neutral / 39% positive)
    is preserved across train, dev and test.
    """
    if split_path.exists():
        return SplitIndices.load(split_path)

    all_indices = list(range(n_samples))

    train_idx, holdout_idx = train_test_split(
        all_indices,
        test_size=dev_fraction + test_fraction,
        random_state=seed,
        stratify=labels,
    )
    holdout_labels = [labels[i] for i in holdout_idx]
    relative_test_size = test_fraction / (dev_fraction + test_fraction)
    dev_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test_size,
        random_state=seed,
        stratify=holdout_labels,
    )

    splits = SplitIndices(train=train_idx, dev=dev_idx, test=test_idx)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    splits.save(split_path)
    return splits


def apply_split(texts: List[str], labels: List[int], indices: List[int]):
    return [texts[i] for i in indices], [labels[i] for i in indices]
