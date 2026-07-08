"""Run once to create the fixed train/dev/test split used by every
experiment in this project. Safe to re-run: if data/splits.json already
exists it is left untouched, so the split never silently changes.

Usage:
    python scripts/prepare_splits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_or_load_splits, load_raw_corpus

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    texts, labels = load_raw_corpus(ROOT / "data" / "train_text.txt", ROOT / "data" / "train_labels.txt")
    splits = build_or_load_splits(
        n_samples=len(texts),
        labels=labels,
        split_path=ROOT / "data" / "splits.json",
    )
    print(f"train: {len(splits.train)}  dev: {len(splits.dev)}  test: {len(splits.test)}")
    print(f"saved to {ROOT / 'data' / 'splits.json'}")


if __name__ == "__main__":
    main()
