"""Torch Dataset wrapping pre-tokenized tweets for batched training.

Tokenization happens once, eagerly, in a single batched call to the fast
tokenizer instead of once per sample per epoch. The same text is tokenized
identically every time (truncation/max_length are deterministic), so
retokenizing it on every `__getitem__` across every epoch and every
evaluation pass is pure wasted CPU that would otherwise overlap with GPU
compute -- this matters most on runs with many epochs/experiments where the
retokenization cost is paid over and over for no benefit.
"""

from __future__ import annotations

from typing import List

import torch
from torch.utils.data import DataLoader, Dataset


class TweetDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.labels = labels
        # A single batched call lets the Rust fast-tokenizer parallelize
        # internally, which is both faster and avoids the per-item Python
        # call overhead of tokenizing one tweet at a time.
        self._encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {key: values[idx] for key, values in self._encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def make_collate_fn(tokenizer):
    """Dynamic padding per-batch instead of a fixed global length, which
    keeps training noticeably faster than padding every sample to
    `max_length` up front.
    """
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collate(batch):
        labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.long)
        padded = collator(batch)
        padded["labels"] = labels
        return padded

    return collate


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Centralizes DataLoader construction so num_workers/pin_memory are
    applied consistently everywhere a loader is built. Worker processes are
    only worth keeping alive (`persistent_workers`) when there is more than
    one epoch to amortize their startup cost over, and only when
    num_workers > 0 in the first place.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
