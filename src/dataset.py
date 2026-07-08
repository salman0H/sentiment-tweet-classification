"""Torch Dataset wrapping pre-tokenized tweets for batched training."""

from __future__ import annotations

from typing import List

import torch
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    """Tokenizes lazily and pads dynamically via the collate function below,
    so a single dataset instance can be reused across max-length experiments
    just by pointing it at a different tokenizer/max_length pair.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        encoding["labels"] = self.labels[idx]
        return encoding


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
