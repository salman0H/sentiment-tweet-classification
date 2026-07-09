"""Fast pipeline sanity check, run before committing to a full experiment.

A full experiment can take from minutes to hours depending on hardware. If
the config, tokenizer, or model registry entry has a mistake, the cheapest
place to find that out is a run over a tiny slice of data for one epoch --
seconds to low minutes -- not after an entire real run has already burned
the time. This reuses `run_training` unchanged (same code path as the real
run, so a pass here is a meaningful signal) with a throwaway config variant.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import List

from .config import ExperimentConfig
from .trainer import run_training

SMOKE_TEST_SUBDIR = "_smoke"


def _quick_variant(config: ExperimentConfig) -> ExperimentConfig:
    return dataclasses.replace(
        config,
        name=f"{config.name}__smoke",
        epochs=1,
        output_dir=str(Path(config.output_dir) / SMOKE_TEST_SUBDIR),
        monitor_hardware=False,
        early_stopping_patience=None,
        dataloader_num_workers=0,
    )


def run_smoke_test(
    config: ExperimentConfig,
    train_texts: List[str],
    train_labels: List[int],
    dev_texts: List[str],
    dev_labels: List[int],
    n_samples: int = 200,
) -> None:
    """Raises on failure; the caller decides whether to proceed to the real
    run. Cleans up its own scratch output directory afterwards either way.
    """
    quick_config = _quick_variant(config)
    n_train = min(n_samples, len(train_texts))
    n_dev = min(max(n_samples // 4, 8), len(dev_texts))

    quick_dir = Path(quick_config.output_dir) / quick_config.name
    try:
        run_training(
            quick_config,
            train_texts[:n_train],
            train_labels[:n_train],
            dev_texts[:n_dev],
            dev_labels[:n_dev],
        )
    finally:
        if quick_dir.exists():
            shutil.rmtree(quick_dir, ignore_errors=True)
