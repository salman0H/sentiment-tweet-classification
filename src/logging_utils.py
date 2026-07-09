"""Structured logging: every experiment gets its own log file under
`results/<name>/train.log`, in addition to console output, so the dashboard
can tail a specific experiment's progress independently of everything else
that's happening in the same process.
"""

from __future__ import annotations

import logging
from pathlib import Path


def get_experiment_logger(name: str, output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"experiment.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Re-configuring the same logger across repeated calls (e.g. smoke test
    # followed by the real run) would otherwise stack duplicate handlers.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
