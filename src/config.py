"""Single source of truth for what defines one experiment.

Every axis the assignment asks us to study (model choice, learning rate,
epochs, optimizer, preprocessing, max sequence length, fine-tuning
strategy) is a field here, so a full experiment is fully described by one
small YAML file and its name doubles as the run's identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .preprocessing import PreprocessingConfig


@dataclass
class ExperimentConfig:
    name: str

    # data
    text_path: str = "data/train_text.txt"
    label_path: str = "data/train_labels.txt"
    split_path: str = "data/splits.json"
    dev_fraction: float = 0.10
    test_fraction: float = 0.15
    seed: int = 42

    # model
    model_key: str = "bert"
    freeze_encoder: bool = False
    max_length: int = 64

    # optimization
    optimizer: str = "adamw"          # adamw | sgd
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 32
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    mixed_precision: bool = True      # automatic mixed precision (GPU only, ignored on CPU)

    # hardware monitoring
    monitor_hardware: bool = True
    monitor_interval_seconds: float = 1.0

    # preprocessing
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    # output
    output_dir: str = "results"

    _FLOAT_FIELDS = (
        "dev_fraction",
        "test_fraction",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "max_grad_norm",
        "monitor_interval_seconds",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw["preprocessing"] = PreprocessingConfig(**raw.get("preprocessing", {}))
        # PyYAML reads unquoted values like "2e-5" as strings, not floats,
        # because they are missing a decimal point. Cast explicitly so a
        # config author doesn't have to remember to write "2.0e-5" instead.
        for field_name in cls._FLOAT_FIELDS:
            if field_name in raw:
                raw[field_name] = float(raw[field_name])
        return cls(**raw)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
