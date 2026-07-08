"""Run a single experiment end to end: load data, clean text, fine-tune,
evaluate on the dev set, and write results under results/<experiment_name>/.

Usage:
    python scripts/run_experiment.py configs/bert_baseline.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.data import apply_split, build_or_load_splits, load_raw_corpus
from src.preprocessing import clean_corpus
from src.trainer import evaluate_on_texts, run_training

ROOT = Path(__file__).resolve().parents[1]


def main(config_path: str) -> None:
    config = ExperimentConfig.from_yaml(config_path)

    texts, labels = load_raw_corpus(ROOT / config.text_path, ROOT / config.label_path)
    splits = build_or_load_splits(
        n_samples=len(texts),
        labels=labels,
        split_path=ROOT / config.split_path,
        dev_fraction=config.dev_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )

    cleaned_texts = clean_corpus(texts, config.preprocessing)

    train_texts, train_labels = apply_split(cleaned_texts, labels, splits.train)
    dev_texts, dev_labels = apply_split(cleaned_texts, labels, splits.dev)

    print(f"Running experiment '{config.name}'")
    print(f"  train={len(train_texts)}  dev={len(dev_texts)}  model={config.model_key}")

    outcome = run_training(config, train_texts, train_labels, dev_texts, dev_labels)

    dev_metrics, dev_report, _, _ = evaluate_on_texts(
        outcome["model"], outcome["tokenizer"], config, dev_texts, dev_labels
    )

    result_dir = ROOT / config.output_dir / config.name
    (result_dir / "dev_metrics.json").write_text(json.dumps(dev_metrics, indent=2), encoding="utf-8")
    (result_dir / "dev_classification_report.txt").write_text(dev_report, encoding="utf-8")

    print("\nFinal dev metrics:")
    print(json.dumps(dev_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to an experiment YAML config")
    args = parser.parse_args()
    main(args.config)
