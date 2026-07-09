"""Evaluate exactly one model -- the experiment with the best dev F1 -- on
the held-out test split, exactly once.

The assignment (Project.pdf) is explicit: hyperparameter/model selection
must only use the dev set, and the test set is touched a single time for
the final report. This script enforces that in practice, not just in
intent: it writes a lock file recording which experiment's test score was
already reported, and refuses to silently re-evaluate on a repeat run
unless the winning experiment has actually changed (e.g. after adding more
experiments to the sweep).

Usage:
    python scripts/final_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_results import build_comparison_table  # noqa: E402

from src.config import ExperimentConfig  # noqa: E402
from src.data import apply_split, build_or_load_splits, load_raw_corpus  # noqa: E402
from src.model import load_model, load_tokenizer  # noqa: E402
from src.preprocessing import clean_corpus  # noqa: E402
from src.trainer import evaluate_on_texts  # noqa: E402

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
LOCK_PATH = RESULTS_DIR / ".test_set_evaluated.json"


def _load_lock() -> dict | None:
    if not LOCK_PATH.exists():
        return None
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def run_final_report(results_dir: Path = RESULTS_DIR, force: bool = False) -> dict | None:
    comparison = build_comparison_table(results_dir)
    if comparison is None or comparison.empty:
        print("No completed experiments to select a winner from yet.")
        return None

    best_row = comparison.iloc[0]
    winner_name = best_row["experiment"]

    lock = _load_lock()
    if lock is not None and lock.get("experiment") == winner_name and not force:
        print(
            f"Test set was already evaluated once for '{winner_name}' at {lock['evaluated_at']} "
            f"-- not touching it again (this is required by the assignment). "
            f"Pass force=True / --force if you intentionally want to re-evaluate."
        )
        return json.loads((results_dir / "final_test_report.json").read_text(encoding="utf-8"))

    if lock is not None and lock.get("experiment") != winner_name:
        print(
            f"Warning: the best dev-F1 experiment changed from '{lock['experiment']}' to "
            f"'{winner_name}' since the last test-set evaluation. Re-evaluating on test for "
            f"the new winner (still exactly once for this experiment)."
        )

    winner_dir = results_dir / winner_name
    config = ExperimentConfig.from_dict(json.loads((winner_dir / "config.json").read_text(encoding="utf-8")))

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
    test_texts, test_labels = apply_split(cleaned_texts, labels, splits.test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(config.model_key)
    model = load_model(config.model_key, freeze_encoder=config.freeze_encoder)
    model.load_state_dict(torch.load(winner_dir / "best_model.pt", map_location=device))
    model.to(device)

    print(f"Evaluating '{winner_name}' on the held-out test set ({len(test_texts)} examples)...")
    test_metrics, test_report, _, _ = evaluate_on_texts(model, tokenizer, config, test_texts, test_labels)

    payload = {
        "experiment": winner_name,
        "dev_f1": float(best_row["f1"]),
        "test_metrics": test_metrics,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    (results_dir / "final_test_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (results_dir / "final_test_report.md").write_text(
        f"# Final test-set report\n\n"
        f"**Winning experiment (selected on dev F1):** `{winner_name}`\n\n"
        f"**Dev F1:** {best_row['f1']:.4f}\n\n"
        f"## Test metrics\n\n"
        f"| accuracy | precision | recall | f1 |\n|---|---|---|---|\n"
        f"| {test_metrics['accuracy']:.4f} | {test_metrics['precision']:.4f} "
        f"| {test_metrics['recall']:.4f} | {test_metrics['f1']:.4f} |\n\n"
        f"## Classification report\n\n```\n{test_report}\n```\n",
        encoding="utf-8",
    )
    LOCK_PATH.write_text(
        json.dumps({"experiment": winner_name, "evaluated_at": payload["evaluated_at"]}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if already locked")
    args = parser.parse_args()
    run_final_report(force=args.force)
