"""Collect every experiment's dev-set metrics into a single comparison
table (markdown + CSV), for the "Results" section of the report.

Usage:
    python scripts/compare_results.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def main() -> None:
    rows = []
    for experiment_dir in sorted(RESULTS_DIR.iterdir()):
        metrics_path = experiment_dir / "dev_metrics.json"
        config_path = experiment_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))

        row = {
            "experiment": experiment_dir.name,
            "model": config["model_key"],
            "lr": config["learning_rate"],
            "optimizer": config["optimizer"],
            "epochs": config["epochs"],
            "max_length": config["max_length"],
            "frozen_encoder": config["freeze_encoder"],
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
        }

        hardware_log_path = experiment_dir / "hardware_log.csv"
        if hardware_log_path.exists():
            hw_df = pd.read_csv(hardware_log_path)
            if hw_df["gpu_util_percent"].notna().any():
                row["avg_gpu_util_%"] = round(hw_df["gpu_util_percent"].mean(), 1)
                row["peak_gpu_mem_mb"] = round(hw_df["gpu_memory_used_mb"].max(), 0)
            row["avg_cpu_%"] = round(hw_df["cpu_percent"].mean(), 1)
            row["train_wall_seconds"] = round(hw_df["elapsed_seconds"].max(), 1)

        rows.append(row)

    if not rows:
        print("No completed experiments found under results/.")
        return

    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))

    df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    (RESULTS_DIR / "comparison_table.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(f"\nSaved to {RESULTS_DIR / 'comparison_table.csv'} and comparison_table.md")


if __name__ == "__main__":
    main()
