"""Turn one experiment's hardware_log.csv into a chart showing CPU, RAM,
and GPU utilization/memory over the course of training, with vertical
lines marking epoch boundaries.

Usage:
    python scripts/plot_hardware.py bert_baseline
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(experiment_name: str) -> None:
    result_dir = ROOT / "results" / experiment_name
    log_path = result_dir / "hardware_log.csv"
    history_path = result_dir / "history.json"

    if not log_path.exists():
        raise FileNotFoundError(
            f"{log_path} not found. Did you run this experiment with "
            f"monitor_hardware: true (the default)?"
        )

    df = pd.read_csv(log_path)
    has_gpu = df["gpu_util_percent"].notna().any()

    epoch_boundaries = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epoch_boundaries = [record["cumulative_elapsed_seconds"] for record in history]

    n_panels = 3 if has_gpu else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3 * n_panels), sharex=True)
    if n_panels == 2:
        axes = list(axes) + [None]

    axes[0].plot(df["elapsed_seconds"], df["cpu_percent"], color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("CPU (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title(f"Hardware utilization during training — {experiment_name}")

    axes[1].plot(df["elapsed_seconds"], df["ram_percent"], color="#2ca02c", linewidth=1.2)
    axes[1].set_ylabel("RAM (%)")
    axes[1].set_ylim(0, 100)

    if has_gpu:
        ax_gpu = axes[2]
        ax_gpu.plot(
            df["elapsed_seconds"], df["gpu_util_percent"], color="#d62728",
            linewidth=1.2, label="GPU utilization (%)",
        )
        ax_gpu.set_ylabel("GPU utilization (%)")
        ax_gpu.set_ylim(0, 100)

        ax_mem = ax_gpu.twinx()
        ax_mem.plot(
            df["elapsed_seconds"], df["gpu_memory_used_mb"], color="#9467bd",
            linewidth=1.0, linestyle="--", label="GPU memory used (MB)",
        )
        ax_mem.set_ylabel("GPU memory (MB)")

        lines_1, labels_1 = ax_gpu.get_legend_handles_labels()
        lines_2, labels_2 = ax_mem.get_legend_handles_labels()
        ax_gpu.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)
    else:
        fig.text(
            0.5, 0.02,
            "No NVIDIA GPU detected via NVML during this run — showing CPU/RAM only.",
            ha="center", fontsize=9, color="gray",
        )

    for ax in axes:
        if ax is None:
            continue
        for boundary in epoch_boundaries:
            ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8)

    axes[-1 if has_gpu else 1].set_xlabel("Elapsed time (seconds)")
    fig.tight_layout()

    output_path = result_dir / "hardware_usage.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", help="Name of a run under results/, e.g. bert_baseline")
    args = parser.parse_args()
    main(args.experiment_name)
