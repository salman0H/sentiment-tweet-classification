"""Turn `results/comparison_table.csv` into a short, human-readable
findings summary -- a first draft of the report's "Results analysis" and
"Conclusion" sections (Project.pdf sections د and و), generated
automatically instead of requiring a manual pass over the numbers.

Every experiment in configs/ varies exactly one axis relative to
`bert_baseline`; this script diffs each experiment's dev metrics against
the baseline row and states the direction and size of the effect.

Usage:
    python scripts/conclusions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_results import build_comparison_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
BASELINE_NAME = "bert_baseline"

# What each experiment name is expected to vary, for a readable label. Any
# experiment not listed here still gets compared to the baseline, just with
# a generic label.
AXIS_LABELS = {
    "roberta": "model architecture -> RoBERTa",
    "distilbert": "model architecture -> DistilBERT",
    "albert": "model architecture -> ALBERT",
    "bert_lr_high": "learning rate 2e-5 -> 5e-5",
    "bert_lr_low": "learning rate 2e-5 -> 1e-5",
    "bert_sgd_optimizer": "optimizer AdamW -> SGD",
    "bert_max_len_128": "max sequence length 64 -> 128",
    "bert_frozen_encoder": "fine-tuning strategy -> frozen encoder (head-only)",
    "bert_preprocessing_aggressive": "preprocessing -> aggressive (lowercase, mention/URL masking, char collapsing)",
}


def _fmt_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}"


def build_findings(results_dir: Path = RESULTS_DIR) -> str:
    df = build_comparison_table(results_dir)
    if df is None or df.empty:
        return "No completed experiments yet -- run the pipeline first.\n"

    lines = ["# Findings (auto-generated draft)\n"]

    best = df.iloc[0]
    lines.append(
        f"**Best experiment by dev F1:** `{best['experiment']}` "
        f"(f1={best['f1']:.4f}, accuracy={best['accuracy']:.4f})\n"
    )

    baseline_rows = df[df["experiment"] == BASELINE_NAME]
    if baseline_rows.empty:
        lines.append(
            "_No `bert_baseline` run found -- per-axis comparisons below are skipped; "
            "run `configs/bert_baseline.yaml` to enable them._\n"
        )
        base = None
    else:
        base = baseline_rows.iloc[0]
        lines.append(
            f"**Baseline (`bert_baseline`):** f1={base['f1']:.4f}, "
            f"accuracy={base['accuracy']:.4f}\n"
        )

    lines.append("\n## Effect of each axis vs. baseline\n")
    if base is None:
        lines.append("_(skipped, see above)_\n")
    else:
        for _, row in df.iterrows():
            name = row["experiment"]
            if name == BASELINE_NAME:
                continue
            label = AXIS_LABELS.get(name, name)
            f1_delta = row["f1"] - base["f1"]
            acc_delta = row["accuracy"] - base["accuracy"]
            direction = "improved" if f1_delta > 0.001 else ("worsened" if f1_delta < -0.001 else "no meaningful change")
            extra = ""
            if "train_wall_seconds" in row and "train_wall_seconds" in base:
                if row["train_wall_seconds"] == row["train_wall_seconds"]:  # not NaN
                    extra = f", wall time {row['train_wall_seconds']:.0f}s"
            lines.append(
                f"- **{label}** (`{name}`): F1 {_fmt_delta(f1_delta)} "
                f"({direction}), accuracy {_fmt_delta(acc_delta)}{extra}"
            )

    lines.append("\n## Full comparison table\n")
    lines.append(df.to_markdown(index=False))
    lines.append("")

    return "\n".join(lines)


def write_findings(results_dir: Path = RESULTS_DIR) -> Path:
    content = build_findings(results_dir)
    out_path = results_dir / "findings.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = write_findings()
    print(f"Wrote {path}")
