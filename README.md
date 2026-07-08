# Tweet Sentiment Classification with Transformer Language Models

Final project for *Fundamentals of Language and Speech Processing*.
Three-way sentiment classification of tweets (`0` negative, `1` neutral,
`2` positive) using fine-tuned transformer encoders.

## Why this is structured differently from the starter notebook

The provided starter notebook fine-tunes `bert-base-cased` by feeding one
tweet at a time through the model, with the classification head
hand-written on top of the pooler output. That works, but it makes every
one of the experiments the assignment asks for — a different model, a
different learning rate, a different max sequence length — mean copying
and re-editing the whole notebook.

This project pulls the same logic apart into a small, reusable package
(`src/`) plus one YAML file per experiment (`configs/`), so that:

- Switching models (BERT / RoBERTa / DistilBERT / ALBERT) is a one-line
  change (`model_key` in the config).
- Hyperparameters, preprocessing, max length, and fine-tuning strategy are
  all explicit, versioned config fields instead of edits scattered across
  notebook cells.
- Training runs in real mini-batches with dynamic padding and a linear
  warmup/decay schedule, instead of a single-sample loop.
- Every experiment is reproducible from its config file alone, and every
  experiment reuses the exact same train/dev/test split.

## Project layout

```
.
├── configs/                  one YAML file per experiment
├── data/
│   ├── train_text.txt        raw corpus (provided)
│   ├── train_labels.txt      raw labels (provided)
│   └── splits.json           cached train/dev/test indices (generated)
├── notebooks/
│   └── run_baseline.ipynb    end-to-end runnable walkthrough
├── results/
│   └── <experiment_name>/    metrics, best checkpoint, history, hardware log per run
├── scripts/
│   ├── prepare_splits.py     builds the fixed split, once
│   ├── run_experiment.py     runs one experiment end to end
│   ├── compare_results.py    aggregates results into one table
│   ├── plot_hardware.py      charts CPU/RAM/GPU usage for one run
│   └── verify_gpu.py         confirms torch can see your GPU
└── src/
    ├── config.py             ExperimentConfig: everything an experiment needs
    ├── data.py                corpus loading + stratified split
    ├── preprocessing.py       configurable text cleaning pipeline
    ├── dataset.py             torch Dataset + dynamic-padding collator
    ├── model.py               model registry (bert/roberta/distilbert/albert)
    ├── trainer.py             training loop, evaluation
    ├── hardware_monitor.py    background CPU/RAM/GPU sampling
    ├── metrics.py             accuracy / precision / recall / F1
    └── error_analysis.py      misclassified examples, confusion matrix
```

## Setup

Torch is installed separately from the rest of the dependencies — it's a
large download (500MB+), and bundling it into one `pip install` means a
single flaky connection aborts the install of everything else, including
lightweight packages like `PyYAML` and `scikit-learn` that the rest of the
project needs.

```bash
python -m venv .venv && source .venv/bin/activate

# 1. Everything except torch
pip install -r requirements.txt

# 2a. If you have an NVIDIA GPU:
pip install --timeout 1000 torch>=2.2

# 2b. CPU-only machine (most laptops) — smaller download, no CUDA wheels:
pip install --timeout 1000 --index-url https://download.pytorch.org/whl/cpu torch>=2.2
```

If a download still times out, `pip`'s default timeout (15s of no
progress) is too short for a 500MB file on a slow connection — the
`--timeout 1000` above raises that. `pip install -v ...` also lets you see
progress and confirm it's still downloading rather than stuck.

A CUDA-capable GPU is strongly recommended; fine-tuning any of these
models on the full corpus is impractical on CPU. If you only have CPU
available, use `configs/distilbert.yaml` and a small subset of the data
to sanity-check the pipeline before committing to a full run elsewhere
(e.g. Google Colab, which gives free GPU access).

## Using a local GPU

1. Confirm the driver is installed and check which CUDA version it supports:
   ```bash
   nvidia-smi
   ```
   Look at the `CUDA Version` field in the header.

2. Install the matching torch build from the official selector at
   https://pytorch.org/get-started/locally/ (choose Linux / Pip / Python /
   your CUDA version). You do **not** need to install the CUDA toolkit
   separately — the pip wheel bundles the CUDA runtime it needs; only the
   driver has to be new enough.

3. Verify torch actually sees the GPU:
   ```bash
   python scripts/verify_gpu.py
   ```
   This prints the detected device name and runs a small real computation
   on it, not just a boolean flag.

4. Run experiments as usual — `src/trainer.py` already moves the model and
   batches to `cuda` automatically whenever `torch.cuda.is_available()` is
   `True`, no config changes needed.

Two things in this project specifically take advantage of having a GPU:

- **Automatic mixed precision** (`mixed_precision: true` in each config,
  on by default) runs the forward pass in float16 where safe, which
  roughly halves memory use and speeds up training on modern GPUs. It has
  no effect on CPU runs — the flag is silently ignored there.
- **Batch size**: the default configs use `train_batch_size: 32`, sized
  for a GPU rather than a CPU. If you hit a `CUDA out of memory` error on
  a smaller GPU (e.g. 4–6GB of VRAM), lower `train_batch_size` (and
  `eval_batch_size`) in the config you're running — everything else stays
  the same, since the scheduler and metrics don't depend on batch size.

## Hardware monitoring

Every run started through `run_experiment.py` also logs CPU load, RAM
usage, and (on an NVIDIA GPU) GPU utilization, memory, and temperature
once per second on a background thread, via `src/hardware_monitor.py`.
This has nothing to do with model quality — it's there to answer "how
hard was the machine actually working during training", which is a
separate, useful question from "how good is the model".

Each run writes an extra `results/<name>/hardware_log.csv` with one row
per sample. To turn it into a chart:

```bash
python scripts/plot_hardware.py bert_baseline
```

This produces `results/bert_baseline/hardware_usage.png`: CPU%, RAM%, and
(if available) GPU utilization% + GPU memory over time, with dotted
vertical lines marking epoch boundaries so you can see, for example,
whether GPU utilization dips between epochs (usually the dev-set
evaluation pass, which is smaller and faster than training).

If no NVIDIA GPU is detected (no driver, no `pynvml`, or running on CPU),
the GPU columns are simply left empty and the chart falls back to
CPU/RAM only — nothing breaks.

`compare_results.py` also picks up `avg_gpu_util_%`, `peak_gpu_mem_mb`,
`avg_cpu_%`, and `train_wall_seconds` from these logs when present, so
the comparison table can show accuracy/F1 next to how expensive each run
actually was.

Monitoring is on by default (`monitor_hardware: true`, sampling every
`monitor_interval_seconds: 1.0`); set either to `false` / a larger number
in a config if you don't want the background thread running.

## Running an experiment

```bash
# 1. Build the fixed split once (safe to skip — run_experiment.py does this
#    automatically on first use, but running it explicitly makes the
#    "same split for every experiment" guarantee visible).
python scripts/prepare_splits.py

# 2. Run any experiment
python scripts/run_experiment.py configs/bert_baseline.yaml

# 3. After running several experiments, compare them
python scripts/compare_results.py
```

Each run writes `results/<name>/` containing:

- `best_model.pt` — the checkpoint with the highest dev F1
- `history.json` — per-epoch train loss and dev metrics
- `config.json` — the exact config that produced this run
- `dev_metrics.json`, `dev_classification_report.txt`
- `hardware_log.csv` — CPU/RAM/GPU samples taken during training (see
  "Hardware monitoring" below)

`scripts/compare_results.py` reads all of these and produces
`results/comparison_table.csv` / `.md`.

## What each config varies

| Config | Axis under test |
|---|---|
| `bert_baseline.yaml` | Reference run: BERT, AdamW, lr 2e-5, 3 epochs |
| `roberta.yaml` / `distilbert.yaml` / `albert.yaml` | Model architecture |
| `bert_lr_high.yaml` / `bert_lr_low.yaml` | Learning rate (5e-5 vs 1e-5) |
| `bert_sgd_optimizer.yaml` | Optimizer (SGD instead of AdamW) |
| `bert_max_len_128.yaml` | Max sequence length (128 vs 64 tokens) |
| `bert_frozen_encoder.yaml` | Fine-tuning strategy: frozen encoder, head-only |
| `bert_preprocessing_aggressive.yaml` | Preprocessing: lowercase, mention/URL masking, repeated-character collapsing |

Each of these maps directly to one of the axes the assignment asks the
project to study (model choice, hyperparameters, preprocessing, max
length, fine-tuning strategy). Adding a new experiment is just adding a
new config file — no code changes needed.

## Preprocessing notes

The raw corpus has a few artifacts from how it was originally exported,
found by inspecting the data directly rather than assumed:

- Literal `\uXXXX` sequences in the text instead of the actual character
  (e.g. `can\u2019t` instead of `can't`).
- Some tweets wrapped in a stray pair of double quotes.
- Internal quotes escaped as a literal backslash + quote instead of a
  real quote character.

`src/preprocessing.py` fixes these by default (`fix_literal_unicode_escapes`,
`strip_wrapping_quotes`, `unescape_literal_quotes`) since they are
encoding artifacts rather than meaningful signal. Everything else
(mention masking, URL masking, lowercasing, repeated-character
collapsing) is off by default and only enabled in the
`bert_preprocessing_aggressive` config, so its effect on dev performance
can be measured directly against the baseline.

## Notes on methodology

- The train/dev/test split is stratified by label and cached to
  `data/splits.json` on first use — every experiment shares it, satisfying
  the requirement that all comparisons use the same split.
- Model selection and hyperparameter tuning should only look at
  `dev_metrics.json`; the test set is evaluated once, at the end, for the
  final report.
- `results/<name>/history.json` keeps the full per-epoch curve, which is
  what the report's "results analysis" and "error analysis" sections draw
  on.
# sentiment-tweet-classification
