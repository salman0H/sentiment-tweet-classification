راهنمای شما با دقت به زبان انگلیسی ترجمه، بومی‌سازی و ساختاردهی شد. تمام دستورات، توضیحات فنی، جدول عیب‌یابی و نمونه‌کدها مطابق با استانداردهای مستندسازی پروژه‌های پایتون/یادگیری عمیق تطبیق داده شده‌اند:

---

# Complete Guide to Twitter Sentiment Analysis with Transformer Models

This document serves as a step-by-step guide for installing, configuring, and executing the project on Linux systems (with a specific focus on hardware-constrained environments, such as GPUs with 2 GB VRAM). All commands, optimization tips, and troubleshooting steps are consolidated here.

---

## 1. Project Overview

This project provides a full end-to-end pipeline for **fine-tuning** Transformer-based language models (BERT, RoBERTa, DistilBERT, ALBERT) on a dataset of 45,615 tweets for 3-class sentiment classification (Negative, Neutral, Positive).

**Experimental Objectives**:

* **Architecture Comparison**: Evaluate 4 distinct model architectures.
* **Learning Rate Impact**: Assess performance across 3 learning rate settings.
* **Optimizer Comparison**: Compare AdamW against SGD.
* **Sequence Length Sensitivity**: Evaluate max token length limits ($64$ vs. $128$).
* **Fine-Tuning Strategies**: Compare full-model fine-tuning against head-only (encoder-frozen) adaptation.
* **Preprocessing Impact**: Test aggressive text preprocessing (casing, user mention replacement, URL stripping).

---

## 2. Directory Structure

```text
sentiment-tweet-classification/
├── configs/                  # Configuration files (YAML) for each experiment
│   ├── bert_baseline.yaml
│   ├── roberta.yaml
│   ├── distilbert.yaml
│   ├── albert.yaml
│   ├── bert_lr_high.yaml
│   ├── bert_lr_low.yaml
│   ├── bert_sgd_optimizer.yaml
│   ├── bert_max_len_128.yaml
│   ├── bert_frozen_encoder.yaml
│   └── bert_preprocessing_aggressive.yaml
├── data/                     # Raw dataset and generated splits
│   ├── train_text.txt
│   ├── train_labels.txt
│   └── splits.json           (Auto-generated)
├── src/                      # Core codebase
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── hardware_monitor.py
│   ├── metrics.py
│   ├── error_analysis.py
│   ├── smoke_test.py
│   └── run_state.py
├── scripts/                  # Execution and utility scripts
│   ├── prepare_splits.py
│   ├── run_experiment.py
│   ├── run_all.py
│   ├── final_report.py
│   ├── compare_results.py
│   ├── conclusions.py
│   ├── plot_hardware.py
│   └── verify_gpu.py
├── results/                  # Generated artifacts per experiment (Auto-created)
│   ├── <experiment_name>/
│   │   ├── best_model.pt
│   │   ├── config.json
│   │   ├── dev_metrics.json
│   │   ├── dev_classification_report.txt
│   │   ├── history.json
│   │   ├── hardware_log.csv
│   │   └── train.log
│   ├── comparison_table.csv / .md
│   ├── final_test_report.json / .md
│   ├── findings.md
│   └── run_state.json        (Real-time state for the dashboard)
├── app.py                    # Streamlit interactive dashboard
├── requirements.txt          # Python dependencies (excluding PyTorch)
├── requirements-torch.txt    # PyTorch installation instructions
├── run_no_proxy.sh           # Proxy bypass script
└── README.md                 # Project documentation

```

---

## 3. System Requirements

* **Operating System**: Linux (Ubuntu 20.04 LTS or newer recommended)
* **Python**: Version 3.10 or higher
* **GPU** *(Optional but strongly recommended)*: NVIDIA GPU with at least 2 GB VRAM (sufficient when paired with low batch sizes)
* **Disk Space**: At least 10 GB (for downloading pretrained checkpoints and saving experiment artifacts)

---

## 4. Environment Setup

### 4.1. Clone the Repository

```bash
git clone <repository-url>
cd sentiment-tweet-classification

```

### 4.2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### 4.3. Install Dependencies

```bash
pip install --timeout 1000 -r requirements.txt

```

*This installs essential libraries including `streamlit`, `plotly`, `scikit-learn`, `pandas`, `PyYAML`, and `matplotlib`.*

### 4.4. Install PyTorch Based on Available Hardware

Verify GPU detection:

```bash
nvidia-smi

```

* **NVIDIA GPU Available**:
```bash
pip install --timeout 1000 torch>=2.2

```


* **CPU-Only Environment**:
```bash
pip install --timeout 1000 --index-url https://download.pytorch.org/whl/cpu torch>=2.2

```



### 4.5. Verify GPU Setup

```bash
python scripts/verify_gpu.py

```

*Expected output: GPU device details alongside a successful tensor operation verification. Refer to Troubleshooting if errors occur.*

---

## 5. Low-VRAM Optimization Guidelines (2 GB VRAM)

> [!WARNING]
> The default configuration value `train_batch_size: 32` will trigger **`CUDA out of memory`** on 2 GB GPUs (e.g., NVIDIA MX450).

Before launching experiments, adjust your parameters: set `train_batch_size` to **`8`**, `eval_batch_size` to **`16`**, and lower `dataloader_num_workers` to **`1`** (or `0`) across all YAML configurations.

### Automated In-Place Update:

```bash
sed -i 's/^train_batch_size:.*/train_batch_size: 8/' configs/*.yaml
sed -i 's/^eval_batch_size:.*/eval_batch_size: 16/' configs/*.yaml
sed -i 's/^dataloader_num_workers:.*/dataloader_num_workers: 1/' configs/*.yaml

```

### Sample Optimized Configuration (`bert_baseline.yaml`):

```yaml
name: bert_baseline
model_key: bert
freeze_encoder: false
max_length: 64
optimizer: adamw
learning_rate: 2e-5
weight_decay: 0.01
epochs: 3
train_batch_size: 8          # Reduced from 32 to 8
eval_batch_size: 16          # Reduced from 64 to 16
warmup_ratio: 0.06
dataloader_num_workers: 1    # Lowered to reduce system memory overhead
preprocessing:
  fix_literal_unicode_escapes: true
  unescape_literal_quotes: true
  strip_wrapping_quotes: true
  normalize_whitespace: true
  replace_user_mentions: false
  strip_urls: false
  lowercase: false
  collapse_repeated_chars: false

```

---

## 6. Execution Guide

### 6.1. Running the Full Pipeline

Execute the end-to-end workflow (dataset split, experiment execution, result aggregation, test set evaluation, and finding generation) using a single entry script:

```bash
python scripts/run_all.py

```

* Completed runs are automatically skipped unless the `--force` flag is appended.
* Each run is prefaced by a **smoke test** (200 samples, 1 epoch) to validate setup integrity and guard against out-of-memory errors early.

**Useful Execution Flags**:

```bash
# Force re-running all experiments regardless of prior state
python scripts/run_all.py --force

# Target specific experiment configurations
python scripts/run_all.py --configs bert_baseline distilbert

# Bypass initial smoke tests (not recommended)
python scripts/run_all.py --skip-smoke

```

### 6.2. Executing a Single Experiment

```bash
python scripts/run_experiment.py configs/bert_baseline.yaml

```

### 6.3. Generating Dataset Splits

```bash
python scripts/prepare_splits.py

```

*Output:*

```text
train: 34211  dev: 4561  test: 6843
Saved to data/splits.json

```

### 6.4. Aggregating Experiment Results

```bash
python scripts/compare_results.py

```

*Generates summary reports at `results/comparison_table.csv` and `results/comparison_table.md`.*

### 6.5. Evaluating the Test Set

```bash
python scripts/final_report.py

```

*Evaluates the best-performing model (based on validation Macro F1 score) against the test set once. Results are saved to `results/final_test_report.json` and `.md`. A lock file (`results/.test_set_evaluated.json`) prevents unintended re-evaluations.*

### 6.6. Generating Findings Summaries

```bash
python scripts/conclusions.py

```

*Generates `results/findings.md`, providing comparative analysis across model architectures and hyperparameters against the baseline.*

### 6.7. Interactive Streamlit Dashboard

Launch the web interface in a separate terminal session:

```bash
streamlit run app.py

```

*(Alternative syntax if `streamlit` isn't added to system `PATH`: `python -m streamlit run app.py`)*

Access the dashboard via **`http://localhost:8501`** to monitor:

* Pipeline workflow states (Environment Checks → Splitting → Training → Evaluation).
* Live `pipeline.log` and `train.log` streams.
* Plotly performance metrics (Accuracy, Precision, Recall, Macro F1).
* Real-time system resource consumption (CPU, RAM, VRAM usage).
* Rendered markdown reports (`findings.md` and `final_test_report.md`).

---

## 7. Artifacts & Logging Directory

Output files are systematically organized under `results/<experiment_name>/`:

* `best_model.pt`: Model checkpoint with highest dev Macro F1 score.
* `config.json`: Configuration specifications for the run.
* `dev_metrics.json`: Final metrics evaluated on the validation set.
* `dev_classification_report.txt`: Per-class precision, recall, and F1-score report.
* `history.json`: Epoch-wise training loss and validation progression.
* `hardware_log.csv`: Time-series hardware utilization data (CPU, RAM, GPU).
* `train.log`: Detailed runtime logs.

**Aggregated Workspace Artifacts** (located in `results/` root):

* `comparison_table.csv` / `.md`: Cross-experiment comparative tables.
* `final_test_report.json` / `.md`: Test-set evaluation metrics for the top model.
* `findings.md`: Automated analysis summarizing experimental insights.
* `run_state.json`: Live state tracking file for dashboard rendering.

---

## 8. Troubleshooting

| Symptom / Error | Resolution |
| --- | --- |
| `ModuleNotFoundError: No module named 'plotly'` | Run `pip install plotly>=5.20` |
| `streamlit: command not found` | Run `python -m streamlit run app.py` or `pip install streamlit` |
| `CUDA out of memory` | Reduce `train_batch_size` to `4` or `8`; lower `max_length` to `32` or `48`; ensure `mixed_precision: true` is enabled. |
| `Warning: You are sending unauthenticated requests to the HF Hub` | Authenticate via `huggingface-cli login` or route through `run_no_proxy.sh`. |
| `No module named 'torch'` | Install PyTorch following instructions in **Section 4.4**. |
| `Killed` or `MemoryError` | Insufficient system RAM. Set `dataloader_num_workers: 0` and pass `pin_memory=False` in `src/trainer.py`. |
| Smoke Test Failure | Check your configuration parameters (`model_key` validity, compatibility between `freeze_encoder` and `learning_rate`). |

---

## 9. Performance Optimization Best Practices

* **Mixed Precision (FP16/BF16)**: Enabled by default, offering roughly ~30% VRAM savings and accelerated computation speeds.
* **Early Stopping**: Automatically halts training if validation Macro F1 fails to improve after 2 consecutive epochs.
* **In-Memory Tokenization**: Datasets are tokenized once during initialization rather than dynamically per epoch.
* **DataLoader Adjustments**: Setting `dataloader_num_workers=1` and `pin_memory=True` overlaps data loading operations with GPU execution (when host RAM permits).
* **Lightweight Architectures**: On constrained hardware, utilize **DistilBERT** or **ALBERT** to maintain high throughput and lower VRAM consumption.

---

## 10. Quick Start Example

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Batch adjust configuration batch sizes for low VRAM
sed -i 's/^train_batch_size:.*/train_batch_size: 8/' configs/*.yaml
sed -i 's/^eval_batch_size:.*/eval_batch_size: 16/' configs/*.yaml

# 3. Test setup with limited runs
python scripts/run_all.py --configs bert_baseline distilbert

# 4. Run full pipeline across all configs
python scripts/run_all.py

# 5. Monitor progress in Streamlit dashboard (separate terminal)
streamlit run app.py

```

---

## 11. Conclusion

This modular pipeline enables efficient, reproducible experimentation with Transformer models on sentiment classification tasks. By following the optimized guidelines for resource-constrained GPUs, full experiment cycles can be executed without facing out-of-memory errors.