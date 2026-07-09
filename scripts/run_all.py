"""Single entry point for the whole pipeline: environment self-check, the
fixed train/dev/test split, every experiment in configs/ (each preceded by
a fast smoke test), the comparison table, the one-time test-set report, and
an auto-generated findings summary.

This exists so running the full study is one command instead of nine
(`python scripts/run_experiment.py configs/X.yaml` repeated) with no shared
visibility into what's happening or how far along it is. Every state
transition is written to `results/run_state.json` (src/run_state.py) so
`app.py` can render live progress without parsing log files.

Usage:
    python scripts/run_all.py                  # run everything, skip completed
    python scripts/run_all.py --force           # re-run everything
    python scripts/run_all.py --configs bert_baseline roberta
    python scripts/run_all.py --skip-smoke       # skip the pre-flight sanity run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_results import write_comparison_table  # noqa: E402
from scripts.conclusions import write_findings  # noqa: E402
# scripts.final_report is imported lazily below: it pulls in src.model,
# which imports transformers at module scope, and the env-check/split
# stages of this pipeline must still work on a machine that only has
# torch/transformers installed later (or on a different machine entirely).

from src.config import ExperimentConfig  # noqa: E402
from src.data import apply_split, build_or_load_splits, load_raw_corpus  # noqa: E402
from src.preprocessing import clean_corpus  # noqa: E402
from src.run_state import (  # noqa: E402
    RunState,
    STAGE_COMPARE,
    STAGE_CONCLUSIONS,
    STAGE_DONE,
    STAGE_ENV_CHECK,
    STAGE_EXPERIMENTS,
    STAGE_FINAL_REPORT,
    STAGE_SPLIT,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SKIPPED,
    STATUS_SMOKE_TEST,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"


def check_environment() -> dict:
    info = {
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    torch_spec = importlib.util.find_spec("torch")
    transformers_spec = importlib.util.find_spec("transformers")
    info["torch_installed"] = torch_spec is not None
    info["transformers_installed"] = transformers_spec is not None

    if torch_spec is not None:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["device_name"] = torch.cuda.get_device_name(0) if info["cuda_available"] else "cpu"
    else:
        info["torch_version"] = None
        info["cuda_available"] = False
        info["device_name"] = None

    disk_usage = shutil.disk_usage(ROOT)
    info["free_disk_gb"] = round(disk_usage.free / 1e9, 1)
    info["ready_to_train"] = info["torch_installed"] and info["transformers_installed"]
    return info


def load_configs(names: Optional[List[str]]) -> List[Path]:
    all_paths = sorted(CONFIGS_DIR.glob("*.yaml"))
    if not names:
        return all_paths
    wanted = set(names)
    selected = [p for p in all_paths if p.stem in wanted]
    missing = wanted - {p.stem for p in selected}
    if missing:
        raise SystemExit(f"Unknown config name(s): {sorted(missing)}")
    return selected


def run_pipeline(
    config_names: Optional[List[str]] = None,
    force: bool = False,
    skip_smoke: bool = False,
) -> RunState:
    state = RunState.create(RESULTS_DIR)

    state.set_stage(STAGE_ENV_CHECK, "checking python/torch/transformers/CUDA")
    env_info = check_environment()
    state.set_env_check(env_info)
    print(
        f"[env] python={env_info['python_version']} torch={env_info['torch_version']} "
        f"cuda={env_info['cuda_available']} device={env_info['device_name']} "
        f"cpus={env_info['cpu_count']} free_disk={env_info['free_disk_gb']}GB"
    )
    if not env_info["ready_to_train"]:
        print(
            "\n[env] torch and/or transformers are not installed in this environment. "
            "Install them (see README.md -> Setup) before running experiments. "
            "Continuing with the parts of the pipeline that don't need torch "
            "(building the split) so the rest of the state is still useful.\n"
        )

    config_paths = load_configs(config_names)

    state.set_stage(STAGE_SPLIT, "loading corpus and building/loading the fixed split")
    first_config = ExperimentConfig.from_yaml(config_paths[0])
    texts, labels = load_raw_corpus(ROOT / first_config.text_path, ROOT / first_config.label_path)
    splits = build_or_load_splits(
        n_samples=len(texts),
        labels=labels,
        split_path=ROOT / first_config.split_path,
        dev_fraction=first_config.dev_fraction,
        test_fraction=first_config.test_fraction,
        seed=first_config.seed,
    )
    print(f"[split] train={len(splits.train)} dev={len(splits.dev)} test={len(splits.test)}")

    for path in config_paths:
        state.set_experiment_status(path.stem, STATUS_PENDING)

    state.set_stage(STAGE_EXPERIMENTS, f"{len(config_paths)} experiment(s) queued")

    if not env_info["ready_to_train"]:
        for path in config_paths:
            state.set_experiment_status(
                path.stem, STATUS_FAILED, detail="torch/transformers not installed"
            )
        state.fail("torch/transformers not installed -- cannot train. See README.md -> Setup.")
        return state

    from src.smoke_test import run_smoke_test  # deferred: needs torch
    from src.trainer import evaluate_on_texts, run_training  # deferred: needs torch

    for path in config_paths:
        config = ExperimentConfig.from_yaml(path)
        result_dir = RESULTS_DIR / config.name
        dev_metrics_path = result_dir / "dev_metrics.json"

        if dev_metrics_path.exists() and not force:
            print(f"[{config.name}] already completed -- skipping (use --force to re-run)")
            state.set_experiment_status(config.name, STATUS_SKIPPED, detail="already completed")
            continue

        cleaned_texts = clean_corpus(texts, config.preprocessing)
        train_texts, train_labels = apply_split(cleaned_texts, labels, splits.train)
        dev_texts, dev_labels = apply_split(cleaned_texts, labels, splits.dev)

        try:
            if not skip_smoke:
                state.set_experiment_status(config.name, STATUS_SMOKE_TEST, detail="sanity-checking pipeline on a tiny subset")
                print(f"[{config.name}] running smoke test...")
                smoke_start = time.time()
                run_smoke_test(config, train_texts, train_labels, dev_texts, dev_labels)
                print(f"[{config.name}] smoke test passed in {time.time() - smoke_start:.1f}s")

            print(f"[{config.name}] starting full run: train={len(train_texts)} dev={len(dev_texts)} model={config.model_key}")
            outcome = run_training(config, train_texts, train_labels, dev_texts, dev_labels, state=state)

            dev_metrics, dev_report, _, _ = evaluate_on_texts(
                outcome["model"], outcome["tokenizer"], config, dev_texts, dev_labels
            )
            (result_dir / "dev_metrics.json").write_text(json.dumps(dev_metrics, indent=2), encoding="utf-8")
            (result_dir / "dev_classification_report.txt").write_text(dev_report, encoding="utf-8")
            print(f"[{config.name}] done: dev_f1={dev_metrics['f1']:.4f} dev_acc={dev_metrics['accuracy']:.4f}")
        except Exception as exc:  # noqa: BLE001 - one bad experiment must not kill the sweep
            traceback.print_exc()
            state.set_experiment_status(config.name, STATUS_FAILED, detail=str(exc)[:300])
            print(f"[{config.name}] FAILED: {exc}")
            continue

    state.set_stage(STAGE_COMPARE, "aggregating dev metrics")
    df = write_comparison_table()
    if df is not None:
        print(df.to_string(index=False))

    state.set_stage(STAGE_FINAL_REPORT, "evaluating the best experiment on the test set (once)")
    if env_info["ready_to_train"]:
        from scripts.final_report import run_final_report

        run_final_report(force=False)
    else:
        print("[final report] skipped -- torch/transformers not installed")

    state.set_stage(STAGE_CONCLUSIONS, "writing findings.md")
    findings_path = write_findings()
    print(f"[conclusions] wrote {findings_path}")

    state.set_stage(STAGE_DONE, "pipeline complete")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=None, help="Config names (stems) to run; default: all")
    parser.add_argument("--force", action="store_true", help="Re-run experiments even if already completed")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the pre-flight smoke test")
    args = parser.parse_args()

    run_pipeline(config_names=args.configs, force=args.force, skip_smoke=args.skip_smoke)


if __name__ == "__main__":
    main()
