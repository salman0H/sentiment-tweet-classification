"""
Fine-tuning loop.

Compared to the starter notebook, the functional changes are:
  1. Real mini-batches through a DataLoader with dynamic padding, instead
     of one forward/backward pass per tweet.
  2. A linear warmup + decay learning-rate schedule, which matters a lot
     at the small learning rates transformer fine-tuning needs.
  3. Eager, single-pass tokenization (src/dataset.py) plus optional
     DataLoader worker processes, so CPU-side batch prep can overlap with
     GPU compute instead of serializing with it.
  4. Early stopping on dev F1, so an experiment that has plateaued doesn't
     keep burning wall-clock time for epochs that won't be selected anyway
     (the saved checkpoint is still whichever epoch had the best dev F1).

Everything else (device placement, checkpoint saving, dev-set selection)
is kept deliberately simple so each experiment stays easy to read end to
end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW, SGD
from transformers import get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .dataset import TweetDataset, make_collate_fn, make_dataloader
from .hardware_monitor import HardwareMonitor
from .logging_utils import get_experiment_logger
from .metrics import compute_metrics, full_report
from .model import load_model, load_tokenizer
from .run_state import RunState, STATUS_DONE, STATUS_EVALUATING, STATUS_FAILED, STATUS_TRAINING


def build_optimizer(model, config: ExperimentConfig):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "sgd":
        return SGD(trainable_params, lr=config.learning_rate, momentum=0.9)
    return AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)


def _make_grad_scaler(device_type: str, enabled: bool):
    # torch.cuda.amp.GradScaler is deprecated in favor of the unified
    # torch.amp API in recent torch versions; fall back for older ones so
    # this keeps working across the range of torch>=2.2 installs.
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool = False) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def run_training(
    config: ExperimentConfig,
    train_texts: List[str],
    train_labels: List[int],
    dev_texts: List[str],
    dev_labels: List[int],
    state: Optional[RunState] = None,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_experiment_logger(config.name, output_dir)

    if state is not None:
        state.set_experiment_status(config.name, STATUS_TRAINING, detail="loading model/tokenizer")

    tokenizer = load_tokenizer(config.model_key)
    model = load_model(config.model_key, freeze_encoder=config.freeze_encoder).to(device)
    collate_fn = make_collate_fn(tokenizer)

    pin_memory = device.type == "cuda"
    train_loader = make_dataloader(
        TweetDataset(train_texts, train_labels, tokenizer, config.max_length),
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.dataloader_num_workers,
        pin_memory=pin_memory,
    )
    dev_loader = make_dataloader(
        TweetDataset(dev_texts, dev_labels, tokenizer, config.max_length),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.dataloader_num_workers,
        pin_memory=pin_memory,
    )

    optimizer = build_optimizer(model, config)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    # Mixed precision only makes sense on CUDA; on CPU it's a silent no-op.
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = _make_grad_scaler(device.type, enabled=use_amp)

    history = []
    best_dev_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    monitor = HardwareMonitor(interval_seconds=config.monitor_interval_seconds)
    if config.monitor_hardware:
        monitor.start()
    training_start = time.time()

    logger.info(
        "starting '%s': train=%d dev=%d model=%s device=%s amp=%s early_stop_patience=%s",
        config.name, len(train_texts), len(dev_texts), config.model_key, device.type,
        use_amp, config.early_stopping_patience,
    )

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss = 0.0
            start = time.time()

            for batch in train_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(**batch, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            if state is not None:
                state.set_experiment_status(
                    config.name, STATUS_EVALUATING, detail=f"epoch {epoch}/{config.epochs} dev eval",
                    epoch=epoch, total_epochs=config.epochs,
                )
            dev_true, dev_pred = evaluate(model, dev_loader, device, use_amp=use_amp)
            dev_metrics = compute_metrics(dev_true, dev_pred)

            epoch_record = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "dev": dev_metrics,
                "elapsed_seconds": round(time.time() - start, 1),
                "cumulative_elapsed_seconds": round(time.time() - training_start, 1),
            }
            history.append(epoch_record)
            logger.info(
                "epoch %d/%d train_loss=%.4f dev_f1=%.4f dev_acc=%.4f (%.1fs)",
                epoch, config.epochs, avg_train_loss, dev_metrics["f1"], dev_metrics["accuracy"],
                epoch_record["elapsed_seconds"],
            )
            if state is not None:
                state.set_experiment_status(
                    config.name, STATUS_TRAINING, detail=f"epoch {epoch}/{config.epochs} done",
                    metrics=dev_metrics, epoch=epoch, total_epochs=config.epochs,
                )

            if dev_metrics["f1"] > best_dev_f1:
                best_dev_f1 = dev_metrics["f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                patience = config.early_stopping_patience
                if patience is not None and epochs_without_improvement >= patience:
                    logger.info(
                        "early stopping: dev F1 hasn't improved for %d epoch(s) "
                        "(best=%.4f) -- keeping best checkpoint, skipping remaining epochs",
                        epochs_without_improvement, best_dev_f1,
                    )
                    break
    except Exception:
        if state is not None:
            state.set_experiment_status(config.name, STATUS_FAILED, detail="exception during training")
        logger.exception("training failed for '%s'", config.name)
        raise
    finally:
        if config.monitor_hardware:
            monitor.stop()
            monitor.save_csv(output_dir / "hardware_log.csv")
            if not monitor.gpu_available:
                logger.info(
                    "[hardware monitor] no NVIDIA GPU detected via NVML -- "
                    "hardware_log.csv only contains CPU/RAM columns."
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    if state is not None:
        state.set_experiment_status(config.name, STATUS_DONE, detail="training complete", metrics={"f1": best_dev_f1})

    return {"model": model, "tokenizer": tokenizer, "history": history, "best_dev_f1": best_dev_f1}


def evaluate_on_texts(model, tokenizer, config: ExperimentConfig, texts: List[str], labels: List[int]):
    device = next(model.parameters()).device
    collate_fn = make_collate_fn(tokenizer)
    loader = make_dataloader(
        TweetDataset(texts, labels, tokenizer, config.max_length),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.dataloader_num_workers,
        pin_memory=device.type == "cuda",
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    y_true, y_pred = evaluate(model, loader, device, use_amp=use_amp)
    metrics = compute_metrics(y_true, y_pred)
    report = full_report(y_true, y_pred)
    return metrics, report, y_true, y_pred
