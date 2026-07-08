"""
Fine-tuning loop.

Compared to the starter notebook, the two functional changes are:
  1. Real mini-batches through a DataLoader with dynamic padding, instead
     of one forward/backward pass per tweet.
  2. A linear warmup + decay learning-rate schedule, which matters a lot
     at the small learning rates transformer fine-tuning needs.

Everything else (device placement, checkpoint saving, dev-set selection)
is kept deliberately simple so each experiment stays easy to read end to
end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .dataset import TweetDataset, make_collate_fn
from .hardware_monitor import HardwareMonitor
from .metrics import compute_metrics, full_report
from .model import load_model, load_tokenizer


def build_optimizer(model, config: ExperimentConfig):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "sgd":
        return SGD(trainable_params, lr=config.learning_rate, momentum=0.9)
    return AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)


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
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    tokenizer = load_tokenizer(config.model_key)
    model = load_model(config.model_key, freeze_encoder=config.freeze_encoder).to(device)
    collate_fn = make_collate_fn(tokenizer)

    train_loader = DataLoader(
        TweetDataset(train_texts, train_labels, tokenizer, config.max_length),
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        TweetDataset(dev_texts, dev_labels, tokenizer, config.max_length),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_dev_f1 = -1.0
    best_state = None

    monitor = HardwareMonitor(interval_seconds=config.monitor_interval_seconds)
    if config.monitor_hardware:
        monitor.start()
    training_start = time.time()

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
        print(
            f"[{config.name}] epoch {epoch}/{config.epochs} "
            f"train_loss={avg_train_loss:.4f} "
            f"dev_f1={dev_metrics['f1']:.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f}"
        )

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if config.monitor_hardware:
        monitor.stop()
        monitor.save_csv(output_dir / "hardware_log.csv")
        if not monitor.gpu_available:
            print(
                "[hardware monitor] no NVIDIA GPU detected via NVML — "
                "hardware_log.csv only contains CPU/RAM columns."
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    return {"model": model, "tokenizer": tokenizer, "history": history, "best_dev_f1": best_dev_f1}


def evaluate_on_texts(model, tokenizer, config: ExperimentConfig, texts: List[str], labels: List[int]):
    device = next(model.parameters()).device
    collate_fn = make_collate_fn(tokenizer)
    loader = DataLoader(
        TweetDataset(texts, labels, tokenizer, config.max_length),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    y_true, y_pred = evaluate(model, loader, device, use_amp=use_amp)
    metrics = compute_metrics(y_true, y_pred)
    report = full_report(y_true, y_pred)
    return metrics, report, y_true, y_pred
