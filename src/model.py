"""
Model construction.

Rather than the manual "encoder + linear head on the pooler output" setup,
this uses `AutoModelForSequenceClassification`, which already ships a
classification head, handles padding via the attention mask correctly,
and lets every architecture below (encoder-only, distilled, or
parameter-shared) be swapped in through a single config field.
"""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REGISTRY = {
    "bert": "bert-base-cased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-cased",
    "albert": "albert-base-v2",
}

NUM_LABELS = 3


def resolve_checkpoint(model_key: str) -> str:
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]
    # Allow passing a raw Hugging Face checkpoint name directly.
    return model_key


def load_tokenizer(model_key: str):
    checkpoint = resolve_checkpoint(model_key)
    return AutoTokenizer.from_pretrained(checkpoint)


def load_model(model_key: str, freeze_encoder: bool = False):
    checkpoint = resolve_checkpoint(model_key)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=NUM_LABELS
    )

    if freeze_encoder:
        # Fine-tune only the classification head, keep the pretrained
        # encoder weights fixed. Useful as one of the fine-tuning
        # strategies compared in the report.
        base_model = getattr(model, model.base_model_prefix)
        for param in base_model.parameters():
            param.requires_grad = False

    return model
