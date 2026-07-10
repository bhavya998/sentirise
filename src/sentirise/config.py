"""Configuration - model IDs, paths, hyperparameters."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = PROJECT_ROOT / "data" / "adapters"

LABELS = ["negative", "positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

QLoRA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

TRAIN_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "logging_steps": 20,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "max_seq_length": 256,
}

MAX_TRAIN_SAMPLES = 2000
MAX_EVAL_SAMPLES = 400
