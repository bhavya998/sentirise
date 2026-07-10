"""QLoRA fine-tuning script - trains a LoRA adapter on Qwen3-0.6B.

Usage:
    sentirise train              # fine-tune with defaults
    sentirise train --epochs 5   # custom epochs
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from sentirise.config import (
    ADAPTER_DIR,
    BASE_MODEL,
    MAX_EVAL_SAMPLES,
    MAX_TRAIN_SAMPLES,
    QLoRA_CONFIG,
    TRAIN_ARGS,
)
from sentirise.dataset import load_imdb_dataset


def train(
    epochs: int = 3,
    max_train: int = MAX_TRAIN_SAMPLES,
    max_eval: int = MAX_EVAL_SAMPLES,
    output_dir: str = str(ADAPTER_DIR),
) -> str:
    """Run QLoRA fine-tuning on Qwen3-0.6B for sentiment classification.

    Args:
        epochs: Number of training epochs.
        max_train: Maximum training samples.
        max_eval: Maximum evaluation samples.
        output_dir: Where to save the LoRA adapter.

    Returns:
        Path to the saved adapter.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[sentirise] Loading {BASE_MODEL} in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(**QLoRA_CONFIG)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sentirise] Loading IMDB dataset ({max_train} train, {max_eval} eval)...")
    train_ds, eval_ds = load_imdb_dataset(max_train=max_train, max_eval=max_eval)

    sft_config = SFTConfig(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAIN_ARGS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_ARGS["per_device_eval_batch_size"],
        learning_rate=TRAIN_ARGS["learning_rate"],
        warmup_ratio=TRAIN_ARGS["warmup_ratio"],
        logging_steps=TRAIN_ARGS["logging_steps"],
        save_strategy=TRAIN_ARGS["save_strategy"],
        eval_strategy=TRAIN_ARGS["eval_strategy"],
        max_length=TRAIN_ARGS["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("[sentirise] Starting training...")
    trainer.train()

    print(f"[sentirise] Saving adapter to {output_path}...")
    trainer.save_model(str(output_path))

    print(f"[sentirise] Done. Adapter saved at {output_path}")
    return str(output_path)
