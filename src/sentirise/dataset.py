"""Dataset loading and preprocessing for sentiment fine-tuning.

Downloads IMDB reviews, formats them as instruction prompts for QLoRA,
and splits into train/eval sets.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset

from sentirise.config import MAX_EVAL_SAMPLES, MAX_TRAIN_SAMPLES

PROMPT_TEMPLATE = (
    "Classify the sentiment of the following review.\n\n"
    "Review: {text}\n\n"
    "Sentiment:"
)


def load_imdb_dataset(
    max_train: int = MAX_TRAIN_SAMPLES,
    max_eval: int = MAX_EVAL_SAMPLES,
) -> tuple[Dataset, Dataset]:
    """Load IMDB dataset subsampled to train/eval sizes.

    Args:
        max_train: Maximum number of training samples.
        max_eval: Maximum number of evaluation samples.

    Returns:
        Tuple of (train_dataset, eval_dataset) formatted with instruction prompts.
    """
    ds = load_dataset("stanfordnlp/imdb", split="train")
    ds = ds.shuffle(seed=42)

    train_raw = ds.select(range(max_train))
    eval_raw = ds.select(range(max_train, max_train + max_eval))

    train_ds = train_raw.map(_format_example, remove_columns=["text", "label"])
    eval_ds = eval_raw.map(_format_example, remove_columns=["text", "label"])

    train_ds = train_ds.rename_column("formatted_text", "text")
    eval_ds = eval_ds.rename_column("formatted_text", "text")

    return train_ds, eval_ds


def _format_example(row: dict) -> dict:
    """Format a single IMDB row into a QLoRA instruction prompt."""
    label_name = "positive" if row["label"] == 1 else "negative"
    text = row["text"].strip().replace("<br />", " ")
    if len(text) > 1500:
        text = text[:1500]
    formatted = PROMPT_TEMPLATE.format(text=text) + f" {label_name}"
    return {"formatted_text": formatted}


def format_for_inference(text: str) -> str:
    """Format user input text for the model during inference.

    Args:
        text: Raw user text to classify.

    Returns:
        Formatted prompt string ready for model input.
    """
    return PROMPT_TEMPLATE.format(text=text) + " "
