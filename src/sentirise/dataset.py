"""Dataset loading and preprocessing for sentiment fine-tuning.

Downloads IMDB reviews, formats them as chat-template prompts for QLoRA
on Qwen3-0.6B, and splits into train/eval sets.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from sentirise.config import BASE_MODEL, MAX_EVAL_SAMPLES, MAX_TRAIN_SAMPLES

SYSTEM_PROMPT = (
    "You are a sentiment classifier. Given a movie review, respond with "
    "exactly one word: positive or negative. Do not explain."
)

USER_TEMPLATE = "Classify the sentiment of this review:\n\n{text}"


def load_imdb_dataset(
    max_train: int = MAX_TRAIN_SAMPLES,
    max_eval: int = MAX_EVAL_SAMPLES,
) -> tuple[Dataset, Dataset]:
    """Load IMDB dataset subsampled to train/eval sizes.

    Args:
        max_train: Maximum number of training samples.
        max_eval: Maximum number of evaluation samples.

    Returns:
        Tuple of (train_dataset, eval_dataset) formatted with chat-template prompts.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    ds = load_dataset("stanfordnlp/imdb", split="train")
    ds = ds.shuffle(seed=42)

    train_raw = ds.select(range(max_train))
    eval_raw = ds.select(range(max_train, max_train + max_eval))

    fmt = _make_formatter(tokenizer)
    train_ds = train_raw.map(fmt, remove_columns=["text", "label"])
    eval_ds = eval_raw.map(fmt, remove_columns=["text", "label"])

    train_ds = train_ds.rename_column("formatted_text", "text")
    eval_ds = eval_ds.rename_column("formatted_text", "text")

    return train_ds, eval_ds


def _make_formatter(tokenizer: AutoTokenizer):
    """Return a map function that formats examples using the chat template."""

    def _format_example(row: dict) -> dict:
        label_name = "positive" if row["label"] == 1 else "negative"
        review = row["text"].strip().replace("<br />", " ")
        if len(review) > 1500:
            review = review[:1500]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=review)},
            {"role": "assistant", "content": label_name},
        ]

        # tokenize=False → get the formatted string for SFTTrainer
        # enable_thinking=False → /no_think prefix for Qwen3
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"formatted_text": formatted}

    return _format_example


def format_for_inference(text: str, tokenizer: AutoTokenizer) -> str:
    """Format user input text for the model during inference.

    Uses the chat template with a generation prompt so the model
    continues with the assistant's answer.

    Args:
        text: Raw user text to classify.
        tokenizer: The tokenizer with the chat template.

    Returns:
        Formatted prompt string ready for model input.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
