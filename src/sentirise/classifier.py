"""Sentiment classifier - loads base model + LoRA adapter, classifies text."""

from __future__ import annotations

from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sentirise.config import ADAPTER_DIR, BASE_MODEL, LABELS
from sentirise.dataset import format_for_inference


class SentimentClassifier:
    """QLoRA fine-tuned sentiment classifier wrapping Qwen3-0.6B.

    Loads the base model in 4-bit, merges the LoRA adapter, and provides
    a clean classify() method that returns label + confidence.
    """

    def __init__(
        self,
        adapter_path: str = str(ADAPTER_DIR),
        model_id: str = BASE_MODEL,
        quantize_4bit: bool = True,
        max_new_tokens: int = 5,
    ) -> None:
        """Initialize the classifier with paths and inference settings.

        Args:
            adapter_path: Path to the fine-tuned LoRA adapter directory.
            model_id: HuggingFace model ID for the base model.
            quantize_4bit: Whether to load the base model in 4-bit quantization.
            max_new_tokens: Maximum tokens to generate during classification.
        """
        self._adapter_path = adapter_path
        self._model_id = model_id
        self._quantize = quantize_4bit
        self._max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        """Lazily load the base model + LoRA adapter. Called on first classify()."""
        if self._model is not None:
            return

        from pathlib import Path

        print(f"[sentirise] Loading {self._model_id} in 4-bit...")
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if self._quantize and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif not torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        base = AutoModelForCausalLM.from_pretrained(self._model_id, **kwargs)

        adapter_path = Path(self._adapter_path)
        if (adapter_path / "adapter_config.json").exists():
            print(f"[sentirise] Loading LoRA adapter from {adapter_path}...")
            self._model = PeftModel.from_pretrained(base, str(adapter_path))
        else:
            print("[sentirise] No adapter found - using base model (untrained)")
            self._model = base

        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def classify(self, text: str) -> dict[str, Any]:
        """Classify the sentiment of the given text.

        Args:
            text: Raw user text to classify.

        Returns:
            Dict with 'label' (str), 'confidence' (float), and 'raw_output' (str).
        """
        self._load()
        prompt = format_for_inference(text)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = self._tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()

        label, confidence = self._parse_output(generated)
        return {
            "label": label,
            "confidence": confidence,
            "raw_output": generated,
        }

    @staticmethod
    def _parse_output(generated: str) -> tuple[str, float]:
        """Parse the model output to extract label and a heuristic confidence.

        Confidence tiers:
          0.85 — output starts with the label (clean generation)
          0.70 — label found as substring (noisy generation)
          0.00 — no label found → "unknown"

        Args:
            generated: Raw model generation after the prompt.

        Returns:
            Tuple of (label, confidence) where label is in LABELS.
        """
        gen_lower = generated.lower().strip()

        for label in LABELS:
            if gen_lower.startswith(label):
                return label, 0.85

        if "positive" in gen_lower and "negative" not in gen_lower:
            return "positive", 0.7
        if "negative" in gen_lower and "positive" not in gen_lower:
            return "negative", 0.7

        return "unknown", 0.0
