"""Tests for sentirise.config."""

from sentirise.config import LABELS, QLoRA_CONFIG, TRAIN_ARGS


class TestConfig:
    def test_labels(self) -> None:
        assert LABELS == ["negative", "positive"]

    def test_qlora_config_has_required_keys(self) -> None:
        assert "r" in QLoRA_CONFIG
        assert "lora_alpha" in QLoRA_CONFIG
        assert "target_modules" in QLoRA_CONFIG
        assert "task_type" in QLoRA_CONFIG
        assert QLoRA_CONFIG["task_type"] == "CAUSAL_LM"

    def test_train_args(self) -> None:
        assert "per_device_train_batch_size" in TRAIN_ARGS
        assert "learning_rate" in TRAIN_ARGS
        assert "max_seq_length" in TRAIN_ARGS
