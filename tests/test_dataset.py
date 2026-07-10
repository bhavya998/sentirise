"""Tests for sentirise.dataset."""

from __future__ import annotations

from unittest.mock import MagicMock

from sentirise.dataset import SYSTEM_PROMPT, USER_TEMPLATE, format_for_inference, load_imdb_dataset


class TestFormatForInference:
    def test_contains_review_text(self) -> None:
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "<formatted>Great movie!</formatted>"
        result = format_for_inference("Great movie!", mock_tok)
        assert "Great movie!" in result
        # Verify chat template was called with correct structure
        call_args = mock_tok.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Great movie!" in messages[1]["content"]

    def test_system_prompt_present(self) -> None:
        assert "positive" in SYSTEM_PROMPT or "sentiment" in SYSTEM_PROMPT.lower()

    def test_user_template_has_placeholder(self) -> None:
        assert "{text}" in USER_TEMPLATE


class TestLoadImdbDataset:
    def test_returns_two_datasets(self) -> None:
        train, eval_ds = load_imdb_dataset(max_train=10, max_eval=5)
        assert len(train) == 10
        assert len(eval_ds) == 5

    def test_datasets_have_text_column(self) -> None:
        train, _ = load_imdb_dataset(max_train=5, max_eval=2)
        assert "text" in train.column_names

    def test_formatted_text_contains_label(self) -> None:
        train, _ = load_imdb_dataset(max_train=10, max_eval=2)
        sample = train[0]["text"]
        assert "positive" in sample or "negative" in sample
