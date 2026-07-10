"""Tests for sentirise.dataset."""

from __future__ import annotations

from sentirise.dataset import format_for_inference, load_imdb_dataset


class TestFormatForInference:
    def test_contains_prompt(self) -> None:
        result = format_for_inference("Great movie!")
        assert "Classify the sentiment" in result
        assert "Great movie!" in result
        assert "Sentiment:" in result

    def test_ends_with_space(self) -> None:
        result = format_for_inference("Test text")
        assert result.endswith(" ")


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
