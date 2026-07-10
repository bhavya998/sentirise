"""Tests for sentirise.classifier - mocked model."""

from __future__ import annotations

from sentirise.classifier import SentimentClassifier


class TestParseOutput:
    def test_positive(self) -> None:
        label, conf = SentimentClassifier._parse_output("positive")
        assert label == "positive"
        assert 0 < conf <= 1.0

    def test_negative(self) -> None:
        label, conf = SentimentClassifier._parse_output("negative")
        assert label == "negative"
        assert 0 < conf <= 1.0

    def test_positive_in_sentence(self) -> None:
        label, _ = SentimentClassifier._parse_output("This is clearly positive.")
        assert label == "positive"

    def test_negative_in_sentence(self) -> None:
        label, _ = SentimentClassifier._parse_output("The sentiment is negative.")
        assert label == "negative"

    def test_unknown(self) -> None:
        label, conf = SentimentClassifier._parse_output("xyz abc")
        assert label == "unknown"
        assert conf == 0.0

    def test_case_insensitive(self) -> None:
        label, _ = SentimentClassifier._parse_output("POSITIVE")
        assert label == "positive"

    def test_case_insensitive_negative(self) -> None:
        label, _ = SentimentClassifier._parse_output("NEGATIVE")
        assert label == "negative"
