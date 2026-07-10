"""Shared test fixtures."""

from __future__ import annotations

from typing import Any

import pytest


class MockClassifier:
    """Drop-in mock classifier that avoids loading the real model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def classify(self, text: str) -> dict[str, Any]:
        text_lower = text.lower()
        if any(w in text_lower for w in ["love", "great", "amazing", "good", "fantastic"]):
            return {"label": "positive", "confidence": 0.92, "raw_output": "positive"}
        if any(w in text_lower for w in ["hate", "terrible", "awful", "bad", "worst"]):
            return {"label": "negative", "confidence": 0.88, "raw_output": "negative"}
        return {"label": "unknown", "confidence": 0.5, "raw_output": "unknown"}


@pytest.fixture
def mock_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the classifier with a mock for tests."""
    import sentirise.api as api

    monkeypatch.setattr(api, "_classifier", MockClassifier())
    monkeypatch.setattr("sentirise.api.SentimentClassifier", MockClassifier)
    monkeypatch.setattr("sentirise.classifier.SentimentClassifier", MockClassifier)
