"""Tests for sentirise.api - FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentirise.api import app


@pytest.fixture(autouse=True)
def reset_globals() -> None:
    import sentirise.api as api
    api._classifier = None


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestInfo:
    def test_info(self, client: TestClient) -> None:
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "labels" in data
        assert isinstance(data["labels"], list)


class TestClassify:
    def test_classify_positive(self, client: TestClient, mock_classifier: None) -> None:
        resp = client.post("/classify", json={"text": "I love this movie, it's fantastic!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "positive"
        assert "confidence" in data
        assert "raw_output" in data

    def test_classify_negative(self, client: TestClient, mock_classifier: None) -> None:
        resp = client.post("/classify", json={"text": "This was terrible and awful."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "negative"

    def test_classify_empty_text_returns_422(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"text": ""})
        assert resp.status_code == 422

    def test_classify_missing_text_returns_422(self, client: TestClient) -> None:
        resp = client.post("/classify", json={})
        assert resp.status_code == 422

    def test_classify_too_long_returns_422(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"text": "x" * 10001})
        assert resp.status_code == 422
