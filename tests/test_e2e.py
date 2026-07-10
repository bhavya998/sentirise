"""End-to-end test: live server + mocked model."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import pytest
import requests
from uvicorn import Config, Server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubClassifier:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def classify(self, text: str) -> dict[str, Any]:
        if "love" in text.lower():
            return {"label": "positive", "confidence": 0.95, "raw_output": "positive"}
        if "hate" in text.lower():
            return {"label": "negative", "confidence": 0.91, "raw_output": "negative"}
        return {"label": "unknown", "confidence": 0.5, "raw_output": "unknown"}


@pytest.fixture(scope="module")
def e2e_server() -> str:
    import sentirise.api as api_mod

    api_mod._classifier = _StubClassifier()

    port = _free_port()
    config = Config(app=api_mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = Server(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            r = requests.get(f"{base}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("Server did not start")

    yield base
    server.should_exit = True
    thread.join(timeout=5)


class TestE2EHealth:
    def test_health(self, e2e_server: str) -> None:
        resp = requests.get(f"{e2e_server}/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestE2EClassify:
    def test_positive(self, e2e_server: str) -> None:
        resp = requests.post(
            f"{e2e_server}/classify", json={"text": "I love this product!"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "positive"
        assert data["confidence"] > 0.5

    def test_negative(self, e2e_server: str) -> None:
        resp = requests.post(
            f"{e2e_server}/classify", json={"text": "I hate everything about this."}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "negative"


class TestE2ECORS:
    def test_cors_headers(self, e2e_server: str) -> None:
        resp = requests.options(
            f"{e2e_server}/classify",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"
