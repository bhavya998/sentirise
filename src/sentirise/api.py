"""FastAPI application - sentiment classification API.

POST /classify  - upload text, get sentiment + confidence
GET  /health    - liveness check
GET  /info      - model info
"""

from __future__ import annotations

import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentirise.classifier import SentimentClassifier

_classifier: SentimentClassifier | None = None
_classifier_lock = threading.Lock()


def _get_classifier() -> SentimentClassifier:
    """Get or lazily initialize the global classifier singleton (thread-safe)."""
    global _classifier  # noqa: PLW0603
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = SentimentClassifier()
    return _classifier


class ClassifyRequest(BaseModel):
    """Request body for the /classify endpoint."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")


class ClassifyResponse(BaseModel):
    """Response body for the /classify endpoint."""

    label: str
    confidence: float
    raw_output: str


class InfoResponse(BaseModel):
    """Response body for the /info endpoint."""

    model: str
    adapter_loaded: bool
    labels: list[str]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application with all routes.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(
        title="sentirise - QLoRA Sentiment Classifier",
        version="0.1.0",
    )

    cors_origins = os.getenv("SENTIRISE_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check endpoint."""
        return {"status": "ok"}

    @app.get("/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        """Get model information."""
        from pathlib import Path

        from sentirise.config import ADAPTER_DIR, BASE_MODEL, LABELS

        return InfoResponse(
            model=BASE_MODEL,
            adapter_loaded=(Path(str(ADAPTER_DIR)) / "adapter_config.json").exists(),
            labels=LABELS,
        )

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest) -> ClassifyResponse:
        """Classify the sentiment of the provided text.

        Runs synchronously so FastAPI dispatches it to a threadpool,
        avoiding event-loop blocking during GPU inference.
        """
        clf = _get_classifier()
        result = clf.classify(req.text)
        return ClassifyResponse(**result)

    return app


app = create_app()
