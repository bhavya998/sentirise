"""Real E2E API test — live server + real model."""

from __future__ import annotations

import socket
import sys
import threading
import time

import requests
from uvicorn import Config, Server

sys.stdout.reconfigure(encoding="utf-8")

from sentirise.api import app  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    port = _free_port()
    config = Config(app=app, host="127.0.0.1", port=port, log_level="warning")
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

    print(f"Server up on port {port}")

    tests = [
        ("This movie was absolutely fantastic!", "positive"),
        ("Terrible waste of time. Awful.", "negative"),
        ("I loved every minute of it!", "positive"),
        ("One of the worst movies ever.", "negative"),
    ]

    print("\nPOST /classify tests:\n")
    correct = 0
    for text, expected in tests:
        r = requests.post(f"{base}/classify", json={"text": text})
        data = r.json()
        hit = data["label"] == expected
        correct += hit
        mark = "OK  " if hit else "MISS"
        print(f"[{mark}] {data['label']:10s} conf={data['confidence']:.0%} raw={data['raw_output'][:20]!r} | {text[:40]}")

    print(f"\nAPI Accuracy: {correct}/{len(tests)} ({correct/len(tasks):.0%})" if False else f"\nAPI Accuracy: {correct}/{len(tests)} ({correct/len(tests):.0%})")
    server.should_exit = True


if __name__ == "__main__":
    main()
