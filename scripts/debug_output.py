"""Debug: see raw model output to fix parsing."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from sentirise.classifier import SentimentClassifier  # noqa: E402

texts = [
    "I loved every minute of it. Highly recommend!",
    "This was the worst movie ever. Terrible.",
]

clf = SentimentClassifier()
for text in texts:
    result = clf.classify(text)
    raw = result["raw_output"]
    print(f"Text: {text}")
    print(f"  raw_output: {repr(raw)}")
    print(f"  label: {result['label']}")
    print(f"  confidence: {result['confidence']}")
    print()
