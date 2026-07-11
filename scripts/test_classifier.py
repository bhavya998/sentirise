"""Test the ChatML-trained classifier end-to-end."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from sentirise.classifier import SentimentClassifier

samples = [
    ("This movie was absolutely fantastic! Best film I've ever seen.", "positive"),
    ("Terrible waste of time. The acting was awful.", "negative"),
    ("I loved every minute of it. Highly recommend!", "positive"),
    ("Boring, predictable, and completely disappointing.", "negative"),
    ("The cinematography was stunning and performances brilliant.", "positive"),
    ("One of the worst movies ever made. A complete disaster.", "negative"),
]

clf = SentimentClassifier()

correct = 0
for text, expected in samples:
    result = clf.classify(text)
    hit = result["label"] == expected
    correct += hit
    mark = "OK  " if hit else "MISS"
    raw = result["raw_output"][:30]
    print(f"[{mark}] {result['label']:10s} (exp: {expected:10s}) conf={result['confidence']:.0%} raw={repr(raw)} | {text[:50]}")

print(f"\nAccuracy: {correct}/{len(samples)} ({correct/len(samples):.0%})")
