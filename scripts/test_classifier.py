"""Test the fine-tuned classifier on sample texts."""

from sentirise.classifier import SentimentClassifier

samples = [
    ("This movie was absolutely fantastic! Best film I've ever seen.", "positive"),
    ("Terrible waste of time. The acting was awful and the plot made no sense.", "negative"),
    ("I loved every minute of it. Highly recommend to everyone!", "positive"),
    ("Boring, predictable, and completely disappointing. Do not watch.", "negative"),
    ("The cinematography was stunning and the performances were brilliant.", "positive"),
    ("One of the worst movies ever made. A complete disaster.", "negative"),
]

clf = SentimentClassifier()

correct = 0
for text, expected in samples:
    result = clf.classify(text)
    hit = result["label"] == expected
    correct += hit
    mark = "OK" if hit else "MISS"
    print(f"[{mark}] {result['label']:10s} (exp: {expected:10s}) | {text[:60]}")

print(f"\nAccuracy: {correct}/{len(samples)} ({correct/len(samples):.0%})")
