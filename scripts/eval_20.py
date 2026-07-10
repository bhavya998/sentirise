"""Evaluate the fine-tuned model on 20 hand-crafted samples."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from sentirise.classifier import SentimentClassifier  # noqa: E402

samples = [
    # --- 10 positive ---
    ("This film was an absolute masterpiece, I was moved to tears.", "positive"),
    ("Incredible acting and a gripping storyline from start to finish.", "positive"),
    ("I can't recommend this movie enough, truly a wonderful experience.", "positive"),
    ("The soundtrack was beautiful and perfectly complemented the visuals.", "positive"),
    ("A heartwarming story that left me smiling for hours.", "positive"),
    ("One of the best performances I have ever seen on screen.", "positive"),
    ("Brilliantly directed with stunning cinematography throughout.", "positive"),
    ("Every scene was captivating, I didn't want it to end.", "positive"),
    ("A delightful comedy that had the whole audience laughing.", "positive"),
    ("This is the kind of movie that restores your faith in cinema.", "positive"),
    # --- 10 negative ---
    ("What a complete waste of two hours, absolutely dreadful.", "negative"),
    ("The plot made no sense and the dialogue was painfully bad.", "negative"),
    ("I walked out halfway through, it was that boring.", "negative"),
    ("Awful special effects and a storyline that went nowhere.", "negative"),
    ("The worst movie I have seen this year, totally forgettable.", "negative"),
    ("Predictable, cliched, and lacking any originality whatsoever.", "negative"),
    ("Terrible pacing and the characters were completely unlikable.", "negative"),
    ("An embarrassing attempt at drama with zero emotional depth.", "negative"),
    ("I regret spending money on this, it was pure garbage.", "negative"),
    ("Overhyped and underdelivered in every possible way.", "negative"),
]

print("Loading model + adapter...\n")
clf = SentimentClassifier()

correct = 0
results = []
for i, (text, expected) in enumerate(samples, 1):
    result = clf.classify(text)
    hit = result["label"] == expected
    correct += hit
    mark = "OK  " if hit else "MISS"
    results.append((mark, result["label"], expected, result["confidence"], result["raw_output"], text))
    print(f"[{mark}] {i:2d}. {result['label']:10s} (exp: {expected:10s}) conf={result['confidence']:.2f}  | {text[:65]}")

print(f"\n{'='*80}")
print(f"Accuracy: {correct}/{len(samples)} ({correct/len(samples):.0%})")
print(f"{'='*80}")

# Show misses in detail
misses = [r for r in results if r[0] == "MISS"]
if misses:
    print(f"\n--- {len(misses)} MISSES ---")
    for mark, label, expected, conf, raw, text in misses:
        print(f"  Text:     {text}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {label} (raw: {repr(raw)})")
        print()
else:
    print("\nPerfect score!")
