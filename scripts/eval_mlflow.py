"""MLflow evaluation: compare base model vs fine-tuned adapter on IMDB test set.

Logs accuracy, F1, precision, recall to MLflow for both models.
Run: uv run python scripts/eval_mlflow.py
"""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import mlflow
from datasets import load_dataset

from sentirise.classifier import SentimentClassifier
from sentirise.config import ADAPTER_DIR, BASE_MODEL


def evaluate_model(classifier: SentimentClassifier, samples: list[dict]) -> dict[str, float]:
    """Evaluate classifier on samples, return metrics dict."""
    tp = fp = fn = tn = 0
    for sample in samples:
        result = classifier.classify(sample["text"])
        pred = 1 if result["label"] == "positive" else 0
        actual = sample["label"]

        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    accuracy = (tp + tn) / max(len(samples), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    n_samples = 50

    print(f"Loading IMDB test set ({n_samples} samples)...")
    ds = load_dataset("stanfordnlp/imdb", split="test")
    ds = ds.shuffle(seed=42).select(range(n_samples))
    samples = [{"text": r["text"][:500], "label": r["label"]} for r in ds]

    mlflow.set_experiment("sentirise-sentiment")

    # --- Evaluate fine-tuned model ---
    print("\n[1/2] Evaluating fine-tuned model...")
    clf_finetuned = SentimentClassifier(adapter_path=str(ADAPTER_DIR))
    t0 = time.time()
    metrics_ft = evaluate_model(clf_finetuned, samples)
    ft_time = time.time() - t0
    print(f"  Done ({ft_time:.0f}s): acc={metrics_ft['accuracy']:.1%} f1={metrics_ft['f1']:.1%}")

    # --- Evaluate base model ---
    print("\n[2/2] Evaluating base model (no adapter)...")
    clf_base = SentimentClassifier(adapter_path="/nonexistent", model_id=BASE_MODEL)
    t0 = time.time()
    metrics_base = evaluate_model(clf_base, samples)
    base_time = time.time() - t0
    print(f"  Done ({base_time:.0f}s): acc={metrics_base['accuracy']:.1%} f1={metrics_base['f1']:.1%}")

    # --- Log to MLflow ---
    print("\nLogging to MLflow...")
    with mlflow.start_run(run_name="qlora-finetuned"):
        mlflow.log_params({
            "model": BASE_MODEL,
            "method": "QLoRA",
            "train_samples": 500,
            "epochs": 1,
            "adapter": str(ADAPTER_DIR),
        })
        mlflow.log_metrics({
            "ft_accuracy": metrics_ft["accuracy"],
            "ft_f1": metrics_ft["f1"],
            "ft_precision": metrics_ft["precision"],
            "ft_recall": metrics_ft["recall"],
            "base_accuracy": metrics_base["accuracy"],
            "base_f1": metrics_base["f1"],
            "improvement_accuracy": metrics_ft["accuracy"] - metrics_base["accuracy"],
            "improvement_f1": metrics_ft["f1"] - metrics_base["f1"],
        })

    # --- Summary ---
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<15} {'Base':>8} {'Fine-tuned':>12} {'Δ':>8}")
    print("-" * 50)
    for metric in ["accuracy", "f1", "precision", "recall"]:
        b = metrics_base[metric]
        f = metrics_ft[metric]
        delta = f - b
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<15} {b:>8.1%} {f:>12.1%} {sign}{delta:>7.1%}")
    print("=" * 50)
    print(f"\nSamples: {n_samples} | Eval time: ft={ft_time:.0f}s base={base_time:.0f}s")


if __name__ == "__main__":
    main()
