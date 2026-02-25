#!/usr/bin/env python3
"""
Sentinel Baseline Evaluation
=============================
Runs the open-source Sentinel model (qualifire/prompt-injection-sentinel,
ModernBERT-large 395M) on HIVE's 10,958-sample unified benchmark for a
direct head-to-head comparison.

Requirements:
    pip install transformers torch pandas scikit-learn numpy

Usage:
    python sentinel_baseline_evaluation.py

Output:
    results/sentinel_baseline_results.json  — metrics + per-source breakdown
    results/sentinel_per_sample.csv         — per-sample predictions
"""

import csv
import json
import time
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------- CONFIG ----------
MODEL_NAME = "qualifire/prompt-injection-sentinel"
BATCH_SIZE = 64
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATASET_PATH = PROJECT_ROOT / "data" / "unified_dataset.csv"


def load_unified_dataset():
    """Load the same unified dataset used in HIVE evaluation."""
    rows = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "text": row["text"],
                "label": int(row["label"]),
                "source": row.get("source", "unknown"),
            })
    print(f"Loaded {len(rows)} samples from {DATASET_PATH.name}")
    return rows


def load_sentinel_model():
    """Load Sentinel model from HuggingFace."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    print(f"Loading Sentinel model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    return model, tokenizer, device


def run_sentinel_evaluation(model, tokenizer, device, samples):
    """Run Sentinel on all samples and collect predictions."""
    import torch

    results = []
    total = len(samples)
    start_time = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_samples = samples[batch_start:batch_end]

        texts = [s["text"] for s in batch_samples]

        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Inference
        with torch.no_grad():
            t0 = time.time()
            outputs = model(**inputs)
            batch_time = (time.time() - t0) * 1000  # ms

        # Get predictions and probabilities
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        per_sample_time = batch_time / len(texts)

        for i, sample in enumerate(batch_samples):
            pred_label = preds[i].item()
            # Sentinel: label 1 = injection, label 0 = benign
            # Check model's label mapping
            confidence = probs[i][1].item()  # probability of injection class

            results.append({
                "idx": batch_start + i,
                "text": sample["text"][:200],  # truncate for CSV
                "label": sample["label"],
                "source": sample.get("source", "unknown"),
                "sentinel_pred": pred_label,
                "sentinel_confidence": round(confidence, 4),
                "latency_ms": round(per_sample_time, 2),
            })

        if (batch_start // BATCH_SIZE) % 10 == 0:
            elapsed = time.time() - start_time
            pct = batch_end / total * 100
            print(f"  [{batch_end}/{total}] {pct:.1f}% — {elapsed:.1f}s elapsed")

    total_time = time.time() - start_time
    print(f"Evaluation complete: {total} samples in {total_time:.1f}s "
          f"({total/total_time:.1f} samples/sec)")

    return results


def compute_metrics(results):
    """Compute precision, recall, F1, AUC-ROC with bootstrap CIs."""
    labels = np.array([r["label"] for r in results])
    preds = np.array([r["sentinel_pred"] for r in results])
    confs = np.array([r["sentinel_confidence"] for r in results])

    # Overall metrics
    metrics = {
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "auc_roc": round(roc_auc_score(labels, confs), 4),
    }

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0][0]), "fp": int(cm[0][1]),
        "fn": int(cm[1][0]), "tp": int(cm[1][1])
    }

    # Bootstrap CIs (500 iterations)
    n = len(labels)
    boot_metrics = {"precision": [], "recall": [], "f1": []}
    rng = np.random.RandomState(42)
    for _ in range(500):
        idx = rng.randint(0, n, size=n)
        bl, bp = labels[idx], preds[idx]
        if len(np.unique(bl)) < 2:
            continue
        boot_metrics["precision"].append(precision_score(bl, bp, zero_division=0))
        boot_metrics["recall"].append(recall_score(bl, bp, zero_division=0))
        boot_metrics["f1"].append(f1_score(bl, bp, zero_division=0))

    metrics["bootstrap_ci_95"] = {}
    for k, vals in boot_metrics.items():
        vals = sorted(vals)
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals))]
        metrics["bootstrap_ci_95"][k] = [round(lo, 4), round(hi, 4)]

    # Per-source breakdown
    df = pd.DataFrame(results)
    per_source = {}
    for source, group in df.groupby("source"):
        gl = group["label"].values
        gp = group["sentinel_pred"].values
        gc = group["sentinel_confidence"].values
        per_source[source] = {
            "n": len(group),
            "precision": round(precision_score(gl, gp, zero_division=0), 4),
            "recall": round(recall_score(gl, gp, zero_division=0), 4),
            "f1": round(f1_score(gl, gp, zero_division=0), 4),
            "auc_roc": round(roc_auc_score(gl, gc), 4) if len(np.unique(gl)) > 1 else None,
        }
    metrics["per_source"] = per_source

    # Latency stats
    latencies = [r["latency_ms"] for r in results]
    metrics["latency"] = {
        "mean_ms": round(np.mean(latencies), 2),
        "std_ms": round(np.std(latencies), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
    }

    return metrics


def main():
    print("=" * 60)
    print("SENTINEL BASELINE EVALUATION")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Load dataset
    samples = load_unified_dataset()

    # Load model
    model, tokenizer, device = load_sentinel_model()

    # Run evaluation
    results = run_sentinel_evaluation(model, tokenizer, device, samples)

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["model"] = MODEL_NAME
    metrics["n_samples"] = len(results)
    metrics["device"] = device

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1:        {metrics['f1']}")
    print(f"  AUC-ROC:   {metrics['auc_roc']}")
    print(f"  Accuracy:  {metrics['accuracy']}")
    cm = metrics["confusion_matrix"]
    print(f"  TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")
    print(f"  Latency:   {metrics['latency']['mean_ms']:.1f} ms/sample")
    print(f"\n  Bootstrap 95% CIs:")
    for k, ci in metrics["bootstrap_ci_95"].items():
        print(f"    {k}: [{ci[0]}, {ci[1]}]")
    print(f"\n  Per-source breakdown:")
    for src, m in metrics["per_source"].items():
        print(f"    {src} (n={m['n']}): F1={m['f1']}, Prec={m['precision']}, Rec={m['recall']}")

    # Save results
    json_path = RESULTS_DIR / "sentinel_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {json_path}")

    # Save per-sample CSV
    csv_path = RESULTS_DIR / "sentinel_per_sample.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Per-sample results saved to: {csv_path}")

    # Print comparison with HIVE
    print("\n" + "=" * 60)
    print("COMPARISON WITH HIVE")
    print("=" * 60)
    print(f"{'Metric':<15} {'Sentinel':<12} {'HIVE Full':<12} {'HIVE Local':<12}")
    print("-" * 51)
    print(f"{'Precision':<15} {metrics['precision']:<12} {'0.921':<12} {'0.841':<12}")
    print(f"{'Recall':<15} {metrics['recall']:<12} {'0.859':<12} {'0.360':<12}")
    print(f"{'F1':<15} {metrics['f1']:<12} {'0.889':<12} {'0.504':<12}")
    print(f"{'AUC-ROC':<15} {metrics['auc_roc']:<12} {'0.910':<12} {'0.665':<12}")


if __name__ == "__main__":
    main()
