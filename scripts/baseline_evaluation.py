#!/usr/bin/env python3
"""
Baseline Model Evaluation
=========================
Evaluates open-source prompt-injection detection models on HIVE's unified
benchmark for direct head-to-head comparison.

Default model: protectai/deberta-v3-base-prompt-injection-v2
  (DeBERTa-v3-base, ~249M params, reported F1≈0.95, Apache 2.0, ungated)

Alternative: qualifire/prompt-injection-sentinel
  (ModernBERT-large, 395M params, F1≈0.94, gated — requires HF login)

Requirements:
    pip install transformers torch pandas scikit-learn numpy

Usage:
    python baseline_evaluation.py                          # ProtectAI (default)
    python baseline_evaluation.py --model sentinel         # Sentinel (needs HF auth)
    python baseline_evaluation.py --model protectai        # ProtectAI (explicit)

Output:
    results/baseline_<model>_results.json  — metrics + per-source breakdown
    results/baseline_<model>_per_sample.csv — per-sample predictions
"""

import argparse
import csv
import json
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------- MODEL REGISTRY ----------
MODELS = {
    "protectai": {
        "hf_name": "protectai/deberta-v3-base-prompt-injection-v2",
        "display_name": "ProtectAI DeBERTa-v3",
        "params": "249M",
        "injection_label": 1,  # label 1 = INJECTION
    },
    "sentinel": {
        "hf_name": "qualifire/prompt-injection-sentinel",
        "display_name": "Sentinel ModernBERT-large",
        "params": "395M",
        "injection_label": 1,  # label 1 = INJECTION
    },
    "deepset": {
        "hf_name": "deepset/deberta-v3-base-injection",
        "display_name": "Deepset DeBERTa-v3",
        "params": "249M",
        "injection_label": 1,
    },
    "promptguard86m": {
        "hf_name": "meta-llama/Llama-Prompt-Guard-2-86M",
        "display_name": "Llama Prompt Guard 2 86M",
        "params": "86M",
        "injection_label": 2,  # label 2 = INJECTION (0=BENIGN, 1=JAILBREAK, 2=INJECTION)
    },
    "promptguard22m": {
        "hf_name": "meta-llama/Llama-Prompt-Guard-2-22M",
        "display_name": "Llama Prompt Guard 2 22M",
        "params": "22M",
        "injection_label": 2,  # label 2 = INJECTION
    },
}

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

    # Quick class distribution
    n_pos = sum(r["label"] for r in rows)
    n_neg = len(rows) - n_pos
    print(f"  Class distribution: {n_neg} benign / {n_pos} malicious "
          f"({n_neg/len(rows)*100:.1f}% / {n_pos/len(rows)*100:.1f}%)")
    return rows


def load_model(model_key):
    """Load model from HuggingFace."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    cfg = MODELS[model_key]
    hf_name = cfg["hf_name"]
    print(f"\nLoading model: {cfg['display_name']}")
    print(f"  HuggingFace: {hf_name}")
    print(f"  Parameters:  {cfg['params']}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    model.eval()

    # Detect label mapping
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
    print(f"  Device:      {device}")
    print(f"  Labels:      {id2label}")

    return model, tokenizer, device, cfg


def detect_malicious_indices(model, cfg):
    """Figure out which output indices correspond to malicious classes.
    Checks the actual classifier head output size to avoid index-out-of-bounds.
    """
    import torch
    id2label = getattr(model.config, "id2label", {})

    malicious_keywords = {"injection", "attack", "malicious", "unsafe", "jailbreak"}

    # Get actual output size from the classifier head weights
    actual_num_classes = None
    for name, param in model.named_parameters():
        if "classifier" in name and param.dim() == 2:
            actual_num_classes = param.shape[0]
            break
    if actual_num_classes is None:
        actual_num_classes = getattr(model.config, "num_labels", 999)
    print(f"  Actual classifier output size: {actual_num_classes}")

    malicious_indices = []
    for idx, label_name in id2label.items():
        int_idx = int(idx)
        if int_idx >= actual_num_classes:
            print(f"  Skipping label {idx} ('{label_name}') — beyond output size {actual_num_classes}")
            continue
        if any(kw in str(label_name).lower() for kw in malicious_keywords):
            malicious_indices.append(int_idx)
            print(f"  Malicious class index: {idx} ('{label_name}')")

    if malicious_indices:
        return malicious_indices

    # Fallback: last index (typically the non-benign class)
    fallback = min(cfg.get("injection_label", 1), actual_num_classes - 1)
    print(f"  Malicious class index: {fallback} (fallback)")
    return [fallback]


def run_evaluation(model, tokenizer, device, cfg, samples):
    """Run model on all samples and collect predictions."""
    import torch

    malicious_indices = detect_malicious_indices(model, cfg)
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

        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        per_sample_time = batch_time / len(texts)

        for i, sample in enumerate(batch_samples):
            raw_pred = preds[i].item()
            # Map to binary: 1 = any malicious class, 0 = benign
            binary_pred = 1 if raw_pred in malicious_indices else 0
            # Confidence = sum of probabilities across all malicious classes
            confidence = sum(probs[i][idx].item() for idx in malicious_indices)

            results.append({
                "idx": batch_start + i,
                "text": sample["text"][:200],
                "label": sample["label"],
                "source": sample.get("source", "unknown"),
                "model_pred": binary_pred,
                "model_confidence": round(confidence, 4),
                "latency_ms": round(per_sample_time, 2),
            })

        batch_num = batch_start // BATCH_SIZE
        if batch_num % 10 == 0:
            elapsed = time.time() - start_time
            pct = batch_end / total * 100
            print(f"  [{batch_end:>5}/{total}] {pct:5.1f}% — {elapsed:.1f}s elapsed")

    total_time = time.time() - start_time
    print(f"\nEvaluation complete: {total} samples in {total_time:.1f}s "
          f"({total/total_time:.1f} samples/sec)")
    return results


def compute_metrics(results):
    """Compute precision, recall, F1, AUC-ROC with bootstrap CIs."""
    labels = np.array([r["label"] for r in results])
    preds = np.array([r["model_pred"] for r in results])
    confs = np.array([r["model_confidence"] for r in results])

    metrics = {
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "auc_roc": round(roc_auc_score(labels, confs), 4),
        "fpr": round(
            (labels == 0).sum() and
            ((preds == 1) & (labels == 0)).sum() / (labels == 0).sum(),
            4
        ),
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
        gp = group["model_pred"].values
        gc = group["model_confidence"].values
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
    parser = argparse.ArgumentParser(description="Baseline model evaluation on HIVE benchmark")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="protectai",
                        help="Model to evaluate (default: protectai)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    model_key = args.model

    print("=" * 60)
    print("BASELINE MODEL EVALUATION")
    print(f"Model: {MODELS[model_key]['display_name']}")
    print(f"HuggingFace: {MODELS[model_key]['hf_name']}")
    print("=" * 60)

    # Load dataset
    samples = load_unified_dataset()

    # Load model
    model, tokenizer, device, cfg = load_model(model_key)

    # Run evaluation
    results = run_evaluation(model, tokenizer, device, cfg, samples)

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["model"] = MODELS[model_key]["hf_name"]
    metrics["model_key"] = model_key
    metrics["display_name"] = MODELS[model_key]["display_name"]
    metrics["n_samples"] = len(results)
    metrics["device"] = device

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Model:     {MODELS[model_key]['display_name']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1:        {metrics['f1']}")
    print(f"  AUC-ROC:   {metrics['auc_roc']}")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  FPR:       {metrics['fpr']}")
    cm = metrics["confusion_matrix"]
    print(f"  TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")
    print(f"  Latency:   {metrics['latency']['mean_ms']:.1f} ± {metrics['latency']['std_ms']:.1f} ms/sample")

    print(f"\n  Bootstrap 95% CIs:")
    for k, ci in metrics["bootstrap_ci_95"].items():
        print(f"    {k}: [{ci[0]}, {ci[1]}]")

    print(f"\n  Per-source breakdown:")
    for src, m in metrics["per_source"].items():
        print(f"    {src} (n={m['n']}): F1={m['f1']}, Prec={m['precision']}, Rec={m['recall']}")

    # Save results
    slug = model_key
    json_path = RESULTS_DIR / f"baseline_{slug}_results.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {json_path}")

    csv_path = RESULTS_DIR / f"baseline_{slug}_per_sample.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Per-sample results saved to: {csv_path}")

    # Print comparison with HIVE
    print("\n" + "=" * 60)
    print("COMPARISON WITH HIVE")
    print("=" * 60)
    print(f"{'Metric':<15} {'Baseline':<14} {'HIVE Full':<12} {'HIVE Local':<12}")
    print("-" * 53)
    print(f"{'Precision':<15} {metrics['precision']:<14} {'0.921':<12} {'0.841':<12}")
    print(f"{'Recall':<15} {metrics['recall']:<14} {'0.859':<12} {'0.360':<12}")
    print(f"{'F1':<15} {metrics['f1']:<14} {'0.889':<12} {'0.504':<12}")
    print(f"{'AUC-ROC':<15} {metrics['auc_roc']:<14} {'0.910':<12} {'0.665':<12}")
    print(f"{'FPR':<15} {metrics['fpr']:<14} {'0.079':<12} {'0.050':<12}")


if __name__ == "__main__":
    main()
