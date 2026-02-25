#!/usr/bin/env python3
"""Compare HIVE baselines on the unified dataset."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.honey_prompt_detector.baselines.heuristic_baseline import HeuristicBaseline
from src.honey_prompt_detector.baselines.keyword_baseline import KeywordBaseline
from src.honey_prompt_detector.monitoring.evaluation_metrics import EvaluationMetrics, DetectionResult


def evaluate_detector(name, detector, samples):
    """Run a detector on samples and return metrics."""
    evaluator = EvaluationMetrics()

    for row in samples:
        text = row["text"]
        actual_attack = row["label"] == "1"
        result = detector.detect(text)

        evaluator.add_result(DetectionResult(
            text=text[:200],
            predicted_attack=result["detection"],
            confidence=result["confidence"],
            actual_attack=actual_attack,
            category=row.get("source", "unknown"),
        ))

    report = evaluator.generate_report(threshold=0.5)
    auc = evaluator.calculate_auc()
    report["auc"] = auc
    return report


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/baseline_comparison.json")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load dataset
    rows = []
    with open("data/unified_dataset.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    benign = [r for r in rows if r["label"] == "0"]
    malicious = [r for r in rows if r["label"] == "1"]
    n_benign = min(args.samples // 2, len(benign))
    n_malicious = min(args.samples - n_benign, len(malicious))
    sample = random.sample(benign, n_benign) + random.sample(malicious, n_malicious)
    random.shuffle(sample)

    print(f"Running baseline comparison on {len(sample)} samples")
    print(f"  Benign: {n_benign}, Malicious: {n_malicious}\n")

    # Run detectors
    detectors = {
        "keyword_baseline": KeywordBaseline(),
        "heuristic_baseline": HeuristicBaseline(),
    }

    results = {"dataset": {"samples": len(sample), "benign": n_benign, "malicious": n_malicious}}

    for name, detector in detectors.items():
        print(f"Evaluating: {name}...")
        report = evaluate_detector(name, detector, sample)
        m = report["metrics"]
        results[name] = {
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "accuracy": round(m["accuracy"], 4),
            "fpr": round(m["fpr"], 4),
            "auc": round(report["auc"], 4),
        }
        print(f"  Precision={m['precision']:.4f} Recall={m['recall']:.4f} F1={m['f1']:.4f} AUC={report['auc']:.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
