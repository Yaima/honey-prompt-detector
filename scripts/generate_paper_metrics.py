#!/usr/bin/env python3
"""Generate paper-quality evaluation metrics on the unified dataset."""

import argparse
import csv
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.honey_prompt_detector.core.heuristic_rules import HeuristicRulesEngine
from src.honey_prompt_detector.monitoring.evaluation_metrics import EvaluationMetrics, DetectionResult


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation metrics for paper")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="results/evaluation_report.json")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    # Load dataset
    dataset_path = Path("data/unified_dataset.csv")
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Stratified sample
    benign = [r for r in rows if r["label"] == "0"]
    malicious = [r for r in rows if r["label"] == "1"]

    n_benign = min(args.samples // 2, len(benign))
    n_malicious = min(args.samples - n_benign, len(malicious))

    sample = random.sample(benign, n_benign) + random.sample(malicious, n_malicious)
    random.shuffle(sample)

    print(f"Evaluating {len(sample)} samples ({n_benign} benign, {n_malicious} malicious)")

    # Run evaluation
    engine = HeuristicRulesEngine()
    evaluator = EvaluationMetrics()

    for i, row in enumerate(sample):
        text = row["text"]
        actual_attack = row["label"] == "1"

        # Run heuristic detection
        quick_match = engine.scan_quick(text)
        if quick_match:
            predicted = True
            confidence = quick_match.confidence
        else:
            matches = engine.scan(text)
            if matches:
                predicted = True
                confidence = matches[0].confidence
            else:
                predicted = False
                confidence = 0.0

        evaluator.add_result(DetectionResult(
            text=text[:200],
            predicted_attack=predicted,
            confidence=confidence,
            actual_attack=actual_attack,
            category=row.get("source", "unknown"),
        ))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(sample)}...")

    # Generate report
    report = evaluator.generate_report(threshold=0.5)

    # Add ROC curve data
    roc_data = evaluator.calculate_roc_curve(num_thresholds=50)
    report["roc_curve"] = [{"threshold": t, "fpr": f, "tpr": tp} for t, f, tp in roc_data]

    # Add AUC
    report["auc"] = evaluator.calculate_auc()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    m = report["metrics"]
    print(f"Precision:   {m['precision']:.4f}")
    print(f"Recall:      {m['recall']:.4f}")
    print(f"F1:          {m['f1']:.4f}")
    print(f"Accuracy:    {m['accuracy']:.4f}")
    print(f"FPR:         {m['fpr']:.4f}")
    print(f"AUC:         {report['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = report["confusion_matrix"]
    print(f"  TP={cm['tp']} FP={cm['fp']}")
    print(f"  FN={cm['fn']} TN={cm['tn']}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
