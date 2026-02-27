#!/usr/bin/env python3
"""
Stage 3 FPR Test on Full Benign Dataset
========================================

Tests Stage 3 (honey-token matching) on ALL 7,554 benign samples from the
unified benchmark to confirm FPR = 0.000.

For each benign sample:
  1. Generate a fresh 128-bit URL-safe random token via secrets.token_urlsafe(16)
  2. Create HoneyPrompt with token and variations [token.lower(), token.upper(), token.replace("-", "_")]
  3. Run detector.analyze_text() with skip_heuristics=True, skip_memory=True (isolates Stage 3)
  4. Record: text_id, source, stage3_detected, match_type, confidence

Computes:
  - FPR = (# detected) / (total benign samples)
  - 95% bootstrap CI (500 iterations)

Expected result: FPR = 0.000 [0.000, 0.000]
  A 128-bit random URL-safe string cannot collide with natural text.

Usage:
  cd honey-prompt-detector
  python scripts/experiment_stage3_fpr_full_benign.py
"""

import json
import secrets
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.honey_prompt_detector.core.detector import Detector
from src.honey_prompt_detector.core.honey_prompt import HoneyPrompt


class SimpleContextEvaluator:
    """Minimal context evaluator for testing."""
    def adjust_confidence(self, confidence, context, expected_context):
        return confidence


def bootstrap_ci(detections, n_bootstrap=500, ci=0.95):
    """
    Compute bootstrap confidence interval for FPR.
    
    Args:
        detections: array of 0/1 flags (1 = detected as false positive)
        n_bootstrap: number of bootstrap iterations (default 500)
        ci: confidence level (default 0.95 for 95% CI)
    
    Returns:
        (lower_bound, upper_bound) percentiles
    """
    arr = np.array(detections)
    boot_fprs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_fprs.append(sample.mean())
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_fprs, alpha * 100)
    upper = np.percentile(boot_fprs, (1 - alpha) * 100)
    return lower, upper


def main():
    print("[*] Stage 3 FPR Test: Full Benign Dataset")
    print("=" * 70)
    
    # ===== Load Data =====
    data_path = Path(__file__).resolve().parent.parent / "data" / "unified_dataset.csv"
    print(f"[*] Loading unified dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    # Filter for benign samples (label == 0)
    benign_df = df[df['label'] == 0].reset_index(drop=True)
    print(f"[*] Total samples: {len(df)}")
    print(f"[*] Benign samples (label=0): {len(benign_df)}")
    
    # ===== Initialize Detector =====
    print("\n[*] Initializing detector...")
    detector = Detector(
        context_evaluator=SimpleContextEvaluator(),
        initial_threshold=0.70,
        heuristic_engine=None,
        attack_memory=None,
    )
    print("[+] Detector initialized")
    
    # ===== Run Stage 3 Analysis =====
    print(f"\n[*] Running Stage 3 analysis on {len(benign_df)} benign samples...")
    
    results = []
    detected_count = 0
    detections = []  # For bootstrap CI
    
    start_time = time.time()
    
    for idx, row in benign_df.iterrows():
        text_id = idx
        source = row['source']
        sample_text = row['text']
        
        # Generate fresh random token
        token = secrets.token_urlsafe(16)
        
        # Create HoneyPrompt with variations
        variations = [
            token.lower(),
            token.upper(),
            token.replace("-", "_"),
        ]
        honey_prompt = HoneyPrompt(
            base_token=token,
            category="test",
            sensitivity=0.9,
            context="",
            variations=variations,
        )
        
        # Run Stage 3 analysis (skip heuristics and memory)
        try:
            analysis = detector.analyze_text(
                text=sample_text,
                honey_prompt=honey_prompt,
                context_window_size=200,
                skip_heuristics=True,
                skip_memory=True,
            )
            
            # Extract Stage 3 results
            stage3_detected = analysis.get('stage3', {}).get('detected', False)
            match_type = analysis.get('stage3', {}).get('match_type', None)
            confidence = analysis.get('stage3', {}).get('confidence', 0.0)
            
            if stage3_detected:
                detected_count += 1
                detections.append(1)
            else:
                detections.append(0)
            
            results.append({
                'text_id': text_id,
                'source': source,
                'stage3_detected': stage3_detected,
                'match_type': match_type,
                'confidence': float(confidence),
                'token': token,
                'sample_text': sample_text[:100],  # First 100 chars for reference
            })
            
        except Exception as e:
            print(f"[!] Error analyzing sample {idx}: {e}")
            detections.append(0)
            results.append({
                'text_id': text_id,
                'source': source,
                'stage3_detected': False,
                'match_type': None,
                'confidence': 0.0,
                'token': token,
                'sample_text': sample_text[:100],
                'error': str(e),
            })
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"  [{idx + 1}/{len(benign_df)}] - {detected_count} detections - {rate:.1f} samples/sec")
    
    elapsed = time.time() - start_time
    
    # ===== Compute FPR and Bootstrap CI =====
    print(f"\n[*] Computing FPR and bootstrap CI...")
    
    fpr = detected_count / len(benign_df)
    lower_ci, upper_ci = bootstrap_ci(detections, n_bootstrap=500, ci=0.95)
    
    print(f"\n[+] Results Summary")
    print("=" * 70)
    print(f"Total benign samples tested: {len(benign_df)}")
    print(f"False positives (Stage 3 detected): {detected_count}")
    print(f"FPR: {fpr:.6f}")
    print(f"95% Bootstrap CI: [{lower_ci:.6f}, {upper_ci:.6f}]")
    print(f"Analysis time: {elapsed:.2f} seconds ({elapsed/len(benign_df):.4f} sec/sample)")
    
    # ===== Save Results =====
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = results_dir / "experiment_stage3_fpr_full_benign.json"
    
    output_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'configuration': {
            'total_benign_samples': len(benign_df),
            'token_generator': 'secrets.token_urlsafe(16)',
            'variations': ['token.lower()', 'token.upper()', 'token.replace("-", "_")'],
            'context_window_size': 200,
            'skip_heuristics': True,
            'skip_memory': True,
        },
        'results': {
            'false_positives': detected_count,
            'total_samples': len(benign_df),
            'fpr': float(fpr),
            'bootstrap_ci_lower': float(lower_ci),
            'bootstrap_ci_upper': float(upper_ci),
            'bootstrap_iterations': 500,
            'confidence_level': 0.95,
        },
        'performance': {
            'total_time_seconds': elapsed,
            'avg_time_per_sample': elapsed / len(benign_df),
        },
        'per_sample_results': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n[+] Results saved to {output_file}")
    print("\n[+] Experiment completed successfully!")
    
    return output_data


if __name__ == "__main__":
    main()
