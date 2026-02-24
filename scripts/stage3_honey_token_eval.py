#!/usr/bin/env python3
"""
Stage 3 Honey-Token Detection Evaluation
==========================================

Evaluates honey-token detection under realistic obfuscation scenarios
using HIVE's actual Detector pipeline (canonicalization + exact/variation/obfuscation matching).

Addresses reviewer Q5: "Can you provide a dedicated evaluation of Stage 3
(honey tokens) with planted tokens in realistically obfuscated outputs
(e.g., Base64, URL-encoded, homoglyphs) to quantify FPR/TPR?"

Produces per-obfuscation TPR and overall FPR on benign text.
"""

import sys
import json
import time
import base64
import urllib.parse
import codecs
import secrets
import string
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.honey_prompt_detector.core.detector import Detector
from src.honey_prompt_detector.core.honey_prompt import HoneyPrompt


# ---------------------------------------------------------------------------
# Simple context evaluator (same as full_pipeline_evaluation.py)
# ---------------------------------------------------------------------------
class SimpleContextEvaluator:
    def adjust_confidence(self, confidence, context, expected_context):
        return confidence


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------
def generate_honey_token(length=32):
    """Generate a cryptographically random honey token (same format as production)."""
    alphabet = string.ascii_letters + string.digits + "-_"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


# ---------------------------------------------------------------------------
# Obfuscation functions — simulate how an attacker-controlled LLM output
# might leak a honey token in various encoded forms
# ---------------------------------------------------------------------------
HOMOGLYPH_MAP = {
    'a': '\u0430', 'e': '\u0435', 'o': '\u043e', 'p': '\u0440',
    'c': '\u0441', 'x': '\u0445', 'i': '\u0456', 's': '\u0455',
}

def obf_plain(token):
    return token

def obf_base64(token):
    return base64.b64encode(token.encode()).decode()

def obf_url(token):
    return urllib.parse.quote(token, safe='')

def obf_html(token):
    return ''.join(f'&#{ord(c)};' for c in token)

def obf_rot13(token):
    return codecs.encode(token, 'rot_13')

def obf_homoglyph(token):
    return ''.join(HOMOGLYPH_MAP.get(c.lower(), c) for c in token)

def obf_zero_width(token):
    return '\u200b'.join(token)

def obf_base64_url(token):
    return urllib.parse.quote(base64.b64encode(token.encode()).decode(), safe='')

def obf_in_sentence(token):
    return f"The configuration key is {token} and it should remain private."

def obf_split(token):
    mid = len(token) // 2
    return f"First half: {token[:mid]}. Second half: {token[mid:]}."

def obf_case_swap(token):
    return token.swapcase()

def obf_hex(token):
    return token.encode().hex()


OBFUSCATIONS = {
    "plain":            obf_plain,
    "base64":           obf_base64,
    "url_encoded":      obf_url,
    "html_entities":    obf_html,
    "rot13":            obf_rot13,
    "homoglyph":        obf_homoglyph,
    "zero_width":       obf_zero_width,
    "base64+url":       obf_base64_url,
    "in_sentence":      obf_in_sentence,
    "split_halves":     obf_split,
    "case_swap":        obf_case_swap,
    "hex_encoded":      obf_hex,
}


# ---------------------------------------------------------------------------
# Benign outputs (no token — for FPR measurement)
# ---------------------------------------------------------------------------
BENIGN_OUTPUTS = [
    "The weather today is sunny with a high of 75 degrees.",
    "Python is a versatile programming language used in many domains.",
    "The capital of France is Paris, located along the Seine River.",
    "To make pasta, boil water, add salt, cook for 8-10 minutes.",
    "Machine learning models learn patterns from training data.",
    "The mitochondria is the powerhouse of the cell.",
    "HTTP status code 200 means the request was successful.",
    "Shakespeare wrote Romeo and Juliet in the late 16th century.",
    "The speed of light is approximately 299,792,458 meters per second.",
    "React is a JavaScript library for building user interfaces.",
    "Photosynthesis converts carbon dioxide and water into glucose.",
    "The Great Wall of China stretches over 13,000 miles.",
    "SQL is used for managing and querying relational databases.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The human genome contains approximately 3 billion base pairs.",
    "Git is a distributed version control system for tracking changes.",
    "The Amazon River is the largest river by discharge volume.",
    "TCP provides reliable, ordered delivery of data between applications.",
    "The Pythagorean theorem states that a² + b² = c².",
    "Kubernetes orchestrates containerized applications across clusters.",
]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_evaluation(n_tokens=100):
    print("=" * 70)
    print("STAGE 3 HONEY-TOKEN DETECTION EVALUATION")
    print("=" * 70)
    print(f"Tokens: {n_tokens} | Obfuscation types: {len(OBFUSCATIONS)} | Benign: {len(BENIGN_OUTPUTS)}")
    print()

    context_eval = SimpleContextEvaluator()
    results = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "times_ms": []})

    for token_idx in range(n_tokens):
        token = generate_honey_token()

        # Create HoneyPrompt with realistic variations
        hp = HoneyPrompt(
            base_token=token,
            category="security_canary",
            sensitivity=1.0,
            context="system security token",
            variations=[
                token.replace("-", "_"),
                token.lower(),
                token.upper(),
            ],
        )

        # Create fresh detector per token (no attack memory — isolate Stage 3)
        detector = Detector(
            context_evaluator=context_eval,
            initial_threshold=0.70,
            heuristic_engine=None,  # We'll skip heuristics
            attack_memory=None,     # No Stage 2
        )

        # --- Test each obfuscation (should detect = TP) ---
        for obf_name, obf_func in OBFUSCATIONS.items():
            obfuscated = obf_func(token)

            t0 = time.perf_counter()
            result = detector.analyze_text(
                text=obfuscated,
                honey_prompt=hp,
                context_window_size=100,
                skip_heuristics=True,   # Isolate Stage 3
                skip_memory=True,       # Isolate Stage 3
            )
            elapsed = (time.perf_counter() - t0) * 1000

            results[obf_name]["times_ms"].append(elapsed)
            if result["matched"]:
                results[obf_name]["tp"] += 1
            else:
                results[obf_name]["fn"] += 1

        # --- Test benign outputs (should NOT detect = TN) ---
        for benign in BENIGN_OUTPUTS:
            t0 = time.perf_counter()
            result = detector.analyze_text(
                text=benign,
                honey_prompt=hp,
                context_window_size=100,
                skip_heuristics=True,
                skip_memory=True,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            results["_benign"]["times_ms"].append(elapsed)
            if result["matched"]:
                results["_benign"]["fp"] += 1
            else:
                results["_benign"]["tn"] += 1

        if (token_idx + 1) % 20 == 0:
            print(f"  Processed {token_idx + 1}/{n_tokens} tokens...")

    # --- Print results ---
    print("\n" + "=" * 70)
    print("RESULTS: Per-Obfuscation True Positive Rate")
    print("=" * 70)
    print(f"\n{'Obfuscation':<20} {'TPR':>8} {'Detected':>12} {'Avg ms':>10}")
    print("-" * 55)

    summary = {}
    for obf_name in OBFUSCATIONS:
        r = results[obf_name]
        total = r["tp"] + r["fn"]
        tpr = r["tp"] / total if total > 0 else 0.0
        avg_ms = sum(r["times_ms"]) / len(r["times_ms"]) if r["times_ms"] else 0

        print(f"{obf_name:<20} {tpr:>8.3f} {r['tp']:>5}/{total:<5} {avg_ms:>10.3f}")
        summary[obf_name] = {
            "tpr": round(tpr, 4), "tp": r["tp"], "fn": r["fn"],
            "total": total, "avg_latency_ms": round(avg_ms, 3),
        }

    # Benign FPR
    rb = results["_benign"]
    total_benign = rb["fp"] + rb["tn"]
    fpr = rb["fp"] / total_benign if total_benign > 0 else 0.0
    avg_ms_b = sum(rb["times_ms"]) / len(rb["times_ms"]) if rb["times_ms"] else 0

    print(f"\n{'BENIGN FPR':<20} {fpr:>8.4f} {rb['fp']:>5}/{total_benign:<5} {avg_ms_b:>10.3f}")

    # Overall
    all_tp = sum(results[o]["tp"] for o in OBFUSCATIONS)
    all_total = sum(results[o]["tp"] + results[o]["fn"] for o in OBFUSCATIONS)
    overall_tpr = all_tp / all_total if all_total > 0 else 0

    print(f"\n{'OVERALL TPR':<20} {overall_tpr:>8.3f} {all_tp:>5}/{all_total}")

    summary["_benign_fpr"] = {"fpr": round(fpr, 4), "fp": rb["fp"], "tn": rb["tn"], "total": total_benign}
    summary["_overall"] = {
        "tpr": round(overall_tpr, 4), "fpr": round(fpr, 4),
        "n_tokens": n_tokens, "n_obfuscations": len(OBFUSCATIONS), "n_benign": total_benign,
    }

    # Save
    out = Path(__file__).parent.parent / "results" / "stage3_honey_token_eval.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out}")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, default=100)
    args = p.parse_args()
    run_evaluation(n_tokens=args.n_tokens)
