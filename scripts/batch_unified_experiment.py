#!/usr/bin/env python3
import asyncio
import logging
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# Adjust import to your project structure:
from src.honey_prompt_detector.main import HoneyPromptSystem
from src.honey_prompt_detector.utils.config import Config
from src.honey_prompt_detector.utils.logging import setup_logger

config = Config()
logger = setup_logger(
    name="",    # your chosen root logger name
    log_file=config.log_file,
    level=logging.DEBUG, # or logging.DEBUG if you want debug logs
    retention_days=config.retention_days
)

# Paths and constants
DATA_FILE = Path(__file__).parent.parent / "data" / "unified_dataset.csv"
MISMATCH_CSV = Path(__file__).parent.parent / "results" / "mismatches.csv"
METRICS_JSON = Path(__file__).parent.parent / "results" / "batch_experiment_metrics.json"
MAX_SAMPLES = 500  # how many rows to sample

class BatchUnifiedExperiment:
    """
    1) Loads unified_dataset.csv (with 'text' and 'label'),
    2) Randomly samples up to MAX_SAMPLES rows,
    3) For each row, calls monitor_text() on 'text',
       prints results in a style like basic_usage,
       tracks mismatches if predicted != label,
    4) Computes confusion matrix, accuracy, average confidence/time,
       saves mismatches.csv and metrics.json.
    """

    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.mismatches = []
        self.results = []  # row-level info
        self.system = None

    async def run(self):
        # 1) Load data
        df = pd.read_csv(self.data_file, encoding="utf-8", keep_default_na=False)
        if "text" not in df.columns or "label" not in df.columns:
            print(f"Error: dataset must have 'text' and 'label' columns. Found {df.columns.tolist()}")
            return

        total_rows = len(df)
        if total_rows == 0:
            print("Dataset is empty.")
            return

        # 2) Randomly sample up to MAX_SAMPLES
        if total_rows > MAX_SAMPLES:
            df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
            print(f"Sampling {MAX_SAMPLES} rows out of {total_rows} total.")
        else:
            print(f"Dataset has {total_rows} rows (<= {MAX_SAMPLES}), using all.")

        # 3) Initialize HoneyPromptSystem
        self.system = HoneyPromptSystem()
        started = await self.system.start()
        if not started:
            print("Failed to initialize honey-prompt detector.")
            return

        print("\nRunning Unified Dataset Classification")
        print("=========================================")
        start_time = time.time()

        correct_count = 0
        for idx, row in df.iterrows():
            text = row["text"]
            true_label = row["label"]  # 0=benign, 1=malicious
            ground_truth_str = "malicious" if true_label == 1 else "benign"

            t0 = time.time()
            # 4) Call monitor_text
            result = await self.system.monitor_text(text)
            elapsed = time.time() - t0

            # Convert detection => predicted_label
            detected = result.get("detection", False)
            predicted_label = 1 if detected else 0
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "")
            risk_level = result.get("risk_level", "unknown")

            match = (predicted_label == true_label)
            if match:
                correct_count += 1

            # 5) Print in basic_usage style:
            status = "✓" if match else "✗"
            truncated_text = text[:50].replace("\n", "\\n")  # avoid newlines
            print(f"{status} [{ground_truth_str}] {truncated_text}... "
                  f"Confidence: {confidence:.2f} | Time: {elapsed:.2f}s")

            # Store row-level results
            self.results.append({
                "index": idx,
                "text": text[:100],  # store up to 100 chars
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "risk_level": risk_level,
                "explanation": explanation,
                "elapsed_s": elapsed
            })

            # If mismatch, store in self.mismatches
            if not match:
                self.mismatches.append({
                    "index": idx,
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "explanation": explanation
                })

        total_time = time.time() - start_time
        print(f"\nProcessed {len(df)} rows in {total_time:.2f}s.")
        accuracy = correct_count / len(df)
        print(f"Accuracy: {accuracy*100:.2f}% ({correct_count}/{len(df)})")

        # 6) Save mismatches if any
        if self.mismatches:
            mismatch_df = pd.DataFrame(self.mismatches)
            MISMATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
            mismatch_df.to_csv(MISMATCH_CSV, index=False, encoding="utf-8")
            print(f"Saved {len(self.mismatches)} mismatches to {MISMATCH_CSV}")

        # 7) Analyze + save metrics
        metrics = self._analyze_metrics()
        METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
        METRICS_JSON.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to {METRICS_JSON}")

        # 8) Stop system
        await self.system.stop()

    def _analyze_metrics(self) -> Dict[str, Any]:
        df_res = pd.DataFrame(self.results)
        total = len(df_res)
        correct = (df_res["true_label"] == df_res["predicted_label"]).sum()
        accuracy = correct / total if total else 0.0

        # confusion matrix
        tp = len(df_res[(df_res["true_label"] == 1) & (df_res["predicted_label"] == 1)])
        tn = len(df_res[(df_res["true_label"] == 0) & (df_res["predicted_label"] == 0)])
        fp = len(df_res[(df_res["true_label"] == 0) & (df_res["predicted_label"] == 1)])
        fn = len(df_res[(df_res["true_label"] == 1) & (df_res["predicted_label"] == 0)])

        avg_conf = df_res["confidence"].mean() if not df_res.empty else 0.0
        avg_time = df_res["elapsed_s"].mean() if not df_res.empty else 0.0

        return {
            "total_rows": total,
            "accuracy": accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "avg_confidence": avg_conf,
            "avg_elapsed_s": avg_time
        }

async def main():
    runner = BatchUnifiedExperiment(DATA_FILE)
    await runner.run()

if __name__ == "__main__":
    asyncio.run(main())
