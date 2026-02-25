#!/usr/bin/env python3
"""
Self-Tuner Calibration and Stability Analysis

Simulates the EnhancedSelfTuner over 1000+ rounds to test:
1. Stability under normal conditions (balanced data)
2. Stability under distribution shift (sudden attack ratio change)
3. Oscillation behavior with adversarial confidence perturbations
4. Thompson Sampling convergence
5. Pseudo-label accuracy evolution
"""

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from honey_prompt_detector.core.self_tuner import EnhancedSelfTuner, ThresholdArm
from honey_prompt_detector.utils.config import Config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""

    num_rounds: int = 1500
    normal_attack_ratio: float = 0.3  # 30% of samples are attacks
    shift_attack_ratio: float = 0.7  # 70% after shift
    shift_start_round: int = 750  # When distribution shift occurs
    seed: int = 42
    adversarial_perturbation_ratio: float = 0.1  # 10% of confidences perturbed


class MockDetector:
    """Mock Detector for simulation."""

    def __init__(self, initial_threshold: float = 0.80):
        self.current_threshold = initial_threshold


class SimulatedDataGenerator:
    """Generate simulated detection results."""

    def __init__(self, attack_ratio: float = 0.3, seed: int = 42):
        self.attack_ratio = attack_ratio
        self.rng = random.Random(seed)
        self.attack_seed = seed

    def generate_sample(self) -> Tuple[bool, float]:
        """
        Generate a simulated detection result.

        Returns:
            (is_attack, confidence): Whether it's an attack and model confidence
        """
        is_attack = self.rng.random() < self.attack_ratio

        if is_attack:
            # Attack: confidence should be higher on average
            confidence = self.rng.betavariate(alpha=5, beta=2)  # Skewed towards 1
        else:
            # Benign: confidence should be lower on average
            confidence = self.rng.betavariate(alpha=2, beta=5)  # Skewed towards 0

        return is_attack, confidence

    def update_attack_ratio(self, new_ratio: float):
        """Update attack ratio for distribution shift simulation."""
        self.attack_ratio = new_ratio


class TunerSimulator:
    """Simulates self-tuner behavior over multiple rounds."""

    def __init__(self, config: SimulationConfig, real_config: Any):
        self.config = config
        self.rng = random.Random(config.seed)
        self.detector = MockDetector()
        self.tuner = EnhancedSelfTuner(
            detector_agent=self.detector,
            config=real_config,
            high_confidence_threshold=0.95,
            low_confidence_threshold=0.30,
            consistency_window=5,
            drift_detection_window=100,
        )
        self.data_gen = SimulatedDataGenerator(
            attack_ratio=config.normal_attack_ratio, seed=config.seed
        )

        # Tracking metrics
        self.round_history: List[Dict[str, Any]] = []
        self.threshold_history: List[float] = []
        self.confidence_history: List[float] = []
        self.pseudo_label_accuracy: List[float] = []
        self.beta_params_history: List[Dict[str, List[Tuple[int, int]]]] = []
        self.arm_selection_history: List[int] = []
        self.oscillation_count = 0

    def run_simulation(self) -> Dict[str, Any]:
        """Run the full simulation."""
        logger.info(f"Starting simulation with {self.config.num_rounds} rounds")

        for round_num in range(self.config.num_rounds):
            # Handle distribution shift
            if round_num == self.config.shift_start_round:
                logger.info(
                    f"Distribution shift at round {round_num}: "
                    f"Attack ratio {self.config.normal_attack_ratio} -> {self.config.shift_attack_ratio}"
                )
                self.data_gen.update_attack_ratio(self.config.shift_attack_ratio)

            # Generate sample
            is_attack, confidence = self.data_gen.generate_sample()

            # Apply adversarial perturbation occasionally
            if round_num > 500 and self.rng.random() < self.config.adversarial_perturbation_ratio:
                confidence = self._perturb_confidence(confidence, is_attack)

            # Record detection
            detection_result = {
                "detection": is_attack,
                "confidence": confidence,
                "features": {},
            }
            text_hash = f"text_{round_num}"
            record = self.tuner.record_detection(detection_result, text_hash)

            # Track metrics
            self.threshold_history.append(self.tuner.detector_agent.current_threshold)
            self.confidence_history.append(confidence)
            self.arm_selection_history.append(self.tuner.current_arm_idx)

            # Simulate human label with high confidence predictions
            if record.pseudo_label is not None and self.rng.random() < 0.05:
                # 5% of pseudo-labeled items get human review
                self.tuner.update_with_human_label(text_hash, record.pseudo_label)

            # Periodically adjust threshold using Thompson Sampling
            if round_num % 50 == 0 and round_num > 0:
                old_threshold = self.tuner.detector_agent.current_threshold
                self.tuner.adjust_threshold_thompson_sampling()
                new_threshold = self.tuner.detector_agent.current_threshold

                if abs(new_threshold - old_threshold) > 0.001:
                    self.oscillation_count += 1

            # Also adjust using pseudo-labels
            if round_num % 100 == 0 and round_num > 0:
                self.tuner.adjust_threshold_pseudo_labels()

            # Track beta parameters for current arm
            if round_num % 100 == 0:
                beta_params = {
                    f"arm_{i}": (arm.successes, arm.failures)
                    for i, arm in enumerate(self.tuner.threshold_arms)
                }
                self.beta_params_history.append(beta_params)

            # Calculate pseudo-label accuracy
            if self.tuner.pseudo_labeled_count > 0:
                self._update_pseudo_label_accuracy()

            # Store round summary
            if round_num % 50 == 0:
                metrics = self.tuner.get_metrics()
                self.round_history.append(
                    {
                        "round": round_num,
                        "threshold": self.tuner.detector_agent.current_threshold,
                        "attack_ratio": self.data_gen.attack_ratio,
                        **metrics,
                    }
                )

        logger.info("Simulation completed")
        return self._generate_results()

    def _perturb_confidence(self, confidence: float, is_attack: bool) -> float:
        """Apply adversarial perturbation to confidence."""
        direction = self.rng.choice([-1, 1])
        perturbation = self.rng.gauss(0, 0.15) * direction
        perturbed = max(0.0, min(1.0, confidence + perturbation))
        return perturbed

    def _count_arm_switches(self) -> int:
        """Count how many times the Thompson Sampling arm switched."""
        switches = 0
        for i in range(1, len(self.arm_selection_history)):
            if self.arm_selection_history[i] != self.arm_selection_history[i - 1]:
                switches += 1
        return switches

    def _update_pseudo_label_accuracy(self):
        """Track pseudo-label accuracy based on pseudo estimates."""
        total = (
            self.tuner.estimated_tp
            + self.tuner.estimated_fp
            + self.tuner.estimated_fn
            + self.tuner.estimated_tn
        )
        if total > 0:
            accuracy = (self.tuner.estimated_tp + self.tuner.estimated_tn) / total
            self.pseudo_label_accuracy.append(accuracy)

    def _generate_results(self) -> Dict[str, Any]:
        """Generate summary results."""

        # Calculate convergence metrics
        convergence_speed = self._estimate_convergence_speed()
        shift_recovery = self._estimate_shift_recovery()
        stability = self._estimate_stability()
        oscillation_freq = self._estimate_oscillation_frequency()

        # Get final metrics
        final_metrics = self.tuner.get_metrics()

        # Calculate TP/FP/FN/TN for final state
        total_pseudo = (
            self.tuner.estimated_tp
            + self.tuner.estimated_fp
            + self.tuner.estimated_fn
            + self.tuner.estimated_tn
        )

        return {
            "simulation_config": asdict(self.config),
            "convergence_metrics": {
                "rounds_to_convergence": convergence_speed,
                "final_threshold": self.threshold_history[-1],
                "threshold_std_dev": float(np.std(self.threshold_history[-500:])),
                "threshold_mean": float(np.mean(self.threshold_history[-500:])),
            },
            "distribution_shift_metrics": {
                "shift_round": self.config.shift_start_round,
                "recovery_time": shift_recovery,
                "pre_shift_threshold": float(np.mean(self.threshold_history[700:750])),
                "post_shift_threshold": float(np.mean(self.threshold_history[900:950])),
            },
            "stability_metrics": {
                "threshold_variance": stability["variance"],
                "threshold_range": stability["range"],
                "threshold_changes_per_100_rounds": stability["change_frequency"],
            },
            "oscillation_metrics": {
                "oscillation_count": self.oscillation_count,
                "oscillation_frequency": oscillation_freq,
                "arm_switches": self._count_arm_switches(),
            },
            "final_tuner_state": {
                "current_threshold": final_metrics["current_threshold"],
                "total_evaluations": final_metrics["total_evaluations"],
                "pseudo_labeled_count": final_metrics["pseudo_labeled_count"],
                "human_labeled_count": final_metrics["human_labeled_count"],
                "estimated_precision": final_metrics["estimated_precision"],
                "estimated_recall": final_metrics["estimated_recall"],
                "arm_successes": final_metrics["current_arm"]["successes"],
                "arm_failures": final_metrics["current_arm"]["failures"],
            },
            "pseudo_label_accuracy": {
                "mean_accuracy": float(
                    np.mean(self.pseudo_label_accuracy) if self.pseudo_label_accuracy else 0.0
                ),
                "samples": len(self.pseudo_label_accuracy),
            },
        }

    def _estimate_convergence_speed(self) -> int:
        """Estimate rounds to convergence (within 0.05 of optimal threshold)."""
        # Assume optimal is around 0.70-0.75 based on typical detector tuning
        optimal_threshold = 0.72
        tolerance = 0.05

        for i, threshold in enumerate(self.threshold_history):
            if abs(threshold - optimal_threshold) <= tolerance:
                return i

        return -1  # Did not converge

    def _estimate_shift_recovery(self) -> int:
        """Estimate rounds to recover from distribution shift."""
        shift_start = self.config.shift_start_round
        baseline = np.mean(self.threshold_history[700:750])

        # Look for stabilization after shift
        for i in range(shift_start + 50, min(shift_start + 400, len(self.threshold_history))):
            window_mean = np.mean(self.threshold_history[i : i + 50])
            if abs(window_mean - baseline) < 0.01:  # Re-stabilized
                return i - shift_start

        return -1  # Did not recover

    def _estimate_stability(self) -> Dict[str, float]:
        """Estimate stability metrics."""
        recent_thresholds = self.threshold_history[-500:]
        return {
            "variance": float(np.var(recent_thresholds)),
            "range": float(max(recent_thresholds) - min(recent_thresholds)),
            "change_frequency": float(
                sum(
                    1
                    for i in range(1, len(recent_thresholds))
                    if abs(recent_thresholds[i] - recent_thresholds[i - 1]) > 0.001
                )
                / 5
            ),  # Per 100 rounds
        }

    def _estimate_oscillation_frequency(self) -> float:
        """Estimate frequency of threshold oscillations."""
        oscillations = 0
        for i in range(1, len(self.threshold_history) - 1):
            prev = self.threshold_history[i - 1]
            curr = self.threshold_history[i]
            next_val = self.threshold_history[i + 1]

            # Check for direction change (local extrema)
            if (curr > prev and curr > next_val) or (curr < prev and curr < next_val):
                oscillations += 1

        return oscillations / len(self.threshold_history) if self.threshold_history else 0.0


def save_history_for_plotting(simulator: TunerSimulator, output_dir: Path):
    """Save detailed history for external plotting."""
    history_file = output_dir / "tuner_history.json"

    history_data = {
        "threshold_history": simulator.threshold_history,
        "confidence_history": simulator.confidence_history,
        "arm_selection_history": simulator.arm_selection_history,
        "pseudo_label_accuracy": simulator.pseudo_label_accuracy,
        "shift_round": simulator.config.shift_start_round,
    }

    with open(history_file, "w") as f:
        json.dump(history_data, f, indent=2)

    logger.info(f"Saved history to {history_file}")


def main():
    """Run the self-tuner analysis."""
    # Setup
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Load real config
    try:
        real_config = Config()
    except Exception as e:
        logger.warning(f"Could not load real config: {e}, using minimal config")

        class MinimalConfig:
            tuning_batch_size = 10
            learning_rate = 0.01
            fn_penalty_weight = 2.0

        real_config = MinimalConfig()

    # Run simulation
    sim_config = SimulationConfig(
        num_rounds=1500,
        normal_attack_ratio=0.3,
        shift_attack_ratio=0.7,
        shift_start_round=750,
        seed=42,
    )

    simulator = TunerSimulator(sim_config, real_config)
    results = simulator.run_simulation()

    # Save results
    results_file = results_dir / "self_tuner_analysis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {results_file}")

    # Save history for plotting
    save_history_for_plotting(simulator, results_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SELF-TUNER CALIBRATION AND STABILITY ANALYSIS")
    print("=" * 70)

    print("\nCONVERGENCE METRICS:")
    conv = results["convergence_metrics"]
    print(f"  Rounds to convergence (within 0.05 of optimal): {conv['rounds_to_convergence']}")
    print(f"  Final threshold: {conv['final_threshold']:.4f}")
    print(f"  Threshold stability (std dev): {conv['threshold_std_dev']:.4f}")

    print("\nDISTRIBUTION SHIFT METRICS:")
    shift = results["distribution_shift_metrics"]
    print(f"  Shift occurred at round: {shift['shift_round']}")
    print(f"  Recovery time (rounds): {shift['recovery_time']}")
    print(f"  Pre-shift threshold: {shift['pre_shift_threshold']:.4f}")
    print(f"  Post-shift threshold: {shift['post_shift_threshold']:.4f}")

    print("\nSTABILITY METRICS:")
    stab = results["stability_metrics"]
    print(f"  Threshold variance: {stab['threshold_variance']:.6f}")
    print(f"  Threshold range: {stab['threshold_range']:.4f}")
    print(f"  Threshold changes per 100 rounds: {stab['threshold_changes_per_100_rounds']:.1f}")

    print("\nOSCILLATION METRICS:")
    osc = results["oscillation_metrics"]
    print(f"  Total oscillation events: {osc['oscillation_count']}")
    print(f"  Oscillation frequency: {osc['oscillation_frequency']:.4f}")
    print(f"  Thompson arm switches: {osc['arm_switches']}")

    print("\nFINAL TUNER STATE:")
    final = results["final_tuner_state"]
    print(f"  Current threshold: {final['current_threshold']:.4f}")
    print(f"  Total evaluations: {final['total_evaluations']}")
    print(f"  Pseudo-labeled: {final['pseudo_labeled_count']}")
    print(f"  Human-labeled: {final['human_labeled_count']}")
    print(f"  Estimated precision: {final['estimated_precision']:.4f}")
    print(f"  Estimated recall: {final['estimated_recall']:.4f}")

    print("\nPSEUDO-LABEL ACCURACY:")
    pla = results["pseudo_label_accuracy"]
    print(f"  Mean accuracy: {pla['mean_accuracy']:.4f}")
    print(f"  Samples: {pla['samples']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
