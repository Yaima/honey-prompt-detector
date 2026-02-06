"""
Intelligent Decision Engine for Agents

Provides real autonomy beyond simple if-then rules:
- Multi-factor decision making (considers tradeoffs)
- Predictive modeling (forecasts outcomes)
- Causal reasoning (understands why)
- Cost-benefit analysis
- Meta-learning (learns about learning)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("honey_prompt")


@dataclass
class Decision:
    """Represents a decision with reasoning"""

    action: str
    confidence: float
    reasoning: str
    expected_outcome: Dict[str, float]
    alternatives_considered: List[Dict[str, Any]]


@dataclass
class Factor:
    """A factor to consider in decision-making"""

    name: str
    value: float
    weight: float  # How important is this factor?
    trend: float  # Is it getting better/worse?


class PredictiveModel:
    """Simple predictive model for forecasting outcomes"""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.observations: List[Tuple[Dict[str, float], float]] = []

    def learn(self, features: Dict[str, float], outcome: float) -> None:
        """Learn from an observation"""
        self.observations.append((features, outcome))
        if len(self.observations) > self.history_size:
            self.observations = self.observations[-self.history_size :]

    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict outcome given features.
        Returns: (prediction, confidence)
        """
        if len(self.observations) < 5:
            return 0.5, 0.0  # Not enough data

        # Simple weighted k-nearest neighbors
        similarities = []
        for obs_features, obs_outcome in self.observations:
            # Calculate similarity (inverse euclidean distance)
            distance = self._feature_distance(features, obs_features)
            similarity = 1.0 / (1.0 + distance)
            similarities.append((similarity, obs_outcome))

        # Weight by similarity, take top k=5
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[: min(5, len(similarities))]

        # Weighted average
        total_weight = sum(sim for sim, _ in top_k)
        if total_weight == 0:
            return 0.5, 0.0

        prediction = sum(sim * outcome for sim, outcome in top_k) / total_weight

        # Confidence based on agreement among neighbors
        outcomes = [outcome for _, outcome in top_k]
        variance = np.var(outcomes) if len(outcomes) > 1 else 1.0
        confidence = 1.0 / (1.0 + variance)

        return prediction, confidence

    def _feature_distance(self, f1: Dict[str, float], f2: Dict[str, float]) -> float:
        """Calculate distance between feature sets"""
        all_keys = set(f1.keys()) | set(f2.keys())
        distance_sq = sum((f1.get(k, 0) - f2.get(k, 0)) ** 2 for k in all_keys)
        return np.sqrt(distance_sq)


class CausalReasoner:
    """Understands causal relationships between actions and outcomes"""

    def __init__(self):
        # Track: action -> outcome correlations
        self.action_outcomes: Dict[str, List[float]] = defaultdict(list)

        # Track: context -> action -> outcome
        self.contextual_effects: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def record_outcome(self, action: str, outcome: float, context: Optional[str] = None) -> None:
        """Record an action outcome"""
        self.action_outcomes[action].append(outcome)

        if context:
            self.contextual_effects[context][action].append(outcome)

    def explain_causality(self, action: str, context: Optional[str] = None) -> str:
        """Explain why an action leads to outcomes"""
        if action not in self.action_outcomes or len(self.action_outcomes[action]) < 3:
            return f"Insufficient data to explain causality of '{action}'"

        # Overall effect
        outcomes = self.action_outcomes[action]
        avg_outcome = np.mean(outcomes)
        std_outcome = np.std(outcomes)

        explanation = f"Action '{action}' typically results in {avg_outcome:.2f} ± {std_outcome:.2f}. "

        # Contextual effects
        if context and context in self.contextual_effects:
            ctx_outcomes = self.contextual_effects[context][action]
            if ctx_outcomes:
                ctx_avg = np.mean(ctx_outcomes)
                diff = ctx_avg - avg_outcome

                if abs(diff) > 0.1:
                    direction = "better" if diff > 0 else "worse"
                    explanation += f"In context '{context}', it performs {abs(diff):.2f} {direction} than average. "

        # Trend analysis
        if len(outcomes) >= 10:
            recent = np.mean(outcomes[-5:])
            older = np.mean(outcomes[-10:-5])
            trend = recent - older

            if abs(trend) > 0.1:
                direction = "improving" if trend > 0 else "degrading"
                explanation += f"Recent trend shows {direction} performance ({abs(trend):.2f} change)."

        return explanation

    def predict_causal_effect(self, action: str, context: Optional[str] = None) -> Tuple[float, float]:
        """
        Predict the causal effect of an action.
        Returns: (expected_outcome, confidence)
        """
        if action not in self.action_outcomes:
            return 0.5, 0.0

        outcomes = self.action_outcomes[action]

        # Context-specific prediction
        if context and context in self.contextual_effects and action in self.contextual_effects[context]:
            ctx_outcomes = self.contextual_effects[context][action]
            if ctx_outcomes:
                outcomes = ctx_outcomes

        if not outcomes:
            return 0.5, 0.0

        expected = np.mean(outcomes)
        confidence = 1.0 / (1.0 + np.std(outcomes))  # Lower variance = higher confidence

        return expected, min(1.0, confidence)


class DecisionEngine:
    """
    Intelligent decision engine that goes beyond if-then rules.

    Makes decisions by:
    - Considering multiple factors and their tradeoffs
    - Predicting outcomes using learned patterns
    - Understanding causal relationships
    - Performing cost-benefit analysis
    - Learning from decision outcomes
    """

    def __init__(self):
        self.predictive_model = PredictiveModel()
        self.causal_reasoner = CausalReasoner()

        # Track decision outcomes for meta-learning
        self.decision_history: List[Tuple[Decision, float]] = []

    def make_decision(
        self, situation: str, options: List[str], factors: Dict[str, Factor], context: Optional[str] = None
    ) -> Decision:
        """
        Make an intelligent decision considering multiple factors.

        Args:
            situation: Description of the situation
            options: List of possible actions
            factors: Relevant factors to consider
            context: Optional context about the situation
        """

        # Evaluate each option
        evaluations = []

        for option in options:
            # Predict outcome using learned patterns
            feature_vector = {f.name: f.value for f in factors.values()}
            predicted_outcome, prediction_confidence = self.predictive_model.predict(feature_vector)

            # Get causal understanding
            causal_outcome, causal_confidence = self.causal_reasoner.predict_causal_effect(option, context)

            # Combine predictions (weighted by confidence)
            total_confidence = prediction_confidence + causal_confidence
            if total_confidence > 0:
                combined_outcome = (
                    predicted_outcome * prediction_confidence + causal_outcome * causal_confidence
                ) / total_confidence
                combined_confidence = total_confidence / 2.0
            else:
                # Fallback to factor-based heuristic
                combined_outcome = self._evaluate_by_factors(option, factors)
                combined_confidence = 0.3

            # Cost-benefit analysis
            cost = self._estimate_cost(option)
            benefit = combined_outcome
            net_value = benefit - cost

            evaluations.append(
                {
                    "option": option,
                    "predicted_outcome": combined_outcome,
                    "confidence": combined_confidence,
                    "cost": cost,
                    "benefit": benefit,
                    "net_value": net_value,
                    "reasoning": self.causal_reasoner.explain_causality(option, context),
                }
            )

        # Select best option
        evaluations.sort(key=lambda x: x["net_value"], reverse=True)
        best = evaluations[0]

        decision = Decision(
            action=best["option"],
            confidence=best["confidence"],
            reasoning=(
                f"Selected '{best['option']}' with {best['confidence']:.2f} confidence. "
                f"Expected outcome: {best['predicted_outcome']:.2f}, "
                f"Cost: {best['cost']:.2f}, Net value: {best['net_value']:.2f}. "
                f"{best['reasoning']}"
            ),
            expected_outcome={
                "outcome": best["predicted_outcome"],
                "cost": best["cost"],
                "net_value": best["net_value"],
            },
            alternatives_considered=[{"option": e["option"], "net_value": e["net_value"]} for e in evaluations[1:]],
        )

        logger.info(f"Decision: {decision.action} (confidence: {decision.confidence:.2f})")
        logger.debug(f"Reasoning: {decision.reasoning}")

        return decision

    def _evaluate_by_factors(self, option: str, factors: Dict[str, Factor]) -> float:
        """Fallback: Evaluate option by weighted factor analysis"""
        total_weight = sum(f.weight for f in factors.values())
        if total_weight == 0:
            return 0.5

        # Weighted sum of factors
        # Options that align with improving negative trends score higher
        score = 0.0
        for factor in factors.values():
            # If factor is below target and trending down, action to improve it scores higher
            if factor.value < 0.5 and factor.trend < 0:
                score += factor.weight * (1.0 - factor.value)  # Want to improve it
            elif factor.value > 0.5 and factor.trend > 0:
                score += factor.weight * factor.value  # Maintain it
            else:
                score += factor.weight * 0.5  # Neutral

        return score / total_weight

    def _estimate_cost(self, option: str) -> float:
        """Estimate cost of an action"""
        # Simple heuristic - can be enhanced
        cost_map = {
            "increase_threshold": 0.1,  # Slightly costly (might miss attacks)
            "decrease_threshold": 0.2,  # More costly (more false positives)
            "rotate_token": 0.3,  # Expensive (API call + disruption)
            "switch_strategy": 0.15,  # Moderate cost
            "no_change": 0.0,  # Free
        }

        return cost_map.get(option, 0.1)  # Default moderate cost

    def learn_from_outcome(self, decision: Decision, actual_outcome: float) -> None:
        """Learn from decision outcomes (meta-learning)"""
        # Record for history
        self.decision_history.append((decision, actual_outcome))

        # Update predictive model
        # Extract features from decision context
        features = decision.expected_outcome.copy()
        features["confidence"] = decision.confidence

        self.predictive_model.learn(features, actual_outcome)

        # Update causal reasoner
        self.causal_reasoner.record_outcome(action=decision.action, outcome=actual_outcome)

        # Log learning
        expected = decision.expected_outcome.get("outcome", 0.5)
        error = abs(expected - actual_outcome)

        if error > 0.2:
            logger.warning(
                f"Decision '{decision.action}' had high error: " f"expected {expected:.2f}, got {actual_outcome:.2f}"
            )
        else:
            logger.debug(
                f"Decision '{decision.action}' was accurate: " f"expected {expected:.2f}, got {actual_outcome:.2f}"
            )

    def explain_decision(self, decision: Decision) -> str:
        """Provide detailed explanation of why a decision was made"""
        explanation = [
            f"Decision: {decision.action}",
            f"Confidence: {decision.confidence:.2%}",
            "",
            "Reasoning:",
            decision.reasoning,
            "",
            f"Expected Outcome: {decision.expected_outcome.get('outcome', 0):.2f}",
            f"Expected Cost: {decision.expected_outcome.get('cost', 0):.2f}",
            f"Net Value: {decision.expected_outcome.get('net_value', 0):.2f}",
        ]

        if decision.alternatives_considered:
            explanation.append("")
            explanation.append("Alternatives Considered:")
            for alt in decision.alternatives_considered:
                explanation.append(f"  - {alt['option']}: net value {alt['net_value']:.2f}")

        return "\n".join(explanation)

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about what the engine has learned"""
        return {
            "decisions_made": len(self.decision_history),
            "prediction_accuracy": self._calculate_prediction_accuracy(),
            "causal_relationships_learned": len(self.causal_reasoner.action_outcomes),
            "training_samples": len(self.predictive_model.observations),
        }

    def _calculate_prediction_accuracy(self) -> float:
        """Calculate how accurate predictions have been"""
        if not self.decision_history:
            return 0.0

        errors = [
            abs(decision.expected_outcome.get("outcome", 0.5) - actual_outcome)
            for decision, actual_outcome in self.decision_history
        ]

        # Convert error to accuracy (1 - normalized error)
        avg_error = np.mean(errors)
        return max(0.0, 1.0 - avg_error)
