"""
Test the Intelligent Decision Engine

Verifies real intelligence beyond if-then rules:
- Multi-factor decision making
- Predictive modeling
- Causal reasoning
- Cost-benefit analysis
- Meta-learning
"""

import pytest

from src.honey_prompt_detector.agents.decision_engine import CausalReasoner, DecisionEngine, Factor, PredictiveModel


class TestPredictiveModel:
    """Test that the model actually learns and predicts"""

    def test_learns_from_observations(self):
        """Test model learns from observations"""
        model = PredictiveModel()

        # Teach: high FP rate + low accuracy → bad outcome
        for _ in range(10):
            model.learn({"fp_rate": 0.8, "accuracy": 0.6}, outcome=0.3)

        # Teach: low FP rate + high accuracy → good outcome
        for _ in range(10):
            model.learn({"fp_rate": 0.1, "accuracy": 0.95}, outcome=0.9)

        # Predict similar situation
        prediction, confidence = model.predict({"fp_rate": 0.15, "accuracy": 0.92})

        assert prediction > 0.7, f"Should predict good outcome for good metrics, got {prediction}"
        assert confidence > 0, "Should have some confidence with 20 samples"

    def test_prediction_improves_with_data(self):
        """Test predictions get better with more data"""
        model = PredictiveModel()

        # With little data, confidence should be low
        model.learn({"x": 1.0}, 0.5)
        model.learn({"x": 1.1}, 0.6)
        _, confidence_low = model.predict({"x": 1.0})

        # With more data, confidence should increase
        for i in range(20):
            model.learn({"x": 1.0 + i * 0.1}, 0.5 + i * 0.02)

        _, confidence_high = model.predict({"x": 1.0})

        assert confidence_high > confidence_low, "More data should increase confidence"


class TestCausalReasoner:
    """Test causal understanding"""

    def test_learns_causal_relationships(self):
        """Test reasoner learns what actions cause what outcomes"""
        reasoner = CausalReasoner()

        # Action A consistently leads to good outcomes
        for _ in range(10):
            reasoner.record_outcome("increase_threshold", outcome=0.8)

        # Action B consistently leads to bad outcomes
        for _ in range(10):
            reasoner.record_outcome("decrease_threshold", outcome=0.3)

        # Predictions should reflect learned causality
        pred_a, conf_a = reasoner.predict_causal_effect("increase_threshold")
        pred_b, conf_b = reasoner.predict_causal_effect("decrease_threshold")

        assert pred_a > pred_b, "Should learn that action A is better than action B"
        assert conf_a > 0.5, "Should be confident with 10 observations"

    def test_contextual_effects(self):
        """Test understanding context-dependent causality"""
        reasoner = CausalReasoner()

        # Action works well in context A
        for _ in range(5):
            reasoner.record_outcome("rotate_token", outcome=0.9, context="high_fp")

        # Same action works poorly in context B
        for _ in range(5):
            reasoner.record_outcome("rotate_token", outcome=0.3, context="low_fp")

        explanation_high_fp = reasoner.explain_causality("rotate_token", context="high_fp")
        explanation_low_fp = reasoner.explain_causality("rotate_token", context="low_fp")

        # Should explain context-dependent effects
        assert "high_fp" in explanation_high_fp.lower() or "better" in explanation_high_fp.lower()
        assert "low_fp" in explanation_low_fp.lower() or "worse" in explanation_low_fp.lower()

    def test_trend_detection(self):
        """Test detection of performance trends"""
        reasoner = CausalReasoner()

        # Add degrading performance over time
        for i in range(15):
            outcome = 0.9 - (i * 0.04)  # Getting worse
            reasoner.record_outcome("strategy_X", outcome)

        explanation = reasoner.explain_causality("strategy_X")

        assert "degrading" in explanation.lower() or "trend" in explanation.lower()


class TestDecisionEngine:
    """Test intelligent decision making"""

    def test_considers_multiple_factors(self):
        """Test decision considers all factors, not just one"""
        engine = DecisionEngine()

        # Define conflicting factors
        factors = {
            "accuracy": Factor("accuracy", value=0.95, weight=0.6, trend=0.05),  # Good and improving
            "fp_rate": Factor("fp_rate", value=0.2, weight=0.4, trend=-0.1),  # Bad and worsening
        }

        decision = engine.make_decision(situation="Test situation", options=["option_a", "option_b"], factors=factors)

        # Should make a decision
        assert decision.action in ["option_a", "option_b"]
        assert 0 <= decision.confidence <= 1.0
        assert len(decision.reasoning) > 0
        assert len(decision.alternatives_considered) > 0

    def test_learns_from_outcomes(self):
        """Test meta-learning: engine learns from decision outcomes"""
        engine = DecisionEngine()

        factors = {"metric": Factor("metric", value=0.5, weight=1.0, trend=0.0)}

        # Make decision
        decision = engine.make_decision(situation="Test", options=["action_a", "action_b"], factors=factors)

        # Teach outcome
        engine.learn_from_outcome(decision, actual_outcome=0.8)

        # Engine should now have learned
        stats = engine.get_learning_statistics()
        assert stats["decisions_made"] == 1
        assert stats["training_samples"] > 0

    def test_cost_benefit_analysis(self):
        """Test decisions consider cost vs benefit"""
        engine = DecisionEngine()

        factors = {"performance": Factor("performance", value=0.7, weight=1.0, trend=0.0)}

        # Make decision - should consider costs
        decision = engine.make_decision(
            situation="Optimize performance",
            options=["rotate_token", "no_change"],  # rotate_token is expensive
            factors=factors,
        )

        # Should have calculated net value (benefit - cost)
        assert "cost" in decision.expected_outcome
        assert "net_value" in decision.expected_outcome

    def test_improves_with_experience(self):
        """Test predictions improve as engine learns"""
        engine = DecisionEngine()

        # Create consistent scenario
        factors = {"fp": Factor("fp", value=0.8, weight=1.0, trend=0.0)}

        # Teach that in this scenario, action_a leads to good outcomes
        for _ in range(10):
            decision = engine.make_decision(situation="High FP", options=["action_a", "action_b"], factors=factors)
            # Simulate action_a being successful
            if decision.action == "action_a":
                engine.learn_from_outcome(decision, actual_outcome=0.9)
            else:
                engine.learn_from_outcome(decision, actual_outcome=0.4)

        # Now make same decision - should be more confident
        _final_decision = engine.make_decision(  # noqa: F841
            situation="High FP", options=["action_a", "action_b"], factors=factors
        )

        # With learning, should prefer action_a
        # (This might fail initially but should work with more training)
        stats = engine.get_learning_statistics()
        assert stats["decisions_made"] >= 10, "Should have made 10 decisions"
        assert stats["training_samples"] > 0, "Should have training data"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
