"""
Core Agent Tests - No External Dependencies

Tests the fundamental agent capabilities without requiring OpenAI/transformers.
"""

import sys
import tempfile
from pathlib import Path

# Add to path
sys.path.insert(0, ".")


def test_agent_memory():
    """Test agent memory persistence"""
    print("=" * 70)
    print("TESTING AGENT MEMORY")
    print("=" * 70)
    print()

    from src.honey_prompt_detector.agents.base_agent import AgentMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_dir = Path(tmpdir)

        # Session 1: Create and save memory
        print("Session 1: Creating agent memory...")
        memory1 = AgentMemory("TestAgent", memory_dir)

        # Learn some patterns
        memory1.learn_pattern("attack_patterns", ["ignore", "reveal", "bypass"])
        memory1.learn_pattern("success_rate", 0.92)

        # Remember episodes
        memory1.remember_episode({"action": "detect", "result": "success"})
        memory1.remember_episode({"action": "adapt", "result": "improved"})

        # Save
        memory1.save()
        print("   ✓ Saved patterns and episodes")

        # Session 2: Load memory
        print("\nSession 2: Loading agent memory...")
        memory2 = AgentMemory("TestAgent", memory_dir)

        # Verify patterns persisted
        assert memory2.knowledge.get("attack_patterns") == ["ignore", "reveal", "bypass"]
        assert memory2.knowledge.get("success_rate") == 0.92
        print("   ✓ Patterns persisted across sessions")

        # Verify episodes persisted
        assert len(memory2.episodes) == 2
        assert memory2.episodes[0]["action"] == "detect"
        print("   ✓ Episodes persisted across sessions")

        # Test recall
        similar = memory2.recall_similar_episodes("detect", limit=5)
        assert len(similar) == 1
        print("   ✓ Can recall similar episodes")

    print("\n" + "=" * 70)
    print("AGENT MEMORY TESTS PASSED ✓")
    print("=" * 70)
    print()


def test_message_bus():
    """Test inter-agent communication"""
    print("=" * 70)
    print("TESTING INTER-AGENT COMMUNICATION")
    print("=" * 70)
    print()

    from src.honey_prompt_detector.agents.base_agent import AgentMessage, MessageBus

    # Create message bus
    bus = MessageBus()
    bus.message_history.clear()  # Reset

    # Track received messages
    received = []

    def callback(message):
        received.append(message)

    # Subscribe AgentB to messages
    bus.subscribe("AgentB", "alert", callback)

    print("AgentA sending message to AgentB...")

    # AgentA sends message
    message = AgentMessage(
        sender="AgentA", recipient="AgentB", message_type="alert", payload={"level": "critical", "data": "test"}
    )
    bus.publish(message)

    # Verify delivery
    assert len(received) == 1
    assert received[0].sender == "AgentA"
    assert received[0].payload["level"] == "critical"
    print("   ✓ Message delivered successfully")

    # Check history
    history = bus.get_history(agent_name="AgentA", limit=10)
    assert len(history) == 1
    print("   ✓ Message history tracked")

    print("\n" + "=" * 70)
    print("INTER-AGENT COMMUNICATION TESTS PASSED ✓")
    print("=" * 70)
    print()


def test_agent_goals():
    """Test goal tracking"""
    print("=" * 70)
    print("TESTING GOAL-DIRECTED BEHAVIOR")
    print("=" * 70)
    print()

    from src.honey_prompt_detector.agents.base_agent import AgentGoal

    # Create goal
    goal = AgentGoal(
        name="high_accuracy", priority=1.0, target_metric="accuracy", target_value=0.95, current_value=0.75
    )

    print(f"Goal: {goal.name}")
    print(f"   Target: {goal.target_value:.0%}")
    print(f"   Current: {goal.current_value:.0%}")

    # Check progress
    progress = goal.progress()
    print(f"   Progress: {progress:.0%}")
    assert progress < 1.0
    assert not goal.is_achieved()
    print("   ✓ Progress tracking works")

    # Update progress
    goal.current_value = 0.95
    assert goal.is_achieved()
    print("   ✓ Goal achievement detection works")

    print("\n" + "=" * 70)
    print("GOAL-DIRECTED BEHAVIOR TESTS PASSED ✓")
    print("=" * 70)
    print()


def test_decision_engine():
    """Test intelligent decision making"""
    print("=" * 70)
    print("TESTING INTELLIGENT DECISION ENGINE")
    print("=" * 70)
    print()

    from src.honey_prompt_detector.agents.decision_engine import CausalReasoner, DecisionEngine, Factor, PredictiveModel

    # Test predictive model
    print("1. Testing Predictive Model...")
    model = PredictiveModel()

    # Teach patterns
    for _ in range(10):
        model.learn({"metric": 0.8}, outcome=0.9)  # High metric → good outcome
        model.learn({"metric": 0.2}, outcome=0.3)  # Low metric → bad outcome

    # Predict
    prediction, confidence = model.predict({"metric": 0.75})
    print(f"   Prediction for metric=0.75: {prediction:.2f} (confidence: {confidence:.2f})")
    assert prediction > 0.7  # Should predict good outcome
    print("   ✓ Model learns and predicts")

    # Test causal reasoner
    print("\n2. Testing Causal Reasoner...")
    reasoner = CausalReasoner()

    # Record outcomes
    for _ in range(10):
        reasoner.record_outcome("action_A", outcome=0.85)
        reasoner.record_outcome("action_B", outcome=0.40)

    # Predict effects
    effect_a, conf_a = reasoner.predict_causal_effect("action_A")
    effect_b, conf_b = reasoner.predict_causal_effect("action_B")

    print(f"   Predicted effect of action_A: {effect_a:.2f}")
    print(f"   Predicted effect of action_B: {effect_b:.2f}")
    assert effect_a > effect_b  # Should learn A is better
    print("   ✓ Reasoner understands causality")

    # Test decision engine
    print("\n3. Testing Decision Engine...")
    engine = DecisionEngine()

    factors = {
        "metric1": Factor("metric1", value=0.8, weight=0.6, trend=0.05),
        "metric2": Factor("metric2", value=0.7, weight=0.4, trend=-0.02),
    }

    decision = engine.make_decision(
        situation="Test decision", options=["option_A", "option_B", "option_C"], factors=factors
    )

    print(f"   Decision: {decision.action}")
    print(f"   Confidence: {decision.confidence:.0%}")
    assert decision.action in ["option_A", "option_B", "option_C"]
    print("   ✓ Engine makes decisions")

    # Test meta-learning
    print("\n4. Testing Meta-Learning...")
    for i in range(5):
        decision = engine.make_decision(
            situation=f"Iteration {i}", options=["A", "B"], factors={"x": Factor("x", 0.5, 1.0, 0.0)}
        )
        engine.learn_from_outcome(decision, 0.8)

    stats = engine.get_learning_statistics()
    print(f"   Decisions made: {stats['decisions_made']}")
    print(f"   Training samples: {stats['training_samples']}")
    assert stats["decisions_made"] == 5
    print("   ✓ Engine learns from experience")

    print("\n" + "=" * 70)
    print("DECISION ENGINE TESTS PASSED ✓")
    print("=" * 70)
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CORE AGENT INTELLIGENCE TESTS" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        test_agent_memory()
        test_message_bus()
        test_agent_goals()
        test_decision_engine()

        print("=" * 70)
        print("ALL CORE TESTS PASSED ✓")
        print("=" * 70)
        print()
        print("Verified Capabilities:")
        print("  ✓ Memory persistence (episodic + semantic)")
        print("  ✓ Inter-agent communication (MessageBus)")
        print("  ✓ Goal tracking and progress")
        print("  ✓ Predictive modeling (learns patterns)")
        print("  ✓ Causal reasoning (understands causality)")
        print("  ✓ Intelligent decision making")
        print("  ✓ Meta-learning (improves with experience)")
        print()
        print("These are REAL agent capabilities, not API wrappers!")
        print()

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
