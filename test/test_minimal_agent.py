#!/usr/bin/env python3
"""
Minimal test that verifies agents WITHOUT requiring OpenAI API or heavy dependencies.

This tests the ACTUAL integration, not the AI features.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Test without importing the full system - just test BaseAgent directly
sys.path.insert(0, str(Path(__file__).parent))

from src.honey_prompt_detector.agents.base_agent import AgentGoal, AgentMemory, BaseAgent


class MockAgent(BaseAgent):
    """Simple test agent without external dependencies"""

    def __init__(self, name: str):
        super().__init__(name=name, memory_dir=Path(tempfile.mkdtemp()))
        self.proactive_call_count = 0
        self.messages_received = []

    def _setup_message_handlers(self) -> None:
        """Setup message subscriptions"""
        self.message_bus.subscribe(self.name, "test_message", self._handle_test_message)

    def _handle_test_message(self, message):
        """Handle incoming test messages"""
        self.messages_received.append(message)
        print(f"  {self.name} received message from {message.sender}")

    async def learn_from_feedback(self, feedback: dict) -> None:
        """Learn from feedback (required abstract method)"""
        pass  # Not needed for basic test

    async def proactive_action(self):
        """Count how many times proactive loop runs"""
        self.proactive_call_count += 1
        if self.proactive_call_count <= 3:
            print(f"  {self.name} proactive action #{self.proactive_call_count}")


async def test_real_agent_behavior():
    """Test that agents actually behave as autonomous entities"""

    print("\n" + "=" * 70)
    print("MINIMAL AGENT INTEGRATION TEST")
    print("Testing ACTUAL agent behavior without external dependencies")
    print("=" * 70)

    # Create two agents
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")

    try:
        # TEST 1: Can agents start?
        print("\n[TEST 1] Starting agents...")
        await agent1.start()
        await agent2.start()

        assert agent1.running, "Agent1 not running"
        assert agent2.running, "Agent2 not running"
        print("✅ Both agents started successfully")

        # TEST 2: Are proactive tasks actually running?
        print("\n[TEST 2] Verifying proactive loops are active...")
        assert agent1._proactive_task is not None, "Agent1 has no proactive task"
        assert not agent1._proactive_task.done(), "Agent1 proactive task not running"
        assert agent2._proactive_task is not None, "Agent2 has no proactive task"
        assert not agent2._proactive_task.done(), "Agent2 proactive task not running"
        print("✅ Proactive tasks are running in background")

        # TEST 3: Do proactive loops actually execute?
        print("\n[TEST 3] Waiting for proactive actions (5 seconds)...")
        await asyncio.sleep(5)

        assert agent1.proactive_call_count > 0, f"Agent1 proactive never called (count: {agent1.proactive_call_count})"
        assert agent2.proactive_call_count > 0, f"Agent2 proactive never called (count: {agent2.proactive_call_count})"
        print(f"✅ Agent1 executed {agent1.proactive_call_count} proactive actions")
        print(f"✅ Agent2 executed {agent2.proactive_call_count} proactive actions")

        # TEST 4: Can agents communicate?
        print("\n[TEST 4] Testing inter-agent communication...")
        agent1.send_message(recipient="Agent2", message_type="test_message", payload={"data": "Hello from Agent1"})

        # Give time for message processing
        await asyncio.sleep(0.1)

        assert len(agent2.messages_received) > 0, "Agent2 received no messages"
        assert agent2.messages_received[0].sender == "Agent1", "Wrong sender"
        print(f"✅ Agent2 received {len(agent2.messages_received)} message(s) from Agent1")

        # TEST 5: Can agents track goals?
        print("\n[TEST 5] Testing goal tracking...")
        agent1.add_goal(AgentGoal(name="test_goal", priority=1.0, target_metric="test_metric", target_value=100.0))

        agent1.update_goal_progress("test_goal", 75.0)

        # Find the goal
        goal = None
        for g in agent1.goals:
            if g.name == "test_goal":
                goal = g
                break

        assert goal is not None, "Goal not found"
        assert goal.current_value == 75.0, f"Goal value wrong: {goal.current_value}"
        assert abs(goal.progress() - 0.75) < 0.01, f"Goal progress wrong: {goal.progress()}"
        print(f"✅ Goal tracking works: {goal.progress()*100:.0f}% progress toward target")

        # TEST 6: Does memory work?
        print("\n[TEST 6] Testing memory system...")
        agent1.memory.remember_episode({"action": "test_action", "result": "success"})

        agent1.memory.learn_pattern("test_pattern", {"value": 42})

        assert len(agent1.memory.episodes) > 0, "No episodes stored"
        assert "test_pattern" in agent1.memory.knowledge, "Pattern not in knowledge"
        assert agent1.memory.knowledge["test_pattern"]["value"] == 42, "Wrong pattern value"
        print(
            f"✅ Memory works: {len(agent1.memory.episodes)} episodes, {len(agent1.memory.knowledge)} knowledge items"
        )

        # TEST 7: Does memory persist?
        print("\n[TEST 7] Testing memory persistence...")

        # Store the memory file path before stopping
        memory_file = agent1.memory.memory_file
        memory_dir = memory_file.parent

        await agent1.stop()

        # Check if memory file was created
        assert memory_file.exists(), "Memory file not created"
        print(f"✅ Memory persisted to: {memory_file}")

        # Restart agent and check memory loaded
        agent1_new = MockAgent("Agent1")
        agent1_new.memory = AgentMemory("Agent1", memory_dir)

        assert "test_pattern" in agent1_new.memory.knowledge, "Memory not loaded"
        print("✅ Memory loaded from disk successfully")

        # TEST 8: Graceful shutdown
        print("\n[TEST 8] Testing graceful shutdown...")
        await agent2.stop()

        assert not agent2.running, "Agent2 still running"
        assert agent2._proactive_task.done(), "Agent2 task not cancelled"
        print("✅ Agent stopped gracefully")

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
        print("\nVerified behaviors:")
        print("  ✅ Agents start and stop properly")
        print("  ✅ Proactive loops run autonomously in background")
        print("  ✅ Inter-agent communication works")
        print("  ✅ Goal tracking functional")
        print("  ✅ Memory persists across sessions")
        print("\n🎉 These are REAL autonomous agents!\n")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if agent1.running:
            await agent1.stop()
        if agent2.running:
            await agent2.stop()


if __name__ == "__main__":
    result = asyncio.run(test_real_agent_behavior())
    sys.exit(0 if result else 1)
