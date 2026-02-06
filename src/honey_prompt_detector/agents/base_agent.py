"""
Base Agent Class with Real Agent Capabilities

Provides core agent functionality:
- Autonomy: Independent decision-making
- Learning: Adaptation from experience
- Proactivity: Initiates actions based on goals
- Social Ability: Communication with other agents
- State/Memory: Maintains context across interactions
- Goal-Directed: Works toward objectives
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("honey_prompt")


@dataclass
class AgentMessage:
    """Message passed between agents"""

    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class AgentGoal:
    """Represents an agent's goal"""

    name: str
    priority: float  # 0.0 to 1.0
    target_metric: str
    target_value: float
    current_value: float = 0.0

    def is_achieved(self) -> bool:
        return self.current_value >= self.target_value

    def progress(self) -> float:
        return min(1.0, self.current_value / self.target_value) if self.target_value > 0 else 0.0


class AgentMemory:
    """Persistent memory system for agents"""

    def __init__(self, agent_name: str, memory_dir: Path):
        self.agent_name = agent_name
        self.memory_file = memory_dir / f"{agent_name}_memory.json"
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

        # Short-term memory (in-memory, current session)
        self.short_term: Dict[str, Any] = {}

        # Long-term memory (persistent across sessions)
        self.long_term: Dict[str, Any] = self._load_long_term()

        # Episodic memory (experiences)
        self.episodes: List[Dict[str, Any]] = self.long_term.get("episodes", [])

        # Semantic memory (learned facts/patterns)
        self.knowledge: Dict[str, Any] = self.long_term.get("knowledge", {})

    def _load_long_term(self) -> Dict[str, Any]:
        """Load persistent memory from disk"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load memory for {self.agent_name}: {e}")
        return {"episodes": [], "knowledge": {}}

    def save(self) -> None:
        """Persist memory to disk"""
        try:
            self.long_term["episodes"] = self.episodes[-1000:]  # Keep last 1000 episodes
            self.long_term["knowledge"] = self.knowledge

            with open(self.memory_file, "w") as f:
                json.dump(self.long_term, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory for {self.agent_name}: {e}")

    def remember_episode(self, episode: Dict[str, Any]) -> None:
        """Store an experience in episodic memory"""
        episode["timestamp"] = datetime.now().isoformat()
        self.episodes.append(episode)

    def learn_pattern(self, pattern_name: str, pattern_data: Any) -> None:
        """Store learned knowledge in semantic memory"""
        self.knowledge[pattern_name] = pattern_data

    def recall_similar_episodes(self, context: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar past experiences"""
        # Simple keyword matching for now - could use embeddings
        relevant = [ep for ep in self.episodes if context.lower() in str(ep).lower()]
        return relevant[-limit:]  # Most recent matches


class MessageBus:
    """Central message bus for inter-agent communication"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_history: List[AgentMessage] = []

    def subscribe(self, agent_name: str, message_type: str, callback: Callable) -> None:
        """Subscribe to messages of a specific type"""
        key = f"{agent_name}:{message_type}"
        self.subscribers[key].append(callback)

    def publish(self, message: AgentMessage) -> None:
        """Publish a message to interested agents"""
        self.message_history.append(message)

        # Deliver to specific recipient
        key = f"{message.recipient}:{message.message_type}"
        for callback in self.subscribers.get(key, []):
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error delivering message to {message.recipient}: {e}")

        # Also deliver to broadcast listeners
        broadcast_key = f"*:{message.message_type}"
        for callback in self.subscribers.get(broadcast_key, []):
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

    def get_history(self, agent_name: Optional[str] = None, limit: int = 100) -> List[AgentMessage]:
        """Get message history for an agent"""
        if agent_name:
            relevant = [m for m in self.message_history if m.sender == agent_name or m.recipient == agent_name]
            return relevant[-limit:]
        return self.message_history[-limit:]


class BaseAgent(ABC):
    """
    Base class for all agents with real agent capabilities.

    Real agents exhibit:
    1. Autonomy - Make independent decisions
    2. Learning - Adapt from experience
    3. Proactivity - Initiate actions based on goals
    4. Social Ability - Communicate with other agents
    5. State/Memory - Maintain context
    6. Goal-Directed - Work toward objectives
    """

    def __init__(self, name: str, memory_dir: Path):
        self.name = name
        self.memory = AgentMemory(name, memory_dir)
        self.message_bus = MessageBus()
        self.goals: List[AgentGoal] = []
        self.running = False
        self._proactive_task: Optional[asyncio.Task] = None

        # Performance metrics for learning
        self.metrics: Dict[str, List[float]] = defaultdict(list)

        # Subscribe to messages
        self._setup_message_handlers()

    @abstractmethod
    def _setup_message_handlers(self) -> None:
        """Setup message subscriptions (implemented by subclasses)"""

    @abstractmethod
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn and adapt from feedback (implemented by subclasses)"""

    @abstractmethod
    async def proactive_action(self) -> None:
        """Autonomous proactive behavior (implemented by subclasses)"""

    def add_goal(self, goal: AgentGoal) -> None:
        """Add a goal for the agent to work toward"""
        self.goals.append(goal)
        logger.info(f"{self.name}: New goal added - {goal.name} (priority: {goal.priority})")

    def update_goal_progress(self, goal_name: str, value: float) -> None:
        """Update progress toward a goal"""
        for goal in self.goals:
            if goal.name == goal_name:
                goal.current_value = value
                if goal.is_achieved():
                    logger.info(f"{self.name}: Goal achieved - {goal_name}!")

    def get_active_goals(self) -> List[AgentGoal]:
        """Get goals not yet achieved, sorted by priority"""
        active = [g for g in self.goals if not g.is_achieved()]
        return sorted(active, key=lambda g: g.priority, reverse=True)

    def send_message(self, recipient: str, message_type: str, payload: Dict[str, Any]) -> None:
        """Send a message to another agent"""
        message = AgentMessage(sender=self.name, recipient=recipient, message_type=message_type, payload=payload)
        self.message_bus.publish(message)
        logger.debug(f"{self.name} -> {recipient}: {message_type}")

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric for learning"""
        self.metrics[metric_name].append(value)

        # Keep only recent metrics
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]

    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = self.metrics.get(metric_name, [])
        if not values:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "recent_mean": sum(values[-10:]) / min(10, len(values)),
        }

    async def start(self) -> None:
        """Start the agent's proactive behavior"""
        if self.running:
            return

        self.running = True

        # Setup message subscriptions if the subclass defined them
        if hasattr(self, "setup_subscriptions") and callable(self.setup_subscriptions):
            self.setup_subscriptions()
            logger.info(f"{self.name}: Message subscriptions configured")

        logger.info(f"{self.name}: Agent started")

        # Start proactive behavior in background
        self._proactive_task = asyncio.create_task(self._proactive_loop())

    async def stop(self) -> None:
        """Stop the agent"""
        self.running = False

        if self._proactive_task and not self._proactive_task.done():
            self._proactive_task.cancel()
            try:
                await self._proactive_task
            except asyncio.CancelledError:
                pass

        # Save memory
        self.memory.save()
        logger.info(f"{self.name}: Agent stopped")

    async def _proactive_loop(self) -> None:
        """Background loop for proactive behavior"""
        while self.running:
            try:
                await self.proactive_action()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self.name}: Error in proactive loop: {e}")
                await asyncio.sleep(60)  # Back off on error

    def __del__(self):
        """Ensure memory is saved on cleanup"""
        try:
            self.memory.save()
        except Exception:
            # Suppress exceptions during __del__ to avoid issues during interpreter shutdown
            pass
