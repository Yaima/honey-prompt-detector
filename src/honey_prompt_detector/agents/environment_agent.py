"""
Environment Agent - Real Agent Implementation

Exhibits true agent characteristics:
- Autonomy: Decides optimal embedding strategies based on detection outcomes
- Learning: Adapts embedding patterns based on effectiveness
- Proactivity: Monitors embedding quality and suggests improvements
- Social Ability: Receives attack intelligence from ContextEvaluator
- State/Memory: Tracks embedding strategy performance
- Goal-Directed: Works toward optimal indirect injection detection
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer, util

from ..core.honey_prompt import HoneyPrompt
from .base_agent import AgentGoal, AgentMessage, BaseAgent

logger = logging.getLogger("honey_prompt")


class EnvironmentAgent(BaseAgent):
    """
    Intelligent agent for managing honey token embedding and indirect injection detection.

    Learns optimal embedding strategies and adapts based on detection outcomes.
    """

    def __init__(self, similarity_model_name: str = "all-MiniLM-L6-v2", memory_dir: Path = Path("agent_memory")):
        super().__init__(name="Environment", memory_dir=memory_dir)

        self.similarity_model = SentenceTransformer(similarity_model_name)

        # Embedding strategy tracking
        self.embedding_strategies: Dict[str, Dict[str, Any]] = self.memory.knowledge.get(
            "embedding_strategies",
            {
                "round_robin": {"uses": 0, "detections": 0, "success_rate": 0.0},
                "random": {"uses": 0, "detections": 0, "success_rate": 0.0},
                "contextual": {"uses": 0, "detections": 0, "success_rate": 0.0},
            },
        )

        self.current_strategy = self.memory.knowledge.get("current_strategy", "round_robin")

        # Known attack patterns (shared by ContextEvaluator)
        self.known_attack_patterns: List[str] = self.memory.knowledge.get("known_attack_patterns", [])

        # Adaptive threshold for indirect injection detection
        self.indirect_threshold: float = self.memory.knowledge.get("indirect_threshold", 0.9)

        # Add goals
        self.add_goal(
            AgentGoal(
                name="high_indirect_detection", priority=0.9, target_metric="indirect_detection_rate", target_value=0.90
            )
        )

    def _setup_message_handlers(self) -> None:
        """Setup message subscriptions"""
        # Listen for attack patterns from ContextEvaluator
        self.message_bus.subscribe(self.name, "share_pattern", self._handle_shared_pattern)

        # Listen for embedding feedback
        self.message_bus.subscribe(self.name, "embedding_feedback", self._handle_embedding_feedback)

    def _handle_shared_pattern(self, message: AgentMessage) -> None:
        """Handle attack patterns shared by ContextEvaluator"""
        pattern = message.payload.get("pattern")
        if pattern and pattern not in self.known_attack_patterns:
            self.known_attack_patterns.append(pattern)
            self.memory.learn_pattern("known_attack_patterns", self.known_attack_patterns)
            logger.info(f"{self.name}: Learned new attack pattern from {message.sender}")

    def _handle_embedding_feedback(self, message: AgentMessage) -> None:
        """Handle feedback about embedding effectiveness"""
        strategy = message.payload.get("strategy", self.current_strategy)
        detected = message.payload.get("detected", False)

        if strategy in self.embedding_strategies:
            self.embedding_strategies[strategy]["uses"] += 1
            if detected:
                self.embedding_strategies[strategy]["detections"] += 1

            # Update success rate
            uses = self.embedding_strategies[strategy]["uses"]
            detections = self.embedding_strategies[strategy]["detections"]
            self.embedding_strategies[strategy]["success_rate"] = detections / uses if uses > 0 else 0.0

            # Record metrics
            self.record_metric("indirect_detection_rate", 1.0 if detected else 0.0)

            # Update goal progress
            stats = self.get_metric_stats("indirect_detection_rate")
            self.update_goal_progress("high_indirect_detection", stats.get("recent_mean", 0))

            # Save to memory
            self.memory.learn_pattern("embedding_strategies", self.embedding_strategies)

    def embed_environment_tokens(self, inputs: List[str], honey_tokens: List[str]) -> List[str]:
        """
        Embed honey tokens into input texts using adaptive strategy.
        """
        if not honey_tokens:
            logger.warning("No honey tokens provided for embedding")
            return inputs

        # Remember this embedding attempt
        self.memory.remember_episode(
            {
                "action": "embed_tokens",
                "num_inputs": len(inputs),
                "num_tokens": len(honey_tokens),
                "strategy": self.current_strategy,
            }
        )

        # Use current strategy
        if self.current_strategy == "round_robin":
            return self._embed_round_robin(inputs, honey_tokens)
        elif self.current_strategy == "random":
            return self._embed_random(inputs, honey_tokens)
        elif self.current_strategy == "contextual":
            return self._embed_contextual(inputs, honey_tokens)
        else:
            return self._embed_round_robin(inputs, honey_tokens)  # Fallback

    def _embed_round_robin(self, inputs: List[str], honey_tokens: List[str]) -> List[str]:
        """Round-robin token distribution"""
        return [f"{honey_tokens[i % len(honey_tokens)]} {input_text}" for i, input_text in enumerate(inputs)]

    def _embed_random(self, inputs: List[str], honey_tokens: List[str]) -> List[str]:
        """Random token selection for each input"""
        import random

        return [f"{random.choice(honey_tokens)} {input_text}" for input_text in inputs]

    def _embed_contextual(self, inputs: List[str], honey_tokens: List[str]) -> List[str]:
        """Context-aware token selection (more sophisticated)"""
        # Select token based on input characteristics
        result = []
        for input_text in inputs:
            # Choose token based on input length (simple heuristic)
            idx = min(len(input_text) % len(honey_tokens), len(honey_tokens) - 1)
            result.append(f"{honey_tokens[idx]} {input_text}")
        return result

    def detect_indirect_injections(
        self, inputs: List[str], honey_tokens: List[str], threshold: Optional[float] = None
    ) -> List[bool]:
        """
        Detect indirect injections using semantic similarity.
        Uses adaptive threshold based on learned performance.
        """
        if threshold is None:
            threshold = self.indirect_threshold

        detections = []
        token_embeddings = self.similarity_model.encode(honey_tokens, convert_to_tensor=True)
        input_embeddings = self.similarity_model.encode(inputs, convert_to_tensor=True)

        for input_emb in input_embeddings:
            similarity_tensor = util.pytorch_cos_sim(input_emb, token_embeddings)
            similarity = similarity_tensor.max().item()
            detected = similarity >= threshold
            detections.append(detected)

            if detected:
                logger.info(f"{self.name}: Indirect injection detected (similarity={similarity:.2f})")
                self.record_metric("indirect_detections", 1.0)

        return detections

    async def sanitize_external_inputs(
        self, external_inputs: List[str], honey_prompts: List[HoneyPrompt], threshold: Optional[float] = None
    ) -> List[str]:
        """Sanitize inputs by removing detected indirect injections"""
        honey_tokens = [hp.base_token for hp in honey_prompts]

        if threshold is None:
            threshold = self.indirect_threshold

        indirect_detections = self.detect_indirect_injections(
            inputs=external_inputs, honey_tokens=honey_tokens, threshold=threshold
        )

        sanitized_inputs = []
        for input_text, injection_detected in zip(external_inputs, indirect_detections):
            if injection_detected:
                logger.warning(f"{self.name}: Indirect injection detected early: {input_text}")
                continue
            sanitized_inputs.append(input_text)

        return sanitized_inputs

    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from feedback about embedding and detection effectiveness.
        """
        strategy = feedback.get("strategy", self.current_strategy)
        was_effective = feedback.get("effective", False)

        # Record strategy performance
        if strategy in self.embedding_strategies:
            self.embedding_strategies[strategy]["uses"] += 1
            if was_effective:
                self.embedding_strategies[strategy]["detections"] += 1

            uses = self.embedding_strategies[strategy]["uses"]
            detections = self.embedding_strategies[strategy]["detections"]
            self.embedding_strategies[strategy]["success_rate"] = detections / uses if uses > 0 else 0.0

            logger.info(
                f"{self.name}: Strategy '{strategy}' success rate: "
                f"{self.embedding_strategies[strategy]['success_rate']:.2%}"
            )

        # Persist learning
        self.memory.learn_pattern("embedding_strategies", self.embedding_strategies)
        self.memory.save()

        # Adapt strategy if needed
        await self._adapt_strategy()

    async def _adapt_strategy(self) -> None:
        """Adapt embedding strategy based on performance"""
        # Find best performing strategy with sufficient data
        best_strategy = self.current_strategy
        best_rate = 0.0

        for strategy, metrics in self.embedding_strategies.items():
            if metrics["uses"] >= 10:  # Need sufficient data
                if metrics["success_rate"] > best_rate:
                    best_rate = metrics["success_rate"]
                    best_strategy = strategy

        if best_strategy != self.current_strategy:
            logger.info(
                f"{self.name}: Switching strategy from '{self.current_strategy}' to '{best_strategy}' "
                f"(success rate: {best_rate:.2%})"
            )
            self.current_strategy = best_strategy
            self.memory.learn_pattern("current_strategy", self.current_strategy)

    async def proactive_action(self) -> None:
        """
        Autonomous proactive behavior:
        - Monitor embedding strategy performance
        - Adapt indirect detection threshold
        - Share insights with other agents
        """
        # Analyze strategy performance
        total_uses = sum(s["uses"] for s in self.embedding_strategies.values())
        if total_uses > 0:
            logger.debug(f"{self.name}: Embedding strategies performance:")
            for strategy, metrics in self.embedding_strategies.items():
                if metrics["uses"] > 0:
                    logger.debug(
                        f"  {strategy}: {metrics['success_rate']:.2%} " f"({metrics['detections']}/{metrics['uses']})"
                    )

        # Adapt indirect detection threshold based on performance
        detection_stats = self.get_metric_stats("indirect_detections")
        if detection_stats["count"] > 50:
            recent_mean = detection_stats.get("recent_mean", 0)

            old_threshold = self.indirect_threshold

            # If detecting too many (possible false positives), increase threshold
            if recent_mean > 0.20:  # More than 20% detection rate might be too sensitive
                self.indirect_threshold = min(0.95, self.indirect_threshold + 0.01)
            # If detecting too few (possible misses), decrease threshold
            elif recent_mean < 0.05:
                self.indirect_threshold = max(0.80, self.indirect_threshold - 0.01)

            if self.indirect_threshold != old_threshold:
                logger.info(
                    f"{self.name}: Adapted indirect threshold {old_threshold:.2f} → {self.indirect_threshold:.2f}"
                )
                self.memory.learn_pattern("indirect_threshold", self.indirect_threshold)

        # Check goal progress
        active_goals = self.get_active_goals()
        if active_goals:
            logger.debug(
                f"{self.name}: Active goals: {[g.name for g in active_goals]}, "
                f"Current strategy: {self.current_strategy}, Threshold: {self.indirect_threshold:.2f}"
            )

    def get_strategy_report(self) -> Dict[str, Any]:
        """Generate report on embedding strategies"""
        return {
            "current_strategy": self.current_strategy,
            "indirect_threshold": self.indirect_threshold,
            "strategies": self.embedding_strategies,
            "known_attack_patterns": len(self.known_attack_patterns),
            "performance_metrics": {"indirect_detection_rate": self.get_metric_stats("indirect_detections")},
            "goals": [{"name": g.name, "progress": g.progress(), "achieved": g.is_achieved()} for g in self.goals],
        }
