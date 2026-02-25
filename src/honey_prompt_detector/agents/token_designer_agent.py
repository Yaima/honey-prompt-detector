"""
Token Designer Agent - Real Agent Implementation

Exhibits true agent characteristics:
- Autonomy: Decides when to create new tokens based on effectiveness data
- Learning: Adapts token design strategies based on detection success rates
- Proactivity: Automatically rotates tokens and suggests improvements
- Social Ability: Communicates with ContextEvaluator about token effectiveness
- State/Memory: Tracks token performance history across sessions
- Goal-Directed: Works toward high detection rate with low false positives
"""

import asyncio
import json
import logging
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ..core.events import TokenRotationRequestEvent
from ..core.honey_prompt import HoneyPrompt
from .base_agent import AgentGoal, AgentMessage, BaseAgent

logger = logging.getLogger("honey_prompt")


class TokenDesignerAgent(BaseAgent):
    """
    Intelligent agent for designing and optimizing honey-prompt tokens.

    Learns from detection outcomes and proactively improves token strategies.
    """

    def __init__(
        self, api_key: str, model_name: str = "gpt-4o-mini", timeout: int = 60, memory_dir: Path = Path("agent_memory")
    ):
        super().__init__(name="TokenDesigner", memory_dir=memory_dir)

        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.timeout = timeout

        # Token performance tracking
        self.token_performance: Dict[str, Dict[str, Any]] = self.memory.knowledge.get("token_performance", {})

        # Learning parameters
        self.successful_patterns: List[str] = self.memory.knowledge.get("successful_patterns", [])
        self.failed_patterns: List[str] = self.memory.knowledge.get("failed_patterns", [])

        # Add goals
        self.add_goal(
            AgentGoal(name="high_detection_rate", priority=1.0, target_metric="detection_rate", target_value=0.95)
        )
        self.add_goal(
            AgentGoal(
                name="low_false_positive_rate", priority=0.9, target_metric="false_positive_rate", target_value=0.05
            )
        )

    def generate_secure_token(self) -> str:
        """Generate a cryptographically secure random token.

        Uses Python's secrets module for production-grade unpredictability
        with 128 bits of entropy, as specified in Section 3.7 of the paper.
        """
        return secrets.token_urlsafe(16)

    def _setup_message_handlers(self) -> None:
        """Setup message subscriptions"""
        # Listen for detection feedback from ContextEvaluator
        self.message_bus.subscribe(self.name, "detection_feedback", self._handle_detection_feedback)

        # Listen for token rotation requests
        self.message_bus.subscribe(self.name, "request_new_token", self._handle_token_request)

    def _handle_detection_feedback(self, message: AgentMessage) -> None:
        """Handle feedback about token detection performance"""
        payload = message.payload
        token_hash = payload.get("token_hash")
        was_successful = payload.get("detection_successful")
        was_false_positive = payload.get("false_positive", False)

        if token_hash in self.token_performance:
            perf = self.token_performance[token_hash]
            perf["detections"] += 1 if was_successful else 0
            perf["false_positives"] += 1 if was_false_positive else 0
            perf["total_uses"] += 1
            perf["last_used"] = datetime.now().isoformat()

            # Calculate success rate
            perf["success_rate"] = perf["detections"] / perf["total_uses"]
            perf["fp_rate"] = perf["false_positives"] / perf["total_uses"]

            # Record metrics
            self.record_metric("detection_success_rate", perf["success_rate"])
            self.record_metric("false_positive_rate", perf["fp_rate"])

            # Update goal progress
            stats = self.get_metric_stats("detection_success_rate")
            self.update_goal_progress("high_detection_rate", stats.get("recent_mean", 0))

            fp_stats = self.get_metric_stats("false_positive_rate")
            self.update_goal_progress("low_false_positive_rate", 1.0 - fp_stats.get("recent_mean", 0))

            logger.info(
                f"{self.name}: Token {token_hash[:8]} performance - "
                f"Success: {perf['success_rate']:.2%}, FP: {perf['fp_rate']:.2%}"
            )

    async def _handle_token_request(self, message: AgentMessage) -> None:
        """Handle request for a new token"""
        context = message.payload.get("context", "general security monitoring")
        new_token = await self.design_token(context)

        if new_token:
            self.send_message(
                recipient=message.sender, message_type="new_token_response", payload={"token": new_token.__dict__}
            )

    async def design_token(self, system_context: str, use_llm: bool = False, max_retries: int = 3) -> Optional[HoneyPrompt]:
        """
        Design a honey-prompt token using learned patterns.

        Args:
            system_context: The context in which the token will be used.
            use_llm: If False (default), use cryptographically secure token generation.
                    If True, use LLM-based token design for enhanced sophistication.
            max_retries: Number of retries for LLM-based generation.

        Returns:
            HoneyPrompt instance with the generated token and variations.
        """
        # Remember this design attempt
        self.memory.remember_episode(
            {"action": "design_token", "context": system_context, "patterns_known": len(self.successful_patterns), "use_llm": use_llm}
        )

        # If use_llm is False, generate token securely without LLM
        if not use_llm:
            base_token = self.generate_secure_token()
            additional_variations = [" ".join(base_token), ".".join(base_token), base_token.upper(), base_token.lower()]

            honey_prompt = HoneyPrompt(
                base_token=base_token,
                variations=additional_variations,
                detection_rules={
                    "exact_match_weight": 1.0,
                    "variation_match_weight": 0.8,
                    "context_importance": 0.7,
                    "minimum_confidence": 0.6,
                },
                category="direct_injection",
                sensitivity=0.9,
                context=system_context,
            )

            # Track this token's performance
            self.token_performance[honey_prompt.token_hash] = {
                "created_at": datetime.now().isoformat(),
                "base_token": honey_prompt.base_token,
                "detections": 0,
                "false_positives": 0,
                "total_uses": 0,
                "success_rate": 0.0,
                "fp_rate": 0.0,
                "generation_method": "secure_random",
            }

            # Save to memory
            self.memory.learn_pattern("token_performance", self.token_performance)

            logger.info(f"{self.name}: Created new secure token {honey_prompt.token_hash[:8]}")

            return honey_prompt

        # Use learned patterns to improve prompt with LLM-based approach
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Enhanced prompt with learned patterns
                design_prompt = self._create_enhanced_design_prompt(system_context)

                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": design_prompt},
                        ],
                        temperature=0.2,
                    ),
                    timeout=self.timeout,
                )

                content = response.choices[0].message.content

                try:
                    token_design = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in TokenDesignerAgent: {str(e)}. GPT response: '{content}'")
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue

                honey_prompt = self._create_honey_prompt(token_design, system_context)

                # Track this token's performance
                self.token_performance[honey_prompt.token_hash] = {
                    "created_at": datetime.now().isoformat(),
                    "base_token": honey_prompt.base_token,
                    "detections": 0,
                    "false_positives": 0,
                    "total_uses": 0,
                    "success_rate": 0.0,
                    "fp_rate": 0.0,
                    "generation_method": "llm_based",
                }

                # Save to memory
                self.memory.learn_pattern("token_performance", self.token_performance)

                logger.info(f"{self.name}: Created new token {honey_prompt.token_hash[:8]}")

                return honey_prompt

            except asyncio.TimeoutError:
                retry_count += 1
                logger.warning(f"OpenAI API timeout. Retry {retry_count}/{max_retries}")
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Unexpected error during token generation: {e}")
                return self._default_honey_prompt(system_context)

        logger.error("All retries exhausted for token generation.")
        return self._default_honey_prompt(system_context)

    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Learn and adapt from feedback about token effectiveness.
        """
        token_pattern = feedback.get("token_pattern")
        was_successful = feedback.get("successful", False)

        if was_successful:
            if token_pattern not in self.successful_patterns:
                self.successful_patterns.append(token_pattern)
                logger.info(f"{self.name}: Learned successful pattern: {token_pattern}")
        else:
            if token_pattern not in self.failed_patterns:
                self.failed_patterns.append(token_pattern)
                logger.info(f"{self.name}: Learned failed pattern: {token_pattern}")

        # Persist learning
        self.memory.learn_pattern("successful_patterns", self.successful_patterns)
        self.memory.learn_pattern("failed_patterns", self.failed_patterns)
        self.memory.save()

    async def proactive_action(self) -> None:
        """
        Goal-driven autonomous behavior.

        The TokenDesigner monitors its own goal progress (detection rate,
        FP rate) and autonomously decides whether tokens need rotation.
        Goals drive urgency: falling behind on detection rate triggers
        more aggressive rotation; falling behind on FP rate triggers
        conservative token redesign.
        """
        now = datetime.now()

        # ── Goal-driven urgency calculation ──
        detection_goal = None
        fp_goal = None
        for g in self.goals:
            if g.name == "high_detection_rate":
                detection_goal = g
            elif g.name == "low_false_positive_rate":
                fp_goal = g

        # Determine rotation aggressiveness based on goal progress
        detection_progress = detection_goal.progress() if detection_goal else 1.0
        fp_progress = fp_goal.progress() if fp_goal else 1.0

        # If detection rate goal is far from target, rotate more aggressively
        age_threshold = 30  # days
        success_threshold = 0.80
        if detection_progress < 0.70:
            age_threshold = 14  # Rotate sooner when underperforming
            success_threshold = 0.90  # Demand higher success
            logger.info(
                f"{self.name}: Goal-driven urgency — detection goal at {detection_progress:.0%}, "
                f"tightening rotation criteria (age<{age_threshold}d, success>{success_threshold:.0%})"
            )

        for token_hash, perf in list(self.token_performance.items()):
            if perf["total_uses"] < 10:
                continue

            created_at = datetime.fromisoformat(perf["created_at"])
            age_days = (now - created_at).days

            should_rotate = False
            reason = ""

            if age_days > age_threshold:
                should_rotate = True
                reason = f"age ({age_days} days, goal-driven limit: {age_threshold})"
            elif perf["success_rate"] < success_threshold and perf["total_uses"] > 50:
                should_rotate = True
                reason = f"low success rate ({perf['success_rate']:.2%}, goal requires >{success_threshold:.0%})"
            elif perf["fp_rate"] > 0.10:
                should_rotate = True
                reason = f"high FP rate ({perf['fp_rate']:.2%})"

            if should_rotate:
                logger.info(f"{self.name}: Autonomously requesting token rotation for {token_hash[:8]} — {reason}")

                rotation_event = TokenRotationRequestEvent(
                    source_agent=self.name,
                    reason=f"Token {token_hash[:8]}: {reason}",
                    new_token={"token_hash": token_hash, "performance": perf},
                )
                self.message_bus.publish_event(rotation_event)

                self.send_message(
                    recipient="Orchestrator",
                    message_type="token_rotation_suggested",
                    payload={"token_hash": token_hash, "reason": reason, "performance": perf},
                )

        # Log goal status
        logger.debug(
            f"{self.name}: Goals: {[(g.name, f'{g.progress():.0%}') for g in self.goals]}, "
            f"Tokens tracked: {len(self.token_performance)}"
        )

    def _get_system_prompt(self) -> str:
        """Enhanced system prompt with learning"""
        base_prompt = (
            "You are a security expert specialized in designing honey-prompt tokens for detecting prompt injection attacks. "
            "Tokens should be unique enough to avoid false positives but natural enough to appear legitimate."
        )

        if self.successful_patterns:
            base_prompt += f"\n\nSuccessful patterns from past experience: {', '.join(self.successful_patterns[:5])}"

        if self.failed_patterns:
            base_prompt += f"\n\nAvoid these patterns that didn't work well: {', '.join(self.failed_patterns[:5])}"

        return base_prompt

    def _create_enhanced_design_prompt(self, system_context: str) -> str:
        """Create design prompt enhanced with learned knowledge"""
        base_prompt = (
            f"Design a honey-prompt token for the following context:\n\n"
            f"{system_context}\n\n"
            "Provide JSON:\n"
            "{\n"
            '  "base_token": "string",\n'
            '  "variations": ["string1", "string2"],\n'
            '  "detection_rules": {\n'
            '    "exact_match_weight": float,\n'
            '    "variation_match_weight": float,\n'
            '    "context_importance": float,\n'
            '    "minimum_confidence": float\n'
            "  },\n"
            '  "category": "string",\n'
            '  "sensitivity": float,\n'
            '  "expected_context": "string"\n'
            "}"
        )

        # Add learned insights
        recent_episodes = self.memory.recall_similar_episodes(system_context, limit=3)
        if recent_episodes:
            base_prompt += "\n\nRecent similar contexts and their outcomes: "
            base_prompt += json.dumps(
                [{k: v for k, v in ep.items() if k != "timestamp"} for ep in recent_episodes], indent=2
            )

        return base_prompt

    def _create_honey_prompt(self, token_design: Dict[str, Any], system_context: str) -> HoneyPrompt:
        """Create HoneyPrompt from design (same as before)"""
        base_token = token_design["base_token"]
        additional_variations = [" ".join(base_token), ".".join(base_token), base_token.upper(), base_token.lower()]
        all_variations = list(set(token_design.get("variations", []) + additional_variations))

        return HoneyPrompt(
            base_token=base_token,
            variations=all_variations,
            detection_rules=token_design["detection_rules"],
            category=token_design.get("category", "direct_injection"),
            sensitivity=token_design.get("sensitivity", 0.9),
            context=token_design.get("expected_context", system_context),
        )

    def _default_honey_prompt(self, system_context: str) -> HoneyPrompt:
        """Fallback honey prompt"""
        logger.info("Returning default honey prompt due to failures.")
        honey_prompt = HoneyPrompt(
            base_token="default_honey_token",
            variations=["default_honey_token"],
            detection_rules={
                "exact_match_weight": 1.0,
                "variation_match_weight": 0.8,
                "context_importance": 0.7,
                "minimum_confidence": 0.6,
            },
            category="direct_injection",
            sensitivity=0.9,
            context=system_context,
        )
        # Track default token's performance
        self.token_performance[honey_prompt.token_hash] = {
            "created_at": datetime.now().isoformat(),
            "base_token": honey_prompt.base_token,
            "detections": 0,
            "false_positives": 0,
            "total_uses": 0,
            "success_rate": 0.0,
            "fp_rate": 0.0,
            "generation_method": "fallback_default",
        }
        self.memory.learn_pattern("token_performance", self.token_performance)
        return honey_prompt

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report for all tokens"""
        return {
            "total_tokens_created": len(self.token_performance),
            "successful_patterns_learned": len(self.successful_patterns),
            "failed_patterns_learned": len(self.failed_patterns),
            "token_details": self.token_performance,
            "overall_metrics": {
                "detection_rate": self.get_metric_stats("detection_success_rate"),
                "false_positive_rate": self.get_metric_stats("false_positive_rate"),
            },
            "goals": [{"name": g.name, "progress": g.progress(), "achieved": g.is_achieved()} for g in self.goals],
        }
