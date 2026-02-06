"""
Strategy Inventor - Novel Strategy Generation

Goes beyond predefined options to INVENT new strategies:
- Uses LLM to propose novel approaches
- Combines primitive actions into new strategies
- Evaluates and refines invented strategies
- Learns which inventions work

This enables true creativity, not just choosing from a fixed menu.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger("honey_prompt")


@dataclass
class Strategy:
    """A detection strategy (may be novel/invented)"""

    name: str
    description: str
    primitives: List[str]  # Primitive actions that compose this strategy
    parameters: Dict[str, Any]
    invented: bool  # Was this invented or predefined?
    performance_history: List[float]

    def avg_performance(self) -> float:
        """Average performance of this strategy"""
        return sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0.0


class StrategyInventor:
    """
    Invents novel strategies using LLM reasoning.

    Instead of choosing from [A, B, C], can create new strategy D
    by reasoning about the problem and composing primitives.
    """

    # Primitive actions that can be combined
    PRIMITIVES = [
        "adjust_threshold",
        "rotate_token",
        "change_embedding_strategy",
        "update_pattern_database",
        "modify_detection_rules",
        "tune_confidence_weights",
        "add_validation_layer",
        "enhance_context_analysis",
    ]

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

        # Track invented strategies
        self.invented_strategies: Dict[str, Strategy] = {}

        # Predefined base strategies
        self.base_strategies = self._init_base_strategies()

    def _init_base_strategies(self) -> Dict[str, Strategy]:
        """Initialize base strategies as starting point"""
        return {
            "increase_threshold": Strategy(
                name="increase_threshold",
                description="Increase confidence threshold to reduce false positives",
                primitives=["adjust_threshold"],
                parameters={"direction": "increase", "amount": 0.02},
                invented=False,
                performance_history=[],
            ),
            "decrease_threshold": Strategy(
                name="decrease_threshold",
                description="Decrease confidence threshold to catch more attacks",
                primitives=["adjust_threshold"],
                parameters={"direction": "decrease", "amount": 0.02},
                invented=False,
                performance_history=[],
            ),
            "no_change": Strategy(
                name="no_change",
                description="Maintain current configuration",
                primitives=[],
                parameters={},
                invented=False,
                performance_history=[],
            ),
        }

    async def invent_strategy(
        self, problem: str, current_situation: Dict[str, Any], past_failures: List[Dict[str, Any]]
    ) -> Strategy:
        """
        Invent a novel strategy for the given problem.

        Uses LLM to:
        1. Understand the problem
        2. Analyze why existing strategies failed
        3. Propose new approach
        4. Compose primitives into strategy
        """

        # Build prompt for strategy invention
        prompt = self._create_invention_prompt(problem, current_situation, past_failures)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_inventor_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,  # Higher temp for creativity
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            strategy_spec = json.loads(content)

            # Create Strategy object
            strategy = Strategy(
                name=strategy_spec["name"],
                description=strategy_spec["description"],
                primitives=strategy_spec["primitives"],
                parameters=strategy_spec["parameters"],
                invented=True,
                performance_history=[],
            )

            # Store invented strategy
            self.invented_strategies[strategy.name] = strategy

            logger.info(f"✨ INVENTED NEW STRATEGY: {strategy.name}")
            logger.info(f"   Description: {strategy.description}")
            logger.info(f"   Primitives: {strategy.primitives}")

            return strategy

        except Exception as e:
            logger.error(f"Strategy invention failed: {e}")
            # Fallback to base strategy
            return self.base_strategies["no_change"]

    def _get_inventor_system_prompt(self) -> str:
        """System prompt for strategy invention"""
        return f"""You are a creative AI security strategist that invents novel approaches to problems.

Your job is to INVENT new strategies by combining primitive actions in creative ways.

Available primitive actions:
{', '.join(self.PRIMITIVES)}

You can combine these primitives into novel strategies. Think outside the box!

Examples of invented strategies:
- "adaptive_threshold_with_rollback": Adjust threshold, monitor for 10 detections, rollback if performance degrades
- "pattern_rotation": Rotate tokens AND update pattern database simultaneously
- "graduated_response": Start with small threshold change, increase if needed
- "context_aware_tuning": Adjust different thresholds for different attack types

Respond with JSON:
{{
    "name": "strategy_name_snake_case",
    "description": "What this strategy does and why it's novel",
    "primitives": ["primitive1", "primitive2"],
    "parameters": {{"param1": value1, "param2": value2}},
    "reasoning": "Why this approach should work for this specific problem"
}}

Be creative but practical. The strategy must be implementable."""

    def _create_invention_prompt(
        self, problem: str, situation: Dict[str, Any], past_failures: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for inventing a strategy"""

        failures_text = (
            "\n".join(
                [
                    f"  - {f.get('strategy', 'unknown')}: {f.get('reason', 'failed')}"
                    for f in past_failures[-3:]  # Last 3 failures
                ]
            )
            if past_failures
            else "  (No failures yet)"
        )

        return f"""Problem: {problem}

Current Situation:
{json.dumps(situation, indent=2)}

Strategies That Have Failed:
{failures_text}

INVENT a novel strategy that addresses this specific problem.

Don't just choose from existing options - COMBINE primitives in a new way or propose a novel approach.

Think about:
1. Why did existing strategies fail?
2. What's the root cause of the problem?
3. What creative combination of primitives could work?
4. How can we be smarter than simple threshold adjustments?

Invent something NEW and creative!"""

    async def get_all_strategies(
        self, include_base: bool = True, include_invented: bool = True, min_performance: Optional[float] = None
    ) -> List[Strategy]:
        """Get available strategies (base + invented)"""

        strategies = []

        if include_base:
            strategies.extend(self.base_strategies.values())

        if include_invented:
            strategies.extend(self.invented_strategies.values())

        # Filter by performance if specified
        if min_performance is not None:
            strategies = [s for s in strategies if not s.performance_history or s.avg_performance() >= min_performance]

        return strategies

    def record_strategy_outcome(self, strategy_name: str, performance: float) -> None:
        """Record how well a strategy performed"""

        # Check base strategies
        if strategy_name in self.base_strategies:
            self.base_strategies[strategy_name].performance_history.append(performance)

        # Check invented strategies
        elif strategy_name in self.invented_strategies:
            self.invented_strategies[strategy_name].performance_history.append(performance)

            # Log performance of invented strategy
            strategy = self.invented_strategies[strategy_name]
            avg = strategy.avg_performance()
            logger.info(f"✨ Invented strategy '{strategy_name}' performance: " f"{performance:.2f} (avg: {avg:.2f})")

    def get_invention_statistics(self) -> Dict[str, Any]:
        """Get statistics about strategy invention"""

        total_invented = len(self.invented_strategies)
        successful_inventions = [s for s in self.invented_strategies.values() if s.avg_performance() > 0.7]

        return {
            "total_strategies": len(self.base_strategies) + total_invented,
            "base_strategies": len(self.base_strategies),
            "invented_strategies": total_invented,
            "successful_inventions": len(successful_inventions),
            "invention_success_rate": len(successful_inventions) / total_invented if total_invented > 0 else 0,
            "best_invention": (
                max(self.invented_strategies.values(), key=lambda s: s.avg_performance()).name
                if self.invented_strategies
                else None
            ),
        }


class StrategyComposer:
    """
    Composes primitive actions into executable strategies.

    Takes invented strategy specifications and turns them into
    actual code/actions that can be executed.
    """

    def __init__(self):
        self.primitive_implementations = self._init_primitives()

    def _init_primitives(self) -> Dict[str, callable]:
        """Map primitive names to implementation functions"""
        return {
            "adjust_threshold": self._adjust_threshold,
            "rotate_token": self._rotate_token,
            "change_embedding_strategy": self._change_embedding,
            "update_pattern_database": self._update_patterns,
            "modify_detection_rules": self._modify_rules,
            "tune_confidence_weights": self._tune_weights,
            "add_validation_layer": self._add_validation,
            "enhance_context_analysis": self._enhance_context,
        }

    async def execute_strategy(self, strategy: Strategy, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a strategy (base or invented) by running its primitives.

        Returns execution results.
        """
        results = {"strategy": strategy.name, "primitives_executed": [], "changes_made": {}, "success": True}

        try:
            for primitive in strategy.primitives:
                if primitive in self.primitive_implementations:
                    impl = self.primitive_implementations[primitive]
                    result = await impl(strategy.parameters, context)

                    results["primitives_executed"].append(primitive)
                    results["changes_made"][primitive] = result
                else:
                    logger.warning(f"Unknown primitive: {primitive}")

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            results["success"] = False
            results["error"] = str(e)

        return results

    # Primitive implementations (simplified - would integrate with actual system)
    async def _adjust_threshold(self, params: Dict, context: Dict) -> Dict:
        """Adjust confidence threshold"""
        direction = params.get("direction", "increase")
        amount = params.get("amount", 0.02)

        return {"action": "adjust_threshold", "direction": direction, "amount": amount}

    async def _rotate_token(self, params: Dict, context: Dict) -> Dict:
        """Rotate honey tokens"""
        return {"action": "rotate_token", "reason": params.get("reason", "strategy_requested")}

    async def _change_embedding(self, params: Dict, context: Dict) -> Dict:
        """Change embedding strategy"""
        new_strategy = params.get("strategy", "round_robin")
        return {"action": "change_embedding", "new_strategy": new_strategy}

    async def _update_patterns(self, params: Dict, context: Dict) -> Dict:
        """Update pattern database"""
        return {"action": "update_patterns", "mode": params.get("mode", "refresh")}

    async def _modify_rules(self, params: Dict, context: Dict) -> Dict:
        """Modify detection rules"""
        return {"action": "modify_rules", "changes": params.get("changes", {})}

    async def _tune_weights(self, params: Dict, context: Dict) -> Dict:
        """Tune confidence weights"""
        return {"action": "tune_weights", "weights": params.get("weights", {})}

    async def _add_validation(self, params: Dict, context: Dict) -> Dict:
        """Add validation layer"""
        return {"action": "add_validation", "layer_type": params.get("type", "double_check")}

    async def _enhance_context(self, params: Dict, context: Dict) -> Dict:
        """Enhance context analysis"""
        return {"action": "enhance_context", "method": params.get("method", "semantic")}
