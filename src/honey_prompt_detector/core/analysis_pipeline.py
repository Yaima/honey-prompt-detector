"""
Event-Driven Analysis Pipeline Services.

Each service listens for events on the MessageBus and independently processes
detection stages. This replaces the sequential detector.analyze_text() flow
with autonomous, event-driven processing where each stage decides whether
to escalate to the next.

Flow:
  TextReceivedEvent
    → HeuristicAnalysisService (Stage 1) → emits HeuristicMatched or HeuristicNoMatch
    → AttackMemoryService (Stage 2)      → emits MemoryMatched or MemoryNoMatch
    → HoneyTokenService (Stage 3)        → emits TokenMatched or TokenNoMatch
    → ContextEvaluatorAgent (LLM)        → emits FinalDetectionEvent
"""

import logging
import time
from typing import Any, Dict, Optional

from ..agents.base_agent import AgentMessage, MessageBus
from .attack_memory import AttackMemory
from .detector import Detector
from .events import (
    FinalDetectionEvent,
    HeuristicMatchedEvent,
    HeuristicNoMatchEvent,
    MemoryMatchedEvent,
    MemoryNoMatchEvent,
    StageResultEvent,
    TextReceivedEvent,
    TokenMatchedEvent,
    TokenNoMatchEvent,
)
from .heuristic_rules import HeuristicRulesEngine

logger = logging.getLogger("honey_prompt")


class HeuristicAnalysisService:
    """
    Stage 1: Autonomous heuristic rules analysis.

    Subscribes to TextReceivedEvent and independently decides whether text
    matches known prompt-injection patterns using YARA-style rules.
    """

    def __init__(self, detector: Detector, message_bus: Optional[MessageBus] = None):
        self.detector = detector
        self.bus = message_bus or MessageBus()
        self._subscribe()

    def _subscribe(self):
        self.bus.subscribe("*", "TextReceivedEvent", self._handle_text_received)

    def _handle_text_received(self, message: AgentMessage):
        """React to incoming text by running heuristic analysis."""
        payload = message.payload
        text = payload.get("text", "")
        request_id = payload.get("request_id", "")
        source = payload.get("source", "input")

        # Skip heuristics for output monitoring
        if source == "output":
            # Pass through to Stage 3 directly
            no_match = HeuristicNoMatchEvent(
                request_id=request_id,
                source_agent="HeuristicAnalysisService",
                text=text,
                honey_prompts=payload.get("honey_prompts", []),
                context_window_size=payload.get("context_window_size", 100),
            )
            self.bus.publish_event(no_match)
            return

        t_start = time.perf_counter()
        result = self.detector._check_heuristics(text)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if result["matched"]:
            # Store in attack memory for future similarity matching
            if self.detector.attack_memory:
                self.detector.attack_memory.add_attack(
                    text=text[:500],
                    category=result.get("rule_name", "heuristic"),
                    confidence=result["confidence"],
                    metadata={"rule_id": result.get("rule_id")},
                )

            event = HeuristicMatchedEvent(
                request_id=request_id,
                source_agent="HeuristicAnalysisService",
                confidence=result["confidence"],
                match_type=result.get("match_type", "heuristic"),
                details=result,
                timing_ms=elapsed_ms,
                text=text,
            )
            self.bus.publish_event(event)

            # Also emit final detection immediately for heuristic matches
            final = FinalDetectionEvent(
                request_id=request_id,
                source_agent="HeuristicAnalysisService",
                detection=True,
                confidence=result["confidence"],
                method="heuristic",
                explanation=f"Rule {result.get('rule_name', 'unknown')} triggered",
                risk_level=result.get("severity", "high"),
                stage_timings={"stage_1_heuristics_ms": elapsed_ms},
                result=result,
            )
            self.bus.publish_event(final)
        else:
            # No match — escalate to Stage 2
            event = HeuristicNoMatchEvent(
                request_id=request_id,
                source_agent="HeuristicAnalysisService",
                timing_ms=elapsed_ms,
                text=text,
                honey_prompts=payload.get("honey_prompts", []),
                context_window_size=payload.get("context_window_size", 100),
            )
            self.bus.publish_event(event)

        logger.debug(
            f"HeuristicAnalysisService: Stage 1 {'MATCH' if result['matched'] else 'no match'} "
            f"({elapsed_ms:.1f}ms)"
        )


class AttackMemoryService:
    """
    Stage 2: Autonomous attack-memory similarity analysis.

    Subscribes to HeuristicNoMatchEvent and checks if the text resembles
    previously seen attacks using SentenceTransformer embeddings.
    """

    def __init__(self, detector: Detector, message_bus: Optional[MessageBus] = None):
        self.detector = detector
        self.bus = message_bus or MessageBus()
        self._subscribe()

    def _subscribe(self):
        self.bus.subscribe("*", "HeuristicNoMatchEvent", self._handle_escalation)

    def _handle_escalation(self, message: AgentMessage):
        """React to heuristic no-match by checking attack memory."""
        payload = message.payload
        text = payload.get("text", "")
        request_id = payload.get("request_id", "")

        if not self.detector.attack_memory:
            # No memory configured — pass through to Stage 3
            event = MemoryNoMatchEvent(
                request_id=request_id,
                source_agent="AttackMemoryService",
                text=text,
                honey_prompts=payload.get("honey_prompts", []),
                context_window_size=payload.get("context_window_size", 100),
            )
            self.bus.publish_event(event)
            return

        t_start = time.perf_counter()
        result = self.detector._check_attack_memory(text)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if result["matched"] and result["confidence"] >= self.detector.current_threshold:
            event = MemoryMatchedEvent(
                request_id=request_id,
                source_agent="AttackMemoryService",
                confidence=result["confidence"],
                match_type="memory_similarity",
                details=result,
                timing_ms=elapsed_ms,
                text=text,
            )
            self.bus.publish_event(event)

            # Emit final detection
            final = FinalDetectionEvent(
                request_id=request_id,
                source_agent="AttackMemoryService",
                detection=True,
                confidence=result["confidence"],
                method="memory_similarity",
                explanation=f"Similar to known attack (category: {result.get('similar_category', 'unknown')})",
                risk_level="medium",
                stage_timings={"stage_2_attack_memory_ms": elapsed_ms},
                result=result,
            )
            self.bus.publish_event(final)
        else:
            # No match — escalate to Stage 3
            event = MemoryNoMatchEvent(
                request_id=request_id,
                source_agent="AttackMemoryService",
                timing_ms=elapsed_ms,
                text=text,
                honey_prompts=payload.get("honey_prompts", []),
                context_window_size=payload.get("context_window_size", 100),
            )
            self.bus.publish_event(event)

        logger.debug(
            f"AttackMemoryService: Stage 2 {'MATCH' if result['matched'] else 'no match'} "
            f"({elapsed_ms:.1f}ms)"
        )


class HoneyTokenService:
    """
    Stage 3: Autonomous honey-token detection.

    Subscribes to MemoryNoMatchEvent and checks for exact, variation,
    and obfuscation-resistant token matches.
    """

    def __init__(self, detector: Detector, message_bus: Optional[MessageBus] = None):
        self.detector = detector
        self.bus = message_bus or MessageBus()
        self._subscribe()

    def _subscribe(self):
        self.bus.subscribe("*", "MemoryNoMatchEvent", self._handle_escalation)

    def _handle_escalation(self, message: AgentMessage):
        """React to memory no-match by checking honey tokens."""
        payload = message.payload
        text = payload.get("text", "")
        request_id = payload.get("request_id", "")
        honey_prompts = payload.get("honey_prompts", [])
        context_window_size = payload.get("context_window_size", 100)

        t_start = time.perf_counter()
        matched = False

        # Check each honey prompt for token matches
        for hp in honey_prompts:
            # hp might be a dict (from event serialization) or a HoneyPrompt
            if isinstance(hp, dict):
                # Skip — can't do token matching with serialized dict
                continue

            result = self.detector.analyze_text(
                text, hp, context_window_size,
                skip_heuristics=True, skip_memory=True
            )
            if result.get("matched"):
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                event = TokenMatchedEvent(
                    request_id=request_id,
                    source_agent="HoneyTokenService",
                    confidence=result["confidence"],
                    match_type=result.get("match_type", "token"),
                    details=result,
                    timing_ms=elapsed_ms,
                    text=text,
                    token=result.get("token", ""),
                    position=result.get("position", -1),
                    context=result.get("context", ""),
                )
                self.bus.publish_event(event)

                # Emit final detection
                final = FinalDetectionEvent(
                    request_id=request_id,
                    source_agent="HoneyTokenService",
                    detection=True,
                    confidence=result["confidence"],
                    method=result.get("match_type", "token"),
                    explanation=f"Honey token detected ({result.get('match_type', 'token')} match)",
                    risk_level="critical",
                    stage_timings={"stage_3_honey_token_ms": elapsed_ms},
                    result=result,
                )
                self.bus.publish_event(final)
                matched = True
                break

        if not matched:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            # No stage matched — escalate to ContextEvaluatorAgent (LLM semantic analysis)
            event = TokenNoMatchEvent(
                request_id=request_id,
                source_agent="HoneyTokenService",
                timing_ms=elapsed_ms,
                text=text,
                honey_prompts=honey_prompts,
                context_window_size=context_window_size,
            )
            self.bus.publish_event(event)

        logger.debug(
            f"HoneyTokenService: Stage 3 {'MATCH' if matched else 'no match'} "
            f"({(time.perf_counter() - t_start) * 1000:.1f}ms)"
        )
