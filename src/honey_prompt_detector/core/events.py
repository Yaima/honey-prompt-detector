"""
Event Definitions for HIVE's Event-Driven Agent Architecture.

All inter-agent communication flows through typed events published
on the MessageBus. Each event carries a request_id for correlation
and a source_agent for traceability.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BaseEvent:
    """Base class for all HIVE events."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source_agent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["timestamp"] = self.timestamp.isoformat()
        d["event_type"] = self.__class__.__name__
        return d


# ─── Pipeline Input Events ──────────────────────────────────────────


@dataclass
class TextReceivedEvent(BaseEvent):
    """Emitted when new text arrives for analysis."""

    text: str = ""
    honey_prompts: List[Any] = field(default_factory=list)
    context_window_size: int = 100
    source: str = "input"  # "input" or "output"


# ─── Stage Completion Events ────────────────────────────────────────


@dataclass
class StageResultEvent(BaseEvent):
    """Emitted when a detection stage completes."""

    stage: str = ""  # "heuristic", "memory", "token"
    matched: bool = False
    confidence: float = 0.0
    match_type: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    honey_prompts: List[Any] = field(default_factory=list)
    context_window_size: int = 100
    timing_ms: float = 0.0


@dataclass
class HeuristicMatchedEvent(StageResultEvent):
    """Stage 1 found a match via heuristic rules."""

    stage: str = "heuristic"
    matched: bool = True


@dataclass
class HeuristicNoMatchEvent(StageResultEvent):
    """Stage 1 found no match — escalate to Stage 2."""

    stage: str = "heuristic"
    matched: bool = False


@dataclass
class MemoryMatchedEvent(StageResultEvent):
    """Stage 2 found a similar attack in memory."""

    stage: str = "memory"
    matched: bool = True


@dataclass
class MemoryNoMatchEvent(StageResultEvent):
    """Stage 2 found no similar attack — escalate to Stage 3."""

    stage: str = "memory"
    matched: bool = False


@dataclass
class TokenMatchedEvent(StageResultEvent):
    """Stage 3 found a honey-token match."""

    stage: str = "token"
    matched: bool = True
    token: str = ""
    position: int = -1
    context: str = ""


@dataclass
class TokenNoMatchEvent(StageResultEvent):
    """No stage found a match — escalate to LLM semantic evaluator."""

    stage: str = "token"
    matched: bool = False


# ─── Final Detection Events ─────────────────────────────────────────


@dataclass
class FinalDetectionEvent(BaseEvent):
    """Final detection result emitted by the decision authority (ContextEvaluatorAgent)."""

    detection: bool = False
    confidence: float = 0.0
    method: str = ""  # "heuristic", "memory", "token", "llm_semantic", "pattern_match"
    explanation: str = ""
    risk_level: str = "low"
    timing_ms: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    # Full result dict for backward compat
    result: Dict[str, Any] = field(default_factory=dict)


# ─── Agent Autonomy Events ──────────────────────────────────────────


@dataclass
class TokenRotationRequestEvent(BaseEvent):
    """TokenDesignerAgent autonomously requests token rotation."""

    reason: str = ""
    new_token: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyChangedEvent(BaseEvent):
    """EnvironmentAgent autonomously changed embedding strategy."""

    old_strategy: str = ""
    new_strategy: str = ""
    reason: str = ""


@dataclass
class ThresholdAdjustedEvent(BaseEvent):
    """SelfTuner adjusted detection threshold."""

    old_threshold: float = 0.0
    new_threshold: float = 0.0
    method: str = ""  # "pseudo_label", "thompson_sampling", "human_feedback"
