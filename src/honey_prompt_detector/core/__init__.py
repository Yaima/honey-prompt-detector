"""
Core detection components for Honey-Prompt Injection Detection.
"""

from .attack_memory import AttackMemory, AttackRecord, SimilarityMatch
from .detector import Detector
from .heuristic_rules import BUILTIN_RULES, HeuristicRule, HeuristicRulesEngine, RuleMatch, Severity, scan_for_injection
from .honey_prompt import HoneyPrompt

__all__ = [
    # Main detector
    "Detector",
    "HoneyPrompt",
    # Heuristic rules (YARA-style)
    "HeuristicRulesEngine",
    "HeuristicRule",
    "RuleMatch",
    "Severity",
    "BUILTIN_RULES",
    "scan_for_injection",
    # Attack memory (VectorDB)
    "AttackMemory",
    "AttackRecord",
    "SimilarityMatch",
]
