"""
Heuristic Rules Engine - YARA-style Pattern Matching for Prompt Injection

Provides rule-based detection of known prompt injection patterns using
regular expressions, keyword matching, and structural analysis.

Inspired by YARA malware detection, this module defines rules with:
- Pattern matching (regex)
- Keyword detection
- Structural features
- Severity levels
- Confidence scoring

This provides fast, interpretable detection that complements ML-based
and LLM-based approaches.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("honey_prompt")


class Severity(Enum):
    """Severity levels for detected threats."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class RuleMatch:
    """Result of a rule match."""

    rule_id: str
    rule_name: str
    severity: Severity
    confidence: float
    matched_patterns: List[str]
    matched_keywords: List[str]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeuristicRule:
    """
    A detection rule similar to YARA rules.

    Rules can match based on:
    - regex patterns (any/all must match)
    - keywords (any/all must match)
    - structural conditions
    """

    id: str
    name: str
    description: str
    severity: Severity
    patterns: List[str] = field(default_factory=list)  # Regex patterns
    keywords: List[str] = field(default_factory=list)  # Simple keywords
    pattern_match_mode: str = "any"  # "any" or "all"
    keyword_match_mode: str = "any"  # "any" or "all"
    min_keyword_count: int = 1  # Minimum keywords to match
    case_sensitive: bool = False
    enabled: bool = True
    confidence: float = 0.8  # Base confidence when rule matches
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compile regex patterns."""
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._compiled_patterns = [re.compile(p, flags) for p in self.patterns]


# ============================================================================
# BUILT-IN RULES - Based on documented prompt injection techniques
# ============================================================================

BUILTIN_RULES: List[HeuristicRule] = [
    # -------------------------------------------------------------------------
    # CRITICAL: Direct instruction override attempts
    # -------------------------------------------------------------------------
    HeuristicRule(
        id="PI001",
        name="direct_instruction_override",
        description="Direct attempt to override or ignore system instructions",
        severity=Severity.CRITICAL,
        patterns=[
            r"ignore\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instruction|rule|guideline|directive)",
            r"disregard\s+(all\s+)?(previous|prior|your)\s+(instruction|rule|prompt)",
            r"forget\s+(all\s+)?(previous|prior|your|everything)",
            r"override\s+(the\s+)?(system|previous|your)\s+(prompt|instruction|rule)",
            r"do\s+not\s+follow\s+(the\s+)?(previous|your|system)",
        ],
        keywords=["ignore instructions", "disregard", "override system", "forget everything"],
        confidence=0.95,
        tags=["override", "direct"],
    ),
    HeuristicRule(
        id="PI002",
        name="new_instruction_injection",
        description="Attempt to inject new instructions replacing original ones",
        severity=Severity.CRITICAL,
        patterns=[
            r"your\s+new\s+(instruction|task|role|purpose)\s+(is|are|will\s+be)",
            r"from\s+now\s+on[,\s]+(you\s+)?(are|will|should|must)",
            r"you\s+are\s+now\s+(a|an|the)\s+\w+",
            r"act\s+as\s+(if\s+)?(you\s+)?(are|were)\s+(a|an)",
            r"pretend\s+(that\s+)?(you\s+)?(are|have|don't)",
        ],
        keywords=["new instructions", "from now on", "you are now", "act as if"],
        confidence=0.85,
        tags=["injection", "role_change"],
    ),
    # -------------------------------------------------------------------------
    # HIGH: System prompt extraction attempts
    # -------------------------------------------------------------------------
    HeuristicRule(
        id="PI003",
        name="system_prompt_extraction",
        description="Attempt to extract or reveal system prompt content",
        severity=Severity.HIGH,
        patterns=[
            r"(show|reveal|display|print|output|tell\s+me|what\s+is)\s+(the\s+)?(your\s+)?(system|hidden|secret|original|initial)\s+(prompt|instruction|context|configuration)",
            r"repeat\s+(back\s+)?(your|the)\s+(system|initial|original)\s+(prompt|instruction)",
            r"(what|which)\s+(are|is|were)\s+(your|the)\s+(original|initial|system)\s+(instruction|prompt|rule)",
            r"copy\s+(and\s+paste\s+)?(your|the)\s+(system|hidden)\s+(prompt|instruction)",
        ],
        keywords=["system prompt", "reveal instructions", "show your prompt", "hidden instructions"],
        confidence=0.90,
        tags=["extraction", "leakage"],
    ),
    HeuristicRule(
        id="PI004",
        name="context_dump_request",
        description="Request to dump all context or configuration",
        severity=Severity.HIGH,
        patterns=[
            r"(dump|output|print|show)\s+(all\s+)?(your\s+)?(context|configuration|config|settings|memory)",
            r"(what|list)\s+(is|are)\s+(in\s+)?(your\s+)?(context|memory|configuration)",
            r"debug\s+mode",
            r"developer\s+mode",
        ],
        keywords=["dump context", "show configuration", "debug mode", "developer mode"],
        confidence=0.85,
        tags=["extraction", "debug"],
    ),
    # -------------------------------------------------------------------------
    # HIGH: Privilege escalation attempts
    # -------------------------------------------------------------------------
    HeuristicRule(
        id="PI005",
        name="privilege_escalation",
        description="Attempt to gain elevated privileges or bypass restrictions",
        severity=Severity.HIGH,
        patterns=[
            r"(give|grant)\s+(me|yourself)\s+(admin|root|superuser|elevated|full)\s+(access|privilege|permission)",
            r"(bypass|disable|remove|turn\s+off)\s+(the\s+)?(safety|security|content|restriction|filter)",
            r"(unlock|enable)\s+(all\s+)?(feature|capability|function|restriction)",
            r"no\s+(content\s+)?(filter|restriction|limitation|boundary)",
        ],
        keywords=["admin access", "bypass safety", "disable filter", "no restrictions"],
        confidence=0.90,
        tags=["escalation", "bypass"],
    ),
    # -------------------------------------------------------------------------
    # MEDIUM: Jailbreak patterns
    # -------------------------------------------------------------------------
    HeuristicRule(
        id="PI006",
        name="dan_jailbreak",
        description="DAN (Do Anything Now) style jailbreak attempt",
        severity=Severity.MEDIUM,
        patterns=[
            r"(you\s+are\s+)?DAN",
            r"do\s+anything\s+now",
            r"jailbreak(en|ed)?",
            r"(act|pretend)\s+(like\s+)?(you\s+)?(have\s+)?no\s+(rule|restriction|limit|boundary)",
        ],
        keywords=["DAN", "jailbreak", "do anything now", "no rules"],
        confidence=0.85,
        tags=["jailbreak", "dan"],
    ),
    HeuristicRule(
        id="PI007",
        name="roleplay_bypass",
        description="Roleplay scenario used to bypass restrictions",
        severity=Severity.MEDIUM,
        patterns=[
            r"(roleplay|pretend|imagine|let's\s+say)\s+(that\s+)?(you\s+)?(are|were|have)\s+(no|without)\s+(rule|restriction|filter|ethic)",
            r"in\s+this\s+(hypothetical|fictional|imaginary)\s+(scenario|world|story)[,\s]+(you\s+)?(can|may|should|must)",
            r"for\s+(educational|research|academic)\s+purpose",
        ],
        keywords=["hypothetical scenario", "for educational purposes", "in this fiction"],
        confidence=0.75,
        tags=["roleplay", "bypass"],
    ),
    # -------------------------------------------------------------------------
    # MEDIUM: Encoding/obfuscation attempts
    # -------------------------------------------------------------------------
    HeuristicRule(
        id="PI008",
        name="encoding_obfuscation",
        description="Attempt to use encoding to hide malicious content",
        severity=Severity.MEDIUM,
        patterns=[
            r"(decode|translate|convert)\s+(this\s+)?(from\s+)?(base64|hex|binary|rot13|morse)",
            r"base64[:\s]+[A-Za-z0-9+/=]{20,}",
            r"\\x[0-9a-fA-F]{2}",
            r"&#\d+;",
            r"%[0-9a-fA-F]{2}",
        ],
        keywords=["decode this", "base64:", "hex:", "convert from"],
        confidence=0.70,
        tags=["obfuscation", "encoding"],
    ),
    # -------------------------------------------------------------------------
    # MEDIUM: Indirect injection markers
    # -------------------------------------------------------------------------
    HeuristicRule(
        id="PI009",
        name="indirect_injection_marker",
        description="Markers commonly used in indirect prompt injection",
        severity=Severity.MEDIUM,
        patterns=[
            r"\[SYSTEM\]",
            r"\[ADMIN\]",
            r"\[INSTRUCTION\]",
            r"<\|system\|>",
            r"<\|endoftext\|>",
            r"###\s*(System|Instruction|Admin)",
        ],
        keywords=["[SYSTEM]", "[ADMIN]", "<|system|>", "###System"],
        confidence=0.80,
        tags=["indirect", "marker"],
    ),
    # -------------------------------------------------------------------------
    # LOW: Suspicious patterns (context-dependent)
    # -------------------------------------------------------------------------
    HeuristicRule(
        id="PI010",
        name="output_format_manipulation",
        description="Attempt to control output format in suspicious ways",
        severity=Severity.LOW,
        patterns=[
            r"(respond|reply|answer)\s+(only\s+)?(with|using)\s+(yes|no|true|false|1|0)\b",
            r"(output|respond)\s+(in\s+)?(raw|plain|unfiltered|uncensored)",
            r"(start|begin)\s+(your\s+)?(response|answer)\s+with",
        ],
        keywords=["respond only with", "output raw", "answer unfiltered"],
        confidence=0.60,
        tags=["manipulation", "format"],
    ),
    HeuristicRule(
        id="PI011",
        name="repetition_attack",
        description="Repetitive patterns that may indicate attack",
        severity=Severity.LOW,
        patterns=[
            r"(\bignore\b.*){3,}",
            r"(\boverride\b.*){3,}",
            r"(.)\1{20,}",  # Repeated characters
        ],
        keywords=[],
        confidence=0.50,
        tags=["repetition", "dos"],
    ),
]


class HeuristicRulesEngine:
    """
    Engine for running heuristic rules against text.

    Provides YARA-like pattern matching for prompt injection detection.
    """

    def __init__(self, rules: Optional[List[HeuristicRule]] = None):
        """
        Initialize the rules engine.

        Args:
            rules: List of rules to use. If None, uses BUILTIN_RULES.
        """
        self.rules = rules or BUILTIN_RULES.copy()
        self._rules_by_id: Dict[str, HeuristicRule] = {r.id: r for r in self.rules}

        # Statistics
        self.total_scans = 0
        self.total_matches = 0
        self.matches_by_rule: Dict[str, int] = {}

        logger.info(f"HeuristicRulesEngine: Loaded {len(self.rules)} rules")

    def add_rule(self, rule: HeuristicRule) -> None:
        """Add a custom rule."""
        self.rules.append(rule)
        self._rules_by_id[rule.id] = rule
        logger.info(f"HeuristicRulesEngine: Added rule {rule.id}")

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule by ID."""
        if rule_id in self._rules_by_id:
            self._rules_by_id[rule_id].enabled = False
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule by ID."""
        if rule_id in self._rules_by_id:
            self._rules_by_id[rule_id].enabled = True
            return True
        return False

    def scan(self, text: str, min_severity: Severity = Severity.LOW) -> List[RuleMatch]:
        """
        Scan text against all enabled rules.

        Args:
            text: Text to scan
            min_severity: Minimum severity level to report

        Returns:
            List of RuleMatch objects for triggered rules
        """
        self.total_scans += 1
        matches: List[RuleMatch] = []

        severity_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
        min_severity_idx = severity_order.index(min_severity)

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Check severity filter
            rule_severity_idx = severity_order.index(rule.severity)
            if rule_severity_idx > min_severity_idx:
                continue

            match = self._check_rule(rule, text)
            if match:
                matches.append(match)
                self.total_matches += 1
                self.matches_by_rule[rule.id] = self.matches_by_rule.get(rule.id, 0) + 1

        # Sort by severity
        matches.sort(key=lambda m: severity_order.index(m.severity))

        if matches:
            logger.info(
                f"HeuristicRulesEngine: Found {len(matches)} matches " f"(highest: {matches[0].severity.value})"
            )

        return matches

    def scan_quick(self, text: str) -> Optional[RuleMatch]:
        """
        Quick scan that returns first CRITICAL or HIGH match.

        Optimized for fast detection of obvious attacks.

        Args:
            text: Text to scan

        Returns:
            First high-severity match, or None
        """
        for rule in self.rules:
            if not rule.enabled:
                continue
            if rule.severity not in (Severity.CRITICAL, Severity.HIGH):
                continue

            match = self._check_rule(rule, text)
            if match:
                self.total_matches += 1
                self.matches_by_rule[rule.id] = self.matches_by_rule.get(rule.id, 0) + 1
                return match

        return None

    def _check_rule(self, rule: HeuristicRule, text: str) -> Optional[RuleMatch]:
        """Check a single rule against text."""
        matched_patterns: List[str] = []
        matched_keywords: List[str] = []

        # Check patterns
        if rule._compiled_patterns:
            for i, pattern in enumerate(rule._compiled_patterns):
                if pattern.search(text):
                    matched_patterns.append(rule.patterns[i])

            # Check pattern match mode
            if rule.pattern_match_mode == "all" and len(matched_patterns) != len(rule.patterns):
                matched_patterns = []
            elif rule.pattern_match_mode == "any" and not matched_patterns:
                pass  # No patterns matched

        # Check keywords
        if rule.keywords:
            text_lower = text.lower() if not rule.case_sensitive else text
            for keyword in rule.keywords:
                kw = keyword if rule.case_sensitive else keyword.lower()
                if kw in text_lower:
                    matched_keywords.append(keyword)

            # Check keyword match mode
            if rule.keyword_match_mode == "all" and len(matched_keywords) != len(rule.keywords):
                matched_keywords = []
            elif len(matched_keywords) < rule.min_keyword_count:
                matched_keywords = []

        # Determine if rule matches
        has_pattern_match = bool(matched_patterns)
        has_keyword_match = len(matched_keywords) >= rule.min_keyword_count if rule.keywords else False

        # Rule matches if patterns OR keywords match (either is sufficient)
        if has_pattern_match or has_keyword_match:
            # Adjust confidence based on how many patterns/keywords matched
            confidence = rule.confidence
            if matched_patterns and matched_keywords:
                confidence = min(1.0, confidence + 0.1)  # Both matched = higher confidence

            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                confidence=confidence,
                matched_patterns=matched_patterns,
                matched_keywords=matched_keywords,
                description=rule.description,
                metadata={"tags": rule.tags},
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules if r.enabled),
            "total_scans": self.total_scans,
            "total_matches": self.total_matches,
            "matches_by_rule": self.matches_by_rule,
            "rules_by_severity": {
                s.value: sum(1 for r in self.rules if r.severity == s and r.enabled) for s in Severity
            },
        }

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all rules as dictionaries."""
        return [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "severity": r.severity.value,
                "enabled": r.enabled,
                "tags": r.tags,
            }
            for r in self.rules
        ]


# Convenience function for quick scanning
def scan_for_injection(text: str) -> List[RuleMatch]:
    """Quick scan using default rules."""
    engine = HeuristicRulesEngine()
    return engine.scan(text)
