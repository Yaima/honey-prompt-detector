"""Heuristic-only baseline detector (Stage 1 only)."""

from typing import Any, Dict
from ..core.heuristic_rules import HeuristicRulesEngine, Severity


class HeuristicBaseline:
    """Baseline detector using only YARA-style heuristic rules.

    This represents Stage 1 of the HIVE pipeline in isolation,
    serving as an ablation baseline for comparison.
    """

    def __init__(self):
        self.engine = HeuristicRulesEngine()

    def detect(self, text: str) -> Dict[str, Any]:
        """Run heuristic-only detection."""
        quick = self.engine.scan_quick(text)
        if quick:
            return {
                "detection": True,
                "confidence": quick.confidence,
                "rule_id": quick.rule_id,
                "method": "heuristic_only",
            }

        matches = self.engine.scan(text, min_severity=Severity.MEDIUM)
        if matches:
            return {
                "detection": True,
                "confidence": matches[0].confidence,
                "rule_id": matches[0].rule_id,
                "method": "heuristic_only",
            }

        return {"detection": False, "confidence": 0.0, "method": "heuristic_only"}
