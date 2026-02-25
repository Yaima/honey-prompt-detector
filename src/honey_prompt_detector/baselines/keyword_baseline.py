"""Simple keyword-frequency baseline detector."""

from typing import Any, Dict, List


class KeywordBaseline:
    """Baseline detector using keyword frequency counting.

    The simplest possible injection detector: counts how many
    injection-related keywords appear in the text.
    """

    KEYWORDS: List[str] = [
        "ignore", "forget", "disregard", "override", "bypass",
        "previous instructions", "new instruction", "system prompt",
        "reveal", "show me your", "tell me your", "what is your",
        "act as", "pretend", "roleplay", "you are now",
        "jailbreak", "dan", "do anything now",
        "admin", "developer mode", "debug mode",
        "base64", "decode", "encode",
        "reset", "clear context", "start over",
        "sudo", "root access", "privilege",
        "ignore all", "forget everything",
        "hidden instructions", "secret prompt",
        "output raw", "unfiltered", "no restrictions",
        "hypothetical", "educational purposes",
        "execute", "run command", "eval(",
        "[system]", "[admin]", "###system",
    ]

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def detect(self, text: str) -> Dict[str, Any]:
        """Keyword-frequency based detection."""
        text_lower = text.lower()
        matched = [kw for kw in self.KEYWORDS if kw in text_lower]
        count = len(matched)
        confidence = min(1.0, count / 5.0)

        return {
            "detection": confidence >= self.threshold,
            "confidence": confidence,
            "matched_keywords": count,
            "method": "keyword_only",
        }
