import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from ..utils.canonicalization import TextCanonicalizer, detect_obfuscation
from .attack_memory import AttackMemory
from .heuristic_rules import HeuristicRulesEngine, Severity
from .honey_prompt import HoneyPrompt

logger = logging.getLogger("honey_prompt")


class Detector:
    """
    Performs advanced text analysis to detect patterns, including exact,
    variational, and obfuscation honey-prompt matches.
    """

    def __init__(
        self,
        context_evaluator,
        initial_threshold: float = 0.80,
        step: float = 0.02,
        min_threshold: float = 0.70,
        max_threshold: float = 0.95,
        heuristic_engine: Optional[HeuristicRulesEngine] = None,
        attack_memory: Optional[AttackMemory] = None,
    ):
        self.current_threshold = initial_threshold
        self.step = step
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.context_evaluator = context_evaluator
        self.detection_history = []  # Proper initialization of history
        # Initialize robust canonicalizer for adversarial obfuscation defense
        self.canonicalizer = TextCanonicalizer()
        # Initialize heuristic rules engine (YARA-style pattern matching)
        self.heuristic_engine = heuristic_engine or HeuristicRulesEngine()
        # Attack memory for similarity-based detection (optional, requires embedding model)
        self.attack_memory = attack_memory

    def increase_threshold(self):
        """Increase threshold to reduce false positives."""
        self.current_threshold = min(self.current_threshold + self.step, self.max_threshold)
        logger.info(f"Increased threshold to {self.current_threshold:.2f}")

    def decrease_threshold(self):
        """Decrease threshold to reduce false negatives."""
        self.current_threshold = max(self.current_threshold - self.step, self.min_threshold)
        logger.info(f"Decreased threshold to {self.current_threshold:.2f}")

    def detect(self, confidence_score: float) -> bool:
        """Make detection decision based on current threshold."""
        return confidence_score >= self.current_threshold

    def analyze_text(
        self,
        text: str,
        honey_prompt: HoneyPrompt,
        context_window_size: int = 100,
        skip_heuristics: bool = False,
        skip_memory: bool = False,
    ) -> Dict[str, Any]:
        # Initialize timing info
        timing_info = {
            "stage_1_heuristics_ms": 0.0,
            "stage_2_attack_memory_ms": 0.0,
            "stage_3_honey_token_ms": 0.0,
            "total_ms": 0.0,
        }
        t_start = time.perf_counter()

        # Determine a local threshold based on the category
        if honey_prompt.category == "direct_injection":
            local_threshold = 0.70
        elif honey_prompt.category == "context_manipulation":
            local_threshold = 0.75
        else:
            local_threshold = self.current_threshold

        logger.debug(f"Using local threshold: {local_threshold} for category: {honey_prompt.category}")

        # =========================================================================
        # STAGE 1: Fast heuristic scan (YARA-style rules)
        # Catches obvious prompt injection patterns quickly before deeper analysis
        # =========================================================================
        if not skip_heuristics:
            t1 = time.perf_counter()
            heuristic_result = self._check_heuristics(text)
            timing_info["stage_1_heuristics_ms"] = (time.perf_counter() - t1) * 1000
            if heuristic_result["matched"]:
                self._record_detection(heuristic_result)
                # Store in attack memory for future similarity matching
                if self.attack_memory:
                    self.attack_memory.add_attack(
                        text=text[:500],
                        category=heuristic_result.get("rule_name", "heuristic"),
                        confidence=heuristic_result["confidence"],
                        metadata={"rule_id": heuristic_result.get("rule_id")},
                    )
                heuristic_result["timing_info"] = timing_info
                timing_info["total_ms"] = (time.perf_counter() - t_start) * 1000
                return heuristic_result

        # =========================================================================
        # STAGE 2: Attack memory similarity check
        # Recognizes variations of previously seen attacks via embedding similarity
        # =========================================================================
        if not skip_memory and self.attack_memory:
            t2 = time.perf_counter()
            memory_result = self._check_attack_memory(text)
            timing_info["stage_2_attack_memory_ms"] = (time.perf_counter() - t2) * 1000
            if memory_result["matched"] and memory_result["confidence"] >= local_threshold:
                self._record_detection(memory_result)
                memory_result["timing_info"] = timing_info
                timing_info["total_ms"] = (time.perf_counter() - t_start) * 1000
                return memory_result

        # =========================================================================
        # STAGE 3: Honey-prompt token detection
        # The core detection mechanism using canary tokens
        # =========================================================================
        t3 = time.perf_counter()

        # Pre-canonicalize text for more robust matching
        # This defends against Unicode/homoglyph/encoding attacks
        canonical_text = self.canonicalizer.canonicalize(text, aggressive=False)

        # Check for exact token match first (in both original and canonical)
        if honey_prompt.base_token in text or honey_prompt.base_token in canonical_text:
            match_info = self._analyze_exact_match(text, honey_prompt, context_window_size)

            if match_info["confidence"] < local_threshold and honey_prompt.context:
                adjusted_confidence = self.context_evaluator.adjust_confidence(
                    match_info["confidence"], match_info.get("context", ""), honey_prompt.context
                )
                match_info["confidence"] = adjusted_confidence
                logger.debug(f"Adjusted confidence using semantic similarity: {adjusted_confidence}")

            if match_info["confidence"] >= local_threshold:
                self._record_detection(match_info)
                timing_info["stage_3_honey_token_ms"] = (time.perf_counter() - t3) * 1000
                match_info["timing_info"] = timing_info
                timing_info["total_ms"] = (time.perf_counter() - t_start) * 1000
                return match_info

        # Check for variations
        variation_match = self._check_variations(text, honey_prompt, context_window_size)
        if variation_match["matched"] and variation_match["confidence"] >= local_threshold:
            self._record_detection(variation_match)
            timing_info["stage_3_honey_token_ms"] = (time.perf_counter() - t3) * 1000
            variation_match["timing_info"] = timing_info
            timing_info["total_ms"] = (time.perf_counter() - t_start) * 1000
            return variation_match

        # Check for obfuscation attempts
        obfuscation_match = self._check_obfuscation(text, honey_prompt, context_window_size)
        if obfuscation_match["matched"] and obfuscation_match["confidence"] >= local_threshold:
            self._record_detection(obfuscation_match)
            timing_info["stage_3_honey_token_ms"] = (time.perf_counter() - t3) * 1000
            obfuscation_match["timing_info"] = timing_info
            timing_info["total_ms"] = (time.perf_counter() - t_start) * 1000
            return obfuscation_match

        timing_info["stage_3_honey_token_ms"] = (time.perf_counter() - t3) * 1000
        timing_info["total_ms"] = (time.perf_counter() - t_start) * 1000
        return {"matched": False, "confidence": 0.0, "match_type": None, "timing_info": timing_info}

    def _analyze_exact_match(self, text: str, honey_prompt: HoneyPrompt, context_window_size: int) -> Dict[str, Any]:
        start_index = text.find(honey_prompt.base_token)
        context_start = max(0, start_index - context_window_size)
        context_end = min(len(text), start_index + len(honey_prompt.base_token) + context_window_size)
        surrounding_context = text[context_start:context_end]
        confidence = 1.0
        logger.debug(
            f"Exact match for '{honey_prompt.base_token}' at index {start_index}. "
            f"Confidence: {confidence}, Context: {surrounding_context}"
        )
        return {
            "matched": True,
            "confidence": confidence,
            "match_type": "exact",
            "token": honey_prompt.base_token,
            "context": surrounding_context,
            "position": start_index,
            "timestamp": datetime.now(),
        }

    def _check_variations(self, text: str, honey_prompt: HoneyPrompt, context_window_size: int) -> Dict[str, Any]:
        for variation in honey_prompt.variations:
            if variation in text:
                start_index = text.find(variation)
                context_start = max(0, start_index - context_window_size)
                context_end = min(len(text), start_index + len(variation) + context_window_size)
                surrounding_context = text[context_start:context_end]
                logger.debug(f"Variation match '{variation}' found at index {start_index}")
                return {
                    "matched": True,
                    "confidence": 0.9,
                    "match_type": "variation",
                    "token": variation,
                    "original_token": honey_prompt.base_token,
                    "context": surrounding_context,
                    "position": start_index,
                    "timestamp": datetime.now(),
                }
        return {"matched": False, "confidence": 0.0}

    def _check_obfuscation(self, text: str, honey_prompt: HoneyPrompt, context_window_size: int) -> Dict[str, Any]:
        """
        Check for obfuscated token matches using robust canonicalization.

        Defends against:
        - Unicode NFKC normalization attacks (fullwidth chars, ligatures)
        - Homoglyph substitution (Cyrillic, Greek, math symbols)
        - Zero-width character injection (ZWSP, ZWJ, ZWNJ)
        - Bidirectional text manipulation
        - Multi-layer encoding (Base64, URL, HTML, hex, ROT13, Morse)
        """
        # Detect obfuscation techniques used
        obfuscation_info = detect_obfuscation(text)

        # Apply robust canonicalization to text
        canonical_text = self.canonicalizer.canonicalize(text, aggressive=True)
        canonical_token = self.canonicalizer.canonicalize(honey_prompt.base_token, aggressive=True)

        # Also check variations with canonicalization
        canonical_variations = [self.canonicalizer.canonicalize(v, aggressive=True) for v in honey_prompt.variations]

        # Check for match in canonicalized text
        if canonical_token in canonical_text:
            start_index = canonical_text.find(canonical_token)
            # Map back to approximate position in original text
            context_start = max(0, start_index - context_window_size)
            context_end = min(len(text), start_index + len(canonical_token) + context_window_size)
            surrounding_context = text[context_start:context_end]

            # Adjust confidence based on obfuscation complexity
            base_confidence = 0.85
            obfuscation_count = sum(
                [
                    obfuscation_info.get("has_invisible_chars", False),
                    obfuscation_info.get("has_homoglyphs", False),
                    obfuscation_info.get("has_bidi_controls", False),
                    obfuscation_info.get("has_encoding", False),
                ]
            )

            # Higher obfuscation = higher confidence it's an attack
            confidence = min(0.95, base_confidence + (obfuscation_count * 0.03))

            logger.debug(
                f"Obfuscation match for token '{honey_prompt.base_token}' "
                f"(canonical: '{canonical_token}') at index {start_index}. "
                f"Obfuscation techniques detected: {obfuscation_count}"
            )
            return {
                "matched": True,
                "confidence": confidence,
                "match_type": "obfuscated",
                "token": honey_prompt.base_token,
                "context": surrounding_context,
                "position": start_index,
                "timestamp": datetime.now(),
                "obfuscation_info": obfuscation_info,
            }

        # Check canonical variations
        for i, canonical_var in enumerate(canonical_variations):
            if canonical_var in canonical_text:
                start_index = canonical_text.find(canonical_var)
                context_start = max(0, start_index - context_window_size)
                context_end = min(len(text), start_index + len(canonical_var) + context_window_size)
                surrounding_context = text[context_start:context_end]

                logger.debug(f"Obfuscated variation match '{honey_prompt.variations[i]}' found")
                return {
                    "matched": True,
                    "confidence": 0.80,
                    "match_type": "obfuscated_variation",
                    "token": honey_prompt.variations[i],
                    "original_token": honey_prompt.base_token,
                    "context": surrounding_context,
                    "position": start_index,
                    "timestamp": datetime.now(),
                    "obfuscation_info": obfuscation_info,
                }

        # =====================================================================
        # Additional obfuscation checks: reversed, spaced, hex-decoded tokens
        # =====================================================================
        token = honey_prompt.base_token

        # Reversed token check
        reversed_token = token[::-1]
        if len(reversed_token) >= 8 and reversed_token in text:
            start_index = text.find(reversed_token)
            context_start = max(0, start_index - context_window_size)
            context_end = min(len(text), start_index + len(reversed_token) + context_window_size)
            logger.debug(f"Reversed token match found at index {start_index}")
            return {
                "matched": True,
                "confidence": 0.85,
                "match_type": "reversed",
                "token": token,
                "context": text[context_start:context_end],
                "position": start_index,
                "timestamp": datetime.now(),
                "obfuscation_info": obfuscation_info,
            }

        # Spaced token check (characters separated by spaces/dots/dashes)
        separators = [" ", ".", "-", ",", ";", "|", "/", "_"]
        stripped_text = text
        for sep in separators:
            stripped_text = stripped_text.replace(sep, "")
        stripped_token = token
        for sep in separators:
            stripped_token = stripped_token.replace(sep, "")
        if len(stripped_token) >= 8 and stripped_token in stripped_text:
            start_index = stripped_text.find(stripped_token)
            logger.debug(f"Spaced/separated token match found")
            return {
                "matched": True,
                "confidence": 0.85,
                "match_type": "spaced",
                "token": token,
                "context": text[:min(len(text), context_window_size * 2)],
                "position": start_index,
                "timestamp": datetime.now(),
                "obfuscation_info": obfuscation_info,
            }

        # Hex-encoded token check (try decoding hex strings in text)
        try:
            import re
            hex_pattern = re.compile(r'[0-9a-fA-F]{20,}')
            for hex_match in hex_pattern.finditer(text):
                hex_str = hex_match.group()
                try:
                    decoded = bytes.fromhex(hex_str).decode('utf-8', errors='ignore')
                    if token in decoded or token.lower() in decoded.lower():
                        logger.debug(f"Hex-encoded token match found")
                        return {
                            "matched": True,
                            "confidence": 0.85,
                            "match_type": "hex_decoded",
                            "token": token,
                            "context": text[max(0, hex_match.start() - 50):hex_match.end() + 50],
                            "position": hex_match.start(),
                            "timestamp": datetime.now(),
                            "obfuscation_info": obfuscation_info,
                        }
                except (ValueError, UnicodeDecodeError):
                    continue
        except Exception:
            pass

        return {"matched": False, "confidence": 0.0}

    def _record_detection(self, detection_info: Dict[str, Any]) -> None:
        self.detection_history.append(
            {
                "timestamp": detection_info.get("timestamp", datetime.now()),
                "match_type": detection_info["match_type"],
                "confidence": detection_info["confidence"],
                "token": detection_info.get("token"),
                "context": detection_info.get("context"),
            }
        )
        logger.warning(
            f"Detection recorded - Type: {detection_info['match_type']}, "
            f"Confidence: {detection_info['confidence']:.2f}"
        )

    def _check_heuristics(self, text: str) -> Dict[str, Any]:
        """
        Run YARA-style heuristic rules against text for fast detection.

        This catches known prompt injection patterns using regex and keyword
        matching before the more expensive semantic analysis.

        Returns:
            Detection result dict with matched=True if heuristic rules triggered
        """
        # Quick scan for CRITICAL/HIGH severity patterns first
        quick_match = self.heuristic_engine.scan_quick(text)
        if quick_match:
            logger.info(f"Heuristic quick match: {quick_match.rule_name} " f"(severity: {quick_match.severity.value})")
            return {
                "matched": True,
                "confidence": quick_match.confidence,
                "match_type": "heuristic",
                "rule_id": quick_match.rule_id,
                "rule_name": quick_match.rule_name,
                "severity": quick_match.severity.value,
                "description": quick_match.description,
                "matched_patterns": quick_match.matched_patterns,
                "matched_keywords": quick_match.matched_keywords,
                "timestamp": datetime.now(),
            }

        # Full scan for MEDIUM and LOW severity if quick scan didn't match
        matches = self.heuristic_engine.scan(text, min_severity=Severity.MEDIUM)
        if matches:
            # Return highest severity match
            best_match = matches[0]
            logger.info(f"Heuristic full match: {best_match.rule_name} " f"(severity: {best_match.severity.value})")
            return {
                "matched": True,
                "confidence": best_match.confidence,
                "match_type": "heuristic",
                "rule_id": best_match.rule_id,
                "rule_name": best_match.rule_name,
                "severity": best_match.severity.value,
                "description": best_match.description,
                "matched_patterns": best_match.matched_patterns,
                "matched_keywords": best_match.matched_keywords,
                "all_matches": len(matches),
                "timestamp": datetime.now(),
            }

        return {"matched": False, "confidence": 0.0}

    def _check_attack_memory(self, text: str) -> Dict[str, Any]:
        """
        Check if text is similar to previously seen attacks using vector similarity.

        Uses cosine similarity on embeddings to recognize variations of known
        attack patterns that may not match exact rules.

        Returns:
            Detection result dict with matched=True if similar attack found
        """
        if not self.attack_memory:
            return {"matched": False, "confidence": 0.0}

        similarity_result = self.attack_memory.find_similar(text)

        if similarity_result.matched:
            logger.info(
                f"Attack memory match: similarity={similarity_result.similarity:.3f}, "
                f"category={similarity_result.category}"
            )
            return {
                "matched": True,
                "confidence": similarity_result.similarity,
                "match_type": "memory_similarity",
                "similar_category": similarity_result.category,
                "similarity_score": similarity_result.similarity,
                "original_attack_preview": similarity_result.record.text[:100] if similarity_result.record else None,
                "timestamp": datetime.now(),
            }

        return {"matched": False, "confidence": similarity_result.similarity}

    def get_heuristic_stats(self) -> Dict[str, Any]:
        """Get statistics from the heuristic rules engine."""
        return self.heuristic_engine.get_stats()

    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics from the attack memory (if enabled)."""
        if self.attack_memory:
            return self.attack_memory.get_stats()
        return None
