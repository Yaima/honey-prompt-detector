import asyncio
import base64
import binascii
import logging
import uuid
from typing import Any, Dict, List

from src.honey_prompt_detector.core.self_tuner import EnhancedSelfTuner

from ..agents.base_agent import MessageBus
from ..agents.context_evaluator_agent import ContextEvaluatorAgent
from ..agents.environment_agent import EnvironmentAgent
from ..agents.token_designer_agent import TokenDesignerAgent
from ..core.analysis_pipeline import (
    AttackMemoryService,
    HeuristicAnalysisService,
    HoneyTokenService,
)
from ..core.attack_memory import AttackMemory
from ..core.detector import Detector
from ..core.events import FinalDetectionEvent, TextReceivedEvent, TokenRotationRequestEvent
from ..core.honey_prompt import HoneyPrompt

logger = logging.getLogger("honey_prompt")


class Orchestrator:
    """
    Coordinates detection logic for prompt injection attacks.

    This class serves as a central orchestrator that utilizes a token designing
    agent and context evaluation agent to detect potential prompt injection
    attacks. It initializes \"honey prompts\" that are used as detection markers
    in monitored text and performs text monitoring to identify possible threats.

    Attributes:
      token_designer: Responsible for designing honey prompts.
      context_evaluator: Evaluates context for risk analysis.
      config: Configuration object for system behavior.
      honey_prompts: List of designed honey prompts.
      detector: Instance of Detector used for matching honey-prompts in text.
    """

    def __init__(self, token_designer: TokenDesignerAgent, context_evaluator: ContextEvaluatorAgent, config: Any):
        self.token_designer = token_designer
        self.context_evaluator = context_evaluator
        self.config = config
        self.honey_prompts: List[HoneyPrompt] = []
        self.environment_agent = EnvironmentAgent(similarity_model_name=self.config.similarity_model_name)

        # Initialize AttackMemory with the embedding model from environment_agent
        attack_memory = AttackMemory(
            embedding_model=self.environment_agent.similarity_model,
            similarity_threshold=0.85,
            max_records=10000,
            persistence_path=None,  # Disabled persistence for now - can add to config later
        )

        self.detector = Detector(
            context_evaluator=self.context_evaluator,
            attack_memory=attack_memory,
        )
        self.self_tuner = EnhancedSelfTuner(detector_agent=self.detector, config=config)
        self.current_threshold = config.initial_threshold

        # ── Event-Driven Pipeline Setup ──────────────────────────────
        # Shared message bus (singleton) for inter-agent communication
        self.message_bus = MessageBus()

        # Initialize autonomous analysis pipeline services
        # Each subscribes to events and processes detection stages independently
        self.heuristic_service = HeuristicAnalysisService(self.detector, self.message_bus)
        self.memory_service = AttackMemoryService(self.detector, self.message_bus)
        self.token_service = HoneyTokenService(self.detector, self.message_bus)

        # ContextEvaluatorAgent already subscribes to TokenNoMatchEvent
        # via _setup_message_handlers() → _handle_pipeline_no_match()

        # Subscribe to token rotation requests from TokenDesignerAgent
        self.message_bus.subscribe("*", "TokenRotationRequestEvent", self._handle_token_rotation)

        # Event-driven mode flag (True = event pipeline, False = synchronous fallback)
        self.use_event_driven = getattr(config, "use_event_driven", True)

    async def initialize_system(self) -> None:
        """Set up initial honey-prompts for detection."""
        try:
            honey_prompt = await self.token_designer.design_token(self.config.system_context)
            if honey_prompt:
                self.honey_prompts.append(honey_prompt)
                logger.info(f"Initialized system with honey-prompt: {honey_prompt.token_hash}")
            else:
                raise RuntimeError("Failed to generate initial honey-prompt")
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise

    async def monitor_text(self, text: str) -> Dict[str, Any]:
        """
        Monitor text for potential prompt injection attacks.

        Uses the event-driven autonomous agent pipeline:
          1. Emit TextReceivedEvent on the MessageBus
          2. Agents independently process stages (heuristic → memory → token → LLM)
          3. Await FinalDetectionEvent with timeout
          4. Fallback to synchronous detector if event pipeline times out
        """
        original_text = text
        text = self._decode_base64_if_needed(text)

        if original_text != text:
            logger.info("Base64 input detected and decoded.")

        if self.use_event_driven:
            return await self._monitor_text_event_driven(text, original_text)
        else:
            return await self._monitor_text_synchronous(text, original_text)

    async def _monitor_text_event_driven(self, text: str, original_text: str) -> Dict[str, Any]:
        """Event-driven pipeline: agents autonomously process detection stages."""
        request_id = str(uuid.uuid4())

        # Emit TextReceivedEvent — agents subscribe and react independently
        event = TextReceivedEvent(
            request_id=request_id,
            source_agent="Orchestrator",
            text=text,
            honey_prompts=self.honey_prompts,
            context_window_size=self.config.context_window_size,
            source="input",
        )
        self.message_bus.publish_event(event)

        # Await FinalDetectionEvent from whichever agent resolves first
        final_msg = await self.message_bus.await_event(
            "FinalDetectionEvent", request_id, timeout=5.0
        )

        if final_msg:
            payload = final_msg.payload
            result = {
                "detection": payload.get("detection", False),
                "confidence": payload.get("confidence", 0.0),
                "explanation": payload.get("explanation", ""),
                "risk_level": payload.get("risk_level", "low"),
                "was_base64_encoded": original_text != text,
                "match_type": payload.get("method", "event_pipeline"),
                "context": text,
                "stage_timings": payload.get("stage_timings", {}),
                "source_agent": payload.get("source_agent", ""),
            }

            # Notify agents for learning
            self._notify_agents_of_detection(result, None, text)
            return result

        # Timeout — fall back to synchronous path
        logger.warning("Event pipeline timed out — falling back to synchronous detection")
        return await self._monitor_text_synchronous(text, original_text)

    async def _monitor_text_synchronous(self, text: str, original_text: str) -> Dict[str, Any]:
        """Synchronous fallback: direct detector + context evaluator calls."""
        for honey_prompt in self.honey_prompts:
            detection_result = self.detector.analyze_text(text, honey_prompt, self.config.context_window_size)
            if detection_result["matched"]:
                evaluation = await self.context_evaluator.evaluate_detection(
                    text=text,
                    token=honey_prompt.base_token,
                    surrounding_context=detection_result.get("context", ""),
                    expected_context=honey_prompt.context,
                )
                if evaluation.get("is_attack"):
                    result = {
                        "detection": True,
                        "confidence": evaluation.get("confidence", detection_result["confidence"]),
                        "explanation": evaluation.get("explanation", ""),
                        "risk_level": evaluation.get("risk_level", "high"),
                        "token_hash": honey_prompt.token_hash,
                        "was_base64_encoded": original_text != text,
                        "match_type": "honey_prompt_match",
                        "context": detection_result.get("context", text),
                    }
                    self._notify_agents_of_detection(result, honey_prompt, text)
                    return result

        # Fallback scenario
        evaluation = await self.context_evaluator.evaluate_detection(
            text=text, token="(no_token)", surrounding_context=text, expected_context=""
        )
        result = {
            "detection": evaluation.get("is_attack", False),
            "confidence": evaluation.get("confidence", 0.0),
            "explanation": evaluation.get("explanation", ""),
            "risk_level": evaluation.get("risk_level", "low"),
            "was_base64_encoded": original_text != text,
            "match_type": "contextual_evaluation",
            "context": text,
        }
        self._notify_agents_of_detection(result, None, text)
        return result

    async def monitor_output(self, response: str) -> Dict[str, Any]:
        """
        Monitor LLM output for potential honey token leaks.
        Decodes Base64 input if detected before analysis.
        Uses Stage 3 (semantic analysis) only for output detection.
        """
        original_response = response
        response = self._decode_base64_if_needed(response)

        if original_response != response:
            logger.info("Base64 input detected in output and decoded.")

        # Iterate over honey prompts and check for leaks in output
        for honey_prompt in self.honey_prompts:
            detection_result = self.detector.analyze_text(
                response,
                honey_prompt,
                self.config.context_window_size,
                skip_heuristics=True,
                skip_memory=True,
            )
            if detection_result["matched"]:
                logger.warning("ALERT: Honey token leaked in model output!")
                result = {
                    "detection": True,
                    "confidence": detection_result.get("confidence", 0.85),
                    "explanation": "Honey token detected in LLM output - potential information leak",
                    "risk_level": "high",
                    "token_hash": honey_prompt.token_hash,
                    "source": "output",
                    "match_type": "output_token_leak",
                    "was_base64_encoded": original_response != response,
                    "context": detection_result.get("context", response),
                }

                # Notify agents of output leak detection
                self._notify_agents_of_detection(result, honey_prompt, response)

                return result

        # No token leak detected in output
        result = {
            "detection": False,
            "confidence": 0.0,
            "source": "output",
            "match_type": "no_leak",
            "was_base64_encoded": original_response != response,
        }

        return result

    def _decode_base64_if_needed(self, text: str) -> str:
        try:
            if self._looks_like_base64(text):
                decoded_bytes = base64.b64decode(text, validate=True)
                return decoded_bytes.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
        return text

    def _looks_like_base64(self, text: str) -> bool:
        import re

        return bool(re.match(r"^[A-Za-z0-9+/]+={0,2}$", text))

    async def sanitize_and_monitor_text(
        self, external_inputs: List[str], expected_detection: List[bool]
    ) -> List[Dict[str, Any]]:
        if not self.honey_prompts:
            await self.initialize_system()

        honey_tokens = [hp.base_token for hp in self.honey_prompts]
        embedded_inputs = self.environment_agent.embed_environment_tokens(
            inputs=external_inputs, honey_tokens=honey_tokens
        )

        indirect_injections = self.environment_agent.detect_indirect_injections(
            inputs=embedded_inputs, honey_tokens=honey_tokens, threshold=0.85
        )

        final_inputs = []
        for input_text, injection_detected in zip(embedded_inputs, indirect_injections):
            if injection_detected:
                logger.warning(f"Indirect injection detected: {input_text}")
                continue
            sanitized_input = await self.environment_agent.sanitize_external_inputs(
                external_inputs=[input_text],  # Single-element list
                honey_prompts=self.honey_prompts,
                threshold=self.detector.current_threshold,
            )
            final_inputs.extend(sanitized_input)  # Use extend to flatten list

        results = []
        for text, expected in zip(final_inputs, expected_detection):
            result = await self.monitor_text(text)
            results.append(result)
            self.self_tuner.update_metrics(result, expected)

        # Adjust threshold based on accumulated metrics
        self.current_threshold = self.self_tuner.adjust_threshold_pseudo_labels()
        logger.info(f"Adjusted threshold to {self.current_threshold}")

        return results

    def _extract_context(self, text: str, token: str, window_size: int) -> str:
        start_idx = text.find(token)
        if start_idx == -1:
            return ""
        context_start = max(0, start_idx - window_size)
        context_end = min(len(text), start_idx + len(token) + window_size)
        return text[context_start:context_end]

    def _handle_token_rotation(self, message) -> None:
        """
        Handle autonomous token rotation requests from TokenDesignerAgent.

        When the TokenDesignerAgent detects degraded token performance (age,
        low success rate, high FP rate), it emits a TokenRotationRequestEvent.
        The Orchestrator coordinates the actual swap.
        """
        payload = message.payload
        reason = payload.get("reason", "unknown")
        new_token_data = payload.get("new_token", {})

        logger.info(f"Orchestrator: Token rotation requested — reason: {reason}")

        # The actual rotation is initiated by the proactive_action loop in TokenDesigner
        # Here we just log and could trigger a re-initialization if needed

    def _notify_agents_of_detection(self, result: Dict[str, Any], honey_prompt: Any, text: str) -> None:
        """
        Notify agents of detection results so they can learn autonomously.

        This enables:
        - TokenDesigner to track token performance
        - ContextEvaluator to learn patterns
        - Environment to adapt embedding strategies
        """

        # Notify TokenDesigner about token performance
        if honey_prompt:
            self.token_designer.send_message(
                recipient="TokenDesigner",
                message_type="detection_feedback",
                payload={
                    "token_hash": honey_prompt.token_hash,
                    "detection_successful": result["detection"],
                    "confidence": result["confidence"],
                    "risk_level": result.get("risk_level", "unknown"),
                    "false_positive": False,  # Would need ground truth to know
                },
            )

        # Notify ContextEvaluator for pattern learning
        self.context_evaluator.send_message(
            recipient="ContextEvaluator",
            message_type="detection_result",
            payload={"text": text, "result": result, "expected_result": None},  # Would need ground truth
        )

        # Notify Environment agent about embedding effectiveness
        if honey_prompt:
            self.environment_agent.send_message(
                recipient="Environment",
                message_type="embedding_feedback",
                payload={
                    "strategy": getattr(self.environment_agent, "current_strategy", "unknown"),
                    "detected": result["detection"],
                    "confidence": result["confidence"],
                },
            )
