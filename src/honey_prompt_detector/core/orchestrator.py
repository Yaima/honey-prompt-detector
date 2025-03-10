import base64
import binascii
from typing import Dict, Any, List
import logging

from ..agents.environment_agent import EnvironmentAgent
from src.honey_prompt_detector.core.self_tuner import SelfTunerAgent
from ..agents.token_designer_agent import TokenDesignerAgent
from ..agents.context_evaluator_agent import ContextEvaluatorAgent
from ..core.honey_prompt import HoneyPrompt
from ..core.detector import Detector  # Import the Detector

logger = logging.getLogger(__name__)


class DetectionOrchestrator:
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

    def __init__(
            self,
            token_designer: TokenDesignerAgent,
            context_evaluator: ContextEvaluatorAgent,
            config: Any
    ):
        self.token_designer = token_designer
        self.context_evaluator = context_evaluator
        self.config = config
        self.honey_prompts: List[HoneyPrompt] = []
        self.detector = Detector(context_evaluator=self.context_evaluator)
        self.environment_agent = EnvironmentAgent(similarity_model_name=self.config.similarity_model_name)
        self.self_tuner = SelfTunerAgent(detector_agent=self.detector, config=config)
        self.current_threshold = config.initial_threshold

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
        Decodes Base64 input if detected before analysis.
        """
        original_text = text
        text = self._decode_base64_if_needed(text)

        if original_text != text:
            logger.info("Base64 input detected and decoded.")

        # Leverage detector logic comprehensively
        for honey_prompt in self.honey_prompts:
            detection_result = self.detector.analyze_text(text, honey_prompt, self.config.context_window_size)
            if detection_result['matched']:
                evaluation = await self.context_evaluator.evaluate_detection(
                    text=text,
                    token=honey_prompt.base_token,
                    surrounding_context=detection_result.get('context', ''),
                    expected_context=honey_prompt.context
                )
                if evaluation.get('is_attack'):
                    return {
                        'detection': True,
                        'confidence': evaluation.get('confidence', detection_result['confidence']),
                        'explanation': evaluation.get('explanation', ''),
                        'risk_level': evaluation.get('risk_level', 'high'),
                        'token_hash': honey_prompt.token_hash,
                        'was_base64_encoded': original_text != text,
                        'match_type': 'honey_prompt_match',  # explicitly set
                        'context': detection_result.get('context', text)  # provide fallback if no context
                    }

        # Fallback scenario:
        evaluation = await self.context_evaluator.evaluate_detection(
            text=text,
            token="(no_token)",
            surrounding_context=text,
            expected_context=""
        )
        return {
            'detection': evaluation.get('is_attack', False),
            'confidence': evaluation.get('confidence', 0.0),
            'explanation': evaluation.get('explanation', ''),
            'risk_level': evaluation.get('risk_level', 'low'),
            'was_base64_encoded': original_text != text,
            'match_type': 'contextual_evaluation',  # explicitly set
            'context': text
        }

    def _decode_base64_if_needed(self, text: str) -> str:
        try:
            if self._looks_like_base64(text):
                decoded_bytes = base64.b64decode(text, validate=True)
                return decoded_bytes.decode('utf-8')
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
        return text

    def _looks_like_base64(self, text: str) -> bool:
        import re
        return bool(re.match(r'^[A-Za-z0-9+/]+={0,2}$', text))

    async def sanitize_and_monitor_text(self, external_inputs: List[str], expected_detection: List[bool]) -> List[Dict[str, Any]]:
        if not self.honey_prompts:
            await self.initialize_system()

        honey_tokens = [hp.base_token for hp in self.honey_prompts]
        embedded_inputs = self.environment_agent.embed_environment_tokens(
            inputs=external_inputs, honey_tokens=honey_tokens
        )

        indirect_injections = self.environment_agent.detect_indirect_injections(
            inputs=embedded_inputs,
            honey_tokens=honey_tokens,
            threshold=0.85
        )

        final_inputs = []
        for input_text, injection_detected in zip(embedded_inputs, indirect_injections):
            if injection_detected:
                logger.warning(f"Indirect injection detected: {input_text}")
                continue
            sanitized_input = await self.environment_agent.sanitize_external_inputs(
                external_inputs=[input_text],  # Single-element list
                honey_prompts=self.honey_prompts,
                threshold=self.detector.current_threshold
            )
            final_inputs.extend(sanitized_input)  # Use extend to flatten list

        results = []
        for text, expected in zip(final_inputs, expected_detection):
            result = await self.monitor_text(text)
            results.append(result)
            self.self_tuner.update_metrics(result, expected)

        # Adjust threshold based on accumulated metrics
        self.current_threshold = self.self_tuner.adjust_threshold_if_needed()
        logger.info(f"Adjusted threshold to {self.current_threshold}")

        return results

    def _extract_context(self, text: str, token: str, window_size: int) -> str:
        start_idx = text.find(token)
        if start_idx == -1:
            return ""
        context_start = max(0, start_idx - window_size)
        context_end = min(len(text), start_idx + len(token) + window_size)
        return text[context_start:context_end]

