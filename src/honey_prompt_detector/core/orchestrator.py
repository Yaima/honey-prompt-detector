import base64
from typing import Dict, Any, List
import logging

from ..agents.environment_agent import EnvironmentAgent
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
    attacks. It initializes "honey prompts" that are used as detection markers
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
        self.detector = Detector()  # Expose the Detector instance
        self.environment_agent = EnvironmentAgent(similarity_model_name=self.config.similarity_model_name)

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

        # Proceed with existing honey token matching logic
        for honey_prompt in self.honey_prompts:
            match_result = honey_prompt.matches_text(text)
            if match_result['matched']:
                context_window = self._extract_context(
                    text, honey_prompt.base_token, self.config.context_window_size)
                evaluation = await self.context_evaluator.evaluate_detection(
                    text=text,
                    token=honey_prompt.base_token,
                    surrounding_context=context_window,
                    expected_context=honey_prompt.context
                )
                if evaluation.get('is_attack'):
                    return {
                        'detection': True,
                        'confidence': evaluation.get('confidence', 0.9),
                        'explanation': evaluation.get('explanation', ''),
                        'risk_level': evaluation.get('risk_level', 'high'),
                        'token_hash': honey_prompt.token_hash,
                        'was_base64_encoded': original_text != text
                    }

        # Fallback: Evaluate entire text
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
            'was_base64_encoded': original_text != text
        }

    def _decode_base64_if_needed(self, text: str) -> str:
        try:
            # Only attempt decoding if it looks Base64-encoded
            if self._looks_like_base64(text):
                decoded_bytes = base64.b64decode(text, validate=True)
                return decoded_bytes.decode('utf-8')
        except (base64.binascii.Error, UnicodeDecodeError, ValueError):
            pass  # If decoding fails, simply return original text
        return text

    def _looks_like_base64(self, text: str) -> bool:
        # A simple check for Base64 encoding validity
        import re
        return bool(re.match(r'^[A-Za-z0-9+/]+={0,2}$', text))

    async def sanitize_and_monitor_text(self, external_inputs: List[str]) -> List[Dict[str, Any]]:
        """
        Integrates the EnvironmentAgent directly for pre-processing inputs
        before running through LLM and context evaluation.
        """
        # Initialize if needed
        if not hasattr(self, 'honey_prompts') or not self.honey_prompts:
            await self.initialize_system()

        # Extract tokens for embedding
        honey_tokens = [hp.base_token for hp in self.honey_prompts]

        # Use EnvironmentAgent to embed and detect early injections
        embedded_and_sanitized_inputs = self.environment_agent.embed_environment_tokens(
            inputs=external_inputs, honey_tokens=honey_tokens
        )

        indirect_injections = self.environment_agent.detect_indirect_injections(
            inputs=embedded_and_sanitized_inputs,
            honey_tokens=honey_tokens,
            threshold=0.85
        )

        final_inputs = []
        for input_text, injection_detected in zip(embedded_and_sanitized_inputs, indirect_injections):
            if injection_detected:
                logger.warning(f"Indirect injection detected by EnvironmentAgent: {input_text}")
                await self.alert_manager.send_alert({
                    'detection': True,
                    'confidence': 1.0,
                    'explanation': "Indirect injection detected early by EnvironmentAgent",
                    'risk_level': 'high'
                })
                continue
            final_sanitized_input = self.output_sanitizer.sanitize(input_text)
            final_inputs.append(final_sanitized_input)  # Corrected, use final_inputs

        return [await self.monitor_text(text) for text in final_inputs]

    def _extract_context(self, text: str, token: str, window_size: int) -> str:
        """Extract text surrounding a token match."""
        start_idx = text.find(token)
        if start_idx == -1:
            return ""
        context_start = max(0, start_idx - window_size)
        context_end = min(len(text), start_idx + len(token) + window_size)
        return text[context_start:context_end]
