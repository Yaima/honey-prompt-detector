from typing import Dict, Any, List
import logging
from ..agents.token_designer import TokenDesignerAgent
from ..agents.context_evaluator import ContextEvaluatorAgent
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

        1) Check if text contains any honey prompt token.
        2) If no token is matched, use the context evaluator on the entire text.
        """
        # 1) Check for honey token matches
        for honey_prompt in self.honey_prompts:
            match_result = honey_prompt.matches_text(text)
            if match_result['matched']:
                context_window = self._extract_context(text, honey_prompt.base_token, self.config.context_window_size)
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
                        'token_hash': honey_prompt.token_hash
                    }
        # 2) Fallback: Evaluate the entire text
        evaluation = await self.context_evaluator.evaluate_detection(
            text=text,
            token="(no_token)",
            surrounding_context=text,
            expected_context=""
        )
        if evaluation.get('is_attack'):
            return {
                'detection': True,
                'confidence': evaluation.get('confidence', 0.9),
                'explanation': evaluation.get('explanation', ''),
                'risk_level': evaluation.get('risk_level', 'high'),
            }
        return {'detection': False, 'confidence': evaluation.get('confidence', 0.0)}

    def _extract_context(self, text: str, token: str, window_size: int) -> str:
        """Extract text surrounding a token match."""
        start_idx = text.find(token)
        if start_idx == -1:
            return ""
        context_start = max(0, start_idx - window_size)
        context_end = min(len(text), start_idx + len(token) + window_size)
        return text[context_start:context_end]
