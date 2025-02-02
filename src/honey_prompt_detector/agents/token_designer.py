# src/honey_prompt_detector/agents/token_designer.py
import json
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from ..core.honey_prompt import HoneyPrompt
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class TokenDesignerAgent:
    """
    TokenDesignerAgent is responsible for creating honey-prompt tokens for detecting
    potential prompt injection attacks in AI systems. The tokens are designed to be
    unique yet plausible in system instructions, with detailed detection rules and
    variations.

    This class provides methods to interact with OpenAI API for token generation,
    process the response, and produce a structured HoneyPrompt object as the output.

    :ivar client: An instance of AsyncOpenAI client initialized with the provided API key.
    :type client: AsyncOpenAI
    :ivar model_name: Name of the OpenAI model used for token generation.
    :type model_name: str
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        """
        Initialize the token designer agent.

        Args:
            api_key: OpenAI API key
            model_name: Name of the model to use for token generation
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name

    async def design_token(self, system_context: str) -> Optional[HoneyPrompt]:
        """
        Generate a new honey-prompt token suitable for the given context.

        Args:
            system_context: Description of the system's purpose and behavior

        Returns:
            HoneyPrompt if generation successful, None otherwise
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": self._create_design_prompt(system_context)}
                ],
                temperature=0.2
            )

            token_design = json.loads(response.choices[0].message.content)
            return self._create_honey_prompt(token_design, system_context)

        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            # Fallback option: use a default token if desired.
            default_token_design = {
                "base_token": "default_honey_token",
                "variations": ["default_honey_token"],
                "detection_rules": {
                    "exact_match_weight": 1.0,
                    "variation_match_weight": 0.8,
                    "context_importance": 0.7,
                    "minimum_confidence": 0.6
                },
                "category": "direct_injection",
                "sensitivity": 0.9,
                "expected_context": "default expected usage context"
            }
            return self._create_honey_prompt(default_token_design, system_context)

    def _get_system_prompt(self) -> str:
        """Create the system prompt for the token designer."""
        return """You are a security expert specialized in designing honey-prompt 
        tokens for detecting prompt injection attacks in AI systems. Your tokens 
        should be unique enough to avoid false positives but natural enough to 
        appear legitimate in system instructions."""

    def _create_design_prompt(self, system_context: str) -> str:
        """Create the design prompt for token generation."""
        return f"""
        Design a honey-prompt token for detecting prompt injection attacks.

        System Context: {system_context}

        Requirements:
        1. Token should be unique but plausible in system instructions
        2. Include multiple variations for fuzzy matching
        3. Specify detection rules and expected contexts

        Format response as JSON with:
        {{
            "base_token": "primary token string",
            "variations": ["list", "of", "acceptable", "variations"],
            "detection_rules": {{
                "exact_match_weight": 1.0,
                "variation_match_weight": 0.8,
                "context_importance": 0.7,
                "minimum_confidence": 0.6
            }},
            "category": "category name",
            "sensitivity": 0.9,
            "expected_context": "description of normal usage context"
        }}
        """

    def _create_honey_prompt(self, token_design: Dict[str, Any], system_context: str) -> HoneyPrompt:
        # Optionally, augment the variations list with additional obfuscations
        base_token = token_design['base_token']
        additional_variations = [
            " ".join(list(base_token)),  # insert spaces between characters
            ".".join(list(base_token)),  # insert dots between characters
            base_token.upper(),  # all uppercase
            base_token.lower()  # all lowercase
        ]
        # Merge the provided variations with the additional ones
        all_variations = list(set(token_design.get('variations', []) + additional_variations))

        return HoneyPrompt(
            base_token=base_token,
            category=token_design['category'],
            sensitivity=token_design['sensitivity'],
            context=token_design['expected_context'],
            variations=all_variations,
            detection_rules=token_design['detection_rules']
        )