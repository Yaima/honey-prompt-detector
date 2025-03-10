import asyncio
import json
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from ..core.honey_prompt import HoneyPrompt
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class TokenDesignerAgent:
    """
    Uses GPT models via OpenAI API for designing honey-prompt tokens.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4", timeout: int = 60):
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.timeout = timeout

    async def design_token(self, system_context: str, max_retries: int = 3) -> Optional[HoneyPrompt]:
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": self._create_design_prompt(system_context)}
                        ],
                        temperature=0.2
                    ),
                    timeout=self.timeout
                )

                content = response.choices[0].message.content

                try:
                    token_design = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in TokenDesignerAgent: {str(e)}. GPT response: '{content}'")
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue  # Retry on JSON errors specifically

                return self._create_honey_prompt(token_design, system_context)

            except asyncio.TimeoutError:
                retry_count += 1
                logger.warning(f"OpenAI API timeout. Retry {retry_count}/{max_retries}")
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Unexpected error during token generation: {e}")
                return self._default_honey_prompt(system_context)

        logger.error("All retries exhausted for token generation.")
        return self._default_honey_prompt(system_context)

    def _default_honey_prompt(self, system_context: str) -> HoneyPrompt:
        logger.info("Returning default honey prompt due to failures.")
        return HoneyPrompt(
            base_token="default_honey_token",
            variations=["default_honey_token"],
            detection_rules={
                "exact_match_weight": 1.0,
                "variation_match_weight": 0.8,
                "context_importance": 0.7,
                "minimum_confidence": 0.6
            },
            category="direct_injection",
            sensitivity=0.9,
            context=system_context
        )

    def _get_system_prompt(self) -> str:
        return (
            "You are a security expert specialized in designing honey-prompt tokens for detecting prompt injection attacks."
            " Tokens should be unique enough to avoid false positives but natural enough to appear legitimate."
        )

    def _create_design_prompt(self, system_context: str) -> str:
        return (f"Design a honey-prompt token for the following context:\n\n"
                f"{system_context}\n\n"
                "Provide JSON:\n"
                "{\n"
                "  \"base_token\": \"string\",\n"
                "  \"variations\": [\"string1\", \"string2\"],\n"
                "  \"detection_rules\": {\n"
                "    \"exact_match_weight\": float,\n"
                "    \"variation_match_weight\": float,\n"
                "    \"context_importance\": float,\n"
                "    \"minimum_confidence\": float\n"
                "  },\n"
                "  \"category\": \"string\",\n"
                "  \"sensitivity\": float,\n"
                "  \"expected_context\": \"string\"\n"
                "}")

    def _create_honey_prompt(self, token_design: Dict[str, Any], system_context: str) -> HoneyPrompt:
        base_token = token_design['base_token']
        additional_variations = [
            " ".join(base_token),
            ".".join(base_token),
            base_token.upper(),
            base_token.lower()
        ]
        all_variations = list(set(token_design.get('variations', []) + additional_variations))

        return HoneyPrompt(
            base_token=base_token,
            variations=all_variations,
            detection_rules=token_design['detection_rules'],
            category=token_design.get('category', 'direct_injection'),
            sensitivity=token_design.get('sensitivity', 0.9),
            context=token_design.get('expected_context', system_context)
        )