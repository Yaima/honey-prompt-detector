from sentence_transformers import SentenceTransformer, util
import logging
from typing import List
from ..core.honey_prompt import HoneyPrompt

logger = logging.getLogger(__name__)

class EnvironmentAgent:
    """
    Uses SentenceTransformer models for semantic embedding and similarity checks.
    """
    def __init__(self, similarity_model_name: str = 'all-MiniLM-L6-v2'):
        self.similarity_model = SentenceTransformer(similarity_model_name)

    def embed_environment_tokens(self, inputs: List[str], honey_tokens: List[str]) -> List[str]:
        return [
            f"{honey_tokens[0]} {input_text}"  # Consistent embedding of one token for simplicity
            for input_text in inputs
        ]

    def detect_indirect_injections(
            self,
            inputs: List[str],
            honey_tokens: List[str],
            threshold: float = 0.9
    ) -> List[bool]:
        detections = []
        token_embeddings = self.similarity_model.encode(honey_tokens, convert_to_tensor=True)
        input_embeddings = self.similarity_model.encode(inputs, convert_to_tensor=True)

        # Correctly iterate over input embeddings
        for input_emb in input_embeddings:
            similarity_tensor = util.pytorch_cos_sim(input_emb, token_embeddings)
            similarity = similarity_tensor.max().item()  # Extract maximum similarity value directly
            detected = similarity >= threshold
            detections.append(detected)

            if detected:
                logger.info(f"Indirect injection detected (similarity={similarity:.2f})")

        return detections

    async def sanitize_external_inputs(
        self, external_inputs: List[str], honey_prompts: List[HoneyPrompt], threshold: float = 0.9
    ) -> List[str]:
        honey_tokens = [hp.base_token for hp in honey_prompts]

        indirect_detections = self.detect_indirect_injections(
            inputs=external_inputs, honey_tokens=honey_tokens, threshold=threshold
        )

        sanitized_inputs = []
        for input_text, injection_detected in zip(external_inputs, indirect_detections):
            if injection_detected:
                logger.warning(f"Indirect injection detected early: {input_text}")
                continue
            sanitized_inputs.append(input_text)

        return sanitized_inputs
