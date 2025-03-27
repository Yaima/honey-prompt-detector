import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger("honey_prompt")

class LocalClassifierAgent:
    """
    Quickly classifies text as malicious (prompt injection) vs benign.
    Fine-tune or adapt a model for your domain if possible.
    """

    def __init__(self, model_name="microsoft/deberta-v3-base", threshold=0.7):
        self.model_name = model_name
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-base",
            cache_dir="./../my_deberta_cache",
            use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base",
            cache_dir="./../my_deberta_cache"
        )
        self.model.eval()

        logger.info(f"[LocalClassifierAgent] Loaded model: {model_name}, threshold={threshold}")

    def is_malicious(self, text: str) -> bool:
        """
        Returns True if the local model thinks 'text' is malicious,
        otherwise False.
        """
        logger.info(f"[LocalClassifierAgent] Checking text: {text[:30]}")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Suppose model is a 2-class classifier: [benign, malicious]
            logits = outputs.logits.squeeze()
            probs = torch.softmax(logits, dim=-1)
            malicious_prob = probs[1].item()

        logger.debug(f"[LocalClassifierAgent] text='{text[:30]}...', malicious_prob={malicious_prob:.3f}")
        return malicious_prob > self.threshold
