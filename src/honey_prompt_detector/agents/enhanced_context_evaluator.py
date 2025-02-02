from sentence_transformers import SentenceTransformer, util

class EnhancedContextEvaluator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def evaluate_similarity(self, observed_context: str, expected_context: str) -> float:
        emb_obs = self.model.encode(observed_context, convert_to_tensor=True)
        emb_exp = self.model.encode(expected_context, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_obs, emb_exp).item()
        return similarity

    def adjust_confidence(self, base_confidence: float, observed_context: str, expected_context: str) -> float:
        similarity = self.evaluate_similarity(observed_context, expected_context)
        adjustment = 1 - similarity
        return min(1.0, base_confidence * (1 + adjustment))
