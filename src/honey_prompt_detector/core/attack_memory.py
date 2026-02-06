"""
VectorDB Attack Memory - Lightweight Vector-Based Attack Pattern Storage

Stores embeddings of detected attacks and finds similar patterns using
cosine similarity. Provides defense-in-depth by recognizing variations
of previously seen attacks.

Features:
- In-memory vector storage (no external DB required)
- Cosine similarity search for attack pattern matching
- Persistence to disk for cross-session learning
- Configurable similarity thresholds
- Attack categorization and metadata tracking
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("honey_prompt")


@dataclass
class AttackRecord:
    """Record of a detected attack for memory storage."""

    text: str
    embedding: List[float]  # Stored as list for JSON serialization
    category: str
    confidence: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackRecord":
        return cls(**data)


@dataclass
class SimilarityMatch:
    """Result of a similarity search."""

    matched: bool
    similarity: float
    record: Optional[AttackRecord] = None
    category: Optional[str] = None


class AttackMemory:
    """
    Vector-based attack pattern memory using cosine similarity.

    Stores embeddings of detected attacks and can find similar patterns
    in new inputs. This provides a "memory" of attacks that helps detect
    variations of previously seen injection attempts.
    """

    def __init__(
        self,
        embedding_model,
        similarity_threshold: float = 0.85,
        max_records: int = 10000,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize attack memory.

        Args:
            embedding_model: SentenceTransformer model for encoding text
            similarity_threshold: Minimum cosine similarity to consider a match
            max_records: Maximum number of records to store (FIFO eviction)
            persistence_path: Path to save/load memory (optional)
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_records = max_records
        self.persistence_path = persistence_path

        # Storage
        self.records: List[AttackRecord] = []
        self._embedding_matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True

        # Statistics
        self.total_attacks_stored = 0
        self.total_queries = 0
        self.total_matches = 0

        # Load from persistence if available
        if persistence_path and persistence_path.exists():
            self._load_from_disk()

    def add_attack(
        self, text: str, category: str, confidence: float, metadata: Optional[Dict[str, Any]] = None
    ) -> AttackRecord:
        """
        Add a detected attack to memory.

        Args:
            text: The attack text
            category: Attack category (e.g., 'direct_injection', 'obfuscated')
            confidence: Detection confidence score
            metadata: Additional metadata about the attack

        Returns:
            The created AttackRecord
        """
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)

        record = AttackRecord(
            text=text[:500],  # Truncate for storage efficiency
            embedding=embedding.tolist(),
            category=category,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        # Add to storage
        self.records.append(record)
        self.total_attacks_stored += 1
        self._matrix_dirty = True

        # Evict oldest if over limit
        if len(self.records) > self.max_records:
            self.records.pop(0)
            logger.debug(f"AttackMemory: Evicted oldest record (limit: {self.max_records})")

        logger.info(f"AttackMemory: Stored attack [{category}] (total: {len(self.records)})")

        # Persist if configured
        if self.persistence_path:
            self._save_to_disk()

        return record

    def find_similar(self, text: str, threshold: Optional[float] = None) -> SimilarityMatch:
        """
        Find similar attacks in memory.

        Args:
            text: Text to search for
            threshold: Override default similarity threshold

        Returns:
            SimilarityMatch with results
        """
        self.total_queries += 1

        if not self.records:
            return SimilarityMatch(matched=False, similarity=0.0)

        threshold = threshold or self.similarity_threshold

        # Generate query embedding
        query_embedding = self.embedding_model.encode(text, convert_to_numpy=True)

        # Build/update embedding matrix
        if self._matrix_dirty or self._embedding_matrix is None:
            self._rebuild_matrix()

        # Compute cosine similarities
        similarities = self._cosine_similarity(query_embedding, self._embedding_matrix)

        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_idx])

        if best_similarity >= threshold:
            self.total_matches += 1
            record = self.records[best_idx]
            logger.info(
                f"AttackMemory: Found similar attack (similarity: {best_similarity:.3f}, "
                f"category: {record.category})"
            )
            return SimilarityMatch(matched=True, similarity=best_similarity, record=record, category=record.category)

        return SimilarityMatch(matched=False, similarity=best_similarity)

    def find_top_k(self, text: str, k: int = 5, threshold: Optional[float] = None) -> List[Tuple[float, AttackRecord]]:
        """
        Find top-k similar attacks.

        Args:
            text: Text to search for
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (similarity, record) tuples sorted by similarity
        """
        if not self.records:
            return []

        threshold = threshold or 0.0  # No threshold for top-k by default

        # Generate query embedding
        query_embedding = self.embedding_model.encode(text, convert_to_numpy=True)

        # Build/update embedding matrix
        if self._matrix_dirty or self._embedding_matrix is None:
            self._rebuild_matrix()

        # Compute cosine similarities
        similarities = self._cosine_similarity(query_embedding, self._embedding_matrix)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append((sim, self.records[idx]))

        return results

    def get_category_stats(self) -> Dict[str, int]:
        """Get count of attacks by category."""
        stats = {}
        for record in self.records:
            stats[record.category] = stats.get(record.category, 0) + 1
        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_records": len(self.records),
            "total_attacks_stored": self.total_attacks_stored,
            "total_queries": self.total_queries,
            "total_matches": self.total_matches,
            "match_rate": self.total_matches / max(1, self.total_queries),
            "category_distribution": self.get_category_stats(),
            "similarity_threshold": self.similarity_threshold,
            "max_records": self.max_records,
        }

    def clear(self) -> None:
        """Clear all records from memory."""
        self.records = []
        self._embedding_matrix = None
        self._matrix_dirty = True
        logger.info("AttackMemory: Cleared all records")

    def _rebuild_matrix(self) -> None:
        """Rebuild the embedding matrix from records."""
        if not self.records:
            self._embedding_matrix = None
            return

        embeddings = [np.array(r.embedding) for r in self.records]
        self._embedding_matrix = np.vstack(embeddings)
        self._matrix_dirty = False

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all rows in matrix."""
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        # Normalize matrix rows
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        matrix_normalized = matrix / matrix_norms

        # Dot product gives cosine similarity for normalized vectors
        similarities = np.dot(matrix_normalized, query_norm)

        return similarities

    def _save_to_disk(self) -> None:
        """Save memory to disk."""
        if not self.persistence_path:
            return

        data = {
            "records": [r.to_dict() for r in self.records],
            "stats": {
                "total_attacks_stored": self.total_attacks_stored,
                "total_queries": self.total_queries,
                "total_matches": self.total_matches,
            },
        }

        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, "w") as f:
            json.dump(data, f)

        logger.debug(f"AttackMemory: Saved {len(self.records)} records to {self.persistence_path}")

    def _load_from_disk(self) -> None:
        """Load memory from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)

            self.records = [AttackRecord.from_dict(r) for r in data.get("records", [])]
            stats = data.get("stats", {})
            self.total_attacks_stored = stats.get("total_attacks_stored", len(self.records))
            self.total_queries = stats.get("total_queries", 0)
            self.total_matches = stats.get("total_matches", 0)
            self._matrix_dirty = True

            logger.info(f"AttackMemory: Loaded {len(self.records)} records from {self.persistence_path}")
        except Exception as e:
            logger.error(f"AttackMemory: Failed to load from disk: {e}")
