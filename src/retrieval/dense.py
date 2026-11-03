import math
from typing import Dict, List, Tuple


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two numeric vectors."""
    if len(vec_a) != len(vec_b) or len(vec_a) == 0:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class DenseVectorStore:
    """
    In-memory dense vector store supporting high-speed cosine similarity lookup.
    """

    def __init__(self):
        self.documents: List[Dict] = []
        self.embeddings: List[List[float]] = []

    def add(self, document: Dict, embedding: List[float]):
        self.documents.append(document)
        self.embeddings.append(embedding)

    def add_many(self, documents: List[Dict], embeddings: List[List[float]]):
        for doc, emb in zip(documents, embeddings):
            self.add(doc, emb)

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Find top_k nearest neighbors by cosine similarity."""
        if not self.embeddings:
            return []

        scored = []
        for i, emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, emb)
            scored.append((self.documents[i], round(sim, 4)))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self.documents)
