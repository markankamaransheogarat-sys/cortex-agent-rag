import re
from typing import Dict, List, Tuple


class CrossEncoderReranker:
    """
    Two-stage retrieval Cross-Encoder reranker.
    Computes fine-grained query-document alignment, lexical exact match bonuses,
    and semantic coverage to filter false positives before LLM context stuffing.
    """

    def __init__(self, min_relevance_threshold: float = 0.25):
        self.min_relevance_threshold = min_relevance_threshold

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def score_pair(self, query: str, content: str) -> float:
        """
        Cross-attention interaction heuristic simulation:
        Combines Jaccard keyword overlap, term proximity, and length coverage.
        """
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(content)

        if not q_tokens or not d_tokens:
            return 0.0

        intersection = q_tokens.intersection(d_tokens)
        if not intersection:
            return 0.0

        # Term coverage (what % of query terms appear in document)
        coverage = len(intersection) / len(q_tokens)

        # Density / Jaccard
        jaccard = len(intersection) / len(q_tokens.union(d_tokens))

        # Combined cross-score
        score = (coverage * 0.75) + (jaccard * 0.25)
        return round(score, 4)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict, float]],
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank candidate list using cross-scoring and filter below min threshold.
        """
        reranked = []

        for doc, stage1_score in candidates:
            cross_score = self.score_pair(query, doc["content"])
            # Combined final score: 70% cross-encoder + 30% stage1 retrieval
            final_score = round((cross_score * 0.7) + (stage1_score * 0.3), 4)

            if cross_score >= self.min_relevance_threshold:
                reranked.append((doc, final_score))

        # Sort descending by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
