from typing import Dict, List, Tuple


class HybridFusion:
    """
    Reciprocal Rank Fusion (RRF) for merging multi-modal retrieval streams
    (Dense semantic embeddings + Sparse lexical BM25).
    """

    @staticmethod
    def rrf(
        dense_results: List[Tuple[Dict, float]],
        sparse_results: List[Tuple[Dict, float]],
        k: int = 60,
        top_k: int = 10,
    ) -> List[Tuple[Dict, float]]:
        """
        Merge two ranked lists using standard RRF:
        RRF_Score(d) = sum(1.0 / (k + rank_i(d)))
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        # 1. Score Dense Ranks
        for rank, (doc, _) in enumerate(dense_results, start=1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # 2. Score Sparse (BM25) Ranks
        for rank, (doc, _) in enumerate(sparse_results, start=1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # 3. Sort by aggregated RRF score descending
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        merged = []
        for doc_id, score in ranked[:top_k]:
            merged.append((doc_map[doc_id], round(score, 6)))

        return merged
