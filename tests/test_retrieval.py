import unittest
from src.retrieval.bm25 import BM25Okapi
from src.retrieval.dense import DenseVectorStore, cosine_similarity
from src.retrieval.hybrid import HybridFusion
from src.retrieval.reranker import CrossEncoderReranker


class TestRetrieval(unittest.TestCase):
    def test_bm25_ranking(self):
        docs = [
            {"id": "doc1", "content": "PostgreSQL database indexing and query tuning"},
            {"id": "doc2", "content": "ClickHouse columnar analytical storage engine"},
            {"id": "doc3", "content": "Machine learning RAG pipelines and vector search"},
        ]
        bm25 = BM25Okapi()
        bm25.fit(docs)

        results = bm25.search("ClickHouse analytics", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0]["id"], "doc2")

    def test_dense_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

    def test_hybrid_rrf_merging(self):
        dense_hits = [
            ({"id": "d1", "content": "A"}, 0.95),
            ({"id": "d2", "content": "B"}, 0.85),
        ]
        sparse_hits = [
            ({"id": "d2", "content": "B"}, 4.2),
            ({"id": "d3", "content": "C"}, 3.1),
        ]

        fused = HybridFusion.rrf(dense_hits, sparse_hits, k=60, top_k=3)
        self.assertEqual(len(fused), 3)
        # d2 appeared in both top ranks, so it should rank highest in RRF
        self.assertEqual(fused[0][0]["id"], "d2")

    def test_cross_encoder_reranking(self):
        reranker = CrossEncoderReranker(min_relevance_threshold=0.2)
        candidates = [
            ({"id": "irrelevant", "content": "Weather forecast in Paris today"}, 0.5),
            ({"id": "relevant", "content": "Rust high concurrency tokio actor model"}, 0.4),
        ]

        reranked = reranker.rerank("Rust concurrency tokio", candidates, top_k=2)
        self.assertEqual(len(reranked), 1)
        self.assertEqual(reranked[0][0]["id"], "relevant")


if __name__ == "__main__":
    unittest.main()
