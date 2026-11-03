import time
from typing import Any, Dict, List, Optional

from .retrieval.bm25 import BM25Okapi
from .retrieval.dense import DenseVectorStore
from .retrieval.hybrid import HybridFusion
from .retrieval.reranker import CrossEncoderReranker
from .cache.semantic_cache import SemanticCache
from .agents.orchestrator import MultiAgentOrchestrator


class CortexRAGPipeline:
    """
    End-to-End Enterprise Production Multi-Agent RAG Pipeline.
    Combines Sub-Millisecond Semantic Caching, Hybrid Dense+BM25 Search,
    Reciprocal Rank Fusion, Cross-Encoder Reranking, and Self-Correcting Agentic Verification.
    """

    def __init__(
        self,
        cache_threshold: float = 0.92,
        rerank_threshold: float = 0.25,
    ):
        self.bm25 = BM25Okapi()
        self.dense_store = DenseVectorStore()
        self.semantic_cache = SemanticCache(similarity_threshold=cache_threshold)
        self.reranker = CrossEncoderReranker(min_relevance_threshold=rerank_threshold)
        self.orchestrator = MultiAgentOrchestrator()

    def index_documents(
        self,
        documents: List[Dict],
        embeddings: Optional[List[List[float]]] = None,
    ):
        """Index documents into both BM25 and Dense vector stores."""
        self.bm25.fit(documents)
        if embeddings:
            self.dense_store.add_many(documents, embeddings)

    def query(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute full production RAG pipeline:
        Cache -> Hybrid Retrieval -> RRF -> Reranking -> Agent Synthesis & Verification
        """
        start_time = time.perf_counter()
        timing = {}

        # 1. Semantic Cache Lookup
        if query_embedding:
            cache_start = time.perf_counter()
            cached = self.semantic_cache.get(query, query_embedding)
            timing["cache_lookup_ms"] = round((time.perf_counter() - cache_start) * 1000, 3)

            if cached:
                timing["total_elapsed_ms"] = round((time.perf_counter() - start_time) * 1000, 3)
                return {
                    "query": query,
                    "answer": cached["response"]["answer"],
                    "from_cache": True,
                    "similarity": cached["similarity"],
                    "faithfulness_score": cached["response"]["faithfulness_score"],
                    "timing": timing,
                    "citations": cached["response"].get("citations", []),
                }

        # 2. Stage 1: Hybrid Retrieval (Dense + BM25)
        retrieval_start = time.perf_counter()
        sparse_hits = self.bm25.search(query, top_k=20)
        dense_hits = []
        if query_embedding and self.dense_store.count() > 0:
            dense_hits = self.dense_store.search(query_embedding, top_k=20)

        # 3. Reciprocal Rank Fusion (RRF)
        fused_candidates = HybridFusion.rrf(dense_hits, sparse_hits, k=60, top_k=15)
        timing["hybrid_retrieval_ms"] = round((time.perf_counter() - retrieval_start) * 1000, 3)

        # 4. Stage 2: Cross-Encoder Reranking
        rerank_start = time.perf_counter()
        reranked_docs = self.reranker.rerank(query, fused_candidates, top_k=top_k)
        timing["reranker_ms"] = round((time.perf_counter() - rerank_start) * 1000, 3)

        top_contexts = [doc for doc, score in reranked_docs]

        # 5. Multi-Agent Synthesis & Verification Loop
        agent_start = time.perf_counter()
        agent_result = self.orchestrator.execute(query, top_contexts)
        timing["agent_synthesis_ms"] = round((time.perf_counter() - agent_start) * 1000, 3)

        timing["total_elapsed_ms"] = round((time.perf_counter() - start_time) * 1000, 3)

        # 6. Save verified response in Semantic Cache
        if query_embedding:
            self.semantic_cache.put(query, query_embedding, agent_result)

        return {
            "query": query,
            "answer": agent_result["answer"],
            "from_cache": False,
            "faithfulness_score": agent_result["faithfulness_score"],
            "is_hallucination": agent_result["is_hallucination"],
            "iterations": agent_result["iterations"],
            "citations": agent_result["citations"],
            "timing": timing,
        }
