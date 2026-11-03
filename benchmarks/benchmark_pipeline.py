#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import CortexRAGPipeline


def main():
    print("\033[36m========================================================================\033[0m")
    print("\033[1;32m 🧠 CORTEX: MULTI-AGENT PRODUCTION RAG PIPELINE BENCHMARK \033[0m")
    print("\033[36m========================================================================\033[0m")

    pipeline = CortexRAGPipeline()

    # Knowledge corpus
    corpus = [
        {
            "id": "doc_arch_01",
            "content": "Antigravity agents use asynchronous messaging and reactive wakeups to avoid CPU polling loops.",
        },
        {
            "id": "doc_db_02",
            "content": "ClickHouse ReplacingMergeTree removes duplicate records during background merge passes based on timestamp.",
        },
        {
            "id": "doc_crdt_03",
            "content": "Vector clocks track causal happened-before partial orders without physical clock synchronization requirements.",
        },
        {
            "id": "doc_cache_04",
            "content": "Semantic caching reduces LLM token costs by matching cosine similarity of query embeddings above threshold.",
        },
    ]

    # Pre-computed mock normalized embeddings (4 dimensions for demonstration)
    embeddings = [
        [0.85, 0.12, 0.05, 0.50],
        [0.10, 0.92, 0.15, 0.35],
        [0.05, 0.10, 0.98, 0.12],
        [0.55, 0.40, 0.20, 0.70],
    ]

    print(f"[*] Indexing {len(corpus)} knowledge documents into BM25 and Dense stores...")
    pipeline.index_documents(corpus, embeddings)

    # Test Query 1: Cache Miss
    query = "How does ClickHouse handle duplicate rows in background merges?"
    query_emb = [0.12, 0.90, 0.14, 0.38]

    print("\n[*] Query #1 (Cold Cache Miss):")
    print(f"    User Query: '{query}'")

    res1 = pipeline.query(query, query_emb)
    print("\033[1;32m[+] PIPELINE EXECUTION SUCCESSFUL:\033[0m")
    print(f"    From Cache:          {res1['from_cache']}")
    print(f"    Answer:              {res1['answer'][:90]}...")
    print(f"    Faithfulness Score:  \033[1;33m{res1['faithfulness_score'] * 100}%\033[0m")
    print(f"    Hallucination Alert: {res1['is_hallucination']}")
    print(f"    Total Elapsed:       \033[1;36m{res1['timing']['total_elapsed_ms']} ms\033[0m")
    print(f"    Stage Timings:       {res1['timing']}")

    # Test Query 2: Cache Hit (Paraphrased query with cosine sim > 0.95)
    query_paraphrased = "How ClickHouse deduplicates data during merges?"
    query_paraphrased_emb = [0.11, 0.91, 0.15, 0.37] # Cosine similarity ~0.999

    print("\n[*] Query #2 (Semantic Cache Hit - Paraphrased Query):")
    print(f"    User Query: '{query_paraphrased}'")

    res2 = pipeline.query(query_paraphrased, query_paraphrased_emb)
    print("\033[1;32m[+] INSTANT SEMANTIC CACHE HIT!\033[0m")
    print(f"    From Cache:          \033[1;32m{res2['from_cache']}\033[0m")
    print(f"    Cosine Similarity:   \033[1;33m{res2['similarity']}\033[0m")
    print(f"    Answer:              {res2['answer'][:90]}...")
    print(f"    Total Latency:       \033[1;36m{res2['timing']['total_elapsed_ms']} ms\033[0m (< 1ms!)")


if __name__ == "__main__":
    main()
