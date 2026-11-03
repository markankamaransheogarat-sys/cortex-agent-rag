import math
import re
from typing import Dict, List, Tuple


class BM25Okapi:
    """
    Production-grade BM25 (Best Matching 25) sparse lexical retrieval engine.
    Computes exact probabilistic relevance without external dependencies.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: List[int] = []
        self.doc_token_counts: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.documents: List[Dict] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple lowercase alphanumeric tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, documents: List[Dict]):
        """
        Index a collection of documents.
        Each document is a dict with at least 'id' and 'content'.
        """
        self.documents = documents
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        total_length = 0
        self.doc_lengths = []
        self.doc_token_counts = []
        df: Dict[str, int] = {}

        for doc in documents:
            tokens = self.tokenize(doc["content"])
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1

            self.doc_token_counts.append(counts)

            # Document frequency
            for term in counts.keys():
                df[term] = df.get(term, 0) + 1

        self.avgdl = total_length / self.corpus_size

        # Compute IDF with Lucene/Robertson smoothing
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Score and rank indexed documents for a given query.
        """
        if self.corpus_size == 0:
            return []

        query_tokens = self.tokenize(query)
        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            idf_val = self.idf.get(term, 0.0)
            if idf_val <= 0.0:
                continue

            for i in range(self.corpus_size):
                tf = self.doc_token_counts[i].get(term, 0)
                if tf > 0:
                    doc_len = self.doc_lengths[i]
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[i] += idf_val * (numerator / denominator)

        # Rank documents by score descending
        ranked_indices = sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked_indices[:top_k]:
            if scores[i] > 0.0:
                results.append((self.documents[i], round(scores[i], 4)))

        return results
