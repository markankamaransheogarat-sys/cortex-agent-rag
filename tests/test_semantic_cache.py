import unittest
from src.cache.semantic_cache import SemanticCache


class TestSemanticCache(unittest.TestCase):
    def test_cache_hit_on_high_similarity(self):
        cache = SemanticCache(similarity_threshold=0.90)

        emb_original = [0.1, 0.9, 0.2]
        emb_similar = [0.11, 0.89, 0.21] # Cosine similarity > 0.99
        emb_different = [0.9, 0.1, 0.0]  # Very low similarity

        cache.put("How to partition PostgreSQL?", emb_original, {"answer": "Use PARTITION BY RANGE"})

        # Hit test
        hit = cache.get("How to partition PostgreSQL tables?", emb_similar)
        self.assertIsNotNone(hit)
        self.assertTrue(hit["hit"])
        self.assertEqual(hit["response"]["answer"], "Use PARTITION BY RANGE")

        # Miss test
        miss = cache.get("Tell me about cats", emb_different)
        self.assertIsNone(miss)

    def test_hit_rate_tracking(self):
        cache = SemanticCache(similarity_threshold=0.90)
        cache.put("Q1", [1.0, 0.0], "A1")

        cache.get("Q1", [1.0, 0.0]) # Hit
        cache.get("Q2", [0.0, 1.0]) # Miss

        self.assertEqual(cache.hit_rate, 50.0)


if __name__ == "__main__":
    unittest.main()
