import unittest
from src.agents.critic import HallucinationCriticAgent


class TestCriticGuard(unittest.TestCase):
    def test_grounded_answer_verified(self):
        critic = HallucinationCriticAgent()
        context = [
            {"id": "doc1", "content": "Vector clocks track causality without physical clocks."}
        ]
        answer = "Vector clocks track causality without physical clocks."

        res = critic.verify_faithfulness(answer, context)
        self.assertEqual(res["faithfulness_score"], 1.0)
        self.assertFalse(res["is_hallucination"])
        self.assertEqual(len(res["unsupported_claims"]), 0)

    def test_hallucination_detected(self):
        critic = HallucinationCriticAgent()
        context = [
            {"id": "doc1", "content": "Python is an interpreted programming language."}
        ]
        # Fabricated claim completely absent from context
        answer = "Quantum computers teleported electrons to the Andromeda galaxy yesterday."

        res = critic.verify_faithfulness(answer, context)
        self.assertTrue(res["is_hallucination"])
        self.assertEqual(len(res["unsupported_claims"]), 1)


if __name__ == "__main__":
    unittest.main()
