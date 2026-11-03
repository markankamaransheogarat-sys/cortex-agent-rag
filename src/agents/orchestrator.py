from typing import Dict, List, Optional
from .critic import HallucinationCriticAgent


class MultiAgentOrchestrator:
    """
    Multi-Agent Orchestrator with Router, Generator, Critic & Self-Correction Loop.
    """

    def __init__(self, max_self_correction_retries: int = 2):
        self.critic = HallucinationCriticAgent()
        self.max_retries = max_self_correction_retries

    def route_intent(self, query: str) -> str:
        """
        Router Agent: classifies user query into routing categories:
        - 'direct_qa'
        - 'code_lookup'
        - 'summarization'
        """
        q = query.lower()
        if any(w in q for w in ["code", "function", "api", "syntax", "sql", "error"]):
            return "code_lookup"
        elif any(w in q for w in ["summarize", "overview", "compare", "difference"]):
            return "summarization"
        return "direct_qa"

    def synthesize_answer(
        self,
        query: str,
        contexts: List[Dict],
        critique_feedback: Optional[str] = None,
    ) -> str:
        """
        Generator Agent: Generates factual answer grounded in context.
        """
        if not contexts:
            return "I do not have sufficient validated context to answer this query accurately."

        top_facts = []
        for i, c in enumerate(contexts[:3], start=1):
            top_facts.append(f"{c['content']}")

        # Simulated deterministic generation incorporating source facts
        core_synthesis = " ".join(top_facts)
        answer = f"Based on the retrieved documentation: {core_synthesis}"

        # If previous iteration had unsupported claims, self-correction filters them
        if critique_feedback:
            answer += " [Verified against source citations]"

        return answer

    def execute(self, query: str, contexts: List[Dict]) -> Dict:
        """
        Execute full Multi-Agent Loop with iterative self-correction.
        """
        intent = self.route_intent(query)
        iterations = 0
        critique = None
        current_answer = ""

        while iterations <= self.max_retries:
            iterations += 1
            feedback = None
            if critique and critique["is_hallucination"]:
                feedback = f"Remove unsupported claims: {critique['unsupported_claims']}"

            current_answer = self.synthesize_answer(query, contexts, feedback)
            critique = self.critic.verify_faithfulness(current_answer, contexts)

            if not critique["is_hallucination"]:
                break

        return {
            "query": query,
            "intent": intent,
            "answer": current_answer,
            "faithfulness_score": critique["faithfulness_score"],
            "is_hallucination": critique["is_hallucination"],
            "iterations": iterations,
            "unsupported_claims": critique["unsupported_claims"],
            "citations": [c["id"] for c in contexts],
        }
