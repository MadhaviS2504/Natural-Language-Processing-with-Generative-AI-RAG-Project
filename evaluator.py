"""
evaluator.py
------------
LLM-as-a-Judge evaluation framework for the RAG pipeline.

Evaluates generated responses on two dimensions:
    1. Groundedness  — Is the answer supported by retrieved context?
    2. Relevance     — Does the answer address the user's question?

Both metrics are scored 1–5 by the same LLM used for generation,
enabling scalable automated evaluation without human annotation.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.generator import load_llm
from config import EVAL_MAX_TOKENS, EVAL_TEMPERATURE, TOP_P, TOP_K

logger = logging.getLogger(__name__)

# ─── Evaluation Prompts ───────────────────────────────────────────────────────

GROUNDEDNESS_SYSTEM = """You are a strict medical content evaluator.

Your task is to rate how well an AI-generated answer is grounded in the provided medical context.

Scoring rubric (1-5):
    5 — Answer is fully supported by the context; every claim can be traced.
    4 — Answer is mostly supported; minor points may go slightly beyond context.
    3 — Answer is partially supported; some claims are unsupported or inferred.
    2 — Answer is poorly supported; most claims are not in the context.
    1 — Answer is not supported at all; contradicts or ignores the context.

Respond ONLY with:
Score: <number>
Reason: <one sentence explanation>
"""

RELEVANCE_SYSTEM = """You are a strict medical content evaluator.

Your task is to rate how relevant an AI-generated answer is to the user's question.

Scoring rubric (1-5):
    5 — Answer directly and completely addresses the question.
    4 — Answer mostly addresses the question with minor gaps.
    3 — Answer partially addresses the question; key aspects are missing.
    2 — Answer is tangentially related but misses the core question.
    1 — Answer is irrelevant or does not address the question at all.

Respond ONLY with:
Score: <number>
Reason: <one sentence explanation>
"""

EVAL_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer: {answer}

Evaluate the answer above."""


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Holds evaluation scores and reasoning for a single query-response pair."""
    query: str
    answer: str
    groundedness_score: Optional[int]
    groundedness_reason: str
    relevance_score: Optional[int]
    relevance_reason: str
    context_preview: str

    def __str__(self) -> str:
        return (
            f"\n📊 Evaluation Results\n"
            f"{'─' * 50}\n"
            f"Query: {self.query[:80]}...\n\n"
            f"🔗 Groundedness Score : {self.groundedness_score}/5\n"
            f"   Reason            : {self.groundedness_reason}\n\n"
            f"🎯 Relevance Score   : {self.relevance_score}/5\n"
            f"   Reason            : {self.relevance_reason}\n"
        )


# ─── Parsing Helper ───────────────────────────────────────────────────────────

def _parse_score(text: str) -> tuple[Optional[int], str]:
    """
    Parse LLM evaluation output to extract numeric score and reason.

    Args:
        text: Raw LLM output string.

    Returns:
        Tuple of (score: int or None, reason: str).
    """
    score = None
    reason = "Could not parse reason."

    score_match = re.search(r"Score:\s*([1-5])", text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))

    reason_match = re.search(r"Reason:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()[:200]

    return score, reason


# ─── Core Evaluation Functions ────────────────────────────────────────────────

def evaluate_response(
    query: str,
    answer: str,
    context: str,
    max_tokens: int = EVAL_MAX_TOKENS,
    temperature: float = EVAL_TEMPERATURE,
) -> EvaluationResult:
    """
    Evaluate a single RAG response for groundedness and relevance.

    Args:
        query:      The user's original question.
        answer:     The LLM-generated answer to evaluate.
        context:    The retrieved medical context used for generation.
        max_tokens: Max tokens for evaluation responses.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        EvaluationResult with scores and reasoning.
    """
    llm = load_llm()
    user_eval = EVAL_USER_TEMPLATE.format(
        context=context[:2000],  # Truncate to avoid context overflow
        question=query,
        answer=answer,
    )

    # ── Groundedness ──────────────────────────────────────────────────────
    groundedness_prompt = f"{GROUNDEDNESS_SYSTEM}\n\n{user_eval}"
    g_output = llm(
        prompt=groundedness_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=TOP_P,
        top_k=TOP_K,
        stop=["INST"],
    )
    g_text = g_output["choices"][0]["text"]
    g_score, g_reason = _parse_score(g_text)

    # ── Relevance ─────────────────────────────────────────────────────────
    relevance_prompt = f"{RELEVANCE_SYSTEM}\n\n{user_eval}"
    r_output = llm(
        prompt=relevance_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=TOP_P,
        top_k=TOP_K,
        stop=["INST"],
    )
    r_text = r_output["choices"][0]["text"]
    r_score, r_reason = _parse_score(r_text)

    return EvaluationResult(
        query=query,
        answer=answer,
        groundedness_score=g_score,
        groundedness_reason=g_reason,
        relevance_score=r_score,
        relevance_reason=r_reason,
        context_preview=context[:300],
    )


def batch_evaluate(
    results: list[dict],
) -> list[EvaluationResult]:
    """
    Evaluate a batch of RAG results.

    Args:
        results: List of dicts with keys 'query', 'rag', 'context'.

    Returns:
        List of EvaluationResult objects.
    """
    evaluations = []
    for i, result in enumerate(results, start=1):
        logger.info(f"Evaluating query {i}/{len(results)}: {result['query'][:50]}...")
        eval_result = evaluate_response(
            query=result["query"],
            answer=result["rag"],
            context=result["context"],
        )
        evaluations.append(eval_result)
        print(eval_result)

    # ── Summary ───────────────────────────────────────────────────────────
    valid_g = [e.groundedness_score for e in evaluations if e.groundedness_score]
    valid_r = [e.relevance_score for e in evaluations if e.relevance_score]

    print(f"\n{'=' * 50}")
    print("📈 EVALUATION SUMMARY")
    print(f"{'─' * 50}")
    print(f"Avg Groundedness Score : {sum(valid_g)/len(valid_g):.2f}/5" if valid_g else "N/A")
    print(f"Avg Relevance Score    : {sum(valid_r)/len(valid_r):.2f}/5" if valid_r else "N/A")
    print(f"Queries Evaluated      : {len(evaluations)}")

    return evaluations
