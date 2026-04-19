"""
src/generator.py
----------------
Handles LLM model loading and response generation.

Uses HuggingFace Inference API for Mistral-7B model.
Supports both local and remote inference.

Prompting approaches:
    1. Baseline     — raw query, no system prompt
    2. Prompt-Eng   — structured system prompt, no retrieval context
    3. RAG          — system prompt + retrieved medical context
"""

import logging
import os
from functools import lru_cache
from typing import Optional

from huggingface_hub import InferenceClient

from src.config import (
    MAX_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    HF_API_KEY,
)

logger = logging.getLogger(__name__)

# Global LLM instance for reuse
_client: Optional[InferenceClient] = None

# ─── System Prompts ───────────────────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = """You are a knowledgeable medical assistant trained on the Merck Manual.
Your role is to provide accurate, evidence-based medical information to healthcare professionals.

Guidelines:
- Answer only based on the provided medical context.
- If the context does not contain sufficient information, clearly state that.
- Use precise medical terminology.
- Structure your response clearly with relevant clinical details.
- Do NOT fabricate medical facts or treatment protocols.
- Always recommend consulting a qualified physician for patient care decisions.
"""

QNA_USER_TEMPLATE = """Use the following medical context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_ENG_SYSTEM = """You are a concise, expert medical assistant.
Answer the following clinical question clearly and accurately.
Structure your response with:
1. Brief overview
2. Key clinical points
3. Recommended actions or treatments
"""


# ─── Model Loading ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_client_cached(
    api_key: str = HF_API_KEY,
) -> InferenceClient:
    """
    Initialize the HuggingFace Inference client.

    Uses LRU cache to prevent re-loading across calls in the same session.

    Args:
        api_key: HuggingFace API key for authentication.

    Returns:
        Initialized InferenceClient ready for inference.
    """
    logger.info("Initializing HuggingFace Inference client...")
    client = InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.2",
        token=api_key,
    )
    logger.info("Inference client initialized successfully.")
    return client


def load_llm(
    api_key: str = HF_API_KEY,
    force_reload: bool = False,
) -> InferenceClient:
    """
    Load (or reuse) the HuggingFace Inference client.

    Args:
        api_key: HuggingFace API key.
        force_reload: If True, bypass cache and reload.

    Returns:
        InferenceClient instance.
    """
    global _client
    if _client is None or force_reload:
        _client = _load_client_cached(api_key)
    return _client


# ─── Response Generation ──────────────────────────────────────────────────────

def generate_baseline_response(
    query: str,
    api_key: str = HF_API_KEY,
) -> str:
    """
    Generate a baseline response (raw query, no system prompt).

    Args:
        query: User's question.
        api_key: HuggingFace API key.

    Returns:
        Model's raw response.
    """
    client = load_llm(api_key)
    logger.info(f"Generating baseline response for query: {query[:50]}...")

    response = client.text_generation(
        query,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    return response


def generate_prompt_engineered_response(
    query: str,
    api_key: str = HF_API_KEY,
) -> str:
    """
    Generate a response using prompt engineering (system prompt, no RAG).

    Args:
        query: User's question.
        api_key: HuggingFace API key.

    Returns:
        Model's response with system prompt.
    """
    client = load_llm(api_key)
    logger.info(f"Generating prompt-engineered response for query: {query[:50]}...")

    full_prompt = f"{PROMPT_ENG_SYSTEM}\n\nQuestion: {query}\nAnswer:"

    response = client.text_generation(
        full_prompt,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    return response


def generate_rag_response(
    query: str,
    context: str,
    api_key: str = HF_API_KEY,
) -> str:
    """
    Generate a RAG response (system prompt + retrieved context).

    Args:
        query: User's question.
        context: Retrieved context from the vector store.
        api_key: HuggingFace API key.

    Returns:
        Model's response grounded in the provided context.
    """
    client = load_llm(api_key)
    logger.info(f"Generating RAG response for query: {query[:50]}...")

    user_prompt = QNA_USER_TEMPLATE.format(context=context, question=query)
    full_prompt = f"{MEDICAL_SYSTEM_PROMPT}\n\n{user_prompt}"

    response = client.text_generation(
        full_prompt,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    return response