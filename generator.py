"""
generator.py
------------
Handles LLM model loading and response generation.

Uses llama-cpp-python to run Mistral-7B locally (GGUF quantized format).
Supports both GPU (CUDA) and CPU inference.

Prompting approaches:
    1. Baseline     — raw query, no system prompt
    2. Prompt-Eng   — structured system prompt, no retrieval context
    3. RAG          — system prompt + retrieved medical context
"""

import logging
from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from config import (
    MAX_TOKENS,
    MODEL_CONTEXT_WINDOW,
    MODEL_FILENAME,
    MODEL_GPU_LAYERS,
    MODEL_REPO_ID,
    MODEL_THREADS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

logger = logging.getLogger(__name__)

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
def load_llm(
    repo_id: str = MODEL_REPO_ID,
    filename: str = MODEL_FILENAME,
    n_gpu_layers: int = MODEL_GPU_LAYERS,
    n_ctx: int = MODEL_CONTEXT_WINDOW,
    n_threads: int = MODEL_THREADS,
) -> Llama:
    """
    Download (if needed) and load the Mistral-7B GGUF model.

    Uses LRU cache to prevent re-loading across calls in the same session.
    Model is downloaded to HuggingFace cache (~4GB for Q4_K_M quantization).

    Args:
        repo_id:       HuggingFace repository ID.
        filename:      GGUF filename within the repository.
        n_gpu_layers:  Number of layers to offload to GPU (0 = CPU only).
        n_ctx:         Context window size in tokens.
        n_threads:     CPU threads for computation.

    Returns:
        Initialized Llama instance ready for inference.
    """
    logger.info(f"Downloading model '{filename}' from '{repo_id}'...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info(f"Model path: {model_path}")

    logger.info(f"Loading LLM (GPU layers: {n_gpu_layers}, context: {n_ctx})...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )
    logger.info("LLM loaded successfully.")
    return llm


# ─── Generation Functions ─────────────────────────────────────────────────────

def generate_baseline_response(
    query: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
) -> str:
    """
    Generate a response using the LLM with no system prompt or context (baseline).

    Args:
        query:       User's raw question.
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature (0 = deterministic).
        top_p:       Nucleus sampling probability.
        top_k:       Top-k sampling.

    Returns:
        Generated text string.
    """
    llm = load_llm()
    output = llm(
        prompt=query,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return output["choices"][0]["text"].strip()


def generate_prompt_engineered_response(
    query: str,
    system_prompt: str = PROMPT_ENG_SYSTEM,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
) -> str:
    """
    Generate a response using structured prompt engineering (no RAG context).

    Args:
        query:         User's clinical question.
        system_prompt: System-level instruction for the LLM.
        max_tokens:    Maximum tokens to generate.
        temperature:   Sampling temperature.
        top_p:         Nucleus sampling probability.
        top_k:         Top-k sampling.

    Returns:
        Generated text string.
    """
    llm = load_llm()
    full_prompt = f"{system_prompt}\n\nQuestion: {query}\n\nAnswer:"
    output = llm(
        prompt=full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return output["choices"][0]["text"].strip()


def generate_rag_response(
    query: str,
    context: str,
    system_prompt: str = MEDICAL_SYSTEM_PROMPT,
    user_template: str = QNA_USER_TEMPLATE,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
) -> str:
    """
    Generate a grounded RAG response using retrieved medical context.

    Args:
        query:         User's clinical question.
        context:       Retrieved medical text from ChromaDB.
        system_prompt: System-level instruction.
        user_template: Template with {context} and {question} placeholders.
        max_tokens:    Maximum tokens to generate.
        temperature:   Sampling temperature.
        top_p:         Nucleus sampling probability.
        top_k:         Top-k sampling.

    Returns:
        Generated, context-grounded text string.
    """
    llm = load_llm()

    user_message = user_template.replace("{context}", context).replace("{question}", query)
    full_prompt = f"{system_prompt}\n{user_message}"

    try:
        output = llm(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return f"Error during generation: {e}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    query = "What is the treatment for appendicitis?"
    print("\n📡 Generating baseline response...")
    print(generate_baseline_response(query, max_tokens=100))
