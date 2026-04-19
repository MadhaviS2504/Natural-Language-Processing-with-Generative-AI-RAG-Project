"""
rag_pipeline.py
---------------
End-to-end RAG pipeline orchestrator.

Provides a single entry point to:
    1. Ingest the Merck Manual PDF
    2. Build (or load) the ChromaDB vector store
    3. Run queries through all three approaches (Baseline, PE, RAG)
    4. Print and return results

Usage:
    python src/rag_pipeline.py
"""

import logging
from pathlib import Path
from typing import Optional

from config import CHROMA_PERSIST_DIR, PDF_PATH, SAMPLE_QUERIES, TOP_K_RETRIEVAL
from src.ingestion import ingest_pipeline
from src.retriever import build_vector_store, get_retriever, load_vector_store, retrieve_context
from src.generator import (
    generate_baseline_response,
    generate_prompt_engineered_response,
    generate_rag_response,
)

logger = logging.getLogger(__name__)

DIVIDER = "─" * 70


def setup_vector_store(force_rebuild: bool = False):
    """
    Load existing ChromaDB store or build one from the PDF.

    Args:
        force_rebuild: If True, re-index even if a store already exists.

    Returns:
        Loaded or newly built Chroma vector store.
    """
    store_exists = Path(CHROMA_PERSIST_DIR).exists()

    if store_exists and not force_rebuild:
        logger.info("Existing vector store found. Loading from disk...")
        return load_vector_store()
    else:
        logger.info("Building vector store from PDF (this may take several minutes)...")
        chunks = ingest_pipeline(PDF_PATH)
        return build_vector_store(chunks)


def run_all_approaches(query: str, retriever, k: int = TOP_K_RETRIEVAL) -> dict:
    """
    Run a query through all three generation approaches.

    Args:
        query:     The medical question to answer.
        retriever: Initialized LangChain retriever.
        k:         Number of chunks to retrieve.

    Returns:
        Dictionary with keys: 'baseline', 'prompt_engineered', 'rag', 'context'.
    """
    print(f"\n{'═' * 70}")
    print(f"🔍 QUERY: {query}")
    print(f"{'═' * 70}")

    # ── Baseline ──────────────────────────────────────────────────────────
    print("\n📌 [1/3] BASELINE LLM RESPONSE")
    print(DIVIDER)
    baseline = generate_baseline_response(query)
    print(baseline)

    # ── Prompt Engineered ─────────────────────────────────────────────────
    print("\n🎯 [2/3] PROMPT-ENGINEERED LLM RESPONSE")
    print(DIVIDER)
    pe_response = generate_prompt_engineered_response(query)
    print(pe_response)

    # ── RAG ───────────────────────────────────────────────────────────────
    print("\n🧠 [3/3] RAG RESPONSE (Grounded in Merck Manual)")
    print(DIVIDER)
    context, docs = retrieve_context(query, retriever, k=k)
    rag_response = generate_rag_response(query, context)
    print(rag_response)

    print(f"\n📄 Retrieved {len(docs)} context chunks from Merck Manual.")

    return {
        "query": query,
        "baseline": baseline,
        "prompt_engineered": pe_response,
        "rag": rag_response,
        "context": context,
        "retrieved_docs": docs,
    }


def main(queries: Optional[list] = None, force_rebuild: bool = False):
    """
    Main entry point for the RAG pipeline.

    Args:
        queries:       List of medical queries. Defaults to SAMPLE_QUERIES.
        force_rebuild: Force re-indexing of the PDF.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    queries = queries or SAMPLE_QUERIES
    results = []

    print("\n🩺 MedRAG — Medical Knowledge Retrieval-Augmented Generation System")
    print("=" * 70)

    # Setup
    vector_store = setup_vector_store(force_rebuild=force_rebuild)
    retriever = get_retriever(vector_store)

    # Run queries
    for i, query in enumerate(queries, start=1):
        print(f"\n\n{'#' * 70}")
        print(f"  Query {i}/{len(queries)}")
        result = run_all_approaches(query, retriever)
        results.append(result)

    print("\n\n✅ All queries processed.")
    return results


if __name__ == "__main__":
    main()
