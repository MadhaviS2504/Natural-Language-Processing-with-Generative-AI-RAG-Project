"""
src/retriever.py
----------------
Manages ChromaDB vector store creation, persistence, and document retrieval.

Responsibilities:
    - Build the ChromaDB vector store from document chunks
    - Persist the store to disk for reuse across sessions
    - Expose a retriever interface for similarity search
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    TOP_K_RETRIEVAL,
)
from src.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

# Global vector store instance for reuse
_vector_store: Optional[Chroma] = None


def build_vector_store(
    chunks: List[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Build a ChromaDB vector store from document chunks and persist it to disk.

    This operation is expensive (~minutes for 4,000 pages) and should only be
    run once. On subsequent runs, use `load_vector_store()` instead.

    Args:
        chunks:           List of chunked Document objects.
        persist_directory: Directory to save the ChromaDB files.
        collection_name:   Name for the ChromaDB collection.

    Returns:
        Populated Chroma vector store instance.
    """
    logger.info(f"Building vector store with {len(chunks)} chunks...")
    embedding_fn = get_embedding_function()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    vector_store.persist()
    logger.info(f"Vector store saved to: {persist_directory}")
    return vector_store


def load_vector_store(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        persist_directory: Directory where ChromaDB files are stored.
        collection_name:   Name of the ChromaDB collection.

    Returns:
        Loaded Chroma vector store instance.

    Raises:
        FileNotFoundError: If no persisted store exists at the given path.
    """
    if not Path(persist_directory).exists():
        raise FileNotFoundError(
            f"No vector store found at '{persist_directory}'. "
            "Run `build_vector_store()` first to index the documents."
        )

    logger.info(f"Loading vector store from: {persist_directory}")
    embedding_fn = get_embedding_function()

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_fn,
        collection_name=collection_name,
    )
    logger.info("Vector store loaded successfully.")
    return vector_store


def get_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Get or create the vector store (singleton pattern).

    Args:
        force_rebuild: If True, rebuild the vector store even if it exists.

    Returns:
        Chroma vector store instance.
    """
    global _vector_store
    
    if _vector_store is None or force_rebuild:
        try:
            _vector_store = load_vector_store()
        except FileNotFoundError:
            logger.info("Vector store not found. Building new one...")
            from src.ingestion import ingest_pipeline
            from src.config import PDF_PATH
            chunks = ingest_pipeline(PDF_PATH)
            _vector_store = build_vector_store(chunks)
    
    return _vector_store


def get_retriever(
    vector_store: Optional[Chroma] = None,
    k: int = TOP_K_RETRIEVAL,
):
    """
    Create a retriever from a Chroma vector store.

    Args:
        vector_store: Populated Chroma instance. If None, uses get_vector_store().
        k:            Number of top documents to retrieve per query.

    Returns:
        LangChain retriever object.
    """
    if vector_store is None:
        vector_store = get_vector_store()
    
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    logger.info(f"Retriever created with top_k={k}.")
    return retriever


def retrieve_context(
    query: str,
    retriever=None,
    k: int = TOP_K_RETRIEVAL,
) -> tuple[str, List[Document]]:
    """
    Retrieve relevant document chunks for a given query.

    Args:
        query:     The user's natural language question.
        retriever: LangChain retriever instance. If None, creates one.
        k:         Number of chunks to retrieve.

    Returns:
        Tuple of (combined_context_string, list_of_documents).
    """
    if retriever is None:
        retriever = get_retriever(k=k)
    
    docs = retriever.invoke(query)
    context = ". ".join([d.page_content for d in docs])
    logger.debug(f"Retrieved {len(docs)} chunks for query: '{query[:60]}...'")
    return context, docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        store = get_vector_store()
        retriever = get_retriever(store)
        test_query = "What is the protocol for managing sepsis?"
        context, docs = retrieve_context(test_query, retriever)
        print(f"\n✅ Retrieved {len(docs)} relevant chunks.")
        print(f"📄 First chunk preview:\n{docs[0].page_content[:400]}...")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")