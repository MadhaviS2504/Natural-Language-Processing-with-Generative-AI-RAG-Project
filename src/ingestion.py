"""
src/ingestion.py
----------------
Handles PDF loading and document chunking for the RAG pipeline.

Responsibilities:
    - Load the Merck Manual PDF using PyMuPDF
    - Split documents into semantically coherent chunks
    - Expose chunk statistics for debugging and tuning
"""

import logging
from pathlib import Path
from typing import List, Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    PDF_PATH,
)

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: Optional[str | Path] = None) -> List[Document]:
    """
    Load a PDF document using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file. Defaults to PDF_PATH from config.

    Returns:
        List of LangChain Document objects (one per page).

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
    """
    pdf_path = Path(pdf_path or PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at '{pdf_path}'. "
            "Please place the Merck Manual PDF in the data/ directory. "
            "See data/README.md for sourcing instructions."
        )

    logger.info(f"Loading PDF from: {pdf_path}")
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from PDF.")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: List[str] = CHUNK_SEPARATORS,
) -> List[Document]:
    """
    Split documents into overlapping text chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on paragraphs,
    then sentences, then words — preserving semantic coherence.

    Args:
        documents:     List of LangChain Documents (pages).
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Characters to overlap between consecutive chunks.
        separators:    Ordered list of split points.

    Returns:
        List of chunked Document objects with source metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        f"Created {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})."
    )
    return chunks


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in a string using tiktoken (approximation for non-OpenAI models)."""
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Compute summary statistics for a list of chunks.

    Args:
        chunks: List of Document chunks.

    Returns:
        Dictionary with min, max, mean, and total character counts.
    """
    lengths = [len(c.page_content) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "total_chars": sum(lengths),
    }


def ingest_pipeline(pdf_path: Optional[str | Path] = None) -> List[Document]:
    """
    Full ingestion pipeline: load PDF → chunk → return chunks.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        List of text chunks ready for embedding.
    """
    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)
    stats = get_chunk_stats(chunks)
    logger.info(f"Chunk statistics: {stats}")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = ingest_pipeline()
    print(f"\n✅ Ingestion complete. Total chunks: {len(chunks)}")
    print(f"📄 Sample chunk:\n{chunks[0].page_content[:300]}...")