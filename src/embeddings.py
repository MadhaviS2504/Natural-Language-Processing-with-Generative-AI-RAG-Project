"""
src/embeddings.py
-----------------
Handles embedding model loading and generation.

Uses sentence-transformers to generate dense vector representations
of text chunks for semantic similarity search.
"""

import logging
from functools import lru_cache

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from src.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_function(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformerEmbeddings:
    """
    Load and cache the embedding model.

    Uses LRU cache to prevent re-loading across calls in the same session.

    Args:
        model_name: Name of the sentence-transformers model.

    Returns:
        Initialized SentenceTransformerEmbeddings instance.
    """
    logger.info(f"Loading embedding model: {model_name}")
    embedding_fn = SentenceTransformerEmbeddings(model_name=model_name)
    
    # Quick sanity check
    test_vec = embedding_fn.embed_query("test query")
    logger.info(f"Embedding model loaded. Vector dimension: {len(test_vec)}")
    
    return embedding_fn


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.
    """
    embedding_fn = get_embedding_function()
    return embedding_fn.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """
    Generate embedding for a single query.

    Args:
        query: Query string to embed.

    Returns:
        Embedding vector.
    """
    embedding_fn = get_embedding_function()
    return embedding_fn.embed_query(query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedding_fn = get_embedding_function()
    test_embedding = embedding_fn.embed_query("What is sepsis?")
    print(f"✅ Embedding generated. Dimension: {len(test_embedding)}")