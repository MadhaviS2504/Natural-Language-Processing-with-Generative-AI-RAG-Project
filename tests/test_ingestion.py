"""
tests/test_ingestion.py
-----------------------
Unit tests for the ingestion module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document


class TestIngestion:
    """Test cases for the ingestion module."""
    
    def test_chunk_documents_basic(self):
        """Test basic document chunking functionality."""
        from src.ingestion import chunk_documents
        
        # Create mock documents
        docs = [
            Document(
                page_content="This is a test document. " * 50,
                metadata={"source": "test.pdf", "page": 1}
            )
        ]
        
        # Chunk the documents
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        
        # Verify chunks were created
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
    
    def test_get_chunk_stats(self):
        """Test chunk statistics calculation."""
        from src.ingestion import get_chunk_stats
        
        chunks = [
            Document(page_content="Short", metadata={}),
            Document(page_content="Medium length text", metadata={}),
            Document(page_content="This is a much longer text for testing", metadata={}),
        ]
        
        stats = get_chunk_stats(chunks)
        
        assert stats["total_chunks"] == 3
        assert stats["min_chars"] == 5
        assert stats["max_chars"] == 40
    
    @patch("src.ingestion.PyMuPDFLoader")
    def test_load_pdf_not_found(self, mock_loader):
        """Test that load_pdf raises error for missing PDF."""
        from src.ingestion import load_pdf
        
        with pytest.raises(FileNotFoundError):
            load_pdf("/nonexistent/path.pdf")
    
    def test_count_tokens(self):
        """Test token counting functionality."""
        from src.ingestion import count_tokens
        
        text = "This is a test sentence."
        token_count = count_tokens(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0


class TestConfig:
    """Test cases for configuration."""
    
    def test_config_values(self):
        """Test that config values are properly set."""
        from src.config import (
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            TOP_K_RETRIEVAL,
            MAX_TOKENS,
            TEMPERATURE,
        )
        
        assert CHUNK_SIZE == 1000
        assert CHUNK_OVERLAP == 200
        assert TOP_K_RETRIEVAL == 3
        assert MAX_TOKENS == 512
        assert TEMPERATURE == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])