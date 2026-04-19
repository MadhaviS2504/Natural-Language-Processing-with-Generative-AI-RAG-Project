"""
api/routes/rag.py
-----------------
RAG pipeline endpoints for the API.

Provides endpoints for:
- Submitting queries to the RAG pipeline
- Checking RAG pipeline status
"""

import logging
from flask import Blueprint, request, jsonify

from src.rag_pipeline import run_rag_only
from src.config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

rag_bp = Blueprint("rag", __name__)


@rag_bp.route("/query", methods=["POST"])
def rag_query():
    """
    Submit a query to the RAG pipeline.
    
    Request body:
        {
            "query": "What is the treatment for sepsis?",
            "top_k": 3  // optional
        }
    
    Returns:
        JSON response with answer and sources.
    """
    import time
    start_time = time.time()
    
    # Validate request
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        query = data.get("query")
        if not query:
            return jsonify({"error": "Field 'query' is required"}), 400
        
        top_k = data.get("top_k", TOP_K_RETRIEVAL)
        
    except Exception as e:
        logger.error(f"Request validation error: {e}")
        return jsonify({"error": "Invalid request format", "message": str(e)}), 400
    
    try:
        logger.info(f"Processing RAG query: {query[:50]}...")
        
        # Run RAG pipeline
        result = run_rag_only(query, k=top_k)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "query": result["query"],
            "answer": result["answer"],
            "sources": result["sources"],
            "processing_time_ms": round(processing_time, 2),
        })
        
    except FileNotFoundError as e:
        logger.error(f"Vector store not found: {e}")
        return jsonify({
            "error": "Vector store not initialized",
            "message": "Please ensure the PDF has been processed and the vector store is built.",
        }), 503
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
        }), 500


@rag_bp.route("/status", methods=["GET"])
def rag_status():
    """
    Get RAG pipeline status.
    
    Returns:
        JSON response with status information.
    """
    try:
        # Check vector store
        from src.retriever import get_vector_store
        store = get_vector_store()
        
        # Get collection info
        collection = store._collection
        count = collection.count()
        
        return jsonify({
            "status": "ready",
            "vector_store": {
                "initialized": True,
                "document_count": count,
            },
            "config": {
                "top_k": TOP_K_RETRIEVAL,
                "chunk_size": 1000,
                "chunk_overlap": 200,
            }
        })
        
    except FileNotFoundError:
        return jsonify({
            "status": "not_initialized",
            "vector_store": {
                "initialized": False,
                "document_count": 0,
            },
            "message": "Vector store not found. Please process the PDF first.",
        }), 200
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
        }), 500


@rag_bp.route("/rebuild", methods=["POST"])
def rag_rebuild():
    """
    Rebuild the vector store from the PDF.
    
    This is a long-running operation and should be used carefully.
    
    Returns:
        JSON response with rebuild status.
    """
    try:
        from src.retriever import get_vector_store, build_vector_store
        from src.ingestion import ingest_pipeline
        from src.config import PDF_PATH
        
        logger.info("Starting vector store rebuild...")
        
        # Ingest PDF
        chunks = ingest_pipeline(PDF_PATH)
        
        # Build vector store
        store = build_vector_store(chunks)
        
        return jsonify({
            "status": "rebuilt",
            "document_count": len(chunks),
            "message": "Vector store rebuilt successfully.",
        })
        
    except FileNotFoundError as e:
        return jsonify({
            "error": "PDF not found",
            "message": str(e),
        }), 404
        
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        return jsonify({
            "error": "Rebuild failed",
            "message": str(e),
        }), 500