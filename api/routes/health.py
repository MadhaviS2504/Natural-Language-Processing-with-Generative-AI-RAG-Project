"""
api/routes/health.py
--------------------
Health check endpoint for the API.

Provides status information about:
- API status
- Vector store availability
- LLM availability
"""

import logging
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with health status.
    """
    # Check vector store
    vector_store_status = "unknown"
    try:
        from src.retriever import get_vector_store
        store = get_vector_store()
        if store is not None:
            vector_store_status = "ready"
    except Exception as e:
        vector_store_status = f"error: {str(e)}"
    
    # Check LLM
    llm_status = "unknown"
    try:
        from src.generator import load_llm
        llm = load_llm()
        if llm is not None:
            llm_status = "ready"
    except Exception as e:
        llm_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy",
        "components": {
            "api": "ready",
            "vector_store": vector_store_status,
            "llm": llm_status,
        }
    })


@health_bp.route("/ready", methods=["GET"])
def readiness_check():
    """
    Readiness check endpoint for Kubernetes/load balancers.
    
    Returns:
        JSON response indicating if the service is ready to accept traffic.
    """
    try:
        # Try to initialize components
        from src.retriever import get_vector_store
        from src.generator import load_llm
        
        # These will be cached after first call
        get_vector_store()
        load_llm()
        
        return jsonify({"ready": True}), 200
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({"ready": False, "error": str(e)}), 503