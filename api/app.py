"""
api/app.py
----------
Flask application factory and configuration.

Creates and configures the Flask app with:
- CORS support
- JSON encoding configuration
- Error handlers
- Blueprint registration
"""

import logging
from flask import Flask, jsonify
from flask_cors import CORS

from src.config import API_HOST, API_PORT, API_DEBUG

logger = logging.getLogger(__name__)


def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)
    
    # Configuration
    app.config["JSON_SORT_KEYS"] = False
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
    
    # Enable CORS for all routes
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    # Register blueprints
    from api.routes.health import health_bp
    from api.routes.rag import rag_bp
    
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(rag_bp, url_prefix="/api/rag")
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found", "message": str(error)}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return jsonify({"error": "Internal server error", "message": str(error)}), 500
    
    @app.route("/")
    def index():
        return jsonify({
            "name": "MedRAG API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/api/health",
                "rag_query": "/api/rag/query",
                "rag_status": "/api/rag/status",
            }
        })
    
    logger.info("Flask application created successfully.")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)