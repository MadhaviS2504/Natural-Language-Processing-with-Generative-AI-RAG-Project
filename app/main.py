"""
app/main.py
-----------
Streamlit web application for MedRAG.

Provides a user-friendly interface for:
- Medical question answering
- Viewing retrieved sources
- Adjusting retrieval parameters
"""

import streamlit as st
import requests
import time
from typing import Optional, Dict

# ─── Configuration ────────────────────────────────────────────────────────────

# API base URL - can be configured via environment variable
API_BASE_URL = st.session_state.get("API_BASE_URL", "http://localhost:5000")

# Page configuration
st.set_page_config(
    page_title="MedRAG - Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
    }
    .answer-box {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1E88E5;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ─── API Helper Functions ────────────────────────────────────────────────────

def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_rag_status() -> Dict:
    """Get RAG pipeline status."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/rag/status", timeout=10)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def submit_query(query: str, top_k: int = 3) -> Optional[Dict]:
    """
    Submit a query to the RAG API.
    
    Args:
        query: The medical question to answer.
        top_k: Number of chunks to retrieve.
    
    Returns:
        Response dictionary or None if error.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/rag/query",
            json={"query": query, "top_k": top_k},
            timeout=120,  # Long timeout for LLM inference
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json()
            st.error(f"Error: {error_data.get('error', 'Unknown error')}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may be loading or processing.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure the Flask server is running.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


# ─── UI Components ───────────────────────────────────────────────────────────

def render_header():
    """Render the application header."""
    st.markdown('<div class="main-header">🩺 MedRAG - Medical Assistant</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #757575; margin-bottom: 2rem;">
        Retrieval-Augmented Generation powered by Mistral-7B and Merck Manual
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with settings and information."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Configuration
        st.subheader("API Configuration")
        api_url = st.text_input(
            "API Base URL",
            value=API_BASE_URL,
            help="The base URL of the Flask API server"
        )
        st.session_state["API_BASE_URL"] = api_url
        
        # Retrieval Settings
        st.subheader("Retrieval Settings")
        top_k = st.slider(
            "Number of sources (top_k)",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant document chunks to retrieve"
        )
        
        # System Info
        st.subheader("📊 System Status")
        
        if check_api_health():
            st.success("✅ API Connected")
            
            # Get RAG status
            status = get_rag_status()
            if status.get("status") == "ready":
                vector_store = status.get("vector_store", {})
                st.info(f"📚 Documents indexed: {vector_store.get('document_count', 'N/A')}")
            else:
                st.warning("⚠️ Vector store not initialized")
        else:
            st.error("❌ API Not Connected")
            st.info("Please start the Flask API server:")
            st.code("python -m api.app", language="bash")
        
        # Sample Queries
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "What is the protocol for managing sepsis?",
            "What are the symptoms of appendicitis?",
            "What causes patchy hair loss?",
            "How is traumatic brain injury treated?",
            "What is the treatment for a leg fracture?",
        ]
        
        for sq in sample_queries:
            if st.button(sq, key=f"sample_{sq[:20]}"):
                st.session_state["query_input"] = sq
                st.rerun()
        
        st.markdown("---")
        st.caption("""
        **MedRAG** uses Retrieval-Augmented Generation to provide 
        evidence-based medical information from the Merck Manual.
        
        ⚠️ This is for informational purposes only. 
        Always consult a qualified healthcare professional.
        """)


def render_chat_interface():
    """Render the main chat interface."""
    st.subheader("💬 Ask a Medical Question")
    
    # Query input
    query = st.text_area(
        "Enter your medical question:",
        value=st.session_state.get("query_input", ""),
        height=100,
        placeholder="e.g., What is the treatment for sepsis?",
        key="query_text"
    )
    
    # Clear query input from session state
    if "query_input" in st.session_state:
        del st.session_state["query_input"]
    
    # Submit button
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("🔍 Get Answer", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.session_state["query_text"] = ""
        st.rerun()
    
    # Process query
    if submit_button and query.strip():
        with st.spinner("🔎 Retrieving relevant medical context..."):
            result = submit_query(query, top_k=top_k)
        
        if result:
            # Display answer
            st.markdown("---")
            st.subheader("📝 Answer")
            
            st.markdown(f"""
            <div class="answer-box">
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources
            st.subheader("📚 Sources")
            
            for i, source in enumerate(result.get("sources", []), 1):
                st.markdown(f"""
                <div class="source-box">
                    <strong>Source {i}</strong> (Page {source.get('page', 'Unknown')})<br>
                    <small>{source.get('content', '')[:300]}...</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Display processing time
            if result.get("processing_time_ms"):
                st.caption(f"⏱️ Processed in {result['processing_time_ms']:.2f}ms")


# ─── Main Application ─────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    render_header()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()