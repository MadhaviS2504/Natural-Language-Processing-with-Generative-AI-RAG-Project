"""
config.py
---------
Central configuration for the MedRAG pipeline.
All tunable parameters live here — no magic numbers scattered across files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

PDF_PATH = DATA_DIR / "merck_manual.pdf"
CHROMA_PERSIST_DIR = str(EMBEDDINGS_DIR / "chroma_db")

# ─── Model Configuration ──────────────────────────────────────────────────────
# Mistral-7B GGUF (quantized for local inference)
MODEL_REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_FILENAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# llama-cpp parameters
MODEL_CONTEXT_WINDOW = 4096
MODEL_GPU_LAYERS = 35          # Set to 0 for CPU-only inference
MODEL_THREADS = 8

# ─── Embedding Configuration ──────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ─── Chunking Configuration ───────────────────────────────────────────────────
CHUNK_SIZE = 1000              # Characters per chunk
CHUNK_OVERLAP = 200            # Overlap to preserve cross-boundary context
CHUNK_SEPARATORS = ["\n\n", "\n", ".", " ", ""]

# ─── Retrieval Configuration ──────────────────────────────────────────────────
TOP_K_RETRIEVAL = 3            # Number of chunks to retrieve per query
COLLECTION_NAME = "merck_manual"

# ─── Generation Configuration ─────────────────────────────────────────────────
MAX_TOKENS = 512
TEMPERATURE = 0.0              # Deterministic for medical context
TOP_P = 0.95
TOP_K = 50

# ─── Evaluation Configuration ─────────────────────────────────────────────────
EVAL_MAX_TOKENS = 128
EVAL_TEMPERATURE = 0.0

# ─── Sample Queries ───────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    "What is the protocol for managing sepsis in a critical care unit?",
    "What are the common symptoms for appendicitis, and can it be cured via medicine? "
    "If not, what surgical procedure should be followed to treat it?",
    "What are the effective treatments or solutions for addressing sudden patchy hair loss, "
    "commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?",
    "What treatments are recommended for a person who has sustained a physical injury to brain tissue, "
    "resulting in temporary or permanent impairment of brain function?",
    "What are the necessary precautions and treatment steps for a person who has fractured their leg "
    "during a hiking trip, and what should be considered for their care and recovery?",
]
