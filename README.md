# 🩺 MedRAG — AI Medical Assistant (RAG-based LLM System)

> **An end-to-end Retrieval-Augmented Generation (RAG) system that empowers healthcare professionals with instant, grounded, context-aware answers from the Merck Medical Manual — 4,000+ pages of clinical knowledge, at your fingertips.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0.3.27-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LLaMA-cpp-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Sentence%20Transformers-Embeddings-E76F51?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Mistral-7B-264653?style=for-the-badge"/>
</p>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution & Architecture](#-solution--architecture)
- [Tech Stack](#-tech-stack)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Evaluation Methodology](#-evaluation-methodology)
- [Results & Key Findings](#-results--key-findings)
- [Future Improvements](#-future-improvements)
- [Business Impact](#-business-impact)

---

## 🏥 Problem Statement

Healthcare professionals face a **critical information bottleneck**: thousands of pages of clinical knowledge, yet no fast way to extract accurate, context-aware answers.

| Challenge | Impact |
|---|---|
| Information overload from 4,000+ page manuals | Slow decision-making in critical moments |
| Traditional keyword search lacks medical context | Missed diagnoses, incomplete treatment plans |
| LLMs alone hallucinate clinical facts | Unsafe for medical settings without grounding |
| No standardized knowledge retrieval tool | Inconsistent care practices across departments |

> **Real queries this system answers:**
> - *"What is the protocol for managing sepsis in a critical care unit?"*
> - *"Can appendicitis be cured via medicine, or is surgery required?"*
> - *"What are the diagnostic steps for suspected endocrine disorders?"*

---

## 💡 Solution & Architecture

This project implements a **production-grade RAG pipeline** that grounds every LLM response in verified medical text from the Merck Manual, dramatically reducing hallucinations.

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
│                                                                 │
│  📄 Merck Manual PDF (4,000+ pages)                            │
│         │                                                       │
│         ▼                                                       │
│  🔪 Text Chunking (RecursiveCharacterTextSplitter)              │
│         │                                                       │
│         ▼                                                       │
│  🔢 Embedding Generation (sentence-transformers)                │
│         │                                                       │
│         ▼                                                       │
│  🗃️  Vector Storage (ChromaDB)                                  │
│         │                                                       │
│         ▼                                                       │
│  🔍 Semantic Retrieval (Top-K relevant chunks)                  │
│         │                                                       │
│         ▼                                                       │
│  🧠 LLM Generation (Mistral-7B via llama-cpp)                  │
│         │                                                       │
│         ▼                                                       │
│  ✅ Grounded, Context-Aware Medical Answer                      │
└─────────────────────────────────────────────────────────────────┘
```

### Three-Stage Comparison

| Approach | Description | Quality |
|---|---|---|
| **Baseline LLM** | Raw LLM response, no context | ⚠️ Prone to hallucination |
| **Prompt-Engineered LLM** | Structured prompts, no retrieval | 🔶 Better structure, still ungrounded |
| **RAG System** | Retrieval + Prompt Engineering + LLM | ✅ Grounded, accurate, reliable |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Mistral-7B (GGUF via llama-cpp) | Text generation |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Semantic vector representation |
| **Vector DB** | ChromaDB | Efficient similarity search |
| **Orchestration** | LangChain 0.3.27 | Pipeline management |
| **PDF Parsing** | PyMuPDF | Document ingestion |
| **Tokenization** | tiktoken | Token counting & management |
| **Runtime** | Google Colab (GPU: T4) | Accelerated inference |
| **Language** | Python 3.10+ | Core implementation |

---

## ⚡ Key Features

- 🔍 **Semantic Search** — Finds clinically relevant passages even with varied phrasing
- 📎 **Context Grounding** — Every answer is sourced from verified Merck Manual content
- 🧪 **LLM-as-a-Judge Evaluation** — Automated quality scoring using groundedness + relevance metrics
- 🎛️ **Configurable Retrieval** — Adjustable top-k, chunk size, and overlap parameters
- 🧩 **Modular Design** — Ingestion, retrieval, and generation are fully decoupled
- 📊 **Prompt Engineering Baseline** — Side-by-side comparison of all three approaches
- 🚀 **GPU-Accelerated** — CUDA-enabled llama-cpp for fast inference

---

## 📂 Project Structure

```
rag-medical-assistant/
│
├── 📁 src/
│   ├── ingestion.py          # PDF loading & chunking logic
│   ├── embeddings.py         # Embedding generation & management
│   ├── retriever.py          # ChromaDB vector store & retrieval
│   ├── generator.py          # LLM loading & response generation
│   ├── rag_pipeline.py       # End-to-end RAG orchestration
│   └── evaluator.py          # LLM-as-a-judge evaluation module
│
├── 📁 notebooks/
│   └── MedRAG_Full_Pipeline.ipynb   # Complete experiment notebook
│
├── 📁 data/
│   └── README.md             # Data sourcing instructions (Merck Manual)
│
├── 📁 embeddings/
│   └── .gitkeep              # ChromaDB persistence directory
│
├── 📁 tests/
│   ├── test_ingestion.py     # Unit tests for data pipeline
│   ├── test_retriever.py     # Unit tests for retrieval
│   └── test_generator.py     # Unit tests for generation
│
├── 📁 assets/
│   └── architecture.png      # System architecture diagram
│
├── app.py                    # (Future) Streamlit UI entrypoint
├── requirements.txt          # All dependencies pinned
├── config.py                 # Centralized configuration
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU
- ~8GB RAM minimum

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-medical-assistant.git
cd rag-medical-assistant
```

### 2. Install Dependencies

```bash
# GPU (recommended)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.85
pip install -r requirements.txt

# CPU fallback
CMAKE_ARGS="-DLLAMA_CUBLAS=off" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.85
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your HuggingFace token and model path
```

### 4. Add Data

```bash
# Place the Merck Manual PDF in the data/ directory
# See data/README.md for sourcing instructions
```

### 5. Run the Pipeline

```bash
# Run full RAG pipeline
python src/rag_pipeline.py

# Or open the notebook
jupyter notebook notebooks/MedRAG_Full_Pipeline.ipynb
```

---

## 🔬 How It Works

### Step 1 — Document Ingestion
The Merck Manual PDF (4,000+ pages, 23 sections) is loaded using `PyMuPDFLoader` and split into semantically meaningful chunks using `RecursiveCharacterTextSplitter` with carefully tuned `chunk_size` and `chunk_overlap` to preserve clinical context across boundaries.

### Step 2 — Embedding & Indexing
Each chunk is embedded using `sentence-transformers` to generate dense vector representations. These are stored in a **ChromaDB** persistent vector store, enabling sub-second similarity search at query time.

### Step 3 — Retrieval
At query time, the user's question is embedded and compared against all stored chunks via cosine similarity. The top-k most relevant passages are retrieved and injected into the LLM context window.

### Step 4 — Prompt Engineering
A carefully designed system prompt instructs the Mistral-7B model to answer **strictly from the provided context**, cite limitations where information is absent, and structure responses clearly for medical professionals.

### Step 5 — Generation
The Mistral-7B model (running locally via llama-cpp with CUDA) generates a grounded, coherent response using the retrieved medical context.

---

## 📏 Evaluation Methodology

This project uses **LLM-as-a-Judge** — a self-evaluation framework where the same LLM scores response quality on two critical axes:

| Metric | Description | What it measures |
|---|---|---|
| **Groundedness** | Is the answer supported by the retrieved context? | Hallucination prevention |
| **Relevance** | Does the answer actually address the user's question? | Response quality |

Both scores are generated automatically, enabling scalable, consistent evaluation without human annotation.

---

## 📈 Results & Key Findings

### Qualitative Comparison (Query: Sepsis Protocol)

| Approach | Response Quality | Observations |
|---|---|---|
| Baseline LLM | Generic, vague | No specific protocols, possible hallucinations |
| Prompt-Engineered | Structured, clearer | Better format but still not grounded in manual |
| **RAG System** | **Specific, cited** | **Directly references Merck Manual content** |

### Key Takeaways

- RAG **significantly reduces hallucination** compared to baseline LLM
- Prompt engineering alone improves structure but cannot substitute grounding
- ChromaDB retrieval consistently surfaces relevant clinical passages
- LLM-as-a-Judge provides a scalable, automated quality gate

---

## 🔮 Future Improvements

| Enhancement | Description | Priority |
|---|---|---|
| 🔀 Hybrid Search | Combine semantic + BM25 keyword search for better recall | High |
| 🌐 FastAPI Backend | RESTful API for production deployment | High |
| 🖥️ Streamlit UI | Interactive chat interface for clinical staff | Medium |
| ☁️ HuggingFace Spaces | One-click cloud deployment | Medium |
| 📊 Evaluation Dashboard | Visual metrics tracking over time | Medium |
| 🔄 Re-ranking | Cross-encoder re-ranking for better precision | High |
| 📚 Multi-document | Support for multiple medical knowledge bases | Low |
| 🔐 Auth & Audit | Logging & access control for clinical settings | High |

---

## 💼 Business Impact

This system directly addresses healthcare operational challenges:

- **⏱️ Time Savings**: Reduces information retrieval time from minutes → seconds
- **🎯 Decision Support**: Grounds clinical decisions in authoritative medical literature
- **📋 Standardization**: Consistent reference to the same canonical source (Merck Manual)
- **🧠 Knowledge Democratization**: Junior clinicians access senior-level knowledge instantly
- **⚕️ Patient Safety**: Reduced hallucination = reduced risk of erroneous recommendations

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> ⚠️ **Medical Disclaimer**: This system is intended for research and educational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 🙋 About

Built as an end-to-end demonstration of production-grade RAG architecture for AI/ML engineering roles. Showcases skills in LLM integration, vector databases, prompt engineering, and evaluation frameworks.

**Relevant Skills Demonstrated**: RAG • LLMs • LangChain • Vector Databases • Prompt Engineering • NLP • Python • ChromaDB • Sentence Transformers • LLM Evaluation

---

<p align="center">⭐ Star this repo if you found it useful!</p>
