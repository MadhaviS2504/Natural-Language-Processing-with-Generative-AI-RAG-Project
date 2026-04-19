# Contributing to MedRAG

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/rag-medical-assistant.git
cd rag-medical-assistant
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Code Style

This project uses:
- **Black** for formatting: `black src/`
- **Flake8** for linting: `flake8 src/`

## Running Tests

```bash
pytest tests/ -v
```

## Pull Request Guidelines

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a PR with a clear description of changes

## Areas for Contribution

- Hybrid search (BM25 + semantic)
- Streamlit UI
- FastAPI backend
- Additional evaluation metrics
- Support for additional document formats
