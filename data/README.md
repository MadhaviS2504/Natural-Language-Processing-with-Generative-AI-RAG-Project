# Data Directory

This directory should contain the Merck Manual PDF file.

## Required File

Place the **Merck Manual (Professional Version)** PDF here:

```
data/merck_manual.pdf
```

## How to Obtain the Merck Manual

1. **Download from Official Source**:
   - Visit: https://www.merckmanuals.com/professional
   - Click "Download PDF" or similar option
   - Save as `merck_manual.pdf` in this directory

2. **Alternative Sources**:
   - Check your institution's library resources
   - Some universities provide access through their libraries

## File Requirements

- **Format**: PDF
- **Size**: ~50-100 MB (full professional version)
- **Pages**: ~4,000+ pages covering 23 sections

## Note

The ingestion pipeline will automatically:
1. Load the PDF using PyMuPDF
2. Split it into chunks (1000 chars with 200 overlap)
3. Generate embeddings using sentence-transformers
4. Store in ChromaDB for semantic search

The first run will take several minutes to process the full document.