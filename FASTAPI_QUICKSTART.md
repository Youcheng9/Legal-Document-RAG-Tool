# FastAPI Quick Start Guide

## Overview

Your FastAPI application is now complete with the following features:
- **Upload PDF documents** and process them into chunks
- **Query documents** using RAG (Retrieval Augmented Generation)
- **Health check** endpoint to verify services
- **Delete documents** from the vector store
- **List documents** in the store

## Prerequisites

1. **Ollama installed and running**
   ```bash
   # Start Ollama (if not already running)
   ollama serve
   
   # Pull the required model
   ollama pull llama3.1:8b
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy model** (for entity extraction)
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the API

### Option 1: Using the run script (Recommended)
```bash
python run_api.py
```

### Option 2: Using uvicorn directly
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Run directly from main.py
```bash
python src/api/main.py
```

The API will be available at: **http://localhost:8000**

## API Endpoints

### 1. Health Check
```bash
GET /health
```

Check if the API and Ollama are available.

**Response:**
```json
{
  "status": "healthy",
  "ollama_available": true,
  "documents_count": 0
}
```

### 2. Upload Document
```bash
POST /upload
Content-Type: multipart/form-data
```

Upload and process a PDF document.

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@path/to/your/document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "chunks_created": 42,
  "message": "Successfully processed 42 chunks from document.pdf"
}
```

### 3. Query Documents
```bash
POST /query
Content-Type: application/json
```

Query documents and get an AI-generated answer.

**Request:**
```json
{
  "question": "What is the salary?",
  "source": "document.pdf",  // Optional: filter by source
  "top_k": 10                 // Optional: number of chunks to retrieve
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the salary?",
    "top_k": 10
  }'
```

**Response:**
```json
{
  "answer": "According to page 3, the salary is $100,000 per year...",
  "sources": ["document.pdf"],
  "context_chunks": ["chunk1...", "chunk2..."],
  "confidence": 0.85
}
```

### 4. Delete Document
```bash
DELETE /documents
Content-Type: application/json
```

Delete a document and all its chunks from the vector store.

**Request:**
```json
{
  "source": "document.pdf"
}
```

### 5. List Documents
```bash
GET /documents
```

Get information about documents in the vector store.

### 6. Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Example Workflow

1. **Start Ollama** (if not running):
   ```bash
   ollama serve
   ```

2. **Start the API**:
   ```bash
   python run_api.py
   ```

3. **Upload a document**:
   ```bash
   curl -X POST "http://localhost:8000/upload" \
     -F "file=@your_legal_document.pdf"
   ```

4. **Query the document**:
   ```bash
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the key terms of this agreement?"}'
   ```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": f}
    )
    print(response.json())

# Query documents
response = requests.post(
    f"{BASE_URL}/query",
    json={
        "question": "What is the salary?",
        "top_k": 10
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## Configuration

You can configure the API using environment variables:

```bash
# Ollama configuration
export OLLAMA_HOST="http://127.0.0.1:11434"
export OLLAMA_MODEL="llama3.1:8b"

# ChromaDB configuration
export COLLECTION_NAME="legal_documents"
export EMBED_MODEL="all-MiniLM-L6-v2"

# Chunking configuration
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=200
```

## Troubleshooting

1. **Ollama not available**: Ensure Ollama is running (`ollama serve`)
2. **Import errors**: Make sure you're in the project root directory
3. **PDF processing errors**: Check that the PDF is not encrypted or corrupted
4. **Port already in use**: Change the port in `run_api.py` or use `--port` flag with uvicorn

## Project Structure

```
src/
├── api/
│   ├── main.py       # FastAPI application
│   └── schemas.py    # Pydantic models
├── config.py         # Configuration settings
├── ingest/           # Document ingestion
│   ├── pdf_loader.py
│   └── chunking.py
├── llm/              # LLM client (Ollama)
│   └── ollama_client.py
├── rag/              # RAG pipeline
│   ├── pipeline.py
│   └── prompts.py
└── vectorstore/      # Vector database (ChromaDB)
    └── chroma_store.py
```
