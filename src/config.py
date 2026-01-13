from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_PAGE_CHARS = int(os.getenv("MIN_PAGE_CHARS", "50"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "100"))

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
