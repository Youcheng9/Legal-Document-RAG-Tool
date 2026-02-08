from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class UploadResponse(BaseModel):
    file_id: str
    filename: str


class IngestResponse(BaseModel):
    file_id: str
    source: str
    chunks: int
    status: str
    entities: Optional[Dict[str, List[str]]] = None


class SourceItem(BaseModel):
    # For traceability + citations
    chunk_id: Optional[str] = None
    file_id: Optional[str] = None          # UUID doc key (source_name)
    filename: Optional[str] = None         # Optional: only if you store it somewhere

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    snippet: Optional[str] = None   # What you show in the UI (avoid returning full chunk text)

    score: Optional[float] = None  # Retrieval score (if available)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    file_id: Optional[str] = None
    top_k: int = 8


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    retrieved: int


class ErrorResponse(BaseModel):
    detail: str
    extra: Optional[Dict[str, Any]] = None
