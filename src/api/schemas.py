from pydantic import BaseModel
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
    id: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    retrieved: int


# Optional: for standardized errors if you want to return structured errors
class ErrorResponse(BaseModel):
    detail: str
    extra: Optional[Dict[str, Any]] = None
