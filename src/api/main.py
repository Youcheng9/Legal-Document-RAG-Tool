from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import uuid

from src.rag.pipeline import RagPipeline
from src.config import UPLOADS_DIR

from src.api.schemas import UploadResponse, IngestResponse, QueryRequest, QueryResponse

app = FastAPI(title="Legal RAG API")

rag = RagPipeline()
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Legal RAG API is running. Visit /docs"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{file_id}.pdf"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return UploadResponse(file_id=file_id, filename=file.filename)


@app.post("/ingest/{file_id}", response_model=IngestResponse)
def ingest(file_id: str):
    """
    Ingest a previously uploaded PDF.
    """
    result = rag.ingest_file_id(file_id)

    # If your pipeline returns a dict already matching IngestResponse, this works:
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a question against ingested documents.
    If file_id is provided, search is restricted to that document.
    """
    result = rag.answer(
        user_query=req.question,
        source_name=req.file_id,
        top_k=req.top_k,
    )

    # If your pipeline returns a dict already matching QueryResponse, this works:
    return QueryResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}
