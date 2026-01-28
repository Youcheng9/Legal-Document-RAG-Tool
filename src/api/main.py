from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

from src.rag.pipeline import RagPipeline
from src.config import UPLOADS_DIR, DEFAULT_TOP_K

app = FastAPI(title="Legal RAG API")

# Initialize pipeline once
rag = RagPipeline()

# Ensure uploads directory exists
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Legal RAG API is running. Visit /docs"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{file_id}.pdf"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "file_id": file_id,
        "filename": file.filename
    }


@app.post("/ingest/{file_id}")
def ingest(file_id: str):
    """
    Ingest a previously uploaded PDF.
    """
    result = rag.ingest_file_id(file_id)
    return result


@app.post("/query")
def query(
    question: str,
    file_id: str | None = None,
    top_k: int = DEFAULT_TOP_K
):
    """
    Ask a question against ingested documents.
    If file_id is provided, search is restricted to that document.
    """
    result = rag.answer(
        user_query=question,
        source_name=file_id,
        top_k=top_k
    )
    return result


@app.get("/health")
def health():
    return {"status": "ok"}

# from fastapi import FastAPI, UploadFile, File
# import uvicorn
# from fastapi.responses import HTMLResponse

# app = FastAPI()

# @app.get("/", response_class=HTMLResponse)
# def home():
#     html_content = """
#     <html>
#     <head>
#         <title>Legal Document RAG Tool</title>
#     </head>
#     <body>
#         <h1>Welcome to the Legal Document RAG Tool!</h1>
#         <p>This is a tool that allows you to analyze legal documents using RAG.</p>
#         <p>To get started, please upload a legal document below.</p>
#         <form action="/upload" method="post" enctype="multipart/form-data">
#             <input type="file" name="file" />
#             <button type="submit">Upload</button>
#         </form>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content, status_code=200)

# @app.post("/analyze")
# def analyze(query: str):
#     return {"message": "Analyzing the document..."}

# @app.post("/upload")
# def upload(file: UploadFile = File(...)):
#     return {"message": "Uploading the document..."}

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)