from typing import Dict, Optional
from pathlib import Path

from src.ingest.pdf_loader import load_pdf_and_texts
from src.ingest.chunking import validate_chunks
from src.ingest.entities import extract_entities
from src.rag.prompts import generate_prompt
from src.config import CHROMA_DIR, EMBED_MODEL, COLLECTION_NAME, OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, UPLOADS_DIR

# vectorstore helper functions (from your file)
from src.vectorstore.chroma_store import get_collection, upsert_document, retrieve

# Try to import your wrapper chat function if present; otherwise fall back to ollama.chat
def _load_ollama_chat():
    # Try a few possible module paths for the user's client wrapper
    candidates = [
        "llm.ollama_client",   # common snake_case module name
        "llm.ollama.client",   # if filename had a dot (less common)
        "llm.ollama",          # alternative
    ]
    for modname in candidates:
        try:
            import importlib
            mod = importlib.import_module(modname)
            if hasattr(mod, "chat"):
                return getattr(mod, "chat")
        except Exception:
            continue

    # Fallback: try the official ollama package (must be installed per requirements)
    try:
        import ollama as _ollama

        def _ollama_chat(model: str, prompt: str, temperature: float = 0.1, num_predict: int = 512) -> str:
            resp = _ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_predict": num_predict},
            )
            # defensive extraction
            return (resp.get("message", {}) or {}).get("content", str(resp))

        return _ollama_chat
    except Exception as e:
        # If nothing available, raise an informative error
        raise ImportError(
            "Could not import a chat() function for Ollama. "
            "Ensure your llm wrapper (llm/ollama.client.py) is importable or that 'ollama' package is installed."
        ) from e


# instantiate chat function once
_ollama_chat = _load_ollama_chat()


class RagPipeline:
    def __init__(self,
                chroma_persist_dir: Optional[Path] = None,
                collection_name: Optional[str] = None,
                embed_model: Optional[str] = None,
                ollama_model: Optional[str] = None,
                uploads_dir: Optional[Path] = None,  # add this
                chunk_size: Optional[int] = None,    # add this
                chunk_overlap: Optional[int] = None, # add this
    ):
    
        self.chroma_persist_dir = chroma_persist_dir or CHROMA_DIR
        self.uploads_dir = uploads_dir or UPLOADS_DIR
        self.collection_name = collection_name or COLLECTION_NAME
        self.embed_model = embed_model or EMBED_MODEL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP

        # Create / open collection (Chroma handles embedding function internally)
        self.collection = get_collection(str(self.chroma_persist_dir), self.collection_name, self.embed_model)


    # -----------------------
    # Ingestion
    # -----------------------
    def ingest_file_id(self, file_id: str, force: bool = False) -> Dict:
      
        pdf_path = self.uploads_dir / f"{file_id}.pdf"

        # Error handling for not exisiting PDFs
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found for file_id={file_id}: {pdf_path}")

        pages, _ = load_pdf_and_texts(pdf_path)  # May raise FileNotFoundError / ValueError
        source_name = file_id
        

        # Build 'all_text' list expected by validate_chunks (keys: text, source, page)
        all_texts = []
        for p in pages:
            all_texts.append({
                "text": p.get("text", ""),
                "source": source_name,
                "page": p.get("page")
            })

        # Chunking
        split_texts, chunk_metadatas = validate_chunks(
            all_texts, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap)

        # If no chunks extracted, return info denoting that
        if not split_texts:
            return {"file_id": file_id, "source": source_name, "chunks": 0, "status": "no_text"}

        # Prepare documents and metadatas for Chroma
        documents = [c["text"] for c in split_texts]
        metadatas = chunk_metadatas  # already contains source, page, chunk_length, etc.

        # Upsert to chroma (delete old source chunks inside upsert_document)
        upsert_document(self.collection, source_name, documents, metadatas)

        # Document-level entities (optional) — store/return for downstream use
        entities = extract_entities(all_texts)

        return {
            "file_id": file_id,
            "source": source_name,
            "chunks": len(documents),
            "status": "ingested",
            "entities": entities,
        }

    # -----------------------
    # Querying / Answering
    # -----------------------
    def answer(self, user_query: str, source_name: Optional[str] = None, top_k: int = 5) -> Dict:
      
        # Retrieve from Chroma
        raw = retrieve(self.collection, user_query, source_name, top_k)

        # Chroma returns nested lists for each input query; we used single query -> index 0
        docs = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]  # lower distance = more similar (depends on metric)

        # Build a flat list of candidate chunks
        candidates = []
        for i in range(len(docs)):
            meta = metadatas[i] if i < len(metadatas) else {}
            candidates.append({
                "id": ids[i] if i < len(ids) else None,
                "source": meta.get("source", None),
                "page": meta.get("page", None),
                "text": docs[i],
                "score": (1.0 - distances[i]) if distances and i < len(distances) and distances[i] is not None else None,
            })

        context_parts = []
        for c in candidates:
            src = c.get("source") or "unknown_source"
            page = c.get("page") or "?"
            excerpt = c.get("text") or ""
            # Optionally truncate long excerpts to keep prompt small:
            if len(excerpt) > 1200:
                excerpt = excerpt[:1200].rstrip() + "..."
            context_parts.append(f"[{src} | page:{page}] {excerpt}")
        context = "\n\n".join(context_parts)

        # Build prompt
        prompt = generate_prompt(context, user_query)

        # Call Ollama
        answer_text = _ollama_chat(self.ollama_model, prompt, temperature=0.0, num_predict=512)

        # Return structured response
        sources_out = []
        for c in candidates:
            sources_out.append({
                "id": c.get("id"),
                "source": c.get("source"),
                "page": c.get("page"),
                "text": (c.get("text")[:600] + "...") if c.get("text") and len(c.get("text")) > 600 else c.get("text"),
                "score": c.get("score"),
            })

        return {
            "answer": answer_text,
            "sources": sources_out,
            "retrieved": len(sources_out)
        }
