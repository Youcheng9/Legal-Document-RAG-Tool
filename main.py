# ✅ Real dataset
# ✅ EDA + feature analysis
# ✅ Model comparison
# ✅ Evaluation metrics
# ✅ Visualization
# ✅ Deployment
# ✅ Clear business question

# 1. Get data by scraping or API
# 2. Process data
# 3. EDA for business insights
# 4. Modeling and evaluating
# 5. Deploy



# -- TODO
# 1) Get 8K files PDFs (Upload manually)
#   1.1) Create doc_id + naming/versioning rules (draft/final/signed)
#   1.2) Deduplicate (hash files) + track source + upload timestamps
#   1.3) Store raw PDFs + maintain folder/tenant permissions

# 2) Read the docs (Ingestion + parsing)
#   2.1) Detect digital vs scanned PDFs
#   2.2) Extract text from digital PDFs
#   2.3) OCR scanned pages (only when needed)
#   2.4) Handle edge cases (encrypted/corrupt PDFs, retries, logging)
#   2.5) Save outputs: raw_text + page_map (text ↔ page numbers) + layout hints

# 3) Clean + normalize text
#   3.1) Remove headers/footers/page numbers/boilerplate repeats
#   3.2) Fix hyphenation, broken lines, whitespace, encoding
#   3.3) Preserve structure signals (section numbers, headings, clause numbering)

# 4) Tokenize or text splitting (Chunking)
#   4.1) Legal-aware splitting (by headings/sections/clauses when possible)
#   4.2) Fallback token chunking (size + overlap)
#   4.3) Attach chunk metadata (doc_id, section_title/path, page_start/end, clause_no)
#   4.4) (Optional) Classify chunk type (LoL, indemnity, termination, confidentiality)

# 5) Metadata extraction
#   5.1) Document-level fields (agreement type, parties, effective date, governing law)
#   5.2) Key dates (term, renewal, notice periods) + monetary caps (if present)
#   5.3) Store metadata in a DB (Postgres/SQLite) for filtering + UI

# 6) Create embeddings
#   6.1) Choose embedding model + batching strategy
#   6.2) Embed chunks (not whole docs)
#   6.3) Persist embeddings + chunk ids + metadata
#   6.4) Incremental embedding for new/changed docs only

# 7) Create vector storage (FAISS / Chroma / Pinecone / Qdrant)
#   7.1) Create collection/index with metadata filtering support
#   7.2) Add keyword index (BM25) for hybrid search (recommended for contracts)
#   7.3) Implement backup/rebuild strategy

# 8) Create Retriever for RAG
#   8.1) Query normalization (synonyms, defined terms, party names)
#   8.2) Hybrid retrieval (vector top-k + BM25 top-k)
#   8.3) Merge + dedupe + metadata filters (type, governing law, date ranges)
#   8.4) Rerank (cross-encoder or LLM rerank)
#   8.5) Return top-n chunks with citations (doc + page + section)

# 9) Answer generation
#   9.1) Prompt/template enforcing “answer only from context”
#   9.2) Always include citations + short quotes for each key claim
#   9.3) Structured outputs for extraction tasks (JSON schema)
#   9.4) “Not found” behavior when evidence is missing

# 10) Evaluation + regression
#   10.1) Build gold Q/A set (100–300 questions + expected citations)
#   10.2) Track recall@k, citation accuracy, hallucination rate, latency
#   10.3) Automated regression tests for chunking/embeddings/retrieval changes

# 11) Security + governance
#   11.1) AuthN/AuthZ (doc-level permissions, tenant isolation)
#   11.2) Encryption at rest/in transit + audit logs
#   11.3) Data retention + deletion workflows
#   11.4) PII handling (optional)

# 12) Monitoring + production hardening
#   12.1) Observability (logs, traces, dashboards)
#   12.2) Cost controls (caching, rate limits, batching)
#   12.3) Reliability (retries, dead-letter queue, reindex jobs)
#   12.4) Feedback loop (“helpful?”) → add misses to eval set

# -- SETUP Ollama Once with this function
# Go to 127.0.0.1:11434 to check if Ollama is running

# def setup_ollama():
#   import subprocess
#   import time

#   # Install
#   subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh"], stdout=subprocess.PIPE, shell=True)

#   # Start server
#   subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#   time.sleep(10) # Wait server to open

#   # Pull model
#   subprocess.run(["ollama", "pull", "llama3.1:8b"])

#   print("Ollama ready")

# setup_ollama()

import os
drive_path = '/content/drive/MyDrive/8k Employment Agreement.pdf'

from IPython.core.display import json
# Use pymupdf to open pdf for processing
# -- ERROR HANDLING FOR OPENING PDFs AND LOAD TEXT ---
import pymupdf
from typing import Tuple, List, Dict
import os

def load_pdf_and_texts(file_path: str) -> Tuple[List[Dict], str]:
  try:

    # File does not exist
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"PDF not found: {file_path}")

    with pymupdf.open(file_path) as doc:

      # The PDF has no pages
      if len(doc) == 0:
        raise ValueError("PDF has no pages")

      source_name = os.path.basename(file_path)
      all_text = []

      for page_idx, page in enumerate(doc, start=1):
        text = page.get_text()

        # Skip empty pages
        if text.strip():
          all_text.append({
              "page": page_idx,
              "text": text,
              "source": source_name
          })

      print(f"Loaded {len(doc)} pages from {source_name}")
      return all_text, source_name # Moved outside the loop

  except Exception as e:
      print(f"Error loading PDF: {e}")
      raise

all_text, source_name = load_pdf_and_texts(drive_path)

# print("\n",json.dumps(all_text, indent=2))
# Entites extraction

# import spacy

# nlp = spacy.load("en_core_web_sm")

# target_entities = {
#     "PERSON": [],
#     "ORG": [],
#     "DATE": [],
#     "MONEY": []
# }

# for p in all_text:
#     document = nlp(p["text"])
#     for ent in document.ents:
#       if ent.label_ in target_entities and ent.text not in target_entities[ent.label_]:
#         target_entities[ent.label_].append(ent.text)

# for entity_type, entities in target_entities.items():
#     print(f"{entity_type}: {entities}")



# -- ENTITIES EXTRACTION AND LOOKUP (OPTIMIZED USING SETS)
def extract_entities(all_text: List[Dict]) -> Dict:
  import spacy

  nlp = spacy.load("en_core_web_sm")

  target_entities = {
      "PERSON": set(),
      "ORG": set(),
      "DATE": set(),
      "MONEY": set(),
      "GPE": set(), # Location
      "LAW": set(), # Legal reference
  }

  texts = [p["text"] for p in all_text]

  # Batch processing using nlp.pipe, much faster
  for doc in nlp.pipe(texts, batch_size=50, n_process=1):
    for ent in doc.ents:
      # Corrected: Use ent.label_ instead of ent.labels
      if ent.label_ in target_entities:
        # Clean entity text
        cleaned = ent.text.strip()
        if cleaned and len(cleaned) > 1:
          target_entities[ent.label_].add(cleaned) # Use add for sets

  # Convert set back to sorted list
  return {k: sorted(list(v)) for k, v in target_entities.items()}


target_entities = extract_entities(all_text)


for entity_type, entities in target_entities.items():
    print(f"{entity_type}: {entities}")

# -- Chunking
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200,
#     length_function = len,
#     separators=["\n\n","\n", ". ", " ", ""]
# )

# # split_texts = text_splitter.create_documents(all_text)
# # use .split_documents for multiple documents**********

# # -- Now keeps metadata with chunk
# split_texts = []
# chunk_metadatas = []

# for p in all_text:
#     docs = text_splitter.create_documents([p["text"]])
#     for d in docs:
#         split_texts.append(d)
#         chunk_metadatas.append({
#             "source": source_name,
#             "page": p["page"],
#         })
from langchain_text_splitters import RecursiveCharacterTextSplitter
# -- CHUNKING AND VALIDATION FOR CHUNK QUALITY BEFORE STORING
def validate_chunks(all_texts: List[Dict],
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200) -> tuple:



    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = chunk_size,
      chunk_overlap = chunk_overlap,
      length_function = len,
      separators=["\n\n","\n", ". ", " ", ""]
    )

    split_texts = []
    chunk_metadatas = []

    for p in all_texts:

      # Skip very short pages
      if len(p["text"]) < 50:
        continue

      docs = text_splitter.create_documents([p["text"]])

      for d in docs:

        content = d.page_content.strip()

        # Skip chunks that are way too short or just white space
        if len(content) < 100:
          continue

        split_texts.append(d)
        chunk_metadatas.append({
            "source": p["source"],
            "page": p["page"],
            "chunk_length": len(content),
            "word_count": len(content.split())
        })
    print(f"Created {len(split_texts)} valid chunks")
    return split_texts, chunk_metadatas

split_texts, chunk_metadatas = validate_chunks(all_text)

# Vector base and embeddings

import chromadb
from chromadb.utils import embedding_functions


client = chromadb.PersistentClient(path="/content/chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
    )
collection = client.get_or_create_collection(
    "legal_documents",
    embedding_function=embedding_fn
    )

# -- Delete previous run results if they have
try:
    collection.delete(where={"source": source_name})
except Exception:
    pass

collection.add(
    documents=[text.page_content for text in split_texts],
    metadatas=chunk_metadatas,
    ids=[f"{source_name}_chunk_{i}" for i in range(len(split_texts))]
)

# import os
# import subprocess

# # Set environment variable for Ollama host to listen on all interfaces
# os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'

# # Start the Ollama server in the background
# subprocess.Popen(["ollama", "serve"])
# print("Ollama server launched in the background.")

# !pkill -f "ollama serve" || true
# !OLLAMA_HOST=127.0.0.1:11434 ollama serve

import threading
import subprocess
import time

def run_ollama_serve():
  subprocess.Popen(["ollama", "serve"])

thread = threading.Thread(target=run_ollama_serve)
thread.start()
time.sleep(5)

!nohup ollama serve &

!ollama pull llama3.1:8b

import ollama
# Query - also automatic
# Query retrieval with ollama
def query_with_ollama(user_query, collection, source_name, model="llama3.1:8b"):

    results = collection.query(
        query_texts=[user_query],
        n_results=3,
        where={"source": source_name}
    )

    context = "\n\n".join([
              f"[Chunk {i+1}]\n{doc}"
              for i, doc in enumerate(results['documents'][0])
    ])

    prompt = f"""You are a legal document analyzer. Based on the following excerpts from a legal document, answer the user's question accurately and concisely.

Document Excerpts:
{context}

User Question: {user_query}

Answer:"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response['message']['content']

answer = query_with_ollama(
    "What is the salary?",
    collection,
    source_name
)
print(answer)




