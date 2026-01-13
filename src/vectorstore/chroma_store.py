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
