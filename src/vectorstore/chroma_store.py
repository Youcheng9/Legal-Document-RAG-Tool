from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions


def get_collection(persist_path: str, collection_name: str, embed_model: str):
    client = chromadb.PersistentClient(path=persist_path)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model
    )

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )


def upsert_document(collection, source_name: str, documents: List[str], metadatas: List[Dict]):
    """
    Upsert all chunks for a single document (source_name).
    Old chunks for this source are deleted first.
    """

    # Delete old chunks for this document (safe to ignore failures)
    try:
        collection.delete(where={"source": source_name})
    except Exception:
        pass

    # Stable-ish chunk IDs
    ids = [
        f"{source_name}::chunk::{i}"
        for i in range(len(documents))
    ]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )


def retrieve(collection, query: str, source_name: Optional[str], top_k: int):
    where_filter = {"source": source_name} if source_name else None
    return collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )