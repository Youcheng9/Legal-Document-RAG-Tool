from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

def get_collection(persist_path: str, collection_name: str, embed_model: str):
    client = chromadb.PersistentClient(path=persist_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model
    )
    return client.get_or_create_collection(
        collection_name,
        embedding_function=embedding_fn
    )

def upsert_document(collection, source_name: str, documents: List[str], metadatas: List[Dict]):
    # Delete old chunks for this source
    collection.delete(where={"source": source_name})

    ids = [f"{source_name}_chunk_{i}" for i in range(len(documents))]
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

def retrieve(collection, query: str, source_name: str, top_k: int):
    return collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"source": source_name}
    )
