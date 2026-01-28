"""
Chunking utilities for ingestion.

Expects `all_texts` to be a list of dicts like:
  {"text": "...", "source": "doc-id-or-filename", "page": 1}

Returns a tuple:
  (split_texts, chunk_metadatas)

- split_texts: list of dicts: {"text": <chunk_text>, "source": ..., "page": ..., "chunk_index": ...}
- chunk_metadatas: list of dicts aligned 1:1 with split_texts: {"source":..., "page":..., "chunk_length":..., "word_count":..., "chunk_index":...}
"""

from typing import List, Dict, Tuple
import logging

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    raise ImportError(
        "langchain_text_splitters is required for chunking. "
        "Install the package listed in requirements.txt and try again."
    ) from e

logger = logging.getLogger(__name__)


def validate_chunks(
    all_texts: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Tuple[List[Dict], List[Dict]]:


    # create the splitter with the separators you provided originally
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
)


    split_texts: List[Dict] = []
    chunk_metadatas: List[Dict] = []

    for page in all_texts:
        page_text = page.get("text", "") or ""
        source = page.get("source")
        page_no = page.get("page")

        # skip extremely short pages
        if len(page_text.strip()) < 50:
            continue

        # create_documents expects a list of strings
        docs = text_splitter.create_documents([page_text])

        # docs are LangChain-style Document objects; use page_content and optional metadata
        for idx, d in enumerate(docs):
            # obtain text content
            content = getattr(d, "page_content", None)
            if content is None:
                # fallback if object shape differs
                content = str(d)

            content = content.strip()
            # skip tiny/low-value chunks
            if len(content) < 100:
                continue

            chunk_index = None
            # If the returned Document has metadata with chunk index, prefer it
            doc_meta = getattr(d, "metadata", None) or {}
            if isinstance(doc_meta, dict) and "chunk_index" in doc_meta:
                chunk_index = doc_meta["chunk_index"]
            else:
                # fallback to the order within this page's docs
                chunk_index = idx

            chunk_entry = {
                "text": content,
                "source": source,
                "page": page_no,
                "chunk_index": chunk_index,
            }

            metadata = {
                "source": source,
                "page": page_no,
                "chunk_length": len(content),
                "word_count": len(content.split()),
                "chunk_index": chunk_index,
            }

            split_texts.append(chunk_entry)
            chunk_metadatas.append(metadata)

    logger.info("Created %d valid chunks", len(split_texts))
    return split_texts, chunk_metadatas

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from typing import List, Dict

# def validate_chunks(all_texts: List[Dict],
#                     chunk_size: int = 1000,
#                     chunk_overlap: int = 200) -> tuple:



#     text_splitter = RecursiveCharacterTextSplitter(
#       chunk_size = chunk_size,
#       chunk_ovrlap = chunk_overlap,
#       length_function = len,
#       separators=["\n\n","\n", ". ", " ", ""]
#     )

#     split_texts = []
#     chunk_metadatas = []

#     for p in all_texts:

#       # Skip very short pages
#       if len(p["text"]) < 50:
#         continue

#       docs = text_splitter.create_documents([p["text"]])

#       for d in docs:

#         content = d.page_content.strip()

#         # Skip chunks that are way too short or just white space
#         if len(content) < 100:
#           continue

#         split_texts.append(d)
#         chunk_metadatas.append({
#             "source": p["source"],
#             "page": p["page"],
#             "chunk_length": len(content),
#             "word_count": len(content.split())
#         })
#     print(f"Created {len(split_texts)} valid chunks")
#     return split_texts, chunk_metadatas

# # split_texts, chunk_metadatas = validate_chunks(all_text)