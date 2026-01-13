from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

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

# split_texts, chunk_metadatas = validate_chunks(all_text)