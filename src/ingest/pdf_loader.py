import pymupdf
from typing import Tuple, List, Dict, Optional
import os

def load_pdf_and_texts(file_path: str, source_name: Optional[str] = None) -> Tuple[List[Dict], str]:
  try:

    # File does not exist
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"PDF not found: {file_path}")

    with pymupdf.open(file_path) as doc:

      # The PDF has no pages
      if len(doc) == 0:
        raise ValueError("PDF has no pages")

      actual_source = source_name or os.path.basename(file_path)
      all_text = []

      for page_idx, page in enumerate(doc, start=1):
        text = page.get_text()

        # Skip empty pages
        if text.strip():
          all_text.append({
              "page": page_idx,
              "text": text,
              "source": actual_source
          })

      print(f"Loaded {len(doc)} pages from {source_name}")
      return all_text, actual_source # Moved outside the loop

  except Exception as e:
      print(f"Error loading PDF: {e}")
      raise