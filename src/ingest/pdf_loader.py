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

# all_text, source_name = load_pdf_and_texts(drive_path)
# print("\n",json.dumps(all_text, indent=2))