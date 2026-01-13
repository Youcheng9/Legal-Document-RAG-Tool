from typing import List, Dict
import spacy

def extract_entities(all_text: List[Dict]) -> Dict:
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

# target_entities = extract_entities(all_text)


# for entity_type, entities in target_entities.items():
#     print(f"{entity_type}: {entities}")