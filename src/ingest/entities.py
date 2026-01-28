from typing import List, Dict
import spacy


nlp = spacy.load("en_core_web_sm")


def extract_entities(all_text: List[Dict]) -> Dict:
  
  target_entities = {
      "PERSON": set(),
      "ORG": set(),
      "DATE": set(),
      "MONEY": set(),
      "GPE": set(), # Location
      "LAW": set(), # Legal reference
  }

  texts = [p.get("text", "") for p in all_text if p.get("text")]

  if not texts:
    return {k: [] for k in target_entities.keys()}
  
  for doc in nlp.pipe(texts, batch_size=50):

    for ent in doc.ents:
      label = ent.label_

      if label in target_entities:
        cleaned = ent.text.strip()
        
        if cleaned and len(cleaned) > 1:
          target_entities[ent.label_].add(cleaned) 


  return {label: list(entities) for label, entities in target_entities.items()}

# target_entities = extract_entities(all_text)


# for entity_type, entities in target_entities.items():
#     print(f"{entity_type}: {entities}")