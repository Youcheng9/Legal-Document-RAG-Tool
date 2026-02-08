from typing import List, Dict, Optional
import spacy
import logging

logger = logging.getLogger(__name__)
_NLP: Optional["spacy.language.Language"] = None

# -- Used for loading spacy only when extract_entities() is called
# -- Loads only once and prevents slow app startup and FastAPI crash
def _get_nlp():
  global _NLP

  if _NLP is not None:
    return _NLP
  
  try:
    _NLP = spacy.load("en_core_web_sm")
    logger.info("spaCy model en_core_web_sm loaded")
    return _NLP
  
  except Exception as e:
    logger.warning(
            "spaCy model 'en_core_web_sm' not available. "
            "Entity extraction disabled. Error: %s", e
    )
    _NLP = None
    return None


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
  
  nlp = _get_nlp()
  if nlp is None:
    # spaCy not available, return empty entities safely
    return {k: [] for k in target_entities}
  
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