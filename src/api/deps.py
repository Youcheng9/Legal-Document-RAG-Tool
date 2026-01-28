from functools import lru_cache

from src.rag.pipeline import RagPipeline


@lru_cache()
def get_rag_pipeline() -> RagPipeline:
    return RagPipeline()
