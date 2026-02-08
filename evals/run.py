import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from src.rag.pipeline import RagPipeline

rag = RagPipeline()

with open("evals/queries.jsonl") as f:
    for line in f:
        q = json.loads(line)
        print("=" * 40)
        print("Q:", q["question"])
        res = rag.answer(q["question"], q.get("file_id"))
        print("A:", res["answer"])
        print("Sources:", len(res["sources"]))
