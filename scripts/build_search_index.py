"""Build the deterministic local BM25 index for a demo catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recagent.catalog import generate_catalog
from recagent.retrieval import BM25Index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/workflow-v2/bm25.json"))
    args = parser.parse_args()
    items = generate_catalog()
    documents = {item.id: f"{item.title} {item.genre or ''} {item.tone or ''} {item.description}" for item in items}
    index = BM25Index(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"documents": index.documents, "k1": index.k1, "b": index.b}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "documents": len(documents), "config": str(args.config)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
