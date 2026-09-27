"""Summarize local catalog-gap events; counts are turns, not unique people."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    counts: Counter[str] = Counter()
    turns = invalid_lines = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as events:
            for line in events:
                try:
                    event = json.loads(line)
                    wishes = event["wishes"]
                    if not isinstance(wishes, list):
                        raise ValueError("wishes must be a list")
                    cited = {wish["wish"].casefold().strip() for wish in wishes if isinstance(wish.get("wish"), str)}
                except (KeyError, TypeError, ValueError, json.JSONDecodeError, AttributeError):
                    invalid_lines += 1
                    continue
                turns += 1
                counts.update(wish for wish in cited if wish)
    return {
        "turns_with_catalog_gaps": turns,
        "invalid_lines": invalid_lines,
        "wish_turn_counts": dict(counts.most_common()),
        "unit": "dialogue_turns; not unique users or verified catalog categories",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="JSONL files; default: runtime/catalog-gaps/api.jsonl and demo.jsonl",
    )
    args = parser.parse_args()
    paths = args.paths or [Path("runtime/catalog-gaps/api.jsonl"), Path("runtime/catalog-gaps/demo.jsonl")]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summarize(paths), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
