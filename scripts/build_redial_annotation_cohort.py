"""Build a pinned, open ReDial annotation cohort without future questionnaire labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_redial_pilot import MOVIE_TOKEN, encoded, jsonl, load_training, mapping, prefix, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/redial-annotation-source-v1.json"


def prepare(config, rows):
    splits = config["splits"]
    if list(splits) != ["calibration", "dev", "validation"] or any(type(v) is not int or v <= 0 for v in splits.values()):
        raise ValueError("Expected positive calibration/dev/open-validation sizes")
    if type(config["seed"]) is not int:
        raise ValueError("Integer seed required")
    requested = sum(splits.values())
    eligible = [r for r in rows if prefix(r)]
    ordered = sorted(eligible, key=lambda r: (sha256(f"{config['seed']}:{r['conversationId']}".encode()), str(r["conversationId"])))
    selected, workers = [], set()
    for row in ordered:
        pair = {row["initiatorWorkerId"], row["respondentWorkerId"]}
        if pair.isdisjoint(workers):
            selected.append(row)
            workers.update(pair)
        if len(selected) == requested:
            break
    if len(selected) != requested:
        raise ValueError(f"Only {len(selected)} worker-disjoint prefixes available; requested {requested}")

    split_names = [name for name, size in splits.items() for _ in range(size)]
    inputs, provenance = [], []
    for row, split in zip(selected, split_names, strict=True):
        visible = prefix(row)
        mentions = mapping(row["movieMentions"], "movieMentions")
        ids = sorted({token for message in visible for token in MOVIE_TOKEN.findall(message["text"])})
        if any(token not in mentions for token in ids):
            raise ValueError("Visible entity absent from source mapping")
        identity = f"redial-train-{row['conversationId']}"
        inputs.append(
            {
                "id": identity,
                "split": split,
                "source_language": "en",
                "messages": [
                    {
                        "source_message_id": m["messageId"],
                        "role": "user" if m["senderWorkerId"] == row["initiatorWorkerId"] else "assistant",
                        "text": m["text"],
                    }
                    for m in visible
                ],
                "mentioned_entities": [
                    {
                        "source_id": token,
                        "title": mentions[token],
                        "catalog_mapping_id": None,
                        "attributes": dict.fromkeys(("genre", "tone", "duration", "year", "quality")),
                    }
                    for token in ids
                ],
            }
        )
        provenance.append(
            {
                "id": identity,
                "split": split,
                "source_conversation_id": str(row["conversationId"]),
                "worker_ids": sorted((row["initiatorWorkerId"], row["respondentWorkerId"])),
            }
        )
    excluded = sorted(str(r["conversationId"]) for r in rows if workers.intersection({r["initiatorWorkerId"], r["respondentWorkerId"]}))
    # Questionnaires and post-prefix utterances are never exported by this builder.
    files = {"inputs.en.jsonl": jsonl(inputs), "source-provenance.jsonl": jsonl(provenance)}
    summary = {
        "conversations": len(inputs),
        "workers": len(workers),
        "eligible_training_conversations": len(eligible),
        "splits": splits,
        "future_worker_overlap_exclusions": excluded,
        "source_text_origin": "human English research conversations",
        "annotation_origin": "not annotated by this builder",
        "human_gold_available": False,
        "final_holdout": False,
        "dependence": "Direct worker overlap removed; common movies and indirect worker graph links remain",
    }
    return files, summary


def build(config_path, cache, output):
    if output.exists():
        raise FileExistsError("Choose a new output directory; prior cohorts are immutable")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    source_path = ROOT / config["source_config"]
    source_bytes = source_path.read_bytes()
    if sha256(source_bytes) != config["source_config_sha256"]:
        raise ValueError("Pinned source config hash mismatch")
    source = json.loads(source_bytes)["source"]
    rows = load_training(cache / "redial_dataset.zip", source)
    files, summary = prepare(config, rows)
    if sha256(files["inputs.en.jsonl"]) != config.get("expected_inputs_sha256", sha256(files["inputs.en.jsonl"])):
        raise ValueError("Expected frozen input hash mismatch")
    dependencies = [Path(__file__), ROOT / "scripts/prepare_redial_pilot.py", source_path]
    manifest = {
        "schema_version": config["schema_version"],
        "config_sha256": sha256(config_bytes),
        "source": source,
        "seed": config["seed"],
        "summary": summary,
        "source_hashes": {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()) for p in dependencies},
        "selection": config["selection"],
        "prefix": config["prefix"],
        "limitations": config["limitations"],
        "files": {name: {"sha256": sha256(value), "bytes": len(value)} for name, value in files.items()},
    }
    output.mkdir(parents=True)
    for name, content in {**files, "manifest.json": encoded(manifest)}.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=Path("data/raw/redial-cli-check"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.config, args.cache, args.output)
    print(
        json.dumps(
            {
                "complete": True,
                "conversations": result["summary"]["conversations"],
                "splits": result["summary"]["splits"],
                "inputs": result["files"]["inputs.en.jsonl"],
            }
        )
    )


if __name__ == "__main__":
    main()
