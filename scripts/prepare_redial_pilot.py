"""Acquire pinned ReDial and build a reproducible, unscored English prefix pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/data/redial-pilot-v1.json"
MOVIE_TOKEN = re.compile(r"@(\d+)\b")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def jsonl(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows)


def verified_archive(path: Path, source: dict) -> bytes:
    if path.stat().st_size != source["archive_bytes"]:
        raise ValueError("archive size mismatch")
    payload = path.read_bytes()
    if sha256(payload) != source["archive_sha256"]:
        raise ValueError("archive SHA-256 mismatch")
    return payload


def fetch(config: dict, cache: Path) -> dict:
    """Never replace an existing cache, and publish only complete verified bytes."""
    source = config["source"]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "redial_dataset.zip"
    reused = path.exists()
    if reused:
        verified_archive(path, source)
    else:
        chunks, size = [], 0
        with httpx.stream("GET", source["url"], timeout=60, follow_redirects=True) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > source["max_download_bytes"]:
                    raise ValueError("download exceeds configured byte limit")
                chunks.append(chunk)
        payload = b"".join(chunks)
        if len(payload) != source["archive_bytes"] or sha256(payload) != source["archive_sha256"]:
            raise ValueError("download size/SHA-256 mismatch")
        with path.open("xb") as handle:
            handle.write(payload)
    metadata = {"source": source, "transform": "none; exact upstream ZIP bytes"}
    metadata_path = cache / "download-metadata.json"
    contents = encoded(metadata)
    if metadata_path.exists():
        if metadata_path.read_bytes() != contents:
            raise ValueError("existing download metadata differs")
    else:
        with metadata_path.open("xb") as handle:
            handle.write(contents)
    return {
        "status": "PASS",
        "cache_reused": reused,
        "archive_sha256": source["archive_sha256"],
        "bytes": source["archive_bytes"],
        "url": source["url"],
        "cache": str(cache),
        "observed_at_utc": datetime.now(UTC).isoformat(),
    }


def mapping(value: object, field: str) -> dict:
    # The official file represents some empty dictionaries as [] (107/110 questionnaires).
    if value == []:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"invalid {field} mapping")
    return value


def validate_dialogue(row: dict) -> None:
    required = {
        "conversationId",
        "initiatorWorkerId",
        "respondentWorkerId",
        "messages",
        "movieMentions",
        "initiatorQuestions",
        "respondentQuestions",
    }
    if not isinstance(row, dict) or not required.issubset(row):
        raise ValueError("missing dialogue fields")
    if not isinstance(row["conversationId"], (str, int)) or not str(row["conversationId"]):
        raise ValueError("invalid conversation ID")
    workers = [row["initiatorWorkerId"], row["respondentWorkerId"]]
    if any(type(worker) is not int for worker in workers) or len(set(workers)) != 2:
        raise ValueError("invalid worker IDs")
    mentions = mapping(row["movieMentions"], "movieMentions")
    if any(
        not isinstance(key, str) or not key.isdigit() or (title is not None and not isinstance(title, str))
        for key, title in mentions.items()
    ):
        raise ValueError("invalid movie mapping")
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("invalid messages")
    ids = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or not {"messageId", "text", "senderWorkerId", "timeOffset"}.issubset(message)
            or message["senderWorkerId"] not in workers
            or not isinstance(message["text"], str)
            or type(message["messageId"]) is not int
            or not isinstance(message["timeOffset"], (int, float))
        ):
            raise ValueError("invalid message schema")
        ids.append(message["messageId"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate message IDs")
    for field in ("initiatorQuestions", "respondentQuestions"):
        for movie_id, labels in mapping(row[field], field).items():
            if movie_id not in mentions or not isinstance(labels, dict):
                raise ValueError("invalid questionnaire entity")
            if set(labels) != {"seen", "liked", "suggested"}:
                raise ValueError("invalid questionnaire fields")
            for key, allowed in {"seen": {0, 1, 2}, "liked": {0, 1, 2}, "suggested": {0, 1}}.items():
                if type(labels[key]) is not int or labels[key] not in allowed:
                    raise ValueError("invalid questionnaire value")


def load_training(path: Path, source: dict) -> list[dict]:
    verified_archive(path, source)
    if source["train_member"] != "train_data.jsonl":
        raise ValueError("only the official training member is permitted")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names.count("train_data.jsonl") != 1:
            raise ValueError("missing or duplicate training member")
        info = archive.getinfo("train_data.jsonl")
        if info.file_size > source["max_train_bytes"]:
            raise ValueError("training member exceeds configured byte limit")
        # No extractall: no test member is opened and archive paths never become output paths.
        payload = archive.read("train_data.jsonl")
    if len(payload) != source["train_bytes"] or sha256(payload) != source["train_sha256"]:
        raise ValueError("training size/SHA-256 mismatch")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if len(rows) != source["train_rows"]:
        raise ValueError("training row count mismatch")
    ids = set()
    for row in rows:
        validate_dialogue(row)
        key = str(row["conversationId"])
        if key in ids:
            raise ValueError("duplicate conversation ID")
        ids.add(key)
    return rows


def prefix(row: dict) -> list[dict]:
    """Conservative observable boundary, not a semantic claim 'first recommendation'."""
    messages = row["messages"]
    cut = next(
        (
            index
            for index, message in enumerate(messages)
            if message["senderWorkerId"] == row["respondentWorkerId"] and MOVIE_TOKEN.search(message["text"])
        ),
        None,
    )
    if cut is None:
        return []
    seeker_positions = [index for index in range(cut) if messages[index]["senderWorkerId"] == row["initiatorWorkerId"]]
    return messages[: seeker_positions[-1] + 1] if seeker_positions else []


def prepare(config: dict, rows: list[dict]) -> tuple[dict[str, bytes], dict]:
    size, seed = config["sample_size"], config["seed"]
    if type(size) is not int or not 1 <= size <= 30 or type(seed) is not int:
        raise ValueError("pilot requires an integer seed and 1..30 conversations")
    candidates = [row for row in rows if prefix(row)]
    candidates.sort(key=lambda row: (sha256(f"{seed}:{row['conversationId']}".encode()), str(row["conversationId"])))
    selected, workers = [], set()
    for row in candidates:
        pair = {row["initiatorWorkerId"], row["respondentWorkerId"]}
        if workers.isdisjoint(pair):
            selected.append(row)
            workers.update(pair)
        if len(selected) == size:
            break
    if len(selected) != size:
        raise ValueError("not enough eligible worker-disjoint dialogues")
    inputs, annotations, templates, exclusions = [], [], [], []
    for row in selected:
        conversation_id = str(row["conversationId"])
        visible = prefix(row)
        ids = sorted({match for message in visible for match in MOVIE_TOKEN.findall(message["text"])})
        mentions = mapping(row["movieMentions"], "movieMentions")
        if any(movie_id not in mentions for movie_id in ids):
            raise ValueError("prefix movie token missing from source mapping")
        inputs.append(
            {
                "id": f"redial-train-{conversation_id}",
                "source_language": "en",
                "messages": [
                    {
                        "source_message_id": message["messageId"],
                        "role": "user" if message["senderWorkerId"] == row["initiatorWorkerId"] else "assistant",
                        "text": message["text"],
                    }
                    for message in visible
                ],
                "mentioned_entities": [
                    {
                        "source_id": movie_id,
                        "title": mentions[movie_id],
                        "catalog_mapping_id": None,
                        "attributes": dict.fromkeys(("genre", "tone", "duration", "year", "quality")),
                    }
                    for movie_id in ids
                ],
            }
        )
        annotations.append(
            {
                "id": inputs[-1]["id"],
                "scope": "whole_dialogue_post_dialogue_questionnaire",
                "permitted_as_agent_input": False,
                "prefix_ground_truth": False,
                "movie_mentions_whole_dialogue": row["movieMentions"],
                "initiator_questions_raw": row["initiatorQuestions"],
                "respondent_questions_raw": row["respondentQuestions"],
                "label_meanings": {"seen_liked": "0=no, 1=yes, 2=not stated", "suggested": "0=seeker mention, 1=recommender suggestion"},
            }
        )
        templates.append(
            {
                "id": inputs[-1]["id"],
                "status": "UNANNOTATED",
                "source_language": "en",
                "input_sha256": sha256(encoded(inputs[-1])),
                "translation_ru": None,
                "translation_provenance": None,
                "independent_annotators": [
                    {
                        "annotator_id": None,
                        "explicit_constraints_with_message_spans": None,
                        "allowed_next_actions": None,
                        "unknowns": None,
                        "clarification_opportunity": None,
                    }
                    for _ in range(2)
                ],
                "adjudication": None,
                "gold_labels": None,
            }
        )
    for row in rows:
        if workers.intersection({row["initiatorWorkerId"], row["respondentWorkerId"]}):
            exclusions.append(str(row["conversationId"]))
    files = {
        "inputs.en.jsonl": jsonl(inputs),
        "observed-annotations.DO-NOT-PROMPT.jsonl": jsonl(annotations),
        "annotation-template.jsonl": jsonl(templates),
    }
    summary = {
        "selected_conversation_ids": [str(row["conversationId"]) for row in selected],
        "selected_worker_ids": sorted(workers),
        "eligible_conversations": len(candidates),
        "selected_conversations": len(selected),
        "selected_workers": len(workers),
        "future_eval_excluded_training_conversation_ids": sorted(exclusions),
        "grouping": "Pairwise worker-disjoint selected conversations; later split must reserve selected workers",
        "split_claim": "No train/eval split created; shared films/indirect network dependence remain",
        "status": "PREPARED_NOT_SCORED",
        "human_annotations": 0,
        "russian_gold_labels": 0,
        "catalog_alignment": "NOT_AVAILABLE",
        "agent_calls": 0,
    }
    return files, summary


def build(config_path: Path, cache: Path, output: Path) -> dict:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if output.exists():
        raise FileExistsError("choose a fresh output directory; existing pilot is never overwritten")
    rows = load_training(cache / "redial_dataset.zip", config["source"])
    files, summary = prepare(config, rows)
    manifest = {
        "schema_version": config["schema_version"],
        "config_sha256": sha256(config_bytes),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "source": config["source"],
        "seed": config["seed"],
        "summary": summary,
        "transformations": [config[key] for key in ("selection", "prefix", "labels", "split")],
        "files": {name: {"sha256": sha256(payload), "bytes": len(payload)} for name, payload in files.items()},
    }
    output.mkdir(parents=True)
    for name, payload in {**files, "manifest.json": encoded(manifest)}.items():
        with (output / name).open("xb") as handle:
            handle.write(payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "build"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=Path("data/raw/redial"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        parser.error("receipt already exists; use a fresh path")
    config = json.loads(args.config.read_bytes())
    if args.action == "fetch":
        result = fetch(config, args.cache)
    else:
        if args.output is None:
            parser.error("build requires --output")
        result = build(args.config, args.cache, args.output)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("xb") as handle:
            handle.write(encoded(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
