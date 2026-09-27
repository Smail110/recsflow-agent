"""The lexical audit must detect copied messages without treating distant text as a copy."""

import json

from scripts.audit_slot_evidence_near_overlap import audit


def test_exact_and_distant_messages(tmp_path):
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text(json.dumps({"premise": "Нужен смешной фильм"}, ensure_ascii=False) + "\n", encoding="utf-8")
    dev.write_text(
        "\n".join(
            json.dumps({"premise": message}, ensure_ascii=False)
            for message in ("НУЖЕН смешной фильм!", "Курс по математике")
        ) + "\n",
        encoding="utf-8",
    )
    result = audit(train, dev)
    assert result["exact_matches"] == 1
    assert result["near_matches_at_threshold"] == 1
    assert result["dev_unique_messages"] == 2
