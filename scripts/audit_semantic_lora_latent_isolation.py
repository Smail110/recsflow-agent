"""Run the retrospective split-independent isolation audit for frozen LoRA v1."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.semantic_lora_common import (
    canonical,
    file_sha256,
    latent_isolation_summary,
    load_config,
    read_jsonl,
    resolve,
    sha256_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    splits = {split: read_jsonl(resolve(config["dataset"][split]["path"])) for split in ("train", "dev", "blind")}
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if latent_isolation_summary(splits)["passed"] else "FAIL",
        "config_sha256": file_sha256(args.config),
        "audit": latent_isolation_summary(splits),
        "case_level_blind_content_reported": False,
        "final_holdout_used": False,
    }
    result["report_sha256"] = sha256_bytes(canonical(result).encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical(result))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
