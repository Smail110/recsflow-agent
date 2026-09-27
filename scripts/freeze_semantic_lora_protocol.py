"""Create or verify the immutable semantic-LoRA v2 protocol seal."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.semantic_lora_common import (
    canonical,
    file_sha256,
    load_config,
    protocol_hashes,
    resolve,
    sha256_bytes,
    verify_frozen_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=Path("scripts/evaluate_semantic_lora.py"))
    parser.add_argument("--print-expected", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluator = resolve(args.evaluator)
    hashes = protocol_hashes(config, evaluator)
    if args.print_expected:
        print(json.dumps(hashes, indent=2, sort_keys=True))
        return
    declared = config["protocol_integrity"]
    if declared.get("status") != "FROZEN_BEFORE_BASELINE":
        raise ValueError("set status and printed hashes in config before creating the seal")
    mismatches = {key: {"declared": declared.get(key), "actual": value} for key, value in hashes.items() if declared.get(key) != value}
    if mismatches:
        raise ValueError(f"cannot freeze mismatched protocol: {canonical(mismatches)}")
    blind_seal = json.loads(resolve(config["dataset"]["blind"]["seal"]).read_text(encoding="utf-8"))
    if not blind_seal.get("sealed_before_baseline"):
        raise ValueError("blind dataset was not sealed before baseline")
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "FROZEN_BEFORE_BASELINE",
        "config_sha256": file_sha256(args.config),
        "hashes": hashes,
        "blind_sha256": config["dataset"]["blind"]["sha256"],
        "blind_seal_sha256": file_sha256(resolve(config["dataset"]["blind"]["seal"])),
        "baseline_started": False,
        "final_holdout_used": False,
    }
    result["report_sha256"] = sha256_bytes(canonical(result).encode())
    seal_path = resolve(declared["seal"])
    if seal_path.exists():
        verify_frozen_protocol(args.config, config, evaluator)
        print(canonical({"status": "ALREADY_FROZEN", "seal": str(seal_path)}))
        return
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verify_frozen_protocol(args.config, config, evaluator)
    print(
        canonical(
            {
                "status": result["status"],
                "seal": str(seal_path),
                "report_sha256": result["report_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
