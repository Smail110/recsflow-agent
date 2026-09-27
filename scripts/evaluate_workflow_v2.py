"""Run the frozen DEV dialogue cohort through the integrated workflow-v2 route.

This is deliberately a small E7 entrypoint, not a second evaluator: all
per-dialogue scoring, model identity capture and final-holdout guards live in
``evaluate_llm_first_product.py``. Keeping one implementation prevents an E7
report from silently becoming a one-turn rules smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "product_llm_first_dev.json"


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("workflow config must be a YAML mapping")
    if config.get("implementation") != "workflow-v2":
        raise ValueError("E7 requires implementation: workflow-v2")
    return config


def _assert_dev_only(path: Path) -> None:
    if "holdout" in str(path).casefold() or "final" in str(path).casefold():
        raise ValueError("workflow-v2 evaluator разрешает только DEV dataset")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split") != "dev" or data.get("status") != "frozen_dev_only_before_inference":
        raise ValueError("E7 accepts only the frozen DEV-only cohort")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", choices=("E7",), default="E7")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "report" / "workflow-v2-evaluation.json")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--dialogue-id", action="append", dest="dialogue_ids", help="whole frozen DEV dialogue ID; repeat for a focused subset"
    )
    args = parser.parse_args()

    config_path, data_path = args.config.resolve(), args.data.resolve()
    config = _load_config(config_path)
    _assert_dev_only(data_path)
    metadata = {
        "experiment": args.experiment,
        "workflow_config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "content": config,
        },
        "dataset": {"path": str(data_path), "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()},
    }
    if args.validate_only:
        print(json.dumps({"valid": True, **metadata}, ensure_ascii=False))
        return

    # The delegated evaluator invokes Ollama, pins the digest before and after
    # inference, records every DEV dialogue and refuses non-Ollama modes.
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_llm_first_product",
        "--cohort",
        str(data_path),
        "--output",
        str(args.output),
        "--label",
        "after",
        "--mode",
        "ollama",
        "--implementation",
        "workflow-v2",
        "--workflow-config",
        str(config_path),
        "--timeout",
        str(args.timeout),
    ]
    for dialogue_id in args.dialogue_ids or []:
        command.extend(["--dialogue-id", dialogue_id])
    subprocess.run(command, cwd=ROOT, check=True)
    result = json.loads(args.output.read_text(encoding="utf-8"))
    result.update(metadata)
    result["command"] = command
    # The original report hash covers the product evaluator's output. Extend it
    # after adding the E7 config identity, so consumers can verify both.
    report_without_hash = {key: value for key, value in result.items() if key != "report_sha256"}
    result["report_sha256"] = hashlib.sha256(
        json.dumps(report_without_hash, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"complete": result["complete"], **metadata, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
