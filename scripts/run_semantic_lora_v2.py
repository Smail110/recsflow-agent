"""Run one explicitly selected, config-driven semantic-LoRA v2 pipeline stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.generate_semantic_lora_dataset_v2 import audit_dataset
from scripts.semantic_lora_common import (
    canonical,
    file_sha256,
    load_config,
    read_jsonl,
    resolve,
    sha256_bytes,
)


def _run(module: str, *arguments: str) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], check=True)


def _dataset_audit(config: dict[str, Any]) -> None:
    splits = {split: read_jsonl(resolve(config["dataset"][split]["path"])) for split in ("train", "dev", "blind")}
    schemas = json.loads(resolve(config["dataset"]["schemas"]).read_text(encoding="utf-8"))
    recorded = json.loads(resolve(config["dataset"]["leakage_report"]).read_text(encoding="utf-8"))
    recomputed = audit_dataset(splits, schemas, recorded["registry"])
    if not recomputed["passed"] or canonical(recomputed) != canonical(recorded):
        raise ValueError("recomputed dataset audit does not match the frozen leakage report")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "leakage_report_sha256": file_sha256(resolve(config["dataset"]["leakage_report"])),
        "train_sha256": config["dataset"]["train"]["sha256"],
        "dev_sha256": config["dataset"]["dev"]["sha256"],
        "blind_sha256": config["dataset"]["blind"]["sha256"],
        "checks": {
            key: recomputed[key]
            for key in (
                "exact_text_overlap",
                "normalized_text_overlap",
                "token_jaccard_near_duplicate_pairs",
                "template_family_id_overlap",
                "schema_family_id_overlap",
                "domain_family_id_overlap",
                "schema_id_overlap",
                "field_name_set_overlap",
                "schema_fingerprint_overlap",
                "enum_set_overlap",
                "maximum_cross_split_token_jaccard",
            )
        },
        "final_holdout_used": False,
    }
    result["report_sha256"] = sha256_bytes(canonical(result).encode())
    output = resolve("artifacts/lora/semantic_extraction_v2/dataset-audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"status": "PASS", "output": str(output), **result["checks"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "dataset-build",
            "dataset-audit",
            "preflight",
            "profile",
            "tiny-overfit",
            "protocol-freeze",
            "baseline",
            "train",
            "lora-eval",
        ),
    )
    args = parser.parse_args()
    if args.stage == "dataset-build":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("LoRA config must be a mapping")
    else:
        config = load_config(args.config)
    config_arg = str(args.config)
    root = "artifacts/lora/semantic_extraction_v2"
    if args.stage == "dataset-build":
        _run(
            "scripts.generate_semantic_lora_dataset_v2",
            "--output-dir",
            str(resolve(config["dataset"]["root"])),
            "--seed",
            str(config["seeds"]["generation"]),
        )
    elif args.stage == "dataset-audit":
        _dataset_audit(config)
    elif args.stage == "preflight":
        _run(
            "scripts.preflight_semantic_lora",
            "--config",
            config_arg,
            "--output",
            f"{root}/preflight.json",
        )
    elif args.stage == "profile":
        for name, checkpointing, deterministic in (
            ("checkpoint-on-nondeterministic", "true", "false"),
            ("checkpoint-off-nondeterministic", "false", "false"),
            ("checkpoint-on-deterministic", "true", "true"),
        ):
            _run(
                "scripts.profile_semantic_lora_stack",
                "--config",
                config_arg,
                "--output",
                f"{root}/profiles/{name}.json",
                "--gradient-checkpointing",
                checkpointing,
                "--deterministic-kernels",
                deterministic,
                "--timeout-seconds",
                "600",
            )
    elif args.stage == "tiny-overfit":
        _run(
            "scripts.tiny_overfit_semantic_lora",
            "--config",
            config_arg,
            "--output",
            f"{root}/tiny-overfit/report.json",
        )
    elif args.stage == "protocol-freeze":
        _run("scripts.freeze_semantic_lora_protocol", "--config", config_arg)
    elif args.stage == "baseline":
        _run(
            "scripts.evaluate_semantic_lora",
            "--config",
            config_arg,
            "--arm",
            "base",
            "--output",
            f"{root}/evaluation/base.json",
        )
    elif args.stage == "train":
        _run("scripts.train_semantic_lora", "--config", config_arg)
    else:
        _run(
            "scripts.evaluate_semantic_lora",
            "--config",
            config_arg,
            "--arm",
            "lora",
            "--output",
            f"{root}/evaluation/lora.json",
            "--baseline-report",
            f"{root}/evaluation/base.json",
            "--conversion-gate-report",
            f"{root}/conversion/gate.json",
        )


if __name__ == "__main__":
    main()
