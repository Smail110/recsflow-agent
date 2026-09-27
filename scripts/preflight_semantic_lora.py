"""Validate the frozen semantic-LoRA protocol before baseline or training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from scripts.semantic_lora_common import (
    canonical,
    file_sha256,
    load_config,
    read_jsonl,
    resolve,
    seen_schema_holdout,
    sha256_bytes,
    tokenize_training_row,
)

PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "safetensors",
    "huggingface-hub",
    "PyYAML",
    "pydantic",
)


def _command(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v1/preflight.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = resolve(config["dataset"]["manifest"])
    leakage_path = resolve(config["dataset"]["leakage_report"])
    seal_path = resolve(config["dataset"]["blind"]["seal"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    blind_path = resolve(config["dataset"]["blind"]["path"])
    blind_hash = file_sha256(blind_path)
    if seal["sha256"] != blind_hash:
        raise ValueError("sealed blind hash mismatch")
    if not manifest["leakage_passed"] or not leakage["passed"]:
        raise ValueError("dataset leakage gate failed")
    if set(manifest["TRAIN_SCHEMA_IDS"]) & set(manifest["BLIND_SCHEMA_IDS"]):
        raise ValueError("blind schema appears in TRAIN")

    model = config["base_model"]
    snapshot = Path(
        snapshot_download(
            model["id"],
            revision=model["revision"],
            local_files_only=model["local_files_only"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=model["trust_remote_code"],
    )
    train_rows = read_jsonl(resolve(config["dataset"]["train"]["path"]))
    dev_rows = read_jsonl(resolve(config["dataset"]["dev"]["path"]))
    training_rows, seen_rows = seen_schema_holdout(
        train_rows,
        config["dataset"]["seen_schema_eval_fraction"],
        config["seeds"]["split"],
    )
    max_length = config["training"]["max_sequence_length"]
    prompt_mode = config["training"].get("prompt_mode", "production")
    encoded = [tokenize_training_row(row, tokenizer, max_length, prompt_mode) for row in training_rows + dev_rows]
    lengths = [item["length"] for item in encoded]
    target_lengths = [item["target_length"] for item in encoded]
    overlength = sum(length > max_length for length in lengths)
    if overlength:
        raise ValueError(f"{overlength} examples exceed max_sequence_length")

    llama_dir = resolve(config["export"]["llama_cpp_dir"])
    llama_commit = _command(["git", "-C", str(llama_dir), "rev-parse", "HEAD"])
    if llama_commit != config["export"]["llama_cpp_commit"]:
        raise ValueError(f"llama.cpp commit mismatch: {llama_commit}")
    environment = {
        "python": platform.python_version(),
        "executable": os.path.realpath(os.sys.executable),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "ollama_version": _command(["ollama", "--version"]),
    }
    expected = config["environment"]
    if environment["python"] != expected["python"]:
        raise ValueError("Python version drift")
    if environment["gpu"] != expected["expected_gpu"]:
        raise ValueError("GPU model drift")
    if ".".join(map(str, environment["compute_capability"])) != expected["expected_compute_capability"]:
        raise ValueError("GPU compute capability drift")
    if environment["torch_cuda_runtime"] != expected["cuda_runtime"]:
        raise ValueError("CUDA runtime drift")

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "config_sha256": file_sha256(args.config),
        "manifest_sha256": file_sha256(manifest_path),
        "leakage_report_sha256": file_sha256(leakage_path),
        "blind_sha256": blind_hash,
        "blind_examples_opened_for_preflight": False,
        "dataset": {
            "source_train_rows": len(train_rows),
            "actual_training_rows": len(training_rows),
            "seen_schema_holdout_rows": len(seen_rows),
            "dev_rows": len(dev_rows),
            "training_schema_counts": dict(sorted(Counter(row["schema_id"] for row in training_rows).items())),
        },
        "tokens": {
            "examples_checked": len(encoded),
            "minimum": min(lengths),
            "mean": mean(lengths),
            "maximum": max(lengths),
            "target_minimum": min(target_lengths),
            "target_mean": mean(target_lengths),
            "target_maximum": max(target_lengths),
            "overlength": overlength,
            "max_sequence_length": max_length,
        },
        "model_snapshot": str(snapshot),
        "llama_cpp_commit": llama_commit,
        "llama_converter_sha256": file_sha256(llama_dir / "convert_lora_to_gguf.py"),
        "environment": environment,
        "limitations": [
            "CUDA/bitsandbytes is procedure-reproducible, not promised bit-identical.",
            "The sealed blind file was hash-checked as bytes and not parsed by this preflight.",
        ],
    }
    result["report_sha256"] = sha256_bytes(canonical(result).encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        canonical(
            {
                "status": result["status"],
                "output": str(args.output),
                "report_sha256": result["report_sha256"],
                "tokens": result["tokens"],
                "environment": environment,
            }
        )
    )


if __name__ == "__main__":
    main()
