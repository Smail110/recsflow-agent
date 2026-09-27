"""Convert the trained PEFT adapter to GGUF and create the pinned Ollama tag."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from scripts.evaluate_semantic_lora import model_identity
from scripts.semantic_lora_common import canonical, file_sha256, load_config, resolve, sha256_bytes


def _verified_training_report(path: Path, config_hash: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    integrity = report.pop("report_sha256", None)
    if integrity != sha256_bytes(canonical(report).encode()):
        raise ValueError("training report integrity mismatch")
    if report["status"] != "COMPLETE" or report["config_sha256"] != config_hash:
        raise ValueError("export requires a complete training report for this config")
    report["report_sha256"] = integrity
    for item in report["artifact"]["files"]:
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"adapter artifact drift: {item['path']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--training-report",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v1/training/training-report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v1/export/export-report.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config_hash = file_sha256(args.config)
    training_report = _verified_training_report(args.training_report, config_hash)
    export = config["export"]
    llama_dir = resolve(export["llama_cpp_dir"])
    llama_commit = subprocess.check_output(["git", "-C", str(llama_dir), "rev-parse", "HEAD"], text=True).strip()
    if llama_commit != export["llama_cpp_commit"]:
        raise ValueError("llama.cpp commit drift")
    converter = llama_dir / "convert_lora_to_gguf.py"
    base = config["base_model"]
    snapshot = Path(snapshot_download(base["id"], revision=base["revision"], local_files_only=base["local_files_only"]))
    adapter_dir = resolve(export["adapter_dir"])
    gguf = resolve(export["gguf_adapter"])
    modelfile = resolve(export["modelfile"])
    gguf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(adapter_dir),
            "--base",
            str(snapshot),
            "--outfile",
            str(gguf),
            "--outtype",
            "f16",
        ],
        cwd=llama_dir,
        check=True,
    )
    modelfile.write_text(
        f"FROM {base['ollama_tag']}\nADAPTER ./{gguf.name}\n",
        encoding="utf-8",
    )
    ollama = shutil.which("ollama")
    if not ollama:
        raise FileNotFoundError("ollama executable not found")
    tag = config["evaluation"]["lora_ollama_tag"]
    subprocess.run([ollama, "create", tag, "-f", str(modelfile)], check=True)
    identity = model_identity(config["evaluation"]["base_url"], tag, config["evaluation"]["timeout_seconds"])
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "config_sha256": config_hash,
        "training_report_sha256": training_report["report_sha256"],
        "training_artifact_sha256": training_report["artifact"]["aggregate_sha256"],
        "base_model": {
            "id": base["id"],
            "revision": base["revision"],
            "ollama_tag": base["ollama_tag"],
            "ollama_digest": base["ollama_digest"],
        },
        "llama_cpp_commit": llama_commit,
        "converter_sha256": file_sha256(converter),
        "gguf": {"path": str(gguf), "bytes": gguf.stat().st_size, "sha256": file_sha256(gguf)},
        "modelfile": {
            "path": str(modelfile),
            "sha256": file_sha256(modelfile),
            "content": modelfile.read_text(encoding="utf-8"),
        },
        "ollama_model": identity,
        "compatibility_limitation": (
            "Exact HF BF16 revision to Ollama Q4_K_M blob equivalence is UNKNOWN; "
            "the pinned Ollama base digest and conversion gate bound the evaluated artifact."
        ),
    }
    report["report_sha256"] = sha256_bytes(canonical(report).encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        canonical(
            {
                "status": report["status"],
                "output": str(args.output),
                "report_sha256": report["report_sha256"],
                "gguf": report["gguf"],
                "ollama_model": identity,
            }
        )
    )


if __name__ == "__main__":
    main()
