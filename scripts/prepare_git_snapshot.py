"""Собрать проверяемое дерево файлов для будущей публикации, не создавая Git-репозиторий."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENIED = re.compile(r"blind|holdout|secret|credential|private", re.IGNORECASE)

ROOT_FILES = {
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-hf.txt",
    "Dockerfile",
    "compose.yaml",
    ".dockerignore",
    ".gitattributes",
    "app.py",
    ".streamlit/config.toml",
    "notebooks/demo.ipynb",
}
DOCS = {
    "docs/PROJECT-BRIEF.md",
    "docs/SUBMISSION-STATUS.md",
    "docs/ARCHITECTURE.md",
    "docs/EVALUATION.md",
    "docs/RESEARCH-BASIS.md",
    "docs/DATA-PROVENANCE.md",
    "docs/REPRODUCE.md",
    "docs/runbook.md",
    "docs/CLIENT-ONBOARDING.md",
    "docs/contract/INTEGRATION-CHECKLIST.md",
    "docs/data/PRODUCT-CONTRACT.md",
    "docs/DEVELOPMENT-ROADMAP.md",
    "docs/RELEASE-CHECKLIST.md",
    "docs/contract/openapi.yaml",
    "docs/contract/recsflow-adapter-fixture.json",
}
DATA = {
    "data/README.md",
    "data/dialogues.jsonl",
    "data/product_llm_first_dev.json",
    "data/product_llm_first_dev_v2.json",
    "data/product_contract_ru_v1.json",
    "data/product_contract_ru_v1.manifest.json",
    "data/model_extraction_dev_facts.json",
    "data/model_extraction_electronics_smoke.json",
    "data/e7-006-extraction-contract-v1.json",
    "data/e7-006-nli-contract-v1.json",
    "data/e7-007-canonical-contract-probe-v1.json",
    "artifacts/eval-v2-20260914/recagent-eval-v2.0-dev.jsonl",
    "artifacts/eval-v2-20260914/recagent-eval-v2.0-dev.manifest.json",
}
REPORT = {
    "report/REPORT.md",
    "report/index.html",
    "report/selection.json",
    "report/demo.mp4",
    "report/demo.png",
    "report/video.json",
    "report/evidence/evaluation-contract.json",
    "report/evidence/evaluation-roles.json",
    "report/evidence/evaluation-load-c1.json",
    "report/evidence/evaluation-load-c2.json",
    "report/evidence/evaluation-load-c4.json",
}
TREES = {
    "src": {".py", ".typed"},
    "evals": {".py"},
    "scripts": {".py"},
    "tests": {".py", ".json", ".yaml", ".yml"},
    "configs": {".yaml", ".yml", ".json"},
}
GITIGNORE = """.venv/
.venv-lora/
__pycache__/
.pytest_cache/
.ruff_cache/
*.egg-info/
build/
report/*-local.ipynb
.env
.streamlit/secrets.toml
data/raw/
dist/
runtime/
report/*-local.json
artifacts/*
!artifacts/eval-v2-20260914/
artifacts/eval-v2-20260914/*
!artifacts/eval-v2-20260914/recagent-eval-v2.0-dev.jsonl
!artifacts/eval-v2-20260914/recagent-eval-v2.0-dev.manifest.json
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(root: Path, path: Path) -> Path:
    relative = path.relative_to(root)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Путь выходит за пределы проекта: {relative}")
    if any(
        DENIED.search(part) or part.startswith((".env", ".venv")) or part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
        for part in relative.parts
    ):
        raise ValueError(f"Закрытый или служебный путь: {relative}")
    return relative


def selected_files(root: Path) -> list[Path]:
    selected = {root / name for name in ROOT_FILES | DOCS | DATA | REPORT}
    for folder, suffixes in TREES.items():
        for path in (root / folder).rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                relative = path.relative_to(root)
                if any(DENIED.search(part) for part in relative.parts):
                    continue
                if relative.parts[0] == "tests" and path.name.startswith("test_semantic_lora"):
                    continue
                if relative.as_posix() in {"scripts/package.py", "tests/unit/test_delivery_release.py"}:
                    # Старый ZIP-упаковщик требует архивные документы, исключённые из Git-копии.
                    continue
                selected.add(path)

    selection = json.loads((root / "report/selection.json").read_text(encoding="utf-8"))
    for source in selection["sources"]:
        relative = Path(source["bundled_path"])
        if not relative.as_posix().startswith("report/evidence/"):
            raise ValueError(f"Доказательство вне report/evidence: {relative}")
        path = root / relative
        safe_relative(root, path)
        if not path.resolve().is_relative_to((root / "report/evidence").resolve()):
            raise ValueError(f"Доказательство выходит из report/evidence: {relative}")
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Не совпал SHA-256 доказательства: {relative}")
        selected.add(path)

    for path in selected:
        relative = safe_relative(root, path)
        if not path.is_file():
            raise FileNotFoundError(relative)
    return sorted(selected)


def verify_local_links(destination: Path) -> None:
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for doc in destination.rglob("*.md"):
        for match in pattern.finditer(doc.read_text(encoding="utf-8")):
            target = match.group(1).split("#", 1)[0].strip("<>")
            if not target or re.match(r"^(https?://|mailto:)", target):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.is_relative_to(destination.resolve()) or not resolved.exists():
                raise ValueError(f"Ссылка вне поставки: {doc.relative_to(destination)} -> {target}")


def build_snapshot(root: Path, destination: Path) -> dict:
    root = root.resolve()
    destination = destination.resolve()
    if not destination.is_relative_to((root / "dist").resolve()):
        raise ValueError("Выходная папка должна находиться внутри dist/")
    if destination.exists():
        raise FileExistsError(f"Выходная папка уже существует: {destination}")
    paths = selected_files(root)
    destination.mkdir(parents=True, exist_ok=False)
    for path in paths:
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (destination / ".gitignore").write_text(GITIGNORE, encoding="utf-8", newline="\n")
    # Keep the verified snapshot byte-for-byte across Git index and checkout.
    # The source workspace may contain both LF and CRLF; receipts hash actual bytes.
    (destination / ".gitattributes").write_text(
        "# Preserve checked bytes and hashes; accept CRLF without hiding other whitespace errors.\n"
        "* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n",
        encoding="utf-8", newline="\n",
    )
    verify_local_links(destination)
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    hashes = {path.relative_to(destination).as_posix(): sha256(path) for path in files}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema_version": 1,
        "purpose": "Состав проверяемой поставки и SHA-256 файлов; manifest не включает собственный hash",
        "source_head": head,
        "source_state": "working_tree",
        "files_count": len(hashes),
        "files": hashes,
    }
    (destination / "PUBLIC-MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if any(sha256(destination / name) != digest for name, digest in hashes.items()):
        raise ValueError("Проверка скопированных файлов не прошла")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_snapshot(ROOT, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "files_count": manifest["files_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
