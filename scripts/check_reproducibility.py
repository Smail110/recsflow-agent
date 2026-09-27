"""Generate deterministic research datasets twice and compare every result hash."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "reproducibility"
SCHEMA_VERSION = 1
HASH_SEEDS = ("1", "8675309")
SOURCE_GLOBS = ("src/recagent/**/*.py", "evals/**/*.py")
SOURCE_FILES = (
    "scripts/prepare_hf_dataset.py",
    "scripts/evaluate.py",
    "scripts/check_reproducibility.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-hf.txt",
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_value(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_canonical_json(path: Path, value: object) -> str:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def source_file_hashes() -> dict[str, str]:
    paths: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    paths.update(ROOT / relative for relative in SOURCE_FILES)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing reproducibility inputs: {[str(path) for path in missing]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}


def git_state() -> dict[str, object]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    try:
        porcelain = git("status", "--porcelain", "--untracked-files=all")
        return {
            "available": True,
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(porcelain),
            "porcelain_sha256": sha256_bytes(porcelain.encode("utf-8")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": type(exc).__name__, "dirty": None}


def environment_metadata() -> dict[str, object]:
    packages: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.casefold()] = dist.version
    lock_files = {
        relative: sha256_file(ROOT / relative)
        for relative in ("pyproject.toml", "requirements.txt", "requirements-lock.txt", "requirements-hf.txt")
    }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "packages": dict(sorted(packages.items())),
        "lock_file_sha256": lock_files,
        "git": git_state(),
    }


def _distribution(rows: list[dict[str, Any]], *keys: str) -> dict[str, int]:
    counts = Counter("/".join(str(row[key]) for key in keys) for row in rows)
    return dict(sorted(counts.items()))


def load_static_evaluate_cases(*, seed_series_title: str) -> list[dict[str, object]]:
    """Materialize the static CASES literal without importing the evaluator stack."""

    source_path = ROOT / "scripts" / "evaluate.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    cases_node: ast.expr | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets):
            cases_node = node.value
            break
    if cases_node is None:
        raise ValueError("scripts.evaluate.CASES assignment was not found")

    class ReplaceSeedTitle(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id == "_SEED_SERIES":
                return ast.copy_location(ast.Constant(seed_series_title), node)
            raise ValueError(f"unsupported dynamic name in scripts.evaluate.CASES: {node.id}")

        def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.expr:
            replaced = self.generic_visit(node)
            parts: list[str] = []
            for value in replaced.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                elif (
                    isinstance(value, ast.FormattedValue)
                    and isinstance(value.value, ast.Constant)
                    and value.conversion == -1
                    and value.format_spec is None
                ):
                    parts.append(str(value.value.value))
                else:
                    raise ValueError("unsupported f-string in scripts.evaluate.CASES")
            return ast.copy_location(ast.Constant("".join(parts)), node)

    replaced = ReplaceSeedTitle().visit(cases_node)
    value = ast.literal_eval(replaced)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("scripts.evaluate.CASES must materialize to a list of objects")
    return value


def build_snapshot(output_dir: Path, *, per_family: int) -> dict[str, object]:
    if isinstance(per_family, bool) or not isinstance(per_family, int) or per_family < 1:
        raise ValueError("per_family must be a positive integer")

    from evals.adversarial_ru import CASES as ADVERSARIAL_CASES
    from evals.adversarial_ru import CORPUS_ORIGIN, CORPUS_VERSION, SPLIT, corpus_audit, corpus_sha256
    from evals.dataset_v2 import DATASET_VERSION, SEEDS, generate_public_dataset, write_dataset
    from evals.scenarios import (
        DEV_PROFILE_RANGE,
        HOLDOUT_PROFILE_RANGE,
        SPLIT_SEEDS,
        generate_scenarios,
    )

    from recagent.catalog import catalog_stats, generate_catalog
    from recagent.catalog.users import PROFILE_COUNT, generate_profiles, profile_stats
    from recagent.models import Query
    from scripts.prepare_hf_dataset import TEMPLATE_VERSION, validate_rows
    from scripts.prepare_hf_dataset import build as build_hf

    output_dir.mkdir(parents=True, exist_ok=False)
    source_hashes = source_file_hashes()
    generation_config: dict[str, object] = {
        "catalog": {"seed": 42, "size": 3000},
        "profiles": {"seed": 42, "count": PROFILE_COUNT, "catalog_seed": 42},
        "legacy_dev": {
            "split": "dev",
            "size": 200,
            "seed": SPLIT_SEEDS["dev"],
            "split_seeds": dict(SPLIT_SEEDS),
            "dev_profile_range": list(DEV_PROFILE_RANGE),
            "holdout_profile_range_recorded_only": list(HOLDOUT_PROFILE_RANGE),
        },
        "v2_public": {
            "dataset_version": DATASET_VERSION,
            "splits": ["dev", "validation"],
            "per_family": per_family,
            "seeds": SEEDS,
        },
        "hf_smoke": {"seed": 42, "examples_per_label": 40, "validation_fraction": 0.25},
        "static_evaluate_cases": {"seed": None, "source": "scripts.evaluate.CASES"},
        "adversarial_ru": {"seed": None, "origin": CORPUS_ORIGIN, "version": CORPUS_VERSION, "split": SPLIT},
        "final_holdout": {"generated": False, "read": False},
    }

    catalog = generate_catalog(seed=42, size=3000)
    catalog_values = [item.model_dump(mode="json") for item in catalog]
    catalog_file_hash = write_canonical_json(output_dir / "catalog.json", catalog_values)
    seed_series_title = next(item.title for item in catalog if item.genre == "детектив" and item.kind == "series")
    static_evaluate_cases = load_static_evaluate_cases(seed_series_title=seed_series_title)

    profiles = generate_profiles(seed=42, count=PROFILE_COUNT, catalog=catalog)
    profile_values = [profile.model_dump(mode="json") for profile in profiles]
    profiles_file_hash = write_canonical_json(output_dir / "profiles.json", profile_values)

    legacy, legacy_summary = generate_scenarios(
        "dev",
        size=200,
        seed=SPLIT_SEEDS["dev"],
        catalog=catalog,
        profiles=profiles,
    )
    legacy_values = [scenario.model_dump(mode="json") for scenario in legacy]
    legacy_file_hash = write_canonical_json(output_dir / "legacy-dev.json", legacy_values)

    v2_results: dict[str, object] = {}
    for split in ("dev", "validation"):
        cases, generation_summary = generate_public_dataset(split, per_family=per_family)
        cases_path, manifest_path = write_dataset(cases, output_dir / f"v2-{split}", split=split)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        qa_path = manifest_path.parent / manifest["qa_sample_file"]
        v2_results[split] = {
            "case_count": len(cases),
            "generation_summary": generation_summary,
            "by_family": manifest["by_family"],
            "cases_sha256": manifest["cases_sha256"],
            "cases_file_sha256": sha256_file(cases_path),
            "manifest_payload_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": sha256_file(manifest_path),
            "qa_file_sha256": sha256_file(qa_path),
            "catalog_sha256": manifest["catalog_sha256"],
            "normalized_dialogues_sha256": manifest["normalized_dialogues_sha256"],
            "surface_families_sha256": sha256_value(manifest["surface_families"]),
        }

    hf_rows = build_hf(seed=42, examples_per_label=40, validation_fraction=0.25)
    hf_validation = validate_rows(hf_rows, source="reproducibility snapshot")
    hf_payload = b"".join(canonical_bytes(row) + b"\n" for row in hf_rows)
    hf_path = output_dir / "hf-smoke.jsonl"
    hf_path.write_bytes(hf_payload)

    static_source = ROOT / "scripts" / "evaluate.py"
    static_file_hash = write_canonical_json(output_dir / "static-evaluate-cases.json", static_evaluate_cases)

    query_schema = Query.model_json_schema()
    adversarial_values = [asdict(case) for case in ADVERSARIAL_CASES]
    adversarial_materialized_hash = write_canonical_json(output_dir / "adversarial-ru.json", adversarial_values)

    datasets: dict[str, object] = {
        "catalog": {
            "count": len(catalog),
            "canonical_sha256": sha256_value(catalog_values),
            "file_sha256": catalog_file_hash,
            "stats": catalog_stats(catalog),
        },
        "profiles": {
            "count": len(profiles),
            "canonical_sha256": sha256_value(profile_values),
            "file_sha256": profiles_file_hash,
            "stats": profile_stats(profiles),
        },
        "legacy_dev": {
            "count": len(legacy),
            "canonical_sha256": sha256_value(legacy_values),
            "file_sha256": legacy_file_hash,
            "summary": legacy_summary,
        },
        "v2_public": v2_results,
        "hf_smoke": {
            "count": len(hf_rows),
            "template_version": TEMPLATE_VERSION,
            "canonical_sha256": sha256_value(hf_rows),
            "file_sha256": sha256_bytes(hf_payload),
            "validation_counts": hf_validation["counts"],
            "by_label": _distribution(hf_rows, "label"),
            "by_split": _distribution(hf_rows, "split"),
            "by_label_and_split": _distribution(hf_rows, "label", "split"),
            "template_family_sha256": sha256_value(sorted({row["template_family_id"] for row in hf_rows})),
        },
        "static_evaluate_cases": {
            "count": len(static_evaluate_cases),
            "canonical_sha256": sha256_value(static_evaluate_cases),
            "file_sha256": static_file_hash,
            "source_file_sha256": sha256_file(static_source),
        },
        "adversarial_ru": {
            "count": len(ADVERSARIAL_CASES),
            "corpus_sha256": corpus_sha256(ADVERSARIAL_CASES),
            "query_schema_sha256": sha256_value(query_schema),
            "materialized_canonical_sha256": sha256_value(adversarial_values),
            "materialized_file_sha256": adversarial_materialized_hash,
            "audit": corpus_audit(ADVERSARIAL_CASES),
        },
    }
    source_hashes_after = source_file_hashes()
    if source_hashes_after != source_hashes:
        changed = comparison_differences(source_hashes, source_hashes_after, path="source_file_hashes")
        raise RuntimeError(f"source files changed during snapshot: {changed[:10]}")
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "generation_config": generation_config,
        "source_file_hashes": source_hashes,
        "datasets": datasets,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_sha256": sha256_value(comparison),
        "comparison": comparison,
        "environment": environment_metadata(),
    }


def comparison_differences(left: object, right: object, *, path: str = "comparison") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}"
            if key not in left:
                differences.append(f"{child_path}: missing in run 1")
            elif key not in right:
                differences.append(f"{child_path}: missing in run 2")
            else:
                differences.extend(comparison_differences(left[key], right[key], path=child_path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: list length {len(left)} != {len(right)}"]
        differences: list[str] = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.extend(comparison_differences(left_item, right_item, path=f"{path}[{index}]"))
        return differences
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


def _subprocess_environment(hash_seed: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = hash_seed
    python_path = [str(ROOT / "src"), str(ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return environment


def _snapshot_command(run_dir: Path, *, per_family: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.check_reproducibility",
        "--output-dir",
        str(run_dir),
        "--per-family",
        str(per_family),
        "--_snapshot-output",
        str(run_dir / "snapshot.json"),
    ]


def _run_snapshot(run_dir: Path, *, per_family: int, hash_seed: str) -> dict[str, object]:
    command = _snapshot_command(run_dir, per_family=per_family)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_subprocess_environment(hash_seed),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=flags,
    )
    return {
        "command": command,
        "python_hash_seed": hash_seed,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "snapshot_file": str(run_dir / "snapshot.json"),
    }


def _load_comparison(path: Path) -> tuple[str, object | None]:
    value = json.loads(path.read_text(encoding="utf-8"))
    comparison = value.get("comparison")
    comparison_hash = value.get("comparison_sha256")
    if not isinstance(comparison_hash, str):
        raise ValueError(f"{path} does not contain comparison_sha256")
    if comparison is not None and sha256_value(comparison) != comparison_hash:
        raise ValueError(f"{path} comparison_sha256 does not match comparison payload")
    return comparison_hash, comparison


def run_twice(output_dir: Path, *, per_family: int, expected: Path | None, allow_dirty: bool) -> tuple[dict[str, object], int]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    checkout = git_state()
    if not checkout.get("available"):
        raise RuntimeError("Git state is unavailable; cannot certify a clean checkout")
    if checkout.get("dirty") and not allow_dirty:
        raise RuntimeError("working tree is dirty; commit or isolate the intended source state, or use --allow-dirty for diagnostics")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_receipts = [
        _run_snapshot(output_dir / f"run-{index}", per_family=per_family, hash_seed=hash_seed)
        for index, hash_seed in enumerate(HASH_SEEDS, start=1)
    ]
    subprocess_ok = all(receipt["returncode"] == 0 for receipt in run_receipts)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "compliance_passed": False,
        "clean_checkout_required": not allow_dirty,
        "source_checkout_before_generation": checkout,
        "config": {"per_family": per_family, "python_hash_seeds": list(HASH_SEEDS), "expected": str(expected) if expected else None},
        "runs": run_receipts,
    }
    if not subprocess_ok:
        report["reason"] = "one or more generation subprocesses failed"
        return report, 1

    first_hash, first_comparison = _load_comparison(output_dir / "run-1" / "snapshot.json")
    second_hash, second_comparison = _load_comparison(output_dir / "run-2" / "snapshot.json")
    if first_comparison is None or second_comparison is None:
        raise ValueError("run snapshots must contain comparison payloads")
    differences = comparison_differences(first_comparison, second_comparison)
    run_match = first_hash == second_hash and not differences
    report.update(
        {
            "comparison_sha256": first_hash,
            "comparison": first_comparison,
            "second_comparison_sha256": second_hash,
            "runs_match": run_match,
            "run_differences": differences[:100],
        }
    )

    expected_match: bool | None = None
    if expected is not None:
        expected_hash, expected_comparison = _load_comparison(expected)
        expected_differences = comparison_differences(expected_comparison, first_comparison) if expected_comparison is not None else []
        expected_match = expected_hash == first_hash and not expected_differences
        report.update(
            {
                "expected_comparison_sha256": expected_hash,
                "expected_match": expected_match,
                "expected_differences": expected_differences[:100],
            }
        )

    successful = run_match and expected_match is not False
    compliance_eligible = not allow_dirty and not checkout.get("dirty") and per_family == 25
    if successful and not compliance_eligible:
        diagnostic_reasons = []
        if allow_dirty:
            diagnostic_reasons.append("--allow-dirty was requested")
        if checkout.get("dirty"):
            diagnostic_reasons.append("source checkout was dirty")
        if per_family != 25:
            diagnostic_reasons.append(f"non-default per_family={per_family}")
        report.update(
            {
                "status": "reproducible_diagnostic",
                "reason": "; ".join(diagnostic_reasons),
            }
        )
        return report, 0
    if successful:
        report.update({"status": "passed", "compliance_passed": True})
        return report, 0
    report["reason"] = "generation runs or expected regression manifest did not match"
    return report, 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--per-family", type=int, default=25, help="v2 cases per public scenario family; default is the full pilot configuration"
    )
    parser.add_argument("--expected", type=Path, help="previous snapshot or result JSON whose comparison hash must match")
    parser.add_argument("--allow-dirty", action="store_true", help="run a non-compliant diagnostic from a dirty checkout")
    parser.add_argument("--_snapshot-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.per_family < 1:
        parser.error("--per-family must be positive")
    if args._snapshot_output is not None:
        try:
            snapshot = build_snapshot(args.output_dir, per_family=args.per_family)
            write_canonical_json(args._snapshot_output, snapshot)
        except Exception as exc:
            print(f"snapshot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        report, returncode = run_twice(
            args.output_dir,
            per_family=args.per_family,
            expected=args.expected,
            allow_dirty=args.allow_dirty,
        )
    except Exception as exc:
        print(f"reproducibility check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    result_path = args.output_dir / "result.json"
    write_canonical_json(result_path, report)
    print(json.dumps({"status": report["status"], "result": str(result_path)}, ensure_ascii=False))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
