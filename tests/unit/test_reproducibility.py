from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.check_reproducibility import ROOT, comparison_differences, load_static_evaluate_cases


def test_comparison_detects_changed_hash_and_missing_dataset() -> None:
    first = {
        "datasets": {
            "catalog": {"canonical_sha256": "a", "count": 3},
            "profiles": {"canonical_sha256": "b", "count": 2},
        }
    }
    second = copy.deepcopy(first)
    second["datasets"]["catalog"]["canonical_sha256"] = "changed"
    del second["datasets"]["profiles"]

    differences = comparison_differences(first, second)

    assert any("catalog.canonical_sha256" in difference for difference in differences)
    assert any("datasets.profiles: missing in run 2" in difference for difference in differences)


def test_static_case_materialization_matches_evaluator() -> None:
    from scripts.evaluate import CASES

    from recagent.catalog import generate_catalog

    catalog = generate_catalog(42, 3000)
    seed_series_title = next(item.title for item in catalog if item.genre == "детектив" and item.kind == "series")
    assert load_static_evaluate_cases(seed_series_title=seed_series_title) == CASES


def test_two_subprocess_smoke_uses_different_hash_seeds(tmp_path: Path) -> None:
    output_dir = tmp_path / "reproducibility"
    environment = os.environ.copy()
    python_path = [str(ROOT / "src"), str(ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_reproducibility",
            "--output-dir",
            str(output_dir),
            "--per-family",
            "1",
            "--allow-dirty",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["runs_match"] is True
    assert result["compliance_passed"] is False
    assert result["status"] == "reproducible_diagnostic"
    assert [run["python_hash_seed"] for run in result["runs"]] == ["1", "8675309"]
