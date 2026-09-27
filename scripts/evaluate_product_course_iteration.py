"""Reproduce a paired DEV-only comparison of the course-topic parser fix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from evals.dataset_v2 import load_dataset

import recagent.agent as agent_module
from recagent.catalog import generate_catalog
from scripts.evaluate_baselines import run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, cases = load_dataset(args.manifest)
    if manifest.split != "dev" or manifest.blind_claim:
        raise ValueError("Only public DEV is allowed")
    catalog = generate_catalog(manifest.catalog_seed)
    baseline_ref = "0346198"
    source = subprocess.run(
        ["git", "show", f"{baseline_ref}:src/recagent/parsing.py"], check=True, capture_output=True, encoding="utf-8"
    ).stdout
    namespace = {"__name__": "recagent.parsing", "__package__": "recagent"}
    exec(compile(source, "baseline-parsing.py", "exec"), namespace)
    current_parser = agent_module.rule_parse
    records = {}
    try:
        for label, selected_parser in (("before", namespace["rule_parse"]), ("after", current_parser)):
            agent_module.rule_parse = selected_parser
            records[label] = [run_case(case.scenario, "current", catalog, mode="rules") for case in cases]
    finally:
        agent_module.rule_parse = current_parser
    families = sorted({case.scenario_family for case in cases})
    summary = {}
    for family in ["all", *families]:
        indices = [i for i, case in enumerate(cases) if family == "all" or case.scenario_family == family]
        before = [bool(records["before"][i]["final_success"]) for i in indices]
        after = [bool(records["after"][i]["final_success"]) for i in indices]
        summary[family] = {
            "n": len(indices),
            "before": sum(before),
            "after": sum(after),
            "gains": sum(not a and b for a, b in zip(before, after, strict=True)),
            "losses": sum(a and not b for a, b in zip(before, after, strict=True)),
        }
    result = {
        "scope": "public synthetic DEV; same product except parser; default legacy policy",
        "baseline_parser_ref": baseline_ref,
        "cases_sha256": manifest.cases_sha256,
        "current_source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path("src/recagent").rglob("*.py"))
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
