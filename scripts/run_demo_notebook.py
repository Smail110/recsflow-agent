"""Execute the demo's plain Python notebook cells without optional Jupyter packages.

No magics, shell commands or widgets are supported. This is a sequential Python
runner, not a Jupyter kernel; execution errors are saved and fail the command.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import platform
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def execute_notebook(path: Path, output: Path, *, mode: str = "ollama") -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    namespace = {"__name__": "__main__"}
    old_cwd, old_mode = Path.cwd(), os.environ.get("DEMO_NOTEBOOK_MODE")
    receipt = {
        "runner": "sequential-python-ast (not a Jupyter kernel)",
        "mode_requested": mode,
        "python": platform.python_version(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "code_cells": 0,
        "status": "RUNNING",
    }
    failure = None
    try:
        os.chdir(ROOT)
        os.environ["DEMO_NOTEBOOK_MODE"] = mode
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            receipt["code_cells"] += 1
            cell["execution_count"] = receipt["code_cells"]
            cell["outputs"] = []
            stdout = io.StringIO()
            try:
                tree = ast.parse("".join(cell["source"]), filename=f"{path.name}:cell-{index}")
                # Notebook displays the last expression; preceding statements run normally.
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                with contextlib.redirect_stdout(stdout):
                    exec(compile(tree, f"{path.name}:cell-{index}", "exec"), namespace)
                    value = eval(compile(ast.Expression(last.value), f"{path.name}:cell-{index}", "eval"), namespace) if last else None
                if stdout.getvalue():
                    cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue().splitlines(True)})
                if value is not None:
                    cell["outputs"].append(
                        {
                            "output_type": "execute_result",
                            "execution_count": receipt["code_cells"],
                            "metadata": {},
                            "data": {"text/plain": [repr(value)]},
                        }
                    )
            except Exception as exc:
                failure = exc
                if stdout.getvalue():
                    cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue().splitlines(True)})
                cell["outputs"].append(
                    {"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc), "traceback": traceback.format_exception(exc)}
                )
                receipt["failed_cell"] = index
                break
        receipt["status"] = "FAIL" if failure else "PASS"
        notebook["metadata"]["language_info"]["version"] = platform.python_version()
        notebook["metadata"]["execution_receipt"] = receipt
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    finally:
        os.chdir(old_cwd)
        if old_mode is None:
            os.environ.pop("DEMO_NOTEBOOK_MODE", None)
        else:
            os.environ["DEMO_NOTEBOOK_MODE"] = old_mode
    if failure:
        raise RuntimeError(f"Notebook failed at cell {receipt['failed_cell']}; outputs saved to {output}") from failure
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks" / "demo.ipynb")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["ollama", "rules"], default="ollama")
    args = parser.parse_args()
    print(json.dumps(execute_notebook(args.notebook, args.output or args.notebook, mode=args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
