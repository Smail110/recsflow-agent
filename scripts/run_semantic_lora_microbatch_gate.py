"""Run the execution-path microbatch benchmark with a hard reproducible timeout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deterministic-kernels", choices=("true", "false"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=360)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_semantic_lora_step",
        "--config",
        str(args.config),
        "--deterministic-kernels",
        args.deterministic_kernels,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        print(
            json.dumps(
                {
                    "status": "TIMEOUT",
                    "timeout_seconds": args.timeout_seconds,
                    "deterministic_kernels": args.deterministic_kernels == "true",
                    "stdout": exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout,
                    "stderr": exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        raise SystemExit(2) from None
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
