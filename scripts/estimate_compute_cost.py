"""Рассчитать условную стоимость одной выделенной машины по окнам нагрузки."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def estimate(paths: list[Path], rates: list[float]) -> dict:
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Укажите положительные конечные ставки в рублях за час")
    sources, scenarios = [], []
    for path in paths:
        payload = path.read_bytes()
        data = json.loads(payload)
        seconds, requests = data["elapsed_seconds"], data["requests"]
        if not math.isfinite(seconds) or seconds <= 0 or type(requests) is not int or requests <= 0:
            raise ValueError(f"Некорректная длительность или число запросов: {path}")
        sources.append({"path": path.as_posix(), "sha256": hashlib.sha256(payload).hexdigest()})
        for rate in rates:
            window_cost = rate * seconds / 3600
            scenarios.append({
                "concurrency": data["concurrency"], "requests": requests,
                "http_successes": data["http_successes"], "elapsed_seconds": seconds,
                "assumed_hourly_rub": rate, "window_cost_rub": window_cost,
                "per_request_rub": window_cost / requests,
            })
    return {
        "kind": "conditional_compute_cost",
        "actual_cost_rub": None,
        "assumptions": {
            "pricing": "Illustrative sensitivity rates, not a vendor quote or actual expense.",
            "resource": "One dedicated machine billed for the measured wall-clock window.",
            "formula": "window_cost_rub = assumed_hourly_rub * elapsed_seconds / 3600",
            "exclusions": "Warm-up before the window, idle time, storage, setup and maintenance are excluded.",
            "concurrency": "Overlapping request durations are not summed as machine rental time.",
        },
        "sources": sources,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", type=Path, action="append", required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[50, 100, 200])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = estimate(args.load, args.rates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenarios": len(result["scenarios"])}))


if __name__ == "__main__":
    main()
