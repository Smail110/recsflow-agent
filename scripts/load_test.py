import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .evaluate import percentile


def run(url, count, workers, gpu_hour_cost=None):
    def request_one(index):
        start = time.perf_counter()
        try:
            # Замер включает установку клиентского соединения.
            with httpx.Client(timeout=120, trust_env=False) as client:
                response = client.post(url.rstrip("/")+"/v1/chat", json={"message": "Хочу лёгкий детективный сериал, один сезон", "user_id": f"load-{index}"})
                response.raise_for_status()
                data = response.json()
            return {"latency_ms": (time.perf_counter()-start)*1000, "ok": bool(data["recommendations"]), "mode": data["mode"], "llm_calls": data["llm_calls"], "llm_tokens": data["llm_tokens"],
                    "llm_usage": data.get("llm_usage", {}), "timings_ms": data.get("timings_ms", {}), "degradation": data.get("degradation", "FULL")}
        except Exception as exc:
            return {"latency_ms": (time.perf_counter()-start)*1000, "ok": False, "error": type(exc).__name__}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(request_one, range(count)))
    elapsed = time.perf_counter()-started
    latencies = [r["latency_ms"] for r in results]
    inference_seconds = sum(row.get("llm_usage", {}).get("inference_seconds", 0) for row in results)
    return {"timestamp_utc": datetime.now(UTC).isoformat(), "target": url, "requests": count, "concurrency": workers, "elapsed_seconds": elapsed, "requests_per_second": count/elapsed, "successes": sum(r["ok"] for r in results), "errors": sum(not r["ok"] for r in results), "p50_ms": percentile(latencies, .5), "p95_ms": percentile(latencies, .95), "max_ms": max(latencies), "agent_llm_calls": sum(r.get("llm_calls", 0) for r in results), "agent_tokens": sum(r.get("llm_tokens", 0) for r in results),
            "inference_seconds": inference_seconds, "monetary_cost": inference_seconds * gpu_hour_cost / 3600 if gpu_hour_cost is not None else None,
            "cost_assumptions": {"gpu_hour_cost_rub": gpu_hour_cost, "basis": "Иллюстративная ставка за час вычислений; не фактический счёт за электричество. Время инференса берётся из Ollama."},
            "fallback_requests": sum(row.get("degradation") == "NO_LLM" for row in results), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", default="report/load-test.json")
    parser.add_argument("--gpu-hour-cost", type=float, help="Предполагаемая стоимость часа вычислений в рублях")
    args = parser.parse_args()
    if not 1 <= args.requests <= 400 or not 1 <= args.concurrency <= 32:
        parser.error("Use 1..400 requests, 1..32 concurrency; demo store holds 500 sessions")
    if args.gpu_hour_cost is not None and args.gpu_hour_cost < 0:
        parser.error("Стоимость часа не может быть отрицательной")
    result = run(args.url, args.requests, args.concurrency, args.gpu_hour_cost)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
