import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import httpx
from .evaluate import percentile


def run(url, count, workers):
    def request_one(index):
        start = time.perf_counter()
        try:
            # Include client connection setup in end-to-end timing.
            with httpx.Client(timeout=120, trust_env=False) as client:
                response = client.post(url.rstrip("/")+"/v1/chat", json={"message": "Хочу лёгкий детективный сериал, один сезон", "user_id": f"load-{index}"})
                response.raise_for_status()
                data = response.json()
            return {"latency_ms": (time.perf_counter()-start)*1000, "ok": bool(data["recommendations"]), "mode": data["mode"], "llm_calls": data["llm_calls"], "llm_tokens": data["llm_tokens"]}
        except Exception as exc:
            return {"latency_ms": (time.perf_counter()-start)*1000, "ok": False, "error": type(exc).__name__}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(request_one, range(count)))
    elapsed = time.perf_counter()-started
    latencies = [r["latency_ms"] for r in results]
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "target": url, "requests": count, "concurrency": workers, "elapsed_seconds": elapsed, "requests_per_second": count/elapsed, "successes": sum(r["ok"] for r in results), "errors": sum(not r["ok"] for r in results), "p50_ms": percentile(latencies, .5), "p95_ms": percentile(latencies, .95), "max_ms": max(latencies), "agent_llm_calls": sum(r.get("llm_calls", 0) for r in results), "agent_tokens": sum(r.get("llm_tokens", 0) for r in results), "monetary_cost": None, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", default="report/load-test.json")
    args = parser.parse_args()
    if not 1 <= args.requests <= 400 or not 1 <= args.concurrency <= 32:
        parser.error("Use 1..400 requests, 1..32 concurrency; demo store holds 500 sessions")
    result = run(args.url, args.requests, args.concurrency)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))

