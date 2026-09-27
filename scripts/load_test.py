"""Параллельные HTTP-запросы к API с учётом задержек, ошибок и доступной telemetry."""

import argparse
import json
import math
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .evaluate import percentile


def run(url, count, workers, gpu_hour_cost=None):
    """successes — ответы с рекомендациями; aggregate usage не считает LLM-успехи."""

    def request_one(index):
        start = time.perf_counter()
        request_id = f"load-{uuid.uuid4().hex}"
        http_ok = False
        try:
            # Замер включает установку клиентского соединения.
            with httpx.Client(timeout=120, trust_env=False) as client:
                response = client.post(
                    url.rstrip("/") + "/v1/chat",
                    headers={"X-Request-ID": request_id},
                    json={"message": "Хочу лёгкий детективный сериал, один сезон", "user_id": f"load-{index}"},
                )
                response.raise_for_status()
                http_ok = True
                data = response.json()
            usage = data.get("llm_usage")
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "request_id": data.get("request_id") or request_id,
                "latency_ms": latency_ms,
                "http_ok": http_ok,
                "ok": bool(data.get("recommendations", [])),
                "mode": data.get("mode"),
                "provider": data.get("provider"),
                "llm_calls": data.get("llm_calls"),
                "llm_tokens": data.get("llm_tokens"),
                "llm_usage": usage,
                "timings_ms": data.get("timings_ms", {}),
                "degradation": data.get("degradation", "FULL"),
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "request_id": request_id,
                "latency_ms": latency_ms,
                "http_ok": http_ok,
                "ok": False,
                "error": type(exc).__name__,
                "failure_reason": str(exc)[:200],
                "mode": "unknown",
                "provider": "unknown",
                "degradation": "UNKNOWN",
            }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(request_one, range(count)))
    elapsed = time.perf_counter() - started
    latencies = [r["latency_ms"] for r in results]
    observed_inference = []
    missing_inference = 0
    observed_tokens = 0
    missing_tokens = 0
    for row in results:
        usage = row.get("llm_usage")
        calls = row.get("llm_calls")
        if calls == 0:
            pass
        elif isinstance(usage, dict) and usage and type(row.get("llm_tokens")) is int and row["llm_tokens"] >= 0:
            observed_tokens += row["llm_tokens"]
        else:
            missing_tokens += 1
        seconds = usage.get("inference_seconds") if isinstance(usage, dict) else None
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0:
            observed_inference.append(seconds)
        elif row.get("llm_calls") != 0:
            # Потерянный HTTP-ответ тоже не подтверждает ноль вызовов.
            missing_inference += 1
    observed_inference_seconds = sum(observed_inference)

    def calls_known(row):
        return type(row.get("llm_calls")) is int and row["llm_calls"] >= 0

    inference_seconds = observed_inference_seconds if missing_inference == 0 and all(calls_known(row) for row in results) else None
    agent_tokens = observed_tokens if missing_tokens == 0 else None
    mode_counts = dict(sorted(Counter(str(row["mode"]) if row.get("mode") is not None else "unknown" for row in results).items()))
    observed_token_requests = len(results) - missing_tokens
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "target": url,
        "requests": count,
        "concurrency": workers,
        "elapsed_seconds": elapsed,
        "requests_per_second": count / elapsed,
        "successes": sum(r["ok"] for r in results),
        "errors": sum(not r["ok"] for r in results),
        "http_successes": sum(r["http_ok"] for r in results),
        "http_errors": sum(not r["http_ok"] for r in results),
        "recommendation_successes": sum(r["ok"] for r in results),
        "p50_ms": percentile(latencies, 0.5),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": max(latencies),
        "agent_llm_calls": sum(r.get("llm_calls") or 0 for r in results),
        "requests_with_llm_usage": sum(isinstance(r.get("llm_usage"), dict) and bool(r["llm_usage"]) for r in results),
        "agent_tokens": agent_tokens,
        "observed_agent_tokens": observed_tokens,
        "mode_counts": mode_counts,
        "observed_inference_seconds": observed_inference_seconds,
        "inference_seconds": inference_seconds,
        "monetary_cost": inference_seconds * gpu_hour_cost / 3600 if gpu_hour_cost is not None and inference_seconds is not None else None,
        "observed_monetary_cost": observed_inference_seconds * gpu_hour_cost / 3600 if gpu_hour_cost is not None else None,
        "usage_coverage": {
            "requests_with_reported_llm_calls": sum(calls_known(row) for row in results),
            "requests_with_llm_attempts": sum((row.get("llm_calls") or 0) > 0 for row in results),
            "requests_with_zero_llm_calls": sum(row.get("llm_calls") == 0 for row in results),
            "requests_with_unknown_llm_calls": sum(not calls_known(row) for row in results),
            "requests_with_inference_seconds": len(observed_inference),
            "requests_missing_inference_seconds": missing_inference,
            "request_inference_complete": missing_inference == 0,
            "requests_with_observed_tokens": observed_token_requests,
            "requests_missing_tokens": missing_tokens,
            "request_tokens_complete": missing_tokens == 0,
        },
        "metric_definitions": {
            "successes": "Историческое поле: ответы с непустыми recommendations; errors — все остальные запросы, включая clarify.",
            "http_successes": "HTTP-ответы без ошибки статуса; clarify тоже является HTTP-успехом.",
            "agent_llm_calls": "Сумма сообщённых попыток LLM; для потерянных ответов число попыток неизвестно.",
            "agent_tokens": "Полная сумма токенов, если coverage полная; иначе null.",
            "observed_agent_tokens": "Сумма токенов по запросам с наблюдаемым usage; не полная при missing tokens.",
            "requests_with_llm_usage": "Запросы с непустым aggregate usage, не количество успешных LLM calls.",
            "usage_coverage": "Покрытие HTTP-запросов, не отдельных LLM-вызовов; полнота per-call usage неизвестна.",
            "observed_inference_seconds": "Сумма доступных замеров; при неполном покрытии не является полным временем инференса.",
        },
        "cost_assumptions": {
            "gpu_hour_cost_rub": gpu_hour_cost,
            "basis": "Иллюстративная ставка за час вычислений; не фактический счёт за электричество. Время инференса берётся из aggregate usage API; полнота отдельных LLM-вызовов неизвестна.",
        },
        "fallback_requests": sum(row.get("degradation") == "NO_LLM" for row in results),
        "results": results,
    }


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
