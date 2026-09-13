"""Aggregation and paired intervals; independent of the recommendation agent."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import fmean


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    latencies = [latency for record in records for latency in record["latencies_ms"]]
    claims = sum(record["evidence_count"] for record in records)
    text_claims = sum(record.get("text_claims", 0) for record in records)
    return {
        "scenarios": len(records),
        "success_rate": fmean(record["success"] for record in records),
        "final_success_rate": fmean(record["final_success"] for record in records),
        "mean_slate_size": fmean(len(record["shown_ids"]) for record in records),
        "mean_clarifications": fmean(record["clarifications"] for record in records),
        "hard_constraint_violations": sum(record["hard_constraint_violations"] for record in records),
        "failure_counts": dict(sorted(Counter(reason for record in records for reason in record["failures"]).items())),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "llm_calls": sum(record["llm_calls"] for record in records),
        "llm_tokens": sum(record["llm_tokens"] for record in records),
        "evidence_count": claims,
        "text_claims": text_claims,
        "unsupported_text_claim_rate": sum(record.get("invalid_text_claims", 0) for record in records) / text_claims if text_claims else None,
        "invalid_evidence_rate": sum(record["invalid_evidence"] for record in records) / claims if claims else None,
    }


def paired_success_interval(left: Sequence[dict], right: Sequence[dict], *, seed: int = 42, resamples: int = 1000) -> dict:
    """Right minus left, resampling whole profiles to retain within-user dependence."""
    if resamples < 1 or not left or not right:
        raise ValueError("nonempty pairs and positive resamples are required")
    left_by_id = {row["scenario_id"]: row for row in left}
    right_by_id = {row["scenario_id"]: row for row in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right) or left_by_id.keys() != right_by_id.keys():
        raise ValueError("paired comparison requires identical, unique scenario ids")
    groups: dict[str, list[int]] = defaultdict(list)
    for scenario_id in sorted(left_by_id):
        a, b = left_by_id[scenario_id], right_by_id[scenario_id]
        if a["user_id"] != b["user_id"]:
            raise ValueError("paired rows must refer to the same profile")
        groups[a["user_id"]].append(int(b["success"]) - int(a["success"]))
    clusters = list(groups.values())
    rng = random.Random(seed)
    samples = [fmean(delta for cluster in rng.choices(clusters, k=len(clusters)) for delta in cluster) for _ in range(resamples)]
    return {
        "delta_success_rate": fmean(delta for cluster in clusters for delta in cluster),
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "resamples": resamples,
        "seed": seed,
        "clusters": len(clusters),
        "resampling_unit": "user_id",
    }
