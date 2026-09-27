"""Numerical check of unconditional exact McNemar power, no dependencies."""

import json
import math
from pathlib import Path
from statistics import NormalDist

from scripts.sample_size_design import paired_binary_n


def pmf(k, n, p):
    if p == 0:
        return float(k == 0)
    if p == 1:
        return float(k == n)
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p))


def cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0:
        return 1.0
    if p == 1:
        return 0.0
    if k > n * p:
        return 1 - cdf(n - k - 1, n, 1 - p)
    term = pmf(k, n, p)
    result = term
    for j in range(k, 0, -1):
        term *= j / (n - j + 1) * (1 - p) / p
        result += term
        if term < 1e-16 * max(result, 1e-300):
            break
    return result


def power(n, delta, q, alpha=0.05):
    r = (q + delta) / (2 * q)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    result = 0.0
    omitted_mass = 0.0
    for d in range(n + 1):
        weight = pmf(d, n, q)
        if weight < 1e-16:
            omitted_mass += weight
            continue
        cutoff = max(-1, min(d // 2, math.floor(d / 2 - z * math.sqrt(d) / 2)))
        while cutoff >= 0 and cdf(cutoff, d, 0.5) > alpha / 2:
            cutoff -= 1
        while cutoff + 1 < d / 2 and cdf(cutoff + 1, d, 0.5) <= alpha / 2:
            cutoff += 1
        conditional = cdf(cutoff, d, r) + cdf(cutoff, d, 1 - r)
        result += weight * conditional
    return result, omitted_mass


def main():
    # Exhaustive small-N verification of CDF and rejection-probability calculation.
    for n in range(1, 21):
        for p in (0.1, 0.5, 0.8):
            for k in range(n + 1):
                reference = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))
                assert abs(cdf(k, n, p) - reference) < 1e-12
    for n in range(2, 16):
        brute = 0.0
        for d in range(n + 1):
            rejection = sum(pmf(x, d, 0.625) for x in range(d + 1) if min(1.0, 2 * min(cdf(x, d, 0.5), cdf(d - x, d, 0.5))) <= 0.05)
            brute += pmf(d, n, 0.2) * rejection
        assert abs(power(n, 0.05, 0.2)[0] - brute) < 1e-12

    rows = []
    for q in (0.1, 0.2, 0.3, 0.5, 0.785, 0.8, 1.0):
        n = paired_binary_n(mde=0.05, discordance=q)
        actual, omitted = power(n, 0.05, q)
        rows.append({"q": q, "normal_approx_n": n, "exact_test_power_at_approx_n": actual, "omitted_binomial_mass": omitted})
        candidate = n
        while power(candidate, 0.05, q)[0] < 0.8:
            candidate += 1
        rounded = math.ceil(candidate / 10) * 10
        while power(rounded, 0.05, q)[0] < 0.8:
            rounded += 10
        rows[-1].update(
            first_passing_n_at_or_above_approx=candidate,
            verified_rounded_n=rounded,
            exact_test_power_at_rounded_n=power(rounded, 0.05, q)[0],
        )
        combined = max(1064, rounded)
        rows[-1].update(combined_precision_and_power_n=combined, exact_test_power_at_combined_n=power(combined, 0.05, q)[0])
        print(rows[-1], flush=True)
    report = {
        "delta": 0.05,
        "alpha_two_sided": 0.05,
        "test": "two-sided exact conditional binomial McNemar",
        "alternative": "D~Bin(N,q); X|D~Bin(D,(q+delta)/(2q))",
        "rows": rows,
        "verification": "Small N CDF and power checked by exhaustive enumeration.",
        "limitations": [
            "Numerical probability summation, not symbolic exact arithmetic.",
            "Independent pairs assumed; not a solution to language dependence.",
        ],
    }
    Path("report/exact-power-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
