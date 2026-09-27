"""Reproducible calculations for the evaluation sample-size design.

The paired calculation is the large-sample approximation for a two-sided
McNemar comparison.  It is a planning calculation, not a substitute for the
exact paired analysis of the eventual 2x2 table.
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist


def paired_binary_n(
    *,
    mde: float,
    discordance: float,
    alpha: float = 0.05,
    power: float = 0.80,
    hypotheses: int = 1,
) -> int:
    """Return the normal-approximation N for a paired binary difference.

    ``discordance`` is p10 + p01 and ``mde`` is abs(p10 - p01).  When more
    than one confirmatory hypothesis is requested, Bonferroni alpha is used
    for conservative planning; Holm can be used for the final tests.
    """

    if not 0 < mde <= discordance <= 1:
        raise ValueError("require 0 < mde <= discordance <= 1")
    if not 0 < alpha < 1 or not 0.5 < power < 1:
        raise ValueError("require 0 < alpha < 1 and 0.5 < power < 1")
    if isinstance(hypotheses, bool) or not isinstance(hypotheses, int) or hypotheses < 1:
        raise ValueError("hypotheses must be positive")

    z_alpha = NormalDist().inv_cdf(1 - alpha / (2 * hypotheses))
    z_power = NormalDist().inv_cdf(power)
    variance_alternative = discordance - mde**2
    n = (z_alpha * math.sqrt(discordance) + z_power * math.sqrt(variance_alternative)) ** 2 / mde**2
    return math.ceil(n)


def wilson_interval(successes: int, total: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
        or total < 1
        or not 0 <= successes <= total
    ):
        raise ValueError("require total >= 1 and 0 <= successes <= total")
    if not 0 < alpha < 1:
        raise ValueError("require 0 < alpha < 1")

    z = NormalDist().inv_cdf(1 - alpha / 2)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return centre - radius, centre + radius


def wilson_worst_case_n(*, half_width: float, alpha: float = 0.05, hypotheses: int = 1) -> int:
    """Return N whose worst-case Wilson half-width is at most ``half_width``."""

    if not 0 < half_width < 0.5 or not 0 < alpha < 1:
        raise ValueError("require 0 < half_width < 0.5 and 0 < alpha < 1")
    if isinstance(hypotheses, bool) or not isinstance(hypotheses, int) or hypotheses < 1:
        raise ValueError("hypotheses must be positive")

    z = NormalDist().inv_cdf(1 - alpha / (2 * hypotheses))
    return math.ceil(z**2 * (1 / (4 * half_width**2) - 1))


def zero_failure_n(*, upper_rate: float, alpha_one_sided: float = 0.05) -> int:
    """Return N needed for a one-sided exact upper bound after zero failures."""

    if not 0 < upper_rate < 1 or not 0 < alpha_one_sided < 1:
        raise ValueError("rates must lie strictly between zero and one")
    return math.ceil(math.log(alpha_one_sided) / math.log(1 - upper_rate))


def design_effect(*, mean_cluster_size: float, icc: float) -> float:
    """Return the equal-size illustrative cluster design effect."""

    if mean_cluster_size < 1 or not 0 <= icc <= 1:
        raise ValueError("require mean_cluster_size >= 1 and 0 <= icc <= 1")
    return 1 + (mean_cluster_size - 1) * icc


def build_design() -> dict:
    discordances = (0.10, 0.20, 0.30, 0.50, 0.80, 1.00)
    paired = {f"{q:.2f}": paired_binary_n(mde=0.05, discordance=q) for q in discordances}
    paired_five_claims = {f"{q:.2f}": paired_binary_n(mde=0.05, discordance=q, hypotheses=5) for q in discordances}
    mean_grammar_cluster = 400 / 53
    cluster_sensitivity = []
    for icc in (0.05, 0.10, 0.20):
        effect = design_effect(mean_cluster_size=mean_grammar_cluster, icc=icc)
        cluster_sensitivity.append(
            {
                "icc": icc,
                "mean_cluster_size": mean_grammar_cluster,
                "design_effect": effect,
                "raw_for_effective_1064": math.ceil(1064 * effect),
                "raw_for_exact_rounded_effective_1610": math.ceil(1610 * effect),
                "raw_for_exact_rounded_effective_3170": math.ceil(3170 * effect),
            }
        )
    return {
        "assumptions": {"alpha_two_sided": 0.05, "power": 0.80, "paired_mde": 0.05},
        "paired_normal_approx_n_by_discordance": paired,
        "paired_normal_approx_n_five_confirmatory_claims_bonferroni_planning": paired_five_claims,
        "wilson_worst_case_n": {
            "half_width_0.03": wilson_worst_case_n(half_width=0.03),
            "half_width_0.10": wilson_worst_case_n(half_width=0.10),
            "five_simultaneous_half_width_0.10_bonferroni": wilson_worst_case_n(half_width=0.10, hypotheses=5),
        },
        "diagnostic_allocation": {
            "overall_effective_profiles": 1064,
            "per_critical_semantic_stratum": 93,
            "critical_semantic_strata": ["hard_constraints", "negation", "conflict", "no_results", "domain_change"],
            "per_pilot_zero_success_family": 93,
            "pilot_zero_success_families": [
                "contradiction",
                "course_practical",
                "domain_switch",
                "mood_evening",
                "similar_seed",
                "tone_negation",
            ],
            "allocation_rule": "Pre-labelled cases may overlap strata; do not sum quotas or pad exhausted grammar cores mechanically.",
        },
        "observed_intervals": {
            "current_dev_147_of_400": wilson_interval(147, 400),
            "zero_of_25": wilson_interval(0, 25),
        },
        "zero_failure_exact_one_sided_n": {
            "upper_rate_0.05": zero_failure_n(upper_rate=0.05),
            "upper_rate_0.01": zero_failure_n(upper_rate=0.01),
        },
        "illustrative_grammar_cluster_sensitivity": cluster_sensitivity,
        "limitations": [
            "Future candidate-vs-baseline discordance is unknown.",
            "The paired formula is a normal approximation, not a guaranteed 80%-power sample size after exact-test discreteness.",
            "Exact-test verification and rounded planning values are recorded separately in report/exact-power-check.json.",
            "The equal-cluster design effect is sensitivity analysis, not an estimate of grammar dependence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build_design(), ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
