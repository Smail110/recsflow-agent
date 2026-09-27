import unittest

from scripts.sample_size_design import (
    build_design,
    design_effect,
    paired_binary_n,
    wilson_interval,
    wilson_worst_case_n,
    zero_failure_n,
)


class SampleSizeDesignTests(unittest.TestCase):
    def test_registered_paired_sensitivity_numbers(self) -> None:
        self.assertEqual(
            [paired_binary_n(mde=0.05, discordance=q) for q in (0.10, 0.20, 0.30, 0.50)],
            [312, 626, 940, 1568],
        )

    def test_precision_and_observed_pilot_numbers(self) -> None:
        self.assertEqual(wilson_worst_case_n(half_width=0.03), 1064)
        self.assertEqual(wilson_worst_case_n(half_width=0.10), 93)
        self.assertEqual(wilson_worst_case_n(half_width=0.10, hypotheses=5), 160)
        observed = wilson_interval(147, 400)
        self.assertAlmostEqual(observed[0], 0.32172143852026674)
        self.assertAlmostEqual(observed[1], 0.41579931947832177)
        self.assertAlmostEqual(wilson_interval(0, 25)[0], 0.0, places=12)
        self.assertAlmostEqual(wilson_interval(0, 25)[1], 0.13319225093904843, places=12)

    def test_confirmatory_multiplicity_and_zero_failure_boundaries(self) -> None:
        self.assertEqual(paired_binary_n(mde=0.05, discordance=0.20, hypotheses=5), 932)
        self.assertEqual(zero_failure_n(upper_rate=0.05), 59)
        self.assertEqual(zero_failure_n(upper_rate=0.01), 299)

    def test_cluster_sensitivity_is_explicitly_illustrative(self) -> None:
        self.assertAlmostEqual(design_effect(mean_cluster_size=400 / 53, icc=0.10), 1.6547169811)
        design = build_design()
        self.assertEqual(design["paired_normal_approx_n_by_discordance"]["1.00"], 3138)
        self.assertEqual(design["diagnostic_allocation"]["per_critical_semantic_stratum"], 93)
        self.assertEqual(design["illustrative_grammar_cluster_sensitivity"][1]["raw_for_exact_rounded_effective_1610"], 2665)
        self.assertIn("not an estimate", design["limitations"][3])

    def test_invalid_inputs_fail(self) -> None:
        invalid_calls = [
            (paired_binary_n, {"mde": 0.05, "discordance": 0.04}),
            (paired_binary_n, {"mde": 0.05, "discordance": 0.20, "hypotheses": True}),
            (wilson_interval, {"successes": 2, "total": 1}),
            (wilson_interval, {"successes": True, "total": 1}),
            (wilson_worst_case_n, {"half_width": 0.5}),
            (zero_failure_n, {"upper_rate": 0.0}),
            (design_effect, {"mean_cluster_size": 0.0, "icc": 0.1}),
        ]
        for function, kwargs in invalid_calls:
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(**kwargs)
