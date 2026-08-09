#!/usr/bin/env python3
"""Focused tests for the Part 2 closed-form capacity audit."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from run import (
    AUDIT_ID,
    DEFAULT_SEED,
    METHOD_ORIGINS,
    PAPER_SHA256,
    TOLERANCES,
    AuditFailure,
    _code_identity,
    build_checks,
    build_eq6_audit,
    build_monte_carlo_audit,
    build_quadrature_audit,
    build_ranking_audit,
    build_stability_audit,
    build_zero_shift_audit,
    deterministic_quadrature_self_kernel,
    enforce_checks,
    neuron_capacity,
    pruning_cost,
    pruning_cost_derivative,
    relu_self_kernel,
    run_audit,
    validate_summary,
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ClosedFormKernelTests(unittest.TestCase):
    def test_known_values_include_stable_negative_tail(self) -> None:
        cases = (
            (0.0, 1.0, 0.5),
            (0.0, -2.0, 2.0),
            (1.0, 1.0, 1.9246602166562294),
            (-1.0, 1.0, 0.07533978334377082),
            (-8.0, 1.0, 1.8075064471458513e-17),
            (-10.0, 1.0, 1.4529276957119824e-25),
            (-20.0, 1.0, 1.3599129147073856e-91),
            (-30.0, 1.0, 1.0843724873983507e-200),
        )
        for beta, gamma, expected in cases:
            with self.subTest(beta=beta, gamma=gamma):
                np.testing.assert_allclose(
                    relu_self_kernel(beta, gamma),
                    expected,
                    rtol=3e-11,
                    atol=0.0,
                )

    def test_vectorized_broadcast_matches_scalar_evaluation(self) -> None:
        beta = np.asarray([[-30.0], [-1.0], [0.0], [2.0]])
        gamma = np.asarray([[1.0, -0.5, 0.0]])
        values = relu_self_kernel(beta, gamma)
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values.shape, (4, 3))
        for row in range(beta.shape[0]):
            for column in range(gamma.shape[1]):
                self.assertEqual(
                    values[row, column],
                    relu_self_kernel(beta[row, 0], gamma[0, column]),
                )

    def test_gamma_zero_uses_deterministic_relu(self) -> None:
        beta = np.asarray([-3.0, -0.0, 0.5, 2.0])
        expected = np.asarray([0.0, 0.0, 0.25, 4.0])
        np.testing.assert_array_equal(relu_self_kernel(beta, 0.0), expected)

    def test_zero_shift_is_exact_across_signs_and_scales(self) -> None:
        audit, _ = build_zero_shift_audit()
        self.assertTrue(audit["aggregate"]["all_cases_passed"])
        self.assertEqual(audit["aggregate"]["max_abs_error"], 0.0)
        for record in audit["cases"]:
            self.assertEqual(record["analytical"], record["identity_value"])

    def test_extreme_ratios_are_finite_nonnegative_and_complementary(self) -> None:
        audit, _ = build_stability_audit()
        self.assertTrue(audit["aggregate"]["all_cases_passed"])
        self.assertEqual(audit["aggregate"]["nonfinite_count"], 0)
        self.assertEqual(audit["aggregate"]["negative_count"], 0)
        self.assertEqual(audit["aggregate"]["upper_bound_violation_count"], 0)
        self.assertLessEqual(
            audit["aggregate"]["max_complement_scaled_error"], 1.0
        )

    def test_near_degenerate_upper_bound_uses_declared_one_sided_tolerance(self) -> None:
        audit, _ = build_stability_audit()
        case = next(
            record
            for record in audit["cases"]
            if record["case_id"] == "near_degenerate_positive"
        )
        expected_tolerance = (
            TOLERANCES["stability_upper_bound_rtol"]
            * abs(case["second_moment_upper_bound"])
        )
        self.assertEqual(case["upper_bound_tolerance"], expected_tolerance)
        self.assertLessEqual(case["upper_bound_excess"], expected_tolerance)
        self.assertTrue(case["bounded_by_ungated_second_moment"])
        self.assertLessEqual(case["upper_bound_scaled_excess"], 1.0)

    def test_capacity_factorization_and_validation(self) -> None:
        beta = np.asarray([0.0, 1.0, -1.0])
        gamma = np.asarray([2.0, 1.0, 1.0])
        outgoing = np.asarray([3.0, 0.5, 4.0])
        expected = outgoing * np.sqrt(relu_self_kernel(beta, gamma))
        np.testing.assert_array_equal(
            neuron_capacity(beta, gamma, outgoing), expected
        )
        with self.assertRaises(ValueError):
            neuron_capacity(0.0, 1.0, -1.0)

    def test_nonfinite_inputs_and_true_overflow_fail_explicitly(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(beta=bad):
                with self.assertRaises(ValueError):
                    relu_self_kernel(bad, 1.0)
            with self.subTest(gamma=bad):
                with self.assertRaises(ValueError):
                    relu_self_kernel(0.0, bad)
        with self.assertRaises(FloatingPointError):
            relu_self_kernel(sys.float_info.max, 0.0)
        with self.assertRaises(FloatingPointError):
            relu_self_kernel(0.0, sys.float_info.max)


class IndependentNumericalCheckTests(unittest.TestCase):
    def test_declared_quadrature_grid_passes_and_repeats_exactly(self) -> None:
        audit, _ = build_quadrature_audit(order=256, z_limit=12.0)
        self.assertEqual(audit["aggregate"]["case_count"], 28)
        self.assertTrue(audit["aggregate"]["all_cases_passed"])
        self.assertEqual(audit["aggregate"]["repeat_max_abs_difference"], 0.0)
        self.assertLessEqual(audit["aggregate"]["max_scaled_error"], 1.0)

    def test_quadrature_handles_gamma_zero_without_division(self) -> None:
        values = deterministic_quadrature_self_kernel(
            np.asarray([-2.0, 0.0, 2.0]),
            0.0,
            order=64,
            z_limit=10.0,
        )
        np.testing.assert_array_equal(values, np.asarray([0.0, 0.0, 4.0]))

    def test_invalid_quadrature_grid_fails(self) -> None:
        with self.assertRaises(ValueError):
            deterministic_quadrature_self_kernel(0.0, 1.0, order=8)
        with self.assertRaises(ValueError):
            deterministic_quadrature_self_kernel(0.0, 1.0, z_limit=7.0)

    def test_seeded_monte_carlo_is_reproducible_and_secondary(self) -> None:
        first, _ = build_monte_carlo_audit(DEFAULT_SEED, 20_000, 4_000)
        second, _ = build_monte_carlo_audit(DEFAULT_SEED, 20_000, 4_000)
        self.assertEqual(first, second)
        self.assertEqual(first["role"], "secondary numerical check, not proof")
        self.assertTrue(first["aggregate"]["all_cases_passed"])
        self.assertLessEqual(first["aggregate"]["max_scaled_error"], 1.0)

    def test_monte_carlo_sample_count_is_configurable_but_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_monte_carlo_audit(DEFAULT_SEED, 999, 100)
        with self.assertRaises(ValueError):
            build_monte_carlo_audit(DEFAULT_SEED, 1_000, 0)


class RankingAndEq6Tests(unittest.TestCase):
    def test_four_scores_produce_valid_and_distinct_rankings(self) -> None:
        audit, _ = build_ranking_audit()
        expected_methods = {
            "incoming_l1",
            "abs_gamma",
            "abs_gamma_x_outgoing_l2",
            "full_capacity",
        }
        self.assertEqual(set(audit["orders"]), expected_methods)
        self.assertEqual(audit["aggregate"]["method_count"], 4)
        self.assertGreaterEqual(audit["aggregate"]["unique_order_count"], 3)
        self.assertEqual(
            audit["aggregate"]["baseline_identical_to_full_count"], 0
        )
        self.assertTrue(audit["aggregate"]["all_orders_are_permutations"])

    def test_zero_shift_capacity_matches_refined_bn_route_score(self) -> None:
        audit, _ = build_ranking_audit()
        for neuron in audit["neurons"]:
            if neuron["beta"] == 0.0:
                expected = neuron["abs_gamma_x_outgoing_l2"] / math.sqrt(2.0)
                np.testing.assert_allclose(
                    neuron["full_capacity"], expected, rtol=3e-15, atol=0.0
                )

    def test_eq6_cost_is_strictly_increasing_with_positive_derivative(self) -> None:
        audit, _ = build_eq6_audit()
        aggregate = audit["aggregate"]
        self.assertEqual(aggregate["nonpositive_derivative_count"], 0)
        self.assertEqual(aggregate["nonincreasing_step_count"], 0)
        self.assertTrue(aggregate["capacity_and_cost_order_match"])
        self.assertLessEqual(aggregate["max_secant_scaled_error"], 1.0)

    def test_eq6_closed_forms_and_domain_guards(self) -> None:
        capacities = np.asarray([0.0, 1.0, 3.0])
        expected_cost = 128.0 * capacities / (10.0 - capacities)
        expected_derivative = 128.0 * 10.0 / (10.0 - capacities) ** 2
        np.testing.assert_array_equal(
            pruning_cost(capacities, 128.0, 10.0), expected_cost
        )
        np.testing.assert_array_equal(
            pruning_cost_derivative(capacities, 128.0, 10.0),
            expected_derivative,
        )
        for invalid in (-1.0, 10.0, 11.0):
            with self.subTest(capacity=invalid):
                with self.assertRaises(ValueError):
                    pruning_cost(invalid, 128.0, 10.0)


class OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_context = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls.temp_context.name)
        cls.result_dir = cls.temp_root / "results"
        cls.manifest_path = cls.temp_root / "promoted" / "manifest.json"
        cls.result = run_audit(
            result_dir=cls.result_dir,
            manifest_path=cls.manifest_path,
            seed=DEFAULT_SEED,
            mc_samples=10_000,
            mc_chunk_size=2_000,
            quadrature_order=256,
            quadrature_z_limit=12.0,
            paper_pdf=None,
            invocation=("unit-test", "run_audit"),
        )
        cls.summary = json.loads(
            (cls.result_dir / "summary.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_context.cleanup()

    def test_summary_revalidates_with_public_fail_closed_validator(self) -> None:
        validate_summary(self.summary)
        self.assertEqual(self.summary["audit_id"], AUDIT_ID)
        self.assertEqual(self.summary["status"], "passed")
        self.assertTrue(self.summary["passed"])

    def test_explicit_missing_paper_pdf_fails_closed(self) -> None:
        missing = self.temp_root / "missing-paper.pdf"
        with self.assertRaises(AuditFailure):
            run_audit(
                result_dir=self.temp_root / "missing-paper-results",
                manifest_path=self.temp_root / "missing-paper-manifest.json",
                seed=DEFAULT_SEED,
                mc_samples=10_000,
                mc_chunk_size=2_000,
                quadrature_order=256,
                quadrature_z_limit=12.0,
                paper_pdf=missing,
                invocation=("unit-test", "missing-paper"),
            )

    def test_summary_validator_rejects_changed_metrics_and_config(self) -> None:
        changed_metric = copy.deepcopy(self.summary)
        changed_metric["checks"]["quadrature_case_count"]["value"] = 27.0
        with self.assertRaises(AuditFailure):
            validate_summary(changed_metric)

        changed_config = copy.deepcopy(self.summary)
        changed_config["config"]["seed"] += 1
        with self.assertRaises(AuditFailure):
            validate_summary(changed_config)

        changed_schema = copy.deepcopy(self.summary)
        changed_schema["paper_experiment_reproduced"] = True
        with self.assertRaises(AuditFailure):
            validate_summary(changed_schema)

    def test_summary_validator_rejects_forged_detailed_numerics(self) -> None:
        changed_quadrature = copy.deepcopy(self.summary)
        changed_quadrature["analytical_quadrature"]["cases"][0][
            "analytical"
        ] = 999.0
        with self.assertRaises(AuditFailure):
            validate_summary(changed_quadrature)

        changed_ranking = copy.deepcopy(self.summary)
        changed_ranking["rankings"]["neurons"][0]["full_capacity"] = 999.0
        with self.assertRaises(AuditFailure):
            validate_summary(changed_ranking)
        with self.assertRaises(TypeError):
            validate_summary(changed_ranking, recomputed_sections={})

        changed_tiny_kernel = copy.deepcopy(self.summary)
        tiny_case = next(
            record
            for record in changed_tiny_kernel["stability"]["cases"]
            if record["case_id"] == "near_degenerate_positive"
        )
        tiny_case["analytical"] = 4e-14
        with self.assertRaises(AuditFailure):
            validate_summary(changed_tiny_kernel)

        changed_tiny_diagnostic = copy.deepcopy(self.summary)
        tiny_diagnostic_case = next(
            record
            for record in changed_tiny_diagnostic["stability"]["cases"]
            if record["case_id"] == "near_degenerate_positive"
        )
        tiny_diagnostic_case["upper_bound_excess"] = 4e-13
        with self.assertRaises(AuditFailure):
            validate_summary(changed_tiny_diagnostic)

        changed_exact_zero = copy.deepcopy(self.summary)
        origin_case = next(
            record
            for record in changed_exact_zero["stability"]["cases"]
            if record["case_id"] == "gamma_zero_origin"
        )
        origin_case["analytical"] = 4e-14
        with self.assertRaises(AuditFailure):
            validate_summary(changed_exact_zero)

    def test_check_registry_rejects_missing_and_out_of_range_records(self) -> None:
        checks = copy.deepcopy(self.summary["checks"])
        checks.pop("quadrature_case_count")
        with self.assertRaises(AuditFailure):
            enforce_checks(checks)

        checks = copy.deepcopy(self.summary["checks"])
        checks["quadrature_max_scaled_error"]["value"] = 2.0
        checks["quadrature_max_scaled_error"]["passed"] = False
        with self.assertRaises(AuditFailure):
            enforce_checks(checks)

    def test_exact_output_names_and_separate_manifest_path(self) -> None:
        self.assertEqual(Path(self.result["metrics"]).name, "metrics.csv")
        self.assertEqual(Path(self.result["summary"]).name, "summary.json")
        self.assertEqual(Path(self.result["manifest"]), self.manifest_path)
        self.assertTrue((self.result_dir / "metrics.csv").is_file())
        self.assertTrue((self.result_dir / "summary.json").is_file())
        self.assertTrue(self.manifest_path.is_file())

    def test_manifest_has_complete_hash_and_scope_contract(self) -> None:
        self.assertEqual(self.manifest["paper"]["pdf_sha256"], PAPER_SHA256)
        self.assertEqual(self.manifest["paper"]["local_pdf"]["status"], "not_available")
        self.assertEqual(self.manifest["config_sha256"], self.summary["config_sha256"])
        self.assertEqual(self.manifest["method_origins"], METHOD_ORIGINS)
        self.assertFalse(
            self.manifest["reproduction_scope"]["paper_experiment_reproduced"]
        )
        self.assertEqual(
            set(self.manifest["expected_metric_ranges"]),
            set(self.summary["checks"]),
        )
        self.assertGreaterEqual(len(self.manifest["source_hashes"]), 3)
        self.assertGreaterEqual(len(self.manifest["input_hashes"]), 7)
        source_commit = self.manifest["code"]["source_snapshot_commit"]
        if source_commit is not None:
            self.assertEqual(len(source_commit), 40)
            self.assertTrue(
                all(character in "0123456789abcdef" for character in source_commit)
            )
        self.assertIn(
            self.manifest["code"]["source_snapshot_state"],
            {
                "exact_commit",
                "working_tree_with_individually_hashed_sources",
                "source_archive_with_individually_hashed_sources",
            },
        )
        self.assertIn("python_version", self.manifest["environment"])
        self.assertIn("numpy_version", self.manifest["environment"])

        output_records = {Path(record["path"]).name: record for record in self.manifest["outputs"]}
        self.assertEqual(set(output_records), {"metrics.csv", "summary.json"})
        for filename, record in output_records.items():
            path = self.result_dir / filename
            self.assertEqual(record["sha256"], sha256_path(path))
            self.assertEqual(record["bytes"], path.stat().st_size)

    def test_source_archive_identity_uses_hashed_sources_without_git(self) -> None:
        with mock.patch("run._git_value", return_value="unavailable"):
            identity = _code_identity()
        self.assertEqual(
            identity,
            {
                "source_snapshot_commit": None,
                "source_snapshot_state": (
                    "source_archive_with_individually_hashed_sources"
                ),
                "source_files_match_commit": False,
            },
        )

        workspace_root = str(Path(__file__).resolve().parents[3])
        serialized_manifest = json.dumps(self.manifest, sort_keys=True)
        serialized_summary = json.dumps(self.summary, sort_keys=True)
        self.assertNotIn(workspace_root, serialized_manifest)
        self.assertNotIn(workspace_root, serialized_summary)

    def test_csv_is_long_form_and_labels_simulated_rankings(self) -> None:
        with (self.result_dir / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        sections = {row["section"] for row in rows}
        self.assertEqual(
            sections,
            {
                "analytical_quadrature",
                "monte_carlo",
                "zero_shift",
                "stability",
                "rankings",
                "eq6_monotonicity",
            },
        )
        ranking_rows = [row for row in rows if row["section"] == "rankings"]
        self.assertTrue(ranking_rows)
        self.assertTrue(
            all(
                row["method_origin"]
                == "independent_synthetic_ranking_comparison"
                for row in ranking_rows
            )
        )


if __name__ == "__main__":
    unittest.main()
