#!/usr/bin/env python3
"""Focused tests for the Part 4 maximum-entropy covariance audit."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run import (
    AUDIT_ID,
    BOUNDED_FIELDS,
    DEFAULT_SEED,
    ESCAPE_FIELDS,
    METHOD_ORIGINS,
    PAIR_FIELDS,
    PAPER_SHA256,
    RANK_FIELDS,
    TAU_FIELDS,
    TOLERANCES,
    _manifest_output_path,
    _paper_input_identity,
    AuditFailure,
    build_cosine_half_pair,
    build_duplicate_count_vs_rank,
    build_full_span_control,
    build_infeasible_control,
    build_pair_rank_domain,
    build_rank_deficient_escape,
    build_seeded_rank_cases,
    build_tau_scale_sweep,
    enforce_checks,
    run_audit,
    validate_summary,
)
from hope_repro.paper_literal.maximum_entropy import (
    appendix_identity_precision,
    appendix_pair_correlation,
    canonical_null_direction,
    determinant_lemma_logdet,
    escape_covariance,
    linear_constraint_residual,
    maximum_constraint_residual,
    numerical_rank,
    positive_definite_logdet,
    projection_variances,
    pure_pair_maximum_entropy_covariance,
    pure_stationarity_precision,
    spanning_logdet_upper_bound,
)
from hope_repro.proposed_repairs.relative_entropy import (
    gaussian_covariance_kl,
    pair_relative_entropy_solution,
    reference_pair_correlation,
    reference_pair_kappa,
    reference_precision,
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PaperLiteralMaximumEntropyTests(unittest.TestCase):
    def test_rank_deficient_escape_preserves_constraints_and_matches_lemma(self) -> None:
        directions = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=np.float64,
        )
        base = np.diag(np.asarray([1.0, 4.0, 1.0]))
        targets = np.asarray([1.0, 4.0])
        rank, threshold, _ = numerical_rank(directions)
        self.assertEqual(rank, 2)
        null = canonical_null_direction(directions, rank_tolerance=threshold)
        np.testing.assert_allclose(directions.T @ null, 0.0, rtol=0.0, atol=1e-15)
        escaped = escape_covariance(base, null, 1e6)
        np.testing.assert_array_equal(projection_variances(escaped, directions), targets)
        self.assertEqual(maximum_constraint_residual(escaped, directions, targets), 0.0)
        self.assertEqual(
            positive_definite_logdet(escaped),
            determinant_lemma_logdet(base, null, 1e6),
        )

    def test_full_span_bound_and_unique_fixture_optimum(self) -> None:
        directions = np.eye(2)
        targets = np.asarray([1.0, 4.0])
        bound = spanning_logdet_upper_bound(directions, targets)
        self.assertGreater(bound, math.log(4.0))
        determinants = []
        for off_diagonal in (-1.5, -0.5, 0.0, 0.5, 1.5):
            covariance = np.asarray(
                [[1.0, off_diagonal], [off_diagonal, 4.0]],
                dtype=np.float64,
            )
            determinants.append(positive_definite_logdet(covariance))
            self.assertLessEqual(determinants[-1], bound)
        self.assertEqual(int(np.argmax(determinants)), 2)
        with self.assertRaises(ValueError):
            spanning_logdet_upper_bound(np.asarray([[1.0], [0.0]]), [1.0])

    def test_full_span_bound_allows_redundant_zero_constraint(self) -> None:
        directions = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        targets = np.asarray([1.0, 4.0, 0.0], dtype=np.float64)
        bound = spanning_logdet_upper_bound(directions, targets)
        self.assertGreaterEqual(bound, math.log(4.0))

    def test_duplicate_targets_distinguish_unbounded_from_infeasible(self) -> None:
        directions = np.asarray([[1.0, 1.0], [0.0, 0.0]])
        self.assertLess(linear_constraint_residual(directions, [1.0, 1.0]), 1e-14)
        self.assertGreater(linear_constraint_residual(directions, [1.0, 2.0]), 0.49)

    def test_cosine_half_pair_separates_pure_and_appendix_objectives(self) -> None:
        directions = np.asarray(
            [[1.0, 0.5], [0.0, math.sqrt(3.0) / 2.0]],
            dtype=np.float64,
        )
        pure = pure_pair_maximum_entropy_covariance(directions, [1.0, 1.0])
        output = directions.T @ pure @ directions
        np.testing.assert_allclose(output, np.eye(2), rtol=0.0, atol=4e-16)
        np.testing.assert_allclose(
            pure,
            np.asarray([[1.0, -1.0 / math.sqrt(3.0)], [-1.0 / math.sqrt(3.0), 5.0 / 3.0]]),
            rtol=2e-15,
            atol=2e-16,
        )
        self.assertAlmostEqual(positive_definite_logdet(pure), math.log(4.0 / 3.0))
        self.assertAlmostEqual(appendix_pair_correlation(0.5, 1.0, 1.0, 1.0, 1.0), 0.5)
        np.testing.assert_allclose(
            np.linalg.inv(pure),
            pure_stationarity_precision(directions, np.ones(2)),
            rtol=2e-15,
            atol=2e-16,
        )
        np.testing.assert_array_equal(
            appendix_identity_precision(directions, np.zeros(2)),
            np.eye(2),
        )

    def test_collinear_pair_is_rejected_by_rank_sensitive_formulas(self) -> None:
        record = build_pair_rank_domain()
        self.assertTrue(record["passed"])
        self.assertEqual(record["rank"], 1)
        self.assertEqual(record["rejection_count"], 3)
        self.assertEqual(
            set(record["rejected_operations"]),
            {
                "pure_two_dimensional_pair_solver",
                "appendix_eq80_eq81_formula",
                "relative_entropy_pair_solver",
            },
        )

    def test_invalid_float64_domains_fail_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            numerical_rank([[math.nan]])
        with self.assertRaises(ValueError):
            positive_definite_logdet([[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaises(ValueError):
            appendix_pair_correlation(1.0, 1.0, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            escape_covariance(np.eye(2), [2.0, 0.0], 1.0)


class RelativeEntropyRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directions = np.asarray(
            [[1.0, 0.5], [0.0, math.sqrt(3.0) / 2.0]],
            dtype=np.float64,
        )
        self.targets = np.ones(2)

    def test_tau_one_recovers_appendix_identity_fixture(self) -> None:
        covariance, multipliers, correlation = pair_relative_entropy_solution(
            self.directions,
            self.targets,
            1.0,
        )
        np.testing.assert_allclose(covariance, np.eye(2), rtol=0.0, atol=4e-16)
        np.testing.assert_allclose(multipliers, 0.0, rtol=0.0, atol=4e-16)
        self.assertAlmostEqual(correlation, 0.5)
        self.assertLessEqual(gaussian_covariance_kl(covariance, 1.0), 1e-15)

    def test_tau_changes_correlation_and_satisfies_stationarity(self) -> None:
        correlations = []
        for tau in (0.5, 1.0, 2.0, 10.0):
            covariance, multipliers, correlation = pair_relative_entropy_solution(
                self.directions,
                self.targets,
                tau,
            )
            correlations.append(correlation)
            np.testing.assert_allclose(
                np.linalg.inv(covariance),
                reference_precision(self.directions, multipliers, tau),
                rtol=3e-15,
                atol=2e-15,
            )
            self.assertLessEqual(
                maximum_constraint_residual(covariance, self.directions, self.targets),
                TOLERANCES["constraint_atol"],
            )
            self.assertAlmostEqual(
                correlation,
                reference_pair_correlation(0.5, 1.0, 1.0, 1.0, 1.0, tau),
            )
        self.assertTrue(all(a > b for a, b in zip(correlations, correlations[1:])))
        self.assertAlmostEqual(
            reference_pair_kappa(0.5, 1.0, 1.0, 1.0, 1.0, 2.0),
            reference_pair_kappa(0.5, 1.0, 1.0, 1.0, 1.0, 1.0) / 4.0,
        )

    def test_invalid_reference_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            pair_relative_entropy_solution(self.directions, self.targets, 0.0)
        with self.assertRaises(ValueError):
            pair_relative_entropy_solution(self.directions[:, :1], [1.0], 1.0)
        with self.assertRaises(ValueError):
            gaussian_covariance_kl(np.eye(2), math.inf)


class DeterministicFixtureTests(unittest.TestCase):
    def test_all_declared_fixture_sections_pass(self) -> None:
        rank_escape, _ = build_rank_deficient_escape()
        duplicate, _ = build_duplicate_count_vs_rank()
        full_span, _ = build_full_span_control()
        infeasible = build_infeasible_control()
        pair, _ = build_cosine_half_pair()
        pair_rank_domain = build_pair_rank_domain()
        tau, _ = build_tau_scale_sweep()
        seeded, _ = build_seeded_rank_cases(DEFAULT_SEED)
        self.assertTrue(rank_escape["all_cases_passed"])
        self.assertTrue(duplicate["all_cases_passed"])
        self.assertEqual(duplicate["rank"], 1)
        self.assertEqual(duplicate["nullity"], 1)
        self.assertTrue(full_span["all_cases_passed"])
        self.assertEqual(full_span["observed_maximum_off_diagonal"], 0.0)
        self.assertTrue(infeasible["passed"])
        self.assertEqual(infeasible["classification"], "infeasible")
        self.assertTrue(pair["passed"])
        self.assertGreater(pair["pure_objective_advantage"], 0.0)
        self.assertTrue(pair_rank_domain["passed"])
        self.assertTrue(tau["all_cases_passed"])
        self.assertTrue(seeded["all_cases_passed"])
        self.assertEqual(seeded["rank_deficient_case_count"], 4)
        self.assertEqual(seeded["full_span_case_count"], 2)

    def test_seeded_cases_repeat_exactly(self) -> None:
        first = build_seeded_rank_cases(DEFAULT_SEED)
        second = build_seeded_rank_cases(DEFAULT_SEED)
        self.assertEqual(first, second)


class OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_context = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls.temp_context.name)
        cls.first_dir = cls.temp_root / "first" / "results"
        cls.second_dir = cls.temp_root / "second" / "results"
        cls.first_manifest = cls.temp_root / "first" / "manifest.json"
        cls.second_manifest = cls.temp_root / "second" / "manifest.json"
        cls.first_result = run_audit(
            result_dir=cls.first_dir,
            manifest_path=cls.first_manifest,
            seed=DEFAULT_SEED,
            paper_pdf=None,
        )
        cls.second_result = run_audit(
            result_dir=cls.second_dir,
            manifest_path=cls.second_manifest,
            seed=DEFAULT_SEED,
            paper_pdf=None,
        )
        cls.summary = json.loads(
            (cls.first_dir / "summary.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(cls.first_manifest.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_context.cleanup()

    def test_summary_rebuild_validator_and_check_registry(self) -> None:
        validate_summary(self.summary)
        self.assertEqual(self.summary["audit_id"], AUDIT_ID)
        self.assertEqual(self.summary["paper"]["pdf_sha256"], PAPER_SHA256)
        self.assertEqual(self.summary["method_origins"], METHOD_ORIGINS)

        changed = copy.deepcopy(self.summary)
        changed["rank_deficient_escape"]["logdet_gain"] += 1.0
        with self.assertRaises(AuditFailure):
            validate_summary(changed)

        checks = copy.deepcopy(self.summary["checks"])
        checks.pop("seeded_failure_count")
        with self.assertRaises(AuditFailure):
            enforce_checks(checks)

    def test_all_result_bytes_are_directory_independent(self) -> None:
        filenames = (
            "summary.json",
            "escape_path.csv",
            "bounded_control.csv",
            "pair_comparison.csv",
            "tau_sweep.csv",
            "rank_cases.csv",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                self.assertEqual(
                    (self.first_dir / filename).read_bytes(),
                    (self.second_dir / filename).read_bytes(),
                )
        self.assertEqual(
            self.first_manifest.read_bytes(),
            self.second_manifest.read_bytes(),
        )

    def test_manifest_authenticates_outputs_and_contains_no_absolute_workspace(self) -> None:
        self.assertEqual(self.manifest["paper"]["pdf_sha256"], PAPER_SHA256)
        self.assertFalse(
            self.manifest["reproduction_scope"]["paper_experiment_reproduced"]
        )
        self.assertEqual(
            set(self.manifest["expected_metric_ranges"]),
            set(self.summary["checks"]),
        )
        output_records = {
            Path(record["path"]).name: record
            for record in self.manifest["outputs"]
        }
        expected = {
            "summary.json",
            "escape_path.csv",
            "bounded_control.csv",
            "pair_comparison.csv",
            "tau_sweep.csv",
            "rank_cases.csv",
        }
        self.assertEqual(set(output_records), expected)
        for filename, record in output_records.items():
            path = self.first_dir / filename
            self.assertEqual(record["sha256"], sha256_path(path))
            self.assertEqual(record["bytes"], path.stat().st_size)
        serialized = json.dumps(self.manifest, sort_keys=True)
        self.assertNotIn(str(Path(__file__).resolve().parents[3]), serialized)
        self.assertNotIn(str(self.temp_root), serialized)

    def test_paper_input_uses_repository_relative_identity(self) -> None:
        paper_path = Path(__file__).resolve().parents[3] / "paper" / "2607.21366v1.pdf"
        self.assertEqual(
            _paper_input_identity(paper_path),
            "paper/2607.21366v1.pdf",
        )

    def test_frozen_outputs_use_repository_relative_identities(self) -> None:
        root = Path(__file__).resolve().parents[3]
        result_dir = root / "artifacts" / "part_04" / "results"
        self.assertEqual(
            _manifest_output_path(result_dir / "summary.json", result_dir),
            "artifacts/part_04/results/summary.json",
        )

    def test_csv_schemas_are_exact_and_rows_pass(self) -> None:
        contracts = {
            "escape_path.csv": ESCAPE_FIELDS,
            "bounded_control.csv": BOUNDED_FIELDS,
            "pair_comparison.csv": PAIR_FIELDS,
            "tau_sweep.csv": TAU_FIELDS,
            "rank_cases.csv": RANK_FIELDS,
        }
        for filename, fields in contracts.items():
            with self.subTest(filename=filename):
                with (self.first_dir / filename).open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), fields)
                self.assertTrue(rows)
                self.assertTrue(all(row["passed"] == "True" for row in rows))

    def test_explicit_missing_paper_and_reserved_manifest_fail_closed(self) -> None:
        with self.assertRaises(AuditFailure):
            run_audit(
                result_dir=self.temp_root / "missing" / "results",
                paper_pdf=self.temp_root / "missing-paper.pdf",
            )
        with self.assertRaises(ValueError):
            run_audit(
                result_dir=self.temp_root / "reserved",
                manifest_path=self.temp_root / "reserved" / "summary.json",
            )


if __name__ == "__main__":
    unittest.main()
