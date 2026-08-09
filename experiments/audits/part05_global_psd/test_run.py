#!/usr/bin/env python3
"""Robust unit and artifact-contract tests for the Part 5 global PSD audit."""

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
    AUDIT_SCHEMA_VERSION,
    CHECK_SPECS,
    CLAIM_IDS,
    CSV_CONTRACTS,
    MATRIX_FIELDS,
    METHOD_ORIGINS,
    NEGATIVE_DIRECTION_FIELDS,
    PAIR_FIELDS,
    PAPER_SHA256,
    PROJECTION_FIELDS,
    REPRODUCTION_SCOPE,
    SOURCE_RELATIVE_PATHS,
    SPECTRA_FIELDS,
    TOLERANCES,
    WITNESS_R,
    WITNESS_SCALE_RATIOS,
    AuditFailure,
    _manifest_output_path,
    _paper_input_identity,
    assemble_normalized_eq85,
    assemble_warped_correlation,
    build_audit,
    canonical_eigensystem,
    enforce_checks,
    eq80_warped_correlation,
    eq81_kappa,
    maximum_triangle_violation,
    negative_direction_certificate,
    normalized_eq85_relu_kernel,
    pairwise_squared_distances,
    plain_eigenvalue_clipping,
    psd_renormalized_correlation,
    run_audit,
    validate_summary,
)


ROOT = Path(__file__).resolve().parents[3]


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PaperLiteralFormulaTests(unittest.TestCase):
    def test_fixed_pair_reconstructs_exact_eq80_eq81_values(self) -> None:
        kappa = eq81_kappa(-0.173508, 1.106838, 1.793881)
        self.assertAlmostEqual(kappa, -0.35519961465356764, places=15)
        warped = eq80_warped_correlation(kappa)
        self.assertAlmostEqual(warped, -0.31904414251477403, places=15)
        self.assertLess(abs(warped / (1.0 - warped * warped) - kappa), 2e-16)

    def test_eq80_is_odd_and_strictly_inside_domain(self) -> None:
        for kappa in (0.0, 1e-12, 0.5, 100.0, 1e6):
            positive = eq80_warped_correlation(kappa)
            negative = eq80_warped_correlation(-kappa)
            self.assertEqual(negative, -positive)
            self.assertLess(abs(positive), 1.0)

    def test_eq85_closed_domain_values(self) -> None:
        self.assertEqual(normalized_eq85_relu_kernel(-1.0), 0.0)
        self.assertAlmostEqual(normalized_eq85_relu_kernel(0.0), 1.0 / math.pi)
        self.assertEqual(normalized_eq85_relu_kernel(1.0), 1.0)

    def test_formula_domains_fail_closed(self) -> None:
        for correlation in (-1.0, 1.0, math.nan, math.inf):
            with self.subTest(correlation=correlation):
                with self.assertRaises(ValueError):
                    eq81_kappa(correlation, 1.0, 1.0)
        for ratio_i, ratio_j in ((0.0, 1.0), (-1.0, 1.0), (1.0, math.nan)):
            with self.subTest(ratio_i=ratio_i, ratio_j=ratio_j):
                with self.assertRaises(ValueError):
                    eq81_kappa(0.25, ratio_i, ratio_j)
        for kappa in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                eq80_warped_correlation(kappa)
        for correlation in (-1.01, 1.01, math.nan, math.inf):
            with self.assertRaises(ValueError):
                normalized_eq85_relu_kernel(correlation)


class FixedWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = build_audit()
        cls.raw = np.asarray(cls.audit["matrices"]["input_R"])
        cls.warped = np.asarray(cls.audit["matrices"]["warped_Q"])
        cls.kernel = np.asarray(cls.audit["matrices"]["normalized_eq85_K"])
        cls.shared = np.asarray(cls.audit["matrices"]["shared_eq85_control"])

    def test_declared_constants_are_exact_and_input_is_positive_definite(self) -> None:
        self.assertEqual(tuple(map(tuple, self.raw)), WITNESS_R)
        self.assertEqual(tuple(self.audit["witness"]["scale_ratios"]), WITNESS_SCALE_RATIOS)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(self.raw),
            [0.12600540389160284, 1.1716390523014248, 1.7023555438069724],
            rtol=0.0,
            atol=5e-15,
        )
        self.assertGreater(np.linalg.eigvalsh(self.raw)[0], 0.12)

    def test_warped_matrix_and_eq85_matrix_match_fixed_witness(self) -> None:
        expected_warped = np.asarray(
            [
                [1.0, -0.31904414251477403, 0.7410265449389223],
                [-0.31904414251477403, 1.0, 0.7851452758112819],
                [0.7410265449389223, 0.7851452758112819, 1.0],
            ]
        )
        expected_kernel = np.asarray(
            [
                [1.0, 0.1751298793964040, 0.7811123043000225],
                [0.1751298793964040, 1.0, 0.8153656368172824],
                [0.7811123043000225, 0.8153656368172824, 1.0],
            ]
        )
        np.testing.assert_allclose(self.warped, expected_warped, rtol=0.0, atol=5e-16)
        np.testing.assert_allclose(self.kernel, expected_kernel, rtol=0.0, atol=5e-16)
        np.testing.assert_allclose(assemble_normalized_eq85(self.warped), self.kernel)

    def test_every_local_two_by_two_pair_is_positive_definite(self) -> None:
        self.assertEqual(len(self.audit["local_pairs"]), 3)
        self.assertEqual(
            [record["pair_id"] for record in self.audit["local_pairs"]],
            ["1-2", "1-3", "2-3"],
        )
        for record in self.audit["local_pairs"]:
            with self.subTest(pair=record["pair_id"]):
                self.assertTrue(record["passed"])
                self.assertGreater(record["warped_local_min_eigenvalue"], 0.21)
                self.assertGreater(record["eq85_local_min_eigenvalue"], 0.18)
                self.assertLessEqual(
                    record["eq80_equation_residual"],
                    TOLERANCES["equation_residual_atol"],
                )

    def test_full_assemblies_are_materially_indefinite(self) -> None:
        np.testing.assert_allclose(
            np.linalg.eigvalsh(self.warped),
            [-0.25067976835991007, 1.3183990151721292, 1.932280753187781],
            rtol=0.0,
            atol=5e-15,
        )
        np.testing.assert_allclose(
            np.linalg.eigvalsh(self.kernel),
            [-0.04506976334652169, 0.8250394188974253, 2.220030344449095],
            rtol=0.0,
            atol=5e-15,
        )
        self.assertLess(np.linalg.eigvalsh(self.warped)[0], -0.20)
        self.assertLess(np.linalg.eigvalsh(self.kernel)[0], -0.04)

    def test_negative_directions_are_canonical_and_match_quadratic_forms(self) -> None:
        expected_vectors = {
            "warped_Q": [-0.5260182870219043, -0.5443832096101124, 0.6534153983593697],
            "normalized_eq85_K": [
                -0.4656516215358381,
                -0.4945479775847531,
                0.7338875017520566,
            ],
        }
        for name, expected in expected_vectors.items():
            certificate = self.audit["negative_directions"][name]
            np.testing.assert_allclose(certificate["vector"], expected, rtol=0.0, atol=5e-15)
            self.assertTrue(certificate["passed"])
            self.assertAlmostEqual(certificate["vector_norm"], 1.0)
            self.assertAlmostEqual(
                certificate["quadratic_form"], certificate["minimum_eigenvalue"]
            )
            pivot = int(np.argmax(np.abs(certificate["vector"])))
            self.assertGreater(certificate["vector"][pivot], 0.0)

    def test_all_pairwise_squared_distances_are_nonnegative_but_eq85_is_not_metric(self) -> None:
        for name in ("warped_Q", "normalized_eq85_K"):
            records = self.audit["pairwise_squared_distances"][name]
            self.assertEqual(len(records), 3)
            self.assertTrue(all(record["nonnegative"] for record in records))
            self.assertGreaterEqual(min(record["squared_distance"] for record in records), 0.0)
        triangle = self.audit["triangle_violation"]
        self.assertEqual(triangle["direct_pair"], [0, 1])
        self.assertEqual(triangle["via_index"], 2)
        self.assertAlmostEqual(triangle["violation"], 0.01510131906285106, places=15)
        self.assertFalse(triangle["triangle_inequality_satisfied"])
        self.assertEqual(maximum_triangle_violation(self.kernel), triangle)

    def test_shared_eq85_control_is_positive_definite(self) -> None:
        expected = np.asarray(
            [
                [1.0, 0.23635938028314127, 0.6712159348591652],
                [0.23635938028314127, 1.0, 0.6181972853314059],
                [0.6712159348591652, 0.6181972853314059, 1.0],
            ]
        )
        np.testing.assert_allclose(self.shared, expected, rtol=0.0, atol=1e-15)
        self.assertGreater(np.linalg.eigvalsh(self.shared)[0], 0.19)

    def test_assembly_rejects_invalid_shape_symmetry_domain_and_scale(self) -> None:
        with self.assertRaises(ValueError):
            assemble_warped_correlation(np.eye(2), [1.0])
        with self.assertRaises(ValueError):
            assemble_warped_correlation([[1.0, 0.2], [0.3, 1.0]], [1.0, 1.0])
        with self.assertRaises(ValueError):
            assemble_warped_correlation([[1.0, 1.1], [1.1, 1.0]], [1.0, 1.0])
        with self.assertRaises(ValueError):
            assemble_warped_correlation(np.eye(2), [1.0, 0.0])
        with self.assertRaises(ValueError):
            assemble_normalized_eq85([[1.0, math.nan], [math.nan, 1.0]])


class RepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = build_audit()

    def test_plain_clipping_is_psd_but_does_not_preserve_diagonal(self) -> None:
        for name in ("warped_Q", "normalized_eq85_K"):
            source = np.asarray(self.audit["matrices"][name])
            clipped = plain_eigenvalue_clipping(source)
            self.assertGreaterEqual(
                np.linalg.eigvalsh(clipped)[0],
                -TOLERANCES["positive_semidefinite_atol"],
            )
            self.assertGreater(np.max(np.abs(np.diag(clipped) - 1.0)), 0.02)
            self.assertAlmostEqual(
                np.linalg.norm(clipped - source, ord="fro"),
                abs(np.linalg.eigvalsh(source)[0]),
            )

    def test_renormalized_branch_is_psd_correlation(self) -> None:
        for name in ("warped_Q", "normalized_eq85_K"):
            source = np.asarray(self.audit["matrices"][name])
            repaired = psd_renormalized_correlation(source)
            np.testing.assert_array_equal(np.diag(repaired), np.ones(3))
            self.assertGreaterEqual(
                np.linalg.eigvalsh(repaired)[0],
                -TOLERANCES["positive_semidefinite_atol"],
            )
            self.assertTrue(all(item["nonnegative"] for item in pairwise_squared_distances(repaired)))

    def test_repair_records_compare_both_explicit_method_labels(self) -> None:
        records = self.audit["projection_comparison"]
        self.assertEqual(len(records), 4)
        self.assertEqual(
            [record["method"] for record in records],
            [
                "plain_eigenvalue_clipping",
                "psd_renormalized_correlation",
                "plain_eigenvalue_clipping",
                "psd_renormalized_correlation",
            ],
        )
        self.assertTrue(all(record["psd_passed"] for record in records))
        plain = [record for record in records if record["method"] == "plain_eigenvalue_clipping"]
        renormalized = [
            record for record in records if record["method"] == "psd_renormalized_correlation"
        ]
        self.assertTrue(all(not record["unit_diagonal_passed"] for record in plain))
        self.assertTrue(all(record["unit_diagonal_passed"] for record in renormalized))
        self.assertNotIn("nearest", json.dumps(METHOD_ORIGINS).lower())

    def test_repair_domains_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            plain_eigenvalue_clipping([[1.0, 0.0], [0.1, 1.0]])
        with self.assertRaises(ValueError):
            plain_eigenvalue_clipping([[1.0, math.inf], [math.inf, 1.0]])
        with self.assertRaises(AuditFailure):
            psd_renormalized_correlation([[-1.0, 0.0], [0.0, 1.0]])


class CheckContractTests(unittest.TestCase):
    def test_exactly_22_declared_checks_pass(self) -> None:
        audit = build_audit()
        self.assertEqual(len(CHECK_SPECS), 22)
        self.assertEqual(audit["check_count"], 22)
        self.assertEqual(audit["passing_check_count"], 22)
        enforce_checks(audit["checks"])

    def test_missing_changed_and_failing_checks_are_rejected(self) -> None:
        checks = copy.deepcopy(build_audit()["checks"])
        checks.pop("raw_minimum_eigenvalue")
        with self.assertRaises(AuditFailure):
            enforce_checks(checks)

        checks = copy.deepcopy(build_audit()["checks"])
        checks["raw_minimum_eigenvalue"]["minimum"] = 0.0
        with self.assertRaises(AuditFailure):
            enforce_checks(checks)

        checks = copy.deepcopy(build_audit()["checks"])
        checks["warped_minimum_eigenvalue"]["observed"] = 0.0
        checks["warped_minimum_eigenvalue"]["passed"] = False
        with self.assertRaises(AuditFailure):
            enforce_checks(checks)


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
        )
        cls.second_result = run_audit(
            result_dir=cls.second_dir,
            manifest_path=cls.second_manifest,
        )
        cls.summary = json.loads((cls.first_dir / "summary.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads(cls.first_manifest.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_context.cleanup()

    def test_summary_recomputation_validator_rejects_tampering(self) -> None:
        validate_summary(self.summary)
        changed = copy.deepcopy(self.summary)
        changed["triangle_violation"]["violation"] = 0.0
        with self.assertRaises(AuditFailure):
            validate_summary(changed)
        changed = copy.deepcopy(self.summary)
        changed["negative_directions"]["normalized_eq85_K"]["quadratic_form"] = 0.0
        with self.assertRaises(AuditFailure):
            validate_summary(changed)

    def test_all_result_and_manifest_bytes_are_directory_independent(self) -> None:
        for filename in ("summary.json", *CSV_CONTRACTS):
            with self.subTest(filename=filename):
                self.assertEqual(
                    (self.first_dir / filename).read_bytes(),
                    (self.second_dir / filename).read_bytes(),
                )
        self.assertEqual(self.first_manifest.read_bytes(), self.second_manifest.read_bytes())

    def test_manifest_exact_schema_hashes_and_scope(self) -> None:
        expected_keys = {
            "schema_version",
            "audit_id",
            "status",
            "generation_time_policy",
            "paper",
            "config",
            "config_sha256",
            "seeds",
            "code",
            "environment",
            "expected_metric_ranges",
            "observed_checks",
            "method_origins",
            "reproduction_scope",
            "source_hashes",
            "input_hashes",
            "outputs",
            "manifest_hash_scope",
            "canonical_invocation",
        }
        self.assertEqual(set(self.manifest), expected_keys)
        self.assertEqual(self.manifest["schema_version"], AUDIT_SCHEMA_VERSION)
        self.assertEqual(self.manifest["audit_id"], AUDIT_ID)
        self.assertEqual(self.manifest["status"], "passed")
        self.assertEqual(self.manifest["paper"]["pdf_sha256"], PAPER_SHA256)
        self.assertEqual(self.manifest["method_origins"], METHOD_ORIGINS)
        self.assertEqual(self.manifest["reproduction_scope"], REPRODUCTION_SCOPE)
        self.assertFalse(self.manifest["reproduction_scope"]["random_search_used"])
        self.assertFalse(self.manifest["reproduction_scope"]["random_failure_rate_reported"])
        self.assertEqual(len(self.manifest["observed_checks"]), 22)
        self.assertEqual(
            [record["path"] for record in self.manifest["source_hashes"]],
            list(SOURCE_RELATIVE_PATHS),
        )

        output_records = {Path(record["path"]).name: record for record in self.manifest["outputs"]}
        self.assertEqual(set(output_records), {"summary.json", *CSV_CONTRACTS})
        for filename, record in output_records.items():
            path = self.first_dir / filename
            self.assertEqual(record["sha256"], sha256_path(path))
            self.assertEqual(record["bytes"], path.stat().st_size)
        serialized = json.dumps(self.manifest, sort_keys=True)
        self.assertNotIn(str(ROOT), serialized)
        self.assertNotIn(str(self.temp_root), serialized)

    def test_csv_exact_fields_row_counts_and_order(self) -> None:
        expected_counts = {
            "witness_matrices.csv": 72,
            "spectra.csv": 24,
            "pair_checks.csv": 3,
            "negative_direction.csv": 6,
            "projection_comparison.csv": 4,
        }
        for filename, fields in CSV_CONTRACTS.items():
            with self.subTest(filename=filename):
                with (self.first_dir / filename).open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), fields)
                self.assertEqual(len(rows), expected_counts[filename])
        self.assertEqual(CSV_CONTRACTS["witness_matrices.csv"], MATRIX_FIELDS)
        self.assertEqual(CSV_CONTRACTS["spectra.csv"], SPECTRA_FIELDS)
        self.assertEqual(CSV_CONTRACTS["pair_checks.csv"], PAIR_FIELDS)
        self.assertEqual(CSV_CONTRACTS["negative_direction.csv"], NEGATIVE_DIRECTION_FIELDS)
        self.assertEqual(CSV_CONTRACTS["projection_comparison.csv"], PROJECTION_FIELDS)

        with (self.first_dir / "pair_checks.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            pair_rows = list(csv.DictReader(handle))
        self.assertEqual([row["pair_id"] for row in pair_rows], ["1-2", "1-3", "2-3"])
        self.assertTrue(all(row["passed"] == "True" for row in pair_rows))

    def test_result_sentinel_and_claim_registry(self) -> None:
        self.assertEqual(self.first_result["audit_id"], AUDIT_ID)
        self.assertEqual(self.first_result["status"], "passed")
        self.assertTrue(self.first_result["passed"])
        self.assertEqual(self.first_result["check_count"], 22)
        self.assertEqual(self.first_result["passing_check_count"], 22)
        self.assertEqual(tuple(self.summary["claim_ids"]), CLAIM_IDS)

    def test_pdf_modes_and_reserved_manifest_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            run_audit(result_dir=self.temp_root / "full-missing", full=True)
        wrong_pdf = self.temp_root / "wrong.pdf"
        wrong_pdf.write_bytes(b"not the pinned PDF")
        with self.assertRaises(ValueError):
            run_audit(result_dir=self.temp_root / "pdf-without-full", paper_pdf=wrong_pdf)
        with self.assertRaises(AuditFailure):
            run_audit(result_dir=self.temp_root / "wrong-full", full=True, paper_pdf=wrong_pdf)
        with self.assertRaises(ValueError):
            run_audit(
                result_dir=self.temp_root / "reserved",
                manifest_path=self.temp_root / "reserved" / "summary.json",
            )

    def test_stable_path_identities(self) -> None:
        paper_path = ROOT / "paper" / "2607.21366v1.pdf"
        self.assertEqual(_paper_input_identity(paper_path), "paper/2607.21366v1.pdf")
        frozen = ROOT / "artifacts" / "part_05" / "results"
        self.assertEqual(
            _manifest_output_path(frozen / "summary.json", frozen),
            "artifacts/part_05/results/summary.json",
        )

    def test_owned_sources_are_ascii_dash_safe(self) -> None:
        audit_dir = Path(__file__).resolve().parent
        for name in ("run.py", "test_run.py", "README.md"):
            payload = (audit_dir / name).read_bytes()
            self.assertNotIn(b"\xe2\x80\x93", payload)
            self.assertNotIn(b"\xe2\x80\x94", payload)


if __name__ == "__main__":
    unittest.main()
