#!/usr/bin/env python3
"""Unit tests for the deterministic Part 7 merge-semantics audit."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run


class MergeSemanticsAuditTests(unittest.TestCase):
    def test_relu_kernel_boundaries(self) -> None:
        self.assertAlmostEqual(run.relu_self_kernel(0.0, 1.0), 0.5, places=14)
        self.assertEqual(run.relu_self_kernel(-1.0, 0.0), 0.0)
        self.assertEqual(run.relu_self_kernel(2.0, 0.0), 4.0)
        first = np.array([1.0, 0.0])
        second = np.array([0.0, 1.0])
        self.assertAlmostEqual(run.relu_arc_kernel(first, first), 0.5, places=14)
        self.assertAlmostEqual(
            run.relu_arc_kernel(first, second), 1.0 / (2.0 * math.pi), places=14
        )

    def test_exact_twins_have_zero_reported_and_nonzero_executed_distance(self) -> None:
        q = 1.25
        metrics = run.scale_metrics(2.0 * q * q, 2.0 * q, 4.0 * q * q, 50.0)
        self.assertAlmostEqual(metrics["paper_scale"], q, places=14)
        self.assertEqual(metrics["paper_coordinate_distance"], 0.0)
        self.assertAlmostEqual(metrics["paper_physical_distance"], q, places=14)
        self.assertAlmostEqual(metrics["normalized_scale"], 2.0 * q, places=14)
        self.assertAlmostEqual(metrics["unnormalized_scale"], 2.0 * q, places=14)
        self.assertEqual(metrics["normalized_physical_distance"], 0.0)
        self.assertEqual(metrics["unnormalized_physical_distance"], 0.0)
        self.assertEqual(
            {row["claim_id"] for row in run._twin_rows()},
            {"MERGE-TWIN-COUNTEREXAMPLE"},
        )

    def test_normalized_scale_is_stationary_for_physical_objective(self) -> None:
        geometry = run.pair_geometry(0.9, 0.8)
        remaining = 7.0
        metrics = run.scale_metrics(
            geometry["a"], geometry["b"], geometry["sum_norm_sq"], remaining
        )
        selected = metrics["normalized_scale"]

        def objective(scale: float) -> float:
            numerator = (
                scale * scale - 2.0 * geometry["b"] * scale
                + geometry["sum_norm_sq"]
            )
            return numerator / (remaining + scale) ** 2

        step = 1e-5
        derivative = (objective(selected + step) - objective(selected - step)) / (2.0 * step)
        self.assertLess(abs(derivative), 1e-10)
        self.assertLess(objective(selected), objective(0.9 * selected))
        self.assertLess(objective(selected), objective(1.1 * selected))

    def test_unnormalized_scale_is_fixed_direction_projection(self) -> None:
        geometry = run.pair_geometry(0.95, 0.8)
        selected = geometry["b"]

        def residual_squared(scale: float) -> float:
            return (
                scale * scale - 2.0 * geometry["b"] * scale
                + geometry["sum_norm_sq"]
            )

        self.assertLess(residual_squared(selected), residual_squared(selected - 0.1))
        self.assertLess(residual_squared(selected), residual_squared(selected + 0.1))

    def test_high_correlation_ratios_are_bounded_diagnostics(self) -> None:
        rows = {
            row["rho_input"]: row
            for row in run._correlation_rows() if row["cos_output"] == 1.0
        }
        self.assertTrue(4.49 <= rows[0.95]["true_to_reported_ratio"] <= 4.51)
        self.assertTrue(10.04 <= rows[0.99]["true_to_reported_ratio"] <= 10.06)
        self.assertTrue(31.68 <= rows[0.999]["true_to_reported_ratio"] <= 31.70)
        self.assertEqual(
            {row["claim_id"] for row in run._correlation_rows()},
            {"P07-CORRELATION-GAP"},
        )
        repairs = {
            row["fixture"]: row
            for row in run._repair_rows()
            if row["method"] == "unnormalized_sum_repair"
            and row["fixture"].startswith("aligned")
        }
        self.assertTrue(
            17.0 <= repairs["aligned_rho_095"]["physical_improvement_over_paper"] <= 17.1
        )
        self.assertTrue(
            57.5 <= repairs["aligned_rho_099"]["physical_improvement_over_paper"] <= 57.6
        )
        self.assertTrue(
            324.0
            <= repairs["aligned_rho_0999"]["physical_improvement_over_paper"]
            <= 325.0
        )

    def test_near_cancellation_stays_inside_nonzero_parent_domain(self) -> None:
        geometry = run.pair_geometry(1.0, -1.0, output_scale_j=0.99)
        self.assertGreater(geometry["b"], 0.0)
        self.assertLess(geometry["sum_norm"], 0.01)
        remaining = 32.0 * (geometry["capacity_i"] + geometry["capacity_j"])
        metrics = run.scale_metrics(
            geometry["a"], geometry["b"], geometry["sum_norm_sq"], remaining
        )
        self.assertGreater(
            metrics["paper_coordinate_distance"] / geometry["sum_norm"], 140.0
        )
        self.assertEqual(metrics["normalized_physical_distance"], 0.0)
        self.assertEqual(metrics["unnormalized_physical_distance"], 0.0)
        repair_rows = [
            row for row in run._repair_rows() if row["fixture"] == "near_cancellation"
        ]
        for row in repair_rows:
            if row["method"] != "paper_literal":
                self.assertIsNone(row["physical_improvement_over_paper"])
                self.assertIsNone(row["objective_improvement_over_paper"])

    def test_physical_forward_rows_are_fail_closed(self) -> None:
        rows = run._forward_rows(run.build_config(full=False))
        self.assertEqual(len(rows), 6)
        literal = [row for row in rows if row["method"] == "paper_literal"]
        repairs = [row for row in rows if row["method"] != "paper_literal"]
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(
            all(math.isclose(row["forward_relative_error"], 0.5) for row in literal)
        )
        self.assertTrue(
            all(row["forward_relative_error"] <= run.TOLERANCES["relative_atol"] for row in repairs)
        )

    def test_summary_has_fixed_claim_and_check_registries(self) -> None:
        audit = run.build_audit(full=False)
        summary = audit["summary"]
        run.validate_summary(summary)
        self.assertEqual(tuple(summary["claim_ids"]), run.CLAIM_IDS)
        self.assertEqual(summary["claim_classes"], run.CLAIM_CLASSES)
        self.assertEqual(summary["check_count"], 24)
        self.assertEqual(summary["passing_check_count"], 24)
        self.assertEqual(
            summary["row_counts"],
            {
                "source_map": 10,
                "twin_cases": 8,
                "correlation_sweep": 10,
                "capacity_sweep": 8,
                "repair_comparison": 15,
                "physical_forward": 6,
            },
        )

    def test_output_schemas_and_bytes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_name, tempfile.TemporaryDirectory() as second_name:
            first = Path(first_name)
            second = Path(second_name)
            run.run_audit(result_dir=first)
            run.run_audit(result_dir=second)
            self.assertEqual(
                {path.name for path in first.iterdir()}, set(run.OUTPUT_FILENAMES)
            )
            for filename in run.OUTPUT_FILENAMES:
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            for filename, fields in run.CSV_CONTRACTS.items():
                with (first / filename).open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, strict=True)
                    rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), fields)
                self.assertTrue(rows)
                self.assertTrue(all(row["passed"] == "True" for row in rows))

    def test_runner_refuses_to_overwrite_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            run.run_audit(result_dir=directory)
            with self.assertRaises(run.AuditFailure):
                run.run_audit(result_dir=directory)

    def test_invalid_domains_fail(self) -> None:
        with self.assertRaises(ValueError):
            run.pair_geometry(1.1, 0.0)
        with self.assertRaises(ValueError):
            run.scale_metrics(1.0, 0.0, 1.0, 1.0)
        with self.assertRaises(run.AuditFailure):
            run._sqrt_nonnegative(-1.0, "negative fixture")


if __name__ == "__main__":
    unittest.main()
