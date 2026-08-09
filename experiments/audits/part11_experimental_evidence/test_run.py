#!/usr/bin/env python3
"""Unit tests for the deterministic Part 11 evidence audit."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = Path(__file__).with_name("run.py")


def load_audit():
    spec = importlib.util.spec_from_file_location("part11_experimental_evidence_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Part 11 audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Part11EvidenceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = load_audit()
        cls.bundle = cls.audit.build_audit()
        cls.summary = cls.bundle["summary"]

    def test_claim_registry_is_fixed(self) -> None:
        self.assertEqual(
            self.audit.CLAIM_IDS,
            (
                "P11-COMPRESSION-PLOT",
                "P11-COMPRESSION-EVIDENCE-BOUNDARY",
                "P11-DENSITY-BUDGET-MISMATCH",
                "P11-TABLE2-RECORD",
                "P11-HSCORE-AGGREGATION-GAP",
                "P11-HSCORE-AGGREGATION",
                "P11-UNCERTAINTY-UNDEFINED",
                "P11-CHECKPOINT-RULE-CONFLICT",
                "P11-SOURCE-DATA-SCOPE",
                "P11-ABSOLUTE-SOURCE-ACCURACY",
                "P11-CROSS-STATE-HSCORE",
                "P11-EVIDENCE-SCOPE",
                "P11-PROTOCOL-IDENTITY",
                "P10-SOURCE-STATE-RESTORATION",
                "EWC-TRUNCATED-GRID",
            ),
        )
        self.assertEqual(
            self.audit.CLAIM_CLASSES["P11-HSCORE-AGGREGATION"], "open_question"
        )
        self.assertEqual(
            self.audit.CLAIM_CLASSES["P11-EVIDENCE-SCOPE"], "independent_inference"
        )

    def test_every_registered_claim_has_an_output_row(self) -> None:
        represented = {
            row["claim_id"]
            for filename in self.audit.CSV_CONTRACTS
            for row in self.bundle[filename]
        }
        self.assertEqual(represented, set(self.audit.CLAIM_IDS))

    def test_output_and_check_contracts_are_fixed(self) -> None:
        self.assertEqual(len(self.audit.OUTPUT_FILENAMES), 11)
        self.assertEqual(len(self.audit.CHECK_SPECS), 36)
        self.assertEqual(self.summary["check_count"], 36)
        self.assertEqual(self.summary["passing_check_count"], 36)
        self.assertEqual(tuple(self.summary["checks"]), tuple(self.audit.CHECK_SPECS))
        self.assertEqual(
            tuple(self.audit.CSV_CONTRACTS),
            (
                "source_map.csv",
                "table2_recalculation.csv",
                "aggregation_fixture.csv",
                "uncertainty_interpretations.csv",
                "retention_fixture.csv",
                "evaluation_states.csv",
                "checkpoint_fixture.csv",
                "density_budget_fixture.csv",
                "compression_digitization.csv",
                "missing_experiments.csv",
            ),
        )

    def test_exact_table2_values_and_rounding_disjointness(self) -> None:
        rows = self.bundle["table2_recalculation.csv"]
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [row["method"] for row in rows],
            ["DEFT", "Head-Only", "Full FT", "EWC", "PEFT"],
        )
        for row in rows:
            signature = tuple(
                row[key]
                for key in (
                    "target_mean", "target_plus_minus", "source_mean",
                    "source_plus_minus", "reported_h", "reported_h_plus_minus",
                )
            )
            self.assertEqual(signature, self.audit.TABLE2_EXPECTED_SIGNATURES[row["method"]])
            self.assertFalse(row["rounding_intervals_overlap"])
            self.assertFalse(row["aggregation_reconstructible"])
            self.assertGreater(row["displayed_h_gap"], 0.0)
        self.assertAlmostEqual(rows[0]["harmonic_of_displayed_means"], 65.97126188966392)
        self.assertAlmostEqual(rows[1]["harmonic_of_displayed_means"], 45.94164248286981)

    def test_same_means_can_have_different_mean_task_h(self) -> None:
        fixture = self.summary["aggregation_fixture"]
        self.assertEqual(fixture["balanced"]["mean_target"], 50.0)
        self.assertEqual(fixture["crossed"]["mean_target"], 50.0)
        self.assertEqual(fixture["balanced"]["mean_source"], 50.0)
        self.assertEqual(fixture["crossed"]["mean_source"], 50.0)
        self.assertEqual(fixture["balanced"]["harmonic_of_means"], 50.0)
        self.assertEqual(fixture["crossed"]["harmonic_of_means"], 50.0)
        self.assertEqual(fixture["balanced"]["mean_of_task_h"], 50.0)
        self.assertEqual(fixture["crossed"]["mean_of_task_h"], 18.0)

    def test_uncertainty_rows_forbid_inference_without_inventing_semantics(self) -> None:
        rows = self.bundle["uncertainty_interpretations.csv"]
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertFalse(row["paper_defines_interpretation"])
            self.assertTrue(row["compatible_with_notation"])
            self.assertFalse(row["sampling_unit_known"])
            self.assertFalse(row["inferential_use_allowed"])
            self.assertFalse(row["reported_test"])
            self.assertFalse(row["formal_significance_supported"])

    def test_retention_fixture_keeps_hypothetical_values_separate(self) -> None:
        rows = self.bundle["retention_fixture.csv"]
        paper = rows[0]
        self.assertEqual(paper["row_kind"], "paper_record")
        self.assertEqual(paper["posttransfer_source_accuracy"], 52.14)
        self.assertEqual(paper["pretransfer_source_accuracy"], "not_reported")
        self.assertEqual(paper["normalized_retention_percent"], "not_computable")
        self.assertEqual(paper["claim_id"], "P11-ABSOLUTE-SOURCE-ACCURACY")
        hypothetical = [row for row in rows if row["row_kind"] == "independent_hypothetical_fixture"]
        self.assertEqual(len(hypothetical), 2)
        self.assertAlmostEqual(hypothetical[0]["normalized_retention_percent"], 86.9)
        self.assertAlmostEqual(hypothetical[1]["normalized_retention_percent"], 57.93333333333333)
        self.assertTrue(all(row["claim_id"] == "P11-EVIDENCE-SCOPE" for row in hypothetical))
        self.assertTrue(all(row["paper_reports_pretransfer_accuracy"] is False for row in rows))

    def test_evaluation_state_map_is_cross_state_and_non_numeric(self) -> None:
        rows = self.bundle["evaluation_states.csv"]
        restoration = {
            row["row_id"] for row in rows
            if row["row_id"] in {"source_head", "source_head_mask", "source_bn_buffers"}
            and row["state_differs"]
        }
        self.assertEqual(restoration, {"source_head", "source_head_mask", "source_bn_buffers"})
        h_row = next(row for row in rows if row["row_id"] == "h_score_operands")
        target_row = next(row for row in rows if row["row_id"] == "target_metric")
        self.assertFalse(target_row["state_differs"])
        self.assertFalse(target_row["paper_reports_transform"])
        self.assertEqual(h_row["cross_state_hscore_operand"], "both")
        self.assertTrue(h_row["state_differs"])
        self.assertTrue(all(row["paper_reports_live_source_metric"] is False for row in rows))

    def test_checkpoint_selectors_disagree_on_fixed_fixture(self) -> None:
        rows = self.bundle["checkpoint_fixture.csv"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            next(row["epoch"] for row in rows if row["selected_by_target_rule"]), 2
        )
        self.assertEqual(next(row["epoch"] for row in rows if row["selected_by_h_rule"]), 1)
        self.assertEqual(rows[0]["h_score"], 80.0)
        self.assertEqual(rows[1]["h_score"], 18.0)
        self.assertTrue(all(row["paper_result"] is False for row in rows))

    def test_equal_density_does_not_fix_parameter_payoff(self) -> None:
        rows = self.bundle["density_budget_fixture.csv"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["parameters_before"], 248)
        self.assertEqual(rows[0]["density_reduction"], rows[1]["density_reduction"])
        self.assertEqual(rows[0]["density_reduction"], 1.0 / 6.0)
        self.assertEqual(
            {row["action_id"]: row["parameter_payoff"] for row in rows},
            {"prune_hidden_1": 104, "prune_hidden_2": 12},
        )
        self.assertTrue(all(row["paper_result"] is False for row in rows))

    def test_digitization_is_provenance_bound_and_interval_safe(self) -> None:
        rows = self.bundle["compression_digitization.csv"]
        self.assertEqual(len(rows), 32)
        self.assertEqual(sum(row["vertical_trace"] for row in rows), 2)
        for row in rows:
            self.assertEqual(row["provenance_label"], "digitized_approximation")
            self.assertFalse(row["author_reported_numeric"])
            self.assertEqual(row["source_pdf_sha256"], self.audit.PAPER_SHA256)
            self.assertEqual(
                row["rgb_pixel_sha256"], self.audit.COMPRESSION_RASTER["rgb_pixel_sha256"]
            )
            self.assertLessEqual(row["accuracy_low"], row["accuracy_high"])
            if row["vertical_trace"]:
                self.assertEqual(row["value_kind"], "vertical_trace_interval")
                self.assertEqual(row["accuracy_estimate"], "not_single_valued")

    def test_missing_experiments_have_no_claimed_outcome(self) -> None:
        rows = self.bundle["missing_experiments.csv"]
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["paper_reported"] is False for row in rows))
        self.assertTrue(all(row["outcome_known"] is False for row in rows))
        self.assertTrue(all(row["status"] == "proposed_not_run" for row in rows))

    def test_generator_writes_exact_deterministic_output_set(self) -> None:
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first = Path(first_temp) / "results"
            second = Path(second_temp) / "results"
            self.audit.run_audit(result_dir=first)
            self.audit.run_audit(result_dir=second)
            self.assertEqual(
                {path.name for path in first.iterdir()}, set(self.audit.OUTPUT_FILENAMES)
            )
            for filename in self.audit.OUTPUT_FILENAMES:
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["check_count"], 36)
            for filename, fields in self.audit.CSV_CONTRACTS.items():
                with (first / filename).open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(rows, filename)
                self.assertEqual(tuple(rows[0]), tuple(fields), filename)
                self.assertTrue(all(row["passed"] == "True" for row in rows), filename)

    def test_generator_refuses_overwrite_and_wrong_paper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            result_dir = base / "results"
            self.audit.run_audit(result_dir=result_dir)
            with self.assertRaises(self.audit.AuditFailure):
                self.audit.run_audit(result_dir=result_dir)
            wrong = base / "wrong.pdf"
            wrong.write_bytes(b"wrong")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(AUDIT_PATH),
                    "--full",
                    "--paper-pdf",
                    str(wrong),
                    "--result-dir",
                    str(base / "wrong-results"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("wrong identity", completed.stderr)


if __name__ == "__main__":
    unittest.main()
