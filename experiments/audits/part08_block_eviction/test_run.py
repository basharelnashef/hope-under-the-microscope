#!/usr/bin/env python3
"""Unit tests for the deterministic Part 8 block-eviction audit."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import run


class BlockEvictionAuditTests(unittest.TestCase):
    def test_relu_second_moment_boundaries(self) -> None:
        self.assertAlmostEqual(run.relu_second_moment(0.0, 1.0), 0.5, places=14)
        self.assertEqual(run.relu_second_moment(-1.0, 0.0), 0.0)
        self.assertEqual(run.relu_second_moment(0.0, 0.0), 0.0)
        self.assertEqual(run.relu_second_moment(2.0, 0.0), 4.0)

    def test_zero_shift_factor_is_exact(self) -> None:
        row = next(row for row in run._shift_rows() if row["case_id"] == "alpha_0")
        self.assertAlmostEqual(row["paper_identity_rms"], 1.0, places=14)
        self.assertAlmostEqual(
            row["post_relu_identity_rms"], 1.0 / math.sqrt(2.0), places=14
        )
        self.assertAlmostEqual(row["rms_ratio"], math.sqrt(2.0), places=14)
        self.assertAlmostEqual(
            row["identity_overstatement_fraction"],
            1.0 - 1.0 / math.sqrt(2.0),
            places=14,
        )

    def test_negative_shift_ratio_grows_and_degenerate_boundary_is_zero(self) -> None:
        rows = {row["case_id"]: row for row in run._shift_rows()}
        ratios = [rows[f"alpha_{alpha:g}"]["rms_ratio"] for alpha in (-1, -2, -3, -4)]
        self.assertTrue(all(second > first for first, second in zip(ratios, ratios[1:])))
        self.assertGreater(ratios[-1], 2300.0)
        self.assertEqual(rows["degenerate_negative"]["paper_identity_rms"], 1.0)
        self.assertEqual(rows["degenerate_negative"]["post_relu_identity_rms"], 0.0)
        self.assertFalse(rows["degenerate_negative"]["ratio_defined"])

    def test_quadrature_and_seeded_monte_carlo_are_fail_closed(self) -> None:
        rows = run._moment_rows(run._standard_samples())
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertLessEqual(
            max(row["quadrature_abs_error"] for row in rows),
            run.TOLERANCES["quadrature_atol"],
        )
        self.assertLessEqual(
            max(row["monte_carlo_abs_error"] for row in rows),
            run.TOLERANCES["monte_carlo_atol"],
        )
        self.assertEqual({row["seed"] for row in rows}, {run.MONTE_CARLO_SEED})

    def test_equal_payoff_fixture_reverses_candidate_order(self) -> None:
        rows = [
            row for row in run._ordering_rows()
            if row["fixture_id"] == "negative_shift_flip"
        ]
        self.assertEqual(len(rows), 2)
        paper_first = next(row["candidate"] for row in rows if row["paper_rank"] == 1)
        post_first = next(row["candidate"] for row in rows if row["post_relu_rank"] == 1)
        self.assertEqual(paper_first, "A")
        self.assertEqual(post_first, "B")
        self.assertEqual({row["payoff"] for row in rows}, {1.0})
        self.assertTrue(all(row["passed"] for row in rows))

    def test_ordering_controls_do_not_reverse(self) -> None:
        rows = run._ordering_rows()
        for fixture_id in ("zero_shift_control", "positive_shift_control"):
            selected = [row for row in rows if row["fixture_id"] == fixture_id]
            paper = [
                row["candidate"] for row in sorted(selected, key=lambda row: row["paper_rank"])
            ]
            post = [
                row["candidate"] for row in sorted(selected, key=lambda row: row["post_relu_rank"])
            ]
            self.assertEqual(paper, post)
            self.assertTrue(all(row["passed"] for row in selected))

    def test_relu_idempotence_witness_keeps_function_and_scales_proxy(self) -> None:
        rows = run._depth_rows(run._standard_samples())
        self.assertEqual([row["depth"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(max(row["function_max_abs_difference"] for row in rows), 0.0)
        self.assertEqual(rows[-1]["paper_proxy_over_depth1"], 4.0)
        self.assertEqual(rows[-1]["post_relu_proxy_over_depth1"], 4.0)
        self.assertTrue(all(row["passed"] for row in rows))

    def test_conditional_gaussian_identity_fixture_is_explicitly_scoped(self) -> None:
        rows = run._graph_rows(run._standard_samples())
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["synthetic_identity_copy_max_error"] == 0.0 for row in rows))
        self.assertEqual({row["evidence_label"] for row in rows}, {"independent_inference"})
        self.assertEqual({row["claim_id"] for row in rows}, {"BLOCK-IDENTITY-MOMENT"})
        self.assertTrue(all(row["passed"] for row in rows))
        zero = next(row for row in rows if row["case_id"] == "zero_shift")
        self.assertEqual(zero["conditional_unrectified_second_moment"], 1.0)
        self.assertEqual(zero["conditional_post_relu_second_moment"], 0.5)

    def test_eligibility_rejects_projection_and_stride_shortcuts(self) -> None:
        rows = run._eligibility_rows()
        boundary_eligible = [
            row for row in rows if row["post_relu_boundary_audit_eligible"]
        ]
        projections = [row for row in rows if "projection" in row["shortcut"]]
        self.assertEqual(
            [row["case_id"] for row in boundary_eligible], ["v1_interior_identity"]
        )
        self.assertTrue(all(not row["gaussian_closed_form_established"] for row in rows))
        self.assertEqual(len(projections), 2)
        self.assertTrue(all(not row["eq95_identity_decomposition_eligible"] for row in projections))
        self.assertTrue(all(row["passed"] for row in rows))

    def test_summary_has_fixed_claim_check_and_row_registries(self) -> None:
        summary = run.build_audit(full=False)["summary"]
        run.validate_summary(summary)
        self.assertEqual(tuple(summary["claim_ids"]), run.CLAIM_IDS)
        self.assertEqual(summary["claim_classes"], run.CLAIM_CLASSES)
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["check_count"], 20)
        self.assertEqual(summary["passing_check_count"], 20)
        self.assertEqual(
            summary["row_counts"],
            {
                "source_map": 9,
                "moment_validation": 9,
                "shift_sweep": 14,
                "ordering_fixtures": 6,
                "depth_fixture": 4,
                "graph_boundary": 3,
                "eligibility": 5,
            },
        )

    def test_output_bytes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first = Path(first_temp) / "results"
            second = Path(second_temp) / "results"
            run.run_audit(result_dir=first)
            run.run_audit(result_dir=second)
            for filename in run.OUTPUT_FILENAMES:
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_nonempty_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result_dir = Path(temp) / "results"
            result_dir.mkdir()
            (result_dir / "sentinel.txt").write_text("keep\n", encoding="utf-8")
            with self.assertRaises(run.AuditFailure):
                run.run_audit(result_dir=result_dir)

    def test_full_mode_requires_authenticated_paper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            wrong_pdf = Path(temp) / "wrong.pdf"
            wrong_pdf.write_bytes(b"not the paper")
            with self.assertRaises(run.AuditFailure):
                run.run_audit(
                    result_dir=Path(temp) / "results",
                    paper_pdf=wrong_pdf,
                    full=True,
                )


if __name__ == "__main__":
    unittest.main()
