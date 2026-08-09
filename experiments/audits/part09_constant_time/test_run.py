#!/usr/bin/env python3
"""Unit tests for the deterministic Part 9 complexity and cache audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import run


class ConstantTimeAuditTests(unittest.TestCase):
    def test_exact_candidate_and_trajectory_counts(self) -> None:
        self.assertEqual(run.choose2(4096), 8_386_560)
        self.assertEqual(run.merge_trajectory_evaluations(4096), 11_453_245_440)
        self.assertEqual(
            run.merge_trajectory_evaluations(32),
            sum(run.choose2(width) for width in range(2, 33)),
        )
        self.assertEqual(
            run.total_trajectory_evaluations(32),
            run.merge_trajectory_evaluations(32) + run.prune_trajectory_evaluations(32),
        )

    def test_candidate_rows_are_fail_closed(self) -> None:
        rows = run._candidate_rows()
        self.assertEqual([row["width"] for row in rows], list(run.WIDTHS))
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertGreater(rows[-1]["trajectory_merge_evaluations"], 11_000_000_000)

    def test_operation_count_cache_reduces_geometry_work(self) -> None:
        rows = run._operation_rows()
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(all(row["wall_clock_claim"] is False for row in rows))
        self.assertGreater(rows[-1]["full_over_cache_work_factor"], 108.0)

    def test_cache_memory_count_excludes_implementation_overhead(self) -> None:
        row = next(row for row in run._memory_rows() if row["width"] == 4096)
        self.assertEqual(row["pair_entries"], 8_386_560)
        self.assertAlmostEqual(row["raw_scalar_mib"], 191.953125, places=12)
        self.assertFalse(row["includes_index_overhead"])

    def test_channel_surgery_changes_both_adjacent_layers(self) -> None:
        rows = run._dependency_rows()
        predecessor = [row for row in rows if row["location"] == "predecessor"]
        successor = [row for row in rows if row["location"] == "successor"]
        successor_pairs = [row for row in rows if row["location"] == "successor_pair"]
        self.assertEqual(len(predecessor), 3)
        self.assertEqual(len(successor), 3)
        self.assertEqual(len(successor_pairs), 3)
        self.assertTrue(all(row["absolute_change"] > 0.0 for row in rows))
        self.assertTrue(all(not row["paper_local_update_refreshes"] for row in rows))
        self.assertTrue(all(row["dependency_refreshes"] for row in rows))

    def test_stale_pair_ranking_differs_from_full_refresh(self) -> None:
        rows = run._refresh_rows()
        before = next(row["pair"] for row in rows if row["before_rank"] == 1)
        stale = next(row["pair"] for row in rows if row["after_paper_local_rank"] == 1)
        full = next(row["pair"] for row in rows if row["after_full_rank"] == 1)
        repaired = next(
            row["pair"] for row in rows if row["after_dependency_refresh_rank"] == 1
        )
        self.assertEqual(before, "1-2")
        self.assertEqual(stale, "1-2")
        self.assertEqual(full, "0-2")
        self.assertEqual(repaired, full)
        self.assertTrue(all(row["passed"] for row in rows))

    def test_nonuniform_static_payoff_reverses_order(self) -> None:
        rows = [
            row for row in run._payoff_rows()
            if row["fixture_id"] == "nonuniform_alpha_flip"
        ]
        static_first = next(row["candidate"] for row in rows if row["static_rank"] == 1)
        live_first = next(row["candidate"] for row in rows if row["live_rank"] == 1)
        self.assertEqual(static_first, "A")
        self.assertEqual(live_first, "B")
        self.assertNotEqual(
            rows[0]["static_over_live_alpha"], rows[1]["static_over_live_alpha"]
        )

    def test_payoff_controls_do_not_reverse(self) -> None:
        rows = run._payoff_rows()
        for fixture_id in ("uniform_alpha_control", "distortion_control"):
            selected = [row for row in rows if row["fixture_id"] == fixture_id]
            static = [
                row["candidate"] for row in sorted(selected, key=lambda row: row["static_rank"])
            ]
            live = [
                row["candidate"] for row in sorted(selected, key=lambda row: row["live_rank"])
            ]
            self.assertEqual(static, live)
            self.assertTrue(all(row["passed"] for row in selected))

    def test_summary_has_fixed_claim_check_and_row_registries(self) -> None:
        summary = run.build_audit(full=False)["summary"]
        run.validate_summary(summary)
        self.assertEqual(tuple(summary["claim_ids"]), run.CLAIM_IDS)
        self.assertEqual(summary["claim_classes"], run.CLAIM_CLASSES)
        self.assertEqual(summary["check_count"], 20)
        self.assertEqual(summary["passing_check_count"], 20)
        self.assertEqual(
            summary["row_counts"],
            {
                "source_map": 10,
                "candidate_counts": 10,
                "operation_counts": 10,
                "cache_memory": 10,
                "dependency_invalidation": 9,
                "refresh_comparison": 3,
                "payoff_rankings": 6,
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
