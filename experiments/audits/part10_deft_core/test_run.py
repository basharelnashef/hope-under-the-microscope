#!/usr/bin/env python3
"""Focused tests for the deterministic Part 10 DEFT core-state audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run


class DeftCoreAuditTests(unittest.TestCase):
    def test_summary_has_fixed_claim_check_and_output_registries(self) -> None:
        audit = run.build_audit(full=False)
        summary = audit["summary"]
        run.validate_summary(summary)
        self.assertEqual(tuple(summary["claim_ids"]), run.CLAIM_IDS)
        self.assertEqual(summary["claim_classes"], run.CLAIM_CLASSES)
        self.assertEqual(summary["check_count"], 26)
        self.assertEqual(summary["passing_check_count"], 26)
        self.assertEqual(
            tuple(summary["checks"]),
            (
                "nonfinite_count", "row_failure_count",
                "evidence_class_mismatch_count", "trajectory_threshold",
                "global_capacity_threshold", "trajectory_unassigned_count",
                "trajectory_vs_layer_disagreement_count",
                "trajectory_vs_global_disagreement_count",
                "layer_vs_global_disagreement_count", "partition_signature_count",
                "masked_slack_to_core_weight", "core_dynamic_drift",
                "initialization_shock", "core_to_slack_weight_change",
                "operator_drift_norm", "relative_capacity_change",
                "trainable_parameter_change_norm", "final_running_mean",
                "final_running_variance", "live_source_rmse",
                "zero_slot_gradient_norm", "repair_initial_output",
                "repair_first_outgoing_gradient_magnitude",
                "repair_second_internal_gradient_min", "restored_source_rmse",
                "frozen_source_rmse",
            ),
        )
        self.assertEqual(len(run.OUTPUT_FILENAMES), 9)
        self.assertEqual(
            summary["row_counts"],
            {
                "source_map": 14,
                "partition_assignments": 18,
                "partition_disagreements": 3,
                "routing_transitions": 4,
                "bn_buffer_drift": 4,
                "operator_drift": 1,
                "released_slot_gradients": 4,
                "source_state_controls": 5,
            },
        )
        self.assertEqual(
            run.CLAIM_CLASSES["P10-BN-STATE-REPAIR"], "proposed_repair"
        )

    def test_global_thresholds_are_derived_from_fixture_values(self) -> None:
        self.assertEqual(
            run._trajectory_costs_by_neuron(),
            {
                "A0": 0.3,
                "A1": 0.5,
                "A2": None,
                "B0": 1.0,
                "B1": 1.6,
                "B2": None,
            },
        )
        trajectory, capacity = run._derived_global_thresholds()
        self.assertEqual(trajectory, 0.75)
        self.assertEqual(capacity, 3.5)
        rows = {
            row["neuron"]: row
            for row in run.build_audit()["partition_assignments.csv"]
            if row["rule_id"] == "eq24_27_trajectory_cost"
        }
        self.assertEqual(
            (rows["A0"]["trajectory_n"], rows["A0"]["capacity"],
             rows["A0"]["trajectory_e_b"], rows["A0"]["trajectory_cost"]),
            (3.0, 1.0, 10.0, 0.3),
        )
        self.assertEqual(
            (rows["A1"]["trajectory_n"], rows["A1"]["capacity"],
             rows["A1"]["trajectory_e_b"], rows["A1"]["trajectory_cost"]),
            (2.0, 2.0, 8.0, 0.5),
        )
        self.assertEqual(
            (rows["B0"]["trajectory_n"], rows["B0"]["capacity"],
             rows["B0"]["trajectory_e_b"], rows["B0"]["trajectory_cost"]),
            (3.0, 3.0, 9.0, 1.0),
        )
        self.assertEqual(
            (rows["B1"]["trajectory_n"], rows["B1"]["capacity"],
             rows["B1"]["trajectory_e_b"], rows["B1"]["trajectory_cost"]),
            (2.0, 4.0, 5.0, 1.6),
        )
        self.assertEqual(rows["A2"]["trajectory_n"], "not_recorded")
        self.assertEqual(rows["B2"]["trajectory_e_b"], "not_recorded")

    def test_three_partition_rules_have_declared_assignments(self) -> None:
        summary = run.build_audit()["summary"]
        assignments = summary["partition_fixture"]["assignments"]
        self.assertEqual(
            assignments["eq24_27_trajectory_cost"],
            {
                "A0": "slack",
                "A1": "slack",
                "A2": "unassigned",
                "B0": "core",
                "B1": "core",
                "B2": "unassigned",
            },
        )
        self.assertEqual(
            assignments["definition_h1_layer_capacity"],
            {
                "A0": "slack",
                "A1": "slack",
                "A2": "core",
                "B0": "slack",
                "B1": "slack",
                "B2": "core",
            },
        )
        self.assertEqual(
            assignments["algorithm3_global_capacity"],
            {
                "A0": "slack",
                "A1": "slack",
                "A2": "core",
                "B0": "slack",
                "B1": "core",
                "B2": "core",
            },
        )

    def test_partition_disagreements_use_the_common_assigned_domain(self) -> None:
        audit = run.build_audit()
        rows = audit["partition_disagreements.csv"]
        observed = {
            (row["rule_a"], row["rule_b"]): (
                row["disagreement_count"], row["disagreement_neurons"]
            )
            for row in rows
        }
        self.assertEqual(
            observed[
                ("eq24_27_trajectory_cost", "definition_h1_layer_capacity")
            ],
            (2, "B0|B1"),
        )
        self.assertEqual(
            observed[
                ("eq24_27_trajectory_cost", "algorithm3_global_capacity")
            ],
            (1, "B0"),
        )
        self.assertEqual(
            observed[
                ("definition_h1_layer_capacity", "algorithm3_global_capacity")
            ],
            (1, "B1"),
        )

    def test_destination_gate_preserves_only_the_narrow_routing_barrier(self) -> None:
        audit = run.build_audit()
        summary = audit["summary"]
        self.assertEqual(
            summary["routing_fixture"]["updated_weights"],
            [[2.0, 0.0], [6.0, 8.0]],
        )
        self.assertEqual(summary["routing_fixture"]["slack_to_core_weight_after"], 0.0)
        self.assertEqual(summary["routing_fixture"]["core_to_slack_weight_after"], 6.0)
        self.assertEqual(summary["routing_fixture"]["initialization_shock"], 3.0)
        self.assertEqual(summary["routing_fixture"]["core_dynamic_drift"], 0.0)
        operator = audit["operator_drift.csv"][0]
        self.assertEqual(operator["outgoing_vector_before"], "2|5")
        self.assertEqual(operator["outgoing_vector_after"], "2|6")
        self.assertEqual(operator["outgoing_to_core_after"], 2.0)
        self.assertEqual(operator["outgoing_to_slack_after"], 6.0)
        self.assertEqual(operator["operator_drift_norm"], 1.0)
        self.assertAlmostEqual(
            operator["relative_capacity_change"], 0.17444043902940712, places=15
        )

    def test_batchnorm_buffers_drift_with_fixed_trainable_parameters(self) -> None:
        audit = run.build_audit()
        summary = audit["summary"]
        self.assertEqual(summary["bn_fixture"]["final_running_mean"], 4.0)
        self.assertEqual(summary["bn_fixture"]["final_running_variance"], 5.75)
        self.assertEqual(summary["bn_fixture"]["source_probes"], [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(
            summary["bn_fixture"]["live_source_rmse"],
            1.7346960857596274,
            places=15,
        )
        rows = audit["bn_buffer_drift.csv"]
        live = [row for row in rows if row["mode"] == "live_training_buffers"]
        frozen = [row for row in rows if row["mode"] == "frozen_buffer_control"]
        self.assertTrue(all(row["buffers_updated"] for row in live))
        self.assertTrue(all(row["source_probe_rmse_after"] > 0.0 for row in live))
        self.assertTrue(all(not row["buffers_updated"] for row in frozen))
        self.assertTrue(
            all(row["claim_id"] == "P10-BN-STATE-REPAIR" for row in frozen)
        )
        self.assertTrue(all(row["evidence_label"] == "proposed_repair" for row in frozen))
        self.assertTrue(
            all(row["trainable_parameter_change_norm"] == 0.0 for row in rows)
        )

    def test_released_slot_control_separates_lifecycle_and_gradient_failures(self) -> None:
        rows = {
            row["variant"]: row
            for row in run.build_audit()["released_slot_gradients.csv"]
        }
        purged = rows["algorithm2_purged_child"]
        self.assertEqual(purged["active_layer_status"], "removed_or_purged")
        self.assertEqual(purged["physical_storage_status"], "unspecified_by_algorithm")
        self.assertFalse(purged["reentry_transition_specified"])
        self.assertEqual(purged["incoming_u"], "unspecified")
        self.assertEqual(purged["first_gradient_v"], "not_evaluated")
        released = rows["axiom3_released_child"]
        self.assertEqual(released["physical_storage_status"], "unspecified_by_axiom")
        self.assertFalse(released["reentry_transition_specified"])
        self.assertEqual(released["initial_output"], "not_evaluated")
        dead = rows["all_zero_retained_slot"]
        self.assertEqual(dead["first_gradient_u"], 0.0)
        self.assertEqual(dead["first_gradient_v"], 0.0)
        repair = rows["proposed_fixed_width_reinitialized_slot"]
        self.assertEqual(repair["initial_output"], 0.0)
        self.assertEqual(repair["first_gradient_u"], 0.0)
        self.assertEqual(repair["first_gradient_gamma"], 0.0)
        self.assertEqual(repair["first_gradient_beta"], 0.0)
        self.assertEqual(repair["first_gradient_v"], -1.0)
        self.assertEqual(repair["outgoing_v_after_first_step"], 0.25)
        self.assertEqual(repair["second_output"], 0.25)
        self.assertEqual(repair["second_residual"], -0.75)
        self.assertEqual(repair["second_gradient_u"], -0.1875)
        self.assertEqual(repair["second_gradient_gamma"], -0.1875)
        self.assertEqual(repair["second_gradient_beta"], -0.1875)
        self.assertEqual(repair["second_internal_gradient_min"], 0.1875)
        self.assertTrue(repair["reusable"])

    def test_source_state_controls_restore_or_freeze_batchnorm_buffers(self) -> None:
        rows = {
            row["state_id"]: row
            for row in run.build_audit()["source_state_controls.csv"]
        }
        self.assertAlmostEqual(
            rows["live_target_state"]["source_rmse"],
            1.7346960857596274,
            places=15,
        )
        self.assertEqual(rows["trainable_freeze_only"]["source_rmse"], rows["live_target_state"]["source_rmse"])
        self.assertEqual(rows["restored_source_state"]["source_rmse"], 0.0)
        self.assertEqual(rows["frozen_buffer_control"]["source_rmse"], 0.0)
        self.assertEqual(
            rows["frozen_buffer_control"]["claim_id"], "P10-BN-STATE-REPAIR"
        )
        self.assertEqual(
            rows["restored_source_state"]["claim_id"],
            "P10-SOURCE-STATE-RESTORATION",
        )

    def test_output_bytes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first = Path(first_temp) / "results"
            second = Path(second_temp) / "results"
            run.run_audit(result_dir=first)
            run.run_audit(result_dir=second)
            self.assertEqual({path.name for path in first.iterdir()}, set(run.OUTPUT_FILENAMES))
            for filename in run.OUTPUT_FILENAMES:
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            persisted = json.loads((first / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(tuple(persisted["checks"]), tuple(run.CHECK_SPECS))

    def test_manifest_records_fixed_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result_dir = Path(temp) / "results"
            manifest_path = Path(temp) / "manifest.json"
            run.run_audit(result_dir=result_dir, manifest_path=manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["audit_id"], run.AUDIT_ID)
            self.assertEqual(manifest["expected_metric_ranges"], run.CHECK_SPECS)
            self.assertEqual(manifest["reproduction_scope"], run.REPRODUCTION_SCOPE)
            self.assertEqual(
                [record["path"] for record in manifest["source_hashes"]],
                list(run.SOURCE_RELATIVE_PATHS),
            )
            self.assertEqual(len(manifest["outputs"]), 9)

    def test_nonempty_output_directory_and_existing_manifest_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result_dir = Path(temp) / "results"
            result_dir.mkdir()
            (result_dir / "sentinel.txt").write_text("keep\n", encoding="utf-8")
            with self.assertRaises(run.AuditFailure):
                run.run_audit(result_dir=result_dir)

            empty_result = Path(temp) / "empty-results"
            manifest = Path(temp) / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(run.AuditFailure):
                run.run_audit(result_dir=empty_result, manifest_path=manifest)

    def test_full_mode_requires_authenticated_paper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(run.AuditFailure):
                run.run_audit(result_dir=Path(temp) / "missing", full=True)
            wrong_pdf = Path(temp) / "wrong.pdf"
            wrong_pdf.write_bytes(b"not the paper")
            with self.assertRaises(run.AuditFailure):
                run.run_audit(
                    result_dir=Path(temp) / "wrong-results",
                    paper_pdf=wrong_pdf,
                    full=True,
                )


if __name__ == "__main__":
    unittest.main()
