#!/usr/bin/env python3
"""Unit and artifact-contract tests for the deterministic Part 6 audit."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from run import (
    AUDIT_ID,
    AUDIT_SCHEMA_VERSION,
    CHECK_SPECS,
    CLAIM_CLASSES,
    CLAIM_IDS,
    COUNTERMODEL_TERMINALS,
    CSV_CONTRACTS,
    OUTPUT_FILENAMES,
    P_VALUES,
    SCALE_FACTORS,
    TOLERANCES,
    AuditFailure,
    build_audit,
    countermodel_state,
    endpoint_proxy,
    lp_capacity,
    run_audit,
    scale_invariant_density,
    segment_integral,
    validate_summary,
)


ROOT = Path(__file__).resolve().parents[3]


class FormulaDomainTests(unittest.TestCase):
    def test_lp_capacity_rejects_invalid_inputs(self) -> None:
        for norms, p in (([], 1.0), ([1.0], 0.0), ([-1.0], 1.0), ([math.nan], 1.0)):
            with self.subTest(norms=norms, p=p):
                with self.assertRaises(ValueError):
                    lp_capacity(norms, p)

    def test_endpoint_proxy_rejects_invalid_inputs(self) -> None:
        cases = (
            (0, 1.0, 1.0, 1.0),
            (2, -1.0, 1.0, 1.0),
            (2, 1.0, 0.0, 1.0),
            (2, 1.0, 1.0, 0.0),
            (2, math.inf, 1.0, 1.0),
        )
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    endpoint_proxy(*args)

    def test_countermodel_domain_is_explicit(self) -> None:
        for capacity in (0.0, 1.0, -0.1, math.inf):
            with self.assertRaises(ValueError):
                countermodel_state(capacity)
        with self.assertRaises(ValueError):
            scale_invariant_density((1.0, 0.0))


class PartitionTests(unittest.TestCase):
    def test_three_four_five_vector_has_exact_base_norm(self) -> None:
        self.assertEqual(math.hypot(3.0, 4.0), 5.0)

    def test_partition_formula_matches_fixed_values(self) -> None:
        expected = {
            (0.5, 2): 10.0,
            (0.5, 3): 15.0,
            (0.5, 5): 25.0,
            (1.0, 2): 5.0,
            (1.0, 3): 5.0,
            (1.0, 5): 5.0,
            (2.0, 2): 5.0 / math.sqrt(2.0),
            (2.0, 3): 5.0 / math.sqrt(3.0),
            (2.0, 5): math.sqrt(5.0),
        }
        for (p, count), value in expected.items():
            observed = lp_capacity([5.0 / count] * count, p)
            self.assertAlmostEqual(observed, value, places=14)

    def test_only_p_one_is_nontrivially_invariant_in_fixed_family(self) -> None:
        rows = build_audit()["partition_sweep.csv"]
        for row in rows:
            if row["partition_count"] == 1:
                continue
            self.assertEqual(row["observed_invariant"], row["p"] == 1.0)
            self.assertTrue(row["passed"])


class CapacityAndConnectivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = build_audit()
        cls.paths = cls.audit["path_integrals.csv"]

    def test_constant_density_integral_is_parameterization_invariant(self) -> None:
        rows = [row for row in self.paths if row["section"] == "capacity_drain"]
        self.assertEqual([row["path_id"] for row in rows], ["linear", "geometric", "quadratic"])
        expected = 3.0 * math.log(2.5)
        self.assertAlmostEqual(expected, 2.748872195622465, places=15)
        for row in rows:
            self.assertAlmostEqual(row["exact_integral"], expected, places=15)
            self.assertLessEqual(row["absolute_error"], TOLERANCES["quadrature_atol"])

    def test_density_is_positive_and_scale_invariant_but_equals_E_only_on_path(self) -> None:
        for capacity in (0.75, 0.5, 0.125, 0.015625):
            state = countermodel_state(capacity)
            self.assertAlmostEqual(sum(state), capacity, places=15)
            density = scale_invariant_density(state)
            self.assertAlmostEqual(density, capacity, places=15)
            for scale in SCALE_FACTORS:
                scaled = (scale * state[0], scale * state[1])
                self.assertAlmostEqual(scale_invariant_density(scaled), density, places=15)
        self.assertNotEqual(scale_invariant_density((1.0, 1.0)), sum((1.0, 1.0)))

    def test_countermodel_integral_has_finite_extinction_limit(self) -> None:
        rows = [row for row in self.paths if row["section"] == "connectivity_countermodel"]
        self.assertEqual([row["end_capacity"] for row in rows], list(COUNTERMODEL_TERMINALS))
        for row in rows:
            expected = 0.75 - row["end_capacity"]
            self.assertAlmostEqual(row["exact_integral"], expected, places=15)
            self.assertLess(row["exact_integral"], 0.75)
            self.assertTrue(row["passed"])
        self.assertEqual(self.audit["summary"]["connectivity"]["countermodel_limit"], 0.75)


class GeometricPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = {
            row["path_id"]: row
            for row in build_audit()["path_integrals.csv"]
            if row["section"] == "geometric_projection"
        }

    def test_segment_formula_matches_closed_forms(self) -> None:
        length, exact, numeric = segment_integral((3.0, 1.0), (1.0, 1.0), 2.0)
        self.assertEqual(length, 2.0)
        self.assertAlmostEqual(exact, 2.0 * math.log(2.0), places=15)
        self.assertAlmostEqual(numeric, exact, places=14)

    def test_straight_path_is_below_its_endpoint_bound(self) -> None:
        row = self.rows["straight"]
        self.assertAlmostEqual(row["exact_integral"], 1.3862943611198906, places=15)
        self.assertEqual(row["endpoint_expression"], 2.0)
        self.assertTrue(row["endpoint_is_proved_bound"])
        self.assertLess(row["exact_integral"], row["endpoint_expression"])

    def test_detour_is_monotone_but_exceeds_same_endpoint_expression(self) -> None:
        row = self.rows["detour"]
        expected = 2.0 * (5.0 * math.log(8.0 / 7.0) + math.log(7.0 / 4.0))
        self.assertAlmostEqual(expected, 2.454545502116071, places=15)
        self.assertAlmostEqual(row["exact_integral"], expected, places=14)
        self.assertTrue(row["capacity_monotone"])
        self.assertFalse(row["endpoint_is_proved_bound"])
        self.assertGreater(row["exact_integral"], row["endpoint_expression"])
        self.assertLess(row["exact_integral"], row["arc_upper_bound"])


class ProxyAndRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = build_audit()
        cls.proxy = cls.audit["proxy_family.csv"]
        cls.ranking = cls.audit["ranking_fixture.csv"]

    def test_proxy_family_is_scale_invariant_on_fixed_grid(self) -> None:
        rows = [row for row in self.proxy if row["section"] == "scale_invariance"]
        self.assertEqual(len(rows), len(P_VALUES) * len(SCALE_FACTORS))
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(all(row["absolute_error"] == 0.0 for row in rows))

    def test_positive_p_family_grows_on_declared_extinction_fixture(self) -> None:
        for p in P_VALUES:
            rows = [row for row in self.proxy if row["section"] == "extinction" and row["p"] == p]
            values = [row["proxy_value"] for row in rows]
            self.assertEqual(values, sorted(values))
            self.assertTrue(all(row["monotone_from_previous"] for row in rows))

    def test_endpoint_proxy_is_not_additive_but_log_integral_is(self) -> None:
        nonadd = self.audit["summary"]["nonadditivity"]
        p1 = nonadd["p_1"]
        self.assertEqual(p1["direct"], 2.0)
        self.assertAlmostEqual(p1["via"], 5.0 / 3.0, places=15)
        self.assertAlmostEqual(abs(p1["defect"]), 1.0 / 3.0, places=15)
        control = nonadd["geometric_control"]
        self.assertLessEqual(control["error"], TOLERANCES["additivity_atol"])

    def test_cross_layer_orders_change_but_same_layer_orders_do_not(self) -> None:
        summary = self.audit["summary"]["ranking"]
        self.assertEqual(summary["cross_layer"]["p_0.5"], ["A", "C", "B"])
        self.assertEqual(summary["cross_layer"]["p_1"], ["C", "B", "A"])
        self.assertEqual(summary["cross_layer"]["p_2"], ["B", "C", "A"])
        same_orders = list(summary["same_layer"].values())
        self.assertTrue(all(order == ["S1", "S2", "S4"] for order in same_orders))

    def test_cross_layer_crossover_thresholds_are_exact(self) -> None:
        thresholds = self.audit["summary"]["ranking"]["cross_layer_crossover_p"]
        self.assertAlmostEqual(thresholds["A_vs_C"], 0.5849625007211562, places=15)
        self.assertAlmostEqual(thresholds["A_vs_B"], 0.9036774610288021, places=15)
        self.assertAlmostEqual(thresholds["B_vs_C"], 1.222392421336448, places=15)


class AuditContractTests(unittest.TestCase):
    def test_summary_registry_and_checks_are_exact(self) -> None:
        summary = build_audit()["summary"]
        self.assertEqual(summary["schema_version"], AUDIT_SCHEMA_VERSION)
        self.assertEqual(summary["audit_id"], AUDIT_ID)
        self.assertEqual(tuple(summary["claim_ids"]), CLAIM_IDS)
        self.assertEqual(set(summary["checks"]), set(CHECK_SPECS))
        self.assertEqual(summary["passing_check_count"], len(CHECK_SPECS))
        validate_summary(summary)

    def test_every_claim_has_one_stable_evidence_class(self) -> None:
        audit = build_audit()
        for filename in CSV_CONTRACTS:
            for row in audit[filename]:
                self.assertEqual(
                    row["evidence_label"],
                    CLAIM_CLASSES[row["claim_id"]],
                    msg=f"{filename}: {row['case_id'] if 'case_id' in row else row['step_id']}",
                )
        for method in audit["summary"]["method_origins"].values():
            self.assertEqual(method["evidence_label"], CLAIM_CLASSES[method["claim_id"]])

    def test_validator_rejects_failed_or_nonfinite_summary(self) -> None:
        summary = build_audit()["summary"]
        failed = copy.deepcopy(summary)
        failed["passed"] = False
        with self.assertRaises(AuditFailure):
            validate_summary(failed)
        nonfinite = copy.deepcopy(summary)
        nonfinite["checks"]["straight_integral_error"]["value"] = math.nan
        with self.assertRaises((AuditFailure, ValueError)):
            validate_summary(nonfinite)

    def test_six_outputs_are_byte_deterministic_and_schema_exact(self) -> None:
        with tempfile.TemporaryDirectory() as first_name, tempfile.TemporaryDirectory() as second_name:
            first = Path(first_name)
            second = Path(second_name)
            run_audit(result_dir=first)
            run_audit(result_dir=second)
            self.assertEqual({path.name for path in first.iterdir()}, set(OUTPUT_FILENAMES))
            for filename in OUTPUT_FILENAMES:
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            for filename, fields in CSV_CONTRACTS.items():
                with (first / filename).open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, strict=True)
                    rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), fields)
                self.assertGreater(len(rows), 0)

    def test_manifest_hashes_all_six_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            results = directory / "results"
            manifest_path = directory / "manifest.json"
            run_audit(result_dir=results, manifest_path=manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["audit_id"], AUDIT_ID)
            self.assertEqual(len(manifest["outputs"]), 6)
            by_name = {Path(row["path"]).name: row for row in manifest["outputs"]}
            for filename in OUTPUT_FILENAMES:
                payload = (results / filename).read_bytes()
                self.assertEqual(by_name[filename]["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(by_name[filename]["bytes"], len(payload))

    def test_manifest_cannot_alias_any_result_file(self) -> None:
        for filename in OUTPUT_FILENAMES:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as name:
                result_dir = Path(name) / "results"
                with self.assertRaisesRegex(AuditFailure, "must not alias"):
                    run_audit(
                        result_dir=result_dir,
                        manifest_path=result_dir / filename,
                    )
                self.assertFalse(result_dir.exists())

    def test_symlink_alias_and_existing_manifest_are_rejected_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result_dir = root / "results"
            result_dir.mkdir()
            alias = root / "alias.json"
            alias.symlink_to(result_dir / "summary.json")
            with self.assertRaises(AuditFailure):
                run_audit(result_dir=result_dir, manifest_path=alias)
            self.assertEqual(list(result_dir.iterdir()), [])

            existing = root / "existing.json"
            existing.write_text("keep\n", encoding="utf-8")
            with self.assertRaisesRegex(AuditFailure, "overwrite manifest"):
                run_audit(result_dir=result_dir, manifest_path=existing)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep\n")
            self.assertEqual(list(result_dir.iterdir()), [])

    def test_nonempty_result_directory_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            result_dir = Path(name) / "results"
            result_dir.mkdir()
            sentinel = result_dir / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            with self.assertRaisesRegex(AuditFailure, "nonempty"):
                run_audit(result_dir=result_dir)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
