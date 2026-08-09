"""Focused tests for the deterministic Part 12 corrected HOPE audit."""

from __future__ import annotations

import copy
import csv
from decimal import Decimal, InvalidOperation
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
RUN_PATH = Path(__file__).with_name("run.py")


def load_audit():
    spec = importlib.util.spec_from_file_location("part12_corrected_hope_tested", RUN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Part 12 audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CorrectedHopeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = load_audit()
        cls.bundle = cls.audit.build_audit(full=False)
        cls.summary = cls.bundle["summary"]

    def test_closed_output_schema_claim_and_check_registries(self) -> None:
        expected_outputs = (
            "summary.json",
            "design_contract.csv",
            "common_gram_cases.csv",
            "merge_cases.csv",
            "invalidation_cases.csv",
            "resource_accounting.csv",
            "block_eligibility.csv",
            "deft_transitions.csv",
            "integration_ablations.csv",
            "source_map.csv",
        )
        expected_rows = {
            "design_contract": 12,
            "common_gram_cases": 10,
            "merge_cases": 3,
            "invalidation_cases": 21,
            "resource_accounting": 16,
            "block_eligibility": 8,
            "deft_transitions": 17,
            "integration_ablations": 5,
            "source_map": 12,
        }
        self.assertEqual(self.audit.OUTPUT_FILENAMES, expected_outputs)
        self.assertEqual(("summary.json", *self.audit.CSV_CONTRACTS), expected_outputs)
        self.assertEqual(self.summary["row_counts"], expected_rows)
        self.assertEqual(tuple(self.summary["checks"]), tuple(self.audit.CHECK_SPECS))
        self.assertEqual(self.summary["schema_version"], 2)
        self.assertEqual(self.summary["check_count"], 26)
        self.assertEqual(self.summary["passing_check_count"], 26)
        self.assertEqual(tuple(self.audit.CLAIM_CLASSES), self.audit.CLAIM_IDS)

    def test_every_row_is_passing_finite_and_claim_scoped(self) -> None:
        represented: set[str] = set()
        for filename, fields in self.audit.CSV_CONTRACTS.items():
            rows = self.bundle[filename]
            self.assertTrue(rows, filename)
            for row_index, row in enumerate(rows):
                self.assertEqual(set(row), set(fields), filename)
                self.assertIs(row["passed"], True, f"{filename}[{row_index}]")
                claim_id = row["claim_id"]
                represented.add(claim_id)
                self.assertEqual(
                    row["evidence_label"], self.audit.CLAIM_CLASSES[claim_id]
                )
                for field, value in row.items():
                    if isinstance(value, bool) or value is None:
                        continue
                    try:
                        number = Decimal(str(value))
                    except (InvalidOperation, ValueError):
                        continue
                    self.assertTrue(
                        number.is_finite(), f"{filename}[{row_index}].{field}"
                    )
        self.assertEqual(represented, set(self.audit.CLAIM_IDS))

    def test_fixed_geometry_and_merge_witnesses_are_measured(self) -> None:
        checks = self.summary["checks"]
        self.assertAlmostEqual(
            checks["common_gram_minimum_eigenvalue"]["value"],
            0.08168052937955,
            places=15,
        )
        self.assertEqual(checks["common_gram_symmetry_error"]["value"], 0.0)
        self.assertEqual(checks["bn_fold_max_error"]["value"], 0.0)
        self.assertEqual(checks["kernel_zero_scale_error"]["value"], 0.0)

        merges = {row["case_id"]: row for row in self.bundle["merge_cases.csv"]}
        self.assertEqual(merges["zero_bias_twins"]["absolute_residual"], 0.0)
        self.assertLess(merges["biased_twins"]["absolute_residual"], 4e-16)
        self.assertAlmostEqual(
            merges["noncollinear_pair"]["absolute_residual"],
            0.719667789793448,
            places=15,
        )
        self.assertEqual(merges["noncollinear_pair"]["candidate_count"], 128)
        self.assertEqual(merges["noncollinear_pair"]["span_rank"], 2)

    def test_graph_resource_block_and_deft_witnesses_are_exact(self) -> None:
        checks = self.summary["checks"]
        expected = {
            "dependency_key_count": 25.0,
            "complete_refresh_value_mismatch_count": 0.0,
            "local_only_stale_value_count": 20.0,
            "oracle_missing_key_count": 0.0,
            "oracle_extra_key_count": 0.0,
            "independent_cache_universe_count": 37.0,
            "live_parameter_payoff": 10.0,
            "initial_static_parameter_payoff": 22.0,
            "current_candidate_count": 13.0,
            "trajectory_candidate_evaluations": 24.0,
            "eligible_block_count": 1.0,
            "block_rejection_case_count": 6.0,
            "fixed_width_delta": 0.0,
            "stable_slot_identity_count": 1.0,
            "maximum_masked_parameter_drift": 0.0,
            "dual_track_source_buffer_drift": 0.0,
            "integration_ablation_pass_count": 5.0,
        }
        for name, value in expected.items():
            self.assertEqual(checks[name]["value"], value, name)

        block = {
            row["case_id"]: row for row in self.bundle["block_eligibility.csv"]
        }
        self.assertEqual(block["eligible_identity"]["direct_cost"], 2.3125)
        self.assertEqual(block["eligible_weighted_cost"]["direct_cost"], 3.15625)
        self.assertTrue(
            all(
                row["eligible"] == 0 and row["cost_evaluated"] == 0
                for row in self.bundle["block_eligibility.csv"]
                if row["case_id"] not in {"eligible_identity", "eligible_weighted_cost"}
            )
        )
        core_release = next(
            row
            for row in self.bundle["deft_transitions.csv"]
            if row["case_id"] == "core_release_rejected"
        )
        self.assertEqual(core_release["value"], 1.0)
        self.assertEqual(core_release["state_before"], "core")
        self.assertEqual(core_release["state_after"], "core")
        self.assertEqual(core_release["claim_id"], "P12-CORE-STATE-INVARIANCE")
        self.assertEqual(
            self.summary["headline_metrics"][
                "complete_refresh_value_mismatch_count"
            ],
            0,
        )
        self.assertEqual(
            self.summary["headline_metrics"]["local_only_stale_value_count"],
            20,
        )

    def test_summary_config_hash_scope_and_validation_fail_closed(self) -> None:
        self.assertEqual(self.summary["config"], self.audit.build_config(full=False))
        self.assertEqual(
            self.summary["config_sha256"],
            self.audit._config_hash(self.summary["config"]),
        )
        self.assertEqual(self.summary["method_origins"], self.audit.METHOD_ORIGINS)
        self.assertEqual(
            self.summary["reproduction_scope"], self.audit.REPRODUCTION_SCOPE
        )
        self.assertEqual(
            self.summary["paper"]["pdf_sha256"],
            "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b",
        )
        self.audit.validate_summary(self.summary)
        changed = copy.deepcopy(self.summary)
        changed["checks"]["dependency_key_count"]["value"] = 24.0
        with self.assertRaises(self.audit.AuditFailure):
            self.audit.validate_summary(changed)

    def test_generator_is_byte_deterministic_and_writes_exactly_ten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            self.audit.run_audit(result_dir=first, full=False)
            self.audit.run_audit(result_dir=second, full=False)
            self.assertEqual(
                {path.name for path in first.iterdir()},
                set(self.audit.OUTPUT_FILENAMES),
            )
            for filename in self.audit.OUTPUT_FILENAMES:
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                    filename,
                )
            written_summary = json.loads(
                (first / "summary.json").read_text(encoding="utf-8")
            )
            self.audit.validate_summary(written_summary)
            for filename, fields in self.audit.CSV_CONTRACTS.items():
                with (first / filename).open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), fields)
                self.assertTrue(all(row["passed"] == "True" for row in rows))

    def test_generator_refuses_nonempty_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "results"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("user data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.audit.run_audit(result_dir=destination, full=False)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user data")

    def test_optional_external_manifest_records_complete_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result_dir = Path(temp) / "results"
            manifest_path = Path(temp) / "manifest.json"
            self.audit.run_audit(
                result_dir=result_dir,
                manifest_path=manifest_path,
                full=False,
            )
            self.assertEqual(
                {path.name for path in result_dir.iterdir()},
                set(self.audit.OUTPUT_FILENAMES),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["audit_id"], self.audit.AUDIT_ID)
            self.assertEqual(manifest["config"], self.audit.build_config(full=False))
            self.assertEqual(
                manifest["expected_metric_ranges"], self.audit.CHECK_SPECS
            )
            self.assertEqual(manifest["observed_checks"], self.summary["checks"])
            self.assertEqual(manifest["method_origins"], self.audit.METHOD_ORIGINS)
            self.assertEqual(
                manifest["reproduction_scope"], self.audit.REPRODUCTION_SCOPE
            )
            self.assertEqual(manifest["seeds"], {"used": False, "values": []})
            self.assertEqual(
                [record["path"] for record in manifest["source_hashes"]],
                list(self.audit.SOURCE_RELATIVE_PATHS),
            )
            self.assertEqual(
                [record["path"] for record in manifest["outputs"]],
                list(self.audit.OUTPUT_FILENAMES),
            )
            self.assertEqual(
                manifest["input_hashes"][0]["path"], "paper/manifest.json"
            )
            self.assertEqual(
                manifest["canonical_invocation"],
                {
                    "entrypoint": "experiments/audits/part12_corrected_hope/run.py",
                    "result_dir": "<RESULT_DIR>",
                    "manifest_path": "<MANIFEST_PATH>",
                    "paper_pdf": "not supplied",
                    "full": False,
                },
            )

    def test_repository_relative_outputs_keep_repository_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".part12-path-", dir=ROOT) as temp:
            root = Path(temp)
            result_dir = (root / "results").relative_to(ROOT)
            manifest_path = (root / "manifest.json").relative_to(ROOT)
            self.audit.run_audit(
                result_dir=result_dir,
                manifest_path=manifest_path,
                full=False,
            )
            manifest = json.loads(
                (ROOT / manifest_path).read_text(encoding="utf-8")
            )
            expected = [
                (ROOT / result_dir / filename).relative_to(ROOT).as_posix()
                for filename in self.audit.OUTPUT_FILENAMES
            ]
            self.assertEqual(
                [record["path"] for record in manifest["outputs"]],
                expected,
            )

    def test_existing_or_internal_manifest_is_rejected_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / "manifest.json"
            existing.write_text("user manifest\n", encoding="utf-8")
            destination = Path(temp) / "results"
            with self.assertRaises(FileExistsError):
                self.audit.run_audit(
                    result_dir=destination,
                    manifest_path=existing,
                    full=False,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(
                existing.read_text(encoding="utf-8"), "user manifest\n"
            )

            with self.assertRaises(ValueError):
                self.audit.run_audit(
                    result_dir=destination,
                    manifest_path=destination / "manifest.json",
                    full=False,
                )
            self.assertFalse(destination.exists())

    def test_cli_requires_explicit_result_dir_and_full_requires_paper(self) -> None:
        missing_result = subprocess.run(
            [sys.executable, str(RUN_PATH)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(missing_result.returncode, 0)
        self.assertIn("--result-dir", missing_result.stderr)
        with tempfile.TemporaryDirectory() as temp:
            missing_paper = subprocess.run(
                [
                    sys.executable,
                    str(RUN_PATH),
                    "--result-dir",
                    str(Path(temp) / "results"),
                    "--full",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(missing_paper.returncode, 0)
        self.assertIn("--full requires --paper-pdf", missing_paper.stderr)


if __name__ == "__main__":
    unittest.main()
