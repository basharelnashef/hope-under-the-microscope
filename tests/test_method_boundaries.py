#!/usr/bin/env python3
"""Regression tests for paper-literal and proposed-repair separation."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MethodBoundaryTests(unittest.TestCase):
    def test_sum_scale_policy_lives_only_in_proposed_repairs(self) -> None:
        paper_literal = ROOT / "src" / "hope_repro" / "paper_literal"
        self.assertFalse(paper_literal.is_symlink())
        for path in paper_literal.rglob("*"):
            self.assertFalse(path.is_symlink(), path)
        core = paper_literal / "core.py"
        self.assertTrue(core.is_file())
        self.assertFalse(core.is_symlink())
        paper_sources = list(paper_literal.rglob("*.py"))
        self.assertIn(core, paper_sources)
        for path in paper_sources:
            self.assertTrue(path.is_file(), path)
            self.assertFalse(path.is_symlink(), path)
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("fixed_scale", source, path)
            self.assertNotIn("s_value", source, path)
            self.assertNotIn("sum_preserving_scale", source, path)
        proposed = (
            ROOT
            / "src"
            / "hope_repro"
            / "proposed_repairs"
            / "sum_scale.py"
        ).read_text(encoding="utf-8")
        self.assertIn("class SumScaleHopeEncoder", proposed)
        self.assertIn("sum_preserving_scale", proposed)
        self.assertIn("layer.b_mat[first, second]", proposed)

    def test_maximum_entropy_repair_has_a_separate_namespace(self) -> None:
        paper_path = (
            ROOT
            / "src"
            / "hope_repro"
            / "paper_literal"
            / "maximum_entropy.py"
        )
        repair_path = (
            ROOT
            / "src"
            / "hope_repro"
            / "proposed_repairs"
            / "relative_entropy.py"
        )
        paper_tree = ast.parse(paper_path.read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(paper_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(paper_tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(
            any("proposed_repairs" in name for name in imported_modules)
        )
        paper_source = paper_path.read_text(encoding="utf-8")
        self.assertIn("def pure_stationarity_precision", paper_source)
        self.assertIn("def appendix_identity_precision", paper_source)
        self.assertNotIn("def reference_precision", paper_source)

        repair_source = repair_path.read_text(encoding="utf-8")
        self.assertIn("def gaussian_covariance_kl", repair_source)
        self.assertIn("def reference_precision", repair_source)
        self.assertIn("tau", repair_source)

    def test_part4_audit_labels_paper_and_repair_origins_separately(self) -> None:
        runner = (
            ROOT
            / "experiments"
            / "audits"
            / "part04_maximum_entropy"
            / "run.py"
        )
        spec = importlib.util.spec_from_file_location(
            "part4_method_boundary_contract",
            runner,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        origins = module.METHOD_ORIGINS
        self.assertEqual(
            origins["pure_maximum_entropy"]["namespace"],
            "hope_repro.paper_literal.maximum_entropy",
        )
        self.assertEqual(
            origins["appendix_identity_precision"]["namespace"],
            "hope_repro.paper_literal.maximum_entropy",
        )
        self.assertEqual(
            origins["relative_entropy_reference"],
            {
                "namespace": "hope_repro.proposed_repairs.relative_entropy",
                "evidence_label": "proposed_repair",
                "description": (
                    "Explicit KL projection against N(0, tau^2 I), not a "
                    "paper-reported method."
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
