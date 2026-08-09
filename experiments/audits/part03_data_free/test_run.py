#!/usr/bin/env python3
"""Focused non-Torch tests for the Part 3 data-free audit."""

from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

import numpy as np

from run import (
    AUDIT_ID,
    AUDIT_SCHEMA_VERSION,
    CLAIM_IDS,
    DEFAULT_BVN_ORDER,
    LAYER_MODULE_PATHS,
    LAYER_NAMES,
    METHOD_ORIGINS,
    PAPER_SHA256,
    AuditFailure,
    _configuration,
    _load_checkpoint,
    _sha256_json,
    analyze_layer,
    bivariate_normal_cdf,
    build_checks,
    distribution_diagnostics,
    enforce_checks,
    fixed_formula_audit,
    histogram_rows,
    layer_summary_rows,
    pair_summary_rows,
    projection_metrics,
    relu_gaussian_pair_kernel,
    relu_gaussian_self_kernel,
    select_evenly,
    select_pairs,
    select_stratified_indices,
    standardized_histogram,
    summary_rows,
    validate_summary,
    zero_bias_pair_kernel,
)


class GaussianFormulaTests(unittest.TestCase):
    def test_self_kernel_known_values_and_degenerate_case(self) -> None:
        self.assertEqual(relu_gaussian_self_kernel(0.0, 1.0), 0.5)
        self.assertEqual(relu_gaussian_self_kernel(2.0, 0.0), 4.0)
        self.assertEqual(relu_gaussian_self_kernel(-2.0, 0.0), 0.0)
        np.testing.assert_allclose(
            relu_gaussian_self_kernel(1.0, 1.0),
            1.9246602166562292,
            rtol=2e-15,
            atol=0.0,
        )

    def test_bivariate_cdf_independence_and_perfect_limits(self) -> None:
        np.testing.assert_allclose(
            bivariate_normal_cdf(0.0, 0.0, 0.0), 0.25, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            bivariate_normal_cdf(0.3, -0.2, 1.0),
            0.5 * math.erfc(0.2 / math.sqrt(2.0)),
            rtol=0.0,
            atol=1e-15,
        )
        expected = max(
            0.0,
            0.5 * math.erfc(-0.3 / math.sqrt(2.0))
            - 0.5 * math.erfc(-0.2 / math.sqrt(2.0)),
        )
        np.testing.assert_allclose(
            bivariate_normal_cdf(0.3, -0.2, -1.0),
            expected,
            rtol=0.0,
            atol=1e-15,
        )

    def test_pair_kernel_factorizes_at_independence(self) -> None:
        mean_i, mean_j = 0.7, -0.2
        std_i, std_j = 1.1, 0.8
        observed = relu_gaussian_pair_kernel(
            mean_i, mean_j, std_i, std_j, 0.0
        )
        first = std_i * math.exp(-0.5 * (mean_i / std_i) ** 2) / math.sqrt(
            2.0 * math.pi
        ) + mean_i * 0.5 * math.erfc(-mean_i / (std_i * math.sqrt(2.0)))
        second = std_j * math.exp(-0.5 * (mean_j / std_j) ** 2) / math.sqrt(
            2.0 * math.pi
        ) + mean_j * 0.5 * math.erfc(-mean_j / (std_j * math.sqrt(2.0)))
        np.testing.assert_allclose(observed, first * second, rtol=5e-15, atol=0.0)

    def test_biased_correlated_pair_kernel_matches_independent_reference(self) -> None:
        # Reference evaluated independently by adaptive conditional-Gaussian
        # integration, not by the Plackett expression implemented in run.py.
        observed = relu_gaussian_pair_kernel(0.4, -0.3, 1.2, 0.7, 0.55)
        np.testing.assert_allclose(
            observed,
            0.22355051299879558,
            rtol=2e-14,
            atol=2e-15,
        )

    def test_pair_kernel_degenerate_cases_are_stable_at_max_order(self) -> None:
        fixtures = (
            (-1.0, 0.1860710502876666),
            (1.0, 1.216712414414156),
        )
        for correlation, expected in fixtures:
            with self.subTest(correlation=correlation):
                observed = relu_gaussian_pair_kernel(
                    0.8, 0.6, 1.2, 0.7, correlation, order=512
                )
                self.assertTrue(math.isfinite(observed))
                np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-15)

        point_mass = relu_gaussian_pair_kernel(
            1.0, -0.2, 0.0, 0.7, 0.37, order=512
        )
        expected_first_moment = 0.7 * math.exp(-0.5 * (-0.2 / 0.7) ** 2) / math.sqrt(
            2.0 * math.pi
        ) + (-0.2) * 0.5 * math.erfc(0.2 / (0.7 * math.sqrt(2.0)))
        np.testing.assert_allclose(point_mass, expected_first_moment, rtol=2e-15, atol=0.0)
        self.assertEqual(
            relu_gaussian_pair_kernel(2.0, 3.0, 0.0, 0.0, -1.0, order=512),
            6.0,
        )
        with self.assertRaises(FloatingPointError):
            relu_gaussian_pair_kernel(
                0.0, 0.0, 1e308, 1e308, 1.0, order=512
            )
        with self.assertRaises(ValueError):
            relu_gaussian_pair_kernel(0.0, 0.0, 0.0, 0.0, 1.0, order=513)

    def test_exact_pair_kernel_matches_zero_bias_arc_cosine(self) -> None:
        for correlation in (-0.9, -0.3, 0.0, 0.5, 0.9):
            with self.subTest(correlation=correlation):
                exact = relu_gaussian_pair_kernel(
                    0.0, 0.0, 1.4, 0.6, correlation
                )
                reference = zero_bias_pair_kernel(1.4, 0.6, correlation)
                np.testing.assert_allclose(exact, reference, rtol=3e-13, atol=2e-15)

    def test_formula_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            relu_gaussian_self_kernel(0.0, -1.0)
        with self.assertRaises(ValueError):
            bivariate_normal_cdf(0.0, 0.0, 1.01)
        with self.assertRaises(ValueError):
            relu_gaussian_pair_kernel(0.0, 0.0, 1.0, 1.0, -1.01)

    def test_fixed_formula_audit_passes(self) -> None:
        audit = fixed_formula_audit(DEFAULT_BVN_ORDER)
        self.assertTrue(audit["all_cases_passed"])
        self.assertLessEqual(audit["max_scaled_error"], 1.0)


class DiagnosticsAndSelectionTests(unittest.TestCase):
    def test_distribution_diagnostics_are_deterministic(self) -> None:
        values = np.random.default_rng(11).normal(0.3, 1.2, 4096)
        first = distribution_diagnostics(values)
        second = distribution_diagnostics(values.copy())
        self.assertEqual(first, second)
        self.assertEqual(first["sample_count"], 4096)
        self.assertGreaterEqual(first["ks_distance_fitted_normal"], 0.0)
        self.assertLessEqual(first["ks_distance_fitted_normal"], 1.0)

    def test_standardized_histogram_has_fixed_bins_and_explicit_tails(self) -> None:
        values = np.concatenate((np.linspace(-9.0, 9.0, 1001), np.asarray([30.0])))
        histogram = standardized_histogram(values, bins=40, z_limit=5.0)
        self.assertEqual(histogram["bin_count"], 40)
        self.assertEqual(len(histogram["rows"]), 40)
        width = histogram["rows"][0]["bin_upper"] - histogram["rows"][0]["bin_lower"]
        in_range_mass = sum(row["empirical_density"] * width for row in histogram["rows"])
        total_mass = (
            in_range_mass
            + histogram["lower_tail_fraction"]
            + histogram["upper_tail_fraction"]
        )
        np.testing.assert_allclose(total_mass, 1.0, rtol=0.0, atol=2e-15)

    def test_even_channel_and_pair_selection_is_bounded(self) -> None:
        self.assertEqual(select_evenly(4, 9), [0, 1, 2, 3])
        self.assertEqual(select_evenly(10, 3), [0, 4, 9])
        pairs = select_pairs([0, 4, 9], 2)
        self.assertEqual(pairs, [(0, 4), (4, 9)])

    def test_stratified_selection_is_reproducible_and_balanced(self) -> None:
        targets = np.repeat(np.arange(10), 100)
        first = select_stratified_indices(targets, 203, 9)
        second = select_stratified_indices(targets, 203, 9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 203)
        counts = np.bincount(targets[first], minlength=10)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        self.assertEqual(len(set(first)), len(first))

    def test_projection_identity_does_not_depend_on_expansion_width(self) -> None:
        rng = np.random.default_rng(12)
        weight = rng.normal(size=(12, 3))
        target = weight @ rng.normal(size=3)
        record = projection_metrics(weight, target)
        self.assertGreater(record["rows"], record["columns"])
        self.assertLess(record["relative_residual"], 2e-15)

    def test_projection_detects_an_out_of_column_space_component(self) -> None:
        weight = np.vstack((np.eye(2), np.zeros((1, 2))))
        record = projection_metrics(weight, np.asarray([0.0, 0.0, 1.0]))
        self.assertEqual(record["relative_residual"], 1.0)


def synthetic_capture(name: str, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = 512
    if name == "conv0":
        output_channels, input_features = 8, 3
    else:
        output_channels, input_features = 6, 8
    weight = rng.normal(size=(output_channels, input_features))
    inputs = rng.normal(size=(rows, input_features))
    pre_all = inputs @ weight.T
    running_input_mean = rng.normal(scale=0.2, size=input_features)
    running_mean = weight @ running_input_mean
    running_var = np.var(pre_all, axis=0) + 0.2
    gamma = rng.uniform(0.5, 1.5, size=output_channels)
    beta = rng.normal(scale=0.3, size=output_channels)
    post_all = gamma * (pre_all - running_mean) / np.sqrt(running_var + 1e-5) + beta
    selected = list(range(output_channels))
    return {
        "name": name,
        "pre": pre_all[:, selected],
        "post": post_all[:, selected],
        "channel_indices": selected,
        "pre_mean_all": np.mean(pre_all, axis=0),
        "input_mean": np.mean(inputs, axis=0),
        "observation_count": rows,
        "weight": weight,
        "running_mean": running_mean,
        "running_var": running_var,
        "gamma": gamma,
        "beta": beta,
        "epsilon": 1e-5,
    }


def synthetic_summary() -> dict[str, object]:
    config = _configuration(
        seed=91,
        max_examples=64,
        batch_size=16,
        spatial_positions=8,
        channels_per_layer=8,
        pairs_per_layer=4,
        bvn_order=64,
    )
    layers = [
        analyze_layer(
            synthetic_capture(name, 100 + index),
            pairs_per_layer=4,
            bvn_order=64,
            seed=91,
        )
        for index, name in enumerate(LAYER_NAMES)
    ]
    formula = fixed_formula_audit(64)
    checks = build_checks(layers, formula, config)
    enforce_checks(checks)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "passed": True,
        "claim_ids": list(CLAIM_IDS),
        "paper": {"pdf_sha256": PAPER_SHA256},
        "config": config,
        "config_sha256": _sha256_json(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": {
            "reproduction_scope": "architecture_adaptation",
            "analysis_role": "diagnostic_audit",
            "paper_experiment_reproduced": False,
            "hypothesis_errors_are_pass_gates": False,
            "width_alone_causal_claim": False,
        },
        "input_provenance": {
            "checkpoint": {
                "filename": "fixture.pt",
                "sha256": "0" * 64,
                "state_schema_sha256": "1" * 64,
                "strict_state_dict_load": True,
                "historical_training_provenance": {"status": "incomplete"},
            },
            "dataset": {
                "name": "CIFAR-10",
                "split": "test",
                "dataset_length": 1000,
                "selected_example_count": 64,
                "selected_examples_sha256": "2" * 64,
                "selected_indices_sha256": "3" * 64,
                "selection_rule": config["example_selection"],
                "selection_seed": config["seed"],
                "raw_files": [
                    {"logical_role": "cifar10_test_batch"},
                    {"logical_role": "cifar10_batches_meta"},
                ],
                "preprocessing": {
                    "transform": "ToTensor",
                    "normalization": "none",
                    "augmentation": "none",
                },
            },
        },
        "formula_audit": formula,
        "checks": checks,
        "layers": layers,
    }


class OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = synthetic_summary()

    def test_complete_synthetic_summary_validates_without_torch(self) -> None:
        validate_summary(self.summary)
        self.assertTrue(self.summary["passed"])

    def test_hypothesis_errors_are_never_quality_gates(self) -> None:
        for layer in self.summary["layers"]:
            self.assertTrue(
                all(
                    record["hypothesis_fit_is_quality_gate"] is False
                    for record in layer["self_kernels"]
                )
            )
            self.assertTrue(
                all(
                    record["moment_discrepancies_are_quality_gates"] is False
                    for record in layer["moment_comparisons"]
                )
            )
            self.assertTrue(
                all(
                    record["hypothesis_fit_is_quality_gate"] is False
                    for record in layer["pair_kernels"]
                )
            )
        self.assertEqual(
            self.summary["checks"]["gaussian_hypothesis_pass_gate_count"]["value"],
            0.0,
        )

    def test_flat_tables_cover_layers_pairs_histograms_and_evidence(self) -> None:
        layer_rows = layer_summary_rows(self.summary)
        pair_rows = pair_summary_rows(self.summary)
        histogram_table = histogram_rows(self.summary)
        long_rows = summary_rows(self.summary)
        self.assertEqual([row["layer"] for row in layer_rows], list(LAYER_NAMES))
        self.assertEqual(len(pair_rows), 4 * len(LAYER_NAMES))
        self.assertEqual(len(histogram_table), 40 * len(LAYER_NAMES))
        self.assertEqual(
            {row["module_path"] for row in histogram_table},
            set(LAYER_MODULE_PATHS.values()),
        )
        self.assertIn("independent_inference", {row["evidence_label"] for row in long_rows})
        self.assertIn("counterexample", {row["evidence_label"] for row in long_rows})

    def test_validator_rejects_forged_pair_kernel(self) -> None:
        changed = copy.deepcopy(self.summary)
        changed["layers"][0]["pair_kernels"][0][
            "gaussian_exact_biased_pair_kernel"
        ] += 0.1
        with self.assertRaises(AuditFailure):
            validate_summary(changed)

    def test_validator_rejects_nonfinite_anywhere_in_document(self) -> None:
        changed = copy.deepcopy(self.summary)
        changed["otherwise_unread_extension"] = {"poison": float("nan")}
        with self.assertRaises(AuditFailure):
            validate_summary(changed)

    def test_validator_rejects_normality_and_registry_mutations(self) -> None:
        mutations = []

        changed_std = copy.deepcopy(self.summary)
        changed_std["layers"][0]["normality"][0]["std"] += 0.1
        mutations.append(("std-variance", changed_std))

        changed_sample = copy.deepcopy(self.summary)
        changed_sample["layers"][0]["normality"][0]["sample_count"] += 1
        mutations.append(("sample-registry", changed_sample))

        changed_selection = copy.deepcopy(self.summary)
        changed_selection["layers"][0]["selected_channels"][0] = 1
        mutations.append(("selection-registry", changed_selection))

        for label, changed in mutations:
            with self.subTest(label=label), self.assertRaises(AuditFailure):
                validate_summary(changed)

    def test_validator_rejects_moment_and_self_kernel_mutations(self) -> None:
        changed_moment = copy.deepcopy(self.summary)
        changed_moment["layers"][0]["moment_comparisons"][0][
            "post_bn_mean_symmetric_relative_error"
        ] += 0.1
        with self.assertRaises(AuditFailure):
            validate_summary(changed_moment)

        changed_self = copy.deepcopy(self.summary)
        changed_self["layers"][0]["self_kernels"][0][
            "gaussian_empirical_moments"
        ] += 0.1
        with self.assertRaises(AuditFailure):
            validate_summary(changed_self)

    def test_validator_rejects_pair_derived_field_mutations(self) -> None:
        fields = (
            "empirical_covariance",
            "empirical_correlation",
            "gaussian_exact_biased_pair_kernel",
            "gaussian_fit_error",
            "gaussian_fit_absolute_error",
            "gaussian_fit_symmetric_relative_error",
            "gaussian_fit_relative_error",
        )
        for field in fields:
            changed = copy.deepcopy(self.summary)
            changed["layers"][0]["pair_kernels"][0][field] += 0.1
            with self.subTest(field=field), self.assertRaises(AuditFailure):
                validate_summary(changed)

    def test_validator_rejects_shuffle_eq76_and_histogram_mutations(self) -> None:
        changed_shuffle = copy.deepcopy(self.summary)
        changed_shuffle["layers"][0]["shuffle_witness"][
            "dependence_scaled_change"
        ] += 0.1
        with self.assertRaises(AuditFailure):
            validate_summary(changed_shuffle)

        changed_eq76 = copy.deepcopy(self.summary)
        changed_eq76["layers"][0]["eq76_projection"][
            "empirical_common_map_relative_residual"
        ] += 0.1
        with self.assertRaises(AuditFailure):
            validate_summary(changed_eq76)

        changed_histogram = copy.deepcopy(self.summary)
        changed_histogram["layers"][0]["standardized_post_bn_histogram"]["rows"][0][
            "standard_normal_density"
        ] += 0.1
        with self.assertRaises(AuditFailure):
            validate_summary(changed_histogram)

    def test_validator_rejects_formula_fixture_mutation(self) -> None:
        changed = copy.deepcopy(self.summary)
        changed["formula_audit"]["cases"][0]["reference"] += 0.1
        with self.assertRaises(AuditFailure):
            validate_summary(changed)

    def test_checkpoint_loader_fails_closed_without_weights_only_support(self) -> None:
        class UnsupportedTorch:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def load(self, *args: object, **kwargs: object) -> object:
                self.calls.append((args, kwargs))
                raise TypeError("weights_only is unsupported")

        runtime = UnsupportedTorch()
        with self.assertRaisesRegex(AuditFailure, "weights_only=True"):
            _load_checkpoint(Path(__file__), runtime)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(
            runtime.calls[0][1], {"map_location": "cpu", "weights_only": True}
        )

    def test_validator_rejects_hypothesis_threshold_and_width_claim(self) -> None:
        changed_gate = copy.deepcopy(self.summary)
        changed_gate["layers"][0]["self_kernels"][0][
            "hypothesis_fit_is_quality_gate"
        ] = True
        with self.assertRaises(AuditFailure):
            validate_summary(changed_gate)

        changed_width = copy.deepcopy(self.summary)
        changed_width["layers"][0]["eq76_projection"]["causal_width_claim_made"] = True
        with self.assertRaises(AuditFailure):
            validate_summary(changed_width)

    def test_validator_rejects_config_or_training_provenance_forgery(self) -> None:
        changed_config = copy.deepcopy(self.summary)
        changed_config["config"]["seed"] += 1
        with self.assertRaises(AuditFailure):
            validate_summary(changed_config)

        changed_provenance = copy.deepcopy(self.summary)
        changed_provenance["input_provenance"]["checkpoint"][
            "historical_training_provenance"
        ]["status"] = "complete"
        with self.assertRaises(AuditFailure):
            validate_summary(changed_provenance)

        changed_strict = copy.deepcopy(self.summary)
        changed_strict["input_provenance"]["checkpoint"][
            "strict_state_dict_load"
        ] = False
        with self.assertRaises(AuditFailure):
            validate_summary(changed_strict)

        changed_preprocessing = copy.deepcopy(self.summary)
        changed_preprocessing["input_provenance"]["dataset"]["preprocessing"][
            "normalization"
        ] = "unknown"
        with self.assertRaises(AuditFailure):
            validate_summary(changed_preprocessing)

        changed_scope = copy.deepcopy(self.summary)
        changed_scope["reproduction_scope"]["reproduction_scope"] = "diagnostic_audit"
        with self.assertRaises(AuditFailure):
            validate_summary(changed_scope)

        for flag in (
            "paper_experiment_reproduced",
            "hypothesis_errors_are_pass_gates",
            "width_alone_causal_claim",
        ):
            changed_flag = copy.deepcopy(self.summary)
            changed_flag["reproduction_scope"][flag] = True
            with self.subTest(flag=flag), self.assertRaises(AuditFailure):
                validate_summary(changed_flag)

    def test_enforce_checks_rejects_missing_or_changed_gate(self) -> None:
        missing = copy.deepcopy(self.summary["checks"])
        missing.pop("layer_count")
        with self.assertRaises(AuditFailure):
            enforce_checks(missing)

        changed = copy.deepcopy(self.summary["checks"])
        changed["common_map_failure_count"]["value"] = 1.0
        changed["common_map_failure_count"]["passed"] = False
        with self.assertRaises(AuditFailure):
            enforce_checks(changed)

    def test_no_serialized_synthetic_path_is_absolute(self) -> None:
        encoded = json.dumps(self.summary, sort_keys=True)
        self.assertNotIn(str(Path(__file__).resolve().parents[3]), encoded)


if __name__ == "__main__":
    unittest.main()
