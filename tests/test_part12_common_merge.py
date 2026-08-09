#!/usr/bin/env python3
"""Focused contracts for the framework-independent HOPE-R geometry."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hope_repro.proposed_repairs.hope_r.common import (  # noqa: E402
    CalibrationMeasure,
    FunctionalNeuron,
    calibrated_activation_gram,
    calibrated_cross_kernel,
    fold_batch_norm,
    functional_capacity,
    neuron_capacities,
    relu_self_kernel,
    validate_psd,
    zero_bias_cross_kernel_approximation,
)
from hope_repro.proposed_repairs.hope_r.merge import (  # noqa: E402
    sum_map_optimized_merge,
)


class CommonGeometryTests(unittest.TestCase):
    def test_batch_norm_fold_includes_bias_epsilon_and_zero_scale(self) -> None:
        folded = fold_batch_norm(
            raw_weight=np.array([2.0, -1.0], dtype=np.float64),
            raw_bias=0.75,
            gamma=-1.5,
            beta=0.2,
            running_mean=-0.25,
            running_variance=3.0,
            epsilon=0.125,
        )
        scale = -1.5 / math.sqrt(3.125)
        np.testing.assert_allclose(
            folded.weight,
            scale * np.array([2.0, -1.0], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(folded.bias, 0.2 + scale * (0.75 + 0.25))
        self.assertEqual(folded.batch_norm_scale, scale)
        self.assertEqual(folded.weight.dtype, np.float64)

        zero = fold_batch_norm(
            raw_weight=np.array([np.finfo(np.float64).max], dtype=np.float64),
            raw_bias=np.finfo(np.float64).max,
            gamma=0.0,
            beta=-0.75,
            running_mean=-np.finfo(np.float64).max,
            running_variance=0.0,
            epsilon=1e-5,
        )
        np.testing.assert_array_equal(zero.weight, np.array([0.0]))
        self.assertEqual(zero.bias, -0.75)
        self.assertEqual(zero.batch_norm_scale, 0.0)

    def test_batch_norm_fold_fails_closed_on_invalid_boundaries(self) -> None:
        arguments = {
            "raw_weight": np.array([1.0], dtype=np.float64),
            "raw_bias": 0.0,
            "gamma": 1.0,
            "beta": 0.0,
            "running_mean": 0.0,
            "running_variance": 0.0,
            "epsilon": 0.0,
        }
        with self.assertRaises(ValueError):
            fold_batch_norm(**arguments)
        with self.assertRaises(ValueError):
            fold_batch_norm(**{**arguments, "running_variance": -1.0})
        with self.assertRaises(ValueError):
            fold_batch_norm(**{**arguments, "epsilon": math.nan})
        with self.assertRaises(ValueError):
            fold_batch_norm(**{**arguments, "raw_weight": np.array([1.0j])})

    def test_biased_relu_self_kernel_has_exact_deterministic_limit(self) -> None:
        self.assertEqual(relu_self_kernel(2.5, 0.0), 6.25)
        self.assertEqual(relu_self_kernel(-2.5, -0.0), 0.0)
        self.assertEqual(relu_self_kernel(0.0, 2.0), 2.0)
        self.assertEqual(relu_self_kernel(0.0, -2.0), 2.0)
        expected = 2.0 * 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
        expected += math.exp(-0.5) / math.sqrt(2.0 * math.pi)
        self.assertAlmostEqual(relu_self_kernel(1.0, 1.0), expected, places=15)
        with self.assertRaises(ValueError):
            relu_self_kernel(math.inf, 1.0)

    def test_one_weighted_measure_gives_exact_cross_kernel_and_psd_gram(self) -> None:
        samples = np.array([[-2.0], [0.0], [1.0], [3.0]], dtype=np.float64)
        raw_weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        measure = CalibrationMeasure(samples, raw_weights, label="weighted_fixture")
        first = FunctionalNeuron(np.array([1.0]), 0.5, np.array([2.0, -1.0]))
        second = FunctionalNeuron(np.array([-0.5]), 1.0, np.array([0.5, 3.0]))
        activations = np.array(
            [
                [0.0, 2.0],
                [0.5, 1.0],
                [1.5, 0.5],
                [3.5, 0.0],
            ],
            dtype=np.float64,
        )
        expected = activations.T @ (
            (raw_weights / raw_weights.sum())[:, np.newaxis] * activations
        )
        gram = calibrated_activation_gram(measure, [first, second])
        np.testing.assert_allclose(gram, expected, rtol=0.0, atol=2e-16)
        self.assertAlmostEqual(
            calibrated_cross_kernel(measure, first, second), expected[0, 1], places=15
        )
        certificate = validate_psd(gram)
        self.assertGreaterEqual(
            certificate.minimum_eigenvalue, -certificate.tolerance
        )
        self.assertEqual(certificate.dimension, 2)

        outgoing = np.vstack([first.outgoing_weight, second.outgoing_weight])
        capacities = neuron_capacities(gram, outgoing)
        expected_capacities = np.sqrt(np.diag(expected)) * np.linalg.norm(
            outgoing, axis=1
        )
        np.testing.assert_allclose(capacities, expected_capacities, rtol=1e-15)
        self.assertAlmostEqual(
            functional_capacity(expected[0, 0], first.outgoing_weight),
            expected_capacities[0],
            places=14,
        )

    def test_psd_and_measure_validation_are_explicit(self) -> None:
        with self.assertRaises(ValueError):
            validate_psd(np.array([[1.0, 2.0], [2.0, 1.0]]))
        with self.assertRaises(ValueError):
            validate_psd(np.array([[1.0, 0.2], [0.1, 1.0]]), tolerance=1e-12)
        with self.assertRaises(ValueError):
            CalibrationMeasure(
                np.array([[0.0], [1.0]]), np.array([1.0, -1.0])
            )
        with self.assertRaises(ValueError):
            CalibrationMeasure(
                np.array([[0.0], [math.nan]]), np.array([1.0, 1.0])
            )
        with self.assertRaises(ValueError):
            FunctionalNeuron(np.array([1.0]), 0.0, np.array([math.inf]))

    def test_zero_bias_approximation_is_separate_and_labeled(self) -> None:
        at_zero_correlation = zero_bias_cross_kernel_approximation(0.5, 0.5, 0.0)
        self.assertAlmostEqual(at_zero_correlation, 1.0 / (2.0 * math.pi), places=15)
        self.assertAlmostEqual(
            zero_bias_cross_kernel_approximation(2.0, 3.0, 1.0),
            math.sqrt(6.0),
            places=15,
        )
        self.assertEqual(
            zero_bias_cross_kernel_approximation(2.0, 3.0, -1.0), 0.0
        )
        with self.assertRaises(ValueError):
            zero_bias_cross_kernel_approximation(1.0, 1.0, 1.0001)

    def test_positive_scale_symmetry_preserves_gram_and_capacity(self) -> None:
        samples = np.array(
            [[-2.0, 0.5], [-0.5, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.2]],
            dtype=np.float64,
        )
        measure = CalibrationMeasure.uniform(samples)
        first = FunctionalNeuron(
            np.array([0.7, -0.4]), 0.3, np.array([1.2, -0.8])
        )
        scaled = first.positive_rescaled(7.0)
        np.testing.assert_allclose(
            first.contributions(samples),
            scaled.contributions(samples),
            rtol=2e-15,
            atol=2e-15,
        )
        gram = calibrated_activation_gram(measure, [first])
        scaled_gram = calibrated_activation_gram(measure, [scaled])
        first_capacity = functional_capacity(gram[0, 0], first.outgoing_weight)
        scaled_capacity = functional_capacity(
            scaled_gram[0, 0], scaled.outgoing_weight
        )
        self.assertAlmostEqual(first_capacity, scaled_capacity, places=14)


class SumMapOptimizedMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        grid = np.array(
            [
                [-2.0, -1.0],
                [-2.0, 0.5],
                [-1.0, 2.0],
                [0.0, -2.0],
                [0.0, 0.0],
                [0.5, 1.5],
                [1.0, -1.0],
                [2.0, 0.5],
                [2.0, 2.0],
            ],
            dtype=np.float64,
        )
        self.measure = CalibrationMeasure(
            grid,
            np.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 2.0, 3.0]),
            label="merge_fixture",
        )

    def test_realizable_positive_scale_twins_are_preserved_tightly(self) -> None:
        first = FunctionalNeuron(
            np.array([0.8, -0.35]), 0.25, np.array([1.0, -2.0, 0.5])
        )
        second = first.positive_rescaled(3.25)
        result = sum_map_optimized_merge(
            first, second, self.measure, angular_samples=64
        )
        target = first.contributions(self.measure.samples) + second.contributions(
            self.measure.samples
        )
        parent = result.parent.contributions(self.measure.samples)
        scale = max(1.0, result.source_child_sum_norm)
        self.assertLessEqual(result.absolute_residual, 2e-14 * scale)
        self.assertLessEqual(result.relative_residual, 2e-14)
        np.testing.assert_allclose(parent, target, rtol=2e-14, atol=2e-14)
        self.assertAlmostEqual(result.parent_norm, result.source_child_sum_norm, places=13)
        self.assertEqual(result.span_rank, 1)
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.method, "sum_map_optimized")
        self.assertEqual(result.measure_label, "merge_fixture")

    def test_rank_two_search_is_deterministic_and_least_squares(self) -> None:
        first = FunctionalNeuron(
            np.array([1.0, 0.0]), 0.0, np.array([1.0, 0.0])
        )
        second = FunctionalNeuron(
            np.array([0.0, 1.0]), 0.3, np.array([0.0, 1.0])
        )
        result = sum_map_optimized_merge(
            first, second, self.measure, angular_samples=64
        )
        repeated = sum_map_optimized_merge(
            first, second, self.measure, angular_samples=64
        )
        self.assertEqual(result.span_rank, 2)
        self.assertEqual(result.candidate_count, repeated.candidate_count)
        self.assertGreaterEqual(result.candidate_count, 64)
        np.testing.assert_array_equal(
            result.selected_augmented_direction,
            repeated.selected_augmented_direction,
        )
        np.testing.assert_array_equal(
            result.selected_outgoing_weight,
            repeated.selected_outgoing_weight,
        )
        self.assertEqual(result.absolute_residual, repeated.absolute_residual)

        augmented = np.column_stack(
            [self.measure.samples, np.ones(self.measure.samples.shape[0])]
        )
        activation = np.maximum(
            augmented @ result.selected_augmented_direction, 0.0
        )
        target = first.contributions(self.measure.samples) + second.contributions(
            self.measure.samples
        )
        residual = target - result.parent.contributions(self.measure.samples)
        normal_equation = (self.measure.weights * activation) @ residual
        np.testing.assert_allclose(normal_equation, np.zeros(2), atol=2e-15)

        child_span = np.column_stack(
            [first.augmented_weight, second.augmented_weight]
        )
        coefficients = np.linalg.lstsq(
            child_span, result.selected_augmented_direction, rcond=None
        )[0]
        np.testing.assert_allclose(
            child_span @ coefficients,
            result.selected_augmented_direction,
            rtol=0.0,
            atol=2e-15,
        )

    def test_noncollinear_relu_sum_reports_measured_nonzero_residual(self) -> None:
        first = FunctionalNeuron(
            np.array([1.0, 0.0]), 0.0, np.array([1.0, 0.0])
        )
        second = FunctionalNeuron(
            np.array([0.0, 1.0]), 0.3, np.array([0.0, 1.0])
        )
        result = sum_map_optimized_merge(
            first, second, self.measure, angular_samples=128
        )
        target = first.contributions(self.measure.samples) + second.contributions(
            self.measure.samples
        )
        parent = result.parent.contributions(self.measure.samples)
        residual_squared = float(
            self.measure.weights
            @ np.einsum("ij,ij->i", target - parent, target - parent)
        )
        parent_squared = float(
            self.measure.weights @ np.einsum("ij,ij->i", parent, parent)
        )
        source_squared = float(
            self.measure.weights @ np.einsum("ij,ij->i", target, target)
        )
        self.assertGreater(result.absolute_residual, 0.1)
        self.assertGreater(result.relative_residual, 0.05)
        self.assertAlmostEqual(result.absolute_residual, math.sqrt(residual_squared), places=15)
        self.assertAlmostEqual(result.parent_norm, math.sqrt(parent_squared), places=15)
        self.assertAlmostEqual(result.source_child_sum_norm, math.sqrt(source_squared), places=15)
        self.assertEqual(result.measured_residual, result.absolute_residual)

    def test_merge_is_invariant_to_positive_child_parameter_scaling(self) -> None:
        first = FunctionalNeuron(
            np.array([1.0, -0.2]), 0.1, np.array([1.0, 0.5])
        )
        second = FunctionalNeuron(
            np.array([-0.3, 0.9]), -0.2, np.array([-0.4, 1.2])
        )
        original = sum_map_optimized_merge(
            first, second, self.measure, angular_samples=96
        )
        scaled = sum_map_optimized_merge(
            first.positive_rescaled(5.0),
            second.positive_rescaled(0.25),
            self.measure,
            angular_samples=96,
        )
        self.assertEqual(original.candidate_count, scaled.candidate_count)
        np.testing.assert_allclose(
            original.parent.contributions(self.measure.samples),
            scaled.parent.contributions(self.measure.samples),
            rtol=2e-14,
            atol=2e-14,
        )
        self.assertAlmostEqual(
            original.absolute_residual, scaled.absolute_residual, places=14
        )
        self.assertAlmostEqual(
            original.source_child_sum_norm,
            scaled.source_child_sum_norm,
            places=14,
        )

    def test_merge_rejects_mismatched_physical_shapes(self) -> None:
        first = FunctionalNeuron(np.array([1.0, 0.0]), 0.0, np.array([1.0]))
        wrong_input = FunctionalNeuron(np.array([1.0]), 0.0, np.array([1.0]))
        wrong_output = FunctionalNeuron(
            np.array([0.0, 1.0]), 0.0, np.array([1.0, 2.0])
        )
        with self.assertRaises(ValueError):
            sum_map_optimized_merge(first, wrong_input, self.measure)
        with self.assertRaises(ValueError):
            sum_map_optimized_merge(first, wrong_output, self.measure)
        with self.assertRaises(ValueError):
            sum_map_optimized_merge(first, first, self.measure, angular_samples=3)


if __name__ == "__main__":
    unittest.main()
