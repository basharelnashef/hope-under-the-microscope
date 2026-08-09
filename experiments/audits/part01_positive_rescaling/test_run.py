#!/usr/bin/env python3
"""Unit tests for the Part 1 functional-neuron demonstration."""

from __future__ import annotations

import unittest

import numpy as np

from run import (
    FunctionalNeuron,
    build_edge_cases,
    build_reference,
    collect_sweep,
    relu_second_moment,
    validate_results,
)


class FunctionalNeuronTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.neuron, cls.inputs = build_reference(seed=11, samples=4096)

    def test_ph1_rescaling_is_pointwise_exact(self) -> None:
        reference = self.neuron.contribution(self.inputs)
        for scale in (1e-3, 0.1, 1.0, 10.0, 1e3):
            candidate = self.neuron.ph1_rescale(scale).contribution(self.inputs)
            np.testing.assert_allclose(candidate, reference, rtol=2e-13, atol=2e-13)

    def test_bn_rescaling_preserves_finite_epsilon_map(self) -> None:
        reference_output = self.neuron.contribution(self.inputs)
        reference_effective = self.neuron.effective_parameters()
        for scale in (1e-3, 0.1, 1.0, 10.0, 1e3):
            candidate = self.neuron.bn_normalization_rescale(scale)
            candidate_effective = candidate.effective_parameters()
            np.testing.assert_allclose(
                candidate.contribution(self.inputs),
                reference_output,
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                candidate_effective[0],
                reference_effective[0],
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                candidate_effective[1],
                reference_effective[1],
                rtol=2e-13,
                atol=2e-13,
            )

    def test_full_sweep_invariants(self) -> None:
        rows = collect_sweep(
            self.neuron,
            self.inputs,
            np.logspace(-3.0, 3.0, 13),
        )
        diagnostics = validate_results(rows)
        self.assertLess(diagnostics["max_output_relative_l2_error"], 5e-13)

    def test_disconnected_and_bias_only_cases(self) -> None:
        cases = build_edge_cases(self.neuron, self.inputs)
        self.assertGreater(cases["disconnected"]["raw_l1"], 0.0)
        self.assertEqual(cases["disconnected"]["analytic_capacity"], 0.0)
        self.assertEqual(cases["bias_only"]["raw_l1"], 0.0)
        self.assertGreater(cases["bias_only"]["analytic_capacity"], 0.0)

    def test_deterministic_relu_has_expected_second_moment(self) -> None:
        self.assertAlmostEqual(relu_second_moment(1.2, 0.0), 1.44)
        self.assertEqual(relu_second_moment(-1.2, 0.0), 0.0)

    def test_tiny_positive_scale_is_not_treated_as_deterministic(self) -> None:
        np.testing.assert_allclose(
            relu_second_moment(0.0, 1e-16),
            5e-33,
            rtol=2e-16,
            atol=0.0,
        )
        self.assertGreater(relu_second_moment(-1e-16, 1e-16), 0.0)

    def test_relu_second_moment_rejects_invalid_parameters(self) -> None:
        for mean, std in ((0.0, -1.0), (float("nan"), 1.0), (0.0, float("inf"))):
            with self.assertRaises(ValueError):
                relu_second_moment(mean, std)

    def test_nonpositive_scales_are_rejected(self) -> None:
        for scale in (0.0, -1.0):
            with self.assertRaises(ValueError):
                self.neuron.ph1_rescale(scale)
            with self.assertRaises(ValueError):
                self.neuron.bn_normalization_rescale(scale)
            with self.assertRaises(ValueError):
                self.neuron.bn_normalization_naive(scale)

    def test_finite_epsilon_capacity_is_preserved(self) -> None:
        reference = self.neuron.analytic_capacity(finite_epsilon=True)
        for scale in (1e-3, 0.1, 1.0, 10.0, 1e3):
            normalized = self.neuron.bn_normalization_rescale(scale)
            ph1 = self.neuron.ph1_rescale(scale)
            self.assertAlmostEqual(
                normalized.analytic_capacity(finite_epsilon=True) / reference,
                1.0,
                places=12,
            )
            self.assertAlmostEqual(
                ph1.analytic_capacity(finite_epsilon=True) / reference,
                1.0,
                places=12,
            )

    def test_manual_bias_only_neuron(self) -> None:
        bias_only = FunctionalNeuron(
            raw_in=np.zeros(3),
            running_mean=0.0,
            running_var=0.0,
            epsilon=1e-5,
            gamma=0.0,
            beta=2.0,
            outgoing=np.array([3.0, 4.0]),
        )
        self.assertEqual(bias_only.raw_l1(), 0.0)
        self.assertAlmostEqual(bias_only.analytic_capacity(True), 10.0)

    def test_relu_second_moment_known_values(self) -> None:
        cases = [
            (0.0, 1.0, 0.5),
            (0.0, 2.0, 2.0),
            (1.0, 1.0, 1.9246602166562294),
            (-1.0, 1.0, 0.07533978334377077),
            (-8.0, 1.0, 1.8075064471458522e-17),
            (-10.0, 1.0, 1.4529276957119813e-25),
            (-20.0, 1.0, 1.3599128662753725e-91),
            (0.5, 1.7, 2.2579378411371294),
            (-2.0, 0.7, 0.0001690384752650628),
        ]
        for mean, std, expected in cases:
            np.testing.assert_allclose(
                relu_second_moment(mean, std),
                expected,
                rtol=2e-7,
                atol=1e-300,
            )

    def test_naive_bn_scaling_matches_finite_epsilon_prediction(self) -> None:
        reference_y = self.neuron.preactivation(self.inputs)
        for scale in (1e-3, 0.1, 1.0, 10.0, 1e3):
            candidate = self.neuron.bn_normalization_naive(scale)
            factor = (
                scale
                * np.sqrt(self.neuron.running_var + self.neuron.epsilon)
                / np.sqrt(
                    scale * scale * self.neuron.running_var
                    + self.neuron.epsilon
                )
            )
            predicted = self.neuron.beta + factor * (
                reference_y - self.neuron.beta
            )
            np.testing.assert_allclose(
                candidate.preactivation(self.inputs),
                predicted,
                rtol=2e-13,
                atol=2e-13,
            )

        moved = self.neuron.bn_normalization_naive(1e-3)
        relative_change = np.linalg.norm(
            moved.contribution(self.inputs)
            - self.neuron.contribution(self.inputs)
        ) / np.linalg.norm(self.neuron.contribution(self.inputs))
        self.assertGreater(relative_change, 1e-3)

    def test_zero_variance_and_zero_epsilon_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FunctionalNeuron(
                raw_in=np.zeros(2),
                running_mean=0.0,
                running_var=0.0,
                epsilon=0.0,
                gamma=1.0,
                beta=0.0,
                outgoing=np.ones(2),
            )


if __name__ == "__main__":
    unittest.main()
