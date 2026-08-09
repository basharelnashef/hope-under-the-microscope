#!/usr/bin/env python3
"""Reproduce the scale-invariance demonstrations used in Part 1.

The experiment uses a one-channel 1x1 convolutional path:

    input -> incoming filter -> BatchNorm -> ReLU -> outgoing filter slice

At each spatial position, a 1x1 convolution is a matrix multiplication. This
lets the script expose every term in the proof while retaining the exact
channel semantics of a Conv-BN-ReLU-Conv path.

Only NumPy is required. The script:

1. verifies exact positive PH-1 rescaling,
2. verifies an exact finite-epsilon BatchNorm reparameterization,
3. measures the uncorrected finite-epsilon negative control,
4. compares isolated parameter scores with two capacity formulas,
5. demonstrates disconnected and bias-only edge cases,
6. writes a CSV and a JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = ROOT / "artifacts" / "runs" / "part_01" / "manual"
DEFAULT_RESULT_DIR = DEFAULT_RUN_DIR


def relu(value: np.ndarray) -> np.ndarray:
    """Elementwise ReLU."""
    return np.maximum(value, 0.0)


def normal_pdf(value: float) -> float:
    """Standard-normal probability density."""
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    """Standard-normal cumulative distribution."""
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def relu_second_moment(mean: float, std: float) -> float:
    """E[ReLU(Y)^2] for Y distributed as N(mean, std^2).

    This is the self-kernel in HOPE Eq. 3 and Appendix E, Eq. 79.
    """
    std = float(std)
    mean = float(mean)
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError("mean and std must be finite")
    if std < 0.0:
        raise ValueError("std must be nonnegative")
    if std == 0.0:
        return max(mean, 0.0) ** 2
    ratio = mean / std
    moment = (
        (std * std + mean * mean) * normal_cdf(ratio)
        + mean * std * normal_pdf(ratio)
    )
    return max(moment, 0.0)


@dataclass(frozen=True)
class FunctionalNeuron:
    """One Conv-BN-ReLU channel and its full outgoing 1x1 slice."""

    raw_in: np.ndarray
    running_mean: float
    running_var: float
    epsilon: float
    gamma: float
    beta: float
    outgoing: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_in", np.asarray(self.raw_in, dtype=np.float64))
        object.__setattr__(self, "outgoing", np.asarray(self.outgoing, dtype=np.float64))
        if self.running_var < 0:
            raise ValueError("running_var must be nonnegative")
        if self.epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if self.running_var + self.epsilon <= 0:
            raise ValueError("running_var + epsilon must be positive")

    def effective_parameters(self) -> tuple[np.ndarray, float]:
        """Return HOPE Eq. 1 effective incoming weights and bias."""
        denominator = math.sqrt(self.running_var + self.epsilon)
        effective_in = (self.gamma / denominator) * self.raw_in
        effective_bias = self.beta - (
            self.gamma * self.running_mean / denominator
        )
        return effective_in, effective_bias

    def preactivation(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the inference-mode BN preactivation at many positions."""
        raw_projection = np.asarray(inputs, dtype=np.float64) @ self.raw_in
        centered = raw_projection - self.running_mean
        return (
            self.gamma * centered / math.sqrt(self.running_var + self.epsilon)
            + self.beta
        )

    def contribution(self, inputs: np.ndarray) -> np.ndarray:
        """Return f_i(x) = ReLU(y_i) w_out,i from HOPE Eq. 2."""
        activation = relu(self.preactivation(inputs))
        return activation[:, None] * self.outgoing[None, :]

    def empirical_capacity(self, inputs: np.ndarray) -> float:
        """RMS Euclidean size of the vector-valued local contribution."""
        values = self.contribution(inputs)
        return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))

    def analytic_capacity(self, finite_epsilon: bool = True) -> float:
        """Analytic capacity under the Gaussian marginal used by HOPE.

        If finite_epsilon is True, Eq. 77 is used exactly. If False, the
        paper's negligible-epsilon simplification in Eq. 78 is used.
        """
        if finite_epsilon:
            fraction = self.running_var / (self.running_var + self.epsilon)
            preactivation_std = abs(self.gamma) * math.sqrt(fraction)
        else:
            preactivation_std = abs(self.gamma)
        kernel = relu_second_moment(self.beta, preactivation_std)
        return float(np.linalg.norm(self.outgoing) * math.sqrt(kernel))

    def raw_l1(self) -> float:
        return float(np.sum(np.abs(self.raw_in)))

    def outgoing_l2(self) -> float:
        return float(np.linalg.norm(self.outgoing))

    def ph1_rescale(self, scale: float) -> "FunctionalNeuron":
        """Apply the exact positive PH-1 rescaling.

        gamma and beta are multiplied by scale, which scales the complete
        effective affine signal. Every outgoing connection is divided by the
        same positive scale.
        """
        if scale <= 0:
            raise ValueError("PH-1 rescaling requires a positive scale")
        return replace(
            self,
            gamma=self.gamma * scale,
            beta=self.beta * scale,
            outgoing=self.outgoing / scale,
        )

    def bn_normalization_rescale(self, scale: float) -> "FunctionalNeuron":
        """Rescale a raw filter while preserving the exact finite-epsilon map.

        The raw filter and its empirical BN moments transform naturally:

            w -> scale w, mu -> scale mu, var -> scale^2 var.

        At finite epsilon, gamma needs a small compensating factor. This makes
        the effective parameters from Eq. 1 exactly unchanged.
        """
        if scale <= 0:
            raise ValueError("BatchNorm rescaling requires a positive scale")
        new_var = scale * scale * self.running_var
        gamma_factor = math.sqrt(new_var + self.epsilon) / (
            scale * math.sqrt(self.running_var + self.epsilon)
        )
        return replace(
            self,
            raw_in=self.raw_in * scale,
            running_mean=self.running_mean * scale,
            running_var=new_var,
            gamma=self.gamma * gamma_factor,
        )

    def bn_normalization_naive(self, scale: float) -> "FunctionalNeuron":
        """Scale a raw filter and BN moments without correcting epsilon.

        The paper's negligible-epsilon capacity formula is unchanged, but the
        physical inference function generally moves at fixed finite epsilon.
        This is a negative control, not an invariant transformation.
        """
        if scale <= 0:
            raise ValueError("BatchNorm scaling requires a positive scale")
        return replace(
            self,
            raw_in=self.raw_in * scale,
            running_mean=self.running_mean * scale,
            running_var=self.running_var * scale * scale,
        )


def relative_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1e-300)
    return float(np.linalg.norm(candidate - reference) / denominator)


def max_abs_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(candidate - reference)))


def build_reference(seed: int, samples: int) -> tuple[FunctionalNeuron, np.ndarray]:
    """Construct a deterministic one-channel 1x1 convolutional path."""
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(samples, 9))
    raw_in = rng.normal(size=9)
    projections = inputs @ raw_in
    outgoing = rng.normal(size=6)
    neuron = FunctionalNeuron(
        raw_in=raw_in,
        running_mean=float(np.mean(projections)),
        running_var=float(np.var(projections)),
        epsilon=1e-5,
        gamma=1.35,
        beta=0.30,
        outgoing=outgoing,
    )
    return neuron, inputs


def ratio(value: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if value == 0 else math.inf
    return value / reference


def collect_sweep(
    neuron: FunctionalNeuron,
    inputs: np.ndarray,
    scales: Iterable[float],
) -> list[dict[str, float | str]]:
    """Collect exact symmetries and one finite-epsilon negative control."""
    reference_output = neuron.contribution(inputs)
    reference_effective_in, reference_effective_bias = neuron.effective_parameters()
    reference = {
        "raw_l1": neuron.raw_l1(),
        "bn_scale": abs(neuron.gamma),
        "outgoing_l2": neuron.outgoing_l2(),
        "capacity_finite_epsilon": neuron.analytic_capacity(True),
        "capacity_paper_approx": neuron.analytic_capacity(False),
        "capacity_empirical": neuron.empirical_capacity(inputs),
        "effective_in_l2": float(np.linalg.norm(reference_effective_in)),
    }

    rows: list[dict[str, float | str]] = []
    transformations = {
        "bn_normalization_corrected": neuron.bn_normalization_rescale,
        "bn_normalization_naive": neuron.bn_normalization_naive,
        "ph1_effective": neuron.ph1_rescale,
    }
    for transformation, transform in transformations.items():
        for scale in scales:
            candidate = transform(float(scale))
            output = candidate.contribution(inputs)
            effective_in, effective_bias = candidate.effective_parameters()
            row: dict[str, float | str] = {
                "transformation": transformation,
                "scale": float(scale),
                "raw_l1_ratio": ratio(candidate.raw_l1(), reference["raw_l1"]),
                "bn_scale_ratio": ratio(abs(candidate.gamma), reference["bn_scale"]),
                "outgoing_l2_ratio": ratio(
                    candidate.outgoing_l2(), reference["outgoing_l2"]
                ),
                "capacity_finite_epsilon_ratio": ratio(
                    candidate.analytic_capacity(True),
                    reference["capacity_finite_epsilon"],
                ),
                "capacity_paper_approx_ratio": ratio(
                    candidate.analytic_capacity(False),
                    reference["capacity_paper_approx"],
                ),
                "capacity_empirical_ratio": ratio(
                    candidate.empirical_capacity(inputs),
                    reference["capacity_empirical"],
                ),
                "effective_in_l2_ratio": ratio(
                    float(np.linalg.norm(effective_in)),
                    reference["effective_in_l2"],
                ),
                "effective_bias_delta": float(
                    effective_bias - reference_effective_bias
                ),
                "output_relative_l2_error": relative_l2_error(
                    reference_output, output
                ),
                "output_max_abs_error": max_abs_error(reference_output, output),
            }
            rows.append(row)
    return rows


def build_edge_cases(
    neuron: FunctionalNeuron,
    inputs: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Construct the disconnected and bias-only examples."""
    disconnected = replace(neuron, outgoing=np.zeros_like(neuron.outgoing))
    bias_only = FunctionalNeuron(
        raw_in=np.zeros_like(neuron.raw_in),
        running_mean=0.0,
        running_var=0.0,
        epsilon=neuron.epsilon,
        gamma=0.0,
        beta=1.20,
        outgoing=neuron.outgoing.copy(),
    )
    return {
        "disconnected": {
            "raw_l1": disconnected.raw_l1(),
            "outgoing_l2": disconnected.outgoing_l2(),
            "empirical_capacity": disconnected.empirical_capacity(inputs),
            "analytic_capacity": disconnected.analytic_capacity(True),
        },
        "bias_only": {
            "raw_l1": bias_only.raw_l1(),
            "outgoing_l2": bias_only.outgoing_l2(),
            "empirical_capacity": bias_only.empirical_capacity(inputs),
            "analytic_capacity": bias_only.analytic_capacity(True),
        },
    }


def validate_results(rows: list[dict[str, float | str]]) -> dict[str, float]:
    """Assert the mathematical invariants and return diagnostic maxima."""
    bn_rows = [
        row
        for row in rows
        if row["transformation"] == "bn_normalization_corrected"
    ]
    naive_rows = [
        row
        for row in rows
        if row["transformation"] == "bn_normalization_naive"
    ]
    ph1_rows = [
        row for row in rows if row["transformation"] == "ph1_effective"
    ]
    invariant_rows = bn_rows + ph1_rows

    max_output_error = max(
        float(row["output_relative_l2_error"]) for row in invariant_rows
    )
    max_finite_capacity_error = max(
        abs(float(row["capacity_finite_epsilon_ratio"]) - 1.0)
        for row in invariant_rows
    )
    max_empirical_capacity_error = max(
        abs(float(row["capacity_empirical_ratio"]) - 1.0)
        for row in invariant_rows
    )
    max_paper_capacity_error_corrected_bn = max(
        abs(float(row["capacity_paper_approx_ratio"]) - 1.0) for row in bn_rows
    )
    max_naive_output_change = max(
        float(row["output_relative_l2_error"]) for row in naive_rows
    )
    max_naive_empirical_capacity_change = max(
        abs(float(row["capacity_empirical_ratio"]) - 1.0) for row in naive_rows
    )
    max_bn_effective_error = max(
        abs(float(row["effective_in_l2_ratio"]) - 1.0) for row in bn_rows
    )
    max_bn_effective_bias_error = max(
        abs(float(row["effective_bias_delta"])) for row in bn_rows
    )

    for row in bn_rows:
        np.testing.assert_allclose(
            float(row["raw_l1_ratio"]),
            float(row["scale"]),
            rtol=2e-13,
            atol=2e-13,
        )
    for row in naive_rows:
        np.testing.assert_allclose(
            float(row["capacity_paper_approx_ratio"]),
            1.0,
            rtol=2e-13,
            atol=2e-13,
        )
    for row in ph1_rows:
        scale = float(row["scale"])
        np.testing.assert_allclose(
            float(row["bn_scale_ratio"]), scale, rtol=2e-13, atol=2e-13
        )
        np.testing.assert_allclose(
            float(row["outgoing_l2_ratio"]),
            1.0 / scale,
            rtol=2e-13,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            float(row["capacity_paper_approx_ratio"]),
            1.0,
            rtol=3e-13,
            atol=3e-13,
        )

    if max_output_error > 5e-13:
        raise AssertionError(f"pointwise invariance failed: {max_output_error:.3e}")
    if max_finite_capacity_error > 5e-13:
        raise AssertionError(
            f"finite-epsilon capacity invariance failed: "
            f"{max_finite_capacity_error:.3e}"
        )
    if max_empirical_capacity_error > 5e-13:
        raise AssertionError(
            f"empirical capacity invariance failed: "
            f"{max_empirical_capacity_error:.3e}"
        )
    if max_bn_effective_error > 5e-13 or max_bn_effective_bias_error > 5e-13:
        raise AssertionError("finite-epsilon BatchNorm effective map changed")

    return {
        "max_output_relative_l2_error": max_output_error,
        "max_finite_epsilon_capacity_ratio_error": max_finite_capacity_error,
        "max_empirical_capacity_invariance_error": max_empirical_capacity_error,
        "max_paper_approx_capacity_ratio_error_corrected_bn": (
            max_paper_capacity_error_corrected_bn
        ),
        "max_naive_bn_output_relative_l2_change": max_naive_output_change,
        "max_naive_bn_empirical_capacity_ratio_change": (
            max_naive_empirical_capacity_change
        ),
        "max_bn_effective_in_ratio_error": max_bn_effective_error,
        "max_bn_effective_bias_error": max_bn_effective_bias_error,
    }


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(
    seed: int,
    samples: int,
    result_dir: Path,
) -> dict[str, object]:
    neuron, inputs = build_reference(seed=seed, samples=samples)
    scales = np.logspace(-3.0, 3.0, 25)
    rows = collect_sweep(neuron, inputs, scales)
    diagnostics = validate_results(rows)
    edge_cases = build_edge_cases(neuron, inputs)

    csv_path = result_dir / "invariance_metrics.csv"
    json_path = result_dir / "summary.json"

    write_csv(rows, csv_path)
    payload: dict[str, object] = {
        "seed": seed,
        "samples": samples,
        "dtype": "float64",
        "input_distribution": "seeded standard Gaussian in 9 dimensions",
        "bn_buffer_estimation": (
            "estimated from the same seeded input positions"
        ),
        "scales": [float(value) for value in scales],
        "diagnostics": diagnostics,
        "reference_capacities": {
            "empirical": neuron.empirical_capacity(inputs),
            "finite_epsilon_gaussian_surrogate": neuron.analytic_capacity(True),
            "paper_epsilon_free_gaussian_surrogate": neuron.analytic_capacity(False),
        },
        "edge_cases": edge_cases,
        "paper_anchors": [
            "Eq. 1",
            "Eq. 2",
            "Eq. 3",
            "Eqs. 30-33",
            "Eqs. 73-75",
            "Eq. 77",
            "Eq. 79",
        ],
    }
    write_json(payload, json_path)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        **diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        seed=args.seed,
        samples=args.samples,
        result_dir=args.result_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
