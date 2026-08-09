#!/usr/bin/env python3
"""Fail-closed numerical audit of the Part 2 HOPE capacity formula.

This audit independently checks the biased ReLU self-kernel printed in HOPE
Eq. 3 and Eq. 79, the zero-shift and zero-variance special cases, synthetic
ranking behavior, and the within-layer monotonicity implied by Eq. 6.

The implementation uses only NumPy and the Python standard library. Seeded
Monte Carlo is a secondary check; deterministic fixed-node quadrature is the
primary numerical reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_02" / "manual"
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
AUDIT_ID = "part02_closed_form_capacity"
AUDIT_SCHEMA_VERSION = 1

SQRT_TWO = math.sqrt(2.0)
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
LOG_SQRT_TWO_PI = math.log(SQRT_TWO_PI)
LOG_FLOAT_MAX = math.log(sys.float_info.max)
LOG_FLOAT_MIN_SUBNORMAL = math.log(float(np.nextafter(0.0, 1.0)))
SQRT_FLOAT_MAX = math.sqrt(sys.float_info.max)
NEGATIVE_TAIL_SWITCH = 8.0

DEFAULT_SEED = 20260801
DEFAULT_MC_SAMPLES = 200_000
DEFAULT_MC_CHUNK_SIZE = 50_000
DEFAULT_QUADRATURE_ORDER = 256
DEFAULT_QUADRATURE_Z_LIMIT = 12.0

QUADRATURE_RATIOS = (-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0)
QUADRATURE_SIGNED_SCALES = (-3.0, -0.25, 0.25, 3.0)
ZERO_SHIFT_GAMMAS = (
    -1e150,
    -1e3,
    -1.0,
    -0.25,
    -1e-12,
    -1e-150,
    0.0,
    1e-150,
    1e-12,
    0.25,
    1.0,
    1e3,
    1e150,
)

MONTE_CARLO_CASES = (
    {"case_id": "mc_negative_shift", "beta": -1.5, "gamma": 1.0},
    {"case_id": "mc_negative_gamma", "beta": -0.25, "gamma": -0.5},
    {"case_id": "mc_zero_shift_small", "beta": 0.0, "gamma": 0.4},
    {"case_id": "mc_zero_shift_large", "beta": 0.0, "gamma": -2.0},
    {"case_id": "mc_positive_shift", "beta": 0.7, "gamma": 1.0},
    {"case_id": "mc_positive_ratio", "beta": 2.0, "gamma": -1.0},
)

STABILITY_CASES = (
    {"case_id": "gamma_zero_negative", "beta": -2.0, "gamma": 0.0},
    {"case_id": "gamma_zero_origin", "beta": 0.0, "gamma": 0.0},
    {"case_id": "gamma_zero_positive", "beta": 2.0, "gamma": 0.0},
    {"case_id": "positive_ratio_1e12", "beta": 1.0, "gamma": 1e-12},
    {"case_id": "negative_ratio_1e12", "beta": -1.0, "gamma": -1e-12},
    {"case_id": "negative_tail_ratio_8", "beta": -8.0, "gamma": 1.0},
    {"case_id": "negative_tail_ratio_20", "beta": -20.0, "gamma": -1.0},
    {"case_id": "negative_tail_ratio_30", "beta": -30.0, "gamma": 1.0},
    {"case_id": "near_degenerate_positive", "beta": 1e-140, "gamma": 1e-150},
    {"case_id": "near_degenerate_negative", "beta": -1e-140, "gamma": -1e-150},
    {"case_id": "large_scale_positive", "beta": 3e150, "gamma": 1e149},
    {"case_id": "large_scale_negative", "beta": -3e150, "gamma": -1e149},
)

COMPLEMENT_CASES = (
    {"case_id": "complement_ratio_8", "beta": 8.0, "gamma": 1.0},
    {"case_id": "complement_ratio_20", "beta": 20.0, "gamma": -1.0},
    {"case_id": "complement_ratio_30", "beta": 30.0, "gamma": 1.0},
    {"case_id": "complement_near_degenerate", "beta": 1.0, "gamma": 1e-12},
    {"case_id": "complement_large_scale", "beta": 3e150, "gamma": 1e149},
)

RANKING_NEURONS = (
    {
        "neuron_id": "n0",
        "incoming_l1": 9.0,
        "beta": 0.0,
        "gamma": 0.2,
        "outgoing_l2": 1.0,
    },
    {
        "neuron_id": "n1",
        "incoming_l1": 1.0,
        "beta": 0.0,
        "gamma": 2.0,
        "outgoing_l2": 0.25,
    },
    {
        "neuron_id": "n2",
        "incoming_l1": 3.0,
        "beta": 1.2,
        "gamma": 0.8,
        "outgoing_l2": 0.7,
    },
    {
        "neuron_id": "n3",
        "incoming_l1": 7.0,
        "beta": -2.0,
        "gamma": -1.5,
        "outgoing_l2": 2.0,
    },
    {
        "neuron_id": "n4",
        "incoming_l1": 0.5,
        "beta": 1.1,
        "gamma": 0.0,
        "outgoing_l2": 1.2,
    },
    {
        "neuron_id": "n5",
        "incoming_l1": 5.0,
        "beta": -0.3,
        "gamma": 0.6,
        "outgoing_l2": 3.0,
    },
    {
        "neuron_id": "n6",
        "incoming_l1": 2.0,
        "beta": 0.2,
        "gamma": -3.0,
        "outgoing_l2": 0.4,
    },
    {
        "neuron_id": "n7",
        "incoming_l1": 4.0,
        "beta": 0.0,
        "gamma": 1.1,
        "outgoing_l2": 1.0,
    },
)

EQ6_LAYER_WIDTH = 128.0
EQ6_LAYER_CAPACITY = 10.0
EQ6_CAPACITY_GRID = (0.0, 0.001, 0.1, 1.0, 3.0, 6.0, 9.0, 9.9)

TOLERANCES = {
    "quadrature_rtol": 5e-11,
    "quadrature_atol": 5e-14,
    "zero_shift_rtol": 8e-15,
    "complement_rtol": 2e-13,
    "complement_atol": 1e-300,
    "near_degenerate_rtol": 2e-13,
    "stability_upper_bound_rtol": 2e-13,
    "monte_carlo_sigma_multiplier": 7.0,
    "monte_carlo_atol": 5e-14,
    "eq6_secant_rtol": 2e-13,
    "eq6_secant_atol": 2e-13,
    "semantic_scaled_atol": 1e-4,
    "semantic_diagnostic_rtol": 1e-9,
    "semantic_rel_error_atol": 1e-12,
    "semantic_abs_error_atol": 5e-13,
}

METHOD_ORIGINS = {
    "analytical_self_kernel": {
        "label": "paper_formula_independent_stable_implementation",
        "evidence_label": "verified_fact",
        "description": "Independent float64 implementation of HOPE Eq. 3 and Eq. 79.",
    },
    "deterministic_quadrature": {
        "label": "independent_fixed_node_numerical_check",
        "evidence_label": "verified_fact",
        "description": "Independent fixed-node Gauss-Legendre integration of the one-dimensional Gaussian moment.",
    },
    "seeded_monte_carlo": {
        "label": "independent_seeded_secondary_check",
        "evidence_label": "verified_fact",
        "description": "Seeded simulation used only as a secondary numerical check, not as proof.",
    },
    "ranking_comparison": {
        "label": "independent_synthetic_ranking_comparison",
        "evidence_label": "independent_inference",
        "description": "Synthetic fixture comparing four local scores; it is not a paper experiment.",
    },
    "eq6_monotonicity": {
        "label": "independent_same_layer_algebraic_check",
        "evidence_label": "verified_fact",
        "description": "Independent evaluation of Eq. 6 for fixed N and pre-action layer capacity.",
    },
}

CSV_FIELDS = (
    "section",
    "case_id",
    "metric",
    "method",
    "method_origin",
    "evidence_label",
    "beta",
    "gamma",
    "incoming_l1",
    "outgoing_l2",
    "layer_capacity",
    "layer_width",
    "value",
    "reference_value",
    "abs_error",
    "rel_error",
    "standard_error",
    "tolerance",
    "rank",
    "passed",
    "notes",
)

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part02_capacity/README.md",
    "experiments/audits/part02_capacity/run.py",
    "experiments/audits/part02_capacity/test_run.py",
)


class AuditFailure(RuntimeError):
    """Raised when any declared audit tolerance fails."""


def _as_float_result(value: np.ndarray) -> float | np.ndarray:
    """Return a Python float for a scalar and an ndarray otherwise."""
    if value.ndim == 0:
        return float(value)
    return value


def _erfc_array(values: np.ndarray) -> np.ndarray:
    """Evaluate math.erfc over an ndarray without adding a SciPy dependency."""
    flat = np.fromiter(
        (math.erfc(float(value)) for value in values.ravel()),
        dtype=np.float64,
        count=values.size,
    )
    return flat.reshape(values.shape)


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * _erfc_array(-values / SQRT_TWO)


def _negative_tail_log_second_moment(x: float) -> float:
    """Return log(E[ReLU(Z-x)^2]) for x > 0 and Z standard normal.

    The direct Eq. 79 terms nearly cancel for large x. Below the switch this
    function factors out exp(-x^2/2) and evaluates an erfcx-equivalent
    bracket. At and above the switch it uses the optimally truncated series

        2 phi(x) / x^3 * (1 - 6/x^2 + 45/x^4 - 420/x^6 + ...).

    Returning the logarithm lets the caller combine this very small moment
    with very large or very small gamma without an avoidable intermediate
    overflow or underflow.
    """
    if not x > 0.0:
        raise ValueError("negative-tail argument must be positive")
    if math.isinf(x):
        return -math.inf

    if x < NEGATIVE_TAIL_SWITCH:
        half_x_squared = 0.5 * x * x
        erfcx = math.exp(half_x_squared) * math.erfc(x / SQRT_TWO)
        bracket = 0.5 * (1.0 + x * x) * erfcx - x / SQRT_TWO_PI
        if not bracket > 0.0 or not math.isfinite(bracket):
            raise FloatingPointError(
                "stable negative-tail bracket was not positive and finite"
            )
        return -half_x_squared + math.log(bracket)

    inverse_square = 1.0 / (x * x)
    terms = [1.0]
    term = 1.0
    previous_magnitude = abs(term)
    for index in range(1, 128):
        ratio = -((index + 1.0) * (2.0 * index + 1.0) / index)
        candidate = term * ratio * inverse_square
        magnitude = abs(candidate)
        if magnitude >= previous_magnitude:
            break
        terms.append(candidate)
        term = candidate
        previous_magnitude = magnitude
        if magnitude <= sys.float_info.epsilon * abs(math.fsum(terms)):
            break
    series = math.fsum(terms)
    if not series > 0.0 or not math.isfinite(series):
        raise FloatingPointError(
            "negative-tail asymptotic series was not positive and finite"
        )
    half_x_squared = math.inf if x > math.sqrt(sys.float_info.max) else 0.5 * x * x
    return (
        math.log(2.0)
        - half_x_squared
        - LOG_SQRT_TWO_PI
        - 3.0 * math.log(x)
        + math.log(series)
    )


def relu_self_kernel(
    beta: float | Sequence[float] | np.ndarray,
    gamma: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Vectorized stable evaluation of E[ReLU(Y)^2].

    Y is Gaussian with mean beta and variance gamma^2. Inputs follow NumPy
    broadcasting. When gamma is exactly zero, the deterministic value
    max(0, beta)^2 is returned without division. Non-finite inputs and true
    results outside the float64 range fail explicitly.
    """
    beta_array, gamma_array = np.broadcast_arrays(
        np.asarray(beta, dtype=np.float64),
        np.asarray(gamma, dtype=np.float64),
    )
    if not np.all(np.isfinite(beta_array)):
        raise ValueError("beta must contain only finite values")
    if not np.all(np.isfinite(gamma_array)):
        raise ValueError("gamma must contain only finite values")

    flat_beta = beta_array.ravel()
    flat_scale = np.abs(gamma_array.ravel())
    result = np.empty_like(flat_beta)

    zero_scale = flat_scale == 0.0
    if np.any(zero_scale):
        positive_beta = np.maximum(flat_beta[zero_scale], 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            deterministic = positive_beta * positive_beta
        if not np.all(np.isfinite(deterministic)):
            raise FloatingPointError(
                "deterministic ReLU self-kernel exceeds float64 range"
            )
        result[zero_scale] = deterministic

    zero_beta = (~zero_scale) & (flat_beta == 0.0)
    if np.any(zero_beta):
        with np.errstate(over="ignore", invalid="ignore"):
            exact_zero_shift = (0.5 * flat_scale[zero_beta]) * flat_scale[zero_beta]
        if not np.all(np.isfinite(exact_zero_shift)):
            raise FloatingPointError("zero-shift ReLU self-kernel exceeds float64 range")
        result[zero_beta] = exact_zero_shift

    general = (~zero_scale) & (~zero_beta)
    if np.any(general):
        active_beta = flat_beta[general]
        active_scale = flat_scale[general]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            ratio = active_beta / active_scale
        active_result = np.empty_like(active_beta)

        nonnegative_ratio = ratio >= 0.0
        if np.any(nonnegative_ratio):
            local_beta = active_beta[nonnegative_ratio]
            local_scale = active_scale[nonnegative_ratio]
            local_ratio = ratio[nonnegative_ratio]
            magnitude = np.maximum(local_beta, local_scale)
            normalized_beta = local_beta / magnitude
            normalized_scale = local_scale / magnitude
            cdf = _normal_cdf(local_ratio)
            with np.errstate(under="ignore"):
                pdf = np.exp(-0.5 * local_ratio * local_ratio) / SQRT_TWO_PI
            normalized_moment = (
                (normalized_scale * normalized_scale + normalized_beta * normalized_beta)
                * cdf
                + normalized_beta * normalized_scale * pdf
            )
            positive_values = np.empty_like(local_beta)
            direct_square = magnitude <= SQRT_FLOAT_MAX
            if np.any(direct_square):
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    positive_values[direct_square] = (
                        magnitude[direct_square]
                        * magnitude[direct_square]
                        * normalized_moment[direct_square]
                    )
            logarithmic = ~direct_square
            if np.any(logarithmic):
                log_kernel = (
                    2.0 * np.log(magnitude[logarithmic])
                    + np.log(normalized_moment[logarithmic])
                )
                if np.any(log_kernel > LOG_FLOAT_MAX):
                    raise FloatingPointError("ReLU self-kernel exceeds float64 range")
                with np.errstate(under="ignore"):
                    positive_values[logarithmic] = np.exp(log_kernel)
            if not np.all(np.isfinite(positive_values)):
                raise FloatingPointError("ReLU self-kernel exceeds float64 range")
            active_result[nonnegative_ratio] = positive_values

        negative_ratio = ~nonnegative_ratio
        if np.any(negative_ratio):
            local_ratio = ratio[negative_ratio]
            local_scale = active_scale[negative_ratio]
            tail_logs = np.fromiter(
                (
                    _negative_tail_log_second_moment(float(-value))
                    for value in local_ratio
                ),
                dtype=np.float64,
                count=local_ratio.size,
            )
            log_kernel = 2.0 * np.log(local_scale) + tail_logs
            if np.any(log_kernel > LOG_FLOAT_MAX):
                raise FloatingPointError("ReLU self-kernel exceeds float64 range")
            underflow = log_kernel < LOG_FLOAT_MIN_SUBNORMAL
            values = np.zeros_like(log_kernel)
            with np.errstate(under="ignore"):
                values[~underflow] = np.exp(log_kernel[~underflow])
            active_result[negative_ratio] = values

        if not np.all(np.isfinite(active_result)):
            raise FloatingPointError("ReLU self-kernel produced a non-finite value")
        if np.any(active_result < 0.0):
            raise FloatingPointError("ReLU self-kernel produced a negative value")
        result[general] = active_result

    reshaped = result.reshape(beta_array.shape)
    return _as_float_result(reshaped)


def neuron_capacity(
    beta: float | Sequence[float] | np.ndarray,
    gamma: float | Sequence[float] | np.ndarray,
    outgoing_l2: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Return ||w_out||_2 * sqrt(K(i,i)) with input validation."""
    kernel, outgoing = np.broadcast_arrays(
        np.asarray(relu_self_kernel(beta, gamma), dtype=np.float64),
        np.asarray(outgoing_l2, dtype=np.float64),
    )
    if not np.all(np.isfinite(outgoing)) or np.any(outgoing < 0.0):
        raise ValueError("outgoing_l2 must contain finite nonnegative values")
    with np.errstate(over="ignore", invalid="ignore"):
        capacity = outgoing * np.sqrt(kernel)
    if not np.all(np.isfinite(capacity)):
        raise FloatingPointError("neuron capacity exceeds float64 range")
    return _as_float_result(capacity)


def deterministic_quadrature_self_kernel(
    beta: float | Sequence[float] | np.ndarray,
    gamma: float | Sequence[float] | np.ndarray,
    *,
    order: int = DEFAULT_QUADRATURE_ORDER,
    z_limit: float = DEFAULT_QUADRATURE_Z_LIMIT,
) -> float | np.ndarray:
    """Integrate the one-dimensional Gaussian moment on a fixed grid.

    The declared grid is an order-point Gauss-Legendre rule over the active
    part of [-z_limit, z_limit]. For the declared parameter cases, the audit
    gates the total observed analytical-versus-quadrature discrepancy, which
    includes truncation outside that interval. gamma == 0 uses the exact
    deterministic value.
    """
    if order < 16:
        raise ValueError("quadrature order must be at least 16")
    if not math.isfinite(z_limit) or z_limit < 8.0:
        raise ValueError("quadrature z_limit must be finite and at least 8")

    beta_array, gamma_array = np.broadcast_arrays(
        np.asarray(beta, dtype=np.float64),
        np.asarray(gamma, dtype=np.float64),
    )
    if not np.all(np.isfinite(beta_array)) or not np.all(np.isfinite(gamma_array)):
        raise ValueError("quadrature inputs must be finite")

    nodes, weights = np.polynomial.legendre.leggauss(order)
    flat_beta = beta_array.ravel()
    flat_scale = np.abs(gamma_array.ravel())
    result = np.empty_like(flat_beta)
    for index, (mean, scale) in enumerate(zip(flat_beta, flat_scale)):
        if scale == 0.0:
            result[index] = max(float(mean), 0.0) ** 2
            continue
        threshold = -float(mean) / float(scale)
        lower = max(threshold, -z_limit)
        upper = z_limit
        if lower >= upper:
            result[index] = 0.0
            continue
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        z_values = midpoint + half_width * nodes
        y_values = float(mean) + float(scale) * z_values
        integrand = y_values * y_values * np.exp(-0.5 * z_values * z_values) / SQRT_TWO_PI
        result[index] = half_width * float(np.dot(weights, integrand))

    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("deterministic quadrature produced an invalid value")
    return _as_float_result(result.reshape(beta_array.shape))


def pruning_cost(
    capacity: float | Sequence[float] | np.ndarray,
    layer_width: float,
    pre_action_layer_capacity: float,
) -> float | np.ndarray:
    """Evaluate Eq. 6 pruning cost on 0 <= c < E_a."""
    capacities = np.asarray(capacity, dtype=np.float64)
    if not math.isfinite(layer_width) or layer_width <= 0.0:
        raise ValueError("layer_width N must be finite and positive")
    if not math.isfinite(pre_action_layer_capacity) or pre_action_layer_capacity <= 0.0:
        raise ValueError("pre_action_layer_capacity E_a must be finite and positive")
    if not np.all(np.isfinite(capacities)):
        raise ValueError("capacities must be finite")
    if np.any(capacities < 0.0) or np.any(capacities >= pre_action_layer_capacity):
        raise ValueError("Eq. 6 requires 0 <= capacity < E_a")
    values = layer_width * capacities / (pre_action_layer_capacity - capacities)
    return _as_float_result(values)


def pruning_cost_derivative(
    capacity: float | Sequence[float] | np.ndarray,
    layer_width: float,
    pre_action_layer_capacity: float,
) -> float | np.ndarray:
    """Evaluate dJ/dc = N E_a / (E_a-c)^2 on the Eq. 6 domain."""
    capacities = np.asarray(capacity, dtype=np.float64)
    pruning_cost(capacities, layer_width, pre_action_layer_capacity)
    derivative = (
        layer_width
        * pre_action_layer_capacity
        / (pre_action_layer_capacity - capacities) ** 2
    )
    return _as_float_result(derivative)


def _relative_error(value: float, reference: float) -> float:
    error = abs(value - reference)
    if reference == 0.0:
        return 0.0 if error == 0.0 else error
    return error / abs(reference)


def _scaled_error(error: float, tolerance: float) -> float:
    if tolerance == 0.0:
        return 0.0 if error == 0.0 else sys.float_info.max
    return error / tolerance


def _method_metadata(key: str) -> tuple[str, str]:
    record = METHOD_ORIGINS[key]
    return str(record["label"]), str(record["evidence_label"])


def _blank_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(values)
    return row


def build_quadrature_audit(
    order: int,
    z_limit: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    origins = _method_metadata("deterministic_quadrature")
    for ratio in QUADRATURE_RATIOS:
        for gamma in QUADRATURE_SIGNED_SCALES:
            beta = ratio * abs(gamma)
            analytical = float(relu_self_kernel(beta, gamma))
            quadrature = float(
                deterministic_quadrature_self_kernel(
                    beta,
                    gamma,
                    order=order,
                    z_limit=z_limit,
                )
            )
            repeated = float(
                deterministic_quadrature_self_kernel(
                    beta,
                    gamma,
                    order=order,
                    z_limit=z_limit,
                )
            )
            error = abs(quadrature - analytical)
            tolerance = (
                TOLERANCES["quadrature_atol"]
                + TOLERANCES["quadrature_rtol"] * abs(analytical)
            )
            case_id = f"ratio_{ratio:+g}_gamma_{gamma:+g}"
            record = {
                "case_id": case_id,
                "ratio": ratio,
                "beta": beta,
                "gamma": gamma,
                "analytical": analytical,
                "quadrature": quadrature,
                "repeated_quadrature": repeated,
                "abs_error": error,
                "rel_error": _relative_error(quadrature, analytical),
                "tolerance": tolerance,
                "scaled_error": _scaled_error(error, tolerance),
                "repeat_abs_difference": abs(repeated - quadrature),
                "passed": error <= tolerance and repeated == quadrature,
            }
            cases.append(record)
            rows.append(
                _blank_row(
                    section="analytical_quadrature",
                    case_id=case_id,
                    metric="relu_self_kernel",
                    method="deterministic_quadrature",
                    method_origin=origins[0],
                    evidence_label=origins[1],
                    beta=beta,
                    gamma=gamma,
                    value=quadrature,
                    reference_value=analytical,
                    abs_error=error,
                    rel_error=record["rel_error"],
                    tolerance=tolerance,
                    passed=record["passed"],
                    notes=f"fixed Gauss-Legendre order={order}; z in [-{z_limit:g}, {z_limit:g}]",
                )
            )
    aggregate = {
        "case_count": len(cases),
        "max_abs_error": max(case["abs_error"] for case in cases),
        "max_rel_error": max(case["rel_error"] for case in cases),
        "max_scaled_error": max(case["scaled_error"] for case in cases),
        "repeat_max_abs_difference": max(
            case["repeat_abs_difference"] for case in cases
        ),
        "all_cases_passed": all(case["passed"] for case in cases),
    }
    return {
        "grid": {
            "ratios": list(QUADRATURE_RATIOS),
            "signed_gamma_scales": list(QUADRATURE_SIGNED_SCALES),
            "order": order,
            "z_limit": z_limit,
            "rule": "fixed-node Gauss-Legendre over the active subinterval",
        },
        "tolerances": {
            "rtol": TOLERANCES["quadrature_rtol"],
            "atol": TOLERANCES["quadrature_atol"],
        },
        "cases": cases,
        "aggregate": aggregate,
    }, rows


def build_monte_carlo_audit(
    seed: int,
    samples: int,
    chunk_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if samples < 1_000:
        raise ValueError("mc_samples must be at least 1000")
    if chunk_size < 1:
        raise ValueError("mc_chunk_size must be positive")
    beta = np.asarray([case["beta"] for case in MONTE_CARLO_CASES], dtype=np.float64)
    gamma = np.asarray([case["gamma"] for case in MONTE_CARLO_CASES], dtype=np.float64)
    scales = np.abs(gamma)
    sums = np.zeros(beta.size, dtype=np.float64)
    sums_of_squares = np.zeros(beta.size, dtype=np.float64)
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < samples:
        count = min(chunk_size, samples - completed)
        z_values = rng.standard_normal(count)
        activation_squared = np.maximum(
            beta[:, None] + scales[:, None] * z_values[None, :],
            0.0,
        ) ** 2
        sums += np.sum(activation_squared, axis=1, dtype=np.float64)
        sums_of_squares += np.sum(
            activation_squared * activation_squared,
            axis=1,
            dtype=np.float64,
        )
        completed += count

    estimates = sums / samples
    sample_variances = np.maximum(
        (sums_of_squares - sums * sums / samples) / (samples - 1),
        0.0,
    )
    standard_errors = np.sqrt(sample_variances / samples)
    references = np.asarray(relu_self_kernel(beta, gamma), dtype=np.float64)
    origins = _method_metadata("seeded_monte_carlo")
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, fixture in enumerate(MONTE_CARLO_CASES):
        estimate = float(estimates[index])
        reference = float(references[index])
        standard_error = float(standard_errors[index])
        tolerance = (
            TOLERANCES["monte_carlo_atol"]
            + TOLERANCES["monte_carlo_sigma_multiplier"] * standard_error
        )
        error = abs(estimate - reference)
        record = {
            **fixture,
            "analytical": reference,
            "estimate": estimate,
            "abs_error": error,
            "rel_error": _relative_error(estimate, reference),
            "standard_error": standard_error,
            "tolerance": tolerance,
            "scaled_error": _scaled_error(error, tolerance),
            "passed": error <= tolerance,
        }
        cases.append(record)
        rows.append(
            _blank_row(
                section="monte_carlo",
                case_id=fixture["case_id"],
                metric="relu_self_kernel",
                method="seeded_monte_carlo",
                method_origin=origins[0],
                evidence_label=origins[1],
                beta=fixture["beta"],
                gamma=fixture["gamma"],
                value=estimate,
                reference_value=reference,
                abs_error=error,
                rel_error=record["rel_error"],
                standard_error=standard_error,
                tolerance=tolerance,
                passed=record["passed"],
                notes=f"secondary check; seed={seed}; samples={samples}",
            )
        )
    aggregate = {
        "case_count": len(cases),
        "max_abs_error": max(case["abs_error"] for case in cases),
        "max_rel_error": max(case["rel_error"] for case in cases),
        "max_scaled_error": max(case["scaled_error"] for case in cases),
        "all_cases_passed": all(case["passed"] for case in cases),
    }
    return {
        "seed": seed,
        "samples": samples,
        "chunk_size": chunk_size,
        "role": "secondary numerical check, not proof",
        "tolerances": {
            "sigma_multiplier": TOLERANCES["monte_carlo_sigma_multiplier"],
            "atol": TOLERANCES["monte_carlo_atol"],
        },
        "cases": cases,
        "aggregate": aggregate,
    }, rows


def build_zero_shift_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    origins = _method_metadata("analytical_self_kernel")
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for gamma in ZERO_SHIFT_GAMMAS:
        value = float(relu_self_kernel(0.0, gamma))
        reference = 0.5 * gamma * gamma
        error = abs(value - reference)
        tolerance = TOLERANCES["zero_shift_rtol"] * abs(reference)
        record = {
            "case_id": f"zero_shift_gamma_{gamma:+g}",
            "beta": 0.0,
            "gamma": gamma,
            "analytical": value,
            "identity_value": reference,
            "abs_error": error,
            "rel_error": _relative_error(value, reference),
            "tolerance": tolerance,
            "scaled_error": _scaled_error(error, tolerance),
            "passed": error <= tolerance,
        }
        cases.append(record)
        rows.append(
            _blank_row(
                section="zero_shift",
                case_id=record["case_id"],
                metric="K_gamma_squared_over_2",
                method="analytical_self_kernel",
                method_origin=origins[0],
                evidence_label=origins[1],
                beta=0.0,
                gamma=gamma,
                value=value,
                reference_value=reference,
                abs_error=error,
                rel_error=record["rel_error"],
                tolerance=tolerance,
                passed=record["passed"],
                notes="zero-shift identity across gamma signs and scales",
            )
        )
    return {
        "identity": "K(0,gamma) = gamma^2 / 2",
        "cases": cases,
        "aggregate": {
            "case_count": len(cases),
            "max_abs_error": max(case["abs_error"] for case in cases),
            "max_scaled_error": max(case["scaled_error"] for case in cases),
            "all_cases_passed": all(case["passed"] for case in cases),
        },
    }, rows


def build_stability_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    origins = _method_metadata("analytical_self_kernel")
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for fixture in STABILITY_CASES:
        beta = float(fixture["beta"])
        gamma = float(fixture["gamma"])
        value = float(relu_self_kernel(beta, gamma))
        upper_bound = beta * beta + gamma * gamma
        upper_bound_tolerance = (
            TOLERANCES["stability_upper_bound_rtol"] * abs(upper_bound)
        )
        upper_bound_excess = max(value - upper_bound, 0.0)
        finite = math.isfinite(value)
        nonnegative = value >= 0.0
        bounded = upper_bound_excess <= upper_bound_tolerance
        passed = finite and nonnegative and bounded
        record = {
            **fixture,
            "analytical": value,
            "second_moment_upper_bound": upper_bound,
            "upper_bound_excess": upper_bound_excess,
            "upper_bound_tolerance": upper_bound_tolerance,
            "upper_bound_scaled_excess": _scaled_error(
                upper_bound_excess,
                upper_bound_tolerance,
            ),
            "finite": finite,
            "nonnegative": nonnegative,
            "bounded_by_ungated_second_moment": bounded,
            "passed": passed,
        }
        cases.append(record)
        rows.append(
            _blank_row(
                section="stability",
                case_id=fixture["case_id"],
                metric="finite_nonnegative_bounded_kernel",
                method="analytical_self_kernel",
                method_origin=origins[0],
                evidence_label=origins[1],
                beta=beta,
                gamma=gamma,
                value=value,
                reference_value=upper_bound,
                abs_error=upper_bound_excess,
                tolerance=upper_bound_tolerance,
                passed=passed,
                notes=(
                    "reference_value is the ungated second-moment upper bound; "
                    "abs_error records one-sided numerical excess"
                ),
            )
        )

    complement_cases: list[dict[str, Any]] = []
    for fixture in COMPLEMENT_CASES:
        beta = float(fixture["beta"])
        gamma = float(fixture["gamma"])
        positive = float(relu_self_kernel(beta, gamma))
        negative = float(relu_self_kernel(-beta, gamma))
        value = positive + negative
        reference = beta * beta + gamma * gamma
        error = abs(value - reference)
        tolerance = (
            TOLERANCES["complement_atol"]
            + TOLERANCES["complement_rtol"] * abs(reference)
        )
        record = {
            **fixture,
            "positive_shift_kernel": positive,
            "negative_shift_kernel": negative,
            "sum": value,
            "identity_value": reference,
            "abs_error": error,
            "rel_error": _relative_error(value, reference),
            "tolerance": tolerance,
            "scaled_error": _scaled_error(error, tolerance),
            "passed": error <= tolerance,
        }
        complement_cases.append(record)
        rows.append(
            _blank_row(
                section="stability",
                case_id=fixture["case_id"],
                metric="complement_identity",
                method="analytical_self_kernel",
                method_origin=origins[0],
                evidence_label=origins[1],
                beta=beta,
                gamma=gamma,
                value=value,
                reference_value=reference,
                abs_error=error,
                rel_error=record["rel_error"],
                tolerance=tolerance,
                passed=record["passed"],
                notes="K(beta,gamma) + K(-beta,gamma) = beta^2 + gamma^2",
            )
        )

    near_degenerate_cases: list[dict[str, Any]] = []
    for fixture in (
        {"case_id": "limit_positive", "beta": 1.0, "gamma": 1e-12},
        {"case_id": "limit_negative", "beta": -1.0, "gamma": -1e-12},
        {"case_id": "limit_tiny_positive", "beta": 1e-140, "gamma": 1e-150},
        {"case_id": "limit_tiny_negative", "beta": -1e-140, "gamma": -1e-150},
    ):
        beta = float(fixture["beta"])
        gamma = float(fixture["gamma"])
        value = float(relu_self_kernel(beta, gamma))
        reference = max(beta, 0.0) ** 2
        error = abs(value - reference)
        tolerance = (
            gamma * gamma
            + TOLERANCES["near_degenerate_rtol"] * abs(reference)
        )
        record = {
            **fixture,
            "analytical": value,
            "deterministic_limit": reference,
            "abs_error": error,
            "rel_error": _relative_error(value, reference),
            "tolerance": tolerance,
            "scaled_error": _scaled_error(error, tolerance),
            "passed": error <= tolerance,
        }
        near_degenerate_cases.append(record)
        rows.append(
            _blank_row(
                section="stability",
                case_id=fixture["case_id"],
                metric="near_degenerate_gamma_limit",
                method="analytical_self_kernel",
                method_origin=origins[0],
                evidence_label=origins[1],
                beta=beta,
                gamma=gamma,
                value=value,
                reference_value=reference,
                abs_error=error,
                rel_error=record["rel_error"],
                tolerance=tolerance,
                passed=record["passed"],
                notes="comparison with max(0,beta)^2 as gamma approaches zero",
            )
        )

    return {
        "negative_tail_switch": NEGATIVE_TAIL_SWITCH,
        "cases": cases,
        "complement_identity_cases": complement_cases,
        "near_degenerate_cases": near_degenerate_cases,
        "aggregate": {
            "case_count": len(cases),
            "nonfinite_count": sum(not case["finite"] for case in cases),
            "negative_count": sum(not case["nonnegative"] for case in cases),
            "upper_bound_violation_count": sum(
                not case["bounded_by_ungated_second_moment"] for case in cases
            ),
            "max_complement_scaled_error": max(
                case["scaled_error"] for case in complement_cases
            ),
            "max_near_degenerate_scaled_error": max(
                case["scaled_error"] for case in near_degenerate_cases
            ),
            "all_cases_passed": (
                all(case["passed"] for case in cases)
                and all(case["passed"] for case in complement_cases)
                and all(case["passed"] for case in near_degenerate_cases)
            ),
        },
    }, rows


def _descending_ranks(
    neuron_ids: Sequence[str],
    scores: Sequence[float],
) -> tuple[list[int], list[str]]:
    order_indices = sorted(
        range(len(neuron_ids)),
        key=lambda index: (-float(scores[index]), neuron_ids[index]),
    )
    ranks = [0] * len(neuron_ids)
    for rank, index in enumerate(order_indices, start=1):
        ranks[index] = rank
    return ranks, [neuron_ids[index] for index in order_indices]


def _kendall_tau_without_ties(
    first_ranks: Sequence[int],
    second_ranks: Sequence[int],
) -> float:
    concordant = 0
    discordant = 0
    for left in range(len(first_ranks)):
        for right in range(left + 1, len(first_ranks)):
            first_direction = first_ranks[left] - first_ranks[right]
            second_direction = second_ranks[left] - second_ranks[right]
            if first_direction == 0 or second_direction == 0:
                raise ValueError("ranking fixture unexpectedly contains a tie")
            if first_direction * second_direction > 0:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def build_ranking_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    neuron_ids = [str(record["neuron_id"]) for record in RANKING_NEURONS]
    incoming_l1 = np.asarray(
        [record["incoming_l1"] for record in RANKING_NEURONS], dtype=np.float64
    )
    beta = np.asarray([record["beta"] for record in RANKING_NEURONS], dtype=np.float64)
    gamma = np.asarray([record["gamma"] for record in RANKING_NEURONS], dtype=np.float64)
    outgoing_l2 = np.asarray(
        [record["outgoing_l2"] for record in RANKING_NEURONS], dtype=np.float64
    )
    score_arrays = {
        "incoming_l1": incoming_l1,
        "abs_gamma": np.abs(gamma),
        "abs_gamma_x_outgoing_l2": np.abs(gamma) * outgoing_l2,
        "full_capacity": np.asarray(
            neuron_capacity(beta, gamma, outgoing_l2), dtype=np.float64
        ),
    }
    origin, evidence = _method_metadata("ranking_comparison")
    ranks: dict[str, list[int]] = {}
    orders: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    neuron_records: list[dict[str, Any]] = [dict(record) for record in RANKING_NEURONS]
    for method, scores in score_arrays.items():
        method_ranks, method_order = _descending_ranks(neuron_ids, scores)
        ranks[method] = method_ranks
        orders[method] = method_order
        for index, neuron_id in enumerate(neuron_ids):
            neuron_records[index][method] = float(scores[index])
            neuron_records[index][f"{method}_rank"] = method_ranks[index]
            rows.append(
                _blank_row(
                    section="rankings",
                    case_id=neuron_id,
                    metric=method,
                    method=method,
                    method_origin=origin,
                    evidence_label=evidence,
                    beta=float(beta[index]),
                    gamma=float(gamma[index]),
                    incoming_l1=float(incoming_l1[index]),
                    outgoing_l2=float(outgoing_l2[index]),
                    value=float(scores[index]),
                    rank=method_ranks[index],
                    passed=True,
                    notes="descending rank in a deterministic synthetic same-layer fixture",
                )
            )

    full_ranks = ranks["full_capacity"]
    comparisons: dict[str, dict[str, Any]] = {}
    for method, method_ranks in ranks.items():
        tau = _kendall_tau_without_ties(method_ranks, full_ranks)
        comparisons[method] = {
            "reference": "full_capacity",
            "kendall_tau": tau,
            "identical_order": orders[method] == orders["full_capacity"],
        }
        rows.append(
            _blank_row(
                section="rankings",
                case_id=f"{method}_vs_full_capacity",
                metric="kendall_tau",
                method=method,
                method_origin=origin,
                evidence_label=evidence,
                value=tau,
                reference_value=1.0,
                passed=(method == "full_capacity" or tau < 1.0),
                notes="pairwise order agreement with full capacity on simulated data",
            )
        )

    unique_order_count = len({tuple(order) for order in orders.values()})
    baseline_identical_to_full_count = sum(
        comparison["identical_order"]
        for method, comparison in comparisons.items()
        if method != "full_capacity"
    )
    permutation_valid = all(
        sorted(order) == sorted(neuron_ids) for order in orders.values()
    )
    return {
        "fixture_label": "simulated deterministic same-layer neuron statistics",
        "neurons": neuron_records,
        "orders": orders,
        "comparisons": comparisons,
        "aggregate": {
            "method_count": len(score_arrays),
            "unique_order_count": unique_order_count,
            "baseline_identical_to_full_count": baseline_identical_to_full_count,
            "all_orders_are_permutations": permutation_valid,
        },
    }, rows


def build_eq6_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capacities = np.asarray(EQ6_CAPACITY_GRID, dtype=np.float64)
    costs = np.asarray(
        pruning_cost(capacities, EQ6_LAYER_WIDTH, EQ6_LAYER_CAPACITY),
        dtype=np.float64,
    )
    derivatives = np.asarray(
        pruning_cost_derivative(capacities, EQ6_LAYER_WIDTH, EQ6_LAYER_CAPACITY),
        dtype=np.float64,
    )
    secants = np.diff(costs) / np.diff(capacities)
    expected_secants = (
        EQ6_LAYER_WIDTH
        * EQ6_LAYER_CAPACITY
        / (
            (EQ6_LAYER_CAPACITY - capacities[:-1])
            * (EQ6_LAYER_CAPACITY - capacities[1:])
        )
    )
    secant_errors = np.abs(secants - expected_secants)
    secant_tolerances = (
        TOLERANCES["eq6_secant_atol"]
        + TOLERANCES["eq6_secant_rtol"] * np.abs(expected_secants)
    )
    secant_scaled_errors = secant_errors / secant_tolerances
    capacity_order = list(np.argsort(capacities))
    cost_order = list(np.argsort(costs))
    origin, evidence = _method_metadata("eq6_monotonicity")
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for index, capacity in enumerate(capacities):
        record = {
            "case_id": f"capacity_{capacity:g}",
            "capacity": float(capacity),
            "cost": float(costs[index]),
            "analytic_derivative": float(derivatives[index]),
            "passed": bool(derivatives[index] > 0.0),
        }
        cases.append(record)
        rows.append(
            _blank_row(
                section="eq6_monotonicity",
                case_id=record["case_id"],
                metric="pruning_cost",
                method="eq6_pruning_cost",
                method_origin=origin,
                evidence_label=evidence,
                layer_capacity=float(capacity),
                layer_width=EQ6_LAYER_WIDTH,
                value=float(costs[index]),
                reference_value=float(derivatives[index]),
                passed=record["passed"],
                notes=f"fixed E_a={EQ6_LAYER_CAPACITY:g}; reference_value is dJ/dc",
            )
        )
    return {
        "conditions": {
            "scope": "one fixed layer only",
            "layer_width_N": EQ6_LAYER_WIDTH,
            "pre_action_layer_capacity_E_a": EQ6_LAYER_CAPACITY,
            "domain": "0 <= c < E_a",
        },
        "capacity_grid": list(EQ6_CAPACITY_GRID),
        "cases": cases,
        "adjacent_secants": [float(value) for value in secants],
        "expected_adjacent_secants": [float(value) for value in expected_secants],
        "aggregate": {
            "minimum_derivative": float(np.min(derivatives)),
            "nonpositive_derivative_count": int(np.sum(derivatives <= 0.0)),
            "nonincreasing_step_count": int(np.sum(np.diff(costs) <= 0.0)),
            "capacity_and_cost_order_match": capacity_order == cost_order,
            "max_secant_scaled_error": float(np.max(secant_scaled_errors)),
        },
    }, rows


def _expected_check_ranges() -> dict[str, dict[str, Any]]:
    return {
        "quadrature_case_count": {"minimum": 28.0, "maximum": 28.0, "units": "cases"},
        "quadrature_max_scaled_error": {"minimum": 0.0, "maximum": 1.0, "units": "tolerance multiples"},
        "quadrature_repeat_max_abs_difference": {"minimum": 0.0, "maximum": 0.0, "units": "kernel units"},
        "monte_carlo_case_count": {"minimum": 6.0, "maximum": 6.0, "units": "cases"},
        "monte_carlo_max_scaled_error": {"minimum": 0.0, "maximum": 1.0, "units": "tolerance multiples"},
        "zero_shift_case_count": {"minimum": 13.0, "maximum": 13.0, "units": "cases"},
        "zero_shift_max_scaled_error": {"minimum": 0.0, "maximum": 1.0, "units": "tolerance multiples"},
        "stability_nonfinite_count": {"minimum": 0.0, "maximum": 0.0, "units": "cases"},
        "stability_negative_count": {"minimum": 0.0, "maximum": 0.0, "units": "cases"},
        "stability_upper_bound_violation_count": {"minimum": 0.0, "maximum": 0.0, "units": "cases"},
        "stability_max_complement_scaled_error": {"minimum": 0.0, "maximum": 1.0, "units": "tolerance multiples"},
        "stability_max_near_degenerate_scaled_error": {"minimum": 0.0, "maximum": 1.0, "units": "tolerance multiples"},
        "ranking_method_count": {"minimum": 4.0, "maximum": 4.0, "units": "methods"},
        "ranking_unique_order_count": {"minimum": 3.0, "maximum": 4.0, "units": "orders"},
        "ranking_baseline_identical_to_full_count": {"minimum": 0.0, "maximum": 0.0, "units": "methods"},
        "ranking_all_orders_are_permutations": {"minimum": 1.0, "maximum": 1.0, "units": "boolean"},
        "eq6_minimum_derivative": {"minimum": 1e-300, "maximum": 1e12, "units": "cost per capacity"},
        "eq6_nonpositive_derivative_count": {"minimum": 0.0, "maximum": 0.0, "units": "grid points"},
        "eq6_nonincreasing_step_count": {"minimum": 0.0, "maximum": 0.0, "units": "adjacent pairs"},
        "eq6_capacity_and_cost_order_match": {"minimum": 1.0, "maximum": 1.0, "units": "boolean"},
        "eq6_max_secant_scaled_error": {"minimum": 0.0, "maximum": 1.0, "units": "tolerance multiples"},
    }


def build_checks(
    quadrature: Mapping[str, Any],
    monte_carlo: Mapping[str, Any],
    zero_shift: Mapping[str, Any],
    stability: Mapping[str, Any],
    rankings: Mapping[str, Any],
    eq6: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    observed = {
        "quadrature_case_count": quadrature["aggregate"]["case_count"],
        "quadrature_max_scaled_error": quadrature["aggregate"]["max_scaled_error"],
        "quadrature_repeat_max_abs_difference": quadrature["aggregate"]["repeat_max_abs_difference"],
        "monte_carlo_case_count": monte_carlo["aggregate"]["case_count"],
        "monte_carlo_max_scaled_error": monte_carlo["aggregate"]["max_scaled_error"],
        "zero_shift_case_count": zero_shift["aggregate"]["case_count"],
        "zero_shift_max_scaled_error": zero_shift["aggregate"]["max_scaled_error"],
        "stability_nonfinite_count": stability["aggregate"]["nonfinite_count"],
        "stability_negative_count": stability["aggregate"]["negative_count"],
        "stability_upper_bound_violation_count": stability["aggregate"]["upper_bound_violation_count"],
        "stability_max_complement_scaled_error": stability["aggregate"]["max_complement_scaled_error"],
        "stability_max_near_degenerate_scaled_error": stability["aggregate"]["max_near_degenerate_scaled_error"],
        "ranking_method_count": rankings["aggregate"]["method_count"],
        "ranking_unique_order_count": rankings["aggregate"]["unique_order_count"],
        "ranking_baseline_identical_to_full_count": rankings["aggregate"]["baseline_identical_to_full_count"],
        "ranking_all_orders_are_permutations": float(rankings["aggregate"]["all_orders_are_permutations"]),
        "eq6_minimum_derivative": eq6["aggregate"]["minimum_derivative"],
        "eq6_nonpositive_derivative_count": eq6["aggregate"]["nonpositive_derivative_count"],
        "eq6_nonincreasing_step_count": eq6["aggregate"]["nonincreasing_step_count"],
        "eq6_capacity_and_cost_order_match": float(eq6["aggregate"]["capacity_and_cost_order_match"]),
        "eq6_max_secant_scaled_error": eq6["aggregate"]["max_secant_scaled_error"],
    }
    ranges = _expected_check_ranges()
    checks: dict[str, dict[str, Any]] = {}
    for name, expected in ranges.items():
        value = float(observed[name])
        minimum = float(expected["minimum"])
        maximum = float(expected["maximum"])
        checks[name] = {
            "value": value,
            "expected_minimum": minimum,
            "expected_maximum": maximum,
            "units": expected["units"],
            "passed": math.isfinite(value) and minimum <= value <= maximum,
        }
    return checks


def enforce_checks(checks: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject missing, non-finite, malformed, or out-of-range check values."""
    expected_names = set(_expected_check_ranges())
    actual_names = set(checks)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise AuditFailure(
            f"check registry mismatch: missing={missing}, unexpected={unexpected}"
        )
    failures: list[str] = []
    for name, record in checks.items():
        try:
            value = float(record["value"])
            minimum = float(record["expected_minimum"])
            maximum = float(record["expected_maximum"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{name}: malformed ({error})")
            continue
        computed_pass = (
            math.isfinite(value)
            and math.isfinite(minimum)
            and math.isfinite(maximum)
            and minimum <= maximum
            and minimum <= value <= maximum
        )
        if record.get("passed") is not computed_pass or not computed_pass:
            failures.append(
                f"{name}: value={value!r}, expected=[{minimum!r}, {maximum!r}]"
            )
    if failures:
        raise AuditFailure("Part 2 capacity audit failed: " + "; ".join(failures))


def _semantic_numeric_tolerance(path: str, expected: float) -> float:
    """Return a cross-version comparison tolerance for one audit diagnostic."""
    leaf = path.rsplit(".", 1)[-1]
    quadrature_diagnostic = path.startswith("analytical_quadrature.")
    if quadrature_diagnostic and "scaled_error" in leaf:
        return (
            TOLERANCES["semantic_scaled_atol"]
            + TOLERANCES["semantic_diagnostic_rtol"] * abs(expected)
        )
    if quadrature_diagnostic and "rel_error" in leaf:
        return (
            TOLERANCES["semantic_rel_error_atol"]
            + TOLERANCES["semantic_diagnostic_rtol"] * abs(expected)
        )
    if quadrature_diagnostic and "abs_error" in leaf:
        return (
            TOLERANCES["semantic_abs_error_atol"]
            + TOLERANCES["quadrature_rtol"] * abs(expected)
        )
    return max(
        256.0 * math.ulp(expected),
        TOLERANCES["quadrature_rtol"] * abs(expected),
    )


def _compare_semantic_payload(actual: Any, expected: Any, path: str) -> None:
    """Compare regenerated audit data without requiring last-bit identity."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise AuditFailure(f"summary semantic type mismatch at {path}")
        if set(actual) != set(expected):
            raise AuditFailure(f"summary semantic key mismatch at {path}")
        for key in expected:
            _compare_semantic_payload(
                actual[key],
                expected[key],
                f"{path}.{key}" if path else str(key),
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AuditFailure(f"summary semantic list mismatch at {path}")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _compare_semantic_payload(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
        return
    if isinstance(expected, bool):
        if actual is not expected:
            raise AuditFailure(f"summary semantic boolean mismatch at {path}")
        return
    if isinstance(expected, int):
        if isinstance(actual, bool) or actual != expected:
            raise AuditFailure(f"summary semantic integer mismatch at {path}")
        return
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise AuditFailure(f"summary semantic numeric type mismatch at {path}")
        actual_float = float(actual)
        if not math.isfinite(actual_float) or not math.isfinite(expected):
            raise AuditFailure(f"summary semantic non-finite value at {path}")
        tolerance = _semantic_numeric_tolerance(path, expected)
        if abs(actual_float - expected) > tolerance:
            raise AuditFailure(
                f"summary semantic numeric mismatch at {path}: "
                f"observed={actual_float!r}, regenerated={expected!r}, "
                f"tolerance={tolerance!r}"
            )
        return
    if actual != expected:
        raise AuditFailure(f"summary semantic value mismatch at {path}")


def _expected_config_from_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        quadrature = config["quadrature"]
        if not isinstance(quadrature, Mapping):
            raise TypeError("quadrature is not a mapping")
        return _configuration(
            seed=int(config["seed"]),
            mc_samples=int(config["monte_carlo_samples"]),
            mc_chunk_size=int(config["monte_carlo_chunk_size"]),
            quadrature_order=int(quadrature["order"]),
            quadrature_z_limit=float(quadrature["z_limit"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AuditFailure(f"summary configuration is malformed: {error}") from error


def _regenerate_summary_sections(
    config: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    quadrature = config["quadrature"]
    analytical_quadrature, _ = build_quadrature_audit(
        int(quadrature["order"]),
        float(quadrature["z_limit"]),
    )
    monte_carlo, _ = build_monte_carlo_audit(
        int(config["seed"]),
        int(config["monte_carlo_samples"]),
        int(config["monte_carlo_chunk_size"]),
    )
    zero_shift, _ = build_zero_shift_audit()
    stability, _ = build_stability_audit()
    rankings, _ = build_ranking_audit()
    eq6_monotonicity, _ = build_eq6_audit()
    return {
        "analytical_quadrature": analytical_quadrature,
        "monte_carlo": monte_carlo,
        "zero_shift": zero_shift,
        "stability": stability,
        "rankings": rankings,
        "eq6_monotonicity": eq6_monotonicity,
    }


def _case_number(record: Mapping[str, Any], key: str, path: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditFailure(f"summary internal numeric field is malformed: {path}.{key}")
    value = float(value)
    if not math.isfinite(value):
        raise AuditFailure(f"summary internal numeric field is non-finite: {path}.{key}")
    return value


def _require_internal_number(
    record: Mapping[str, Any],
    key: str,
    expected: float,
    path: str,
) -> None:
    actual = _case_number(record, key, path)
    allowance = max(
        16.0 * math.ulp(expected),
        16.0 * sys.float_info.epsilon * abs(expected),
    )
    if abs(actual - expected) > allowance:
        raise AuditFailure(
            f"summary internal numeric mismatch at {path}.{key}: "
            f"observed={actual!r}, recomputed={expected!r}"
        )


def _require_internal_value(
    record: Mapping[str, Any],
    key: str,
    expected: Any,
    path: str,
) -> None:
    if isinstance(expected, float):
        _require_internal_number(record, key, expected, path)
    elif record.get(key) != expected:
        raise AuditFailure(f"summary internal value mismatch at {path}.{key}")


def _case_records(
    section: Mapping[str, Any],
    key: str,
    path: str,
) -> list[Mapping[str, Any]]:
    records = section.get(key)
    if not isinstance(records, list) or not records:
        raise AuditFailure(f"summary internal case list is malformed: {path}.{key}")
    if any(not isinstance(record, Mapping) for record in records):
        raise AuditFailure(f"summary internal case record is malformed: {path}.{key}")
    return records


def _validate_error_record(
    record: Mapping[str, Any],
    *,
    value_key: str,
    reference_key: str,
    tolerance: float,
    path: str,
    validate_pass: bool = True,
) -> None:
    value = _case_number(record, value_key, path)
    reference = _case_number(record, reference_key, path)
    error = abs(value - reference)
    _require_internal_number(record, "abs_error", error, path)
    _require_internal_number(
        record,
        "rel_error",
        _relative_error(value, reference),
        path,
    )
    _require_internal_number(record, "tolerance", tolerance, path)
    _require_internal_number(
        record,
        "scaled_error",
        _scaled_error(error, tolerance),
        path,
    )
    if validate_pass:
        _require_internal_value(record, "passed", error <= tolerance, path)


def _validate_aggregate(
    section: Mapping[str, Any],
    expected: Mapping[str, Any],
    path: str,
) -> None:
    aggregate = section.get("aggregate")
    if not isinstance(aggregate, Mapping) or set(aggregate) != set(expected):
        raise AuditFailure(f"summary internal aggregate is malformed: {path}")
    for key, value in expected.items():
        _require_internal_value(aggregate, key, value, f"{path}.aggregate")


def _validate_summary_internal_consistency(
    sections: Mapping[str, Mapping[str, Any]],
) -> None:
    quadrature = sections["analytical_quadrature"]
    quadrature_cases = _case_records(quadrature, "cases", "analytical_quadrature")
    for index, record in enumerate(quadrature_cases):
        path = f"analytical_quadrature.cases[{index}]"
        analytical = _case_number(record, "analytical", path)
        repeated = _case_number(record, "repeated_quadrature", path)
        observed = _case_number(record, "quadrature", path)
        tolerance = (
            TOLERANCES["quadrature_atol"]
            + TOLERANCES["quadrature_rtol"] * abs(analytical)
        )
        _validate_error_record(
            record,
            value_key="quadrature",
            reference_key="analytical",
            tolerance=tolerance,
            path=path,
            validate_pass=False,
        )
        repeat_difference = abs(repeated - observed)
        _require_internal_number(
            record,
            "repeat_abs_difference",
            repeat_difference,
            path,
        )
        expected_pass = abs(observed - analytical) <= tolerance and repeated == observed
        _require_internal_value(record, "passed", expected_pass, path)
    _validate_aggregate(
        quadrature,
        {
            "case_count": len(quadrature_cases),
            "max_abs_error": max(float(case["abs_error"]) for case in quadrature_cases),
            "max_rel_error": max(float(case["rel_error"]) for case in quadrature_cases),
            "max_scaled_error": max(
                float(case["scaled_error"]) for case in quadrature_cases
            ),
            "repeat_max_abs_difference": max(
                float(case["repeat_abs_difference"]) for case in quadrature_cases
            ),
            "all_cases_passed": all(case["passed"] is True for case in quadrature_cases),
        },
        "analytical_quadrature",
    )

    monte_carlo = sections["monte_carlo"]
    monte_carlo_cases = _case_records(monte_carlo, "cases", "monte_carlo")
    for index, record in enumerate(monte_carlo_cases):
        path = f"monte_carlo.cases[{index}]"
        standard_error = _case_number(record, "standard_error", path)
        tolerance = (
            TOLERANCES["monte_carlo_atol"]
            + TOLERANCES["monte_carlo_sigma_multiplier"] * standard_error
        )
        _validate_error_record(
            record,
            value_key="estimate",
            reference_key="analytical",
            tolerance=tolerance,
            path=path,
        )
    _validate_aggregate(
        monte_carlo,
        {
            "case_count": len(monte_carlo_cases),
            "max_abs_error": max(float(case["abs_error"]) for case in monte_carlo_cases),
            "max_rel_error": max(float(case["rel_error"]) for case in monte_carlo_cases),
            "max_scaled_error": max(
                float(case["scaled_error"]) for case in monte_carlo_cases
            ),
            "all_cases_passed": all(case["passed"] is True for case in monte_carlo_cases),
        },
        "monte_carlo",
    )

    zero_shift = sections["zero_shift"]
    zero_shift_cases = _case_records(zero_shift, "cases", "zero_shift")
    for index, record in enumerate(zero_shift_cases):
        path = f"zero_shift.cases[{index}]"
        reference = _case_number(record, "identity_value", path)
        _validate_error_record(
            record,
            value_key="analytical",
            reference_key="identity_value",
            tolerance=TOLERANCES["zero_shift_rtol"] * abs(reference),
            path=path,
        )
    _validate_aggregate(
        zero_shift,
        {
            "case_count": len(zero_shift_cases),
            "max_abs_error": max(float(case["abs_error"]) for case in zero_shift_cases),
            "max_scaled_error": max(
                float(case["scaled_error"]) for case in zero_shift_cases
            ),
            "all_cases_passed": all(case["passed"] is True for case in zero_shift_cases),
        },
        "zero_shift",
    )

    stability = sections["stability"]
    stability_cases = _case_records(stability, "cases", "stability")
    for index, record in enumerate(stability_cases):
        path = f"stability.cases[{index}]"
        value = _case_number(record, "analytical", path)
        upper_bound = _case_number(record, "second_moment_upper_bound", path)
        tolerance = TOLERANCES["stability_upper_bound_rtol"] * abs(upper_bound)
        excess = max(value - upper_bound, 0.0)
        finite = math.isfinite(value)
        nonnegative = value >= 0.0
        bounded = excess <= tolerance
        _require_internal_number(record, "upper_bound_excess", excess, path)
        _require_internal_number(record, "upper_bound_tolerance", tolerance, path)
        _require_internal_number(
            record,
            "upper_bound_scaled_excess",
            _scaled_error(excess, tolerance),
            path,
        )
        _require_internal_value(record, "finite", finite, path)
        _require_internal_value(record, "nonnegative", nonnegative, path)
        _require_internal_value(
            record,
            "bounded_by_ungated_second_moment",
            bounded,
            path,
        )
        _require_internal_value(record, "passed", finite and nonnegative and bounded, path)

    complement_cases = _case_records(
        stability,
        "complement_identity_cases",
        "stability",
    )
    for index, record in enumerate(complement_cases):
        path = f"stability.complement_identity_cases[{index}]"
        reference = _case_number(record, "identity_value", path)
        positive = _case_number(record, "positive_shift_kernel", path)
        negative = _case_number(record, "negative_shift_kernel", path)
        _require_internal_number(record, "sum", positive + negative, path)
        _validate_error_record(
            record,
            value_key="sum",
            reference_key="identity_value",
            tolerance=(
                TOLERANCES["complement_atol"]
                + TOLERANCES["complement_rtol"] * abs(reference)
            ),
            path=path,
        )

    near_degenerate_cases = _case_records(
        stability,
        "near_degenerate_cases",
        "stability",
    )
    for index, record in enumerate(near_degenerate_cases):
        path = f"stability.near_degenerate_cases[{index}]"
        gamma = _case_number(record, "gamma", path)
        reference = _case_number(record, "deterministic_limit", path)
        _validate_error_record(
            record,
            value_key="analytical",
            reference_key="deterministic_limit",
            tolerance=(
                gamma * gamma
                + TOLERANCES["near_degenerate_rtol"] * abs(reference)
            ),
            path=path,
        )
    _validate_aggregate(
        stability,
        {
            "case_count": len(stability_cases),
            "nonfinite_count": sum(not case["finite"] for case in stability_cases),
            "negative_count": sum(not case["nonnegative"] for case in stability_cases),
            "upper_bound_violation_count": sum(
                not case["bounded_by_ungated_second_moment"]
                for case in stability_cases
            ),
            "max_complement_scaled_error": max(
                float(case["scaled_error"]) for case in complement_cases
            ),
            "max_near_degenerate_scaled_error": max(
                float(case["scaled_error"]) for case in near_degenerate_cases
            ),
            "all_cases_passed": (
                all(case["passed"] is True for case in stability_cases)
                and all(case["passed"] is True for case in complement_cases)
                and all(case["passed"] is True for case in near_degenerate_cases)
            ),
        },
        "stability",
    )


def _validate_summary(
    payload: Mapping[str, Any],
    *,
    recomputed_sections: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Validate against regenerated sections or trusted in-run section data."""
    if not isinstance(payload, Mapping):
        raise AuditFailure("summary must be a mapping")
    expected_top_level = {
        "schema_version",
        "audit_id",
        "status",
        "passed",
        "paper",
        "config",
        "config_sha256",
        "method_origins",
        "reproduction_scope",
        "checks",
        "analytical_quadrature",
        "monte_carlo",
        "zero_shift",
        "stability",
        "rankings",
        "eq6_monotonicity",
    }
    if set(payload) != expected_top_level:
        raise AuditFailure("summary top-level schema differs from the audit contract")
    if payload.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise AuditFailure("summary schema_version mismatch")
    if payload.get("audit_id") != AUDIT_ID:
        raise AuditFailure("summary audit_id mismatch")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise AuditFailure("summary does not declare a passing audit")
    paper = payload.get("paper")
    if not isinstance(paper, Mapping) or paper.get("pdf_sha256") != PAPER_SHA256:
        raise AuditFailure("summary paper identity mismatch")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise AuditFailure("summary config is missing or malformed")
    if payload.get("config_sha256") != _sha256_json(config):
        raise AuditFailure("summary config hash mismatch")
    expected_config = _expected_config_from_payload(config)
    if _canonical_json_bytes(config) != _canonical_json_bytes(expected_config):
        raise AuditFailure("summary config differs from the executable audit contract")
    if payload.get("method_origins") != METHOD_ORIGINS:
        raise AuditFailure("summary method-origin registry mismatch")
    scope = payload.get("reproduction_scope")
    if not isinstance(scope, Mapping) or scope.get("paper_experiment_reproduced") is not False:
        raise AuditFailure("summary reproduction scope is missing or overclaims")

    section_names = (
        "analytical_quadrature",
        "monte_carlo",
        "zero_shift",
        "stability",
        "rankings",
        "eq6_monotonicity",
    )
    sections: dict[str, Mapping[str, Any]] = {}
    for name in section_names:
        section = payload.get(name)
        if not isinstance(section, Mapping):
            raise AuditFailure(f"summary section is missing or malformed: {name}")
        sections[name] = section

    _validate_summary_internal_consistency(sections)

    regenerated = (
        _regenerate_summary_sections(config)
        if recomputed_sections is None
        else dict(recomputed_sections)
    )
    if set(regenerated) != set(section_names):
        raise AuditFailure("recomputed summary section registry mismatch")
    for name in section_names:
        _compare_semantic_payload(sections[name], regenerated[name], name)

    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise AuditFailure("summary checks are missing or malformed")
    try:
        recomputed = build_checks(
            sections["analytical_quadrature"],
            sections["monte_carlo"],
            sections["zero_shift"],
            sections["stability"],
            sections["rankings"],
            sections["eq6_monotonicity"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AuditFailure(f"summary section aggregate is malformed: {error}") from error
    if _canonical_json_bytes(checks) != _canonical_json_bytes(recomputed):
        raise AuditFailure("summary checks do not match detailed section aggregates")
    enforce_checks(checks)

    detailed_case_groups = (
        sections["analytical_quadrature"].get("cases"),
        sections["monte_carlo"].get("cases"),
        sections["zero_shift"].get("cases"),
        sections["stability"].get("cases"),
        sections["stability"].get("complement_identity_cases"),
        sections["stability"].get("near_degenerate_cases"),
        sections["eq6_monotonicity"].get("cases"),
    )
    for group in detailed_case_groups:
        if not isinstance(group, list) or not group:
            raise AuditFailure("summary contains an empty or malformed case group")
        if any(not isinstance(case, Mapping) or case.get("passed") is not True for case in group):
            raise AuditFailure("summary contains a failing or malformed detailed case")


def validate_summary(payload: Mapping[str, Any]) -> None:
    """Public fail-closed validator that always regenerates expected sections."""
    _validate_summary(payload)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _display_path(path: Path) -> str:
    absolute = path.resolve()
    try:
        return absolute.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(absolute)


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _environment_details() -> dict[str, Any]:
    dirty_output = _git_value("status", "--porcelain", "--untracked-files=all")
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "byteorder": sys.byteorder,
        "float64_epsilon": sys.float_info.epsilon,
        "cpu_count": os.cpu_count(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree_dirty": dirty_output not in ("", "unavailable"),
    }


def _code_identity() -> dict[str, Any]:
    commit = _git_value("rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return {
            "source_snapshot_commit": None,
            "source_snapshot_state": "source_archive_with_individually_hashed_sources",
            "source_files_match_commit": False,
        }
    source_status = _git_value(
        "status",
        "--porcelain",
        "--",
        *SOURCE_RELATIVE_PATHS,
    )
    source_files_match_commit = source_status == ""
    return {
        "source_snapshot_commit": commit,
        "source_snapshot_state": (
            "exact_commit"
            if source_files_match_commit
            else "working_tree_with_individually_hashed_sources"
        ),
        "source_files_match_commit": source_files_match_commit,
    }


def _load_paper_identity(
    paper_pdf: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with PAPER_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        paper_manifest = json.load(handle)
    if paper_manifest.get("pdf_sha256") != PAPER_SHA256:
        raise AuditFailure("paper/manifest.json does not contain the required v1 checksum")
    if paper_manifest.get("version") != "v1":
        raise AuditFailure("paper/manifest.json is not pinned to v1")

    local_pdf: dict[str, Any]
    if paper_pdf is None:
        local_pdf = {
            "path": None,
            "status": "not_available",
            "sha256": None,
        }
    else:
        if not paper_pdf.is_file():
            raise AuditFailure(
                f"explicit paper PDF is not a regular file: {paper_pdf}"
            )
        actual_pdf_hash = _sha256_path(paper_pdf)
        if actual_pdf_hash != PAPER_SHA256:
            raise AuditFailure(
                f"paper PDF checksum mismatch: expected {PAPER_SHA256}, got {actual_pdf_hash}"
            )
        local_pdf = {
            "path": str(paper_pdf.resolve()),
            "status": "verified",
            "sha256": actual_pdf_hash,
        }
    identity = {
        "arxiv_id": paper_manifest["arxiv_id"],
        "version": paper_manifest["version"],
        "title": paper_manifest["title"],
        "pdf_filename": paper_manifest["pdf_filename"],
        "pdf_sha256": PAPER_SHA256,
        "anchors": [
            "PDF page 9, Eq. 3",
            "PDF page 13, Eq. 6",
            "PDF page 52, Eqs. 73-75 and Eqs. 77-78",
            "PDF page 53, Eq. 79 and Theorem E.2",
        ],
        "local_pdf": local_pdf,
    }
    input_records = [
        {
            "kind": "paper_manifest",
            "path": _display_path(PAPER_MANIFEST_PATH),
            "sha256": _sha256_path(PAPER_MANIFEST_PATH),
        }
    ]
    if local_pdf["status"] == "verified":
        input_records.append(
            {
                "kind": "source_paper_pdf",
                "path": local_pdf["path"],
                "sha256": local_pdf["sha256"],
            }
        )
    return identity, input_records


def _configuration(
    *,
    seed: int,
    mc_samples: int,
    mc_chunk_size: int,
    quadrature_order: int,
    quadrature_z_limit: float,
) -> dict[str, Any]:
    return {
        "dtype": "float64",
        "seed": int(seed),
        "monte_carlo_samples": int(mc_samples),
        "monte_carlo_chunk_size": int(mc_chunk_size),
        "quadrature": {
            "order": int(quadrature_order),
            "z_limit": float(quadrature_z_limit),
            "ratios": list(QUADRATURE_RATIOS),
            "signed_gamma_scales": list(QUADRATURE_SIGNED_SCALES),
        },
        "zero_shift_gammas": list(ZERO_SHIFT_GAMMAS),
        "monte_carlo_cases": [dict(record) for record in MONTE_CARLO_CASES],
        "stability_cases": [dict(record) for record in STABILITY_CASES],
        "complement_cases": [dict(record) for record in COMPLEMENT_CASES],
        "ranking_neurons": [dict(record) for record in RANKING_NEURONS],
        "eq6": {
            "layer_width_N": EQ6_LAYER_WIDTH,
            "pre_action_layer_capacity_E_a": EQ6_LAYER_CAPACITY,
            "capacity_grid": list(EQ6_CAPACITY_GRID),
        },
        "negative_tail_switch": NEGATIVE_TAIL_SWITCH,
        "tolerances": dict(TOLERANCES),
    }


def _source_hashes() -> list[dict[str, Any]]:
    source_paths = tuple(ROOT / path for path in SOURCE_RELATIVE_PATHS)
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise AuditFailure(f"audit source file missing: {missing}")
    return [
        {
            "path": _display_path(path),
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
        }
        for path in source_paths
    ]


def _fixture_input_hashes(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = {
        "quadrature_grid": config["quadrature"],
        "zero_shift_grid": config["zero_shift_gammas"],
        "monte_carlo_case_grid": config["monte_carlo_cases"],
        "stability_case_grid": {
            "stability": config["stability_cases"],
            "complement": config["complement_cases"],
        },
        "ranking_fixture": config["ranking_neurons"],
        "eq6_grid": config["eq6"],
    }
    return [
        {
            "kind": "embedded_deterministic_input",
            "name": name,
            "sha256": _sha256_json(payload),
        }
        for name, payload in records.items()
    ]


def run_audit(
    *,
    result_dir: Path = DEFAULT_RESULT_DIR,
    manifest_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    mc_samples: int = DEFAULT_MC_SAMPLES,
    mc_chunk_size: int = DEFAULT_MC_CHUNK_SIZE,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    quadrature_z_limit: float = DEFAULT_QUADRATURE_Z_LIMIT,
    paper_pdf: Path | None = None,
    invocation: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute the complete audit, write artifacts, and fail on any check."""
    result_dir = result_dir.resolve()
    manifest_path = (
        (result_dir / "manifest.json")
        if manifest_path is None
        else manifest_path.resolve()
    )
    if manifest_path in (result_dir / "summary.json", result_dir / "metrics.csv"):
        raise ValueError("manifest_path must not replace summary.json or metrics.csv")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    paper, paper_inputs = _load_paper_identity(paper_pdf)
    config = _configuration(
        seed=seed,
        mc_samples=mc_samples,
        mc_chunk_size=mc_chunk_size,
        quadrature_order=quadrature_order,
        quadrature_z_limit=quadrature_z_limit,
    )
    config_sha256 = _sha256_json(config)

    quadrature, quadrature_rows = build_quadrature_audit(
        quadrature_order, quadrature_z_limit
    )
    monte_carlo, monte_carlo_rows = build_monte_carlo_audit(
        seed, mc_samples, mc_chunk_size
    )
    zero_shift, zero_shift_rows = build_zero_shift_audit()
    stability, stability_rows = build_stability_audit()
    rankings, ranking_rows = build_ranking_audit()
    eq6, eq6_rows = build_eq6_audit()
    rows = (
        quadrature_rows
        + monte_carlo_rows
        + zero_shift_rows
        + stability_rows
        + ranking_rows
        + eq6_rows
    )
    checks = build_checks(
        quadrature, monte_carlo, zero_shift, stability, rankings, eq6
    )
    passed = all(record["passed"] for record in checks.values())
    status = "passed" if passed else "failed"

    reproduction_scope = {
        "claim": "independent audit of the closed-form biased ReLU self-kernel and its same-layer Eq. 6 ordering consequence",
        "includes": [
            "HOPE Eq. 3 and Eq. 79 self-kernel evaluation",
            "zero-shift and gamma-equals-zero special cases",
            "fixed-node quadrature",
            "seeded Monte Carlo secondary checks",
            "synthetic local-score ranking comparisons",
            "Eq. 6 monotonicity for fixed N and fixed pre-action layer capacity E_a",
        ],
        "excludes": [
            "cross-neuron kernels",
            "global positive-semidefiniteness",
            "maximum-entropy construction",
            "cross-layer pruning selection",
            "training, checkpoint, dataset, or headline paper experiment reproduction",
        ],
        "paper_experiment_reproduced": False,
        "ranking_data": "simulated",
        "finite_epsilon_scope": "not evaluated; this audit uses the Eq. 78 negligible-epsilon marginal gamma^2",
    }

    summary: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": status,
        "passed": passed,
        "paper": paper,
        "config": config,
        "config_sha256": config_sha256,
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": reproduction_scope,
        "checks": checks,
        "analytical_quadrature": quadrature,
        "monte_carlo": monte_carlo,
        "zero_shift": zero_shift,
        "stability": stability,
        "rankings": rankings,
        "eq6_monotonicity": eq6,
    }

    metrics_path = result_dir / "metrics.csv"
    summary_path = result_dir / "summary.json"
    _write_csv(metrics_path, rows)
    _write_json(summary_path, summary)

    outputs = [
        {
            "path": _display_path(path),
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
            "media_type": media_type,
        }
        for path, media_type in (
            (metrics_path, "text/csv"),
            (summary_path, "application/json"),
        )
    ]
    expected_ranges = {
        name: {
            "minimum": record["minimum"],
            "maximum": record["maximum"],
            "units": record["units"],
        }
        for name, record in _expected_check_ranges().items()
    }
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper": paper,
        "config": config,
        "config_sha256": config_sha256,
        "seeds": {
            "monte_carlo": seed,
            "deterministic_methods": "no random seed used",
        },
        "code": _code_identity(),
        "environment": _environment_details(),
        "expected_metric_ranges": expected_ranges,
        "observed_checks": checks,
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": reproduction_scope,
        "source_hashes": _source_hashes(),
        "input_hashes": [
            *paper_inputs,
            *_fixture_input_hashes(config),
        ],
        "outputs": outputs,
        "manifest_hash_scope": "manifest excludes its own hash to avoid self-reference",
        "invocation": list(invocation or ("python-api", "run_audit")),
        "working_directory": _display_path(Path.cwd()),
    }
    _write_json(manifest_path, manifest)

    _validate_summary(
        summary,
        recomputed_sections={
            "analytical_quadrature": quadrature,
            "monte_carlo": monte_carlo,
            "zero_shift": zero_shift,
            "stability": stability,
            "rankings": rankings,
            "eq6_monotonicity": eq6,
        },
    )
    return {
        "audit_id": AUDIT_ID,
        "status": status,
        "passed": passed,
        "config_sha256": config_sha256,
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "checks": checks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory for metrics.csv and summary.json.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Manifest destination; defaults to RESULT_DIR/manifest.json.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--mc-chunk-size", type=int, default=DEFAULT_MC_CHUNK_SIZE)
    parser.add_argument(
        "--quadrature-order", type=int, default=DEFAULT_QUADRATURE_ORDER
    )
    parser.add_argument(
        "--quadrature-z-limit", type=float, default=DEFAULT_QUADRATURE_Z_LIMIT
    )
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        default=None,
        help="Optional local v1 PDF to checksum when explicitly supplied.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    invocation = [
        "python3",
        _display_path(Path(__file__).resolve()),
        *(argv or sys.argv[1:]),
    ]
    result = run_audit(
        result_dir=args.result_dir,
        manifest_path=args.manifest_path,
        seed=args.seed,
        mc_samples=args.mc_samples,
        mc_chunk_size=args.mc_chunk_size,
        quadrature_order=args.quadrature_order,
        quadrature_z_limit=args.quadrature_z_limit,
        paper_pdf=args.paper_pdf,
        invocation=invocation,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
