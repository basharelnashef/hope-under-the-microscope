"""Explicit isotropic-reference relative-entropy repair for HOPE v1.

This module is not part of the paper-literal Section 4 optimization. It makes
the reference covariance ``tau^2 I`` explicit and exposes its effect on the
two-projection correlation.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from hope_repro.paper_literal.maximum_entropy import (
    appendix_pair_kappa,
    maximum_constraint_residual,
    numerical_rank,
    positive_definite_logdet,
    projection_variances,
    stable_correlation_root,
)


def _finite_matrix(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite float64 values")
    return matrix


def _positive_tau(tau: float) -> float:
    value = float(tau)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tau must be positive and finite")
    squared = value * value
    if not math.isfinite(squared) or squared == 0.0:
        raise ValueError("tau squared must be positive and representable in float64")
    return value


def gaussian_covariance_kl(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    tau: float,
) -> float:
    """Return KL(N(0, Sigma) || N(0, tau^2 I))."""
    matrix = _finite_matrix(covariance, name="covariance")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    scale = _positive_tau(tau)
    logdet = positive_definite_logdet(matrix)
    dimension = matrix.shape[0]
    trace = float(np.trace(matrix))
    value = 0.5 * (
        trace / (scale * scale)
        - logdet
        + dimension * math.log(scale * scale)
        - dimension
    )
    if not math.isfinite(value):
        raise FloatingPointError("Gaussian covariance KL was non-finite")
    if value < -256.0 * np.finfo(np.float64).eps:
        raise FloatingPointError("Gaussian covariance KL was materially negative")
    return max(value, 0.0)


def reference_precision(
    directions: Sequence[Sequence[float]] | np.ndarray,
    multipliers: Sequence[float] | np.ndarray,
    tau: float,
) -> np.ndarray:
    """Return ``tau^-2 I + W Lambda W.T`` for the proposed repair."""
    matrix = _finite_matrix(directions, name="directions")
    weights = np.asarray(multipliers, dtype=np.float64)
    if weights.ndim != 1 or weights.shape != (matrix.shape[1],):
        raise ValueError("multipliers length must equal the direction count")
    if not np.all(np.isfinite(weights)):
        raise ValueError("multipliers must contain only finite float64 values")
    scale = _positive_tau(tau)
    precision = (
        np.eye(matrix.shape[0], dtype=np.float64) / (scale * scale)
        + (matrix * weights[np.newaxis, :]) @ matrix.T
    )
    if not np.all(np.isfinite(precision)):
        raise FloatingPointError("reference precision was non-finite")
    return 0.5 * (precision + precision.T)


def reference_pair_kappa(
    raw_correlation: float,
    std_first: float,
    std_second: float,
    norm_first: float,
    norm_second: float,
    tau: float,
) -> float:
    """Generalize the Appendix blending constant to declared scale ``tau``."""
    scale = _positive_tau(tau)
    unit_kappa = appendix_pair_kappa(
        raw_correlation,
        std_first,
        std_second,
        norm_first,
        norm_second,
    )
    value = unit_kappa / (scale * scale)
    if not math.isfinite(value):
        raise FloatingPointError("reference-scale kappa was non-finite")
    return value


def reference_pair_correlation(
    raw_correlation: float,
    std_first: float,
    std_second: float,
    norm_first: float,
    norm_second: float,
    tau: float,
) -> float:
    """Return the pair correlation induced by ``N(0, tau^2 I)``."""
    return stable_correlation_root(
        reference_pair_kappa(
            raw_correlation,
            std_first,
            std_second,
            norm_first,
            norm_second,
            tau,
        )
    )


def pair_relative_entropy_solution(
    directions: Sequence[Sequence[float]] | np.ndarray,
    target_variances: Sequence[float] | np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return covariance, diagonal multipliers, and pair correlation.

    The two directions may live in an ambient dimension greater than two but
    must be linearly independent. The returned covariance is the stationary
    solution of the explicitly declared relative-entropy problem.
    """
    matrix = _finite_matrix(directions, name="directions")
    if matrix.shape[1] != 2:
        raise ValueError("pair solution requires exactly two directions")
    rank, _, _ = numerical_rank(matrix)
    if rank != 2:
        raise ValueError("pair directions must be linearly independent")
    targets = np.asarray(target_variances, dtype=np.float64)
    if targets.shape != (2,) or not np.all(np.isfinite(targets)):
        raise ValueError("target_variances must be two finite float64 values")
    if np.any(targets <= 0.0):
        raise ValueError("target_variances must be strictly positive")
    scale = _positive_tau(tau)

    gram = matrix.T @ matrix
    norms = np.sqrt(np.diag(gram))
    raw_correlation = float(gram[0, 1] / (norms[0] * norms[1]))
    standard_deviations = np.sqrt(targets)
    correlation = reference_pair_correlation(
        raw_correlation,
        float(standard_deviations[0]),
        float(standard_deviations[1]),
        float(norms[0]),
        float(norms[1]),
        scale,
    )
    output_covariance = np.asarray(
        [
            [targets[0], correlation * standard_deviations[0] * standard_deviations[1]],
            [correlation * standard_deviations[0] * standard_deviations[1], targets[1]],
        ],
        dtype=np.float64,
    )
    output_precision = np.linalg.inv(output_covariance)
    reference_output_precision = np.linalg.inv(gram) / (scale * scale)
    multiplier_matrix = output_precision - reference_output_precision
    off_diagonal = float(multiplier_matrix[0, 1])
    off_tolerance = (
        512.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(multiplier_matrix))))
    )
    if abs(off_diagonal) > off_tolerance:
        raise FloatingPointError("pair multiplier matrix was not diagonal")
    multipliers = np.diag(multiplier_matrix).astype(np.float64)
    precision = reference_precision(matrix, multipliers, scale)
    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    positive_definite_logdet(covariance)

    residual = maximum_constraint_residual(covariance, matrix, targets)
    constraint_tolerance = (
        2048.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(targets)))
    )
    if residual > constraint_tolerance:
        raise FloatingPointError("relative-entropy pair constraints were not met")
    observed = projection_variances(covariance, matrix)
    if not np.all(np.isfinite(observed)):
        raise FloatingPointError("relative-entropy output covariance was non-finite")
    return covariance, multipliers, correlation
