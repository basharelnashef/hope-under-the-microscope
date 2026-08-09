"""Paper-literal covariance formulas and independent rank diagnostics.

HOPE v1 Section 4 prints a pure Gaussian maximum-entropy problem with
projection-variance constraints. Appendix E.3.1 prints a different precision
shape containing an identity term. This module exposes both printed objects
without adding a reference-prior scale or silently repairing either one.

Directions are columns of a matrix with shape ``(dimension, constraints)``.
All public numerical functions require finite float64 inputs and reject
singular covariance matrices where differential entropy is undefined.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


FLOAT64_EPSILON = np.finfo(np.float64).eps


def _finite_array(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if 0 in array.shape:
        raise ValueError(f"{name} must not have an empty dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite float64 values")
    return array


def _directions(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    return _finite_array(value, name="directions", ndim=2)


def _variances(
    value: Sequence[float] | np.ndarray,
    count: int,
    *,
    strictly_positive: bool = False,
) -> np.ndarray:
    variances = _finite_array(value, name="target_variances", ndim=1)
    if variances.shape != (count,):
        raise ValueError(
            "target_variances length must equal the number of directions"
        )
    if strictly_positive:
        if np.any(variances <= 0.0):
            raise ValueError("target_variances must be strictly positive")
    elif np.any(variances < 0.0):
        raise ValueError("target_variances must be nonnegative")
    return variances


def _covariance(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    dimension: int | None = None,
) -> np.ndarray:
    covariance = _finite_array(value, name="covariance", ndim=2)
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if dimension is not None and covariance.shape != (dimension, dimension):
        raise ValueError("covariance dimension does not match directions")
    symmetry_tolerance = (
        64.0
        * FLOAT64_EPSILON
        * max(1.0, float(np.max(np.abs(covariance))))
    )
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=0.0,
        atol=symmetry_tolerance,
    ):
        raise ValueError("covariance must be symmetric")
    return 0.5 * (covariance + covariance.T)


def numerical_rank(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    tolerance: float | None = None,
) -> tuple[int, float, np.ndarray]:
    """Return float64 SVD rank, the threshold, and singular values."""
    array = _finite_array(matrix, name="matrix", ndim=2)
    singular_values = np.linalg.svd(array, compute_uv=False)
    if not np.all(np.isfinite(singular_values)):
        raise FloatingPointError("singular-value computation was non-finite")
    if tolerance is None:
        threshold = (
            max(array.shape)
            * FLOAT64_EPSILON
            * float(singular_values[0])
        )
    else:
        threshold = float(tolerance)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("rank tolerance must be finite and nonnegative")
    rank = int(np.count_nonzero(singular_values > threshold))
    return rank, threshold, singular_values


def canonical_null_direction(
    directions: Sequence[Sequence[float]] | np.ndarray,
    *,
    rank_tolerance: float | None = None,
) -> np.ndarray:
    """Return one deterministic unit vector orthogonal to all columns."""
    matrix = _directions(directions)
    left_vectors, singular_values, _ = np.linalg.svd(
        matrix,
        full_matrices=True,
    )
    if rank_tolerance is None:
        leading = float(singular_values[0]) if singular_values.size else 0.0
        threshold = max(matrix.shape) * FLOAT64_EPSILON * leading
    else:
        threshold = float(rank_tolerance)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("rank tolerance must be finite and nonnegative")
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank >= matrix.shape[0]:
        raise ValueError("directions span the ambient space")
    vector = np.asarray(left_vectors[:, rank], dtype=np.float64)
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("null direction normalization failed")
    vector = vector / norm
    if not np.all(np.isfinite(vector)):
        raise FloatingPointError("null direction was non-finite")
    return vector


def positive_definite_logdet(
    covariance: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Return ``log(det(covariance))`` and reject non-SPD inputs."""
    matrix = _covariance(covariance)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("covariance eigenvalues were non-finite")
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("covariance must be positive definite")
    sign, value = np.linalg.slogdet(matrix)
    if sign != 1.0 or not math.isfinite(float(value)):
        raise FloatingPointError("positive-definite log determinant failed")
    return float(value)


def projection_variances(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    directions: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Evaluate every printed constraint ``w_i.T Sigma w_i``."""
    matrix = _directions(directions)
    sigma = _covariance(covariance, dimension=matrix.shape[0])
    values = np.einsum("di,de,ei->i", matrix, sigma, matrix)
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("projection variances were non-finite")
    return values


def maximum_constraint_residual(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    directions: Sequence[Sequence[float]] | np.ndarray,
    target_variances: Sequence[float] | np.ndarray,
) -> float:
    matrix = _directions(directions)
    targets = _variances(target_variances, matrix.shape[1])
    residual = np.max(np.abs(projection_variances(covariance, matrix) - targets))
    if not math.isfinite(float(residual)):
        raise FloatingPointError("constraint residual was non-finite")
    return float(residual)


def pure_stationarity_precision(
    directions: Sequence[Sequence[float]] | np.ndarray,
    multipliers: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the identity-free precision shape printed in Section 4."""
    matrix = _directions(directions)
    weights = _finite_array(multipliers, name="multipliers", ndim=1)
    if weights.shape != (matrix.shape[1],):
        raise ValueError("multipliers length must equal the direction count")
    precision = (matrix * weights[np.newaxis, :]) @ matrix.T
    if not np.all(np.isfinite(precision)):
        raise FloatingPointError("paper-literal precision was non-finite")
    return 0.5 * (precision + precision.T)


def appendix_identity_precision(
    directions: Sequence[Sequence[float]] | np.ndarray,
    multipliers: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the unit-identity precision printed in Appendix E.3.1."""
    matrix = _directions(directions)
    return np.eye(matrix.shape[0], dtype=np.float64) + pure_stationarity_precision(
        matrix,
        multipliers,
    )


def escape_covariance(
    base_covariance: Sequence[Sequence[float]] | np.ndarray,
    null_direction: Sequence[float] | np.ndarray,
    amount: float,
) -> np.ndarray:
    """Construct ``Sigma_t = Sigma_0 + t z z.T`` for a supplied unit ``z``."""
    base = _covariance(base_covariance)
    positive_definite_logdet(base)
    vector = _finite_array(null_direction, name="null_direction", ndim=1)
    if vector.shape != (base.shape[0],):
        raise ValueError("null_direction dimension does not match covariance")
    norm = float(np.linalg.norm(vector))
    unit_tolerance = 64.0 * FLOAT64_EPSILON
    if not math.isfinite(norm) or abs(norm - 1.0) > unit_tolerance:
        raise ValueError("null_direction must have unit Euclidean norm")
    scalar = float(amount)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError("escape amount must be finite and nonnegative")
    result = base + scalar * np.outer(vector, vector)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("escape covariance was non-finite")
    positive_definite_logdet(result)
    return result


def determinant_lemma_logdet(
    base_covariance: Sequence[Sequence[float]] | np.ndarray,
    null_direction: Sequence[float] | np.ndarray,
    amount: float,
) -> float:
    """Evaluate the exact rank-one determinant-lemma prediction."""
    base = _covariance(base_covariance)
    base_logdet = positive_definite_logdet(base)
    vector = _finite_array(null_direction, name="null_direction", ndim=1)
    if vector.shape != (base.shape[0],):
        raise ValueError("null_direction dimension does not match covariance")
    scalar = float(amount)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError("escape amount must be finite and nonnegative")
    quadratic = float(vector @ np.linalg.solve(base, vector))
    if not math.isfinite(quadratic) or quadratic <= 0.0:
        raise FloatingPointError("determinant-lemma quadratic was not positive")
    value = base_logdet + math.log1p(scalar * quadratic)
    if not math.isfinite(value):
        raise FloatingPointError("determinant-lemma log determinant was non-finite")
    return value


def spanning_logdet_upper_bound(
    directions: Sequence[Sequence[float]] | np.ndarray,
    target_variances: Sequence[float] | np.ndarray,
    *,
    rank_tolerance: float | None = None,
) -> float:
    """Return the trace/AM-GM upper bound for a full-spanning constraint set."""
    matrix = _directions(directions)
    targets = _variances(target_variances, matrix.shape[1])
    rank, _, _ = numerical_rank(matrix, tolerance=rank_tolerance)
    dimension = matrix.shape[0]
    if rank != dimension:
        raise ValueError("directions must span the ambient space")
    frame = matrix @ matrix.T
    frame_logdet = positive_definite_logdet(frame)
    total = float(np.sum(targets, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("target-variance sum was not positive and finite")
    bound = dimension * math.log(total / dimension) - frame_logdet
    if not math.isfinite(bound):
        raise FloatingPointError("spanning log-determinant bound was non-finite")
    return bound


def linear_constraint_residual(
    directions: Sequence[Sequence[float]] | np.ndarray,
    target_variances: Sequence[float] | np.ndarray,
) -> float:
    """Return least-squares residual for the affine covariance equalities.

    A positive residual proves infeasibility. A zero residual alone does not
    prove that the affine solution set contains a positive-definite matrix.
    """
    matrix = _directions(directions)
    targets = _variances(target_variances, matrix.shape[1])
    dimension = matrix.shape[0]
    upper = [(row, column) for row in range(dimension) for column in range(row, dimension)]
    design = np.empty((matrix.shape[1], len(upper)), dtype=np.float64)
    for direction_index in range(matrix.shape[1]):
        vector = matrix[:, direction_index]
        for coefficient_index, (row, column) in enumerate(upper):
            coefficient = vector[row] * vector[column]
            if row != column:
                coefficient *= 2.0
            design[direction_index, coefficient_index] = coefficient
    solution, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    residual = float(np.max(np.abs(design @ solution - targets)))
    if not math.isfinite(residual):
        raise FloatingPointError("linear constraint residual was non-finite")
    return residual


def pure_pair_maximum_entropy_covariance(
    directions: Sequence[Sequence[float]] | np.ndarray,
    target_variances: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Solve pure maximum entropy for two directions spanning ``R^2``.

    With only the two diagonal projection variances fixed, Hadamard's
    inequality makes the maximum-entropy projection covariance diagonal.
    """
    matrix = _directions(directions)
    if matrix.shape != (2, 2):
        raise ValueError("pure pair fixture requires a 2 by 2 direction matrix")
    rank, _, _ = numerical_rank(matrix)
    if rank != 2:
        raise ValueError("pair directions must be linearly independent")
    targets = _variances(target_variances, 2, strictly_positive=True)
    inverse = np.linalg.inv(matrix)
    covariance = inverse.T @ np.diag(targets) @ inverse
    covariance = 0.5 * (covariance + covariance.T)
    positive_definite_logdet(covariance)
    return covariance


def appendix_pair_kappa(
    raw_correlation: float,
    std_first: float,
    std_second: float,
    norm_first: float,
    norm_second: float,
) -> float:
    """Return the unit-reference blending constant printed in Eq. 81."""
    correlation = float(raw_correlation)
    values = tuple(float(value) for value in (std_first, std_second, norm_first, norm_second))
    if not math.isfinite(correlation) or not -1.0 < correlation < 1.0:
        raise ValueError("raw_correlation must be finite and strictly inside (-1, 1)")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("standard deviations and norms must be positive and finite")
    first_std, second_std, first_norm, second_norm = values
    kappa = (
        correlation
        / (1.0 - correlation * correlation)
        * (first_std / first_norm)
        * (second_std / second_norm)
    )
    if not math.isfinite(kappa):
        raise FloatingPointError("Appendix E.3.1 kappa was non-finite")
    return kappa


def stable_correlation_root(kappa: float) -> float:
    """Solve ``q / (1 - q^2) = kappa`` without squaring overflow."""
    value = float(kappa)
    if not math.isfinite(value):
        raise ValueError("kappa must be finite")
    result = value / (0.5 + math.hypot(0.5, value))
    if not math.isfinite(result) or not -1.0 < result < 1.0:
        raise FloatingPointError("warped correlation root was invalid")
    return result


def appendix_pair_correlation(
    raw_correlation: float,
    std_first: float,
    std_second: float,
    norm_first: float,
    norm_second: float,
) -> float:
    """Return the unit-reference pair correlation printed in Eqs. 80-81."""
    return stable_correlation_root(
        appendix_pair_kappa(
            raw_correlation,
            std_first,
            std_second,
            norm_first,
            norm_second,
        )
    )
