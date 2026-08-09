#!/usr/bin/env python3
"""Deterministic audit of pair-local assembly versus one global PSD Gram matrix.

The audit reconstructs an embedded fixed three-neuron witness. It contains no
random search and reports no empirical failure rate. The paper-literal
formulas, deterministic counterexample, positive control, and proposed repairs
are labeled separately in every output.
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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
AUDIT_ID = "part05_global_psd_audit"
AUDIT_SCHEMA_VERSION = 1
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_05" / "manual"

# These six-decimal values are the exact audit inputs, not rounded output.
WITNESS_R = (
    (1.0, -0.173508, 0.590994),
    (-0.173508, 1.0, 0.513747),
    (0.590994, 0.513747, 1.0),
)
WITNESS_SCALE_RATIOS = (1.106838, 1.793881, 1.634949)

TOLERANCES = {
    "symmetry_atol": 2e-15,
    "unit_diagonal_atol": 2e-15,
    "domain_atol": 2e-15,
    "positive_definite_minimum": 1e-12,
    "positive_semidefinite_atol": 2e-12,
    "equation_residual_atol": 2e-14,
    "quadratic_form_atol": 2e-14,
    "distance_atol": 2e-15,
    "raw_pd_material_minimum": 0.12,
    "warped_negative_maximum": -0.20,
    "eq85_negative_maximum": -0.04,
    "shared_control_pd_minimum": 0.19,
}

CLAIM_IDS = (
    "P05-COMMON-PX-KERNEL",
    "P05-COMMON-GRAM-PSD",
    "P05-PAIR-LOCAL-SURROGATE",
    "P05-LOCAL-PAIR-VALIDITY",
    "P05-GLOBAL-GRAM-COUNTEREXAMPLE",
    "P05-SHARED-SURROGATE-CONTROL",
    "P05-PSD-PROJECTION-REPAIR",
)

METHOD_ORIGINS = {
    "eq80_eq81_pair_warp": {
        "evidence_label": "paper_claim",
        "origin": "HOPE v1 Proposition E.3 and Eqs. 80-81",
        "scope": "applied independently to each declared pair",
    },
    "normalized_eq85_relu": {
        "evidence_label": "paper_claim",
        "origin": "HOPE v1 normalized zero-bias Eq. 85",
        "scope": "applied entrywise to each pair correlation",
    },
    "fixed_global_psd_witness": {
        "evidence_label": "counterexample",
        "origin": "embedded fixed six-decimal three-neuron witness",
        "scope": "one deterministic three-neuron analytical fixture",
    },
    "shared_eq85_control": {
        "evidence_label": "verified_fact",
        "origin": "independent positive control using one valid shared correlation",
        "scope": "Eq. 85 is applied directly to the jointly valid input R",
    },
    "plain_eigenvalue_clipping": {
        "evidence_label": "proposed_repair",
        "origin": "independent spectral projection diagnostic",
        "scope": "negative eigenvalues are set to zero; diagonal is not preserved",
    },
    "psd_renormalized_correlation": {
        "evidence_label": "proposed_repair",
        "origin": "independent clipping followed by positive diagonal congruence",
        "scope": "PSD and unit diagonal; no correlation-matrix optimality claim",
    },
}

REPRODUCTION_SCOPE = {
    "reproduction_scope": "independent_mathematical_audit",
    "analysis_role": "global_psd_counterexample_and_repair_diagnostic",
    "paper_experiment_reproduced": False,
    "checkpoint_or_dataset_used": False,
    "random_search_used": False,
    "random_failure_rate_reported": False,
    "includes": [
        "exact Eqs. 80-81 pairwise warp",
        "normalized zero-bias Eq. 85 map",
        "fixed three-neuron global PSD counterexample",
        "negative-direction quadratic-form certificates",
        "pairwise squared-distance and triangle diagnostics",
        "shared-surrogate positive control",
        "plain clipping and PSD-renormalized correlation diagnostics",
    ],
    "excludes": [
        "trained-network incidence",
        "checkpoint or dataset behavior",
        "paper compression metrics",
        "random failure-rate estimation",
        "downstream repair utility",
    ],
    "claim_boundary": (
        "The fixed witness refutes a universal global-PSD guarantee for the "
        "tested pair-local assembly; it does not estimate incidence in trained layers."
    ),
    "repair_boundary": (
        "Both PSD repairs are independent diagnostics and are not methods reported by "
        "HOPE v1."
    ),
}

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part05_global_psd/README.md",
    "experiments/audits/part05_global_psd/run.py",
    "experiments/audits/part05_global_psd/test_run.py",
)

MATRIX_FIELDS = (
    "matrix_name",
    "row",
    "column",
    "value",
    "evidence_label",
)
SPECTRA_FIELDS = (
    "matrix_name",
    "eigenvalue_index",
    "eigenvalue",
    "classification",
    "evidence_label",
)
PAIR_FIELDS = (
    "pair_id",
    "index_i",
    "index_j",
    "raw_correlation",
    "scale_ratio_i",
    "scale_ratio_j",
    "eq81_kappa",
    "eq80_warped_correlation",
    "eq80_equation_residual",
    "warped_local_min_eigenvalue",
    "eq85_kernel_value",
    "eq85_local_min_eigenvalue",
    "warped_squared_distance",
    "eq85_squared_distance",
    "passed",
)
NEGATIVE_DIRECTION_FIELDS = (
    "matrix_name",
    "component_index",
    "component_value",
    "vector_norm",
    "minimum_eigenvalue",
    "quadratic_form",
    "absolute_match_error",
    "passed",
)
PROJECTION_FIELDS = (
    "source_matrix",
    "repaired_matrix",
    "method",
    "evidence_label",
    "minimum_eigenvalue_before",
    "minimum_eigenvalue_after",
    "frobenius_change",
    "maximum_absolute_change",
    "maximum_diagonal_deviation",
    "minimum_pairwise_squared_distance",
    "psd_passed",
    "unit_diagonal_passed",
    "note",
)

CSV_CONTRACTS = {
    "witness_matrices.csv": MATRIX_FIELDS,
    "spectra.csv": SPECTRA_FIELDS,
    "pair_checks.csv": PAIR_FIELDS,
    "negative_direction.csv": NEGATIVE_DIRECTION_FIELDS,
    "projection_comparison.csv": PROJECTION_FIELDS,
}

# Exactly 22 independent range contracts are intentionally public. The
# manifest copies these ranges and the observed values.
CHECK_SPECS = {
    "witness_nonfinite_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "raw_symmetry_residual": {"minimum": 0.0, "maximum": 2e-15, "units": "absolute"},
    "raw_unit_diagonal_residual": {"minimum": 0.0, "maximum": 2e-15, "units": "absolute"},
    "raw_maximum_pair_absolute_correlation": {"minimum": 0.0, "maximum": 0.999999999999, "units": "correlation"},
    "minimum_scale_ratio": {"minimum": 1e-12, "maximum": 100.0, "units": "ratio"},
    "raw_minimum_eigenvalue": {"minimum": 0.12, "maximum": 0.13, "units": "eigenvalue"},
    "local_pair_count": {"minimum": 3.0, "maximum": 3.0, "units": "count"},
    "minimum_local_pair_eigenvalue": {"minimum": 0.18, "maximum": 1.0, "units": "eigenvalue"},
    "maximum_eq80_equation_residual": {"minimum": 0.0, "maximum": 2e-14, "units": "absolute"},
    "warped_finite_symmetry_residual": {"minimum": 0.0, "maximum": 2e-15, "units": "absolute"},
    "warped_domain_diagonal_violation": {"minimum": 0.0, "maximum": 2e-15, "units": "absolute"},
    "warped_minimum_eigenvalue": {"minimum": -0.26, "maximum": -0.20, "units": "eigenvalue"},
    "warped_negative_direction_error": {"minimum": 0.0, "maximum": 2e-14, "units": "absolute"},
    "eq85_finite_symmetry_residual": {"minimum": 0.0, "maximum": 2e-15, "units": "absolute"},
    "eq85_domain_diagonal_violation": {"minimum": 0.0, "maximum": 2e-15, "units": "absolute"},
    "eq85_minimum_eigenvalue": {"minimum": -0.05, "maximum": -0.04, "units": "eigenvalue"},
    "eq85_negative_direction_error": {"minimum": 0.0, "maximum": 2e-14, "units": "absolute"},
    "minimum_unrepaired_pairwise_squared_distance": {"minimum": 0.36, "maximum": 10.0, "units": "squared_distance"},
    "shared_control_minimum_eigenvalue": {"minimum": 0.19, "maximum": 0.21, "units": "eigenvalue"},
    "plain_clipping_maximum_psd_violation": {"minimum": 0.0, "maximum": 2e-12, "units": "absolute"},
    "renormalized_maximum_psd_or_diagonal_violation": {"minimum": 0.0, "maximum": 2e-12, "units": "absolute"},
    "eq85_maximum_triangle_violation": {"minimum": 0.015, "maximum": 0.0152, "units": "distance"},
}


class AuditFailure(RuntimeError):
    """Raised when an audit invariant or artifact contract fails."""


def _as_finite_square_matrix(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to float64") from error
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix of dimension at least two")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite entries")
    return matrix


def _symmetry_residual(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix - matrix.T)))


def _validate_symmetric(matrix: np.ndarray, name: str) -> None:
    residual = _symmetry_residual(matrix)
    if residual > TOLERANCES["symmetry_atol"]:
        raise ValueError(f"{name} is not symmetric within the declared tolerance")


def _validate_correlation(matrix: np.ndarray, name: str, *, require_psd: bool) -> None:
    _validate_symmetric(matrix, name)
    diagonal_residual = float(np.max(np.abs(np.diag(matrix) - 1.0)))
    if diagonal_residual > TOLERANCES["unit_diagonal_atol"]:
        raise ValueError(f"{name} does not have a unit diagonal")
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    if np.any(off_diagonal < -1.0 - TOLERANCES["domain_atol"]) or np.any(
        off_diagonal > 1.0 + TOLERANCES["domain_atol"]
    ):
        raise ValueError(f"{name} has an entry outside the correlation domain")
    if require_psd:
        minimum = float(np.linalg.eigvalsh(matrix)[0])
        if minimum < -TOLERANCES["positive_semidefinite_atol"]:
            raise ValueError(f"{name} is not positive semidefinite")


def eq81_kappa(raw_correlation: float, scale_ratio_i: float, scale_ratio_j: float) -> float:
    """Return Eq. 81: rho / (1-rho^2) times the two positive scale ratios."""
    values = tuple(float(value) for value in (raw_correlation, scale_ratio_i, scale_ratio_j))
    rho, ratio_i, ratio_j = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Eq. 81 inputs must be finite")
    if not -1.0 < rho < 1.0:
        raise ValueError("Eq. 81 raw correlation must lie strictly inside (-1, 1)")
    if ratio_i <= 0.0 or ratio_j <= 0.0:
        raise ValueError("Eq. 81 scale ratios must be strictly positive")
    result = rho / (1.0 - rho * rho) * ratio_i * ratio_j
    if not math.isfinite(result):
        raise FloatingPointError("Eq. 81 produced a non-finite kappa")
    return result


def eq80_warped_correlation(kappa: float) -> float:
    """Return Eq. 80: 2*kappa / (1 + sqrt(1 + 4*kappa^2)).

    The second algebraic branch avoids overflow for extreme finite inputs. It
    is the same positive-denominator root, not a different approximation.
    """
    value = float(kappa)
    if not math.isfinite(value):
        raise ValueError("Eq. 80 kappa must be finite")
    safe_limit = 0.5 * math.sqrt(np.finfo(np.float64).max - 1.0)
    if abs(value) <= safe_limit:
        result = 2.0 * value / (1.0 + math.sqrt(1.0 + 4.0 * value * value))
    else:
        result = value / (0.5 + math.hypot(0.5, value))
    if not math.isfinite(result) or not -1.0 < result < 1.0:
        raise FloatingPointError("Eq. 80 produced an invalid warped correlation")
    return result


def normalized_eq85_relu_kernel(correlation: float) -> float:
    """Return normalized zero-bias Eq. 85 on the closed domain [-1, 1]."""
    value = float(correlation)
    if not math.isfinite(value):
        raise ValueError("Eq. 85 correlation must be finite")
    if value < -1.0 - TOLERANCES["domain_atol"] or value > 1.0 + TOLERANCES["domain_atol"]:
        raise ValueError("Eq. 85 correlation must lie in [-1, 1]")
    value = min(1.0, max(-1.0, value))
    radicand = max(0.0, 1.0 - value * value)
    result = (
        math.sqrt(radicand) + (math.pi - math.acos(value)) * value
    ) / math.pi
    if not math.isfinite(result):
        raise FloatingPointError("Eq. 85 produced a non-finite value")
    return result


def assemble_warped_correlation(
    raw_correlation: Sequence[Sequence[float]] | np.ndarray,
    scale_ratios: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply Eqs. 80-81 independently to every off-diagonal pair."""
    raw = _as_finite_square_matrix(raw_correlation, "raw_correlation")
    _validate_correlation(raw, "raw_correlation", require_psd=False)
    ratios = np.asarray(scale_ratios, dtype=np.float64)
    if ratios.shape != (raw.shape[0],) or not np.all(np.isfinite(ratios)):
        raise ValueError("scale_ratios must be a finite vector matching the matrix dimension")
    if np.any(ratios <= 0.0):
        raise ValueError("scale_ratios must be strictly positive")

    warped = np.eye(raw.shape[0], dtype=np.float64)
    pairs: list[dict[str, Any]] = []
    for index_i in range(raw.shape[0]):
        for index_j in range(index_i + 1, raw.shape[0]):
            rho = float(raw[index_i, index_j])
            kappa = eq81_kappa(rho, float(ratios[index_i]), float(ratios[index_j]))
            q = eq80_warped_correlation(kappa)
            residual = abs(q / (1.0 - q * q) - kappa)
            eq85_value = normalized_eq85_relu_kernel(q)
            warped[index_i, index_j] = q
            warped[index_j, index_i] = q
            warped_local_minimum = 1.0 - abs(q)
            eq85_local_minimum = 1.0 - abs(eq85_value)
            pairs.append(
                {
                    "pair_id": f"{index_i + 1}-{index_j + 1}",
                    "index_i": index_i,
                    "index_j": index_j,
                    "raw_correlation": rho,
                    "scale_ratio_i": float(ratios[index_i]),
                    "scale_ratio_j": float(ratios[index_j]),
                    "eq81_kappa": kappa,
                    "eq80_warped_correlation": q,
                    "eq80_equation_residual": residual,
                    "warped_local_min_eigenvalue": warped_local_minimum,
                    "eq85_kernel_value": eq85_value,
                    "eq85_local_min_eigenvalue": eq85_local_minimum,
                    "warped_squared_distance": 2.0 - 2.0 * q,
                    "eq85_squared_distance": 2.0 - 2.0 * eq85_value,
                    "passed": bool(
                        warped_local_minimum > TOLERANCES["positive_definite_minimum"]
                        and eq85_local_minimum > TOLERANCES["positive_definite_minimum"]
                        and residual <= TOLERANCES["equation_residual_atol"]
                    ),
                }
            )
    _validate_correlation(warped, "warped_correlation", require_psd=False)
    return warped, pairs


def assemble_normalized_eq85(
    correlation: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Apply normalized Eq. 85 entrywise to a symmetric correlation table."""
    source = _as_finite_square_matrix(correlation, "correlation")
    _validate_correlation(source, "correlation", require_psd=False)
    result = np.eye(source.shape[0], dtype=np.float64)
    for index_i in range(source.shape[0]):
        for index_j in range(index_i + 1, source.shape[0]):
            value = normalized_eq85_relu_kernel(float(source[index_i, index_j]))
            result[index_i, index_j] = value
            result[index_j, index_i] = value
    _validate_correlation(result, "normalized_eq85", require_psd=False)
    return result


def canonical_eigensystem(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending eigenpairs with deterministic per-column signs."""
    value = _as_finite_square_matrix(matrix, "matrix")
    _validate_symmetric(value, "matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    for column in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            eigenvectors[:, column] = -vector
    return eigenvalues, eigenvectors


def negative_direction_certificate(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Return the minimum-eigenvector quadratic-form certificate."""
    value = _as_finite_square_matrix(matrix, "matrix")
    eigenvalues, eigenvectors = canonical_eigensystem(value)
    vector = eigenvectors[:, 0]
    minimum = float(eigenvalues[0])
    quadratic = float(vector @ value @ vector)
    norm = float(np.linalg.norm(vector))
    error = abs(quadratic - minimum)
    return {
        "vector": [float(item) for item in vector],
        "vector_norm": norm,
        "minimum_eigenvalue": minimum,
        "quadratic_form": quadratic,
        "absolute_match_error": error,
        "passed": bool(
            minimum < 0.0
            and quadratic < 0.0
            and abs(norm - 1.0) <= TOLERANCES["quadratic_form_atol"]
            and error <= TOLERANCES["quadratic_form_atol"]
        ),
    }


def pairwise_squared_distances(
    gram: Sequence[Sequence[float]] | np.ndarray,
) -> list[dict[str, Any]]:
    """Return d(i,j)^2 = Gii + Gjj - 2 Gij for all unordered pairs."""
    matrix = _as_finite_square_matrix(gram, "gram")
    _validate_symmetric(matrix, "gram")
    records: list[dict[str, Any]] = []
    for index_i in range(matrix.shape[0]):
        for index_j in range(index_i + 1, matrix.shape[0]):
            distance = float(
                matrix[index_i, index_i]
                + matrix[index_j, index_j]
                - 2.0 * matrix[index_i, index_j]
            )
            records.append(
                {
                    "pair_id": f"{index_i + 1}-{index_j + 1}",
                    "index_i": index_i,
                    "index_j": index_j,
                    "squared_distance": distance,
                    "nonnegative": bool(distance >= -TOLERANCES["distance_atol"]),
                }
            )
    return records


def maximum_triangle_violation(
    gram: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Return the largest direct-minus-two-leg violation from Gram distances.

    Entrywise nonnegative squared distances do not ensure that their square
    roots form a metric. For each unordered triple, this function selects the
    longest side and compares it with the sum of the other two sides.
    """
    matrix = _as_finite_square_matrix(gram, "gram")
    _validate_symmetric(matrix, "gram")
    squared_records = pairwise_squared_distances(matrix)
    squared = {
        tuple(sorted((int(record["index_i"]), int(record["index_j"])))): float(
            record["squared_distance"]
        )
        for record in squared_records
    }
    if any(value < -TOLERANCES["distance_atol"] for value in squared.values()):
        raise ValueError("triangle diagnostic requires nonnegative pairwise squared distances")
    distances = {key: math.sqrt(max(0.0, value)) for key, value in squared.items()}
    candidates: list[dict[str, Any]] = []
    dimension = matrix.shape[0]
    for first in range(dimension):
        for second in range(first + 1, dimension):
            for third in range(second + 1, dimension):
                sides = [
                    ((first, second), distances[(first, second)], third),
                    ((first, third), distances[(first, third)], second),
                    ((second, third), distances[(second, third)], first),
                ]
                sides.sort(key=lambda item: (item[1], item[0]), reverse=True)
                direct_pair, direct_distance, via = sides[0]
                leg_pairs = sorted(
                    (
                        tuple(sorted((direct_pair[0], via))),
                        tuple(sorted((direct_pair[1], via))),
                    )
                )
                leg_distances = [distances[pair] for pair in leg_pairs]
                violation = direct_distance - sum(leg_distances)
                candidates.append(
                    {
                        "triple": [first, second, third],
                        "direct_pair": list(direct_pair),
                        "via_index": via,
                        "direct_distance": direct_distance,
                        "leg_pairs": [list(pair) for pair in leg_pairs],
                        "leg_distances": leg_distances,
                        "violation": violation,
                        "triangle_inequality_satisfied": bool(
                            violation <= TOLERANCES["distance_atol"]
                        ),
                    }
                )
    if not candidates:
        raise ValueError("triangle diagnostic requires dimension at least three")
    return max(candidates, key=lambda item: float(item["violation"]))


def plain_eigenvalue_clipping(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Set negative eigenvalues of a symmetric matrix to zero."""
    value = _as_finite_square_matrix(matrix, "matrix")
    eigenvalues, eigenvectors = canonical_eigensystem(value)
    repaired = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    repaired = 0.5 * (repaired + repaired.T)
    if not np.all(np.isfinite(repaired)):
        raise FloatingPointError("plain eigenvalue clipping produced non-finite entries")
    _validate_symmetric(repaired, "plain_eigenvalue_clipping")
    return repaired


def psd_renormalized_correlation(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Clip to PSD, then restore a unit diagonal by congruence scaling."""
    clipped = plain_eigenvalue_clipping(matrix)
    diagonal = np.diag(clipped)
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise AuditFailure("clipped matrix has a non-positive diagonal")
    scale = np.sqrt(diagonal)
    repaired = clipped / np.outer(scale, scale)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    _validate_correlation(repaired, "psd_renormalized_correlation", require_psd=True)
    return repaired


def _spectrum(matrix: np.ndarray) -> list[float]:
    return [float(value) for value in canonical_eigensystem(matrix)[0]]


def _matrix_domain_diagonal_violation(matrix: np.ndarray) -> float:
    diagonal = float(np.max(np.abs(np.diag(matrix) - 1.0)))
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    domain = max(0.0, float(np.max(np.abs(off_diagonal))) - 1.0)
    return max(diagonal, domain)


def _repair_record(source_name: str, source: np.ndarray, method: str) -> tuple[str, np.ndarray, dict[str, Any]]:
    if method == "plain_eigenvalue_clipping":
        repaired = plain_eigenvalue_clipping(source)
        note = "PSD spectral clipping; the resulting diagonal is recorded and not forced to one."
    elif method == "psd_renormalized_correlation":
        repaired = psd_renormalized_correlation(source)
        note = "PSD clipping plus diagonal congruence renormalization; no optimality claim."
    else:
        raise ValueError(f"unknown repair method: {method}")
    repaired_name = f"{source_name}_{method}"
    before_minimum = float(np.linalg.eigvalsh(source)[0])
    after_minimum = float(np.linalg.eigvalsh(repaired)[0])
    difference = repaired - source
    diagonal_deviation = float(np.max(np.abs(np.diag(repaired) - 1.0)))
    distances = pairwise_squared_distances(repaired)
    minimum_distance = min(float(item["squared_distance"]) for item in distances)
    record = {
        "source_matrix": source_name,
        "repaired_matrix": repaired_name,
        "method": method,
        "evidence_label": "proposed_repair",
        "minimum_eigenvalue_before": before_minimum,
        "minimum_eigenvalue_after": after_minimum,
        "frobenius_change": float(np.linalg.norm(difference, ord="fro")),
        "maximum_absolute_change": float(np.max(np.abs(difference))),
        "maximum_diagonal_deviation": diagonal_deviation,
        "minimum_pairwise_squared_distance": minimum_distance,
        "psd_passed": bool(after_minimum >= -TOLERANCES["positive_semidefinite_atol"]),
        "unit_diagonal_passed": bool(diagonal_deviation <= TOLERANCES["unit_diagonal_atol"]),
        "note": note,
    }
    return repaired_name, repaired, record


def _check_records(observed: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    if set(observed) != set(CHECK_SPECS):
        raise AuditFailure(
            f"check registry mismatch: missing={sorted(set(CHECK_SPECS) - set(observed))}, "
            f"unexpected={sorted(set(observed) - set(CHECK_SPECS))}"
        )
    records: dict[str, dict[str, Any]] = {}
    for name, specification in CHECK_SPECS.items():
        value = float(observed[name])
        minimum = float(specification["minimum"])
        maximum = float(specification["maximum"])
        records[name] = {
            "observed": value,
            "minimum": minimum,
            "maximum": maximum,
            "units": specification["units"],
            "passed": bool(math.isfinite(value) and minimum <= value <= maximum),
        }
    return records


def enforce_checks(checks: Mapping[str, Any]) -> None:
    """Reject missing, altered-range, non-finite, or failing check records."""
    if not isinstance(checks, Mapping) or set(checks) != set(CHECK_SPECS):
        raise AuditFailure("exactly 22 declared checks are required")
    for name, specification in CHECK_SPECS.items():
        record = checks[name]
        if not isinstance(record, Mapping):
            raise AuditFailure(f"check {name} must be a mapping")
        if set(record) != {"observed", "minimum", "maximum", "units", "passed"}:
            raise AuditFailure(f"check {name} schema mismatch")
        if record["minimum"] != specification["minimum"] or record["maximum"] != specification["maximum"]:
            raise AuditFailure(f"check {name} range differs from the declared contract")
        if record["units"] != specification["units"]:
            raise AuditFailure(f"check {name} units differ from the declared contract")
        observed = float(record["observed"])
        expected_pass = bool(
            math.isfinite(observed)
            and float(specification["minimum"]) <= observed <= float(specification["maximum"])
        )
        if record["passed"] is not expected_pass or not expected_pass:
            raise AuditFailure(f"check failed: {name} observed {observed!r}")


def _configuration(full: bool) -> dict[str, Any]:
    return {
        "full_mode": bool(full),
        "float_dtype": "float64",
        "matrix_dimension": 3,
        "fixed_raw_correlation": [list(row) for row in WITNESS_R],
        "fixed_scale_ratios": list(WITNESS_SCALE_RATIOS),
        "fixed_input_policy": "six-decimal embedded witness values are exact inputs",
        "tolerances": dict(TOLERANCES),
        "formula_routes": {
            "pair_warp": "HOPE v1 Eqs. 80-81",
            "activation_kernel": "HOPE v1 normalized zero-bias Eq. 85",
        },
        "random_sampling": False,
    }


def build_audit(*, full: bool = False) -> dict[str, Any]:
    """Build every deterministic matrix, certificate, repair, and check."""
    raw = np.asarray(WITNESS_R, dtype=np.float64)
    ratios = np.asarray(WITNESS_SCALE_RATIOS, dtype=np.float64)
    _validate_correlation(raw, "fixed R", require_psd=True)
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise AuditFailure("fixed scale ratios are invalid")

    warped, local_pairs = assemble_warped_correlation(raw, ratios)
    kernel = assemble_normalized_eq85(warped)
    shared_control = assemble_normalized_eq85(raw)
    _validate_correlation(shared_control, "shared Eq. 85 control", require_psd=True)

    matrices: dict[str, np.ndarray] = {
        "input_R": raw,
        "warped_Q": warped,
        "normalized_eq85_K": kernel,
        "shared_eq85_control": shared_control,
    }
    repair_records: list[dict[str, Any]] = []
    for source_name, source in (("warped_Q", warped), ("normalized_eq85_K", kernel)):
        for method in ("plain_eigenvalue_clipping", "psd_renormalized_correlation"):
            repaired_name, repaired, record = _repair_record(source_name, source, method)
            matrices[repaired_name] = repaired
            repair_records.append(record)

    spectra = {name: _spectrum(matrix) for name, matrix in matrices.items()}
    negative_directions = {
        "warped_Q": negative_direction_certificate(warped),
        "normalized_eq85_K": negative_direction_certificate(kernel),
    }
    distances = {
        name: pairwise_squared_distances(matrix)
        for name, matrix in matrices.items()
    }
    triangle = maximum_triangle_violation(kernel)

    raw_off_diagonal = raw[~np.eye(raw.shape[0], dtype=bool)]
    local_minimum = min(
        min(float(pair["warped_local_min_eigenvalue"]), float(pair["eq85_local_min_eigenvalue"]))
        for pair in local_pairs
    )
    unrepaired_distance_minimum = min(
        float(record["squared_distance"])
        for name in ("warped_Q", "normalized_eq85_K")
        for record in distances[name]
    )
    plain_records = [record for record in repair_records if record["method"] == "plain_eigenvalue_clipping"]
    renormalized_records = [
        record for record in repair_records if record["method"] == "psd_renormalized_correlation"
    ]
    plain_violation = max(
        max(0.0, -float(record["minimum_eigenvalue_after"])) for record in plain_records
    )
    renormalized_violation = max(
        max(
            0.0,
            -float(record["minimum_eigenvalue_after"]),
            float(record["maximum_diagonal_deviation"]),
        )
        for record in renormalized_records
    )
    observed_checks = {
        "witness_nonfinite_count": float(
            np.count_nonzero(~np.isfinite(raw)) + np.count_nonzero(~np.isfinite(ratios))
        ),
        "raw_symmetry_residual": _symmetry_residual(raw),
        "raw_unit_diagonal_residual": float(np.max(np.abs(np.diag(raw) - 1.0))),
        "raw_maximum_pair_absolute_correlation": float(np.max(np.abs(raw_off_diagonal))),
        "minimum_scale_ratio": float(np.min(ratios)),
        "raw_minimum_eigenvalue": spectra["input_R"][0],
        "local_pair_count": float(len(local_pairs)),
        "minimum_local_pair_eigenvalue": local_minimum,
        "maximum_eq80_equation_residual": max(
            float(pair["eq80_equation_residual"]) for pair in local_pairs
        ),
        "warped_finite_symmetry_residual": _symmetry_residual(warped),
        "warped_domain_diagonal_violation": _matrix_domain_diagonal_violation(warped),
        "warped_minimum_eigenvalue": spectra["warped_Q"][0],
        "warped_negative_direction_error": float(
            negative_directions["warped_Q"]["absolute_match_error"]
        ),
        "eq85_finite_symmetry_residual": _symmetry_residual(kernel),
        "eq85_domain_diagonal_violation": _matrix_domain_diagonal_violation(kernel),
        "eq85_minimum_eigenvalue": spectra["normalized_eq85_K"][0],
        "eq85_negative_direction_error": float(
            negative_directions["normalized_eq85_K"]["absolute_match_error"]
        ),
        "minimum_unrepaired_pairwise_squared_distance": unrepaired_distance_minimum,
        "shared_control_minimum_eigenvalue": spectra["shared_eq85_control"][0],
        "plain_clipping_maximum_psd_violation": plain_violation,
        "renormalized_maximum_psd_or_diagonal_violation": renormalized_violation,
        "eq85_maximum_triangle_violation": float(triangle["violation"]),
    }
    checks = _check_records(observed_checks)
    enforce_checks(checks)
    if not all(bool(pair["passed"]) for pair in local_pairs):
        raise AuditFailure("at least one declared local pair failed")
    if not all(bool(record["psd_passed"]) for record in repair_records):
        raise AuditFailure("at least one declared repair is not PSD")
    if not all(
        bool(item["nonnegative"])
        for name in ("warped_Q", "normalized_eq85_K")
        for item in distances[name]
    ):
        raise AuditFailure("an unrepaired pairwise squared distance is negative")

    config = _configuration(full)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "passed": True,
        "claim_ids": list(CLAIM_IDS),
        "config": config,
        "config_sha256": _sha256_json(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": REPRODUCTION_SCOPE,
        "check_count": len(checks),
        "passing_check_count": sum(bool(record["passed"]) for record in checks.values()),
        "checks": checks,
        "witness": {
            "source": "embedded constants in this audit package",
            "raw_correlation": [list(row) for row in WITNESS_R],
            "scale_ratios": list(WITNESS_SCALE_RATIOS),
            "input_decimal_policy": "declared six-decimal values are exact",
        },
        "matrices": {
            name: [[float(value) for value in row] for row in matrix]
            for name, matrix in matrices.items()
        },
        "spectra": spectra,
        "local_pairs": local_pairs,
        "negative_directions": negative_directions,
        "pairwise_squared_distances": distances,
        "triangle_violation": triangle,
        "projection_comparison": repair_records,
    }


def _require_finite_payload(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_payload(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_payload(item, f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise AuditFailure(f"non-finite float in payload at {path}")
        return
    if isinstance(value, (str, bool, int, np.integer)) or value is None:
        return
    raise AuditFailure(f"unsupported payload type at {path}: {type(value)!r}")


def _canonical_json_bytes(payload: Any) -> bytes:
    _require_finite_payload(payload)
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    _require_finite_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    path.write_text(serialized + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise AuditFailure(f"CSV schema mismatch for {path.name}: {sorted(row)}")
            _require_finite_payload(row)
            writer.writerow(row)


def _paper_input_identity(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(Path("external-paper") / resolved.name)


def _paper_identity(*, full: bool, paper_pdf: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if full and paper_pdf is None:
        raise ValueError("full mode requires an explicit --paper-pdf path")
    if not full and paper_pdf is not None:
        raise ValueError("--paper-pdf is accepted only with full mode")
    if not PAPER_MANIFEST_PATH.is_file():
        raise AuditFailure("paper manifest is missing")
    manifest = json.loads(PAPER_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise AuditFailure("paper manifest root is not a mapping")
    if manifest.get("pdf_sha256") != PAPER_SHA256 or manifest.get("version") != "v1":
        raise AuditFailure("paper manifest does not identify the pinned v1 source")
    inputs = [
        {
            "kind": "paper_manifest",
            "path": "paper/manifest.json",
            "sha256": _sha256_path(PAPER_MANIFEST_PATH),
            "bytes": PAPER_MANIFEST_PATH.stat().st_size,
        }
    ]
    local_pdf: dict[str, Any] = {"status": "not_supplied", "authenticated": False}
    if full:
        assert paper_pdf is not None
        resolved = paper_pdf.resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise AuditFailure(f"local paper PDF is not a regular file: {resolved}")
        digest = _sha256_path(resolved)
        if digest != PAPER_SHA256:
            raise AuditFailure(
                f"paper PDF checksum mismatch: expected {PAPER_SHA256}, got {digest}"
            )
        identity = _paper_input_identity(resolved)
        local_pdf = {
            "status": "verified",
            "authenticated": True,
            "path": identity,
            "sha256": digest,
            "bytes": resolved.stat().st_size,
        }
        inputs.append(
            {
                "kind": "paper_pdf",
                "path": identity,
                "sha256": digest,
                "bytes": resolved.stat().st_size,
            }
        )
    paper = {
        "arxiv_id": manifest.get("arxiv_id"),
        "version": manifest.get("version"),
        "title": manifest.get("title"),
        "pdf_filename": manifest.get("pdf_filename"),
        "pdf_sha256": PAPER_SHA256,
        "anchors": [
            "Section 5 and Eqs. 4-5, PDF pages 7-9",
            "Appendix A Eq. 30 and Proposition A.1, PDF page 35",
            "Eq. 73, PDF page 52",
            "Proposition E.3 and Eqs. 80-81, PDF page 54",
            "Eqs. 84-85, PDF page 56",
        ],
        "local_pdf": local_pdf,
    }
    return paper, inputs


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _code_identity() -> dict[str, Any]:
    commit = _git_value("rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return {
            "source_snapshot_commit": None,
            "source_snapshot_state": "source_archive_with_individually_hashed_sources",
            "source_files_match_commit": False,
        }
    status = _git_value("status", "--porcelain", "--", *SOURCE_RELATIVE_PATHS)
    matches_commit = status == ""
    return {
        "source_snapshot_commit": commit,
        "source_snapshot_state": (
            "exact_commit" if matches_commit else "working_tree_with_individually_hashed_sources"
        ),
        "source_files_match_commit": matches_commit,
    }


def _source_hashes() -> list[dict[str, Any]]:
    paths = [ROOT / relative for relative in SOURCE_RELATIVE_PATHS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AuditFailure(f"audit source file missing: {missing}")
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "float64_epsilon": float(np.finfo(np.float64).eps),
        "cpu_count": os.cpu_count(),
    }


def _manifest_output_path(path: Path, result_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        try:
            relative = path.resolve().relative_to(result_dir.resolve())
        except ValueError as error:
            raise AuditFailure("output path is outside the declared result directory") from error
        return str(Path("external-result") / relative)


def _csv_rows(summary: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    matrix_evidence = {
        "input_R": "counterexample",
        "warped_Q": "counterexample",
        "normalized_eq85_K": "counterexample",
        "shared_eq85_control": "verified_fact",
    }
    matrix_rows: list[dict[str, Any]] = []
    spectra_rows: list[dict[str, Any]] = []
    for matrix_name, matrix in summary["matrices"].items():
        evidence = matrix_evidence.get(matrix_name, "proposed_repair")
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                matrix_rows.append(
                    {
                        "matrix_name": matrix_name,
                        "row": row_index,
                        "column": column_index,
                        "value": value,
                        "evidence_label": evidence,
                    }
                )
        eigenvalues = summary["spectra"][matrix_name]
        minimum = float(eigenvalues[0])
        if minimum < -TOLERANCES["positive_semidefinite_atol"]:
            classification = "indefinite"
        elif minimum <= TOLERANCES["positive_semidefinite_atol"]:
            classification = "positive_semidefinite_boundary"
        else:
            classification = "positive_definite"
        for index, value in enumerate(eigenvalues):
            spectra_rows.append(
                {
                    "matrix_name": matrix_name,
                    "eigenvalue_index": index,
                    "eigenvalue": value,
                    "classification": classification,
                    "evidence_label": evidence,
                }
            )

    negative_rows: list[dict[str, Any]] = []
    for matrix_name, certificate in summary["negative_directions"].items():
        for component_index, component in enumerate(certificate["vector"]):
            negative_rows.append(
                {
                    "matrix_name": matrix_name,
                    "component_index": component_index,
                    "component_value": component,
                    "vector_norm": certificate["vector_norm"],
                    "minimum_eigenvalue": certificate["minimum_eigenvalue"],
                    "quadratic_form": certificate["quadratic_form"],
                    "absolute_match_error": certificate["absolute_match_error"],
                    "passed": certificate["passed"],
                }
            )
    return {
        "witness_matrices.csv": matrix_rows,
        "spectra.csv": spectra_rows,
        "pair_checks.csv": list(summary["local_pairs"]),
        "negative_direction.csv": negative_rows,
        "projection_comparison.csv": list(summary["projection_comparison"]),
    }


def validate_summary(payload: Mapping[str, Any]) -> None:
    """Recompute all deterministic sections and reject any altered summary."""
    if not isinstance(payload, Mapping):
        raise AuditFailure("summary must be a mapping")
    _require_finite_payload(payload)
    expected_keys = {
        "schema_version",
        "audit_id",
        "status",
        "passed",
        "claim_ids",
        "paper",
        "config",
        "config_sha256",
        "method_origins",
        "reproduction_scope",
        "check_count",
        "passing_check_count",
        "checks",
        "witness",
        "matrices",
        "spectra",
        "local_pairs",
        "negative_directions",
        "pairwise_squared_distances",
        "triangle_violation",
        "projection_comparison",
    }
    if set(payload) != expected_keys:
        raise AuditFailure(
            f"summary schema mismatch: missing={sorted(expected_keys - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected_keys)}"
        )
    if payload["schema_version"] != AUDIT_SCHEMA_VERSION or payload["audit_id"] != AUDIT_ID:
        raise AuditFailure("summary identity mismatch")
    if payload["status"] != "passed" or payload["passed"] is not True:
        raise AuditFailure("summary does not report a passing audit")
    if tuple(payload["claim_ids"]) != CLAIM_IDS:
        raise AuditFailure("summary claim IDs differ")
    if payload["method_origins"] != METHOD_ORIGINS or payload["reproduction_scope"] != REPRODUCTION_SCOPE:
        raise AuditFailure("summary provenance registry differs")
    if payload["check_count"] != 22 or payload["passing_check_count"] != 22:
        raise AuditFailure("summary must contain 22 passing checks")
    enforce_checks(payload["checks"])
    config = payload["config"]
    if not isinstance(config, Mapping) or not isinstance(config.get("full_mode"), bool):
        raise AuditFailure("summary full-mode configuration is malformed")
    expected = build_audit(full=config["full_mode"])
    for key in expected_keys - {"paper"}:
        if payload[key] != expected[key]:
            raise AuditFailure(f"summary deterministic section mismatch: {key}")
    paper = payload["paper"]
    if not isinstance(paper, Mapping) or paper.get("pdf_sha256") != PAPER_SHA256:
        raise AuditFailure("summary paper identity mismatch")
    local_pdf = paper.get("local_pdf")
    if not isinstance(local_pdf, Mapping):
        raise AuditFailure("summary local PDF record is malformed")
    expected_status = "verified" if config["full_mode"] else "not_supplied"
    if local_pdf.get("status") != expected_status:
        raise AuditFailure("summary local PDF status differs from full-mode configuration")


def run_audit(
    *,
    result_dir: Path = DEFAULT_RESULT_DIR,
    manifest_path: Path | None = None,
    paper_pdf: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Run the audit and write deterministic JSON, CSV, and hash manifest files."""
    if not isinstance(full, bool):
        raise ValueError("full must be a boolean")
    result_dir = result_dir.resolve()
    manifest_path = result_dir / "manifest.json" if manifest_path is None else manifest_path.resolve()
    reserved = {result_dir / "summary.json", *(result_dir / name for name in CSV_CONTRACTS)}
    if manifest_path in reserved:
        raise ValueError("manifest_path must not replace a result file")

    paper, paper_inputs = _paper_identity(full=full, paper_pdf=paper_pdf)
    summary = build_audit(full=full)
    summary["paper"] = paper
    validate_summary(summary)
    rows = _csv_rows(summary)

    summary_path = result_dir / "summary.json"
    _write_json(summary_path, summary)
    for filename, fields in CSV_CONTRACTS.items():
        _write_csv(result_dir / filename, fields, rows[filename])

    output_paths = [summary_path, *(result_dir / filename for filename in CSV_CONTRACTS)]
    outputs = [
        {
            "path": _manifest_output_path(path, result_dir),
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
            "media_type": "application/json" if path.suffix == ".json" else "text/csv",
        }
        for path in output_paths
    ]
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "generation_time_policy": "wall clock omitted for deterministic output",
        "paper": paper,
        "config": summary["config"],
        "config_sha256": summary["config_sha256"],
        "seeds": {
            "random_sampling": "not_used",
            "fixed_witness": "deterministic_six_decimal_constants",
        },
        "code": _code_identity(),
        "environment": _environment(),
        "expected_metric_ranges": CHECK_SPECS,
        "observed_checks": summary["checks"],
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": REPRODUCTION_SCOPE,
        "source_hashes": _source_hashes(),
        "input_hashes": [
            *paper_inputs,
            {
                "kind": "embedded_fixed_witness",
                "path": "summary.witness",
                "sha256": _sha256_json(summary["witness"]),
            },
        ],
        "outputs": outputs,
        "manifest_hash_scope": "manifest excludes its own hash to avoid self-reference",
        "canonical_invocation": {
            "entrypoint": "experiments/audits/part05_global_psd/run.py",
            "result_dir": "<RESULT_DIR>",
            "manifest_path": "<MANIFEST_PATH>",
            "full": full,
            "paper_pdf": "verified v1 PDF" if full else "not supplied",
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "audit_id": AUDIT_ID,
        "status": "passed",
        "passed": True,
        "check_count": 22,
        "passing_check_count": 22,
        "config_sha256": summary["config_sha256"],
        "summary": str(summary_path),
        "csv_outputs": [str(path) for path in output_paths if path.suffix == ".csv"],
        "manifest": str(manifest_path),
        "key_metrics": {
            "raw_minimum_eigenvalue": summary["spectra"]["input_R"][0],
            "warped_minimum_eigenvalue": summary["spectra"]["warped_Q"][0],
            "eq85_minimum_eigenvalue": summary["spectra"]["normalized_eq85_K"][0],
            "shared_control_minimum_eigenvalue": summary["spectra"]["shared_eq85_control"][0],
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Fresh directory for deterministic JSON and CSV outputs.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Manifest destination; defaults to RESULT_DIR/manifest.json.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Authenticate an explicitly supplied local paper PDF.",
    )
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        default=None,
        help="Pinned v1 PDF; accepted only with --full.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_audit(
        result_dir=args.result_dir,
        manifest_path=args.manifest_path,
        paper_pdf=args.paper_pdf,
        full=args.full,
    )
    print(json.dumps(result, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
