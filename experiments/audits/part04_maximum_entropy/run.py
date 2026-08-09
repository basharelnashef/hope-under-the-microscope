#!/usr/bin/env python3
"""Fail-closed audit of HOPE v1's maximum-entropy covariance construction.

The audit uses deterministic analytical fixtures. It does not infer failure
from an optimizer status and does not reproduce a paper experiment.
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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hope_repro.paper_literal.maximum_entropy import (  # noqa: E402
    appendix_identity_precision,
    appendix_pair_correlation,
    canonical_null_direction,
    determinant_lemma_logdet,
    escape_covariance,
    linear_constraint_residual,
    maximum_constraint_residual,
    numerical_rank,
    positive_definite_logdet,
    projection_variances,
    pure_pair_maximum_entropy_covariance,
    pure_stationarity_precision,
    spanning_logdet_upper_bound,
)
from hope_repro.proposed_repairs.relative_entropy import (  # noqa: E402
    gaussian_covariance_kl,
    pair_relative_entropy_solution,
    reference_pair_kappa,
    reference_precision,
)


AUDIT_ID = "part04_maximum_entropy_rank_audit"
AUDIT_SCHEMA_VERSION = 1
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_04" / "manual"
DEFAULT_SEED = 20260803

ESCAPE_AMOUNTS = (0.0, 1.0, 10.0, 1_000.0, 1_000_000.0)
DUPLICATE_ESCAPE_AMOUNTS = (0.0, 1.0, 1_000.0)
FULL_SPAN_OFF_DIAGONALS = (-1.999, -1.5, -0.5, 0.0, 0.5, 1.5, 1.999)
TAU_GRID = (0.5, 0.75, 1.0, 2.0, 4.0, 10.0)
SEEDED_CASES = (
    (3, 1),
    (3, 2),
    (3, 3),
    (5, 2),
    (5, 4),
    (5, 5),
)
SEEDED_ESCAPE_AMOUNT = 10_000.0

TOLERANCES = {
    "constraint_atol": 8e-12,
    "determinant_lemma_atol": 8e-12,
    "trace_atol": 8e-12,
    "bound_atol": 8e-12,
    "pair_atol": 8e-12,
    "stationarity_atol": 2e-11,
    "infeasible_residual_minimum": 0.49,
}

CLAIM_IDS = (
    "P04-PURE-MAXENT-FORMULATION",
    "P04-NONSPANNING-UNBOUNDED",
    "P04-FULL-SPAN-BOUNDARY",
    "P04-LOCAL-IDENTITY-PRECISION",
    "P04-PAIR-RANK-DOMAIN",
    "P04-IDENTITY-PRIOR-INTERPRETATION",
    "P04-RELATIVE-ENTROPY-REPAIR",
    "P04-PRIOR-SCALE-SENSITIVITY",
)

METHOD_ORIGINS = {
    "pure_maximum_entropy": {
        "namespace": "hope_repro.paper_literal.maximum_entropy",
        "evidence_label": "paper_claim",
        "description": "Section 4 log-determinant objective and identity-free stationarity shape.",
    },
    "rank_analysis": {
        "namespace": "experiments.audits.part04_maximum_entropy",
        "evidence_label": "verified_fact",
        "description": "Independent finite-dimensional rank proof and deterministic fixtures.",
    },
    "appendix_identity_precision": {
        "namespace": "hope_repro.paper_literal.maximum_entropy",
        "evidence_label": "paper_claim",
        "description": "Unit-identity precision and warped pair correlation printed in Appendix E.3.1.",
    },
    "relative_entropy_reference": {
        "namespace": "hope_repro.proposed_repairs.relative_entropy",
        "evidence_label": "proposed_repair",
        "description": "Explicit KL projection against N(0, tau^2 I), not a paper-reported method.",
    },
}

SOURCE_RELATIVE_PATHS = (
    "src/hope_repro/paper_literal/maximum_entropy.py",
    "src/hope_repro/proposed_repairs/relative_entropy.py",
    "experiments/audits/part04_maximum_entropy/README.md",
    "experiments/audits/part04_maximum_entropy/run.py",
    "experiments/audits/part04_maximum_entropy/test_run.py",
)

ESCAPE_FIELDS = (
    "case_id",
    "t",
    "dimension",
    "constraint_count",
    "rank",
    "nullity",
    "min_eigenvalue",
    "max_eigenvalue",
    "logdet",
    "max_constraint_residual",
    "determinant_lemma_error",
    "passed",
)
BOUNDED_FIELDS = (
    "case_id",
    "off_diagonal",
    "min_eigenvalue",
    "logdet",
    "trace_value",
    "trace_logdet_upper_bound",
    "is_unique_fixture_maximum",
    "passed",
)
PAIR_FIELDS = (
    "case_id",
    "method",
    "evidence_label",
    "correlation",
    "logdet",
    "projection_variance_1",
    "projection_variance_2",
    "constraint_residual",
    "passed",
)
TAU_FIELDS = (
    "case_id",
    "tau",
    "kappa",
    "correlation",
    "logdet",
    "kl_to_reference",
    "stationarity_residual",
    "constraint_residual",
    "passed",
)
RANK_FIELDS = (
    "case_id",
    "seed",
    "dimension",
    "constraint_count",
    "rank",
    "nullity",
    "classification",
    "base_logdet",
    "test_logdet",
    "max_constraint_residual",
    "determinant_or_bound_residual",
    "passed",
)


class AuditFailure(RuntimeError):
    """Raised when a declared Part 4 invariant fails."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    _require_finite_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        allow_nan=False,
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
                raise AuditFailure(
                    f"CSV row schema mismatch for {path.name}: {sorted(row)}"
                )
            _require_finite_payload(row)
            writer.writerow(row)


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


def _rank_record(directions: np.ndarray) -> tuple[int, int, float, list[float]]:
    rank, threshold, singular_values = numerical_rank(directions)
    return (
        rank,
        directions.shape[0] - rank,
        float(threshold),
        [float(value) for value in singular_values],
    )


def _correlation(output_covariance: np.ndarray) -> float:
    denominator = math.sqrt(
        float(output_covariance[0, 0] * output_covariance[1, 1])
    )
    value = float(output_covariance[0, 1] / denominator)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise AuditFailure("computed pair correlation was invalid")
    return value


def _escape_case(
    *,
    case_id: str,
    directions: np.ndarray,
    base_covariance: np.ndarray,
    target_variances: np.ndarray,
    amounts: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rank, nullity, rank_tolerance, singular_values = _rank_record(directions)
    if nullity <= 0:
        raise AuditFailure(f"{case_id} does not have an escaping nullspace")
    null_direction = canonical_null_direction(
        directions,
        rank_tolerance=rank_tolerance,
    )
    orthogonality_residual = float(
        np.max(np.abs(directions.T @ null_direction))
    )
    base_logdet = positive_definite_logdet(base_covariance)
    rows: list[dict[str, Any]] = []
    for amount in amounts:
        covariance = escape_covariance(base_covariance, null_direction, amount)
        eigenvalues = np.linalg.eigvalsh(covariance)
        logdet = positive_definite_logdet(covariance)
        predicted = determinant_lemma_logdet(
            base_covariance,
            null_direction,
            amount,
        )
        constraint_residual = maximum_constraint_residual(
            covariance,
            directions,
            target_variances,
        )
        determinant_error = abs(logdet - predicted)
        passed = (
            float(eigenvalues[0]) > 0.0
            and constraint_residual <= TOLERANCES["constraint_atol"]
            and determinant_error <= TOLERANCES["determinant_lemma_atol"]
        )
        rows.append(
            {
                "case_id": case_id,
                "t": float(amount),
                "dimension": int(directions.shape[0]),
                "constraint_count": int(directions.shape[1]),
                "rank": rank,
                "nullity": nullity,
                "min_eigenvalue": float(eigenvalues[0]),
                "max_eigenvalue": float(eigenvalues[-1]),
                "logdet": logdet,
                "max_constraint_residual": constraint_residual,
                "determinant_lemma_error": determinant_error,
                "passed": passed,
            }
        )
    section = {
        "case_id": case_id,
        "evidence_label": "counterexample",
        "dimension": int(directions.shape[0]),
        "constraint_count": int(directions.shape[1]),
        "rank": rank,
        "nullity": nullity,
        "rank_tolerance": rank_tolerance,
        "singular_values": singular_values,
        "null_direction": [float(value) for value in null_direction],
        "orthogonality_residual": orthogonality_residual,
        "target_variances": [float(value) for value in target_variances],
        "base_logdet": base_logdet,
        "final_logdet": rows[-1]["logdet"],
        "logdet_gain": rows[-1]["logdet"] - base_logdet,
        "max_constraint_residual": max(
            row["max_constraint_residual"] for row in rows
        ),
        "max_determinant_lemma_error": max(
            row["determinant_lemma_error"] for row in rows
        ),
        "all_cases_passed": all(row["passed"] for row in rows),
        "path": rows,
    }
    return section, rows


def build_rank_deficient_escape() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directions = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    base_covariance = np.diag(np.asarray([1.0, 4.0, 1.0]))
    targets = np.asarray([1.0, 4.0], dtype=np.float64)
    return _escape_case(
        case_id="rank_deficient_r2_d3",
        directions=directions,
        base_covariance=base_covariance,
        target_variances=targets,
        amounts=ESCAPE_AMOUNTS,
    )


def build_duplicate_count_vs_rank() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directions = np.asarray(
        [
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    base_covariance = np.eye(2, dtype=np.float64)
    targets = np.asarray([1.0, 1.0], dtype=np.float64)
    section, rows = _escape_case(
        case_id="duplicate_columns_m2_d2",
        directions=directions,
        base_covariance=base_covariance,
        target_variances=targets,
        amounts=DUPLICATE_ESCAPE_AMOUNTS,
    )
    section["count_equals_dimension"] = True
    section["interpretation"] = (
        "Constraint count equals ambient dimension, but rank one leaves one unseen direction."
    )
    return section, rows


def build_full_span_control() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directions = np.eye(2, dtype=np.float64)
    targets = np.asarray([1.0, 4.0], dtype=np.float64)
    rank, nullity, rank_tolerance, singular_values = _rank_record(directions)
    trace_bound = spanning_logdet_upper_bound(
        directions,
        targets,
        rank_tolerance=rank_tolerance,
    )
    exact_fixture_maximum = math.log(4.0)
    rows: list[dict[str, Any]] = []
    for off_diagonal in FULL_SPAN_OFF_DIAGONALS:
        covariance = np.asarray(
            [[1.0, off_diagonal], [off_diagonal, 4.0]],
            dtype=np.float64,
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        logdet = positive_definite_logdet(covariance)
        trace_value = float(np.trace((directions @ directions.T) @ covariance))
        constraint_residual = maximum_constraint_residual(
            covariance,
            directions,
            targets,
        )
        unique_maximum = off_diagonal == 0.0
        passed = (
            float(eigenvalues[0]) > 0.0
            and constraint_residual <= TOLERANCES["constraint_atol"]
            and abs(trace_value - float(np.sum(targets)))
            <= TOLERANCES["trace_atol"]
            and logdet <= exact_fixture_maximum + TOLERANCES["bound_atol"]
            and logdet <= trace_bound + TOLERANCES["bound_atol"]
        )
        rows.append(
            {
                "case_id": "full_span_identity_d2",
                "off_diagonal": float(off_diagonal),
                "min_eigenvalue": float(eigenvalues[0]),
                "logdet": logdet,
                "trace_value": trace_value,
                "trace_logdet_upper_bound": trace_bound,
                "is_unique_fixture_maximum": unique_maximum,
                "passed": passed,
            }
        )
    maximum_row = max(rows, key=lambda row: row["logdet"])
    section = {
        "case_id": "full_span_identity_d2",
        "evidence_label": "verified_fact",
        "dimension": 2,
        "constraint_count": 2,
        "rank": rank,
        "nullity": nullity,
        "rank_tolerance": rank_tolerance,
        "singular_values": singular_values,
        "target_variances": [1.0, 4.0],
        "trace_logdet_upper_bound": trace_bound,
        "exact_fixture_maximum_logdet": exact_fixture_maximum,
        "observed_maximum_off_diagonal": maximum_row["off_diagonal"],
        "observed_maximum_logdet": maximum_row["logdet"],
        "all_cases_passed": all(row["passed"] for row in rows),
        "path": rows,
    }
    return section, rows


def build_infeasible_control() -> dict[str, Any]:
    directions = np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    targets = np.asarray([1.0, 2.0], dtype=np.float64)
    rank, nullity, rank_tolerance, singular_values = _rank_record(directions)
    residual = linear_constraint_residual(directions, targets)
    passed = residual >= TOLERANCES["infeasible_residual_minimum"]
    return {
        "case_id": "incompatible_duplicate_constraints",
        "evidence_label": "verified_fact",
        "dimension": 2,
        "constraint_count": 2,
        "rank": rank,
        "nullity": nullity,
        "rank_tolerance": rank_tolerance,
        "singular_values": singular_values,
        "target_variances": [1.0, 2.0],
        "minimum_linear_constraint_residual": residual,
        "classification": "infeasible",
        "unboundedness_claim_applied": False,
        "passed": passed,
    }


def _cosine_half_directions() -> np.ndarray:
    return np.asarray(
        [[1.0, 0.5], [0.0, math.sqrt(3.0) / 2.0]],
        dtype=np.float64,
    )


def build_cosine_half_pair() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directions = _cosine_half_directions()
    targets = np.ones(2, dtype=np.float64)
    gram = directions.T @ directions
    raw_correlation = _correlation(gram)

    pure_covariance = pure_pair_maximum_entropy_covariance(directions, targets)
    pure_output = directions.T @ pure_covariance @ directions
    pure_correlation = _correlation(pure_output)
    pure_logdet = positive_definite_logdet(pure_covariance)
    pure_precision = pure_stationarity_precision(directions, np.ones(2))
    pure_stationarity_residual = float(
        np.max(np.abs(np.linalg.inv(pure_covariance) - pure_precision))
    )

    appendix_covariance = np.eye(2, dtype=np.float64)
    appendix_output = directions.T @ appendix_covariance @ directions
    appendix_correlation = appendix_pair_correlation(
        raw_correlation,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    appendix_observed_correlation = _correlation(appendix_output)
    appendix_precision = appendix_identity_precision(directions, np.zeros(2))
    appendix_stationarity_residual = float(
        np.max(np.abs(np.linalg.inv(appendix_covariance) - appendix_precision))
    )
    appendix_logdet = positive_definite_logdet(appendix_covariance)

    pure_constraint_residual = maximum_constraint_residual(
        pure_covariance,
        directions,
        targets,
    )
    appendix_constraint_residual = maximum_constraint_residual(
        appendix_covariance,
        directions,
        targets,
    )
    objective_advantage = pure_logdet - appendix_logdet
    expected_advantage = math.log(4.0 / 3.0)
    common_pass = (
        abs(raw_correlation - 0.5) <= TOLERANCES["pair_atol"]
        and pure_constraint_residual <= TOLERANCES["constraint_atol"]
        and appendix_constraint_residual <= TOLERANCES["constraint_atol"]
        and abs(pure_correlation) <= TOLERANCES["pair_atol"]
        and abs(appendix_correlation - 0.5) <= TOLERANCES["pair_atol"]
        and abs(appendix_observed_correlation - appendix_correlation)
        <= TOLERANCES["pair_atol"]
        and pure_stationarity_residual <= TOLERANCES["stationarity_atol"]
        and appendix_stationarity_residual <= TOLERANCES["stationarity_atol"]
        and abs(objective_advantage - expected_advantage)
        <= TOLERANCES["pair_atol"]
        and objective_advantage > 0.0
    )
    rows = [
        {
            "case_id": "cosine_half_pair",
            "method": "pure_local_maximum_entropy",
            "evidence_label": "counterexample",
            "correlation": pure_correlation,
            "logdet": pure_logdet,
            "projection_variance_1": float(pure_output[0, 0]),
            "projection_variance_2": float(pure_output[1, 1]),
            "constraint_residual": pure_constraint_residual,
            "passed": common_pass,
        },
        {
            "case_id": "cosine_half_pair",
            "method": "appendix_identity_precision_tau_1",
            "evidence_label": "paper_claim",
            "correlation": appendix_observed_correlation,
            "logdet": appendix_logdet,
            "projection_variance_1": float(appendix_output[0, 0]),
            "projection_variance_2": float(appendix_output[1, 1]),
            "constraint_residual": appendix_constraint_residual,
            "passed": common_pass,
        },
    ]
    section = {
        "case_id": "cosine_half_pair",
        "description": "Two unit directions separated by 60 degrees with cosine one half.",
        "evidence_label": "counterexample",
        "raw_correlation": raw_correlation,
        "target_variances": [1.0, 1.0],
        "pure_correlation": pure_correlation,
        "appendix_correlation": appendix_observed_correlation,
        "pure_logdet": pure_logdet,
        "appendix_logdet": appendix_logdet,
        "pure_objective_advantage": objective_advantage,
        "expected_objective_advantage": expected_advantage,
        "pure_stationarity_residual": pure_stationarity_residual,
        "appendix_stationarity_residual": appendix_stationarity_residual,
        "passed": common_pass,
        "methods": rows,
    }
    return section, rows


def build_pair_rank_domain() -> dict[str, Any]:
    """Verify that collinear active directions remain outside Eq. 80's domain."""
    directions = np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    targets = np.asarray([1.0, 4.0], dtype=np.float64)
    rank, nullity, rank_tolerance, singular_values = _rank_record(directions)
    rejected_operations: list[str] = []
    try:
        pure_pair_maximum_entropy_covariance(directions, targets)
    except ValueError:
        rejected_operations.append("pure_two_dimensional_pair_solver")
    try:
        appendix_pair_correlation(1.0, 1.0, 2.0, 1.0, 2.0)
    except ValueError:
        rejected_operations.append("appendix_eq80_eq81_formula")
    try:
        pair_relative_entropy_solution(directions, targets, 1.0)
    except ValueError:
        rejected_operations.append("relative_entropy_pair_solver")
    consistency_residual = linear_constraint_residual(directions, targets)
    passed = (
        rank == 1
        and nullity == 1
        and len(rejected_operations) == 3
        and consistency_residual <= TOLERANCES["constraint_atol"]
    )
    return {
        "case_id": "collinear_active_pair",
        "evidence_label": "verified_fact",
        "dimension": 2,
        "constraint_count": 2,
        "rank": rank,
        "nullity": nullity,
        "rank_tolerance": rank_tolerance,
        "singular_values": singular_values,
        "raw_correlation_magnitude": 1.0,
        "target_variances": [1.0, 4.0],
        "linear_constraint_residual": consistency_residual,
        "rejected_operations": rejected_operations,
        "rejection_count": len(rejected_operations),
        "required_rejection_count": 3,
        "passed": passed,
    }


def build_tau_scale_sweep() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directions = _cosine_half_directions()
    targets = np.ones(2, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for tau in TAU_GRID:
        covariance, multipliers, correlation = pair_relative_entropy_solution(
            directions,
            targets,
            tau,
        )
        precision = reference_precision(directions, multipliers, tau)
        stationarity_residual = float(
            np.max(np.abs(np.linalg.inv(covariance) - precision))
        )
        constraint_residual = maximum_constraint_residual(
            covariance,
            directions,
            targets,
        )
        kappa = reference_pair_kappa(0.5, 1.0, 1.0, 1.0, 1.0, tau)
        passed = (
            stationarity_residual <= TOLERANCES["stationarity_atol"]
            and constraint_residual <= TOLERANCES["constraint_atol"]
            and -1.0 < correlation < 1.0
        )
        rows.append(
            {
                "case_id": "cosine_half_pair",
                "tau": float(tau),
                "kappa": kappa,
                "correlation": correlation,
                "logdet": positive_definite_logdet(covariance),
                "kl_to_reference": gaussian_covariance_kl(covariance, tau),
                "stationarity_residual": stationarity_residual,
                "constraint_residual": constraint_residual,
                "passed": passed,
            }
        )
    correlations = [row["correlation"] for row in rows]
    strictly_decreasing = all(
        first > second
        for first, second in zip(correlations, correlations[1:])
    )
    unit_row = next(row for row in rows if row["tau"] == 1.0)
    section = {
        "case_id": "cosine_half_pair",
        "evidence_label": "proposed_repair",
        "tau_grid": [float(value) for value in TAU_GRID],
        "correlation_minimum": min(correlations),
        "correlation_maximum": max(correlations),
        "correlation_range": max(correlations) - min(correlations),
        "strictly_decreasing_with_tau": strictly_decreasing,
        "tau_one_correlation": unit_row["correlation"],
        "max_stationarity_residual": max(
            row["stationarity_residual"] for row in rows
        ),
        "max_constraint_residual": max(
            row["constraint_residual"] for row in rows
        ),
        "all_cases_passed": strictly_decreasing and all(
            row["passed"] for row in rows
        ),
        "cases": rows,
    }
    return section, rows


def _canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=np.float64).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0.0:
            output[:, column] *= -1.0
    return output


def build_seeded_rank_cases(seed: int = DEFAULT_SEED) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rows: list[dict[str, Any]] = []
    detailed_cases: list[dict[str, Any]] = []
    for index, (dimension, intended_rank) in enumerate(SEEDED_CASES):
        case_seed = seed + 101 * index
        rng = np.random.default_rng(case_seed)
        orthogonal, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        directions = _canonicalize_columns(orthogonal[:, :intended_rank])
        scales = np.linspace(0.75, 1.25, intended_rank, dtype=np.float64)
        directions = directions * scales[np.newaxis, :]
        factor = rng.normal(size=(dimension, dimension))
        base_covariance = (
            factor @ factor.T / dimension
            + 0.5 * np.eye(dimension, dtype=np.float64)
        )
        targets = projection_variances(base_covariance, directions)
        rank, nullity, rank_tolerance, singular_values = _rank_record(directions)
        base_logdet = positive_definite_logdet(base_covariance)
        constraint_residual = maximum_constraint_residual(
            base_covariance,
            directions,
            targets,
        )
        if rank != intended_rank:
            raise AuditFailure(
                f"seeded case rank mismatch: intended {intended_rank}, got {rank}"
            )
        if nullity > 0:
            null_direction = canonical_null_direction(
                directions,
                rank_tolerance=rank_tolerance,
            )
            test_covariance = escape_covariance(
                base_covariance,
                null_direction,
                SEEDED_ESCAPE_AMOUNT,
            )
            test_logdet = positive_definite_logdet(test_covariance)
            predicted = determinant_lemma_logdet(
                base_covariance,
                null_direction,
                SEEDED_ESCAPE_AMOUNT,
            )
            determinant_or_bound_residual = abs(test_logdet - predicted)
            classification = "unbounded_feasible_ray"
            passed = (
                test_logdet > base_logdet
                and maximum_constraint_residual(
                    test_covariance,
                    directions,
                    targets,
                )
                <= TOLERANCES["constraint_atol"]
                and determinant_or_bound_residual
                <= TOLERANCES["determinant_lemma_atol"]
            )
        else:
            bound = spanning_logdet_upper_bound(
                directions,
                targets,
                rank_tolerance=rank_tolerance,
            )
            test_logdet = base_logdet
            determinant_or_bound_residual = max(0.0, test_logdet - bound)
            classification = "bounded_full_span"
            passed = determinant_or_bound_residual <= TOLERANCES["bound_atol"]
        case_id = f"seeded_d{dimension}_r{intended_rank}"
        row = {
            "case_id": case_id,
            "seed": case_seed,
            "dimension": dimension,
            "constraint_count": int(directions.shape[1]),
            "rank": rank,
            "nullity": nullity,
            "classification": classification,
            "base_logdet": base_logdet,
            "test_logdet": test_logdet,
            "max_constraint_residual": constraint_residual,
            "determinant_or_bound_residual": determinant_or_bound_residual,
            "passed": passed,
        }
        rows.append(row)
        detailed_cases.append(
            {
                **row,
                "rank_tolerance": rank_tolerance,
                "singular_values": singular_values,
                "target_variances": [float(value) for value in targets],
            }
        )
    section = {
        "evidence_label": "verified_fact",
        "seed": seed,
        "case_count": len(rows),
        "rank_deficient_case_count": sum(row["nullity"] > 0 for row in rows),
        "full_span_case_count": sum(row["nullity"] == 0 for row in rows),
        "failure_count": sum(not row["passed"] for row in rows),
        "all_cases_passed": all(row["passed"] for row in rows),
        "cases": detailed_cases,
    }
    return section, rows


def _expected_check_ranges() -> dict[str, dict[str, Any]]:
    return {
        "escape_max_constraint_residual": {"minimum": 0.0, "maximum": TOLERANCES["constraint_atol"], "units": "variance"},
        "escape_max_determinant_lemma_error": {"minimum": 0.0, "maximum": TOLERANCES["determinant_lemma_atol"], "units": "log determinant"},
        "escape_logdet_gain": {"minimum": 10.0, "maximum": 20.0, "units": "log determinant"},
        "duplicate_rank": {"minimum": 1.0, "maximum": 1.0, "units": "rank"},
        "duplicate_nullity": {"minimum": 1.0, "maximum": 1.0, "units": "nullity"},
        "full_span_observed_maximum_off_diagonal": {"minimum": 0.0, "maximum": 0.0, "units": "covariance"},
        "full_span_maximum_error": {"minimum": 0.0, "maximum": TOLERANCES["bound_atol"], "units": "log determinant"},
        "infeasible_linear_residual": {"minimum": TOLERANCES["infeasible_residual_minimum"], "maximum": 0.51, "units": "variance"},
        "pair_pure_correlation_abs": {"minimum": 0.0, "maximum": TOLERANCES["pair_atol"], "units": "correlation"},
        "pair_appendix_correlation_error": {"minimum": 0.0, "maximum": TOLERANCES["pair_atol"], "units": "correlation"},
        "pair_objective_advantage_error": {"minimum": 0.0, "maximum": TOLERANCES["pair_atol"], "units": "log determinant"},
        "pair_rank_domain_rejection_count": {"minimum": 3.0, "maximum": 3.0, "units": "operations"},
        "tau_correlation_range": {"minimum": 0.75, "maximum": 0.9, "units": "correlation"},
        "tau_max_stationarity_residual": {"minimum": 0.0, "maximum": TOLERANCES["stationarity_atol"], "units": "precision"},
        "tau_max_constraint_residual": {"minimum": 0.0, "maximum": TOLERANCES["constraint_atol"], "units": "variance"},
        "seeded_case_count": {"minimum": float(len(SEEDED_CASES)), "maximum": float(len(SEEDED_CASES)), "units": "cases"},
        "seeded_failure_count": {"minimum": 0.0, "maximum": 0.0, "units": "cases"},
    }


def build_checks(
    rank_escape: Mapping[str, Any],
    duplicate: Mapping[str, Any],
    full_span: Mapping[str, Any],
    infeasible: Mapping[str, Any],
    pair: Mapping[str, Any],
    pair_rank_domain: Mapping[str, Any],
    tau: Mapping[str, Any],
    seeded: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    observed = {
        "escape_max_constraint_residual": rank_escape["max_constraint_residual"],
        "escape_max_determinant_lemma_error": rank_escape["max_determinant_lemma_error"],
        "escape_logdet_gain": rank_escape["logdet_gain"],
        "duplicate_rank": duplicate["rank"],
        "duplicate_nullity": duplicate["nullity"],
        "full_span_observed_maximum_off_diagonal": abs(full_span["observed_maximum_off_diagonal"]),
        "full_span_maximum_error": abs(full_span["observed_maximum_logdet"] - full_span["exact_fixture_maximum_logdet"]),
        "infeasible_linear_residual": infeasible["minimum_linear_constraint_residual"],
        "pair_pure_correlation_abs": abs(pair["pure_correlation"]),
        "pair_appendix_correlation_error": abs(pair["appendix_correlation"] - 0.5),
        "pair_objective_advantage_error": abs(pair["pure_objective_advantage"] - pair["expected_objective_advantage"]),
        "pair_rank_domain_rejection_count": pair_rank_domain["rejection_count"],
        "tau_correlation_range": tau["correlation_range"],
        "tau_max_stationarity_residual": tau["max_stationarity_residual"],
        "tau_max_constraint_residual": tau["max_constraint_residual"],
        "seeded_case_count": seeded["case_count"],
        "seeded_failure_count": seeded["failure_count"],
    }
    checks: dict[str, dict[str, Any]] = {}
    for name, expected in _expected_check_ranges().items():
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
    expected_names = set(_expected_check_ranges())
    if set(checks) != expected_names:
        raise AuditFailure(
            "check registry mismatch: "
            f"missing={sorted(expected_names - set(checks))}, "
            f"unexpected={sorted(set(checks) - expected_names)}"
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
        computed = (
            math.isfinite(value)
            and math.isfinite(minimum)
            and math.isfinite(maximum)
            and minimum <= maximum
            and minimum <= value <= maximum
        )
        if record.get("passed") is not computed or not computed:
            failures.append(
                f"{name}: value={value!r}, expected=[{minimum!r}, {maximum!r}]"
            )
    if failures:
        raise AuditFailure("Part 4 maximum-entropy audit failed: " + "; ".join(failures))


def _configuration(seed: int) -> dict[str, Any]:
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return {
        "dtype": "float64",
        "seed": seed,
        "escape_amounts": list(ESCAPE_AMOUNTS),
        "duplicate_escape_amounts": list(DUPLICATE_ESCAPE_AMOUNTS),
        "full_span_off_diagonals": list(FULL_SPAN_OFF_DIAGONALS),
        "tau_grid": list(TAU_GRID),
        "seeded_cases": [
            {"dimension": dimension, "rank": rank}
            for dimension, rank in SEEDED_CASES
        ],
        "seeded_escape_amount": SEEDED_ESCAPE_AMOUNT,
        "tolerances": dict(TOLERANCES),
    }


def _reproduction_scope() -> dict[str, Any]:
    return {
        "reproduction_scope": "independent_mathematical_audit",
        "analysis_role": "counterexample_and_boundary_verification",
        "paper_experiment_reproduced": False,
        "checkpoint_or_dataset_used": False,
        "includes": [
            "pure log-determinant rank boundary",
            "positive-definite nullspace escape paths",
            "full-spanning bounded control",
            "infeasible duplicate constraint control",
            "cosine-one-half local pair objective comparison",
            "collinear pair rank-domain rejection",
            "explicit isotropic-reference scale sensitivity",
        ],
        "excludes": [
            "trained-network behavior",
            "paper compression metrics",
            "global pair-kernel positive semidefiniteness",
            "merge or pruning action quality",
        ],
    }


def _paper_identity(paper_pdf: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with PAPER_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("pdf_sha256") != PAPER_SHA256 or manifest.get("version") != "v1":
        raise AuditFailure("paper manifest is not pinned to the required v1 checksum")
    inputs = [
        {
            "kind": "paper_manifest",
            "path": "paper/manifest.json",
            "sha256": _sha256_path(PAPER_MANIFEST_PATH),
            "bytes": PAPER_MANIFEST_PATH.stat().st_size,
        }
    ]
    if paper_pdf is None:
        local_pdf = {"status": "not_supplied", "filename": None, "sha256": None, "bytes": None}
    else:
        resolved = paper_pdf.resolve()
        if not resolved.is_file():
            raise AuditFailure(f"explicit paper PDF is not a regular file: {paper_pdf}")
        digest = _sha256_path(resolved)
        if digest != PAPER_SHA256:
            raise AuditFailure(
                f"paper PDF checksum mismatch: expected {PAPER_SHA256}, got {digest}"
            )
        local_pdf = {
            "status": "verified",
            "filename": resolved.name,
            "sha256": digest,
            "bytes": resolved.stat().st_size,
        }
        paper_input_path = _paper_input_identity(resolved)
        inputs.append(
            {
                "kind": "source_paper_pdf",
                "path": paper_input_path,
                "sha256": digest,
                "bytes": resolved.stat().st_size,
            }
        )
    paper = {
        "arxiv_id": manifest["arxiv_id"],
        "version": manifest["version"],
        "title": manifest["title"],
        "pdf_filename": manifest["pdf_filename"],
        "pdf_sha256": PAPER_SHA256,
        "anchors": [
            "Section 4, PDF pages 6-7",
            "Appendix E.3.1 and Proposition E.3, PDF page 54",
            "Eqs. 80-81, PDF page 54",
            "Eq. 82, PDF page 55",
        ],
        "local_pdf": local_pdf,
    }
    return paper, inputs


def _paper_input_identity(path: Path) -> str:
    """Return a repository-relative or explicitly external paper identity."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(Path("external-paper") / resolved.name)


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
    source_status = _git_value("status", "--porcelain", "--", *SOURCE_RELATIVE_PATHS)
    matches_commit = source_status == ""
    return {
        "source_snapshot_commit": commit,
        "source_snapshot_state": (
            "exact_commit"
            if matches_commit
            else "working_tree_with_individually_hashed_sources"
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


def _manifest_output_path(path: Path, result_dir: Path) -> str:
    """Return a stable bounded output identity without leaking host paths."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        try:
            relative = path.resolve().relative_to(result_dir.resolve())
        except ValueError as error:
            raise AuditFailure("output path is outside the declared result directory") from error
        return str(Path("external-result") / relative)


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


def _build_sections(seed: int) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    rank_escape, rank_escape_rows = build_rank_deficient_escape()
    duplicate, duplicate_rows = build_duplicate_count_vs_rank()
    full_span, bounded_rows = build_full_span_control()
    infeasible = build_infeasible_control()
    pair, pair_rows = build_cosine_half_pair()
    pair_rank_domain = build_pair_rank_domain()
    tau, tau_rows = build_tau_scale_sweep()
    seeded, rank_rows = build_seeded_rank_cases(seed)
    checks = build_checks(
        rank_escape,
        duplicate,
        full_span,
        infeasible,
        pair,
        pair_rank_domain,
        tau,
        seeded,
    )
    sections = {
        "checks": checks,
        "rank_deficient_escape": rank_escape,
        "duplicate_count_vs_rank": duplicate,
        "full_span_control": full_span,
        "infeasible_control": infeasible,
        "cosine_half_pair": pair,
        "pair_rank_domain": pair_rank_domain,
        "tau_scale_sweep": tau,
        "seeded_rank_cases": seeded,
    }
    rows = {
        "escape_path.csv": rank_escape_rows + duplicate_rows,
        "bounded_control.csv": bounded_rows,
        "pair_comparison.csv": pair_rows,
        "tau_sweep.csv": tau_rows,
        "rank_cases.csv": rank_rows,
    }
    return sections, rows


def validate_summary(payload: Mapping[str, Any]) -> None:
    """Rebuild every deterministic section and reject altered summaries."""
    if not isinstance(payload, Mapping):
        raise AuditFailure("summary must be a mapping")
    _require_finite_payload(payload)
    required = {
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
        "checks",
        "rank_deficient_escape",
        "duplicate_count_vs_rank",
        "full_span_control",
        "infeasible_control",
        "cosine_half_pair",
        "pair_rank_domain",
        "tau_scale_sweep",
        "seeded_rank_cases",
    }
    if set(payload) != required:
        raise AuditFailure(
            f"summary schema mismatch: missing={sorted(required - set(payload))}, "
            f"unexpected={sorted(set(payload) - required)}"
        )
    if payload["schema_version"] != AUDIT_SCHEMA_VERSION or payload["audit_id"] != AUDIT_ID:
        raise AuditFailure("summary identity mismatch")
    if payload["status"] != "passed" or payload["passed"] is not True:
        raise AuditFailure("summary does not report a passing audit")
    if tuple(payload["claim_ids"]) != CLAIM_IDS:
        raise AuditFailure("summary claim registry mismatch")
    if payload["method_origins"] != METHOD_ORIGINS:
        raise AuditFailure("summary method-origin registry mismatch")
    if payload["reproduction_scope"] != _reproduction_scope():
        raise AuditFailure("summary reproduction scope mismatch")
    paper = payload["paper"]
    if not isinstance(paper, Mapping) or paper.get("pdf_sha256") != PAPER_SHA256:
        raise AuditFailure("summary paper identity mismatch")
    try:
        seed = int(payload["config"]["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise AuditFailure("summary seed is malformed") from error
    expected_config = _configuration(seed)
    if payload["config"] != expected_config:
        raise AuditFailure("summary configuration mismatch")
    if payload["config_sha256"] != _sha256_json(expected_config):
        raise AuditFailure("summary configuration hash mismatch")
    expected_sections, _ = _build_sections(seed)
    for name, expected in expected_sections.items():
        if payload[name] != expected:
            raise AuditFailure(f"summary deterministic section mismatch: {name}")
    enforce_checks(payload["checks"])


def run_audit(
    *,
    result_dir: Path = DEFAULT_RESULT_DIR,
    manifest_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    paper_pdf: Path | None = None,
) -> dict[str, Any]:
    """Run all fixtures and write deterministic results plus an authenticated manifest."""
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    result_dir = result_dir.resolve()
    manifest_path = (
        result_dir / "manifest.json"
        if manifest_path is None
        else manifest_path.resolve()
    )
    reserved = {
        result_dir / "summary.json",
        result_dir / "escape_path.csv",
        result_dir / "bounded_control.csv",
        result_dir / "pair_comparison.csv",
        result_dir / "tau_sweep.csv",
        result_dir / "rank_cases.csv",
    }
    if manifest_path in reserved:
        raise ValueError("manifest_path must not replace a result file")

    paper, paper_inputs = _paper_identity(paper_pdf)
    config = _configuration(seed)
    sections, csv_rows = _build_sections(seed)
    enforce_checks(sections["checks"])
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "passed": True,
        "claim_ids": list(CLAIM_IDS),
        "paper": paper,
        "config": config,
        "config_sha256": _sha256_json(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": _reproduction_scope(),
        **sections,
    }
    validate_summary(summary)

    summary_path = result_dir / "summary.json"
    csv_contracts = {
        "escape_path.csv": ESCAPE_FIELDS,
        "bounded_control.csv": BOUNDED_FIELDS,
        "pair_comparison.csv": PAIR_FIELDS,
        "tau_sweep.csv": TAU_FIELDS,
        "rank_cases.csv": RANK_FIELDS,
    }
    _write_json(summary_path, summary)
    for filename, fields in csv_contracts.items():
        _write_csv(result_dir / filename, fields, csv_rows[filename])

    output_paths = [
        summary_path,
        *(result_dir / filename for filename in csv_contracts),
    ]
    outputs = [
        {
            "path": _manifest_output_path(path, result_dir),
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
            "media_type": "application/json" if path.suffix == ".json" else "text/csv",
        }
        for path in output_paths
    ]
    expected_ranges = {
        name: {
            "minimum": record["minimum"],
            "maximum": record["maximum"],
            "units": record["units"],
        }
        for name, record in _expected_check_ranges().items()
    }
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "generation_time_policy": "wall clock omitted for deterministic output",
        "paper": paper,
        "config": config,
        "config_sha256": summary["config_sha256"],
        "seeds": {"seeded_rank_cases": seed, "analytical_fixtures": "no random seed used"},
        "code": _code_identity(),
        "environment": _environment(),
        "expected_metric_ranges": expected_ranges,
        "observed_checks": sections["checks"],
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": _reproduction_scope(),
        "source_hashes": _source_hashes(),
        "input_hashes": [
            *paper_inputs,
            {
                "kind": "embedded_deterministic_fixtures",
                "path": "summary.config",
                "sha256": summary["config_sha256"],
            },
        ],
        "outputs": outputs,
        "manifest_hash_scope": "manifest excludes its own hash to avoid self-reference",
        "canonical_invocation": {
            "entrypoint": "experiments/audits/part04_maximum_entropy/run.py",
            "result_dir": "<RESULT_DIR>",
            "manifest_path": "<MANIFEST_PATH>",
            "seed": seed,
            "paper_pdf": "verified v1 PDF" if paper_pdf is not None else "not supplied",
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "audit_id": AUDIT_ID,
        "status": "passed",
        "passed": True,
        "config_sha256": summary["config_sha256"],
        "summary": str(summary_path),
        "csv_outputs": [str(path) for path in output_paths if path.suffix == ".csv"],
        "manifest": str(manifest_path),
        "checks": sections["checks"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory for deterministic JSON and CSV results.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Manifest destination; defaults to RESULT_DIR/manifest.json.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        default=None,
        help="Optional local v1 PDF to authenticate before the audit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_audit(
        result_dir=args.result_dir,
        manifest_path=args.manifest_path,
        seed=args.seed,
        paper_pdf=args.paper_pdf,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
