#!/usr/bin/env python3
"""Deterministic audit of HOPE v1 merge objective versus physical execution."""

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
AUDIT_ID = "part07_merge_semantics_audit"
AUDIT_SCHEMA_VERSION = 1
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
PAPER_BYTES = 1_089_944
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_07" / "manual"
BN_EPSILON = 1e-5

CLAIM_IDS = (
    "P07-DUPLICATED-TARGET",
    "P07-EXECUTED-SUM-TRANSITION",
    "MERGE-TWIN-COUNTEREXAMPLE",
    "P07-CORRELATION-GAP",
    "P07-PAPER-SCALE-LIMIT",
    "P07-PHYSICAL-FORWARD-TWIN",
    "P07-NORMALIZED-SUM-REPAIR",
    "MERGE-SUM-REPAIR",
    "P07-NEAR-CANCELLATION",
    "P07-H4-DISTORTION-GAP",
)
CLAIM_CLASSES = {
    "P07-DUPLICATED-TARGET": "paper_claim",
    "P07-EXECUTED-SUM-TRANSITION": "verified_fact",
    "MERGE-TWIN-COUNTEREXAMPLE": "counterexample",
    "P07-CORRELATION-GAP": "counterexample",
    "P07-PAPER-SCALE-LIMIT": "verified_fact",
    "P07-PHYSICAL-FORWARD-TWIN": "verified_fact",
    "P07-NORMALIZED-SUM-REPAIR": "proposed_repair",
    "MERGE-SUM-REPAIR": "proposed_repair",
    "P07-NEAR-CANCELLATION": "counterexample",
    "P07-H4-DISTORTION-GAP": "independent_inference",
}

RHO_VALUES = (0.7, 0.9, 0.95, 0.99, 0.999)
OUTPUT_COSINES = (1.0, 0.8)
TWIN_REMAINING_RATIOS = (0.0, 0.5, 4.0, 64.0)
CAPACITY_RATIOS = (0.0, 0.25, 1.0, 4.0, 16.0, 64.0, 1e3, 1e6)

TOLERANCES = {
    "formula_atol": 2e-12,
    "forward_atol": 2e-12,
    "relative_atol": 2e-12,
    "limit_atol": 2e-6,
    "high_correlation_ratio_095_min": 4.49,
    "high_correlation_ratio_099_min": 10.04,
    "high_correlation_ratio_0999_min": 31.68,
    "near_cancellation_fraction_max": 0.0051,
    "near_cancellation_charge_ratio_min": 140.0,
}

METHOD_ORIGINS = {
    "duplicated_coordinate_objective": {
        "evidence_label": "paper_claim",
        "claim_id": "P07-DUPLICATED-TARGET",
        "origin": "HOPE v1 Eq. 6 and Eqs. 8-12",
        "scope": "distance from two children to two copies of one parent",
    },
    "executed_sum_transition": {
        "evidence_label": "verified_fact",
        "claim_id": "P07-EXECUTED-SUM-TRANSITION",
        "origin": "independent composition of Eq. 86 with Algorithm 2 lines 12-14",
        "scope": "one local additive layer contribution before and after one merge",
    },
    "exact_twin_witness": {
        "evidence_label": "counterexample",
        "claim_id": "MERGE-TWIN-COUNTEREXAMPLE",
        "origin": "independent exact Hilbert-space witness",
        "scope": "nonzero exact twins; refutes a universal executed-distortion interpretation",
    },
    "correlation_gap_fixtures": {
        "evidence_label": "counterexample",
        "claim_id": "P07-CORRELATION-GAP",
        "origin": "independent deterministic zero-bias ReLU fixtures",
        "scope": "declared input correlations, output alignments, and remaining-capacity rule only",
    },
    "normalized_sum_scale": {
        "evidence_label": "proposed_repair",
        "claim_id": "P07-NORMALIZED-SUM-REPAIR",
        "origin": "independent minimization of the normalized physical sum residual",
        "scope": "fixed parent direction, E_rem>=0, b>0",
    },
    "unnormalized_sum_scale": {
        "evidence_label": "proposed_repair",
        "claim_id": "MERGE-SUM-REPAIR",
        "origin": "independent least-squares projection of f_i+f_j onto a fixed direction",
        "scope": "fixed parent direction; exact preservation only when the sum lies in that direction",
    },
    "physical_forward_fixture": {
        "evidence_label": "verified_fact",
        "claim_id": "P07-PHYSICAL-FORWARD-TWIN",
        "origin": "independent NumPy realization of HOPE v1 Eqs. 15-18",
        "scope": "two deterministic dense-BN-ReLU-dense exact-twin fixtures",
    },
}

REPRODUCTION_SCOPE = {
    "reproduction_scope": "independent_mathematical_audit",
    "paper_experiment_reproduced": False,
    "checkpoint_or_dataset_used": False,
    "random_sampling_used": False,
    "includes": [
        "paper duplicated-coordinate merge algebra",
        "executed local sum-map residual",
        "exact and biased twin fixtures",
        "zero-bias correlation and output-alignment sweeps",
        "remaining-capacity scale limit",
        "near-cancellation fixture",
        "paper-literal physical parent reconstruction",
        "normalized and unnormalized fixed-direction repair comparisons",
    ],
    "excludes": [
        "ResNet-50 or ImageNet reproduction",
        "trained-network incidence or compression accuracy",
        "global-kernel validity",
        "parent-direction optimality outside the declared fixtures",
        "merge correlation-threshold runtime enforcement",
        "the adapted prune fallback",
        "DEFT vessel reuse",
        "Parts 8 through 12",
    ],
    "claim_boundary": (
        "The exact-twin witness concerns the summed local next-affine contribution. "
        "It does not by itself quantify final logits, loss, accuracy, or prevalence."
    ),
}

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part07_merge_semantics/README.md",
    "experiments/audits/part07_merge_semantics/run.py",
    "experiments/audits/part07_merge_semantics/test_run.py",
)

SOURCE_MAP_FIELDS = (
    "step_id", "sequence", "paper_anchor", "priced_or_executed_object",
    "formula_or_action", "result", "evidence_label", "claim_id", "passed",
)
TWIN_FIELDS = (
    "case_id", "beta", "gamma", "remaining_ratio", "q", "E_rem", "a", "b",
    "sum_norm_sq", "paper_scale", "normalized_scale", "unnormalized_scale",
    "paper_coordinate_distance", "paper_physical_distance",
    "normalized_physical_distance", "unnormalized_physical_distance",
    "h4_reported_delta", "h4_executed_gap", "evidence_label", "claim_id", "passed",
)
CORRELATION_FIELDS = (
    "case_id", "rho_input", "cos_output", "E_rem", "capacity_i", "capacity_j",
    "functional_cross", "sum_norm", "a", "b", "paper_scale",
    "paper_coordinate_distance", "paper_physical_distance",
    "true_to_reported_ratio", "normalized_scale", "normalized_physical_distance",
    "unnormalized_scale", "unnormalized_physical_distance", "evidence_label",
    "claim_id", "passed",
)
CAPACITY_FIELDS = (
    "case_id", "remaining_ratio", "E_rem", "a", "b", "sum_norm_sq",
    "paper_scale", "paper_scale_over_b", "paper_half_b_error",
    "normalized_scale", "normalized_scale_over_b", "normalized_b_error",
    "paper_coordinate_distance", "paper_physical_distance", "evidence_label",
    "claim_id", "passed",
)
REPAIR_FIELDS = (
    "case_id", "fixture", "method", "rho_input", "cos_output", "E_rem", "a", "b",
    "sum_norm", "scale", "coordinate_distance", "physical_distance",
    "normalized_objective", "physical_improvement_over_paper",
    "objective_improvement_over_paper", "evidence_label", "claim_id", "passed",
)
FORWARD_FIELDS = (
    "case_id", "fixture", "method", "beta", "gamma", "paper_capacity_q",
    "selected_scale", "expected_scale", "effective_weight_max_error",
    "effective_bias_error", "surrogate_scale_error", "forward_error_frobenius",
    "forward_relative_error", "forward_max_abs_error", "evidence_label", "claim_id",
    "passed",
)

CSV_CONTRACTS = {
    "source_map.csv": SOURCE_MAP_FIELDS,
    "twin_cases.csv": TWIN_FIELDS,
    "correlation_sweep.csv": CORRELATION_FIELDS,
    "capacity_sweep.csv": CAPACITY_FIELDS,
    "repair_comparison.csv": REPAIR_FIELDS,
    "physical_forward.csv": FORWARD_FIELDS,
}
OUTPUT_FILENAMES = ("summary.json", *CSV_CONTRACTS.keys())

CHECK_SPECS = {
    "nonfinite_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "row_failure_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "evidence_class_mismatch_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "twin_paper_scale_max_error": {"minimum": 0.0, "maximum": 2e-12, "units": "absolute"},
    "twin_reported_distance_max": {"minimum": 0.0, "maximum": 2e-12, "units": "capacity"},
    "twin_executed_gap_max_error": {"minimum": 0.0, "maximum": 2e-12, "units": "capacity"},
    "twin_normalized_scale_max_error": {"minimum": 0.0, "maximum": 2e-12, "units": "capacity"},
    "twin_unnormalized_scale_max_error": {"minimum": 0.0, "maximum": 2e-12, "units": "capacity"},
    "twin_repair_residual_max": {"minimum": 0.0, "maximum": 2e-12, "units": "capacity"},
    "forward_literal_relative_error_deviation": {"minimum": 0.0, "maximum": 2e-12, "units": "relative"},
    "forward_repair_relative_error_max": {"minimum": 0.0, "maximum": 2e-12, "units": "relative"},
    "forward_effective_reconstruction_max_error": {"minimum": 0.0, "maximum": 2e-12, "units": "absolute"},
    "correlation_ratio_rho_095": {"minimum": 4.49, "maximum": 4.51, "units": "ratio"},
    "correlation_ratio_rho_099": {"minimum": 10.04, "maximum": 10.06, "units": "ratio"},
    "correlation_ratio_rho_0999": {"minimum": 31.68, "maximum": 31.70, "units": "ratio"},
    "unnormalized_improvement_rho_095": {"minimum": 17.0, "maximum": 17.1, "units": "ratio"},
    "unnormalized_improvement_rho_099": {"minimum": 57.5, "maximum": 57.6, "units": "ratio"},
    "unnormalized_improvement_rho_0999": {"minimum": 324.0, "maximum": 325.0, "units": "ratio"},
    "normalized_objective_order_failures": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "paper_half_b_limit_error": {"minimum": 0.0, "maximum": 2e-6, "units": "relative"},
    "normalized_b_limit_error": {"minimum": 0.0, "maximum": 2e-6, "units": "relative"},
    "near_cancellation_fraction": {"minimum": 0.0050, "maximum": 0.0051, "units": "fraction"},
    "near_cancellation_charge_ratio": {"minimum": 140.0, "maximum": 142.0, "units": "ratio"},
    "h4_zero_delta_executed_gap": {"minimum": 0.1, "maximum": 10.0, "units": "capacity"},
}


class AuditFailure(RuntimeError):
    """Raised when a declared audit invariant fails."""


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sqrt_nonnegative(value: float, name: str, *, scale: float = 1.0) -> float:
    value = _finite(value, name)
    scale = max(1.0, abs(_finite(scale, f"{name} scale")))
    roundoff = TOLERANCES["formula_atol"] * scale
    if value < -roundoff:
        raise AuditFailure(f"materially negative squared quantity: {name}={value}")
    if abs(value) <= roundoff:
        return 0.0
    return math.sqrt(max(0.0, value))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def relu_self_kernel(beta: float, gamma: float) -> float:
    beta = _finite(beta, "beta")
    scale = abs(_finite(gamma, "gamma"))
    if scale <= 1e-15:
        return max(0.0, beta) ** 2
    standardized = beta / scale
    return (
        (scale * scale + beta * beta) * normal_cdf(standardized)
        + beta * scale * normal_pdf(standardized)
    )


def relu_arc_kernel(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    norm_first = float(np.linalg.norm(first))
    norm_second = float(np.linalg.norm(second))
    if norm_first <= 0.0 or norm_second <= 0.0:
        return 0.0
    correlation = float(np.clip(first @ second / (norm_first * norm_second), -1.0, 1.0))
    angle = math.acos(correlation)
    return norm_first * norm_second * (
        math.sin(angle) + (math.pi - angle) * math.cos(angle)
    ) / (2.0 * math.pi)


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-15:
        raise AuditFailure(f"cannot normalize degenerate {name}")
    return vector / norm


def paper_parent_direction(
    incoming_i: np.ndarray,
    incoming_j: np.ndarray,
    outgoing_i: np.ndarray,
    outgoing_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply Eq. 14 and the exact zero-bias phase check from Eq. 11."""
    matrix = np.outer(outgoing_i, incoming_i) + np.outer(outgoing_j, incoming_j)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.T @ matrix)
    if eigenvalues[-1] <= 1e-24:
        candidate = _unit(incoming_i, "fallback incoming direction")
    else:
        candidate = eigenvectors[:, -1]

    def objective(direction: np.ndarray) -> tuple[float, np.ndarray]:
        self_value = relu_arc_kernel(direction, direction)
        z = (
            relu_arc_kernel(direction, incoming_i) * outgoing_i
            + relu_arc_kernel(direction, incoming_j) * outgoing_j
        )
        return float(np.linalg.norm(z)) / math.sqrt(self_value), z

    positive, z_positive = objective(candidate)
    negative, z_negative = objective(-candidate)
    if negative > positive:
        candidate, alignment, z = -candidate, negative, z_negative
    else:
        alignment, z = positive, z_positive
    return candidate, _unit(z, "parent output direction"), alignment


def pair_geometry(
    rho_input: float,
    cos_output: float,
    *,
    output_scale_j: float = 1.0,
) -> dict[str, Any]:
    rho_input = _finite(rho_input, "rho_input")
    cos_output = _finite(cos_output, "cos_output")
    if not -1.0 <= rho_input <= 1.0 or not -1.0 <= cos_output <= 1.0:
        raise ValueError("fixture correlations must lie in [-1,1]")
    incoming_i = np.array([1.0, 0.0], dtype=np.float64)
    incoming_j = np.array(
        [rho_input, math.sqrt(max(0.0, 1.0 - rho_input * rho_input))],
        dtype=np.float64,
    )
    outgoing_i = np.array([1.0, 0.0], dtype=np.float64)
    outgoing_j = output_scale_j * np.array(
        [cos_output, math.sqrt(max(0.0, 1.0 - cos_output * cos_output))],
        dtype=np.float64,
    )
    direction, output_direction, b = paper_parent_direction(
        incoming_i, incoming_j, outgoing_i, outgoing_j
    )
    self_i = relu_arc_kernel(incoming_i, incoming_i)
    self_j = relu_arc_kernel(incoming_j, incoming_j)
    cross_kernel = relu_arc_kernel(incoming_i, incoming_j)
    capacity_i = math.sqrt(self_i) * float(np.linalg.norm(outgoing_i))
    capacity_j = math.sqrt(self_j) * float(np.linalg.norm(outgoing_j))
    functional_cross = cross_kernel * float(outgoing_i @ outgoing_j)
    a = capacity_i * capacity_i + capacity_j * capacity_j
    sum_norm_sq = a + 2.0 * functional_cross
    return {
        "incoming_i": incoming_i,
        "incoming_j": incoming_j,
        "outgoing_i": outgoing_i,
        "outgoing_j": outgoing_j,
        "direction": direction,
        "output_direction": output_direction,
        "capacity_i": capacity_i,
        "capacity_j": capacity_j,
        "functional_cross": functional_cross,
        "a": a,
        "b": b,
        "sum_norm_sq": sum_norm_sq,
        "sum_norm": _sqrt_nonnegative(
            sum_norm_sq,
            "pair sum norm squared",
            scale=max(a, abs(2.0 * functional_cross)),
        ),
    }


def scale_metrics(a: float, b: float, sum_norm_sq: float, E_rem: float) -> dict[str, Any]:
    a = _finite(a, "a")
    b = _finite(b, "b")
    sum_norm_sq = _finite(sum_norm_sq, "sum_norm_sq")
    E_rem = _finite(E_rem, "E_rem")
    if a <= 0.0 or b <= 0.0 or sum_norm_sq < 0.0 or E_rem < 0.0:
        raise ValueError("scale fixture is outside the positive domain")
    paper_scale = (a + b * E_rem) / (2.0 * E_rem + b)
    normalized_scale = (b * E_rem + sum_norm_sq) / (E_rem + b)
    unnormalized_scale = b

    coordinate_floor = a - 0.5 * b * b
    physical_floor = sum_norm_sq - b * b

    def coordinate_distance(scale: float) -> float:
        return _sqrt_nonnegative(
            2.0 * (scale - 0.5 * b) ** 2 + coordinate_floor,
            "duplicated-coordinate distance squared",
            scale=max(a, b * b, scale * scale),
        )

    def physical_distance(scale: float) -> float:
        return _sqrt_nonnegative(
            (scale - b) ** 2 + physical_floor,
            "physical sum distance squared",
            scale=max(sum_norm_sq, b * b, scale * scale),
        )

    return {
        "paper_scale": paper_scale,
        "normalized_scale": normalized_scale,
        "unnormalized_scale": unnormalized_scale,
        "paper_coordinate_distance": coordinate_distance(paper_scale),
        "paper_physical_distance": physical_distance(paper_scale),
        "normalized_coordinate_distance": coordinate_distance(normalized_scale),
        "normalized_physical_distance": physical_distance(normalized_scale),
        "unnormalized_coordinate_distance": coordinate_distance(unnormalized_scale),
        "unnormalized_physical_distance": physical_distance(unnormalized_scale),
    }


def build_config(*, full: bool) -> dict[str, Any]:
    return {
        "float_dtype": "float64",
        "full_mode": bool(full),
        "random_sampling": False,
        "bn_epsilon": BN_EPSILON,
        "rho_values": list(RHO_VALUES),
        "output_cosines": list(OUTPUT_COSINES),
        "twin_remaining_ratios": list(TWIN_REMAINING_RATIOS),
        "capacity_ratios": list(CAPACITY_RATIOS),
        "correlation_remaining_capacity_rule": "E_rem=32*(q_i+q_j)",
        "near_cancellation_output_scale": -0.99,
        "physical_inputs": [
            [-2.0, 0.5, 1.0], [-1.0, -0.5, 0.25], [-0.25, 0.75, -1.5],
            [0.0, 0.0, 0.0], [0.5, -1.0, 0.75], [1.0, 0.25, -0.5],
            [2.0, -0.75, 1.25],
        ],
        "tolerances": dict(TOLERANCES),
    }


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def _source_map_rows() -> list[dict[str, Any]]:
    specifications = [
        (
            "duplicated_target", "Eq. 6 and Eq. 8, PDF page 13", "priced",
            "[f_i,f_j] -> [f_p,f_p]", "the numerator contains two child-to-parent residuals",
            "paper_claim", "P07-DUPLICATED-TARGET",
        ),
        (
            "paper_scale", "Eq. 12, PDF page 15", "priced",
            "s* equals (a+b E_rem) divided by (2 E_rem+b)",
            "the scale minimizes the duplicated-coordinate normalized objective",
            "paper_claim", "P07-DUPLICATED-TARGET",
        ),
        (
            "physical_parent", "Eqs. 15-18, PDF pages 15-16", "realized",
            "recover one physical parent", "the realization reproduces the selected one-parent operator",
            "verified_fact", "P07-PHYSICAL-FORWARD-TWIN",
        ),
        (
            "aggregate_sum", "Eq. 86, PDF page 57", "executed",
            "F(Phi)=sum_i f_i", "the local aggregate before surgery is f_i+f_j",
            "verified_fact", "P07-EXECUTED-SUM-TRANSITION",
        ),
        (
            "delete_child", "Algorithm 2 lines 12-14, PDF page 69", "executed",
            "retain parent in child i vessel and purge child j", "the local aggregate after surgery contains one parent",
            "verified_fact", "P07-EXECUTED-SUM-TRANSITION",
        ),
        (
            "exact_twins", "Eq. 6, Eq. 12, Eq. 86, Algorithm 2", "comparison",
            "[f,f] -> [f,f] -> [f,0]", "reported distance is zero but executed sum error is ||f||",
            "counterexample", "MERGE-TWIN-COUNTEREXAMPLE",
        ),
        (
            "correlation_gap", "bounded independent fixtures", "comparison",
            "D_executed over D_reported under fixed rho, output alignment, and E_rem",
            "the ratio grows from about 4.50 to 31.69 on the declared aligned grid",
            "counterexample", "P07-CORRELATION-GAP",
        ),
        (
            "normalized_repair", "independent derivation", "repair",
            "s equals (b E_rem+||f_i+f_j||^2) divided by (E_rem+b)",
            "minimizes the normalized physical sum residual for fixed psi",
            "proposed_repair", "P07-NORMALIZED-SUM-REPAIR",
        ),
        (
            "unnormalized_repair", "independent derivation", "repair",
            "s=b", "minimizes the unnormalized physical sum residual for fixed psi",
            "proposed_repair", "MERGE-SUM-REPAIR",
        ),
        (
            "h4_delta", "Proposition H.4 and Eqs. 116-118, PDF pages 66-67", "downstream claim",
            "delta_k=min_fp sqrt(sum_m ||f_m-f_p||^2)", "the displayed delta is the duplicated-coordinate error, not the executed aggregate residual",
            "independent_inference", "P07-H4-DISTORTION-GAP",
        ),
    ]
    return [
        {
            "step_id": step_id,
            "sequence": sequence,
            "paper_anchor": anchor,
            "priced_or_executed_object": object_kind,
            "formula_or_action": formula,
            "result": result,
            "evidence_label": evidence_label,
            "claim_id": claim_id,
            "passed": True,
        }
        for sequence, (
            step_id, anchor, object_kind, formula, result, evidence_label, claim_id
        ) in enumerate(specifications, start=1)
    ]


def _twin_rows() -> list[dict[str, Any]]:
    cases = (
        ("zero_shift", 0.0, 1.2, np.array([0.8, -0.4, 0.2])),
        ("biased", 0.35, 1.4, np.array([0.6, -0.3, 0.5])),
    )
    rows: list[dict[str, Any]] = []
    for name, beta, gamma, outgoing in cases:
        q = float(np.linalg.norm(outgoing)) * math.sqrt(relu_self_kernel(beta, gamma))
        a, b, sum_norm_sq = 2.0 * q * q, 2.0 * q, 4.0 * q * q
        for ratio in TWIN_REMAINING_RATIOS:
            E_rem = ratio * q
            metrics = scale_metrics(a, b, sum_norm_sq, E_rem)
            passed = (
                abs(metrics["paper_scale"] - q) <= TOLERANCES["formula_atol"]
                and metrics["paper_coordinate_distance"] <= TOLERANCES["formula_atol"]
                and abs(metrics["paper_physical_distance"] - q) <= TOLERANCES["formula_atol"]
                and abs(metrics["normalized_scale"] - 2.0 * q) <= TOLERANCES["formula_atol"]
                and abs(metrics["unnormalized_scale"] - 2.0 * q) <= TOLERANCES["formula_atol"]
                and metrics["normalized_physical_distance"] <= TOLERANCES["formula_atol"]
                and metrics["unnormalized_physical_distance"] <= TOLERANCES["formula_atol"]
            )
            rows.append({
                "case_id": f"{name}_er{ratio:g}",
                "beta": beta,
                "gamma": gamma,
                "remaining_ratio": ratio,
                "q": q,
                "E_rem": E_rem,
                "a": a,
                "b": b,
                "sum_norm_sq": sum_norm_sq,
                "paper_scale": metrics["paper_scale"],
                "normalized_scale": metrics["normalized_scale"],
                "unnormalized_scale": metrics["unnormalized_scale"],
                "paper_coordinate_distance": metrics["paper_coordinate_distance"],
                "paper_physical_distance": metrics["paper_physical_distance"],
                "normalized_physical_distance": metrics["normalized_physical_distance"],
                "unnormalized_physical_distance": metrics["unnormalized_physical_distance"],
                "h4_reported_delta": metrics["paper_coordinate_distance"],
                "h4_executed_gap": metrics["paper_physical_distance"],
                "evidence_label": "counterexample",
                "claim_id": "MERGE-TWIN-COUNTEREXAMPLE",
                "passed": passed,
            })
    return rows


def _correlation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cos_output in OUTPUT_COSINES:
        for rho_input in RHO_VALUES:
            geometry = pair_geometry(rho_input, cos_output)
            E_rem = 32.0 * (geometry["capacity_i"] + geometry["capacity_j"])
            metrics = scale_metrics(
                geometry["a"], geometry["b"], geometry["sum_norm_sq"], E_rem
            )
            ratio = metrics["paper_physical_distance"] / max(
                metrics["paper_coordinate_distance"], TOLERANCES["formula_atol"]
            )
            rows.append({
                "case_id": f"rho{rho_input:g}_eta{cos_output:g}",
                "rho_input": rho_input,
                "cos_output": cos_output,
                "E_rem": E_rem,
                "capacity_i": geometry["capacity_i"],
                "capacity_j": geometry["capacity_j"],
                "functional_cross": geometry["functional_cross"],
                "sum_norm": geometry["sum_norm"],
                "a": geometry["a"],
                "b": geometry["b"],
                "paper_scale": metrics["paper_scale"],
                "paper_coordinate_distance": metrics["paper_coordinate_distance"],
                "paper_physical_distance": metrics["paper_physical_distance"],
                "true_to_reported_ratio": ratio,
                "normalized_scale": metrics["normalized_scale"],
                "normalized_physical_distance": metrics["normalized_physical_distance"],
                "unnormalized_scale": metrics["unnormalized_scale"],
                "unnormalized_physical_distance": metrics["unnormalized_physical_distance"],
                "evidence_label": "counterexample",
                "claim_id": "P07-CORRELATION-GAP",
                "passed": (
                    metrics["paper_coordinate_distance"] > 0.0
                    and metrics["paper_physical_distance"] > 0.0
                    and metrics["normalized_physical_distance"]
                    <= metrics["paper_physical_distance"] + TOLERANCES["formula_atol"]
                    and metrics["unnormalized_physical_distance"]
                    <= metrics["paper_physical_distance"] + TOLERANCES["formula_atol"]
                ),
            })
    return rows


def _capacity_rows() -> list[dict[str, Any]]:
    geometry = pair_geometry(0.95, 1.0)
    rows: list[dict[str, Any]] = []
    for ratio in CAPACITY_RATIOS:
        E_rem = ratio * geometry["b"]
        metrics = scale_metrics(
            geometry["a"], geometry["b"], geometry["sum_norm_sq"], E_rem
        )
        paper_half_error = abs(metrics["paper_scale"] / geometry["b"] - 0.5)
        normalized_b_error = abs(metrics["normalized_scale"] / geometry["b"] - 1.0)
        rows.append({
            "case_id": f"capacity_ratio_{ratio:g}",
            "remaining_ratio": ratio,
            "E_rem": E_rem,
            "a": geometry["a"],
            "b": geometry["b"],
            "sum_norm_sq": geometry["sum_norm_sq"],
            "paper_scale": metrics["paper_scale"],
            "paper_scale_over_b": metrics["paper_scale"] / geometry["b"],
            "paper_half_b_error": paper_half_error,
            "normalized_scale": metrics["normalized_scale"],
            "normalized_scale_over_b": metrics["normalized_scale"] / geometry["b"],
            "normalized_b_error": normalized_b_error,
            "paper_coordinate_distance": metrics["paper_coordinate_distance"],
            "paper_physical_distance": metrics["paper_physical_distance"],
            "evidence_label": "verified_fact",
            "claim_id": "P07-PAPER-SCALE-LIMIT",
            "passed": metrics["paper_scale"] > 0.0 and metrics["normalized_scale"] > 0.0,
        })
    return rows


def _repair_rows() -> list[dict[str, Any]]:
    fixtures = [
        ("aligned_rho_095", pair_geometry(0.95, 1.0), 32.0),
        ("aligned_rho_099", pair_geometry(0.99, 1.0), 32.0),
        ("aligned_rho_0999", pair_geometry(0.999, 1.0), 32.0),
        ("partial_rho_095", pair_geometry(0.95, 0.8), 32.0),
        ("near_cancellation", pair_geometry(1.0, -1.0, output_scale_j=0.99), 32.0),
    ]
    rows: list[dict[str, Any]] = []
    for fixture, geometry, ratio in fixtures:
        E_rem = ratio * (geometry["capacity_i"] + geometry["capacity_j"])
        metrics = scale_metrics(
            geometry["a"], geometry["b"], geometry["sum_norm_sq"], E_rem
        )
        baseline = metrics["paper_physical_distance"]
        methods = (
            (
                "paper_literal", metrics["paper_scale"],
                metrics["paper_coordinate_distance"], metrics["paper_physical_distance"],
                "counterexample",
                "P07-NEAR-CANCELLATION" if fixture == "near_cancellation" else "P07-CORRELATION-GAP",
            ),
            (
                "normalized_sum_repair", metrics["normalized_scale"],
                metrics["normalized_coordinate_distance"], metrics["normalized_physical_distance"],
                "proposed_repair", "P07-NORMALIZED-SUM-REPAIR",
            ),
            (
                "unnormalized_sum_repair", metrics["unnormalized_scale"],
                metrics["unnormalized_coordinate_distance"], metrics["unnormalized_physical_distance"],
                "proposed_repair", "MERGE-SUM-REPAIR",
            ),
        )
        method_objectives = {
            method: physical / (E_rem + scale)
            for method, scale, _coordinate, physical, _label, _claim in methods
        }
        paper_objective = method_objectives["paper_literal"]
        normalized_is_optimal = (
            method_objectives["normalized_sum_repair"]
            <= method_objectives["unnormalized_sum_repair"] + TOLERANCES["formula_atol"]
            and method_objectives["normalized_sum_repair"]
            <= paper_objective + TOLERANCES["formula_atol"]
        )
        for method, scale, coordinate, physical, label, claim in methods:
            normalized_objective = method_objectives[method]
            rows.append({
                "case_id": f"{fixture}_{method}",
                "fixture": fixture,
                "method": method,
                "rho_input": 1.0 if fixture == "near_cancellation" else (
                    0.999 if "0999" in fixture else 0.99 if "099" in fixture else 0.95
                ),
                "cos_output": -1.0 if fixture == "near_cancellation" else (
                    0.8 if fixture.startswith("partial") else 1.0
                ),
                "E_rem": E_rem,
                "a": geometry["a"],
                "b": geometry["b"],
                "sum_norm": geometry["sum_norm"],
                "scale": scale,
                "coordinate_distance": coordinate,
                "physical_distance": physical,
                "normalized_objective": normalized_objective,
                "physical_improvement_over_paper": (
                    None if physical <= TOLERANCES["formula_atol"]
                    else baseline / physical
                ),
                "objective_improvement_over_paper": (
                    None if normalized_objective <= TOLERANCES["formula_atol"]
                    else paper_objective / normalized_objective
                ),
                "evidence_label": label,
                "claim_id": claim,
                "passed": (
                    normalized_is_optimal if method == "normalized_sum_repair"
                    else physical <= baseline + TOLERANCES["formula_atol"]
                    if method == "unnormalized_sum_repair" else True
                ),
            })
    return rows


def _physical_fixture_definitions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "zero_shift",
            "w_raw": np.array([0.75, -0.25, 0.5], dtype=np.float64),
            "gamma": 1.2,
            "beta": 0.0,
            "mu": 0.0,
            "var": 1.2**2 - BN_EPSILON,
            "w_out": np.array([0.8, -0.4, 0.2], dtype=np.float64),
        },
        {
            "name": "biased",
            "w_raw": np.array([0.4, -0.7, 0.2], dtype=np.float64),
            "gamma": 1.4,
            "beta": 0.35,
            "mu": -0.2,
            "var": 1.4**2 - BN_EPSILON,
            "w_out": np.array([0.6, -0.3, 0.5], dtype=np.float64),
        },
    )


def _realize_twin_parent(fixture: Mapping[str, Any], selected_scale: float) -> dict[str, Any]:
    gamma = float(fixture["gamma"])
    beta = float(fixture["beta"])
    mu = float(fixture["mu"])
    variance = float(fixture["var"])
    w_raw = np.asarray(fixture["w_raw"], dtype=np.float64)
    w_out = np.asarray(fixture["w_out"], dtype=np.float64)
    bn_scale = gamma / math.sqrt(variance + BN_EPSILON)
    w_eff = bn_scale * w_raw
    b_eff = beta - bn_scale * mu
    w_aug = np.concatenate([w_eff, [b_eff]])
    aug_norm = float(np.linalg.norm(w_aug))
    out_norm = float(np.linalg.norm(w_out))
    K_child = relu_self_kernel(beta, gamma)
    K_uu = K_child / (aug_norm * aug_norm)
    ratio_frobenius = aug_norm / out_norm
    scale_in = math.sqrt(selected_scale * ratio_frobenius) * K_uu ** (-0.25)
    scale_out = math.sqrt(selected_scale / ratio_frobenius) * K_uu ** (-0.25)
    unit_in = w_aug / aug_norm
    unit_out = w_out / out_norm
    parent_aug = scale_in * unit_in
    parent_w_eff = parent_aug[:-1]
    parent_b_eff = float(parent_aug[-1])
    parent_w_out = scale_out * unit_out

    alpha = delta = 1.0 / (2.0 * aug_norm)
    coefficient_sum = scale_in * (alpha + delta)
    parent_beta = coefficient_sum * beta
    parent_gamma = abs(coefficient_sum) * abs(gamma)
    if parent_gamma * parent_gamma <= BN_EPSILON:
        raise AuditFailure("physical twin fixture reached the inactive parent boundary")
    parent_mu = parent_beta - parent_b_eff
    parent_var = parent_gamma * parent_gamma - BN_EPSILON
    parent_w_raw = (
        parent_w_eff * math.sqrt(parent_var + BN_EPSILON) / parent_gamma
    )
    recovered_scale = parent_gamma / math.sqrt(parent_var + BN_EPSILON)
    recovered_w_eff = recovered_scale * parent_w_raw
    recovered_b_eff = parent_beta - recovered_scale * parent_mu
    q = out_norm * math.sqrt(K_child)
    return {
        "q": q,
        "w_eff": w_eff,
        "b_eff": b_eff,
        "w_out": w_out,
        "parent_w_eff": recovered_w_eff,
        "parent_b_eff": recovered_b_eff,
        "parent_w_out": parent_w_out,
        "parent_gamma": parent_gamma,
        "parent_beta": parent_beta,
        "parent_mu": parent_mu,
        "parent_var": parent_var,
        "parent_w_raw": parent_w_raw,
        "parent_surrogate_scale": float(np.linalg.norm(parent_w_out)) * math.sqrt(
            relu_self_kernel(parent_beta, parent_gamma)
        ),
        "selected_scale": selected_scale,
    }


def _forward_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = np.asarray(config["physical_inputs"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for fixture in _physical_fixture_definitions():
        out_norm = float(np.linalg.norm(fixture["w_out"]))
        q = out_norm * math.sqrt(relu_self_kernel(fixture["beta"], fixture["gamma"]))
        scales = (
            ("paper_literal", q, "verified_fact", "P07-PHYSICAL-FORWARD-TWIN", 0.5),
            ("normalized_sum_repair", 2.0 * q, "proposed_repair", "P07-NORMALIZED-SUM-REPAIR", 0.0),
            ("unnormalized_sum_repair", 2.0 * q, "proposed_repair", "MERGE-SUM-REPAIR", 0.0),
        )
        base_scale = fixture["gamma"] / math.sqrt(fixture["var"] + BN_EPSILON)
        child_w_eff = base_scale * fixture["w_raw"]
        child_b_eff = fixture["beta"] - base_scale * fixture["mu"]
        child_activation = np.maximum(inputs @ child_w_eff + child_b_eff, 0.0)
        child_output = child_activation[:, None] * fixture["w_out"][None, :]
        reference = 2.0 * child_output
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm <= 0.0:
            raise AuditFailure("physical forward fixture is inactive on every declared input")
        for method, selected_scale, label, claim, expected_relative in scales:
            parent = _realize_twin_parent(fixture, selected_scale)
            parent_activation = np.maximum(
                inputs @ parent["parent_w_eff"] + parent["parent_b_eff"], 0.0
            )
            parent_output = (
                parent_activation[:, None] * parent["parent_w_out"][None, :]
            )
            error = parent_output - reference
            error_norm = float(np.linalg.norm(error))
            relative_error = error_norm / reference_norm
            expected_parent = child_output if method == "paper_literal" else reference
            effective_multiplier = math.sqrt(selected_scale / q)
            expected_w_eff = effective_multiplier * child_w_eff
            expected_b_eff = effective_multiplier * child_b_eff
            effective_error = float(np.max(np.abs(parent["parent_w_eff"] - expected_w_eff)))
            bias_error = abs(parent["parent_b_eff"] - expected_b_eff)
            surrogate_error = abs(parent["parent_surrogate_scale"] - selected_scale)
            max_abs_error = float(np.max(np.abs(parent_output - expected_parent)))
            passed = (
                abs(relative_error - expected_relative) <= TOLERANCES["relative_atol"]
                and effective_error <= TOLERANCES["forward_atol"]
                and bias_error <= TOLERANCES["forward_atol"]
                and surrogate_error <= TOLERANCES["formula_atol"]
                and max_abs_error <= TOLERANCES["forward_atol"]
            )
            rows.append({
                "case_id": f"{fixture['name']}_{method}",
                "fixture": fixture["name"],
                "method": method,
                "beta": fixture["beta"],
                "gamma": fixture["gamma"],
                "paper_capacity_q": q,
                "selected_scale": selected_scale,
                "expected_scale": q if method == "paper_literal" else 2.0 * q,
                "effective_weight_max_error": effective_error,
                "effective_bias_error": bias_error,
                "surrogate_scale_error": surrogate_error,
                "forward_error_frobenius": error_norm,
                "forward_relative_error": relative_error,
                "forward_max_abs_error": max_abs_error,
                "evidence_label": label,
                "claim_id": claim,
                "passed": passed,
            })
    return rows


def _walk_numbers(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    if isinstance(value, Mapping):
        return [number for child in value.values() for number in _walk_numbers(child)]
    if isinstance(value, (list, tuple)):
        return [number for child in value for number in _walk_numbers(child)]
    return []


def build_audit(*, full: bool = False) -> dict[str, Any]:
    config = build_config(full=full)
    source_rows = _source_map_rows()
    twin_rows = _twin_rows()
    correlation_rows = _correlation_rows()
    capacity_rows = _capacity_rows()
    repair_rows = _repair_rows()
    forward_rows = _forward_rows(config)
    all_row_groups = (
        source_rows, twin_rows, correlation_rows, capacity_rows, repair_rows, forward_rows
    )
    all_rows = [row for group in all_row_groups for row in group]

    evidence_mismatches = sum(
        row["evidence_label"] != CLAIM_CLASSES[row["claim_id"]] for row in all_rows
    ) + sum(
        record["evidence_label"] != CLAIM_CLASSES[record["claim_id"]]
        for record in METHOD_ORIGINS.values()
    )
    aligned_rows = {
        row["rho_input"]: row
        for row in correlation_rows if row["cos_output"] == 1.0
    }
    final_capacity = capacity_rows[-1]
    near_literal = next(
        row for row in repair_rows
        if row["fixture"] == "near_cancellation" and row["method"] == "paper_literal"
    )
    repair_by_case = {row["case_id"]: row for row in repair_rows}
    normalized_order_failures = sum(
        not row["passed"]
        for row in repair_rows if row["method"] == "normalized_sum_repair"
    )
    paper_forward = [row for row in forward_rows if row["method"] == "paper_literal"]
    repair_forward = [row for row in forward_rows if row["method"] != "paper_literal"]
    checks = {
        "nonfinite_count": float(sum(
            not math.isfinite(value) for value in _walk_numbers(all_rows)
        )),
        "row_failure_count": float(sum(not row["passed"] for row in all_rows)),
        "evidence_class_mismatch_count": float(evidence_mismatches),
        "twin_paper_scale_max_error": max(
            abs(row["paper_scale"] - row["q"]) for row in twin_rows
        ),
        "twin_reported_distance_max": max(
            row["paper_coordinate_distance"] for row in twin_rows
        ),
        "twin_executed_gap_max_error": max(
            abs(row["paper_physical_distance"] - row["q"]) for row in twin_rows
        ),
        "twin_normalized_scale_max_error": max(
            abs(row["normalized_scale"] - 2.0 * row["q"]) for row in twin_rows
        ),
        "twin_unnormalized_scale_max_error": max(
            abs(row["unnormalized_scale"] - 2.0 * row["q"]) for row in twin_rows
        ),
        "twin_repair_residual_max": max(
            max(row["normalized_physical_distance"], row["unnormalized_physical_distance"])
            for row in twin_rows
        ),
        "forward_literal_relative_error_deviation": max(
            abs(row["forward_relative_error"] - 0.5) for row in paper_forward
        ),
        "forward_repair_relative_error_max": max(
            row["forward_relative_error"] for row in repair_forward
        ),
        "forward_effective_reconstruction_max_error": max(
            max(
                row["effective_weight_max_error"], row["effective_bias_error"],
                row["surrogate_scale_error"], row["forward_max_abs_error"],
            ) for row in forward_rows
        ),
        "correlation_ratio_rho_095": aligned_rows[0.95]["true_to_reported_ratio"],
        "correlation_ratio_rho_099": aligned_rows[0.99]["true_to_reported_ratio"],
        "correlation_ratio_rho_0999": aligned_rows[0.999]["true_to_reported_ratio"],
        "unnormalized_improvement_rho_095": repair_by_case[
            "aligned_rho_095_unnormalized_sum_repair"
        ]["physical_improvement_over_paper"],
        "unnormalized_improvement_rho_099": repair_by_case[
            "aligned_rho_099_unnormalized_sum_repair"
        ]["physical_improvement_over_paper"],
        "unnormalized_improvement_rho_0999": repair_by_case[
            "aligned_rho_0999_unnormalized_sum_repair"
        ]["physical_improvement_over_paper"],
        "normalized_objective_order_failures": float(normalized_order_failures),
        "paper_half_b_limit_error": final_capacity["paper_half_b_error"],
        "normalized_b_limit_error": final_capacity["normalized_b_error"],
        "near_cancellation_fraction": near_literal["sum_norm"] / (
            math.sqrt(0.5) + 0.99 * math.sqrt(0.5)
        ),
        "near_cancellation_charge_ratio": near_literal["coordinate_distance"]
        / near_literal["sum_norm"],
        "h4_zero_delta_executed_gap": min(row["h4_executed_gap"] for row in twin_rows),
    }
    check_rows = {
        name: {
            "value": value,
            "minimum": CHECK_SPECS[name]["minimum"],
            "maximum": CHECK_SPECS[name]["maximum"],
            "units": CHECK_SPECS[name]["units"],
            "passed": CHECK_SPECS[name]["minimum"] <= value <= CHECK_SPECS[name]["maximum"],
        }
        for name, value in checks.items()
    }
    passed = all(row["passed"] for row in check_rows.values()) and all(
        row["passed"] for row in all_rows
    )
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "claim_ids": list(CLAIM_IDS),
        "claim_classes": CLAIM_CLASSES,
        "paper": {
            "arxiv_id": "2607.21366",
            "version": "v1",
            "pdf_sha256": PAPER_SHA256,
            "anchors": [
                "Eq. 6 and Eq. 8, PDF page 13",
                "Eqs. 9-11, PDF page 14",
                "Eqs. 12-15, PDF page 15",
                "Eqs. 16-18, PDF page 16",
                "Eq. 86, PDF page 57",
                "Proposition H.4 and Eqs. 116-118, PDF pages 66-67",
                "Algorithm 2, PDF page 69",
            ],
        },
        "config": config,
        "config_sha256": _config_hash(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": REPRODUCTION_SCOPE,
        "check_count": len(check_rows),
        "passing_check_count": sum(row["passed"] for row in check_rows.values()),
        "checks": check_rows,
        "priced_transition": "[f_i,f_j] -> [f_p,f_p]",
        "executed_transition": "f_i+f_j -> f_p",
        "exact_twin": {
            "paper_scale": "q",
            "paper_coordinate_distance": 0.0,
            "executed_distance": "q",
            "normalized_repair_scale": "2q",
            "unnormalized_repair_scale": "2q",
        },
        "paper_scale_limit": "s* -> b over 2 as E_rem -> infinity",
        "normalized_scale_limit": "s_normalized -> b as E_rem -> infinity",
        "row_counts": {
            "source_map": len(source_rows),
            "twin_cases": len(twin_rows),
            "correlation_sweep": len(correlation_rows),
            "capacity_sweep": len(capacity_rows),
            "repair_comparison": len(repair_rows),
            "physical_forward": len(forward_rows),
        },
    }
    if not passed:
        failed_checks = [name for name, row in check_rows.items() if not row["passed"]]
        failed_rows = [row["case_id"] for row in all_rows if not row["passed"]]
        raise AuditFailure(
            f"audit checks failed: checks={failed_checks}, rows={failed_rows}"
        )
    return {
        "summary": summary,
        "source_map.csv": source_rows,
        "twin_cases.csv": twin_rows,
        "correlation_sweep.csv": correlation_rows,
        "capacity_sweep.csv": capacity_rows,
        "repair_comparison.csv": repair_rows,
        "physical_forward.csv": forward_rows,
    }


def validate_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise AuditFailure("summary schema changed")
    if summary.get("audit_id") != AUDIT_ID:
        raise AuditFailure("summary audit identity changed")
    if summary.get("status") != "passed" or summary.get("passed") is not True:
        raise AuditFailure("summary is not passing")
    if tuple(summary.get("claim_ids", ())) != CLAIM_IDS:
        raise AuditFailure("claim registry changed")
    if summary.get("claim_classes") != CLAIM_CLASSES:
        raise AuditFailure("claim evidence-class registry changed")
    checks = summary.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(CHECK_SPECS):
        raise AuditFailure("summary check registry changed")
    if summary.get("check_count") != len(CHECK_SPECS):
        raise AuditFailure("summary check count changed")
    if summary.get("passing_check_count") != len(CHECK_SPECS):
        raise AuditFailure("summary passing check count changed")
    for name, spec in CHECK_SPECS.items():
        record = checks[name]
        if not isinstance(record, Mapping) or record.get("passed") is not True:
            raise AuditFailure(f"check is not passing: {name}")
        value = _finite(record.get("value"), f"check {name}")
        if not spec["minimum"] <= value <= spec["maximum"]:
            raise AuditFailure(f"check is outside source range: {name}")
    if any(not math.isfinite(value) for value in _walk_numbers(summary)):
        raise AuditFailure("summary contains a non-finite number")


def _write_json(path: Path, payload: Any) -> None:
    if any(not math.isfinite(value) for value in _walk_numbers(payload)):
        raise AuditFailure(f"refusing to write non-finite JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise AuditFailure(f"CSV schema mismatch for {path.name}")
            writer.writerow(row)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paper_identity(paper_pdf: Path | None, full: bool) -> dict[str, Any]:
    manifest = json.loads(PAPER_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("pdf_sha256") != PAPER_SHA256:
        raise AuditFailure("paper manifest identity changed")
    identity = {
        "arxiv_id": "2607.21366",
        "version": "v1",
        "pdf_sha256": PAPER_SHA256,
        "pdf_bytes": PAPER_BYTES,
        "pdf_supplied": paper_pdf is not None,
    }
    if full and paper_pdf is None:
        raise AuditFailure("--full requires --paper-pdf")
    if paper_pdf is not None:
        if (
            not paper_pdf.is_file()
            or paper_pdf.is_symlink()
            or paper_pdf.stat().st_size != PAPER_BYTES
            or _sha256_path(paper_pdf) != PAPER_SHA256
        ):
            raise AuditFailure("supplied paper PDF has the wrong identity")
    return identity


def _git_identity() -> tuple[str, str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *SOURCE_RELATIVE_PATHS],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", "unavailable", False
    return commit, "exact_commit" if not status else "working_tree", not bool(status)


def _manifest(
    result_dir: Path,
    outputs: Sequence[Path],
    summary: Mapping[str, Any],
    paper: Mapping[str, Any],
    full: bool,
) -> dict[str, Any]:
    commit, state, matches = _git_identity()
    source_hashes = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise AuditFailure(f"required source is missing: {relative}")
        source_hashes.append({
            "path": relative,
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
        })
    output_rows = []
    for path in outputs:
        relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        output_rows.append({
            "path": relative,
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
            "media_type": "application/json" if path.suffix == ".json" else "text/csv",
        })
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "generation_time_policy": "deterministic outputs omit wall-clock timestamps",
        "paper": dict(paper),
        "config": summary["config"],
        "config_sha256": summary["config_sha256"],
        "seeds": {"used": False, "value": None},
        "code": {
            "source_snapshot_commit": commit,
            "source_snapshot_state": state,
            "source_files_match_commit": matches,
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "cpu_count": os.cpu_count(),
            "float64_epsilon": float(np.finfo(np.float64).eps),
        },
        "expected_metric_ranges": CHECK_SPECS,
        "observed_checks": summary["checks"],
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": REPRODUCTION_SCOPE,
        "source_hashes": source_hashes,
        "input_hashes": [{
            "kind": "paper_manifest",
            "path": "paper/manifest.json",
            "sha256": _sha256_path(PAPER_MANIFEST_PATH),
            "bytes": PAPER_MANIFEST_PATH.stat().st_size,
        }],
        "outputs": output_rows,
        "manifest_hash_scope": "manifest excludes its own hash to avoid self-reference",
        "canonical_invocation": {
            "entrypoint": "experiments/audits/part07_merge_semantics/run.py",
            "result_dir": "<RESULT_DIR>",
            "manifest_path": "<MANIFEST_PATH>",
            "paper_pdf": "<PAPER_PDF>" if paper["pdf_supplied"] else "not supplied",
            "full": full,
        },
    }


def run_audit(
    *,
    result_dir: Path,
    manifest_path: Path | None = None,
    paper_pdf: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    paper = _paper_identity(paper_pdf, full)
    audit = build_audit(full=full)
    summary = audit["summary"]
    validate_summary(summary)
    if result_dir.exists() and any(result_dir.iterdir()):
        raise AuditFailure(f"refusing to overwrite nonempty result directory: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result_dir / "summary.json", summary)
    for filename, fields in CSV_CONTRACTS.items():
        _write_csv(result_dir / filename, fields, audit[filename])
    outputs = [result_dir / filename for filename in OUTPUT_FILENAMES]
    if manifest_path is not None:
        if manifest_path.exists():
            raise AuditFailure(f"refusing to overwrite manifest: {manifest_path}")
        _write_json(manifest_path, _manifest(result_dir, outputs, summary, paper, full))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--paper-pdf", type=Path)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_audit(
        result_dir=args.result_dir.resolve(),
        manifest_path=args.manifest_path.resolve() if args.manifest_path else None,
        paper_pdf=args.paper_pdf.resolve() if args.paper_pdf else None,
        full=args.full,
    )
    print(
        f"PASS: {summary['passing_check_count']}/{summary['check_count']} Part 7 checks "
        f"passed; seven deterministic outputs written to {args.result_dir}"
    )


if __name__ == "__main__":
    main()
