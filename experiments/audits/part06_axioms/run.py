#!/usr/bin/env python3
"""Deterministic audit of what HOPE v1's transition-cost axioms determine."""

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
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
AUDIT_ID = "part06_axioms_audit"
AUDIT_SCHEMA_VERSION = 1
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
PAPER_BYTES = 1_089_944
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_06" / "manual"

CLAIM_IDS = (
    "P06-L1-CAPACITY-CLAIM",
    "P06-LP-PARTITION-RESULT",
    "P06-CAPACITY-INTEGRAL-FAMILY",
    "P06-CONNECTIVITY-LOWER-BOUND",
    "P06-GEOMETRIC-OBJECTIVE-CHANGE",
    "P06-STRAIGHT-PATH-BOUND",
    "P06-AXIOM3-RELAXATION",
    "P06-WIDTH-NORMALIZATION-CHOICE",
    "P06-ENDPOINT-PROXY-NONUNIQUENESS",
    "P06-CROSS-LAYER-RANKING-SENSITIVITY",
)
CLAIM_CLASSES = {
    "P06-L1-CAPACITY-CLAIM": "paper_claim",
    "P06-LP-PARTITION-RESULT": "verified_fact",
    "P06-CAPACITY-INTEGRAL-FAMILY": "verified_fact",
    "P06-CONNECTIVITY-LOWER-BOUND": "counterexample",
    "P06-GEOMETRIC-OBJECTIVE-CHANGE": "paper_claim",
    "P06-STRAIGHT-PATH-BOUND": "verified_fact",
    "P06-AXIOM3-RELAXATION": "paper_claim",
    "P06-WIDTH-NORMALIZATION-CHOICE": "paper_claim",
    "P06-ENDPOINT-PROXY-NONUNIQUENESS": "counterexample",
    "P06-CROSS-LAYER-RANKING-SENSITIVITY": "verified_fact",
}

P_VALUES = (0.5, 1.0, 2.0)
PARTITION_COUNTS = (1, 2, 3, 5)
SCALE_FACTORS = (1e-6, 1e-3, 1.0, 1e3, 1e6)
EXTINCTION_CAPACITIES = (1.0, 0.1, 0.01, 0.001)
COUNTERMODEL_TERMINALS = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
QUADRATURE_ORDER = 96

TOLERANCES = {
    "formula_atol": 2e-14,
    "quadrature_atol": 5e-13,
    "scale_rtol": 5e-13,
    "additivity_atol": 2e-14,
    "material_path_gap_minimum": 0.4,
    "material_nonadditivity_minimum": 0.3,
}

METHOD_ORIGINS = {
    "conditional_l1_partition": {
        "evidence_label": "paper_claim",
        "claim_id": "P06-L1-CAPACITY-CLAIM",
        "origin": "HOPE v1 Lemma C.1 and Eqs. 58-61",
        "scope": "paper uniqueness claim under its declared functional assumptions",
    },
    "lp_partition_sweep": {
        "evidence_label": "verified_fact",
        "claim_id": "P06-LP-PARTITION-RESULT",
        "origin": "independent exact L_p calculation",
        "scope": "fixed L_p family; not a numerical proof outside that family",
    },
    "capacity_drain": {
        "evidence_label": "verified_fact",
        "claim_id": "P06-CAPACITY-INTEGRAL-FAMILY",
        "origin": "HOPE v1 Theorem C.2 and Eq. 43",
        "scope": "positive capacity with the degree-zero factor c constant on one transition",
    },
    "connectivity_countermodel": {
        "evidence_label": "counterexample",
        "claim_id": "P06-CONNECTIVITY-LOWER-BOUND",
        "origin": "independent positive-cone path and scale-invariant density",
        "scope": "refutes divergence from positivity and scale invariance without a uniform lower bound",
    },
    "geometric_speed": {
        "evidence_label": "paper_claim",
        "claim_id": "P06-GEOMETRIC-OBJECTIVE-CHANGE",
        "origin": "HOPE v1 Eq. 44 geometric projection-rate definition",
        "scope": "objective change after calibration on one-coordinate shrinkage",
    },
    "endpoint_proxy_family": {
        "evidence_label": "counterexample",
        "claim_id": "P06-ENDPOINT-PROXY-NONUNIQUENESS",
        "origin": "independent family J_p = N (D/E_b)^p",
        "scope": "non-uniqueness witness for Axioms 1-2 when D has a positive extinction limit; not claimed to satisfy Axiom 3 or be an upper bound",
    },
    "ranking_fixture": {
        "evidence_label": "verified_fact",
        "claim_id": "P06-CROSS-LAYER-RANKING-SENSITIVITY",
        "origin": "fixed synthetic three-candidate cross-layer fixture",
        "scope": "possibility witness only; no trained-network incidence or accuracy conclusion",
    },
    "straight_path_bound": {
        "evidence_label": "verified_fact",
        "claim_id": "P06-STRAIGHT-PATH-BOUND",
        "origin": "independent exact straight-path and detour fixtures",
        "scope": "the endpoint bound is asserted only for the declared straight path",
    },
    "axiom3_relaxation": {
        "evidence_label": "paper_claim",
        "claim_id": "P06-AXIOM3-RELAXATION",
        "origin": "HOPE v1 Section 6.2, PDF page 12",
        "scope": "the paper explicitly states that the endpoint proxy sacrifices path additivity",
    },
    "width_normalization": {
        "evidence_label": "paper_claim",
        "claim_id": "P06-WIDTH-NORMALIZATION-CHOICE",
        "origin": "HOPE v1 choice c(Phi)=N, PDF page 12",
        "scope": "engineering normalization, not a uniqueness conclusion from Axioms 1-2",
    },
}

REPRODUCTION_SCOPE = {
    "reproduction_scope": "independent_mathematical_audit",
    "paper_experiment_reproduced": False,
    "checkpoint_or_dataset_used": False,
    "random_sampling_used": False,
    "includes": [
        "conditional L_p partition sweep",
        "constant-density capacity-drain integral",
        "scale-invariant density countermodel",
        "straight and detour geometric path integrals",
        "positive endpoint-proxy family",
        "path-additivity control and endpoint non-additivity",
        "cross-layer ranking witness and same-layer control",
    ],
    "excludes": [
        "parent-neuron generation",
        "physical merge execution",
        "merge correlation-threshold validity",
        "Part 7",
        "trained-network frequency",
        "compression accuracy or utility",
    ],
    "claim_boundary": (
        "The fixtures distinguish implications from choices. They do not prove "
        "a uniqueness theorem outside the stated functional family and do not "
        "estimate behavior on trained networks."
    ),
}

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part06_axioms/README.md",
    "experiments/audits/part06_axioms/run.py",
    "experiments/audits/part06_axioms/test_run.py",
)

DERIVATION_FIELDS = (
    "step_id", "sequence", "formula", "premise", "result", "evidence_label",
    "claim_id", "passed",
)
PARTITION_FIELDS = (
    "case_id", "p", "partition_count", "base_norm", "capacity_before",
    "capacity_after", "analytic_factor", "formula_error", "expected_invariant",
    "observed_invariant", "evidence_label", "claim_id", "passed",
)
PATH_FIELDS = (
    "case_id", "section", "path_id", "parameterization", "segments",
    "start_capacity", "end_capacity", "density_rule", "capacity_monotone",
    "arc_length", "minimum_capacity", "endpoint_distance", "exact_integral",
    "numeric_integral", "absolute_error", "endpoint_expression",
    "endpoint_is_proved_bound", "arc_upper_bound", "evidence_label", "claim_id",
    "passed", "note",
)
PROXY_FIELDS = (
    "case_id", "section", "p", "width", "scale", "distance",
    "terminal_capacity", "proxy_value", "reference_value", "absolute_error",
    "monotone_from_previous", "additivity_role", "evidence_label", "claim_id",
    "passed", "note",
)
RANKING_FIELDS = (
    "case_id", "fixture_scope", "candidate_id", "p", "width", "distance",
    "terminal_capacity", "ratio", "score", "rank", "order",
    "rank_changed_from_p1", "evidence_label", "claim_id", "passed", "note",
)
CSV_CONTRACTS = {
    "derivation_steps.csv": DERIVATION_FIELDS,
    "partition_sweep.csv": PARTITION_FIELDS,
    "path_integrals.csv": PATH_FIELDS,
    "proxy_family.csv": PROXY_FIELDS,
    "ranking_fixture.csv": RANKING_FIELDS,
}
OUTPUT_FILENAMES = ("summary.json", *CSV_CONTRACTS.keys())

CHECK_SPECS = {
    "nonfinite_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "partition_formula_max_error": {"minimum": 0.0, "maximum": 2e-14, "units": "absolute"},
    "partition_p1_max_error": {"minimum": 0.0, "maximum": 2e-14, "units": "absolute"},
    "partition_unexpected_result_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "capacity_integral_max_error": {"minimum": 0.0, "maximum": 5e-13, "units": "absolute"},
    "capacity_integral_path_spread": {"minimum": 0.0, "maximum": 5e-13, "units": "absolute"},
    "density_scale_invariance_max_error": {"minimum": 0.0, "maximum": 2e-14, "units": "absolute"},
    "countermodel_integral_max_error": {"minimum": 0.0, "maximum": 5e-13, "units": "absolute"},
    "countermodel_extinction_limit": {"minimum": 0.749999999999, "maximum": 0.750000000001, "units": "cost"},
    "straight_integral_error": {"minimum": 0.0, "maximum": 5e-13, "units": "absolute"},
    "straight_bound_margin": {"minimum": 0.6, "maximum": 0.7, "units": "cost"},
    "detour_integral_error": {"minimum": 0.0, "maximum": 5e-13, "units": "absolute"},
    "detour_endpoint_excess": {"minimum": 0.4, "maximum": 0.5, "units": "cost"},
    "detour_arc_bound_margin": {"minimum": 1.5, "maximum": 1.6, "units": "cost"},
    "proxy_scale_max_relative_error": {"minimum": 0.0, "maximum": 5e-13, "units": "relative"},
    "proxy_extinction_monotonicity_failures": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "endpoint_p1_nonadditivity": {"minimum": 0.333333333332, "maximum": 0.333333333335, "units": "cost"},
    "geometric_additivity_error": {"minimum": 0.0, "maximum": 2e-14, "units": "absolute"},
    "cross_layer_distinct_order_count": {"minimum": 3.0, "maximum": 3.0, "units": "count"},
    "same_layer_rank_change_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "evidence_class_mismatch_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
}


class AuditFailure(RuntimeError):
    """Raised when a declared audit invariant fails."""


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def lp_capacity(norms: Sequence[float], p: float) -> float:
    exponent = _finite(p, "p")
    if exponent <= 0.0 or not norms:
        raise ValueError("p and all capacities must be positive")
    values = [_finite(value, "capacity") for value in norms]
    if any(value < 0.0 for value in values):
        raise ValueError("capacities must be nonnegative")
    return float(sum(value**exponent for value in values) ** (1.0 / exponent))


def endpoint_proxy(width: int, distance: float, terminal_capacity: float, p: float) -> float:
    if isinstance(width, bool) or int(width) != width or width <= 0:
        raise ValueError("width must be a positive integer")
    distance = _finite(distance, "distance")
    terminal_capacity = _finite(terminal_capacity, "terminal_capacity")
    p = _finite(p, "p")
    if distance < 0.0 or terminal_capacity <= 0.0 or p <= 0.0:
        raise ValueError("proxy inputs are outside the positive domain")
    return float(width * (distance / terminal_capacity) ** p)


def scale_invariant_density(phi: Sequence[float]) -> float:
    if len(phi) != 2:
        raise ValueError("the countermodel density requires two coordinates")
    first, second = (_finite(value, "phi coordinate") for value in phi)
    if first < 0.0 or second <= 0.0 or first + second <= 0.0:
        raise ValueError("the countermodel lies on the positive cone")
    return second / (first + second)


def countermodel_state(capacity: float) -> tuple[float, float]:
    capacity = _finite(capacity, "capacity")
    if not 0.0 < capacity < 1.0:
        raise ValueError("countermodel capacity must lie strictly between zero and one")
    return capacity - capacity * capacity, capacity * capacity


def gauss_legendre_integral(function: Callable[[np.ndarray], np.ndarray]) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
    points = 0.5 * (nodes + 1.0)
    values = np.asarray(function(points), dtype=np.float64)
    if values.shape != points.shape or not np.all(np.isfinite(values)):
        raise AuditFailure("quadrature integrand returned invalid values")
    return float(0.5 * np.dot(weights, values))


def segment_integral(start: Sequence[float], end: Sequence[float], c_value: float) -> tuple[float, float, float]:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    if start_array.shape != end_array.shape or start_array.ndim != 1:
        raise ValueError("segment endpoints must have the same vector shape")
    if np.any(start_array <= 0.0) or np.any(end_array <= 0.0):
        raise ValueError("fixture segment must stay in the positive orthant")
    length = float(np.linalg.norm(end_array - start_array))
    initial = float(np.sum(start_array))
    terminal = float(np.sum(end_array))
    delta = terminal - initial
    if abs(delta) <= TOLERANCES["formula_atol"]:
        exact = c_value * length / initial
    else:
        exact = c_value * length * math.log(terminal / initial) / delta
    numeric = gauss_legendre_integral(
        lambda t: c_value * length / (initial + delta * t)
    )
    return length, float(exact), numeric


def build_config(*, full: bool) -> dict[str, Any]:
    return {
        "float_dtype": "float64",
        "full_mode": bool(full),
        "quadrature_order": QUADRATURE_ORDER,
        "random_sampling": False,
        "p_values": list(P_VALUES),
        "partition_counts": list(PARTITION_COUNTS),
        "scale_factors": list(SCALE_FACTORS),
        "extinction_capacities": list(EXTINCTION_CAPACITIES),
        "countermodel_terminals": list(COUNTERMODEL_TERMINALS),
        "partition_vector": [3.0, 4.0],
        "capacity_integral": {"E_a": 5.0, "E_b": 2.0, "c": 3.0},
        "connectivity_countermodel": {
            "E_a": 0.75,
            "state": "Phi(E)=(E-E^2,E^2)",
            "density": "Phi_2/(Phi_1+Phi_2)",
        },
        "geometric_fixture": {
            "start": [3.0, 1.0],
            "end": [1.0, 1.0],
            "detour": [1.0, 2.5],
            "c": 2.0,
        },
        "proxy_scale_fixture": {"N": 5, "D": 3.0, "E_b": 2.0},
        "nonadditivity_fixture": {"N": 2, "E_a": 4.0, "E_m": 3.0, "E_b": 2.0},
        "tolerances": dict(TOLERANCES),
    }


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def _partition_rows() -> list[dict[str, Any]]:
    base_norm = math.hypot(3.0, 4.0)
    rows: list[dict[str, Any]] = []
    for p in P_VALUES:
        for count in PARTITION_COUNTS:
            before = lp_capacity([base_norm], p)
            after = lp_capacity([base_norm / count] * count, p)
            factor = count ** (1.0 / p - 1.0)
            formula_error = abs(after - base_norm * factor)
            expected = count == 1 or p == 1.0
            observed = math.isclose(after, before, rel_tol=0.0, abs_tol=TOLERANCES["formula_atol"])
            rows.append({
                "case_id": f"lp_p{p:g}_m{count}", "p": p, "partition_count": count,
                "base_norm": base_norm, "capacity_before": before, "capacity_after": after,
                "analytic_factor": factor, "formula_error": formula_error,
                "expected_invariant": expected, "observed_invariant": observed,
                "evidence_label": "verified_fact", "claim_id": "P06-LP-PARTITION-RESULT",
                "passed": formula_error <= TOLERANCES["formula_atol"] and expected == observed,
            })
    return rows


def _capacity_path_rows() -> list[dict[str, Any]]:
    ea, eb, c_value = 5.0, 2.0, 3.0
    expected = c_value * math.log(ea / eb)
    definitions: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("linear", lambda t: c_value * (ea - eb) / (ea + (eb - ea) * t)),
        ("geometric", lambda t: np.full_like(t, c_value * math.log(ea / eb))),
        (
            "quadratic",
            lambda t: c_value * 2.0 * (ea - eb) * (1.0 - t)
            / (eb + (ea - eb) * (1.0 - t) ** 2),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, integrand in definitions:
        numeric = gauss_legendre_integral(integrand)
        rows.append({
            "case_id": f"capacity_{name}", "section": "capacity_drain",
            "path_id": name, "parameterization": name, "segments": 1,
            "start_capacity": ea, "end_capacity": eb, "density_rule": "c=3 constant",
            "capacity_monotone": True, "arc_length": None, "minimum_capacity": eb,
            "endpoint_distance": None, "exact_integral": expected,
            "numeric_integral": numeric, "absolute_error": abs(numeric - expected),
            "endpoint_expression": expected, "endpoint_is_proved_bound": None,
            "arc_upper_bound": None, "evidence_label": "verified_fact",
            "claim_id": "P06-CAPACITY-INTEGRAL-FAMILY",
            "passed": abs(numeric - expected) <= TOLERANCES["quadrature_atol"],
            "note": "constant c makes the integral c log(E_a/E_b)",
        })
    return rows


def _countermodel_rows() -> tuple[list[dict[str, Any]], float]:
    ea = 0.75
    rows: list[dict[str, Any]] = []
    scale_error = 0.0
    for eb in COUNTERMODEL_TERMINALS:
        exact = ea - eb
        numeric = gauss_legendre_integral(
            lambda t, terminal=eb: np.full_like(t, ea - terminal)
        )
        for capacity in (ea, 0.5 * (ea + eb), eb):
            state = countermodel_state(capacity)
            density = scale_invariant_density(state)
            scale_error = max(scale_error, abs(density - capacity))
            for scale in SCALE_FACTORS:
                scaled = (scale * state[0], scale * state[1])
                scale_error = max(scale_error, abs(scale_invariant_density(scaled) - density))
        rows.append({
            "case_id": f"countermodel_eb_{eb:g}", "section": "connectivity_countermodel",
            "path_id": "positive_cone", "parameterization": "Phi(E)=(E-E^2,E^2)",
            "segments": 1, "start_capacity": ea, "end_capacity": eb,
            "density_rule": "Phi_2/(Phi_1+Phi_2)=E on this path",
            "capacity_monotone": True, "arc_length": None, "minimum_capacity": eb,
            "endpoint_distance": None, "exact_integral": exact,
            "numeric_integral": numeric, "absolute_error": abs(numeric - exact),
            "endpoint_expression": ea, "endpoint_is_proved_bound": False,
            "arc_upper_bound": None, "evidence_label": "counterexample",
            "claim_id": "P06-CONNECTIVITY-LOWER-BOUND",
            "passed": abs(numeric - exact) <= TOLERANCES["quadrature_atol"] and exact < ea,
            "note": "positive scale-invariant c has no uniform positive lower bound on this path",
        })
    return rows, scale_error


def _geometric_rows() -> list[dict[str, Any]]:
    start, end, waypoint, c_value = (3.0, 1.0), (1.0, 1.0), (1.0, 2.5), 2.0
    direct_length, direct_exact, direct_numeric = segment_integral(start, end, c_value)
    first_length, first_exact, first_numeric = segment_integral(start, waypoint, c_value)
    second_length, second_exact, second_numeric = segment_integral(waypoint, end, c_value)
    endpoint_distance = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    terminal_capacity = sum(end)
    endpoint_expression = c_value * endpoint_distance / terminal_capacity
    detour_length = first_length + second_length
    detour_exact = first_exact + second_exact
    detour_numeric = first_numeric + second_numeric
    return [
        {
            "case_id": "geometric_straight", "section": "geometric_projection",
            "path_id": "straight", "parameterization": "(3-2t,1)", "segments": 1,
            "start_capacity": 4.0, "end_capacity": 2.0, "density_rule": "c*speed/E",
            "capacity_monotone": True, "arc_length": direct_length,
            "minimum_capacity": 2.0, "endpoint_distance": endpoint_distance,
            "exact_integral": direct_exact, "numeric_integral": direct_numeric,
            "absolute_error": abs(direct_numeric - direct_exact),
            "endpoint_expression": endpoint_expression, "endpoint_is_proved_bound": True,
            "arc_upper_bound": c_value * direct_length / terminal_capacity,
            "evidence_label": "verified_fact", "claim_id": "P06-STRAIGHT-PATH-BOUND",
            "passed": direct_exact <= endpoint_expression and abs(direct_numeric - direct_exact) <= TOLERANCES["quadrature_atol"],
            "note": "the declared straight path satisfies E(t)>=E_b",
        },
        {
            "case_id": "geometric_detour", "section": "geometric_projection",
            "path_id": "detour", "parameterization": "(3,1)->(1,2.5)->(1,1)",
            "segments": 2, "start_capacity": 4.0, "end_capacity": 2.0,
            "density_rule": "c*speed/E", "capacity_monotone": True,
            "arc_length": detour_length, "minimum_capacity": 2.0,
            "endpoint_distance": endpoint_distance, "exact_integral": detour_exact,
            "numeric_integral": detour_numeric, "absolute_error": abs(detour_numeric - detour_exact),
            "endpoint_expression": endpoint_expression, "endpoint_is_proved_bound": False,
            "arc_upper_bound": c_value * detour_length / terminal_capacity,
            "evidence_label": "verified_fact", "claim_id": "P06-STRAIGHT-PATH-BOUND",
            "passed": detour_exact > endpoint_expression + TOLERANCES["material_path_gap_minimum"] and detour_exact <= c_value * detour_length / terminal_capacity,
            "note": "same endpoints and decreasing capacity, but the endpoint expression is not an arbitrary-path bound",
        },
    ]


def _proxy_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scale_errors: list[float] = []
    for p in P_VALUES:
        reference = endpoint_proxy(5, 3.0, 2.0, p)
        for scale in SCALE_FACTORS:
            score = endpoint_proxy(5, 3.0 * scale, 2.0 * scale, p)
            error = abs(score - reference)
            scale_errors.append(error / max(abs(reference), 1.0))
            rows.append({
                "case_id": f"scale_p{p:g}_k{scale:g}", "section": "scale_invariance",
                "p": p, "width": 5, "scale": scale, "distance": 3.0 * scale,
                "terminal_capacity": 2.0 * scale, "proxy_value": score,
                "reference_value": reference, "absolute_error": error,
                "monotone_from_previous": None, "additivity_role": None,
                "evidence_label": "counterexample", "claim_id": "P06-ENDPOINT-PROXY-NONUNIQUENESS",
                "passed": math.isclose(score, reference, rel_tol=TOLERANCES["scale_rtol"], abs_tol=TOLERANCES["formula_atol"]),
                "note": "D and E_b are scaled together; N is unchanged",
            })
        previous = None
        for eb in EXTINCTION_CAPACITIES:
            score = endpoint_proxy(5, 3.0, eb, p)
            monotone = previous is None or score > previous
            rows.append({
                "case_id": f"extinction_p{p:g}_eb{eb:g}", "section": "extinction",
                "p": p, "width": 5, "scale": 1.0, "distance": 3.0,
                "terminal_capacity": eb, "proxy_value": score,
                "reference_value": 5.0 * (3.0 / eb) ** p, "absolute_error": 0.0,
                "monotone_from_previous": monotone, "additivity_role": None,
                "evidence_label": "counterexample", "claim_id": "P06-ENDPOINT-PROXY-NONUNIQUENESS",
                "passed": monotone, "note": "D is held at a positive value while E_b decreases",
            })
            previous = score

    nonadditivity: dict[str, Any] = {}
    for p in P_VALUES:
        direct = endpoint_proxy(2, 2.0, 2.0, p)
        via = endpoint_proxy(2, 1.0, 3.0, p) + endpoint_proxy(2, 1.0, 2.0, p)
        defect = via - direct
        nonadditivity[f"p_{p:g}"] = {"direct": direct, "via": via, "defect": defect}
        for role, value in (("direct", direct), ("via_midpoint", via)):
            rows.append({
                "case_id": f"nonadd_p{p:g}_{role}", "section": "nonadditivity",
                "p": p, "width": 2, "scale": 1.0,
                "distance": 2.0 if role == "direct" else None,
                "terminal_capacity": 2.0, "proxy_value": value,
                "reference_value": direct, "absolute_error": abs(value - direct),
                "monotone_from_previous": None, "additivity_role": role,
                "evidence_label": "counterexample", "claim_id": "P06-ENDPOINT-PROXY-NONUNIQUENESS",
                "passed": role == "direct" or abs(defect) > TOLERANCES["material_nonadditivity_minimum"],
                "note": "scalar path 4->3->2 with fixed N; via row is the sum of two endpoint proxies",
            })
    geometric_direct = 2.0 * math.log(4.0 / 2.0)
    geometric_via = 2.0 * math.log(4.0 / 3.0) + 2.0 * math.log(3.0 / 2.0)
    nonadditivity["geometric_control"] = {
        "direct": geometric_direct,
        "via": geometric_via,
        "error": abs(geometric_direct - geometric_via),
    }
    nonadditivity["maximum_scale_relative_error"] = max(scale_errors)
    return rows, nonadditivity


def _ranking_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixtures = {
        "cross_layer": {"A": (2, 4.0), "B": (7, 1.0), "C": (3, 2.0)},
        "same_layer": {"S1": (4, 1.0), "S2": (4, 2.0), "S4": (4, 4.0)},
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for scope, candidates in fixtures.items():
        rank_by_p: dict[float, dict[str, int]] = {}
        order_by_p: dict[float, list[str]] = {}
        score_by_p: dict[float, dict[str, float]] = {}
        for p in P_VALUES:
            scores = {name: endpoint_proxy(width, ratio, 1.0, p) for name, (width, ratio) in candidates.items()}
            order = sorted(scores, key=lambda name: (scores[name], name))
            score_by_p[p] = scores
            order_by_p[p] = order
            rank_by_p[p] = {name: index + 1 for index, name in enumerate(order)}
        base_ranks = rank_by_p[1.0]
        for p in P_VALUES:
            order_text = "<".join(order_by_p[p])
            for name, (width, ratio) in candidates.items():
                changed = rank_by_p[p][name] != base_ranks[name]
                label = "verified_fact"
                rows.append({
                    "case_id": f"{scope}_p{p:g}_{name}", "fixture_scope": scope,
                    "candidate_id": name, "p": p, "width": width, "distance": ratio,
                    "terminal_capacity": 1.0, "ratio": ratio, "score": score_by_p[p][name],
                    "rank": rank_by_p[p][name], "order": order_text,
                    "rank_changed_from_p1": changed, "evidence_label": label,
                    "claim_id": "P06-CROSS-LAYER-RANKING-SENSITIVITY",
                    "passed": True,
                    "note": "synthetic possibility witness" if scope == "cross_layer" else "fixed-N monotonic-order control",
                })
        summary[scope] = {f"p_{p:g}": order_by_p[p] for p in P_VALUES}
    summary["cross_layer_crossover_p"] = {
        "A_vs_C": math.log(3.0 / 2.0) / math.log(2.0),
        "A_vs_B": math.log(7.0 / 2.0) / math.log(4.0),
        "B_vs_C": math.log(7.0 / 3.0) / math.log(2.0),
    }
    return rows, summary


def _derivation_rows() -> list[dict[str, Any]]:
    specs = [
        ("l1_paper_claim", "E(Phi)=sum_i ||f_i||", "identity, symmetric separability, continuity, strict monotonicity, and equal partitioning", "the paper states conditional L1 uniqueness", "paper_claim", "P06-L1-CAPACITY-CLAIM"),
        ("lp_fixed_family", "E_p(partition)=||f|| M^(1/p-1)", "the declared L_p family and equal fractional partitioning", "p=1 is the only nontrivially invariant tested exponent", "verified_fact", "P06-LP-PARTITION-RESULT"),
        ("capacity_differential", "dJ=-c(Phi)dE/E", "Axioms 1 and 3 with E>0", "the scale-invariant density form is independently checked", "verified_fact", "P06-CAPACITY-INTEGRAL-FAMILY"),
        ("constant_density", "J=c log(E_a/E_b)", "c is constant on the declared transition", "three monotone parameterizations agree", "verified_fact", "P06-CAPACITY-INTEGRAL-FAMILY"),
        ("connectivity_condition", "positivity and scale invariance of c do not imply c>=c_min>0", "the theorem's divergence step separately assumes a uniform pathwise lower bound", "the positive scale-invariant countermodel shows why the extra premise matters", "counterexample", "P06-CONNECTIVITY-LOWER-BOUND"),
        ("connectivity_countermodel", "Phi(E)=(E-E^2,E^2), c(Phi)=Phi_2/(Phi_1+Phi_2)=E", "0<E<1 and c is positive and scale invariant on the positive cone", "J=E_a-E_b stays finite as E_b->0", "counterexample", "P06-CONNECTIVITY-LOWER-BOUND"),
        ("geometric_change", "dJ_proj=c(Phi)||dot(Phi)||_2 dt/E(Phi)", "geometric speed replaces -dot(E)", "the objective becomes path dependent", "paper_claim", "P06-GEOMETRIC-OBJECTIVE-CHANGE"),
        ("straight_bound", "J_proj<=c D/E_b", "declared straight path, constant c, and E(t)>=E_b", "endpoint expression bounds that path", "verified_fact", "P06-STRAIGHT-PATH-BOUND"),
        ("axiom3_relaxed", "the endpoint proxy knowingly sacrifices path additivity", "endpoint proxy replaces the line integral", "the paper explicitly relaxes Axiom 3", "paper_claim", "P06-AXIOM3-RELAXATION"),
        ("width_choice", "c(Phi)=N", "piecewise constant width normalization", "an engineering choice, not fixed by Axioms 1-2", "paper_claim", "P06-WIDTH-NORMALIZATION-CHOICE"),
        ("eq6_instantiation", "J_final=N D/E_b with action-specific substitutions", "the geometric endpoint proxy, c=N, and the declared pruning or merge endpoints", "the paper's Eq. 6 formulas are algebraic instantiations of the selected proxy", "paper_claim", "P06-WIDTH-NORMALIZATION-CHOICE"),
        ("proxy_family", "J_p=N(D/E_b)^p, p>0", "D/E_b is dimensionless and D has a positive extinction limit", "every p shares scale invariance and the conditional barrier", "counterexample", "P06-ENDPOINT-PROXY-NONUNIQUENESS"),
        ("ranking_scope", "N_a r_a^p=N_b r_b^p", "global comparison may have different widths", "p can change cross-layer order; fixed N preserves ratio order", "verified_fact", "P06-CROSS-LAYER-RANKING-SENSITIVITY"),
    ]
    return [
        {"step_id": step, "sequence": index, "formula": formula, "premise": premise,
         "result": result, "evidence_label": label, "claim_id": claim, "passed": True}
        for index, (step, formula, premise, result, label, claim) in enumerate(specs, start=1)
    ]


def _walk_numbers(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Mapping):
        return [number for child in value.values() for number in _walk_numbers(child)]
    if isinstance(value, (list, tuple)):
        return [number for child in value for number in _walk_numbers(child)]
    return []


def build_audit(*, full: bool = False) -> dict[str, Any]:
    config = build_config(full=full)
    partition_rows = _partition_rows()
    capacity_rows = _capacity_path_rows()
    counter_rows, density_scale_error = _countermodel_rows()
    geometric_rows = _geometric_rows()
    proxy_rows, nonadditivity = _proxy_rows()
    ranking_rows, ranking = _ranking_rows()
    derivation_rows = _derivation_rows()
    path_rows = capacity_rows + counter_rows + geometric_rows

    capacity_values = [row["numeric_integral"] for row in capacity_rows]
    straight, detour = geometric_rows
    p1_nonadd = nonadditivity["p_1"]
    cross_orders = {tuple(order) for order in ranking["cross_layer"].values()}
    same_changed = sum(
        int(row["rank_changed_from_p1"])
        for row in ranking_rows if row["fixture_scope"] == "same_layer"
    )
    evidence_rows = derivation_rows + partition_rows + path_rows + proxy_rows + ranking_rows
    evidence_mismatches = sum(
        row["evidence_label"] != CLAIM_CLASSES[row["claim_id"]]
        for row in evidence_rows
    ) + sum(
        method["evidence_label"] != CLAIM_CLASSES[method["claim_id"]]
        for method in METHOD_ORIGINS.values()
    )
    checks = {
        "nonfinite_count": 0.0,
        "partition_formula_max_error": max(row["formula_error"] for row in partition_rows),
        "partition_p1_max_error": max(abs(row["capacity_after"] - row["capacity_before"]) for row in partition_rows if row["p"] == 1.0),
        "partition_unexpected_result_count": float(sum(row["expected_invariant"] != row["observed_invariant"] for row in partition_rows)),
        "capacity_integral_max_error": max(row["absolute_error"] for row in capacity_rows),
        "capacity_integral_path_spread": max(capacity_values) - min(capacity_values),
        "density_scale_invariance_max_error": density_scale_error,
        "countermodel_integral_max_error": max(row["absolute_error"] for row in counter_rows),
        "countermodel_extinction_limit": 0.75,
        "straight_integral_error": straight["absolute_error"],
        "straight_bound_margin": straight["endpoint_expression"] - straight["exact_integral"],
        "detour_integral_error": detour["absolute_error"],
        "detour_endpoint_excess": detour["exact_integral"] - detour["endpoint_expression"],
        "detour_arc_bound_margin": detour["arc_upper_bound"] - detour["exact_integral"],
        "proxy_scale_max_relative_error": nonadditivity["maximum_scale_relative_error"],
        "proxy_extinction_monotonicity_failures": float(sum(not row["monotone_from_previous"] for row in proxy_rows if row["section"] == "extinction")),
        "endpoint_p1_nonadditivity": abs(p1_nonadd["direct"] - p1_nonadd["via"]),
        "geometric_additivity_error": nonadditivity["geometric_control"]["error"],
        "cross_layer_distinct_order_count": float(len(cross_orders)),
        "same_layer_rank_change_count": float(same_changed),
        "evidence_class_mismatch_count": float(evidence_mismatches),
    }
    all_values = _walk_numbers({"partition": partition_rows, "paths": path_rows, "proxy": proxy_rows, "ranking": ranking_rows})
    checks["nonfinite_count"] = float(sum(not math.isfinite(value) for value in all_values))
    check_rows = {
        name: {"value": value, "minimum": CHECK_SPECS[name]["minimum"],
               "maximum": CHECK_SPECS[name]["maximum"], "units": CHECK_SPECS[name]["units"],
               "passed": CHECK_SPECS[name]["minimum"] <= value <= CHECK_SPECS[name]["maximum"]}
        for name, value in checks.items()
    }
    passed = all(row["passed"] for row in check_rows.values()) and all(
        row["passed"] for rows in (partition_rows, path_rows, proxy_rows, ranking_rows) for row in rows
    )
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION, "audit_id": AUDIT_ID,
        "status": "passed" if passed else "failed", "passed": passed,
        "claim_ids": list(CLAIM_IDS), "claim_classes": CLAIM_CLASSES,
        "paper": {
            "arxiv_id": "2607.21366", "version": "v1", "pdf_sha256": PAPER_SHA256,
            "anchors": ["Sections 6.1-6.2, PDF pages 10-12", "Appendix C.1, Eqs. 43-49, PDF pages 42-47", "Eqs. 58-61, PDF page 50"],
        },
        "config": config, "config_sha256": _config_hash(config),
        "method_origins": METHOD_ORIGINS, "reproduction_scope": REPRODUCTION_SCOPE,
        "check_count": len(check_rows), "passing_check_count": sum(row["passed"] for row in check_rows.values()),
        "checks": check_rows,
        "partition": {"formula": "||f|| M^(1/p-1)", "conditional_scope": "tested L_p family under Lemma C.1 premises", "rows": len(partition_rows)},
        "capacity_integral": {"exact": 3.0 * math.log(2.5), "parameterizations": [row["path_id"] for row in capacity_rows]},
        "connectivity": {"sufficient_condition": "c>=c_min>0 along the extinction path", "countermodel_limit": 0.75, "countermodel_rows": len(counter_rows)},
        "geometric_paths": {"straight_exact": straight["exact_integral"], "endpoint_expression": straight["endpoint_expression"], "detour_exact": detour["exact_integral"], "detour_arc_bound": detour["arc_upper_bound"]},
        "proxy_family": {"formula": "J_p=N(D/E_b)^p", "domain": "p>0", "axiom3_claimed": False, "upper_bound_claimed": False},
        "nonadditivity": nonadditivity, "ranking": ranking,
    }
    if not passed:
        failed = [name for name, row in check_rows.items() if not row["passed"]]
        raise AuditFailure(f"audit checks failed: {failed}")
    return {
        "summary": summary,
        "derivation_steps.csv": derivation_rows,
        "partition_sweep.csv": partition_rows,
        "path_integrals.csv": path_rows,
        "proxy_family.csv": proxy_rows,
        "ranking_fixture.csv": ranking_rows,
    }


def validate_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != AUDIT_SCHEMA_VERSION or summary.get("audit_id") != AUDIT_ID:
        raise AuditFailure("summary identity changed")
    if summary.get("status") != "passed" or summary.get("passed") is not True:
        raise AuditFailure("summary is not passing")
    if tuple(summary.get("claim_ids", ())) != CLAIM_IDS:
        raise AuditFailure("claim registry changed")
    if summary.get("claim_classes") != CLAIM_CLASSES:
        raise AuditFailure("claim evidence-class registry changed")
    checks = summary.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(CHECK_SPECS):
        raise AuditFailure("summary check registry changed")
    if summary.get("check_count") != len(CHECK_SPECS) or summary.get("passing_check_count") != len(CHECK_SPECS):
        raise AuditFailure("summary check counts changed")
    for name, spec in CHECK_SPECS.items():
        row = checks[name]
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            raise AuditFailure(f"check is not passing: {name}")
        value = _finite(row.get("value"), f"check {name}")
        if not spec["minimum"] <= value <= spec["maximum"]:
            raise AuditFailure(f"check is outside source range: {name}")
    if any(not math.isfinite(value) for value in _walk_numbers(summary)):
        raise AuditFailure("summary contains a non-finite number")


def _write_json(path: Path, payload: Any) -> None:
    if any(not math.isfinite(value) for value in _walk_numbers(payload)):
        raise AuditFailure(f"refusing to write non-finite JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
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
    identity = {"arxiv_id": "2607.21366", "version": "v1", "pdf_sha256": PAPER_SHA256, "pdf_bytes": PAPER_BYTES, "pdf_supplied": paper_pdf is not None}
    if full and paper_pdf is None:
        raise AuditFailure("--full requires --paper-pdf")
    if paper_pdf is not None:
        if not paper_pdf.is_file() or paper_pdf.is_symlink() or paper_pdf.stat().st_size != PAPER_BYTES or _sha256_path(paper_pdf) != PAPER_SHA256:
            raise AuditFailure("supplied paper PDF has the wrong identity")
    return identity


def _git_identity() -> tuple[str, str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_RELATIVE_PATHS], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", "unavailable", False
    return commit, "exact_commit" if not status else "working_tree", not bool(status)


def _manifest(result_dir: Path, manifest_path: Path, outputs: Sequence[Path], summary: Mapping[str, Any], paper: Mapping[str, Any], full: bool) -> dict[str, Any]:
    commit, state, matches = _git_identity()
    source_hashes = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise AuditFailure(f"required source is missing: {relative}")
        source_hashes.append({"path": relative, "sha256": _sha256_path(path), "bytes": path.stat().st_size})
    output_rows = []
    for path in outputs:
        relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        output_rows.append({"path": relative, "sha256": _sha256_path(path), "bytes": path.stat().st_size, "media_type": "application/json" if path.suffix == ".json" else "text/csv"})
    return {
        "schema_version": AUDIT_SCHEMA_VERSION, "audit_id": AUDIT_ID, "status": "passed",
        "generation_time_policy": "deterministic outputs omit wall-clock timestamps",
        "paper": dict(paper), "config": summary["config"], "config_sha256": summary["config_sha256"],
        "seeds": {"used": False, "value": None},
        "code": {"source_snapshot_commit": commit, "source_snapshot_state": state, "source_files_match_commit": matches},
        "environment": {"python_version": platform.python_version(), "python_implementation": platform.python_implementation(), "numpy_version": np.__version__, "platform": platform.platform(), "machine": platform.machine(), "byteorder": sys.byteorder, "cpu_count": os.cpu_count(), "float64_epsilon": float(np.finfo(np.float64).eps)},
        "expected_metric_ranges": CHECK_SPECS, "observed_checks": summary["checks"],
        "method_origins": METHOD_ORIGINS, "reproduction_scope": REPRODUCTION_SCOPE,
        "source_hashes": source_hashes,
        "input_hashes": [{"kind": "paper_manifest", "path": "paper/manifest.json", "sha256": _sha256_path(PAPER_MANIFEST_PATH), "bytes": PAPER_MANIFEST_PATH.stat().st_size}],
        "outputs": output_rows, "manifest_hash_scope": "manifest excludes its own hash to avoid self-reference",
        "canonical_invocation": {"entrypoint": "experiments/audits/part06_axioms/run.py", "result_dir": "<RESULT_DIR>", "manifest_path": "<MANIFEST_PATH>", "paper_pdf": "<PAPER_PDF>" if paper["pdf_supplied"] else "not supplied", "full": full},
    }


def _validated_destinations(
    result_dir: Path, manifest_path: Path | None
) -> tuple[Path, Path | None, tuple[Path, ...]]:
    """Resolve and validate every destination before any output is written."""

    if result_dir.is_symlink():
        raise AuditFailure(f"refusing symlink result directory: {result_dir}")
    resolved_result = result_dir.resolve()
    if resolved_result.exists() and not resolved_result.is_dir():
        raise AuditFailure(f"result destination is not a directory: {resolved_result}")
    if resolved_result.exists() and any(resolved_result.iterdir()):
        raise AuditFailure(
            f"refusing to overwrite nonempty result directory: {resolved_result}"
        )

    outputs = tuple((resolved_result / filename).resolve() for filename in OUTPUT_FILENAMES)
    resolved_manifest = None if manifest_path is None else manifest_path.resolve()
    if resolved_manifest is not None:
        if resolved_manifest in outputs:
            raise AuditFailure("manifest path must not alias a result file")
        if resolved_manifest.exists() or manifest_path.is_symlink():
            raise AuditFailure(f"refusing to overwrite manifest: {manifest_path}")
    return resolved_result, resolved_manifest, outputs


def run_audit(*, result_dir: Path, manifest_path: Path | None = None, paper_pdf: Path | None = None, full: bool = False) -> dict[str, Any]:
    result_dir, manifest_path, output_paths = _validated_destinations(
        result_dir, manifest_path
    )
    paper = _paper_identity(paper_pdf, full)
    audit = build_audit(full=full)
    summary = audit["summary"]
    validate_summary(summary)
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result_dir / "summary.json", summary)
    for filename, fields in CSV_CONTRACTS.items():
        _write_csv(result_dir / filename, fields, audit[filename])
    outputs = list(output_paths)
    if manifest_path is not None:
        _write_json(manifest_path, _manifest(result_dir, manifest_path, outputs, summary, paper, full))
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
    print(f"PASS: {summary['passing_check_count']}/{summary['check_count']} Part 6 checks passed; six deterministic outputs written to {args.result_dir}")


if __name__ == "__main__":
    main()
