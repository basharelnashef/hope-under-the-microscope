#!/usr/bin/env python3
"""Fail-closed audit of HOPE v1 residual-block eviction semantics."""

from __future__ import annotations

import argparse
import csv
import functools
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
AUDIT_ID = "part08_block_eviction_audit"
AUDIT_SCHEMA_VERSION = 2
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
PAPER_BYTES = 1_089_944
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_08" / "manual"
MONTE_CARLO_SEED = 260_721_366
MONTE_CARLO_SAMPLES = 500_000
QUADRATURE_LIMIT = 12.0
QUADRATURE_INTERVALS = 240_000

CLAIM_IDS = (
    "P08-EVICTION-SCORE",
    "P08-POST-RELU-BOUNDARY",
    "BLOCK-IDENTITY-MOMENT",
    "P08-BIASED-MOMENT-AUDIT",
    "P08-ORDERING-FLIP",
    "P08-SEQUENTIAL-DEPTH-GAP",
    "P08-IDENTITY-ELIGIBILITY",
    "P08-RELU-MOMENT-REPAIR",
)
CLAIM_CLASSES = {
    "P08-EVICTION-SCORE": "paper_claim",
    "P08-POST-RELU-BOUNDARY": "verified_fact",
    "BLOCK-IDENTITY-MOMENT": "independent_inference",
    "P08-BIASED-MOMENT-AUDIT": "verified_fact",
    "P08-ORDERING-FLIP": "counterexample",
    "P08-SEQUENTIAL-DEPTH-GAP": "counterexample",
    "P08-IDENTITY-ELIGIBILITY": "verified_fact",
    "P08-RELU-MOMENT-REPAIR": "proposed_repair",
}

SHIFT_ALPHAS = (-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
VALIDATION_CASES = (
    ("alpha_minus_4", -4.0, 1.0),
    ("alpha_minus_2", -2.0, 1.0),
    ("alpha_minus_1", -1.0, 1.0),
    ("zero_shift", 0.0, 1.0),
    ("alpha_plus_1", 1.0, 1.0),
    ("alpha_plus_4", 4.0, 1.0),
    ("scaled_bias", 0.75, 1.5),
    ("degenerate_negative", -1.0, 0.0),
    ("degenerate_positive", 1.0, 0.0),
)

TOLERANCES = {
    "formula_atol": 2e-12,
    "quadrature_atol": 2e-8,
    "monte_carlo_atol": 1.5e-2,
    "graph_monte_carlo_atol": 8e-3,
    "function_atol": 0.0,
    "ordering_atol": 1e-12,
}

METHOD_ORIGINS = {
    "paper_identity_energy": {
        "evidence_label": "paper_claim",
        "claim_id": "P08-EVICTION-SCORE",
        "origin": "HOPE v1 Eq. 96",
        "scope": "sum of unrectified coordinate RMS values under the stated BN Gaussian",
    },
    "post_relu_boundary": {
        "evidence_label": "verified_fact",
        "claim_id": "P08-POST-RELU-BOUNDARY",
        "origin": "composition of Figure 3's terminal ReLU with the next interior identity block",
        "scope": "canonical post-activation ResNet V1 identity bottlenecks only",
    },
    "moment_comparison": {
        "evidence_label": "independent_inference",
        "claim_id": "BLOCK-IDENTITY-MOMENT",
        "origin": "conditional comparison of Eq. 96 with a Gaussian pre-ReLU fixture",
        "scope": "synthetic Gaussian coordinate with declared beta and gamma; not a trained-block measurement",
    },
    "post_relu_denominator": {
        "evidence_label": "proposed_repair",
        "claim_id": "P08-RELU-MOMENT-REPAIR",
        "origin": "use the actual boundary second moment; use Eq. 79 only under a validated Gaussian law",
        "scope": "eligible identity shortcuts with a measured or separately derived post-ReLU boundary law",
    },
    "depth_witness": {
        "evidence_label": "counterexample",
        "claim_id": "P08-SEQUENTIAL-DEPTH-GAP",
        "origin": "independent ReLU-idempotence fixture",
        "scope": "refutes a residual-function-only interpretation of the additive local-layer proxy",
    },
}

REPRODUCTION_SCOPE = {
    "reproduction_scope": "independent_mathematical_and_graph_fixture_audit",
    "paper_experiment_reproduced": False,
    "checkpoint_or_dataset_used": False,
    "random_sampling_used": True,
    "includes": [
        "paper source map for Figure 3, Eqs. 19-20, and Appendix F Eqs. 86-98",
        "analytic and deterministic-quadrature Gaussian ReLU second moments",
        "seeded Monte Carlo checks of the conditional Gaussian post-ReLU fixture",
        "bounded beta over gamma sweep",
        "equal-payoff two-candidate ordering fixture",
        "ReLU-idempotence sequential-depth fixture",
        "identity versus projection shortcut eligibility classification",
    ],
    "excludes": [
        "ResNet-50 or ImageNet reproduction",
        "trained-network block frequencies or compression accuracy",
        "the post-add activation law or its parameters in a trained residual block",
        "projection-shortcut capacity derivation or repair",
        "stride-changing shortcut derivation",
        "end-to-end logits, loss, latency, or accuracy",
        "Parts 9 through 12",
    ],
    "claim_boundary": (
        "The audit establishes the Figure 3 topology and conditional Gaussian "
        "sensitivity fixtures. It does not identify a trained block's activation law, "
        "estimate prevalence, or establish downstream performance."
    ),
}

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part08_block_eviction/README.md",
    "experiments/audits/part08_block_eviction/run.py",
    "experiments/audits/part08_block_eviction/test_run.py",
)

SOURCE_MAP_FIELDS = (
    "step_id", "sequence", "paper_anchor", "object", "formula_or_action",
    "result", "evidence_label", "claim_id", "passed",
)
MOMENT_FIELDS = (
    "case_id", "beta", "gamma", "alpha", "paper_second_moment",
    "analytic_post_second_moment", "quadrature_post_second_moment",
    "monte_carlo_post_second_moment", "quadrature_abs_error",
    "monte_carlo_abs_error", "sample_count", "seed", "evidence_label",
    "claim_id", "passed",
)
SHIFT_FIELDS = (
    "case_id", "beta", "gamma", "alpha", "paper_identity_rms",
    "post_relu_identity_rms", "rms_ratio", "ratio_defined",
    "paper_cost_fraction_of_post", "identity_overstatement_fraction",
    "evidence_label", "claim_id", "passed",
)
ORDERING_FIELDS = (
    "fixture_id", "candidate", "numerator", "payoff", "beta", "gamma",
    "paper_identity_rms", "post_relu_identity_rms", "paper_dr",
    "post_relu_dr", "paper_rank", "post_relu_rank", "expected_flip",
    "evidence_label", "claim_id", "passed",
)
DEPTH_FIELDS = (
    "case_id", "depth", "local_capacity", "paper_identity_capacity",
    "post_relu_identity_capacity", "residual_function_rms",
    "function_max_abs_difference", "paper_proxy", "post_relu_proxy",
    "paper_proxy_over_depth1", "post_relu_proxy_over_depth1",
    "evidence_label", "claim_id", "passed",
)
GRAPH_FIELDS = (
    "case_id", "beta", "gamma", "sample_count", "seed",
    "conditional_unrectified_second_moment", "conditional_post_relu_second_moment",
    "empirical_unrectified_second_moment", "empirical_post_relu_second_moment",
    "synthetic_identity_copy_max_error", "post_moment_abs_error", "evidence_label",
    "claim_id", "passed",
)
ELIGIBILITY_FIELDS = (
    "case_id", "architecture", "shortcut", "stride", "input_channels",
    "output_channels", "boundary", "eq95_identity_decomposition_eligible",
    "post_relu_boundary_audit_eligible", "gaussian_closed_form_established",
    "reason", "evidence_label", "claim_id", "passed",
)

CSV_CONTRACTS = {
    "source_map.csv": SOURCE_MAP_FIELDS,
    "moment_validation.csv": MOMENT_FIELDS,
    "shift_sweep.csv": SHIFT_FIELDS,
    "ordering_fixtures.csv": ORDERING_FIELDS,
    "depth_fixture.csv": DEPTH_FIELDS,
    "graph_boundary.csv": GRAPH_FIELDS,
    "eligibility.csv": ELIGIBILITY_FIELDS,
}
OUTPUT_FILENAMES = ("summary.json", *CSV_CONTRACTS.keys())

CHECK_SPECS = {
    "nonfinite_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "row_failure_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "evidence_class_mismatch_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "quadrature_max_abs_error": {"minimum": 0.0, "maximum": 2e-8, "units": "second_moment"},
    "monte_carlo_max_abs_error": {"minimum": 0.0, "maximum": 1.5e-2, "units": "second_moment"},
    "graph_post_moment_max_abs_error": {"minimum": 0.0, "maximum": 8e-3, "units": "second_moment"},
    "synthetic_identity_copy_max_error": {"minimum": 0.0, "maximum": 0.0, "units": "absolute"},
    "zero_shift_rms_ratio_error": {"minimum": 0.0, "maximum": 2e-12, "units": "absolute"},
    "zero_shift_cost_understatement_error": {"minimum": 0.0, "maximum": 2e-12, "units": "fraction"},
    "negative_ratio_monotonic_failures": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "alpha_minus_4_rms_ratio": {"minimum": 2300.0, "maximum": 2400.0, "units": "ratio"},
    "degenerate_negative_post_rms": {"minimum": 0.0, "maximum": 0.0, "units": "rms"},
    "degenerate_negative_paper_rms": {"minimum": 1.0, "maximum": 1.0, "units": "rms"},
    "ordering_expected_flip_count": {"minimum": 1.0, "maximum": 1.0, "units": "count"},
    "ordering_unexpected_flip_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "depth_function_max_difference": {"minimum": 0.0, "maximum": 0.0, "units": "absolute"},
    "depth4_proxy_ratio_error": {"minimum": 0.0, "maximum": 2e-12, "units": "absolute"},
    "eligible_post_relu_boundary_fixture_count": {"minimum": 1.0, "maximum": 1.0, "units": "count"},
    "established_gaussian_closed_form_fixture_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "projection_fixture_count": {"minimum": 2.0, "maximum": 2.0, "units": "count"},
}


class AuditFailure(RuntimeError):
    """Raised when a declared audit invariant fails."""


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def paper_second_moment(beta: float, gamma: float) -> float:
    return float(beta) ** 2 + float(gamma) ** 2


def relu_second_moment(beta: float, gamma: float) -> float:
    beta = float(beta)
    scale = abs(float(gamma))
    if not math.isfinite(beta) or not math.isfinite(scale):
        raise ValueError("beta and gamma must be finite")
    if scale == 0.0:
        return max(0.0, beta) ** 2
    alpha = beta / scale
    value = (scale * scale + beta * beta) * normal_cdf(alpha)
    value += beta * scale * normal_pdf(alpha)
    if value < 0.0 and abs(value) <= 1e-14 * max(1.0, scale * scale + beta * beta):
        return 0.0
    if value < 0.0:
        raise AuditFailure(f"negative ReLU second moment: {value}")
    return value


@functools.lru_cache(maxsize=1)
def _quadrature_grid() -> tuple[np.ndarray, np.ndarray, float]:
    z = np.linspace(
        -QUADRATURE_LIMIT,
        QUADRATURE_LIMIT,
        QUADRATURE_INTERVALS + 1,
        dtype=np.float64,
    )
    weights = np.ones_like(z)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    density = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    step_over_three = (2.0 * QUADRATURE_LIMIT / QUADRATURE_INTERVALS) / 3.0
    return z, weights * density, step_over_three


def quadrature_relu_second_moment(beta: float, gamma: float) -> float:
    scale = abs(float(gamma))
    if scale == 0.0:
        return max(0.0, float(beta)) ** 2
    z, weighted_density, step_over_three = _quadrature_grid()
    values = np.maximum(float(beta) + scale * z, 0.0) ** 2
    return float(step_over_three * np.sum(weighted_density * values, dtype=np.float64))


def build_config(*, full: bool) -> dict[str, Any]:
    return {
        "float_dtype": "float64",
        "full_mode": bool(full),
        "random_sampling": True,
        "monte_carlo_seed": MONTE_CARLO_SEED,
        "monte_carlo_samples": MONTE_CARLO_SAMPLES,
        "quadrature_rule": "composite_simpson_standard_normal",
        "quadrature_limit": QUADRATURE_LIMIT,
        "quadrature_intervals": QUADRATURE_INTERVALS,
        "shift_alphas": list(SHIFT_ALPHAS),
        "tolerances": dict(TOLERANCES),
    }


def _source_map_rows() -> list[dict[str, Any]]:
    specifications = (
        (
            "figure3_boundary", "Figure 3, PDF page 16", "graph boundary",
            "one block ends at Z=ReLU(Y), and the next interior identity block receives X",
            "the audited boundary is post-ReLU for consecutive post-activation V1 identity blocks",
            "verified_fact", "P08-POST-RELU-BOUNDARY",
        ),
        (
            "macro_sum", "Eqs. 19-20, PDF page 17", "paper score",
            "J_evict=sum_l N_active^(l) E_active^(l) / E_identity",
            "two local internal-layer terms are added for the canonical bottleneck",
            "paper_claim", "P08-EVICTION-SCORE",
        ),
        (
            "identity_operator", "Eq. 95, PDF page 58", "skip operator",
            "I_k(x)=x_k e_k and ||I_k||=sqrt(E[x_k^2])",
            "the derivation is for a coordinate identity shortcut",
            "verified_fact", "P08-IDENTITY-ELIGIBILITY",
        ),
        (
            "paper_identity_moment", "Eq. 96, PDF page 58", "paper moment",
            "E_identity=sum_k sqrt(gamma_k^2+beta_k^2)",
            "the printed formula uses an unrectified Gaussian second moment",
            "paper_claim", "P08-EVICTION-SCORE",
        ),
        (
            "relu_self_moment", "Eq. 79, PDF page 53", "activation moment",
            "E[ReLU(Y)^2] for Y distributed as N(beta,gamma^2)",
            "the biased post-ReLU moment is already available in the paper",
            "verified_fact", "P08-BIASED-MOMENT-AUDIT",
        ),
        (
            "boundary_comparison", "Figure 3 and Eq. 96", "independent comparison",
            "sqrt(E[Y^2]) versus sqrt(E[ReLU(Y)^2])",
            "the two RMS values differ under the declared synthetic Gaussian law; Figure 3 does not establish that law",
            "independent_inference", "BLOCK-IDENTITY-MOMENT",
        ),
        (
            "sequential_restriction", "Appendix F.2, PDF page 57", "state construction",
            "sequential layers cannot be grouped as additive components of one macro state",
            "the appendix explicitly recognizes the composition-versus-addition conflict",
            "paper_claim", "P08-EVICTION-SCORE",
        ),
        (
            "linear_sum", "Eqs. 94 and 97, PDF page 58", "paper proxy",
            "separate local-layer bounds are summed after individual null projections",
            "the resulting score depends on the number of charged sequential layers",
            "counterexample", "P08-SEQUENTIAL-DEPTH-GAP",
        ),
        (
            "post_relu_repair", "independent scoped repair", "proposed denominator",
            "sum_k sqrt(E[X_k^2]); optionally use Eq. 79 only under a validated Gaussian law",
            "use the actual eligible boundary moment without inferring its distribution from topology alone",
            "proposed_repair", "P08-RELU-MOMENT-REPAIR",
        ),
    )
    return [
        {
            "step_id": step_id,
            "sequence": sequence,
            "paper_anchor": anchor,
            "object": object_name,
            "formula_or_action": formula,
            "result": result,
            "evidence_label": evidence_label,
            "claim_id": claim_id,
            "passed": True,
        }
        for sequence, (
            step_id, anchor, object_name, formula, result, evidence_label, claim_id
        ) in enumerate(specifications, start=1)
    ]


def _standard_samples() -> np.ndarray:
    return np.random.default_rng(MONTE_CARLO_SEED).standard_normal(MONTE_CARLO_SAMPLES)


def _moment_rows(samples: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, beta, gamma in VALIDATION_CASES:
        analytic = relu_second_moment(beta, gamma)
        quadrature = quadrature_relu_second_moment(beta, gamma)
        values = np.maximum(beta + abs(gamma) * samples, 0.0)
        monte_carlo = float(np.mean(values * values, dtype=np.float64))
        quadrature_error = abs(quadrature - analytic)
        monte_carlo_error = abs(monte_carlo - analytic)
        passed = (
            quadrature_error <= TOLERANCES["quadrature_atol"]
            and monte_carlo_error <= TOLERANCES["monte_carlo_atol"]
        )
        rows.append({
            "case_id": case_id,
            "beta": beta,
            "gamma": gamma,
            "alpha": None if gamma == 0.0 else beta / abs(gamma),
            "paper_second_moment": paper_second_moment(beta, gamma),
            "analytic_post_second_moment": analytic,
            "quadrature_post_second_moment": quadrature,
            "monte_carlo_post_second_moment": monte_carlo,
            "quadrature_abs_error": quadrature_error,
            "monte_carlo_abs_error": monte_carlo_error,
            "sample_count": MONTE_CARLO_SAMPLES,
            "seed": MONTE_CARLO_SEED,
            "evidence_label": "verified_fact",
            "claim_id": "P08-BIASED-MOMENT-AUDIT",
            "passed": passed,
        })
    return rows


def _shift_row(case_id: str, beta: float, gamma: float) -> dict[str, Any]:
    paper_rms = math.sqrt(paper_second_moment(beta, gamma))
    post_rms = math.sqrt(relu_second_moment(beta, gamma))
    ratio_defined = post_rms > 0.0
    ratio = paper_rms / post_rms if ratio_defined else None
    paper_cost_fraction = post_rms / paper_rms if paper_rms > 0.0 else None
    overstatement = 1.0 - paper_cost_fraction if paper_cost_fraction is not None else 0.0
    passed = (
        paper_rms >= post_rms >= 0.0
        and (ratio is None or ratio >= 1.0)
        and 0.0 <= overstatement <= 1.0
    )
    return {
        "case_id": case_id,
        "beta": beta,
        "gamma": gamma,
        "alpha": None if gamma == 0.0 else beta / abs(gamma),
        "paper_identity_rms": paper_rms,
        "post_relu_identity_rms": post_rms,
        "rms_ratio": ratio,
        "ratio_defined": ratio_defined,
        "paper_cost_fraction_of_post": paper_cost_fraction,
        "identity_overstatement_fraction": overstatement,
        "evidence_label": "independent_inference",
        "claim_id": "BLOCK-IDENTITY-MOMENT",
        "passed": passed,
    }


def _shift_rows() -> list[dict[str, Any]]:
    rows = [_shift_row(f"alpha_{alpha:g}", alpha, 1.0) for alpha in SHIFT_ALPHAS]
    rows.extend((
        _shift_row("degenerate_negative", -1.0, 0.0),
        _shift_row("degenerate_zero", 0.0, 0.0),
        _shift_row("degenerate_positive", 1.0, 0.0),
    ))
    return rows


def _ordering_rows() -> list[dict[str, Any]]:
    fixtures = (
        (
            "negative_shift_flip",
            True,
            (("A", 1.0, 1.0, -1.0, 1.0), ("B", 0.85, 1.0, 0.0, 1.0)),
        ),
        (
            "zero_shift_control",
            False,
            (("A", 1.0, 1.0, 0.0, 1.0), ("B", 1.1, 1.0, 0.0, 2.0)),
        ),
        (
            "positive_shift_control",
            False,
            (("A", 1.0, 1.0, 1.0, 1.0), ("B", 1.2, 1.0, 1.0, 1.0)),
        ),
    )
    rows: list[dict[str, Any]] = []
    for fixture_id, expected_flip, candidates in fixtures:
        working: list[dict[str, Any]] = []
        for candidate, numerator, payoff, beta, gamma in candidates:
            paper_rms = math.sqrt(paper_second_moment(beta, gamma))
            post_rms = math.sqrt(relu_second_moment(beta, gamma))
            working.append({
                "fixture_id": fixture_id,
                "candidate": candidate,
                "numerator": numerator,
                "payoff": payoff,
                "beta": beta,
                "gamma": gamma,
                "paper_identity_rms": paper_rms,
                "post_relu_identity_rms": post_rms,
                "paper_dr": numerator / paper_rms / payoff,
                "post_relu_dr": numerator / post_rms / payoff,
            })
        paper_order = sorted(working, key=lambda row: (row["paper_dr"], row["candidate"]))
        post_order = sorted(working, key=lambda row: (row["post_relu_dr"], row["candidate"]))
        paper_rank = {row["candidate"]: rank for rank, row in enumerate(paper_order, start=1)}
        post_rank = {row["candidate"]: rank for rank, row in enumerate(post_order, start=1)}
        observed_flip = [row["candidate"] for row in paper_order] != [
            row["candidate"] for row in post_order
        ]
        for row in working:
            row.update({
                "paper_rank": paper_rank[row["candidate"]],
                "post_relu_rank": post_rank[row["candidate"]],
                "expected_flip": expected_flip,
                "evidence_label": "counterexample",
                "claim_id": "P08-ORDERING-FLIP",
                "passed": observed_flip is expected_flip,
            })
            rows.append(row)
    return rows


def _depth_rows(samples: np.ndarray) -> list[dict[str, Any]]:
    base = np.maximum(samples, 0.0)
    local_capacity = math.sqrt(0.5)
    paper_identity = 1.0
    post_identity = math.sqrt(0.5)
    rows: list[dict[str, Any]] = []
    for depth in (1, 2, 3, 4):
        output = samples.copy()
        for _ in range(depth):
            output = np.maximum(output, 0.0)
        difference = float(np.max(np.abs(output - base)))
        paper_proxy = depth * local_capacity / paper_identity
        post_proxy = depth * local_capacity / post_identity
        rows.append({
            "case_id": f"relu_depth_{depth}",
            "depth": depth,
            "local_capacity": local_capacity,
            "paper_identity_capacity": paper_identity,
            "post_relu_identity_capacity": post_identity,
            "residual_function_rms": math.sqrt(float(np.mean(output * output))),
            "function_max_abs_difference": difference,
            "paper_proxy": paper_proxy,
            "post_relu_proxy": post_proxy,
            "paper_proxy_over_depth1": float(depth),
            "post_relu_proxy_over_depth1": float(depth),
            "evidence_label": "counterexample",
            "claim_id": "P08-SEQUENTIAL-DEPTH-GAP",
            "passed": difference <= TOLERANCES["function_atol"],
        })
    return rows


def _graph_rows(samples: np.ndarray) -> list[dict[str, Any]]:
    """Exercise a declared synthetic Gaussian-to-ReLU identity fixture.

    These rows validate the conditional calculation only.  They do not model
    the residual addition or claim that beta and gamma describe a trained
    block's post-add pre-activation.
    """

    cases = (("zero_shift", 0.0, 1.0), ("negative_shift", -1.0, 1.0), ("positive_shift", 1.0, 1.0))
    rows: list[dict[str, Any]] = []
    for case_id, beta, gamma in cases:
        pre = beta + gamma * samples
        previous_z = np.maximum(pre, 0.0)
        next_block_x = previous_z.copy()
        empirical_pre = float(np.mean(pre * pre, dtype=np.float64))
        empirical_post = float(np.mean(next_block_x * next_block_x, dtype=np.float64))
        analytic_post = relu_second_moment(beta, gamma)
        identity_error = float(np.max(np.abs(next_block_x - previous_z)))
        post_error = abs(empirical_post - analytic_post)
        rows.append({
            "case_id": case_id,
            "beta": beta,
            "gamma": gamma,
            "sample_count": MONTE_CARLO_SAMPLES,
            "seed": MONTE_CARLO_SEED,
            "conditional_unrectified_second_moment": paper_second_moment(beta, gamma),
            "conditional_post_relu_second_moment": analytic_post,
            "empirical_unrectified_second_moment": empirical_pre,
            "empirical_post_relu_second_moment": empirical_post,
            "synthetic_identity_copy_max_error": identity_error,
            "post_moment_abs_error": post_error,
            "evidence_label": "independent_inference",
            "claim_id": "BLOCK-IDENTITY-MOMENT",
            "passed": (
                identity_error == 0.0
                and post_error <= TOLERANCES["graph_monte_carlo_atol"]
            ),
        })
    return rows


def _eligibility_rows() -> list[dict[str, Any]]:
    specifications = (
        (
            "v1_interior_identity", "post-activation ResNet V1 bottleneck",
            "identity", 1, 256, 256, "previous block post-ReLU", True, True, False,
            "same-shape coordinate identity and post-ReLU topology; Gaussian law not established",
        ),
        (
            "v1_projection_channels", "post-activation ResNet V1 bottleneck",
            "learned 1x1 projection", 1, 256, 512, "previous block post-ReLU", False, False, False,
            "shortcut is a learned operator and changes channel dimension",
        ),
        (
            "v1_projection_stride", "post-activation ResNet V1 bottleneck",
            "learned 1x1 projection", 2, 512, 1024, "previous block post-ReLU", False, False, False,
            "shortcut is learned and changes spatial sampling and channels",
        ),
        (
            "v2_identity", "pre-activation ResNet V2 bottleneck",
            "identity", 1, 256, 256, "no Figure 3 terminal-ReLU contract", True, False, False,
            "Eq. 95 identity algebra applies, but the Part 8 post-ReLU boundary inference does not",
        ),
        (
            "sequential_no_skip", "sequential convolutional block",
            "none", 1, 256, 256, "no parallel survival path", False, False, False,
            "there is no identity shortcut to supply E_identity",
        ),
    )
    return [
        {
            "case_id": case_id,
            "architecture": architecture,
            "shortcut": shortcut,
            "stride": stride,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "boundary": boundary,
            "eq95_identity_decomposition_eligible": eq95_eligible,
            "post_relu_boundary_audit_eligible": boundary_eligible,
            "gaussian_closed_form_established": gaussian_established,
            "reason": reason,
            "evidence_label": "verified_fact",
            "claim_id": "P08-IDENTITY-ELIGIBILITY",
            "passed": (
                eq95_eligible == (shortcut == "identity" and input_channels == output_channels and stride == 1)
                and (not boundary_eligible or eq95_eligible)
                and (not gaussian_established or boundary_eligible)
            ),
        }
        for (
            case_id, architecture, shortcut, stride, input_channels, output_channels,
            boundary, eq95_eligible, boundary_eligible, gaussian_established, reason
        ) in specifications
    ]


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


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def build_audit(*, full: bool = False) -> dict[str, Any]:
    config = build_config(full=full)
    samples = _standard_samples()
    source_rows = _source_map_rows()
    moment_rows = _moment_rows(samples)
    shift_rows = _shift_rows()
    ordering_rows = _ordering_rows()
    depth_rows = _depth_rows(samples)
    graph_rows = _graph_rows(samples)
    eligibility_rows = _eligibility_rows()
    row_groups = (
        source_rows, moment_rows, shift_rows, ordering_rows, depth_rows,
        graph_rows, eligibility_rows,
    )
    all_rows = [row for group in row_groups for row in group]
    evidence_mismatches = sum(
        row["evidence_label"] != CLAIM_CLASSES[row["claim_id"]] for row in all_rows
    ) + sum(
        record["evidence_label"] != CLAIM_CLASSES[record["claim_id"]]
        for record in METHOD_ORIGINS.values()
    )

    shift_by_id = {row["case_id"]: row for row in shift_rows}
    zero = shift_by_id["alpha_0"]
    negative = [shift_by_id[f"alpha_{value:g}"] for value in (-1.0, -2.0, -3.0, -4.0)]
    negative_monotonic_failures = sum(
        not (negative[index + 1]["rms_ratio"] > negative[index]["rms_ratio"])
        for index in range(len(negative) - 1)
    )
    fixture_orders: dict[str, tuple[tuple[str, ...], tuple[str, ...], bool]] = {}
    for fixture_id in sorted({row["fixture_id"] for row in ordering_rows}):
        rows = [row for row in ordering_rows if row["fixture_id"] == fixture_id]
        paper_order = tuple(row["candidate"] for row in sorted(rows, key=lambda row: row["paper_rank"]))
        post_order = tuple(row["candidate"] for row in sorted(rows, key=lambda row: row["post_relu_rank"]))
        fixture_orders[fixture_id] = (paper_order, post_order, bool(rows[0]["expected_flip"]))
    expected_flip_count = sum(
        paper != post and expected for paper, post, expected in fixture_orders.values()
    )
    unexpected_flip_count = sum(
        paper != post and not expected for paper, post, expected in fixture_orders.values()
    )
    depth4 = next(row for row in depth_rows if row["depth"] == 4)

    checks = {
        "nonfinite_count": float(sum(
            not math.isfinite(number) for number in _walk_numbers(all_rows)
        )),
        "row_failure_count": float(sum(not row["passed"] for row in all_rows)),
        "evidence_class_mismatch_count": float(evidence_mismatches),
        "quadrature_max_abs_error": max(row["quadrature_abs_error"] for row in moment_rows),
        "monte_carlo_max_abs_error": max(row["monte_carlo_abs_error"] for row in moment_rows),
        "graph_post_moment_max_abs_error": max(row["post_moment_abs_error"] for row in graph_rows),
        "synthetic_identity_copy_max_error": max(
            row["synthetic_identity_copy_max_error"] for row in graph_rows
        ),
        "zero_shift_rms_ratio_error": abs(zero["rms_ratio"] - math.sqrt(2.0)),
        "zero_shift_cost_understatement_error": abs(
            zero["identity_overstatement_fraction"] - (1.0 - 1.0 / math.sqrt(2.0))
        ),
        "negative_ratio_monotonic_failures": float(negative_monotonic_failures),
        "alpha_minus_4_rms_ratio": shift_by_id["alpha_-4"]["rms_ratio"],
        "degenerate_negative_post_rms": shift_by_id["degenerate_negative"]["post_relu_identity_rms"],
        "degenerate_negative_paper_rms": shift_by_id["degenerate_negative"]["paper_identity_rms"],
        "ordering_expected_flip_count": float(expected_flip_count),
        "ordering_unexpected_flip_count": float(unexpected_flip_count),
        "depth_function_max_difference": max(row["function_max_abs_difference"] for row in depth_rows),
        "depth4_proxy_ratio_error": abs(depth4["paper_proxy_over_depth1"] - 4.0),
        "eligible_post_relu_boundary_fixture_count": float(sum(
            row["post_relu_boundary_audit_eligible"] for row in eligibility_rows
        )),
        "established_gaussian_closed_form_fixture_count": float(sum(
            row["gaussian_closed_form_established"] for row in eligibility_rows
        )),
        "projection_fixture_count": float(sum(
            "projection" in row["shortcut"] for row in eligibility_rows
        )),
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
    passed = all(row["passed"] for row in all_rows) and all(
        row["passed"] for row in check_rows.values()
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
                "Figure 3 and Section 8, PDF pages 16-17",
                "Eqs. 19-20, PDF page 17",
                "Eq. 79, PDF page 53",
                "Appendix F and Eqs. 86-98, PDF pages 56-59",
            ],
        },
        "config": config,
        "config_sha256": _config_hash(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": REPRODUCTION_SCOPE,
        "check_count": len(check_rows),
        "passing_check_count": sum(row["passed"] for row in check_rows.values()),
        "checks": check_rows,
        "zero_shift": {
            "paper_identity_rms": 1.0,
            "post_relu_identity_rms": 1.0 / math.sqrt(2.0),
            "paper_over_post_ratio": math.sqrt(2.0),
            "paper_cost_fraction_of_post": 1.0 / math.sqrt(2.0),
            "paper_cost_understatement_fraction": 1.0 - 1.0 / math.sqrt(2.0),
        },
        "negative_shift_boundary": {
            "finite_fixture_alpha": -4.0,
            "paper_over_post_rms_ratio": shift_by_id["alpha_-4"]["rms_ratio"],
            "degenerate_negative_gamma_zero_paper_rms": 1.0,
            "degenerate_negative_gamma_zero_post_rms": 0.0,
        },
        "ordering_fixture": {
            "fixture_id": "negative_shift_flip",
            "paper_first": fixture_orders["negative_shift_flip"][0][0],
            "post_relu_first": fixture_orders["negative_shift_flip"][1][0],
            "equal_payoff": True,
        },
        "depth_fixture": {
            "function": "ReLU composed with itself depth times",
            "function_is_identical_for_depths": [1, 2, 3, 4],
            "proxy_multiplier_at_depth_4": 4.0,
        },
        "row_counts": {
            "source_map": len(source_rows),
            "moment_validation": len(moment_rows),
            "shift_sweep": len(shift_rows),
            "ordering_fixtures": len(ordering_rows),
            "depth_fixture": len(depth_rows),
            "graph_boundary": len(graph_rows),
            "eligibility": len(eligibility_rows),
        },
    }
    if not passed:
        failed_checks = [name for name, row in check_rows.items() if not row["passed"]]
        failed_rows = [
            row.get("case_id", row.get("fixture_id", row.get("step_id", "unknown")))
            for row in all_rows if not row["passed"]
        ]
        raise AuditFailure(f"audit checks failed: checks={failed_checks}, rows={failed_rows}")
    return {
        "summary": summary,
        "source_map.csv": source_rows,
        "moment_validation.csv": moment_rows,
        "shift_sweep.csv": shift_rows,
        "ordering_fixtures.csv": ordering_rows,
        "depth_fixture.csv": depth_rows,
        "graph_boundary.csv": graph_rows,
        "eligibility.csv": eligibility_rows,
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
        value = float(record.get("value"))
        if not math.isfinite(value) or not spec["minimum"] <= value <= spec["maximum"]:
            raise AuditFailure(f"check is outside source range: {name}")
    if any(not math.isfinite(number) for number in _walk_numbers(summary)):
        raise AuditFailure("summary contains a non-finite number")


def _write_json(path: Path, payload: Any) -> None:
    if any(not math.isfinite(number) for number in _walk_numbers(payload)):
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
        "seeds": {"used": True, "values": [MONTE_CARLO_SEED]},
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
            "entrypoint": "experiments/audits/part08_block_eviction/run.py",
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
        f"PASS: {summary['audit_id']} completed "
        f"{summary['passing_check_count']}/{summary['check_count']} checks"
    )


if __name__ == "__main__":
    try:
        main()
    except (AuditFailure, OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
