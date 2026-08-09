#!/usr/bin/env python3
"""Fail-closed audit of HOPE v1 cache and optimizer complexity claims."""

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
AUDIT_ID = "part09_constant_time_audit"
AUDIT_SCHEMA_VERSION = 1
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
PAPER_BYTES = 1_089_944
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_09" / "manual"
WIDTHS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
INCOMING_DIM = 64
OUTGOING_DIM = 64
SCALARS_PER_PAIR = 3
FLOAT_BYTES = 8
REMOVED_CHANNEL = 1

CLAIM_IDS = (
    "P09-SELECTION-AND-PAYOFF",
    "P09-PER-CANDIDATE-CACHE",
    "P09-GLOBAL-SCAN-SCOPE",
    "P09-TRAJECTORY-CUBIC",
    "P09-CACHE-MEMORY",
    "P09-ADJACENT-INVALIDATION",
    "P09-REFRESH-REPAIR",
    "P09-STATIC-PAYOFF-FLIP",
    "P09-END-TO-END-BOUNDARY",
)
CLAIM_CLASSES = {
    "P09-SELECTION-AND-PAYOFF": "paper_claim",
    "P09-PER-CANDIDATE-CACHE": "paper_claim",
    "P09-GLOBAL-SCAN-SCOPE": "verified_fact",
    "P09-TRAJECTORY-CUBIC": "verified_fact",
    "P09-CACHE-MEMORY": "verified_fact",
    "P09-ADJACENT-INVALIDATION": "counterexample",
    "P09-REFRESH-REPAIR": "proposed_repair",
    "P09-STATIC-PAYOFF-FLIP": "counterexample",
    "P09-END-TO-END-BOUNDARY": "independent_inference",
}

METHOD_ORIGINS = {
    "cached_scalar_lookup": {
        "evidence_label": "paper_claim",
        "claim_id": "P09-PER-CANDIDATE-CACHE",
        "origin": "HOPE v1 Section 10, Footnote 8, Appendix B.4, and Algorithm 1 line 13",
        "scope": "one already enumerated candidate with cached geometry and live residual capacity",
    },
    "trajectory_count": {
        "evidence_label": "verified_fact",
        "claim_id": "P09-TRAJECTORY-CUBIC",
        "origin": "exact finite sums over a declared one-layer merge trajectory",
        "scope": "one candidate removed per action from width N through width 1",
    },
    "dependency_fixture": {
        "evidence_label": "counterexample",
        "claim_id": "P09-ADJACENT-INVALIDATION",
        "origin": "independent deterministic 1x1 Conv-BN-ReLU-Conv channel-surgery fixture",
        "scope": "structural invalidation, not trained-network prevalence or accuracy",
    },
    "dependency_refresh": {
        "evidence_label": "proposed_repair",
        "claim_id": "P09-REFRESH-REPAIR",
        "origin": "independent dependency-aware recomputation policy",
        "scope": "the declared feed-forward fixture with fully resolved consumers",
    },
    "live_payoff_comparison": {
        "evidence_label": "counterexample",
        "claim_id": "P09-STATIC-PAYOFF-FLIP",
        "origin": "independent exact parameter-count fixtures",
        "scope": "refutes a general uniform static-to-live scaling factor",
    },
}

REPRODUCTION_SCOPE = {
    "reproduction_scope": "independent_complexity_and_dependency_fixture_audit",
    "paper_experiment_reproduced": False,
    "checkpoint_or_dataset_used": False,
    "wall_clock_benchmark_used": False,
    "random_sampling_used": False,
    "includes": [
        "paper source map for Sections 9-10, Eqs. 21-23 and 36-42, and Algorithms 1-2",
        "exact candidate, scan, refresh, and cache-memory counts",
        "deterministic operation-count comparison of full recomputation and decoupled caching",
        "deterministic 1x1 convolution dependency and stale-ranking fixture",
        "dependency-aware refresh comparison against full recomputation",
        "static-versus-live payoff ranking reversal and controls",
    ],
    "excludes": [
        "trained-network timing, scaling prevalence, accuracy, or loss",
        "GPU or accelerator wall-clock benchmarks",
        "Python object, allocator, priority-queue, or index overhead in cache byte counts",
        "a complete corrected optimizer implementation",
        "Parts 10 through 12",
    ],
    "claim_boundary": (
        "The audit counts the declared workload and proves invalidation on fixed fixtures. "
        "It does not claim a hardware runtime or frequency on trained models."
    ),
}

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part09_constant_time/README.md",
    "experiments/audits/part09_constant_time/run.py",
    "experiments/audits/part09_constant_time/test_run.py",
)

SOURCE_MAP_FIELDS = (
    "step_id", "sequence", "paper_anchor", "object", "formula_or_action",
    "result", "evidence_label", "claim_id", "passed",
)
CANDIDATE_FIELDS = (
    "width", "prune_candidates", "merge_candidates", "total_candidates",
    "initial_pair_cache_entries", "trajectory_prune_evaluations",
    "trajectory_merge_evaluations", "trajectory_total_evaluations",
    "trajectory_parent_pair_refreshes", "merge_over_width_cubed",
    "evidence_label", "claim_id", "passed",
)
OPERATION_FIELDS = (
    "width", "incoming_dim", "outgoing_dim", "geometry_dim",
    "full_recompute_geometry_coordinate_visits", "full_scan_scalar_evaluations",
    "full_total_work_units", "cache_initial_geometry_coordinate_visits",
    "cache_parent_refresh_coordinate_visits", "cache_scan_scalar_evaluations",
    "cache_total_work_units", "full_over_cache_work_factor",
    "wall_clock_claim", "evidence_label", "claim_id", "passed",
)
MEMORY_FIELDS = (
    "width", "pair_entries", "cached_scalars_per_pair", "float_bytes",
    "raw_scalar_bytes", "raw_scalar_mib", "includes_index_overhead",
    "evidence_label", "claim_id", "passed",
)
DEPENDENCY_FIELDS = (
    "case_id", "location", "quantity", "index", "before", "after",
    "absolute_change", "paper_local_update_refreshes", "dependency_refreshes",
    "evidence_label", "claim_id", "passed",
)
REFRESH_FIELDS = (
    "pair", "before_correlation", "after_full_correlation",
    "after_paper_local_correlation", "after_dependency_refresh_correlation",
    "before_kernel", "after_full_kernel", "after_paper_local_kernel",
    "after_dependency_refresh_kernel", "before_rank", "after_full_rank",
    "after_paper_local_rank", "after_dependency_refresh_rank",
    "evidence_label", "claim_id", "passed",
)
PAYOFF_FIELDS = (
    "fixture_id", "candidate", "distortion", "initial_fan_in",
    "initial_fan_out", "live_fan_in", "live_fan_out", "bn_parameters",
    "static_payoff", "live_payoff", "static_over_live_alpha", "static_dr",
    "live_dr", "static_rank", "live_rank", "expected_flip",
    "evidence_label", "claim_id", "passed",
)

CSV_CONTRACTS = {
    "source_map.csv": SOURCE_MAP_FIELDS,
    "candidate_counts.csv": CANDIDATE_FIELDS,
    "operation_counts.csv": OPERATION_FIELDS,
    "cache_memory.csv": MEMORY_FIELDS,
    "dependency_invalidation.csv": DEPENDENCY_FIELDS,
    "refresh_comparison.csv": REFRESH_FIELDS,
    "payoff_rankings.csv": PAYOFF_FIELDS,
}
OUTPUT_FILENAMES = ("summary.json", *CSV_CONTRACTS.keys())

CHECK_SPECS = {
    "nonfinite_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "row_failure_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "evidence_class_mismatch_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "candidate_formula_error_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "width4096_pair_entries": {"minimum": 8_386_560.0, "maximum": 8_386_560.0, "units": "pairs"},
    "width4096_trajectory_merge_evaluations": {"minimum": 11_453_245_440.0, "maximum": 11_453_245_440.0, "units": "evaluations"},
    "largest_width_empirical_exponent": {"minimum": 2.995, "maximum": 3.005, "units": "log2_ratio"},
    "operation_formula_error_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "largest_width_cache_reduction_factor": {"minimum": 108.0, "maximum": 109.0, "units": "work_ratio"},
    "memory_formula_error_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "width4096_raw_cache_mib": {"minimum": 191.9, "maximum": 192.1, "units": "MiB"},
    "predecessor_capacity_change_count": {"minimum": 3.0, "maximum": 3.0, "units": "count"},
    "successor_variance_change_count": {"minimum": 3.0, "maximum": 3.0, "units": "count"},
    "successor_pair_change_count": {"minimum": 3.0, "maximum": 3.0, "units": "count"},
    "stale_pair_value_count": {"minimum": 3.0, "maximum": 3.0, "units": "count"},
    "stale_best_pair_mismatch_count": {"minimum": 1.0, "maximum": 1.0, "units": "count"},
    "dependency_refresh_max_error": {"minimum": 0.0, "maximum": 0.0, "units": "absolute"},
    "payoff_expected_flip_count": {"minimum": 1.0, "maximum": 1.0, "units": "count"},
    "payoff_unexpected_flip_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "ranking_fixture_alpha_spread": {"minimum": 2.0, "maximum": 2.3, "units": "ratio"},
}


class AuditFailure(RuntimeError):
    """Raised when a declared audit invariant fails."""


def choose2(width: int) -> int:
    return width * (width - 1) // 2


def choose3(value: int) -> int:
    return value * (value - 1) * (value - 2) // 6


def merge_trajectory_evaluations(width: int) -> int:
    return choose3(width + 1)


def prune_trajectory_evaluations(width: int) -> int:
    return width * (width + 1) // 2 - 1


def total_trajectory_evaluations(width: int) -> int:
    return choose3(width + 2) - 1


def parent_pair_refreshes(width: int) -> int:
    return max(0, (width - 2) * (width - 1) // 2)


def relu_normalized_kernel(correlation: float) -> float:
    rho = min(1.0, max(-1.0, float(correlation)))
    return (
        math.sqrt(max(0.0, 1.0 - rho * rho))
        + (math.pi - math.acos(rho)) * rho
    ) / math.pi


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        raise AuditFailure("cosine fixture contains a zero vector")
    return float(np.dot(first, second) / denominator)


def build_config(*, full: bool) -> dict[str, Any]:
    return {
        "float_dtype": "float64",
        "full_mode": bool(full),
        "random_sampling": False,
        "wall_clock_benchmark": False,
        "widths": list(WIDTHS),
        "operation_count_dimensions": {
            "incoming": INCOMING_DIM,
            "outgoing": OUTGOING_DIM,
        },
        "cache_scalar_model": {
            "scalars_per_pair": SCALARS_PER_PAIR,
            "bytes_per_scalar": FLOAT_BYTES,
            "included": ["a", "b", "warped_correlation"],
            "excluded": ["pair_indices", "container_overhead", "allocator_overhead"],
        },
        "trajectory": "one layer, one merge per action, live widths N through 2",
        "removed_channel": REMOVED_CHANNEL,
    }


def _source_map_rows() -> list[dict[str, Any]]:
    specifications = (
        (
            "planning_problem", "Section 9 and Eq. 21, PDF page 18", "global plan",
            "binary action selection under a parameter budget",
            "the paper identifies the full discrete plan as NP-hard",
            "paper_claim", "P09-SELECTION-AND-PAYOFF",
        ),
        (
            "receding_horizon", "Eqs. 22-23, PDF page 19", "one-step selection",
            "arg min over every currently admissible action",
            "one action is executed before the next optimization step",
            "paper_claim", "P09-SELECTION-AND-PAYOFF",
        ),
        (
            "section10_scan", "Section 10, PDF pages 19-20", "global scan",
            "enumerate all live prune, merge, and eviction actions",
            "the main text states O(L N^2) work per global scan",
            "verified_fact", "P09-GLOBAL-SCAN-SCOPE",
        ),
        (
            "footnote8_lookup", "Footnote 8, PDF page 19", "candidate lookup",
            "compute one candidate DR from cached local variables",
            "the explicitly scoped single-candidate arithmetic is O(1)",
            "paper_claim", "P09-PER-CANDIDATE-CACHE",
        ),
        (
            "static_payoff", "Appendix B.2 and Eq. 36, PDF page 38", "parameter payoff",
            "count initial incoming, outgoing, and auxiliary parameters",
            "the denominator is frozen on the initial architecture",
            "paper_claim", "P09-SELECTION-AND-PAYOFF",
        ),
        (
            "uniform_alpha", "Appendix B.3 and Eqs. 38-39, PDF pages 39-40", "ranking claim",
            "Delta P init approximately equals alpha times Delta P live",
            "ranking preservation requires the same alpha for every candidate",
            "paper_claim", "P09-SELECTION-AND-PAYOFF",
        ),
        (
            "appendix_global_scan", "Appendix B.4, PDF page 40", "global scan",
            "scan every remaining pair while recomputing scalar costs",
            "the appendix calls the global scan O(1), conflicting with Section 10",
            "verified_fact", "P09-GLOBAL-SCAN-SCOPE",
        ),
        (
            "algorithm_scan", "Algorithm 1 lines 8-19, PDF page 68", "nested enumeration",
            "loop over each active layer and each active candidate",
            "the pseudocode itself performs a global candidate scan",
            "verified_fact", "P09-GLOBAL-SCAN-SCOPE",
        ),
        (
            "algorithm_invalidation", "Algorithm 1 line 30, PDF page 68", "cache update",
            "zero downstream recalculation under a static payoff approximation",
            "static parameter counts do not freeze function-dependent adjacent tensors",
            "counterexample", "P09-ADJACENT-INVALIDATION",
        ),
        (
            "physical_surgery", "Algorithm 2 lines 2-14, PDF page 69", "physical reduction",
            "remove a neuron or child vessel after pruning or merging",
            "channel removal changes shared predecessor and successor tensor slices",
            "counterexample", "P09-ADJACENT-INVALIDATION",
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


def _candidate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width in WIDTHS:
        prunes = width
        merges = choose2(width)
        trajectory_prunes = prune_trajectory_evaluations(width)
        trajectory_merges = merge_trajectory_evaluations(width)
        trajectory_total = total_trajectory_evaluations(width)
        refreshes = parent_pair_refreshes(width)
        rows.append({
            "width": width,
            "prune_candidates": prunes,
            "merge_candidates": merges,
            "total_candidates": prunes + merges,
            "initial_pair_cache_entries": merges,
            "trajectory_prune_evaluations": trajectory_prunes,
            "trajectory_merge_evaluations": trajectory_merges,
            "trajectory_total_evaluations": trajectory_total,
            "trajectory_parent_pair_refreshes": refreshes,
            "merge_over_width_cubed": trajectory_merges / width ** 3,
            "evidence_label": "verified_fact",
            "claim_id": "P09-TRAJECTORY-CUBIC",
            "passed": (
                merges == width * (width - 1) // 2
                and trajectory_merges == sum(choose2(value) for value in range(2, width + 1))
                and trajectory_prunes == sum(range(2, width + 1))
                and trajectory_total == trajectory_merges + trajectory_prunes
                and refreshes == sum(max(0, value - 2) for value in range(2, width + 1))
            ),
        })
    return rows


def _operation_rows() -> list[dict[str, Any]]:
    geometry_dim = INCOMING_DIM + OUTGOING_DIM
    rows: list[dict[str, Any]] = []
    for width in WIDTHS:
        scans = merge_trajectory_evaluations(width)
        full_geometry = geometry_dim * scans
        full_total = full_geometry + scans
        cache_initial = geometry_dim * choose2(width)
        cache_refresh = geometry_dim * parent_pair_refreshes(width)
        cache_total = cache_initial + cache_refresh + scans
        rows.append({
            "width": width,
            "incoming_dim": INCOMING_DIM,
            "outgoing_dim": OUTGOING_DIM,
            "geometry_dim": geometry_dim,
            "full_recompute_geometry_coordinate_visits": full_geometry,
            "full_scan_scalar_evaluations": scans,
            "full_total_work_units": full_total,
            "cache_initial_geometry_coordinate_visits": cache_initial,
            "cache_parent_refresh_coordinate_visits": cache_refresh,
            "cache_scan_scalar_evaluations": scans,
            "cache_total_work_units": cache_total,
            "full_over_cache_work_factor": full_total / cache_total,
            "wall_clock_claim": False,
            "evidence_label": "independent_inference",
            "claim_id": "P09-END-TO-END-BOUNDARY",
            "passed": (
                full_geometry == geometry_dim * sum(choose2(value) for value in range(2, width + 1))
                and cache_initial == geometry_dim * choose2(width)
                and cache_refresh == geometry_dim * sum(
                    max(0, value - 2) for value in range(2, width + 1)
                )
                and full_total >= cache_total > 0
            ),
        })
    return rows


def _memory_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width in WIDTHS:
        pairs = choose2(width)
        raw_bytes = pairs * SCALARS_PER_PAIR * FLOAT_BYTES
        rows.append({
            "width": width,
            "pair_entries": pairs,
            "cached_scalars_per_pair": SCALARS_PER_PAIR,
            "float_bytes": FLOAT_BYTES,
            "raw_scalar_bytes": raw_bytes,
            "raw_scalar_mib": raw_bytes / (1024.0 ** 2),
            "includes_index_overhead": False,
            "evidence_label": "verified_fact",
            "claim_id": "P09-CACHE-MEMORY",
            "passed": raw_bytes == choose2(width) * SCALARS_PER_PAIR * FLOAT_BYTES,
        })
    return rows


def _fixture_matrices() -> tuple[np.ndarray, np.ndarray]:
    central_incoming = np.asarray(
        [[1.0, 2.0, -1.0], [4.0, -2.0, 1.0], [0.0, 1.0, 3.0], [2.0, -1.0, 2.0]],
        dtype=np.float64,
    )
    successor_incoming = np.asarray(
        [[-5.0, 4.0, -1.0, -1.0], [2.0, -4.0, 0.0, -3.0], [-3.0, -3.0, 2.0, -5.0]],
        dtype=np.float64,
    )
    return central_incoming, successor_incoming


def _dependency_rows() -> list[dict[str, Any]]:
    central, successor = _fixture_matrices()
    reduced_central = np.delete(central, REMOVED_CHANNEL, axis=0)
    reduced_successor = np.delete(successor, REMOVED_CHANNEL, axis=1)
    rows: list[dict[str, Any]] = []

    predecessor_before = np.linalg.norm(central, axis=0) / math.sqrt(2.0)
    predecessor_after = np.linalg.norm(reduced_central, axis=0) / math.sqrt(2.0)
    for index, (before, after) in enumerate(zip(predecessor_before, predecessor_after)):
        rows.append({
            "case_id": "remove_central_channel_1",
            "location": "predecessor",
            "quantity": "functional_capacity_from_outgoing_norm",
            "index": index,
            "before": float(before),
            "after": float(after),
            "absolute_change": abs(float(after - before)),
            "paper_local_update_refreshes": False,
            "dependency_refreshes": True,
            "evidence_label": "counterexample",
            "claim_id": "P09-ADJACENT-INVALIDATION",
            "passed": abs(float(after - before)) > 1e-12,
        })

    variance_before = np.sum(successor * successor, axis=1)
    variance_after = np.sum(reduced_successor * reduced_successor, axis=1)
    for index, (before, after) in enumerate(zip(variance_before, variance_after)):
        rows.append({
            "case_id": "remove_central_channel_1",
            "location": "successor",
            "quantity": "preactivation_variance_under_identity_covariance",
            "index": index,
            "before": float(before),
            "after": float(after),
            "absolute_change": abs(float(after - before)),
            "paper_local_update_refreshes": False,
            "dependency_refreshes": True,
            "evidence_label": "counterexample",
            "claim_id": "P09-ADJACENT-INVALIDATION",
            "passed": abs(float(after - before)) > 1e-12,
        })

    for first in range(successor.shape[0]):
        for second in range(first + 1, successor.shape[0]):
            before = cosine(successor[first], successor[second])
            after = cosine(reduced_successor[first], reduced_successor[second])
            rows.append({
                "case_id": "remove_central_channel_1",
                "location": "successor_pair",
                "quantity": "incoming_cosine",
                "index": f"{first}-{second}",
                "before": before,
                "after": after,
                "absolute_change": abs(after - before),
                "paper_local_update_refreshes": False,
                "dependency_refreshes": True,
                "evidence_label": "counterexample",
                "claim_id": "P09-ADJACENT-INVALIDATION",
                "passed": abs(after - before) > 1e-12,
            })
    return rows


def _rank(values: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: (-values[key], key))
    return {key: rank for rank, key in enumerate(ordered, start=1)}


def _refresh_rows() -> list[dict[str, Any]]:
    _, successor = _fixture_matrices()
    reduced = np.delete(successor, REMOVED_CHANNEL, axis=1)
    before_corr: dict[str, float] = {}
    after_corr: dict[str, float] = {}
    for first in range(successor.shape[0]):
        for second in range(first + 1, successor.shape[0]):
            pair = f"{first}-{second}"
            before_corr[pair] = cosine(successor[first], successor[second])
            after_corr[pair] = cosine(reduced[first], reduced[second])
    before_kernel = {pair: relu_normalized_kernel(value) for pair, value in before_corr.items()}
    after_kernel = {pair: relu_normalized_kernel(value) for pair, value in after_corr.items()}
    before_rank = _rank(before_kernel)
    after_rank = _rank(after_kernel)
    rows: list[dict[str, Any]] = []
    for pair in sorted(before_corr):
        rows.append({
            "pair": pair,
            "before_correlation": before_corr[pair],
            "after_full_correlation": after_corr[pair],
            "after_paper_local_correlation": before_corr[pair],
            "after_dependency_refresh_correlation": after_corr[pair],
            "before_kernel": before_kernel[pair],
            "after_full_kernel": after_kernel[pair],
            "after_paper_local_kernel": before_kernel[pair],
            "after_dependency_refresh_kernel": after_kernel[pair],
            "before_rank": before_rank[pair],
            "after_full_rank": after_rank[pair],
            "after_paper_local_rank": before_rank[pair],
            "after_dependency_refresh_rank": after_rank[pair],
            "evidence_label": "proposed_repair",
            "claim_id": "P09-REFRESH-REPAIR",
            "passed": (
                after_corr[pair] != before_corr[pair]
                and after_kernel[pair] != before_kernel[pair]
                and after_rank[pair] == _rank(after_kernel)[pair]
            ),
        })
    return rows


def _payoff_rows() -> list[dict[str, Any]]:
    fixtures = (
        (
            "nonuniform_alpha_flip", True,
            (
                ("A", 1.0, 8, 8, 1, 1, 4),
                ("B", 1.0, 6, 6, 5, 5, 4),
            ),
        ),
        (
            "uniform_alpha_control", False,
            (
                ("A", 1.0, 8, 8, 3, 3, 4),
                ("B", 1.0, 6, 6, 2, 2, 4),
            ),
        ),
        (
            "distortion_control", False,
            (
                ("A", 0.8, 6, 6, 5, 5, 4),
                ("B", 1.2, 6, 6, 5, 5, 4),
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    for fixture_id, expected_flip, candidates in fixtures:
        working: list[dict[str, Any]] = []
        for candidate, distortion, initial_in, initial_out, live_in, live_out, bn in candidates:
            static_payoff = initial_in + initial_out + bn
            live_payoff = live_in + live_out + bn
            working.append({
                "fixture_id": fixture_id,
                "candidate": candidate,
                "distortion": distortion,
                "initial_fan_in": initial_in,
                "initial_fan_out": initial_out,
                "live_fan_in": live_in,
                "live_fan_out": live_out,
                "bn_parameters": bn,
                "static_payoff": static_payoff,
                "live_payoff": live_payoff,
                "static_over_live_alpha": static_payoff / live_payoff,
                "static_dr": distortion / static_payoff,
                "live_dr": distortion / live_payoff,
            })
        static_order = sorted(working, key=lambda row: (row["static_dr"], row["candidate"]))
        live_order = sorted(working, key=lambda row: (row["live_dr"], row["candidate"]))
        static_rank = {row["candidate"]: rank for rank, row in enumerate(static_order, start=1)}
        live_rank = {row["candidate"]: rank for rank, row in enumerate(live_order, start=1)}
        observed_flip = [row["candidate"] for row in static_order] != [
            row["candidate"] for row in live_order
        ]
        for row in working:
            row.update({
                "static_rank": static_rank[row["candidate"]],
                "live_rank": live_rank[row["candidate"]],
                "expected_flip": expected_flip,
                "evidence_label": "counterexample",
                "claim_id": "P09-STATIC-PAYOFF-FLIP",
                "passed": observed_flip == expected_flip,
            })
            rows.append(row)
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


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def build_audit(*, full: bool = False) -> dict[str, Any]:
    config = build_config(full=full)
    source_rows = _source_map_rows()
    candidate_rows = _candidate_rows()
    operation_rows = _operation_rows()
    memory_rows = _memory_rows()
    dependency_rows = _dependency_rows()
    refresh_rows = _refresh_rows()
    payoff_rows = _payoff_rows()
    row_groups = (
        source_rows, candidate_rows, operation_rows, memory_rows,
        dependency_rows, refresh_rows, payoff_rows,
    )
    all_rows = [row for group in row_groups for row in group]
    evidence_mismatches = sum(
        row["evidence_label"] != CLAIM_CLASSES[row["claim_id"]] for row in all_rows
    ) + sum(
        record["evidence_label"] != CLAIM_CLASSES[record["claim_id"]]
        for record in METHOD_ORIGINS.values()
    )

    by_width = {row["width"]: row for row in candidate_rows}
    operation_by_width = {row["width"]: row for row in operation_rows}
    memory_by_width = {row["width"]: row for row in memory_rows}
    exponent = math.log2(
        by_width[4096]["trajectory_merge_evaluations"]
        / by_width[2048]["trajectory_merge_evaluations"]
    )
    predecessor_changes = [
        row for row in dependency_rows
        if row["location"] == "predecessor" and row["absolute_change"] > 1e-12
    ]
    successor_variance_changes = [
        row for row in dependency_rows
        if row["location"] == "successor" and row["absolute_change"] > 1e-12
    ]
    successor_pair_changes = [
        row for row in dependency_rows
        if row["location"] == "successor_pair" and row["absolute_change"] > 1e-12
    ]
    before_best = next(row["pair"] for row in refresh_rows if row["before_rank"] == 1)
    full_best = next(row["pair"] for row in refresh_rows if row["after_full_rank"] == 1)
    stale_best = next(row["pair"] for row in refresh_rows if row["after_paper_local_rank"] == 1)
    dependency_best = next(
        row["pair"] for row in refresh_rows if row["after_dependency_refresh_rank"] == 1
    )
    refresh_error = max(
        abs(row["after_dependency_refresh_kernel"] - row["after_full_kernel"])
        for row in refresh_rows
    )
    payoff_orders: dict[str, tuple[tuple[str, ...], tuple[str, ...], bool]] = {}
    for fixture_id in sorted({row["fixture_id"] for row in payoff_rows}):
        selected = [row for row in payoff_rows if row["fixture_id"] == fixture_id]
        static = tuple(
            row["candidate"] for row in sorted(selected, key=lambda row: row["static_rank"])
        )
        live = tuple(
            row["candidate"] for row in sorted(selected, key=lambda row: row["live_rank"])
        )
        payoff_orders[fixture_id] = (static, live, bool(selected[0]["expected_flip"]))
    reversal_rows = [
        row for row in payoff_rows if row["fixture_id"] == "nonuniform_alpha_flip"
    ]
    alpha_spread = max(row["static_over_live_alpha"] for row in reversal_rows) - min(
        row["static_over_live_alpha"] for row in reversal_rows
    )

    checks = {
        "nonfinite_count": float(sum(
            not math.isfinite(number) for number in _walk_numbers(all_rows)
        )),
        "row_failure_count": float(sum(not row["passed"] for row in all_rows)),
        "evidence_class_mismatch_count": float(evidence_mismatches),
        "candidate_formula_error_count": float(sum(not row["passed"] for row in candidate_rows)),
        "width4096_pair_entries": float(by_width[4096]["merge_candidates"]),
        "width4096_trajectory_merge_evaluations": float(
            by_width[4096]["trajectory_merge_evaluations"]
        ),
        "largest_width_empirical_exponent": exponent,
        "operation_formula_error_count": float(sum(not row["passed"] for row in operation_rows)),
        "largest_width_cache_reduction_factor": operation_by_width[4096][
            "full_over_cache_work_factor"
        ],
        "memory_formula_error_count": float(sum(not row["passed"] for row in memory_rows)),
        "width4096_raw_cache_mib": memory_by_width[4096]["raw_scalar_mib"],
        "predecessor_capacity_change_count": float(len(predecessor_changes)),
        "successor_variance_change_count": float(len(successor_variance_changes)),
        "successor_pair_change_count": float(len(successor_pair_changes)),
        "stale_pair_value_count": float(sum(
            row["after_paper_local_kernel"] != row["after_full_kernel"] for row in refresh_rows
        )),
        "stale_best_pair_mismatch_count": float(stale_best != full_best),
        "dependency_refresh_max_error": refresh_error,
        "payoff_expected_flip_count": float(sum(
            static != live and expected for static, live, expected in payoff_orders.values()
        )),
        "payoff_unexpected_flip_count": float(sum(
            static != live and not expected for static, live, expected in payoff_orders.values()
        )),
        "ranking_fixture_alpha_spread": alpha_spread,
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
                "Sections 9-10 and Eqs. 21-23, PDF pages 18-20",
                "Appendix B.2-B.4 and Eqs. 36-40, PDF pages 38-40",
                "Algorithms 1-2, PDF pages 68-69",
            ],
        },
        "config": config,
        "config_sha256": _config_hash(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": REPRODUCTION_SCOPE,
        "check_count": len(check_rows),
        "passing_check_count": sum(row["passed"] for row in check_rows.values()),
        "checks": check_rows,
        "complexity_boundary": {
            "single_cached_candidate": "O(1) scalar arithmetic after geometry initialization",
            "one_layer_scan": "Theta(m^2) merge candidates at live width m",
            "complete_one_layer_trajectory": "Theta(N^3) cached merge-score evaluations",
            "pair_initialization": "Theta(N^2(d+c)) coordinate work",
            "pair_cache": "Theta(N^2) entries",
            "wall_clock_claim": False,
        },
        "largest_fixture": {
            "width": 4096,
            "pair_entries": by_width[4096]["merge_candidates"],
            "trajectory_merge_evaluations": by_width[4096]["trajectory_merge_evaluations"],
            "raw_three_float64_cache_mib": memory_by_width[4096]["raw_scalar_mib"],
        },
        "refresh_fixture": {
            "before_best_pair": before_best,
            "paper_local_after_best_pair": stale_best,
            "full_after_best_pair": full_best,
            "dependency_refresh_after_best_pair": dependency_best,
        },
        "payoff_fixture": {
            "fixture_id": "nonuniform_alpha_flip",
            "static_first": payoff_orders["nonuniform_alpha_flip"][0][0],
            "live_first": payoff_orders["nonuniform_alpha_flip"][1][0],
            "alpha_spread": alpha_spread,
        },
        "row_counts": {
            "source_map": len(source_rows),
            "candidate_counts": len(candidate_rows),
            "operation_counts": len(operation_rows),
            "cache_memory": len(memory_rows),
            "dependency_invalidation": len(dependency_rows),
            "refresh_comparison": len(refresh_rows),
            "payoff_rankings": len(payoff_rows),
        },
    }
    if not passed:
        failed_checks = [name for name, row in check_rows.items() if not row["passed"]]
        failed_rows = [
            row.get("case_id", row.get("fixture_id", row.get("step_id", row.get("width", "unknown"))))
            for row in all_rows if not row["passed"]
        ]
        raise AuditFailure(f"audit checks failed: checks={failed_checks}, rows={failed_rows}")
    return {
        "summary": summary,
        "source_map.csv": source_rows,
        "candidate_counts.csv": candidate_rows,
        "operation_counts.csv": operation_rows,
        "cache_memory.csv": memory_rows,
        "dependency_invalidation.csv": dependency_rows,
        "refresh_comparison.csv": refresh_rows,
        "payoff_rankings.csv": payoff_rows,
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
        "seeds": {"used": False, "values": []},
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
            "entrypoint": "experiments/audits/part09_constant_time/run.py",
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
