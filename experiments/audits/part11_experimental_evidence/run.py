#!/usr/bin/env python3
"""Fail-closed deterministic audit of HOPE v1 experimental evidence."""

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
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path
from typing import Any, Mapping, Sequence


getcontext().prec = 40

ROOT = Path(__file__).resolve().parents[3]
AUDIT_ID = "part11_experimental_evidence_audit"
AUDIT_SCHEMA_VERSION = 1
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
PAPER_BYTES = 1_089_944
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_11" / "manual"

# Decoded pixel hashes bind the digitization to the two embedded page-20 rasters,
# independently of PNG container metadata.
COMPRESSION_RASTER = {
    "pdf_page": 20,
    "pdf_object": 1088,
    "width": 2929,
    "height": 1724,
    "rgb_pixel_sha256": "615af1aae17cc4edd4c9b258f6b6baa368f6318f8c4306a2fe039f6db17afe78",
    "smask_pixel_sha256": "f8a655b55823b7c049a97d91589dde7a369c0d4d186448ecf14391e54ffc22e5",
    "x_tick_left_pixel": 375,
    "x_tick_left_value": 0.2,
    "x_tick_right_pixel": 2777,
    "x_tick_right_value": 1.0,
    "y_tick_bottom_pixel": 1466,
    "y_tick_bottom_value": 0.0,
    "y_tick_top_pixel": 122,
    "y_tick_top_value": 0.7,
}

TABLE2_RECORDS = (
    ("DEFT", "89.79", "0.84", "52.14", "5.29", "65.82", "3.96"),
    ("Head-Only", "36.11", "2.79", "63.13", "4.62", "45.79", "2.05"),
    ("Full FT", "94.09", "0.28", "7.52", "1.63", "13.88", "2.84"),
    ("EWC", "93.94", "0.22", "6.74", "1.74", "12.54", "2.99"),
    ("PEFT", "81.91", "0.49", "5.44", "0.98", "10.18", "1.63"),
)

TABLE2_EXPECTED_SIGNATURES = {
    "DEFT": (89.79, 0.84, 52.14, 5.29, 65.82, 3.96),
    "Head-Only": (36.11, 2.79, 63.13, 4.62, 45.79, 2.05),
    "Full FT": (94.09, 0.28, 7.52, 1.63, 13.88, 2.84),
    "EWC": (93.94, 0.22, 6.74, 1.74, 12.54, 2.99),
    "PEFT": (81.91, 0.49, 5.44, 0.98, 10.18, 1.63),
}

AGGREGATION_FIXTURE = {
    "balanced": (("task_1", 50.0, 50.0), ("task_2", 50.0, 50.0)),
    "crossed": (("task_1", 90.0, 10.0), ("task_2", 10.0, 90.0)),
}

UNCERTAINTY_INTERPRETATIONS = (
    "standard_deviation_over_20_scenarios",
    "standard_deviation_over_4_trial_means",
    "standard_error_over_20_scenarios",
    "standard_error_over_4_trial_means",
    "confidence_interval_half_width",
)

CHECKPOINT_FIXTURE = (
    (1, 80.0, 80.0),
    (2, 90.0, 10.0),
)

DENSITY_FIXTURE = {
    "input_width": 100,
    "hidden_1_width": 2,
    "hidden_2_width": 4,
    "output_width": 10,
    "scope": "weight-only dense chain without biases or BatchNorm parameters",
}

# Each row records the union of saturated series-colored pixels in a declared
# x-window around a calibrated density. A wide or disconnected y support is
# retained as an interval instead of being collapsed to an invented midpoint.
DIGITIZATION_PIXELS = (
    # density, method, x_low, x_high, y_low, y_high, vertical_trace
    (0.3, "HOPE", 670, 680, 1364, 1381, False),
    (0.3, "L1-In", 670, 680, 1458, 1461, False),
    (0.3, "L1-Joint", 670, 680, 1440, 1446, False),
    (0.3, "BN-Scale", 670, 680, 1446, 1457, False),
    (0.4, "HOPE", 971, 981, 1107, 1134, False),
    (0.4, "L1-In", 971, 981, 1395, 1415, False),
    (0.4, "L1-Joint", 971, 981, 1335, 1354, False),
    (0.4, "BN-Scale", 971, 981, 1376, 1395, False),
    (0.5, "HOPE", 1271, 1281, 770, 780, False),
    (0.5, "L1-In", 1271, 1281, 995, 1008, False),
    (0.5, "L1-Joint", 1271, 1281, 847, 1347, True),
    (0.5, "BN-Scale", 1271, 1281, 1009, 1055, False),
    (0.6, "HOPE", 1571, 1581, 629, 642, False),
    (0.6, "L1-In", 1571, 1581, 1127, 1147, False),
    (0.6, "L1-Joint", 1571, 1581, 1370, 1405, False),
    (0.6, "BN-Scale", 1571, 1581, 1059, 1196, True),
    (0.7, "HOPE", 1871, 1881, 509, 522, False),
    (0.7, "L1-In", 1871, 1881, 992, 1024, False),
    (0.7, "L1-Joint", 1871, 1881, 1130, 1154, False),
    (0.7, "BN-Scale", 1871, 1881, 893, 931, False),
    (0.8, "HOPE", 2171, 2181, 347, 362, False),
    (0.8, "L1-In", 2171, 2181, 664, 685, False),
    (0.8, "L1-Joint", 2171, 2181, 671, 711, False),
    (0.8, "BN-Scale", 2171, 2181, 582, 601, False),
    (0.9, "HOPE", 2472, 2482, 199, 212, False),
    (0.9, "L1-In", 2472, 2482, 366, 394, False),
    (0.9, "L1-Joint", 2472, 2482, 336, 350, False),
    (0.9, "BN-Scale", 2472, 2482, 320, 342, False),
    (1.0, "HOPE", 2772, 2782, 95, 95, False),
    (1.0, "L1-In", 2772, 2782, 112, 132, False),
    (1.0, "L1-Joint", 2772, 2782, 97, 114, False),
    (1.0, "BN-Scale", 2772, 2782, 93, 117, False),
)

MISSING_EXPERIMENTS = (
    (
        "compression", "action_set_decomposition", "enabled action types",
        "prune-only|merge-only|evict-only|combined", "equal parameter budget",
        "accuracy|action counts by layer and type",
    ),
    (
        "compression", "solitary_score_attribution", "neuron score",
        "BN scale|absolute gamma times outgoing norm|HOPE solitary score",
        "same prune-only action space and parameter budget", "accuracy|ranking agreement",
    ),
    (
        "compression", "merge_semantics", "merge realization",
        "printed parent|sum-preserving parent", "same candidates and parameter budget",
        "accuracy|local map error|selected merges",
    ),
    (
        "compression", "cross_kernel_bias", "pair kernel",
        "zero-bias approximation|exact biased moment", "same candidates and parameter budget",
        "accuracy|ranking agreement|selected merges",
    ),
    (
        "compression", "surrogate_oracle", "geometry source",
        "analytic checkpoint surrogate|empirical activation oracle",
        "same action set and parameter budget", "accuracy|kernel error|ranking agreement",
    ),
    (
        "compression", "parameter_payoff", "payoff accounting",
        "static initial payoff|live exact payoff", "equal final parameter budget",
        "accuracy|action path|actual parameters",
    ),
    (
        "compression", "cache_refresh", "dependency refresh",
        "local refresh|dependency-aware full refresh", "same initial candidates and stopping budget",
        "ranking agreement|action path|accuracy",
    ),
    (
        "DEFT", "matched_partition_controls", "core and slack selector",
        "HOPE capacity|BN gamma magnitude|random mask", "matched frozen fraction",
        "target accuracy|source accuracy|H-score",
    ),
    (
        "DEFT", "merge_component", "redundancy merging",
        "merge enabled|merge disabled", "matched frozen and trainable counts",
        "target accuracy|source accuracy|H-score|released slots",
    ),
    (
        "DEFT", "stagewise_source_damage", "adaptation stage",
        "pre-transfer|post-merge|post-mask|post-target-training", "same source task and seed",
        "live source accuracy|restored source accuracy",
    ),
    (
        "DEFT", "evaluation_state", "source evaluation state",
        "live target-adapted state|restored source-specific state", "same selected checkpoint",
        "source accuracy|H-score",
    ),
    (
        "DEFT", "checkpoint_selector", "epoch selection",
        "peak target validation|peak validation H-score", "same training trajectory",
        "selected epoch|target accuracy|source accuracy|H-score",
    ),
    (
        "DEFT", "source_training_recipe", "source pretraining",
        "paper slack-engineering recipe|standard recipe", "matched architecture and training budget",
        "slack distribution|target accuracy|source accuracy|H-score",
    ),
    (
        "DEFT", "partition_sensitivity", "partition and merge budget",
        "percentile sweep|merge-budget sweep", "same source-target scenarios",
        "frozen fraction|released slots|target accuracy|source accuracy|H-score",
    ),
    (
        "DEFT", "ewc_grid_extension", "EWC regularization grid",
        "reported lambda grid|extended grid through Head-Only limit behavior",
        "same validation and checkpoint protocol", "winning lambda|target accuracy|source accuracy|H-score",
    ),
)

CLAIM_IDS = (
    "P11-COMPRESSION-PLOT",
    "P11-COMPRESSION-EVIDENCE-BOUNDARY",
    "P11-DENSITY-BUDGET-MISMATCH",
    "P11-TABLE2-RECORD",
    "P11-HSCORE-AGGREGATION-GAP",
    "P11-HSCORE-AGGREGATION",
    "P11-UNCERTAINTY-UNDEFINED",
    "P11-CHECKPOINT-RULE-CONFLICT",
    "P11-SOURCE-DATA-SCOPE",
    "P11-ABSOLUTE-SOURCE-ACCURACY",
    "P11-CROSS-STATE-HSCORE",
    "P11-EVIDENCE-SCOPE",
    "P11-PROTOCOL-IDENTITY",
    "P10-SOURCE-STATE-RESTORATION",
    "EWC-TRUNCATED-GRID",
)

CLAIM_CLASSES = {
    "P11-COMPRESSION-PLOT": "paper_claim",
    "P11-COMPRESSION-EVIDENCE-BOUNDARY": "verified_fact",
    "P11-DENSITY-BUDGET-MISMATCH": "independent_inference",
    "P11-TABLE2-RECORD": "paper_claim",
    "P11-HSCORE-AGGREGATION-GAP": "verified_fact",
    "P11-HSCORE-AGGREGATION": "open_question",
    "P11-UNCERTAINTY-UNDEFINED": "verified_fact",
    "P11-CHECKPOINT-RULE-CONFLICT": "verified_fact",
    "P11-SOURCE-DATA-SCOPE": "independent_inference",
    "P11-ABSOLUTE-SOURCE-ACCURACY": "verified_fact",
    "P11-CROSS-STATE-HSCORE": "independent_inference",
    "P11-EVIDENCE-SCOPE": "independent_inference",
    "P11-PROTOCOL-IDENTITY": "open_question",
    "P10-SOURCE-STATE-RESTORATION": "paper_claim",
    "EWC-TRUNCATED-GRID": "independent_inference",
}

METHOD_ORIGINS = {
    "table2_recalculation": {
        "evidence_label": "verified_fact",
        "claim_id": "P11-HSCORE-AGGREGATION-GAP",
        "origin": "Eq. 101 applied to exact displayed Table 2 central values",
        "scope": "arithmetic on displayed two-decimal means, not task-level reproduction",
    },
    "aggregation_fixture": {
        "evidence_label": "verified_fact",
        "claim_id": "P11-HSCORE-AGGREGATION-GAP",
        "origin": "independent exact two-task fixture",
        "scope": "non-equivalence of two aggregation orders only",
    },
    "uncertainty_interpretation_audit": {
        "evidence_label": "verified_fact",
        "claim_id": "P11-UNCERTAINTY-UNDEFINED",
        "origin": "Table 2 caption and complete HOPE v1 experimental protocol",
        "scope": "definition and inferential support, not a significance calculation",
    },
    "source_retention_record": {
        "evidence_label": "verified_fact",
        "claim_id": "P11-ABSOLUTE-SOURCE-ACCURACY",
        "origin": "Table 2 source column and Appendix G.5 final source test evaluation",
        "scope": "paper-record row only; no normalized retention denominator is reported",
    },
    "retention_fixture": {
        "evidence_label": "independent_inference",
        "claim_id": "P11-EVIDENCE-SCOPE",
        "origin": "independent hypothetical denominators applied to the reported DEFT source accuracy",
        "scope": "hypothetical rows only; values are not paper results and the paper-record row has separate provenance",
    },
    "checkpoint_fixture": {
        "evidence_label": "verified_fact",
        "claim_id": "P11-CHECKPOINT-RULE-CONFLICT",
        "origin": "independent two-epoch witness for the two printed selection rules",
        "scope": "selector non-equivalence only; not the paper training trajectory",
    },
    "density_budget_fixture": {
        "evidence_label": "independent_inference",
        "claim_id": "P11-DENSITY-BUDGET-MISMATCH",
        "origin": "independent weight-only dense-chain parameter count",
        "scope": "axis-objective non-equivalence only; not ResNet-50",
    },
    "compression_digitization": {
        "evidence_label": "independent_inference",
        "claim_id": "P11-EVIDENCE-SCOPE",
        "origin": "authenticated embedded raster on HOPE v1 PDF page 20",
        "scope": "digitized approximation with pixel intervals, not author-reported numeric data",
    },
    "missing_experiment_matrix": {
        "evidence_label": "independent_inference",
        "claim_id": "P11-EVIDENCE-SCOPE",
        "origin": "independent proposed experimental matrix",
        "scope": "questions and matched controls only; no outcomes are claimed",
    },
}

REPRODUCTION_SCOPE = {
    "reproduction_scope": "paper_record_and_deterministic_evidence_semantics_audit",
    "paper_experiment_reproduced": False,
    "checkpoint_or_dataset_used": False,
    "trained_result_generated": False,
    "random_sampling_used": False,
    "wall_clock_benchmark_used": False,
    "includes": [
        "exact Table 2 transcription and Eq. 101 arithmetic",
        "conservative two-decimal rounding-bin test",
        "aggregation-order counterexample",
        "uncertainty, retention, evaluation-state, and checkpoint-rule semantics",
        "density-versus-parameter-payoff counterexample",
        "pixel-provenance-bound compression digitization",
        "proposed missing-experiment matrix",
    ],
    "excludes": [
        "CIFAR-100, SVHN, or ImageNet execution",
        "a trained-network reproduction or significance test",
        "recovery of unreported per-scenario values or uncertainty semantics",
        "a claim that a different checkpoint rule changes the published ranking",
        "a claim that parameter-matched compression reverses the plotted ranking",
        "outcomes for any proposed missing experiment",
    ],
    "claim_boundary": (
        "This audit reproduces arithmetic and exposes non-equivalent protocol choices. "
        "It does not reproduce either paper experiment or invent missing statistical semantics."
    ),
}

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part11_experimental_evidence/README.md",
    "experiments/audits/part11_experimental_evidence/run.py",
    "experiments/audits/part11_experimental_evidence/test_run.py",
)

SOURCE_MAP_FIELDS = (
    "step_id", "sequence", "paper_anchor", "object", "paper_record",
    "audit_result", "evidence_label", "claim_id", "passed",
)
TABLE2_FIELDS = (
    "method", "target_mean", "target_plus_minus", "source_mean",
    "source_plus_minus", "reported_h", "reported_h_plus_minus",
    "harmonic_of_displayed_means", "displayed_h_gap", "target_rounding_low",
    "target_rounding_high", "source_rounding_low", "source_rounding_high",
    "implied_h_low", "implied_h_high", "reported_h_rounding_low",
    "reported_h_rounding_high", "rounding_intervals_overlap",
    "aggregation_reconstructible", "aggregation_status", "evidence_label",
    "claim_id", "passed",
)
AGGREGATION_FIELDS = (
    "fixture_id", "task_id", "target_accuracy", "source_accuracy",
    "task_h", "mean_target", "mean_source", "harmonic_of_means",
    "mean_of_task_h", "aggregation_gap", "evidence_label", "claim_id", "passed",
)
UNCERTAINTY_FIELDS = (
    "interpretation_id", "paper_defines_interpretation", "compatible_with_notation",
    "sampling_unit_known", "inferential_use_allowed", "reported_test",
    "formal_significance_supported", "audit_conclusion", "evidence_label",
    "claim_id", "passed",
)
RETENTION_FIELDS = (
    "row_id", "row_kind", "pretransfer_source_accuracy",
    "posttransfer_source_accuracy", "normalization_rule",
    "normalized_retention_percent", "paper_reports_pretransfer_accuracy",
    "paper_reports_normalized_retention", "interpretation", "evidence_label",
    "claim_id", "passed",
)
EVALUATION_STATE_FIELDS = (
    "row_id", "metric_or_component", "live_target_state",
    "reported_source_evaluation_state", "state_differs", "paper_reports_transform",
    "paper_reports_live_source_metric", "cross_state_hscore_operand",
    "audit_conclusion", "evidence_label", "claim_id", "passed",
)
CHECKPOINT_FIELDS = (
    "epoch", "target_validation_accuracy", "source_validation_accuracy",
    "h_score", "selected_by_target_rule", "selected_by_h_rule",
    "paper_result", "evidence_label", "claim_id", "passed",
)
DENSITY_FIELDS = (
    "action_id", "layer", "widths_before", "widths_after", "active_neurons_before",
    "active_neurons_after", "neuron_density_before", "neuron_density_after",
    "density_reduction", "parameters_before", "parameters_after",
    "parameter_payoff", "parameter_retained_fraction", "fixture_scope",
    "paper_result", "evidence_label", "claim_id", "passed",
)
DIGITIZATION_FIELDS = (
    "density", "method", "raw_x_low", "raw_x_high", "raw_y_low", "raw_y_high",
    "accuracy_low", "accuracy_high", "accuracy_estimate", "value_kind",
    "vertical_trace", "provenance_label", "author_reported_numeric",
    "source_pdf_sha256", "pdf_page", "pdf_object", "raster_width",
    "raster_height", "rgb_pixel_sha256", "smask_pixel_sha256",
    "axis_calibration_id", "evidence_label", "claim_id", "passed",
)
MISSING_EXPERIMENT_FIELDS = (
    "application", "experiment_id", "factor", "control_arms", "matched_quantity",
    "readouts", "paper_reported", "outcome_known", "status", "evidence_label",
    "claim_id", "passed",
)

CSV_CONTRACTS = {
    "source_map.csv": SOURCE_MAP_FIELDS,
    "table2_recalculation.csv": TABLE2_FIELDS,
    "aggregation_fixture.csv": AGGREGATION_FIELDS,
    "uncertainty_interpretations.csv": UNCERTAINTY_FIELDS,
    "retention_fixture.csv": RETENTION_FIELDS,
    "evaluation_states.csv": EVALUATION_STATE_FIELDS,
    "checkpoint_fixture.csv": CHECKPOINT_FIELDS,
    "density_budget_fixture.csv": DENSITY_FIELDS,
    "compression_digitization.csv": DIGITIZATION_FIELDS,
    "missing_experiments.csv": MISSING_EXPERIMENT_FIELDS,
}
OUTPUT_FILENAMES = ("summary.json", *CSV_CONTRACTS.keys())

CHECK_SPECS = {
    "nonfinite_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "row_failure_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "evidence_class_mismatch_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "table2_method_count": {"minimum": 5.0, "maximum": 5.0, "units": "methods"},
    "table2_numeric_mismatch_count": {"minimum": 0.0, "maximum": 0.0, "units": "rows"},
    "displayed_harmonic_match_count": {"minimum": 0.0, "maximum": 0.0, "units": "methods"},
    "rounding_interval_overlap_count": {"minimum": 0.0, "maximum": 0.0, "units": "methods"},
    "aggregation_unresolved_count": {"minimum": 5.0, "maximum": 5.0, "units": "methods"},
    "fixture_mean_target_difference": {"minimum": 0.0, "maximum": 0.0, "units": "percentage_points"},
    "fixture_mean_source_difference": {"minimum": 0.0, "maximum": 0.0, "units": "percentage_points"},
    "balanced_mean_task_h": {"minimum": 50.0, "maximum": 50.0, "units": "percentage_points"},
    "crossed_mean_task_h": {"minimum": 18.0, "maximum": 18.0, "units": "percentage_points"},
    "taskwise_h_gap": {"minimum": 32.0, "maximum": 32.0, "units": "percentage_points"},
    "paper_defined_uncertainty_interpretation_count": {"minimum": 0.0, "maximum": 0.0, "units": "interpretations"},
    "uncertainty_inferential_use_allowed_count": {"minimum": 0.0, "maximum": 0.0, "units": "interpretations"},
    "reported_significance_test_count": {"minimum": 0.0, "maximum": 0.0, "units": "tests"},
    "paper_normalized_retention_value_count": {"minimum": 0.0, "maximum": 0.0, "units": "values"},
    "retention_fixture_value_count": {"minimum": 2.0, "maximum": 2.0, "units": "values"},
    "retention_fixture_range": {"minimum": 28.966666666666676, "maximum": 28.966666666666676, "units": "percentage_points"},
    "source_restoration_component_count": {"minimum": 3.0, "maximum": 3.0, "units": "components"},
    "cross_state_hscore_count": {"minimum": 1.0, "maximum": 1.0, "units": "scores"},
    "live_state_source_metric_reported_count": {"minimum": 0.0, "maximum": 0.0, "units": "metrics"},
    "target_rule_selected_epoch": {"minimum": 2.0, "maximum": 2.0, "units": "epoch"},
    "h_rule_selected_epoch": {"minimum": 1.0, "maximum": 1.0, "units": "epoch"},
    "checkpoint_selector_disagreement_count": {"minimum": 1.0, "maximum": 1.0, "units": "fixtures"},
    "density_reduction_difference": {"minimum": 0.0, "maximum": 0.0, "units": "fraction"},
    "hidden_1_parameter_payoff": {"minimum": 104.0, "maximum": 104.0, "units": "parameters"},
    "hidden_2_parameter_payoff": {"minimum": 12.0, "maximum": 12.0, "units": "parameters"},
    "parameter_payoff_difference": {"minimum": 92.0, "maximum": 92.0, "units": "parameters"},
    "digitization_row_count": {"minimum": 32.0, "maximum": 32.0, "units": "rows"},
    "vertical_trace_interval_count": {"minimum": 2.0, "maximum": 2.0, "units": "rows"},
    "author_reported_digitized_row_count": {"minimum": 0.0, "maximum": 0.0, "units": "rows"},
    "digitization_provenance_failure_count": {"minimum": 0.0, "maximum": 0.0, "units": "rows"},
    "missing_experiment_row_count": {"minimum": 15.0, "maximum": 15.0, "units": "rows"},
    "paper_reported_missing_experiment_count": {"minimum": 0.0, "maximum": 0.0, "units": "rows"},
    "known_missing_experiment_outcome_count": {"minimum": 0.0, "maximum": 0.0, "units": "rows"},
}


class AuditFailure(RuntimeError):
    """Raised when a declared Part 11 invariant fails."""


def _d(value: str | float | int | Decimal) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _harmonic(target: Decimal, source: Decimal) -> Decimal:
    if target < 0 or source < 0 or target > 100 or source > 100:
        raise AuditFailure("accuracy must lie in [0, 100]")
    if target + source == 0:
        return Decimal(0)
    return Decimal(2) * target * source / (target + source)


def _round_two(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _rounding_bin(center: Decimal) -> tuple[Decimal, Decimal]:
    half_unit = Decimal("0.005")
    return center - half_unit, center + half_unit


def _float(value: Decimal | float | int) -> float:
    return float(value)


def _source_map_rows() -> list[dict[str, Any]]:
    rows = (
        (
            "compression_plot", "Section 11.1, PDF page 20", "compression evidence",
            "test accuracy versus active-neuron density on a public Keras ResNet-50 ImageNet checkpoint",
            "the paper supplies one raster plot and a qualitative superiority statement",
            "paper_claim", "P11-COMPRESSION-PLOT",
        ),
        (
            "compression_numbers", "Section 11.1, PDF page 20", "numeric compression record",
            "no author-reported numeric table, repeated trials, uncertainty, parameter, FLOP, or latency budget, or complete reproduction protocol accompanies the raster",
            "all Part 11 curve values remain provenance-bound digitized approximations",
            "verified_fact", "P11-COMPRESSION-EVIDENCE-BOUNDARY",
        ),
        (
            "parameter_budget", "Eq. 21, PDF page 18", "compression objective",
            "the constraint is expressed through parameters released and a final parameter footprint",
            "active-neuron density is a different resource coordinate",
            "independent_inference", "P11-DENSITY-BUDGET-MISMATCH",
        ),
        (
            "table2", "Table 2, PDF page 24", "cross-domain results",
            "four trials with five tasks per trial; five rows report target, source, and H-score central values with plus-or-minus fields",
            "exact values are transcribed; displayed means rank DEFT first in H, Full FT first in target, and Head-Only first in source",
            "paper_claim", "P11-TABLE2-RECORD",
        ),
        (
            "h_formula", "Eq. 101, PDF page 61", "H-score",
            "H equals twice target accuracy times source accuracy divided by their sum",
            "the formula can be applied exactly to displayed central values",
            "paper_claim", "P11-TABLE2-RECORD",
        ),
        (
            "h_aggregation", "Table 2 caption, PDF page 24", "aggregation order",
            "all metrics are said to be averaged across tasks and trials",
            "per-scenario rows and an explicit operation order are absent, so the H column cannot be reconstructed",
            "open_question", "P11-HSCORE-AGGREGATION",
        ),
        (
            "uncertainty", "Table 2, PDF page 24", "plus-or-minus notation",
            "the paper does not define the statistic or sampling unit and reports no inferential test",
            "the fields cannot support a formal significance statement without added assumptions",
            "verified_fact", "P11-UNCERTAINTY-UNDEFINED",
        ),
        (
            "absolute_source_accuracy", "Table 2 and Appendix G.5, PDF pages 24 and 62",
            "source metric", "the source column is a post-transfer CIFAR test accuracy",
            "no pre-transfer source accuracy or normalization denominator is reported",
            "verified_fact", "P11-ABSOLUTE-SOURCE-ACCURACY",
        ),
        (
            "source_data_scope", "Table 1, Appendix G.5, and Algorithm 3, PDF pages 24, 61, and 70",
            "source-data independence scope",
            "Table 1 marks DEFT source-data independent while tuning and Algorithm 3 use source validation data",
            "the printed tuning and checkpoint-selection workflow is not independent of source examples; stored task state is a separate issue",
            "independent_inference", "P11-SOURCE-DATA-SCOPE",
        ),
        (
            "source_state", "Appendix G.5, PDF page 62", "source evaluation state",
            "the source head is grafted back, slack-origin head connections are masked, and source BN buffers are restored",
            "the reported source operand is not evaluation of the unchanged live target state",
            "paper_claim", "P10-SOURCE-STATE-RESTORATION",
        ),
        (
            "cross_state_h", "Eq. 101 and Appendix G.5, PDF pages 61-62", "H-score operands",
            "target and source accuracies are evaluated under different task-specific states",
            "the reported H-score is a cross-state tradeoff metric under the printed protocol",
            "independent_inference", "P11-CROSS-STATE-HSCORE",
        ),
        (
            "target_checkpoint", "Appendix G.5 and Table 2 caption, PDF pages 62 and 24",
            "final epoch selection", "select the epoch with peak target validation accuracy",
            "this differs from Algorithm 3's printed selector",
            "verified_fact", "P11-CHECKPOINT-RULE-CONFLICT",
        ),
        (
            "h_checkpoint", "Algorithm 3 lines 27-34, PDF page 70", "final epoch selection",
            "select the epoch with peak validation H-score using target and source validation",
            "the two selectors can choose different checkpoints",
            "verified_fact", "P11-CHECKPOINT-RULE-CONFLICT",
        ),
        (
            "protocol_identity", "Appendix G.5, Table 2, and Algorithm 3", "executed protocol",
            "the printed descriptions omit exact task identities, split and training seeds, winning hyperparameters, aggregation and uncertainty definitions, one reconciled DEFT partition and checkpoint rule, and source-state policy per baseline",
            "the implementation identity that generated Table 2 remains unresolved",
            "open_question", "P11-PROTOCOL-IDENTITY",
        ),
    )
    return [
        {
            "step_id": step_id,
            "sequence": sequence,
            "paper_anchor": anchor,
            "object": object_name,
            "paper_record": record,
            "audit_result": result,
            "evidence_label": evidence_label,
            "claim_id": claim_id,
            "passed": True,
        }
        for sequence, (
            step_id, anchor, object_name, record, result, evidence_label, claim_id,
        ) in enumerate(rows, start=1)
    ]


def _table2_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, target_s, target_pm_s, source_s, source_pm_s, h_s, h_pm_s in TABLE2_RECORDS:
        target, source, reported_h = _d(target_s), _d(source_s), _d(h_s)
        harmonic = _harmonic(target, source)
        target_low, target_high = _rounding_bin(target)
        source_low, source_high = _rounding_bin(source)
        implied_low = _harmonic(target_low, source_low)
        implied_high = _harmonic(target_high, source_high)
        reported_low, reported_high = _rounding_bin(reported_h)
        overlap = max(implied_low, reported_low) <= min(implied_high, reported_high)
        signature = tuple(float(v) for v in (
            target_s, target_pm_s, source_s, source_pm_s, h_s, h_pm_s
        ))
        exact_record = signature == TABLE2_EXPECTED_SIGNATURES[method]
        direct_match = _round_two(harmonic) == reported_h
        rows.append({
            "method": method,
            "target_mean": _float(target),
            "target_plus_minus": float(target_pm_s),
            "source_mean": _float(source),
            "source_plus_minus": float(source_pm_s),
            "reported_h": _float(reported_h),
            "reported_h_plus_minus": float(h_pm_s),
            "harmonic_of_displayed_means": _float(harmonic),
            "displayed_h_gap": _float(harmonic - reported_h),
            "target_rounding_low": _float(target_low),
            "target_rounding_high": _float(target_high),
            "source_rounding_low": _float(source_low),
            "source_rounding_high": _float(source_high),
            "implied_h_low": _float(implied_low),
            "implied_h_high": _float(implied_high),
            "reported_h_rounding_low": _float(reported_low),
            "reported_h_rounding_high": _float(reported_high),
            "rounding_intervals_overlap": overlap,
            "aggregation_reconstructible": False,
            "aggregation_status": "task-level aggregation order and values are not recoverable from Table 2",
            "evidence_label": "verified_fact",
            "claim_id": "P11-HSCORE-AGGREGATION-GAP",
            "passed": exact_record and not direct_match and not overlap,
        })
    return rows


def _aggregation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture_id, task_rows in AGGREGATION_FIXTURE.items():
        targets = [_d(row[1]) for row in task_rows]
        sources = [_d(row[2]) for row in task_rows]
        task_h = [_harmonic(target, source) for target, source in zip(targets, sources)]
        mean_target = sum(targets) / _d(len(targets))
        mean_source = sum(sources) / _d(len(sources))
        harmonic_of_means = _harmonic(mean_target, mean_source)
        mean_of_task_h = sum(task_h) / _d(len(task_h))
        gap = harmonic_of_means - mean_of_task_h
        expected = (Decimal(50), Decimal(0)) if fixture_id == "balanced" else (Decimal(18), Decimal(32))
        for (task_id, target, source), h_value in zip(task_rows, task_h):
            rows.append({
                "fixture_id": fixture_id,
                "task_id": task_id,
                "target_accuracy": target,
                "source_accuracy": source,
                "task_h": _float(h_value),
                "mean_target": _float(mean_target),
                "mean_source": _float(mean_source),
                "harmonic_of_means": _float(harmonic_of_means),
                "mean_of_task_h": _float(mean_of_task_h),
                "aggregation_gap": _float(gap),
                "evidence_label": "verified_fact",
                "claim_id": "P11-HSCORE-AGGREGATION-GAP",
                "passed": bool(
                    mean_target == 50 and mean_source == 50
                    and mean_of_task_h == expected[0] and gap == expected[1]
                ),
            })
    return rows


def _uncertainty_rows() -> list[dict[str, Any]]:
    return [
        {
            "interpretation_id": interpretation,
            "paper_defines_interpretation": False,
            "compatible_with_notation": True,
            "sampling_unit_known": False,
            "inferential_use_allowed": False,
            "reported_test": False,
            "formal_significance_supported": False,
            "audit_conclusion": "possible interpretation only; do not infer significance",
            "evidence_label": "verified_fact",
            "claim_id": "P11-UNCERTAINTY-UNDEFINED",
            "passed": True,
        }
        for interpretation in UNCERTAINTY_INTERPRETATIONS
    ]


def _retention_rows() -> list[dict[str, Any]]:
    post = Decimal("52.14")
    rows = [{
        "row_id": "paper_deft_source_column",
        "row_kind": "paper_record",
        "pretransfer_source_accuracy": "not_reported",
        "posttransfer_source_accuracy": _float(post),
        "normalization_rule": "requires an explicit pre-transfer denominator",
        "normalized_retention_percent": "not_computable",
        "paper_reports_pretransfer_accuracy": False,
        "paper_reports_normalized_retention": False,
        "interpretation": "absolute post-transfer source test accuracy under restored source evaluation state",
        "evidence_label": "verified_fact",
        "claim_id": "P11-ABSOLUTE-SOURCE-ACCURACY",
        "passed": True,
    }]
    for baseline in (Decimal("60"), Decimal("90")):
        normalized = Decimal(100) * post / baseline
        rows.append({
            "row_id": f"hypothetical_pretransfer_{int(baseline)}",
            "row_kind": "independent_hypothetical_fixture",
            "pretransfer_source_accuracy": _float(baseline),
            "posttransfer_source_accuracy": _float(post),
            "normalization_rule": "100 times post-transfer accuracy divided by pre-transfer accuracy",
            "normalized_retention_percent": _float(normalized),
            "paper_reports_pretransfer_accuracy": False,
            "paper_reports_normalized_retention": False,
            "interpretation": "same absolute source accuracy yields a denominator-dependent normalized value",
            "evidence_label": "independent_inference",
            "claim_id": "P11-EVIDENCE-SCOPE",
            "passed": baseline > post and normalized < 100,
        })
    return rows


def _evaluation_state_rows() -> list[dict[str, Any]]:
    specifications = (
        (
            "target_metric", "target test accuracy", "live target-adapted model",
            "not_applicable", False, False, False, "target",
            "the target operand is evaluated on the target-adapted state",
            "independent_inference", "P11-CROSS-STATE-HSCORE",
        ),
        (
            "source_head", "classification head", "target head",
            "original source head grafted back", True, True, False, "source",
            "reported source evaluation changes the task head",
            "paper_claim", "P10-SOURCE-STATE-RESTORATION",
        ),
        (
            "source_head_mask", "source-head slack-origin connections", "not source-masked",
            "slack-origin connections zeroed", True, True, False, "source",
            "reported source evaluation applies a source-specific mask",
            "paper_claim", "P10-SOURCE-STATE-RESTORATION",
        ),
        (
            "source_bn_buffers", "BatchNorm moving mean and variance", "target-adapted buffers",
            "original source buffers restored", True, True, False, "source",
            "reported source evaluation restores source-specific buffers",
            "paper_claim", "P10-SOURCE-STATE-RESTORATION",
        ),
        (
            "h_score_operands", "H-score", "target operand from live target state",
            "source operand from restored source state", True, True, False, "both",
            "the score combines operands evaluated under different states",
            "independent_inference", "P11-CROSS-STATE-HSCORE",
        ),
    )
    return [
        {
            "row_id": row_id,
            "metric_or_component": component,
            "live_target_state": live_state,
            "reported_source_evaluation_state": source_state,
            "state_differs": differs,
            "paper_reports_transform": reports_transform,
            "paper_reports_live_source_metric": live_source_reported,
            "cross_state_hscore_operand": operand,
            "audit_conclusion": conclusion,
            "evidence_label": evidence_label,
            "claim_id": claim_id,
            "passed": True,
        }
        for (
            row_id, component, live_state, source_state, differs, reports_transform,
            live_source_reported, operand, conclusion, evidence_label, claim_id,
        ) in specifications
    ]


def _checkpoint_rows() -> list[dict[str, Any]]:
    h_values = {
        epoch: _harmonic(_d(target), _d(source))
        for epoch, target, source in CHECKPOINT_FIXTURE
    }
    target_epoch = max(CHECKPOINT_FIXTURE, key=lambda row: row[1])[0]
    h_epoch = max(h_values, key=h_values.get)
    rows = []
    for epoch, target, source in CHECKPOINT_FIXTURE:
        rows.append({
            "epoch": epoch,
            "target_validation_accuracy": target,
            "source_validation_accuracy": source,
            "h_score": _float(h_values[epoch]),
            "selected_by_target_rule": epoch == target_epoch,
            "selected_by_h_rule": epoch == h_epoch,
            "paper_result": False,
            "evidence_label": "verified_fact",
            "claim_id": "P11-CHECKPOINT-RULE-CONFLICT",
            "passed": target_epoch == 2 and h_epoch == 1 and target_epoch != h_epoch,
        })
    return rows


def _dense_parameter_count(input_width: int, hidden_1: int, hidden_2: int, output: int) -> int:
    return input_width * hidden_1 + hidden_1 * hidden_2 + hidden_2 * output


def _density_rows() -> list[dict[str, Any]]:
    i = DENSITY_FIXTURE["input_width"]
    h1 = DENSITY_FIXTURE["hidden_1_width"]
    h2 = DENSITY_FIXTURE["hidden_2_width"]
    o = DENSITY_FIXTURE["output_width"]
    initial_parameters = _dense_parameter_count(i, h1, h2, o)
    initial_active = h1 + h2
    specifications = (
        ("prune_hidden_1", "hidden_1", h1 - 1, h2),
        ("prune_hidden_2", "hidden_2", h1, h2 - 1),
    )
    rows = []
    for action_id, layer, after_h1, after_h2 in specifications:
        after_parameters = _dense_parameter_count(i, after_h1, after_h2, o)
        after_active = after_h1 + after_h2
        payoff = initial_parameters - after_parameters
        rows.append({
            "action_id": action_id,
            "layer": layer,
            "widths_before": f"{i}|{h1}|{h2}|{o}",
            "widths_after": f"{i}|{after_h1}|{after_h2}|{o}",
            "active_neurons_before": initial_active,
            "active_neurons_after": after_active,
            "neuron_density_before": 1.0,
            "neuron_density_after": after_active / initial_active,
            "density_reduction": (initial_active - after_active) / initial_active,
            "parameters_before": initial_parameters,
            "parameters_after": after_parameters,
            "parameter_payoff": payoff,
            "parameter_retained_fraction": after_parameters / initial_parameters,
            "fixture_scope": DENSITY_FIXTURE["scope"],
            "paper_result": False,
            "evidence_label": "independent_inference",
            "claim_id": "P11-DENSITY-BUDGET-MISMATCH",
            "passed": initial_parameters == 248 and after_active == 5 and payoff in {104, 12},
        })
    return rows


def _pixel_to_accuracy(pixel_y: int) -> float:
    bottom_pixel = _d(COMPRESSION_RASTER["y_tick_bottom_pixel"])
    top_pixel = _d(COMPRESSION_RASTER["y_tick_top_pixel"])
    bottom_value = _d(COMPRESSION_RASTER["y_tick_bottom_value"])
    top_value = _d(COMPRESSION_RASTER["y_tick_top_value"])
    fraction = (bottom_pixel - _d(pixel_y)) / (bottom_pixel - top_pixel)
    return _float(bottom_value + fraction * (top_value - bottom_value))


def _digitization_rows() -> list[dict[str, Any]]:
    rows = []
    for density, method, x_low, x_high, y_low, y_high, vertical in DIGITIZATION_PIXELS:
        accuracy_low = _pixel_to_accuracy(y_high)
        accuracy_high = _pixel_to_accuracy(y_low)
        estimate: Any = "not_single_valued" if vertical else (accuracy_low + accuracy_high) / 2.0
        value_kind = "vertical_trace_interval" if vertical else "line_width_interval"
        passed = bool(
            0.0 <= accuracy_low <= accuracy_high <= 1.0
            and x_low <= x_high and y_low <= y_high
            and COMPRESSION_RASTER["width"] == 2929
            and COMPRESSION_RASTER["height"] == 1724
            and len(COMPRESSION_RASTER["rgb_pixel_sha256"]) == 64
            and len(COMPRESSION_RASTER["smask_pixel_sha256"]) == 64
        )
        rows.append({
            "density": density,
            "method": method,
            "raw_x_low": x_low,
            "raw_x_high": x_high,
            "raw_y_low": y_low,
            "raw_y_high": y_high,
            "accuracy_low": accuracy_low,
            "accuracy_high": accuracy_high,
            "accuracy_estimate": estimate,
            "value_kind": value_kind,
            "vertical_trace": vertical,
            "provenance_label": "digitized_approximation",
            "author_reported_numeric": False,
            "source_pdf_sha256": PAPER_SHA256,
            "pdf_page": COMPRESSION_RASTER["pdf_page"],
            "pdf_object": COMPRESSION_RASTER["pdf_object"],
            "raster_width": COMPRESSION_RASTER["width"],
            "raster_height": COMPRESSION_RASTER["height"],
            "rgb_pixel_sha256": COMPRESSION_RASTER["rgb_pixel_sha256"],
            "smask_pixel_sha256": COMPRESSION_RASTER["smask_pixel_sha256"],
            "axis_calibration_id": "page20_grid_v1",
            "evidence_label": "independent_inference",
            "claim_id": "P11-EVIDENCE-SCOPE",
            "passed": passed,
        })
    return rows


def _missing_experiment_rows() -> list[dict[str, Any]]:
    rows = []
    for application, experiment_id, factor, arms, matched, readouts in MISSING_EXPERIMENTS:
        claim_id = "EWC-TRUNCATED-GRID" if experiment_id == "ewc_grid_extension" else "P11-EVIDENCE-SCOPE"
        rows.append({
            "application": application,
            "experiment_id": experiment_id,
            "factor": factor,
            "control_arms": arms,
            "matched_quantity": matched,
            "readouts": readouts,
            "paper_reported": False,
            "outcome_known": False,
            "status": "proposed_not_run",
            "evidence_label": CLAIM_CLASSES[claim_id],
            "claim_id": claim_id,
            "passed": True,
        })
    return rows


def build_config(*, full: bool) -> dict[str, Any]:
    return {
        "full_mode": bool(full),
        "numeric_engine": "Python Decimal with precision 40 for H-score and rounding intervals",
        "rounding_model": (
            "closed center plus or minus 0.005 bins for displayed "
            "two-decimal central values"
        ),
        "table2_records": [list(row) for row in TABLE2_RECORDS],
        "aggregation_fixture": {
            key: [list(row) for row in values] for key, values in AGGREGATION_FIXTURE.items()
        },
        "checkpoint_fixture": [list(row) for row in CHECKPOINT_FIXTURE],
        "density_fixture": dict(DENSITY_FIXTURE),
        "compression_raster": dict(COMPRESSION_RASTER),
        "digitization_rule": (
            "map raw saturated-color pixel y-support through calibrated axes; "
            "retain vertical traces as intervals"
        ),
        "random_sampling": False,
    }


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


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def build_audit(*, full: bool = False) -> dict[str, Any]:
    config = build_config(full=full)
    source_rows = _source_map_rows()
    table_rows = _table2_rows()
    aggregation_rows = _aggregation_rows()
    uncertainty_rows = _uncertainty_rows()
    retention_rows = _retention_rows()
    evaluation_rows = _evaluation_state_rows()
    checkpoint_rows = _checkpoint_rows()
    density_rows = _density_rows()
    digitization_rows = _digitization_rows()
    missing_rows = _missing_experiment_rows()
    row_groups = (
        source_rows, table_rows, aggregation_rows, uncertainty_rows, retention_rows,
        evaluation_rows, checkpoint_rows, density_rows, digitization_rows, missing_rows,
    )
    all_rows = [row for group in row_groups for row in group]
    evidence_mismatches = sum(
        row["evidence_label"] != CLAIM_CLASSES[row["claim_id"]] for row in all_rows
    ) + sum(
        record["evidence_label"] != CLAIM_CLASSES[record["claim_id"]]
        for record in METHOD_ORIGINS.values()
    )

    aggregate_by_fixture = {}
    for fixture_id in AGGREGATION_FIXTURE:
        row = next(row for row in aggregation_rows if row["fixture_id"] == fixture_id)
        aggregate_by_fixture[fixture_id] = row
    normalized_values = [
        float(row["normalized_retention_percent"])
        for row in retention_rows if row["row_kind"] == "independent_hypothetical_fixture"
    ]
    checkpoint_target = next(row["epoch"] for row in checkpoint_rows if row["selected_by_target_rule"])
    checkpoint_h = next(row["epoch"] for row in checkpoint_rows if row["selected_by_h_rule"])
    density_by_action = {row["action_id"]: row for row in density_rows}

    checks = {
        "nonfinite_count": float(sum(not math.isfinite(number) for number in _walk_numbers(all_rows))),
        "row_failure_count": float(sum(not row["passed"] for row in all_rows)),
        "evidence_class_mismatch_count": float(evidence_mismatches),
        "table2_method_count": float(len(table_rows)),
        "table2_numeric_mismatch_count": float(sum(
            tuple(row[key] for key in (
                "target_mean", "target_plus_minus", "source_mean", "source_plus_minus",
                "reported_h", "reported_h_plus_minus",
            )) != TABLE2_EXPECTED_SIGNATURES[row["method"]]
            for row in table_rows
        )),
        "displayed_harmonic_match_count": float(sum(
            _round_two(_d(row["harmonic_of_displayed_means"])) == _d(row["reported_h"])
            for row in table_rows
        )),
        "rounding_interval_overlap_count": float(sum(row["rounding_intervals_overlap"] for row in table_rows)),
        "aggregation_unresolved_count": float(sum(not row["aggregation_reconstructible"] for row in table_rows)),
        "fixture_mean_target_difference": abs(
            aggregate_by_fixture["balanced"]["mean_target"] - aggregate_by_fixture["crossed"]["mean_target"]
        ),
        "fixture_mean_source_difference": abs(
            aggregate_by_fixture["balanced"]["mean_source"] - aggregate_by_fixture["crossed"]["mean_source"]
        ),
        "balanced_mean_task_h": aggregate_by_fixture["balanced"]["mean_of_task_h"],
        "crossed_mean_task_h": aggregate_by_fixture["crossed"]["mean_of_task_h"],
        "taskwise_h_gap": abs(
            aggregate_by_fixture["balanced"]["mean_of_task_h"]
            - aggregate_by_fixture["crossed"]["mean_of_task_h"]
        ),
        "paper_defined_uncertainty_interpretation_count": float(sum(
            row["paper_defines_interpretation"] for row in uncertainty_rows
        )),
        "uncertainty_inferential_use_allowed_count": float(sum(
            row["inferential_use_allowed"] for row in uncertainty_rows
        )),
        "reported_significance_test_count": float(sum(row["reported_test"] for row in uncertainty_rows)),
        "paper_normalized_retention_value_count": float(sum(
            row["paper_reports_normalized_retention"] for row in retention_rows
        )),
        "retention_fixture_value_count": float(len(normalized_values)),
        "retention_fixture_range": max(normalized_values) - min(normalized_values),
        "source_restoration_component_count": float(sum(
            row["row_id"] in {"source_head", "source_head_mask", "source_bn_buffers"}
            and row["state_differs"] for row in evaluation_rows
        )),
        "cross_state_hscore_count": float(sum(
            row["row_id"] == "h_score_operands" and row["state_differs"] for row in evaluation_rows
        )),
        "live_state_source_metric_reported_count": float(sum(
            row["paper_reports_live_source_metric"] for row in evaluation_rows
        )),
        "target_rule_selected_epoch": float(checkpoint_target),
        "h_rule_selected_epoch": float(checkpoint_h),
        "checkpoint_selector_disagreement_count": float(checkpoint_target != checkpoint_h),
        "density_reduction_difference": abs(
            density_by_action["prune_hidden_1"]["density_reduction"]
            - density_by_action["prune_hidden_2"]["density_reduction"]
        ),
        "hidden_1_parameter_payoff": float(density_by_action["prune_hidden_1"]["parameter_payoff"]),
        "hidden_2_parameter_payoff": float(density_by_action["prune_hidden_2"]["parameter_payoff"]),
        "parameter_payoff_difference": float(abs(
            density_by_action["prune_hidden_1"]["parameter_payoff"]
            - density_by_action["prune_hidden_2"]["parameter_payoff"]
        )),
        "digitization_row_count": float(len(digitization_rows)),
        "vertical_trace_interval_count": float(sum(row["vertical_trace"] for row in digitization_rows)),
        "author_reported_digitized_row_count": float(sum(
            row["author_reported_numeric"] for row in digitization_rows
        )),
        "digitization_provenance_failure_count": float(sum(
            row["source_pdf_sha256"] != PAPER_SHA256
            or row["rgb_pixel_sha256"] != COMPRESSION_RASTER["rgb_pixel_sha256"]
            or row["smask_pixel_sha256"] != COMPRESSION_RASTER["smask_pixel_sha256"]
            or row["provenance_label"] != "digitized_approximation"
            for row in digitization_rows
        )),
        "missing_experiment_row_count": float(len(missing_rows)),
        "paper_reported_missing_experiment_count": float(sum(row["paper_reported"] for row in missing_rows)),
        "known_missing_experiment_outcome_count": float(sum(row["outcome_known"] for row in missing_rows)),
    }
    if tuple(checks) != tuple(CHECK_SPECS):
        raise AuditFailure("internal check order does not match the fixed Part 11 registry")
    check_rows = {
        name: {
            "value": value,
            "minimum": CHECK_SPECS[name]["minimum"],
            "maximum": CHECK_SPECS[name]["maximum"],
            "units": CHECK_SPECS[name]["units"],
            "passed": bool(CHECK_SPECS[name]["minimum"] <= value <= CHECK_SPECS[name]["maximum"]),
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
                "Section 11.1, PDF page 20",
                "Section 11.2 and Table 2, PDF pages 23-24",
                "Eq. 21, PDF page 18",
                "Appendix G.5 and Eq. 101, PDF pages 61-62",
                "Algorithm 3, PDF page 70",
            ],
        },
        "config": config,
        "config_sha256": _config_hash(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": REPRODUCTION_SCOPE,
        "check_count": len(check_rows),
        "passing_check_count": sum(row["passed"] for row in check_rows.values()),
        "checks": check_rows,
        "table2_recalculation": {
            row["method"]: {
                "reported_h": row["reported_h"],
                "harmonic_of_displayed_means": row["harmonic_of_displayed_means"],
                "rounding_intervals_overlap": row["rounding_intervals_overlap"],
            }
            for row in table_rows
        },
        "aggregation_fixture": {
            fixture_id: {
                "mean_target": row["mean_target"],
                "mean_source": row["mean_source"],
                "harmonic_of_means": row["harmonic_of_means"],
                "mean_of_task_h": row["mean_of_task_h"],
            }
            for fixture_id, row in aggregate_by_fixture.items()
        },
        "checkpoint_fixture": {
            "peak_target_epoch": checkpoint_target,
            "peak_h_epoch": checkpoint_h,
            "paper_result": False,
        },
        "density_budget_fixture": {
            "initial_parameters": density_rows[0]["parameters_before"],
            "density_reduction": density_rows[0]["density_reduction"],
            "payoffs": {
                row["action_id"]: row["parameter_payoff"] for row in density_rows
            },
            "paper_result": False,
        },
        "digitization_provenance": dict(COMPRESSION_RASTER),
        "row_counts": {
            "source_map": len(source_rows),
            "table2_recalculation": len(table_rows),
            "aggregation_fixture": len(aggregation_rows),
            "uncertainty_interpretations": len(uncertainty_rows),
            "retention_fixture": len(retention_rows),
            "evaluation_states": len(evaluation_rows),
            "checkpoint_fixture": len(checkpoint_rows),
            "density_budget_fixture": len(density_rows),
            "compression_digitization": len(digitization_rows),
            "missing_experiments": len(missing_rows),
        },
    }
    if not passed:
        failed_checks = [name for name, row in check_rows.items() if not row["passed"]]
        failed_rows = [
            row.get("step_id", row.get("method", row.get("row_id", row.get("experiment_id", "unknown"))))
            for row in all_rows if not row["passed"]
        ]
        raise AuditFailure(f"audit checks failed: checks={failed_checks}, rows={failed_rows}")
    return {
        "summary": summary,
        "source_map.csv": source_rows,
        "table2_recalculation.csv": table_rows,
        "aggregation_fixture.csv": aggregation_rows,
        "uncertainty_interpretations.csv": uncertainty_rows,
        "retention_fixture.csv": retention_rows,
        "evaluation_states.csv": evaluation_rows,
        "checkpoint_fixture.csv": checkpoint_rows,
        "density_budget_fixture.csv": density_rows,
        "compression_digitization.csv": digitization_rows,
        "missing_experiments.csv": missing_rows,
    }


def validate_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise AuditFailure("summary schema changed")
    if summary.get("audit_id") != AUDIT_ID:
        raise AuditFailure("summary audit identity changed")
    if summary.get("status") != "passed" or summary.get("passed") is not True:
        raise AuditFailure("summary is not passing")
    if tuple(summary.get("claim_ids", ())) != CLAIM_IDS:
        raise AuditFailure("summary claim registry changed")
    if summary.get("claim_classes") != CLAIM_CLASSES:
        raise AuditFailure("summary evidence-class registry changed")
    checks = summary.get("checks")
    if not isinstance(checks, Mapping) or tuple(checks) != tuple(CHECK_SPECS):
        raise AuditFailure("summary check registry changed")
    expected_count = len(CHECK_SPECS)
    if summary.get("check_count") != expected_count or summary.get("passing_check_count") != expected_count:
        raise AuditFailure(f"summary must contain {expected_count} passing checks")
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
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2) + "\n",
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paper_identity(paper_pdf: Path | None, full: bool) -> dict[str, Any]:
    try:
        manifest = json.loads(PAPER_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditFailure(f"could not read paper manifest: {error}") from error
    if (
        manifest.get("arxiv_id") != "2607.21366"
        or manifest.get("version") != "v1"
        or manifest.get("pages") != 70
        or manifest.get("pdf_sha256") != PAPER_SHA256
    ):
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
    if paper_pdf is not None and (
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
            "platform": platform.platform(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "cpu_count": os.cpu_count(),
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
            "entrypoint": "experiments/audits/part11_experimental_evidence/run.py",
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
