#!/usr/bin/env python3
"""Deterministic local audit for the proposed HOPE-R reference design.

The audit uses fixed float64 fixtures only. It writes no canonical artifact,
does not train a model, and does not make dataset, accuracy, convergence,
latency, or universal compression claims.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hope_repro.proposed_repairs.hope_r.common import (  # noqa: E402
    CalibrationMeasure,
    FunctionalNeuron,
    calibrated_activation_gram,
    calibrated_cross_kernel,
    fold_batch_norm,
    functional_capacity,
    relu_self_kernel,
    validate_psd,
    zero_bias_cross_kernel_approximation,
)
from hope_repro.proposed_repairs.hope_r.deft import (  # noqa: E402
    BatchNormBufferPolicy,
    BatchNormBuffers,
    BatchNormOwnership,
    BatchNormState,
    EvaluationDeclaration,
    FixedWidthDeftStateMachine,
    LayerDefinition,
    SlotState,
    StateDeclaration,
    TaskStateSelector,
    clear_optimizer_state,
    momentumw_step,
    source_owned_parameter_masks,
    structural_slack_to_core_mask,
    two_stage_gradient_liveness,
)
from hope_repro.proposed_repairs.hope_r.graph import (  # noqa: E402
    BlockSpec,
    CacheKey,
    ChannelRef,
    ConsumerRef,
    GraphModel,
    LayerSpec,
    UnsupportedGraphError,
    account_resources,
    cache_universe,
    calibrated_branch_cost,
    check_identity_block,
    dependency_invalidation,
    graph_after_channel_surgery,
    initial_static_parameter_payoff,
    live_parameter_payoff,
)
from hope_repro.proposed_repairs.hope_r.merge import (  # noqa: E402
    sum_map_optimized_merge,
)


AUDIT_ID = "part12_corrected_hope"
AUDIT_SCHEMA_VERSION = 2
FIXTURE_VERSION = "hope_r_deterministic_v1"
ANGULAR_SAMPLES = 128
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
PAPER_BYTES = 1_089_944

OUTPUT_FILENAMES = (
    "summary.json",
    "design_contract.csv",
    "common_gram_cases.csv",
    "merge_cases.csv",
    "invalidation_cases.csv",
    "resource_accounting.csv",
    "block_eligibility.csv",
    "deft_transitions.csv",
    "integration_ablations.csv",
    "source_map.csv",
)

CSV_CONTRACTS: dict[str, tuple[str, ...]] = {
    "design_contract.csv": (
        "contract_id",
        "requirement",
        "implementation_source",
        "scope_value",
        "claim_id",
        "evidence_label",
        "passed",
    ),
    "common_gram_cases.csv": (
        "case_id",
        "metric",
        "value",
        "reference_value",
        "tolerance",
        "comparator",
        "measure_label",
        "sample_count",
        "neuron_count",
        "claim_id",
        "evidence_label",
        "passed",
    ),
    "merge_cases.csv": (
        "case_id",
        "expected_exact",
        "span_rank",
        "candidate_count",
        "source_child_sum_norm",
        "parent_norm",
        "absolute_residual",
        "relative_residual",
        "residual_threshold",
        "criterion",
        "claim_id",
        "evidence_label",
        "passed",
    ),
    "invalidation_cases.csv": (
        "case_id",
        "action",
        "category",
        "observed_count",
        "expected_count",
        "missing_count",
        "extra_count",
        "criterion",
        "claim_id",
        "evidence_label",
        "passed",
    ),
    "resource_accounting.csv": (
        "case_id",
        "metric",
        "value",
        "reference_value",
        "tolerance",
        "comparator",
        "unit",
        "scope",
        "claim_id",
        "evidence_label",
        "passed",
    ),
    "block_eligibility.csv": (
        "case_id",
        "eligible",
        "expected_eligible",
        "reason_count",
        "expected_reason",
        "cost_evaluated",
        "direct_cost",
        "reference_cost",
        "tolerance",
        "claim_id",
        "evidence_label",
        "passed",
    ),
    "deft_transitions.csv": (
        "case_id",
        "metric",
        "value",
        "reference_value",
        "tolerance",
        "comparator",
        "state_before",
        "state_after",
        "claim_id",
        "evidence_label",
        "passed",
    ),
    "integration_ablations.csv": (
        "ablation_id",
        "baseline_method",
        "ablated_method",
        "baseline_metric",
        "ablated_metric",
        "absolute_delta",
        "criterion",
        "claim_id",
        "evidence_label",
        "passed",
    ),
    "source_map.csv": (
        "claim_id",
        "evidence_label",
        "implementation_source",
        "audit_output",
        "paper_anchor",
        "scope",
        "source_exists",
        "passed",
    ),
}

NUMERIC_FIELDS: dict[str, tuple[str, ...]] = {
    "design_contract.csv": ("scope_value",),
    "common_gram_cases.csv": (
        "value",
        "reference_value",
        "tolerance",
        "sample_count",
        "neuron_count",
    ),
    "merge_cases.csv": (
        "expected_exact",
        "span_rank",
        "candidate_count",
        "source_child_sum_norm",
        "parent_norm",
        "absolute_residual",
        "relative_residual",
        "residual_threshold",
    ),
    "invalidation_cases.csv": (
        "observed_count",
        "expected_count",
        "missing_count",
        "extra_count",
    ),
    "resource_accounting.csv": (
        "value",
        "reference_value",
        "tolerance",
    ),
    "block_eligibility.csv": (
        "eligible",
        "expected_eligible",
        "reason_count",
        "cost_evaluated",
        "direct_cost",
        "reference_cost",
        "tolerance",
    ),
    "deft_transitions.csv": ("value", "reference_value", "tolerance"),
    "integration_ablations.csv": (
        "baseline_metric",
        "ablated_metric",
        "absolute_delta",
    ),
    "source_map.csv": ("source_exists",),
}

CLAIM_CLASSES = {
    "P12-CALIBRATED-COMMON-MEASURE": "proposed_repair",
    "P12-COMMON-GRAM-PSD": "verified_fact",
    "P12-KERNEL-BOUNDARY-CONTRACT": "proposed_repair",
    "P12-SUM-MAP-MERGE": "proposed_repair",
    "P12-TWIN-PRESERVATION": "verified_fact",
    "P12-DEPENDENCY-CLOSURE": "proposed_repair",
    "P12-REFRESH-EQUIVALENCE": "verified_fact",
    "P12-LIVE-RESOURCE-ACCOUNTING": "proposed_repair",
    "P12-BLOCK-ELIGIBILITY": "proposed_repair",
    "P12-FIXED-WIDTH-DEFT": "proposed_repair",
    "P12-CORE-STATE-INVARIANCE": "verified_fact",
    "P12-GUARANTEE-SCOPE": "independent_inference",
}
CLAIM_IDS = tuple(CLAIM_CLASSES)

METHOD_ORIGINS = {
    "calibrated_common_geometry": {
        "namespace": "hope_repro.proposed_repairs.hope_r.common",
        "evidence_label": "proposed_repair",
        "claim_id": "P12-CALIBRATED-COMMON-MEASURE",
        "scope": "one fixed weighted synthetic calibration measure",
    },
    "physical_sum_map_merge": {
        "namespace": "hope_repro.proposed_repairs.hope_r.merge",
        "evidence_label": "proposed_repair",
        "claim_id": "P12-SUM-MAP-MERGE",
        "scope": "deterministic augmented-span search and measured local residual",
    },
    "graph_and_resource_contract": {
        "namespace": "hope_repro.proposed_repairs.hope_r.graph",
        "evidence_label": "proposed_repair",
        "claim_id": "P12-DEPENDENCY-CLOSURE",
        "scope": "resolved supported abstract channel graph and exact declared counts",
    },
    "fixed_width_deft_contract": {
        "namespace": "hope_repro.proposed_repairs.hope_r.deft",
        "evidence_label": "proposed_repair",
        "claim_id": "P12-FIXED-WIDTH-DEFT",
        "scope": "fixed deterministic slots masks optimizer state and BN ownership",
    },
}

REPRODUCTION_SCOPE = {
    "reproduction_scope": "deterministic_local_hope_r_contract_audit",
    "paper_experiment_reproduced": False,
    "checkpoint_or_dataset_used": False,
    "trained_result_generated": False,
    "wall_clock_benchmark_used": False,
    "random_sampling_used": False,
    "includes": [
        "fixed float64 calibration geometry",
        "physical one-parent merge residuals",
        "explicit graph dependency and resource counts",
        "fixed-width state and ownership transitions",
        "local deterministic ablations",
    ],
    "excludes": [
        "paper experiment execution",
        "checkpoint or benchmark data access",
        "accuracy loss convergence or prevalence measurement",
        "latency measurement",
        "universal function preservation or compression superiority",
    ],
    "claim_boundary": (
        "The fixtures establish local algebra graph state and accounting "
        "contracts only. They do not establish a trained-network outcome."
    ),
}

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part12_corrected_hope/README.md",
    "experiments/audits/part12_corrected_hope/run.py",
    "experiments/audits/part12_corrected_hope/test_run.py",
    "src/hope_repro/proposed_repairs/hope_r/common.py",
    "src/hope_repro/proposed_repairs/hope_r/merge.py",
    "src/hope_repro/proposed_repairs/hope_r/graph.py",
    "src/hope_repro/proposed_repairs/hope_r/deft.py",
    "src/hope_repro/proposed_repairs/hope_r/__init__.py",
    "src/hope_repro/proposed_repairs/hope_r/README.md",
)

CHECK_SPECS = {
    "nonfinite_count": {"minimum": 0.0, "maximum": 0.0, "units": "values"},
    "row_failure_count": {"minimum": 0.0, "maximum": 0.0, "units": "rows"},
    "evidence_class_mismatch_count": {"minimum": 0.0, "maximum": 0.0, "units": "rows"},
    "common_gram_minimum_eigenvalue": {"minimum": -1e-12, "maximum": 100.0, "units": "activation_squared"},
    "common_gram_symmetry_error": {"minimum": 0.0, "maximum": 1e-12, "units": "absolute"},
    "bn_fold_max_error": {"minimum": 0.0, "maximum": 1e-12, "units": "absolute"},
    "kernel_zero_scale_error": {"minimum": 0.0, "maximum": 0.0, "units": "absolute"},
    "maximum_twin_residual": {"minimum": 0.0, "maximum": 1e-12, "units": "weighted_l2"},
    "noncollinear_residual": {"minimum": 0.1, "maximum": 100.0, "units": "weighted_l2"},
    "dependency_key_count": {"minimum": 25.0, "maximum": 25.0, "units": "keys"},
    "complete_refresh_value_mismatch_count": {"minimum": 0.0, "maximum": 0.0, "units": "cache_values"},
    "local_only_stale_value_count": {"minimum": 20.0, "maximum": 20.0, "units": "cache_values"},
    "oracle_missing_key_count": {"minimum": 0.0, "maximum": 0.0, "units": "keys"},
    "oracle_extra_key_count": {"minimum": 0.0, "maximum": 0.0, "units": "keys"},
    "independent_cache_universe_count": {"minimum": 37.0, "maximum": 37.0, "units": "keys"},
    "live_parameter_payoff": {"minimum": 10.0, "maximum": 10.0, "units": "parameters"},
    "initial_static_parameter_payoff": {"minimum": 22.0, "maximum": 22.0, "units": "parameters"},
    "current_candidate_count": {"minimum": 13.0, "maximum": 13.0, "units": "evaluations"},
    "trajectory_candidate_evaluations": {"minimum": 24.0, "maximum": 24.0, "units": "evaluations"},
    "eligible_block_count": {"minimum": 1.0, "maximum": 1.0, "units": "blocks"},
    "block_rejection_case_count": {"minimum": 6.0, "maximum": 6.0, "units": "cases"},
    "fixed_width_delta": {"minimum": 0.0, "maximum": 0.0, "units": "slots"},
    "stable_slot_identity_count": {"minimum": 1.0, "maximum": 1.0, "units": "fixtures"},
    "maximum_masked_parameter_drift": {"minimum": 0.0, "maximum": 0.0, "units": "absolute"},
    "dual_track_source_buffer_drift": {"minimum": 0.0, "maximum": 0.0, "units": "absolute"},
    "integration_ablation_pass_count": {"minimum": 5.0, "maximum": 5.0, "units": "ablations"},
}


class AuditFailure(RuntimeError):
    """Raised when a deterministic Part 12 audit contract fails."""


def _row(claim_id: str, **values: Any) -> dict[str, Any]:
    if claim_id not in CLAIM_CLASSES:
        raise KeyError(f"unknown Part 12 claim ID: {claim_id}")
    return {
        **values,
        "claim_id": claim_id,
        "evidence_label": CLAIM_CLASSES[claim_id],
    }


def _comparison(value: float, reference: float, tolerance: float, comparator: str) -> bool:
    if comparator == "eq":
        return abs(value - reference) <= tolerance
    if comparator == "le":
        return value <= reference + tolerance
    if comparator == "lt":
        return value < reference + tolerance
    if comparator == "ge":
        return value + tolerance >= reference
    if comparator == "gt":
        return value + tolerance > reference
    raise ValueError(f"unknown comparator: {comparator}")


def _metric_row(
    *,
    case_id: str,
    metric: str,
    value: float,
    reference: float,
    tolerance: float,
    comparator: str,
    claim_id: str,
    **extra: Any,
) -> dict[str, Any]:
    return _row(
        claim_id,
        case_id=case_id,
        metric=metric,
        value=float(value),
        reference_value=float(reference),
        tolerance=float(tolerance),
        comparator=comparator,
        passed=_comparison(float(value), float(reference), float(tolerance), comparator),
        **extra,
    )


def _calibration_fixture() -> CalibrationMeasure:
    samples = np.asarray(
        [
            [-2.0, -1.0],
            [-2.0, 0.5],
            [-1.0, 2.0],
            [0.0, -2.0],
            [0.0, 0.0],
            [0.5, 1.5],
            [1.0, -1.0],
            [2.0, 0.5],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 2.0, 3.0],
        dtype=np.float64,
    )
    return CalibrationMeasure(samples, weights, label="part12_weighted_fixture")


def _common_geometry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    measure = _calibration_fixture()
    neurons = (
        FunctionalNeuron(
            np.asarray([0.8, -0.35]), 0.25, np.asarray([1.0, -2.0, 0.5])
        ),
        FunctionalNeuron(
            np.asarray([-0.4, 0.9]), -0.15, np.asarray([0.5, 1.0, -0.75])
        ),
        FunctionalNeuron(
            np.asarray([0.2, 0.3]), 0.4, np.asarray([-0.5, 0.25, 1.25])
        ),
    )
    gram = calibrated_activation_gram(measure, neurons)
    certificate = validate_psd(gram)
    cross = calibrated_cross_kernel(measure, neurons[0], neurons[1])
    cross_error = abs(cross - float(gram[0, 1]))

    folded = fold_batch_norm(
        np.asarray([2.0, -1.0]),
        0.75,
        -1.5,
        0.2,
        -0.25,
        3.0,
        0.125,
    )
    expected_scale = -1.5 / math.sqrt(3.125)
    expected_weight = expected_scale * np.asarray([2.0, -1.0])
    expected_bias = 0.2 + expected_scale
    fold_error = max(
        float(np.max(np.abs(folded.weight - expected_weight))),
        abs(folded.bias - expected_bias),
        abs(folded.batch_norm_scale - expected_scale),
    )
    zero_fold = fold_batch_norm(
        np.asarray([np.finfo(np.float64).max]),
        np.finfo(np.float64).max,
        0.0,
        -0.75,
        -np.finfo(np.float64).max,
        0.0,
        1e-5,
    )
    zero_fold_error = max(
        float(np.max(np.abs(zero_fold.weight))), abs(zero_fold.bias + 0.75)
    )
    positive_zero_scale_error = abs(relu_self_kernel(2.5, 0.0) - 6.25)
    negative_zero_scale_error = abs(relu_self_kernel(-2.5, -0.0))

    scaled = neurons[0].positive_rescaled(7.0)
    scaled_gram = calibrated_activation_gram(measure, (scaled,))
    base_gram = calibrated_activation_gram(measure, (neurons[0],))
    base_capacity = functional_capacity(base_gram[0, 0], neurons[0].outgoing_weight)
    scaled_capacity = functional_capacity(
        scaled_gram[0, 0], scaled.outgoing_weight
    )
    capacity_error = abs(base_capacity - scaled_capacity)
    approximation = zero_bias_cross_kernel_approximation(
        float(gram[0, 0]), float(gram[1, 1]), 0.35
    )

    shared = {
        "measure_label": measure.label,
        "sample_count": measure.samples.shape[0],
        "neuron_count": len(neurons),
    }
    rows = [
        _metric_row(
            case_id="measure_probability",
            metric="normalized_weight_sum",
            value=float(np.sum(measure.weights)),
            reference=1.0,
            tolerance=2e-15,
            comparator="eq",
            claim_id="P12-CALIBRATED-COMMON-MEASURE",
            **shared,
        ),
        _metric_row(
            case_id="weighted_activation_gram",
            metric="minimum_eigenvalue",
            value=certificate.minimum_eigenvalue,
            reference=-certificate.tolerance,
            tolerance=0.0,
            comparator="ge",
            claim_id="P12-COMMON-GRAM-PSD",
            **shared,
        ),
        _metric_row(
            case_id="weighted_activation_gram",
            metric="symmetry_error",
            value=certificate.symmetry_error,
            reference=certificate.tolerance,
            tolerance=0.0,
            comparator="le",
            claim_id="P12-COMMON-GRAM-PSD",
            **shared,
        ),
        _metric_row(
            case_id="exact_empirical_cross",
            metric="cross_vs_gram_error",
            value=cross_error,
            reference=0.0,
            tolerance=2e-15,
            comparator="eq",
            claim_id="P12-KERNEL-BOUNDARY-CONTRACT",
            **shared,
        ),
        _metric_row(
            case_id="finite_epsilon_bn",
            metric="fold_max_absolute_error",
            value=fold_error,
            reference=0.0,
            tolerance=2e-15,
            comparator="eq",
            claim_id="P12-KERNEL-BOUNDARY-CONTRACT",
            **shared,
        ),
        _metric_row(
            case_id="zero_gamma_bn",
            metric="constant_boundary_error",
            value=zero_fold_error,
            reference=0.0,
            tolerance=0.0,
            comparator="eq",
            claim_id="P12-KERNEL-BOUNDARY-CONTRACT",
            **shared,
        ),
        _metric_row(
            case_id="relu_gamma_zero_positive",
            metric="self_kernel_limit_error",
            value=positive_zero_scale_error,
            reference=0.0,
            tolerance=0.0,
            comparator="eq",
            claim_id="P12-KERNEL-BOUNDARY-CONTRACT",
            **shared,
        ),
        _metric_row(
            case_id="relu_gamma_zero_negative",
            metric="self_kernel_limit_error",
            value=negative_zero_scale_error,
            reference=0.0,
            tolerance=0.0,
            comparator="eq",
            claim_id="P12-KERNEL-BOUNDARY-CONTRACT",
            **shared,
        ),
        _metric_row(
            case_id="positive_rescaling",
            metric="capacity_absolute_error",
            value=capacity_error,
            reference=0.0,
            tolerance=2e-14,
            comparator="eq",
            claim_id="P12-KERNEL-BOUNDARY-CONTRACT",
            **shared,
        ),
        _metric_row(
            case_id="zero_bias_approximation",
            metric="finite_approximate_cross",
            value=approximation,
            reference=0.0,
            tolerance=0.0,
            comparator="ge",
            claim_id="P12-KERNEL-BOUNDARY-CONTRACT",
            **shared,
        ),
    ]
    return rows, {
        "measure": measure,
        "neurons": neurons,
        "gram": gram,
        "exact_cross": cross,
        "zero_bias_approximation": approximation,
        "minimum_gram_eigenvalue": certificate.minimum_eigenvalue,
        "gram_symmetry_error": certificate.symmetry_error,
        "bn_fold_max_error": fold_error,
        "kernel_zero_scale_error": max(
            zero_fold_error,
            positive_zero_scale_error,
            negative_zero_scale_error,
        ),
    }


def _merge_geometry(measure: CalibrationMeasure) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    zero_first = FunctionalNeuron(
        np.asarray([0.8, -0.35]), 0.0, np.asarray([1.0, -2.0, 0.5])
    )
    zero_second = zero_first.positive_rescaled(2.75)
    biased_first = FunctionalNeuron(
        np.asarray([0.8, -0.35]), 0.25, np.asarray([1.0, -2.0, 0.5])
    )
    biased_second = biased_first.positive_rescaled(3.25)
    noncollinear_first = FunctionalNeuron(
        np.asarray([1.0, 0.0]), 0.0, np.asarray([1.0, 0.0])
    )
    noncollinear_second = FunctionalNeuron(
        np.asarray([0.0, 1.0]), 0.3, np.asarray([0.0, 1.0])
    )
    cases = (
        ("zero_bias_twins", zero_first, zero_second, True),
        ("biased_twins", biased_first, biased_second, True),
        (
            "noncollinear_pair",
            noncollinear_first,
            noncollinear_second,
            False,
        ),
    )
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for case_id, first, second, expected_exact in cases:
        result = sum_map_optimized_merge(
            first,
            second,
            measure,
            angular_samples=ANGULAR_SAMPLES,
        )
        results[case_id] = result
        threshold = (
            2e-14 * max(1.0, result.source_child_sum_norm)
            if expected_exact
            else 0.1
        )
        passed = (
            result.absolute_residual <= threshold
            if expected_exact
            else result.absolute_residual > threshold
        )
        claim_id = (
            "P12-TWIN-PRESERVATION" if expected_exact else "P12-SUM-MAP-MERGE"
        )
        rows.append(
            _row(
                claim_id,
                case_id=case_id,
                expected_exact=int(expected_exact),
                span_rank=result.span_rank,
                candidate_count=result.candidate_count,
                source_child_sum_norm=result.source_child_sum_norm,
                parent_norm=result.parent_norm,
                absolute_residual=result.absolute_residual,
                relative_residual=result.relative_residual,
                residual_threshold=threshold,
                criterion="residual_le_threshold" if expected_exact else "residual_gt_threshold",
                passed=passed,
            )
        )
    results.update(
        {
            "noncollinear_first": noncollinear_first,
            "noncollinear_second": noncollinear_second,
        }
    )
    return rows, results


def _channels(layer_id: str, width: int) -> tuple[ChannelRef, ...]:
    return tuple(ChannelRef(layer_id, index) for index in range(width))


def _branched_graph() -> GraphModel:
    stem_channels = _channels("stem", 2)
    mid_channels = _channels("mid", 2)
    left_channels = _channels("left", 2)
    right_channels = _channels("right", 2)
    layers = (
        LayerSpec(
            "left",
            left_channels,
            mid_channels,
            initial_input_count=6,
            initial_output_count=5,
            terminal=True,
        ),
        LayerSpec(
            "mid",
            mid_channels,
            stem_channels,
            initial_input_count=8,
            initial_output_count=6,
        ),
        LayerSpec(
            "right",
            right_channels,
            mid_channels,
            initial_input_count=6,
            initial_output_count=5,
            terminal=True,
        ),
        LayerSpec("stem", stem_channels, (None, None)),
    )
    consumers = {
        stem_channels[0]: (ConsumerRef("mid", 0),),
        stem_channels[1]: (ConsumerRef("mid", 1),),
        mid_channels[0]: (ConsumerRef("left", 0), ConsumerRef("right", 0)),
        mid_channels[1]: (ConsumerRef("left", 1), ConsumerRef("right", 1)),
        left_channels[0]: (),
        left_channels[1]: (),
        right_channels[0]: (),
        right_channels[1]: (),
    }
    block = BlockSpec(
        block_id="identity_0",
        spanned_layers=("mid", "right"),
        input_shape=(2, 4, 4),
        output_shape=(2, 4, 4),
    )
    return GraphModel(layers=layers, consumer_map=consumers, blocks=(block,))


def _caught(call: Callable[[], Any], error_type: type[BaseException]) -> bool:
    try:
        call()
    except error_type:
        return True
    return False


def _oracle_cache_snapshot(
    graph: GraphModel,
    *,
    functional_revisions: Mapping[ChannelRef, int] | None = None,
) -> dict[CacheKey, str]:
    """Compute cache values from graph state without consulting an invalidation plan."""

    revisions = {} if functional_revisions is None else dict(functional_revisions)

    def channel_token(channel: ChannelRef | None) -> str:
        return "external" if channel is None else channel.token

    def incoming_signature(layer: LayerSpec) -> list[str]:
        return [channel_token(source) for source in layer.input_sources]

    def outgoing_signature(channel: ChannelRef) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for consumer in graph.consumer_map[channel]:
            target = graph.layer(consumer.layer_id)
            records.append(
                {
                    "target_layer": target.layer_id,
                    "target_live_outputs": target.live_output_count,
                    "target_operator": target.operator,
                    "target_kernel_shape": list(target.kernel_shape),
                }
            )
        return records

    blocks = {block.block_id: block for block in graph.blocks}
    snapshot: dict[CacheKey, str] = {}
    for key in cache_universe(graph):
        record: dict[str, Any] = {"key": key.as_record()}
        if key.kind == "block_score":
            block = blocks[key.owner_id]
            record["spanned_live_widths"] = [
                [layer_id, graph.layer(layer_id).live_output_count]
                for layer_id in block.spanned_layers
            ]
            record["spanned_revisions"] = [
                [channel.token, revisions.get(channel, 0)]
                for layer_id in block.spanned_layers
                for channel in graph.layer(layer_id).channels
            ]
        elif key.kind == "pair_geometry":
            layer = graph.layer(key.owner_id)
            record["incoming_sources"] = incoming_signature(layer)
            record["channels"] = [
                {
                    "token": channel.token,
                    "revision": revisions.get(channel, 0),
                    "outgoing": outgoing_signature(channel),
                }
                for channel in key.channels
            ]
        else:
            channel = key.channels[0]
            layer = graph.layer(channel.layer_id)
            record["revision"] = revisions.get(channel, 0)
            if key.kind in {"channel_capacity", "channel_incoming", "bn_assumption"}:
                record["incoming_sources"] = incoming_signature(layer)
            if key.kind in {"channel_capacity", "channel_outgoing"}:
                record["outgoing"] = outgoing_signature(channel)
        encoded = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        snapshot[key] = hashlib.sha256(encoded).hexdigest()
    return snapshot


def _oracle_invalidated_keys(
    before: Mapping[CacheKey, str], after: Mapping[CacheKey, str]
) -> set[CacheKey]:
    """Return removed or value-changed pre-surgery cache keys."""

    return {key for key, value in before.items() if after.get(key) != value}


def _refreshed_cache(
    before: Mapping[CacheKey, str],
    after: Mapping[CacheKey, str],
    invalidated: Sequence[CacheKey],
) -> dict[CacheKey, str]:
    values = dict(before)
    for key in invalidated:
        values.pop(key, None)
    for key in invalidated:
        if key in after:
            values[key] = after[key]
    for key in set(after) - set(before):
        values[key] = after[key]
    return values


def _cache_mismatch_count(
    observed: Mapping[CacheKey, str], expected: Mapping[CacheKey, str]
) -> int:
    sentinel = object()
    return sum(
        observed.get(key, sentinel) != expected.get(key, sentinel)
        for key in set(observed) | set(expected)
    )


def _graph_geometry() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    graph = _branched_graph()
    prune_plan = dependency_invalidation(graph, ChannelRef("mid", 0))
    merge_plan = dependency_invalidation(
        graph,
        (ChannelRef("mid", 1), ChannelRef("mid", 0)),
        action="merge",
        replacement_channel=ChannelRef("mid", 0),
    )
    expected_categories = {
        "current_layer_capacities": 1,
        "current_layer_outgoing": 1,
        "current_layer_incoming": 1,
        "current_layer_bn_assumptions": 1,
        "current_layer_pairs": 1,
        "predecessor_outgoing": 2,
        "predecessor_capacities": 2,
        "predecessor_pairs": 1,
        "successor_incoming": 4,
        "successor_capacities": 4,
        "successor_pairs": 2,
        "successor_bn_assumptions": 4,
        "spanning_block_scores": 1,
    }
    invalidation_rows: list[dict[str, Any]] = []
    for category, observed in prune_plan.category_counts():
        expected = expected_categories[category]
        invalidation_rows.append(
            _row(
                "P12-DEPENDENCY-CLOSURE",
                case_id=f"prune_{category}",
                action="prune",
                category=category,
                observed_count=observed,
                expected_count=expected,
                missing_count=max(expected - observed, 0),
                extra_count=max(observed - expected, 0),
                criterion="exact_category_count",
                passed=observed == expected,
            )
        )
    before_values = _oracle_cache_snapshot(graph)
    prune_after_graph = graph_after_channel_surgery(graph, prune_plan)
    merge_after_graph = graph_after_channel_surgery(graph, merge_plan)
    prune_after_values = _oracle_cache_snapshot(prune_after_graph)
    merge_after_values = _oracle_cache_snapshot(
        merge_after_graph,
        functional_revisions={merge_plan.replacement_channel: 1},
    )
    prune_oracle = _oracle_invalidated_keys(before_values, prune_after_values)
    merge_oracle = _oracle_invalidated_keys(before_values, merge_after_values)

    for case_id, action, plan, oracle in (
        ("prune_total", "prune", prune_plan, prune_oracle),
        ("merge_total", "merge", merge_plan, merge_oracle),
    ):
        observed = len(plan.cache_keys)
        expected = len(oracle)
        invalidation_rows.append(
            _row(
                "P12-DEPENDENCY-CLOSURE",
                case_id=case_id,
                action=action,
                category="complete_cache_key_union",
                observed_count=observed,
                expected_count=expected,
                missing_count=max(expected - observed, 0),
                extra_count=max(observed - expected, 0),
                criterion="count_matches_independent_post_surgery_oracle",
                passed=observed == expected,
            )
        )

    oracle_missing_keys: set[CacheKey] = set()
    oracle_extra_keys: set[CacheKey] = set()
    for case_id, action, plan, oracle in (
        ("prune_oracle_coverage", "prune", prune_plan, prune_oracle),
        ("merge_oracle_coverage", "merge", merge_plan, merge_oracle),
    ):
        missing = oracle - set(plan.cache_keys)
        extra = set(plan.cache_keys) - oracle
        oracle_missing_keys.update(missing)
        oracle_extra_keys.update(extra)
        invalidation_rows.append(
            _row(
                "P12-DEPENDENCY-CLOSURE",
                case_id=case_id,
                action=action,
                category="independent_oracle_set_equality",
                observed_count=len(plan.cache_keys),
                expected_count=len(oracle),
                missing_count=len(missing),
                extra_count=len(extra),
                criterion="no_missing_or_extra_cache_keys",
                passed=not missing and not extra,
            )
        )

    incremental_values = _refreshed_cache(
        before_values, prune_after_values, prune_plan.cache_keys
    )
    local_only_values = _refreshed_cache(
        before_values, prune_after_values, prune_plan.current_layer_keys
    )
    complete_value_mismatches = _cache_mismatch_count(
        incremental_values, prune_after_values
    )
    local_only_stale_values = _cache_mismatch_count(
        local_only_values, prune_after_values
    )
    for case_id, values, expected_mismatches, criterion in (
        (
            "complete_incremental_refresh",
            incremental_values,
            0,
            "cache_values_match_full_recompute",
        ),
        (
            "local_only_negative_control",
            local_only_values,
            20,
            "counts_stale_cache_values",
        ),
    ):
        mismatches = _cache_mismatch_count(values, prune_after_values)
        invalidation_rows.append(
            _row(
                "P12-REFRESH-EQUIVALENCE",
                case_id=case_id,
                action="prune",
                category="cache_value_refresh_equivalence",
                observed_count=mismatches,
                expected_count=expected_mismatches,
                missing_count=mismatches,
                extra_count=0,
                criterion=criterion,
                passed=mismatches == expected_mismatches,
            )
        )

    unresolved_layers = tuple(
        replace(layer, consumers_resolved=False)
        if layer.layer_id == "mid"
        else layer
        for layer in graph.layers
    )
    unresolved_rejected = _caught(
        lambda: GraphModel(unresolved_layers, graph.consumer_map, graph.blocks),
        UnsupportedGraphError,
    )
    grouped_layers = tuple(
        replace(layer, groups=2) if layer.layer_id == "mid" else layer
        for layer in graph.layers
    )
    grouped_rejected = _caught(
        lambda: GraphModel(grouped_layers, graph.consumer_map, graph.blocks),
        UnsupportedGraphError,
    )
    for case_id, rejected in (
        ("unresolved_fanout", unresolved_rejected),
        ("grouped_operator", grouped_rejected),
    ):
        invalidation_rows.append(
            _row(
                "P12-DEPENDENCY-CLOSURE",
                case_id=case_id,
                action="validation",
                category="fail_closed_rejection",
                observed_count=int(rejected),
                expected_count=1,
                missing_count=int(not rejected),
                extra_count=0,
                criterion="unsupported_graph_rejected",
                passed=rejected,
            )
        )

    current = dict(graph.live_widths)
    after = dict(current)
    after["mid"] = 1
    report = account_resources(
        graph,
        trajectory_states=(current, after),
        dependency_plans=(prune_plan,),
        cache_scalars_per_pair=3,
        scalar_dtype=np.float64,
        cache_indices_per_pair=2,
        index_dtype=np.uint32,
        calibration_shapes=((3, 2),),
    )
    live_payoff = live_parameter_payoff(graph, ChannelRef("mid", 0))
    static_payoff = initial_static_parameter_payoff(graph, ChannelRef("mid", 0))
    resource_specs = (
        ("live_payoff", "live_parameter_payoff", live_payoff, 10, "eq", "parameters"),
        ("static_payoff", "initial_static_payoff", static_payoff, 22, "eq", "parameters"),
        ("payoff_gap", "static_minus_live_payoff", static_payoff - live_payoff, 12, "eq", "parameters"),
        ("current_prunes", "current_prune_candidates", report.current_prune_candidates, 8, "eq", "evaluations"),
        ("current_merges", "current_merge_candidates", report.current_merge_candidates, 4, "eq", "evaluations"),
        ("current_blocks", "current_block_candidates", report.current_block_candidates, 1, "eq", "evaluations"),
        ("current_total", "current_candidate_count", report.current_candidate_count, 13, "eq", "evaluations"),
        ("trajectory_scans", "trajectory_scan_count", report.trajectory_scan_count, 2, "eq", "scans"),
        ("trajectory_total", "trajectory_candidate_evaluations", report.trajectory_candidate_evaluations, 24, "eq", "evaluations"),
        ("pair_cache", "pair_cache_entries", report.pair_cache_entries, 4, "eq", "entries"),
        ("scalar_payload", "cache_scalar_payload_bytes", report.cache_scalar_payload_bytes, 96, "eq", "bytes"),
        ("index_payload", "cache_index_payload_bytes", report.cache_index_payload_bytes, 32, "eq", "bytes"),
        ("key_records", "cache_key_record_bytes", report.cache_key_record_bytes, 1, "ge", "bytes"),
        ("refresh_keys", "refresh_key_evaluations", report.refresh_key_evaluations, 25, "eq", "evaluations"),
        ("calibration_elements", "calibration_output_elements", report.calibration_output_elements, 6, "eq", "elements"),
        ("no_wallclock", "wall_clock_claim", int(report.wall_clock_claim), 0, "eq", "boolean"),
    )
    resource_rows = [
        _metric_row(
            case_id=case_id,
            metric=metric,
            value=value,
            reference=reference,
            tolerance=0.0,
            comparator=comparator,
            unit=unit,
            scope="declared_abstract_counts_not_timing",
            claim_id="P12-LIVE-RESOURCE-ACCOUNTING",
        )
        for case_id, metric, value, reference, comparator, unit in resource_specs
    ]
    return invalidation_rows, resource_rows, {
        "graph": graph,
        "prune_plan": prune_plan,
        "resource_report": report,
        "live_payoff": live_payoff,
        "static_payoff": static_payoff,
        "complete_refresh_value_mismatches": complete_value_mismatches,
        "local_only_stale_values": local_only_stale_values,
        "oracle_missing_keys": len(oracle_missing_keys),
        "oracle_extra_keys": len(oracle_extra_keys),
        "independent_cache_universe_count": len(before_values),
    }


def _block_rows(graph: GraphModel) -> list[dict[str, Any]]:
    valid = graph.blocks[0]
    cases = (
        ("eligible_identity", valid, True, "", 2.3125, 2.3125),
        ("eligible_weighted_cost", valid, True, "", 3.15625, 3.15625),
        ("projection_shortcut", replace(valid, shortcut_kind="projection"), False, "non_identity_shortcut", 0.0, 0.0),
        ("stride_change", replace(valid, stride=2), False, "non_unit_stride", 0.0, 0.0),
        ("resnet_v2", replace(valid, architecture="resnet_v2"), False, "architecture_not_resnet_v1", 0.0, 0.0),
        ("pre_activation", replace(valid, activation_order="pre_activation"), False, "not_post_activation", 0.0, 0.0),
        ("missing_skip", replace(valid, shortcut_kind="none"), False, "missing_skip", 0.0, 0.0),
        ("shape_change", replace(valid, output_shape=(4, 2, 8)), False, "shape_mismatch", 0.0, 0.0),
    )
    shortcut = np.asarray([[-1.0, 1.0], [2.0, -2.0]], dtype=np.float64)
    residual = np.asarray([[2.0, -0.5], [-3.0, 4.0]], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for case_id, spec, expected, expected_reason, stated_cost, reference in cases:
        eligibility = check_identity_block(spec)
        evaluated = expected
        if case_id == "eligible_weighted_cost":
            cost = calibrated_branch_cost(
                residual,
                shortcut,
                spec=spec,
                sample_weights=np.asarray([1.0, 3.0]),
            )
        elif expected:
            cost = calibrated_branch_cost(residual, shortcut, spec=spec)
        else:
            cost = stated_cost
        reason_ok = (
            not expected_reason or expected_reason in eligibility.reasons
        )
        passed = (
            eligibility.eligible == expected
            and reason_ok
            and abs(cost - reference) <= 1e-15
        )
        rows.append(
            _row(
                "P12-BLOCK-ELIGIBILITY",
                case_id=case_id,
                eligible=int(eligibility.eligible),
                expected_eligible=int(expected),
                reason_count=len(eligibility.reasons),
                expected_reason=expected_reason or "none",
                cost_evaluated=int(evaluated),
                direct_cost=cost,
                reference_cost=reference,
                tolerance=1e-15,
                passed=passed,
            )
        )
    return rows


def _deft_geometry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layers = (
        LayerDefinition("hidden-1", ("h1:2", "h1:0", "h1:1"), 2),
        LayerDefinition("hidden-2", ("h2:0", "h2:1"), 1),
    )
    capacities = {
        "h1:0": 4.0,
        "h1:1": 4.0,
        "h1:2": 1.0,
        "h2:0": 7.0,
        "h2:1": 3.0,
    }
    machine = FixedWidthDeftStateMachine(layers, capacities)
    original_ids = machine.stable_slot_ids
    original_width = sum(machine.fixed_widths.values())
    tie_core = machine.partitions[0].core_slot_ids
    core_release_rejected = _caught(lambda: machine.release("h1:0"), ValueError)
    core_state_after_rejection = machine.state("h1:0").value
    if not core_release_rejected or machine.state("h1:0") is not SlotState.CORE:
        raise AuditFailure("CORE slot release did not fail closed")
    machine.release("h1:2")
    released_state = machine.state("h1:2").value
    parameters = machine.reinitialize("h1:2", input_width=4, output_width=2)
    reinitialized_state = machine.state("h1:2").value
    activation, contribution = parameters.forward(np.ones(4, dtype=np.float64))
    machine.activate_reinitialized("h1:2")
    final_state = machine.state("h1:2").value
    final_width = sum(machine.fixed_widths.values())
    stable_ids_preserved = int(machine.stable_slot_ids == original_ids)

    trace = two_stage_gradient_liveness()
    route_mask = structural_slack_to_core_mask(
        (
            SlotState.CORE,
            SlotState.SLACK,
            SlotState.RELEASED,
            SlotState.REINITIALIZED,
        ),
        (SlotState.CORE, SlotState.SLACK),
    )
    blocked_routes = int(np.count_nonzero(route_mask == 0.0))
    masks_bundle = source_owned_parameter_masks(
        ("core", "slack"),
        (SlotState.CORE, SlotState.SLACK),
        input_width=2,
        output_width=3,
    )
    masks = masks_bundle.as_dict()
    core_outgoing_mask_max = float(np.max(masks_bundle.outgoing_weight[0]))
    optimizer_state = {
        name: np.full(mask.shape, 100.0, dtype=np.float64)
        for name, mask in masks.items()
    }
    reset_masks = {
        name: np.zeros(mask.shape, dtype=np.float64)
        for name, mask in masks.items()
    }
    reset_masks["outgoing_weight"][1, 0] = 1.0
    cleared = clear_optimizer_state(
        optimizer_state,
        masks,
        reset_masks=reset_masks,
    )
    cleared_reset_value = float(cleared["outgoing_weight"][1, 0])
    parameters_by_name = {
        name: np.arange(mask.size, dtype=np.float64).reshape(mask.shape) + 1.0
        for name, mask in masks.items()
    }
    gradients = {
        name: np.full(mask.shape, 2.0, dtype=np.float64)
        for name, mask in masks.items()
    }
    step = momentumw_step(
        parameters_by_name,
        gradients,
        optimizer_state,
        masks,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=0.2,
    )

    frozen_ownership = BatchNormOwnership(
        channel_slot_ids=("slot-0", "slot-1"),
        buffer_policy=BatchNormBufferPolicy.FROZEN,
        source_task_id="source-task",
        target_task_id="target-task",
        source_track_id="source-bn",
    )
    frozen_state = BatchNormState(
        ownership=frozen_ownership,
        scale=np.asarray([1.0, 2.0]),
        shift=np.asarray([0.0, 0.5]),
        buffers={
            "source-bn": BatchNormBuffers(
                np.asarray([0.0, 1.0]), np.asarray([1.0, 4.0])
            )
        },
    )
    frozen_updated = frozen_state.updated_for_task(
        np.asarray([[10.0, 20.0], [14.0, 28.0]]),
        task_id="target-task",
        update_rate=0.5,
    )
    frozen_drift = max(
        float(
            np.max(
                np.abs(
                    frozen_updated.buffers["source-bn"].running_mean
                    - frozen_state.buffers["source-bn"].running_mean
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    frozen_updated.buffers["source-bn"].running_variance
                    - frozen_state.buffers["source-bn"].running_variance
                )
            )
        ),
    )

    dual_ownership = BatchNormOwnership(
        channel_slot_ids=("slot-0",),
        buffer_policy="dual_track",
        source_task_id="source-task",
        target_task_id="target-task",
        source_track_id="source-bn",
        target_track_id="target-bn",
    )
    dual_state = BatchNormState(
        ownership=dual_ownership,
        scale=np.asarray([1.0]),
        shift=np.asarray([0.0]),
        buffers={
            "source-bn": BatchNormBuffers(np.asarray([0.0]), np.asarray([1.0])),
            "target-bn": BatchNormBuffers(np.asarray([0.0]), np.asarray([1.0])),
        },
    )
    dual_updated = dual_state.updated_for_task(
        np.asarray([[2.0], [6.0]]),
        task_id="target-task",
        update_rate=0.5,
    )
    dual_source_drift = float(
        np.max(
            np.abs(
                dual_updated.buffers["source-bn"].running_mean
                - dual_state.buffers["source-bn"].running_mean
            )
        )
    )
    target_mean = float(dual_updated.buffers["target-bn"].running_mean[0])
    target_variance = float(
        dual_updated.buffers["target-bn"].running_variance[0]
    )
    source_declaration = StateDeclaration(
        "source-state",
        "source-task",
        "source-head",
        "source-bn",
        ("slot-0",),
        (1.0,),
    )
    target_declaration = StateDeclaration(
        "target-state",
        "target-task",
        "target-head",
        "target-bn",
        ("slot-0",),
        (1.0,),
    )
    selector = TaskStateSelector(
        (source_declaration, target_declaration),
        batch_norm_ownership=dual_ownership,
    )
    source_evaluation = EvaluationDeclaration(
        "source-evaluation",
        "source-state",
        "source-task",
        "source-validation",
        "absolute_accuracy",
        "example_mean",
        "declared-checkpoint",
    )
    target_evaluation = EvaluationDeclaration(
        "target-evaluation",
        "target-state",
        "target-task",
        "target-validation",
        "absolute_accuracy",
        "example_mean",
        "declared-checkpoint",
    )
    probe = np.asarray([[0.0]], dtype=np.float64)
    source_output = float(
        dual_updated.infer_for_evaluation(probe, source_evaluation, selector)[0, 0]
    )
    target_output = float(
        dual_updated.infer_for_evaluation(probe, target_evaluation, selector)[0, 0]
    )
    selected_output_gap = abs(source_output - target_output)

    specs = (
        ("partition_tie", "core_tie_rule_h1_0_first", int(tie_core[0] == "h1:0"), 1.0, 0.0, "eq", "partition", "partitioned", "P12-FIXED-WIDTH-DEFT"),
        ("core_release_rejected", "core_release_rejected", int(core_release_rejected), 1.0, 0.0, "eq", "core", core_state_after_rejection, "P12-CORE-STATE-INVARIANCE"),
        ("release", "fixed_width_delta", final_width - original_width, 0.0, 0.0, "eq", "slack", final_state, "P12-FIXED-WIDTH-DEFT"),
        ("stable_identity", "stable_slot_ids_preserved", stable_ids_preserved, 1.0, 0.0, "eq", released_state, final_state, "P12-FIXED-WIDTH-DEFT"),
        ("reinitialize", "initial_activation", activation, 2.0, 1e-15, "eq", released_state, reinitialized_state, "P12-FIXED-WIDTH-DEFT"),
        ("reinitialize", "initial_contribution_norm", float(np.linalg.norm(contribution)), 0.0, 0.0, "eq", released_state, reinitialized_state, "P12-FIXED-WIDTH-DEFT"),
        ("gradient_stage_one", "outgoing_gradient", trace.first_outgoing_gradient, -1.0, 0.0, "eq", "reinitialized", "stage_one", "P12-CORE-STATE-INVARIANCE"),
        ("gradient_stage_two", "incoming_gradient", trace.second_incoming_gradient, -0.1875, 0.0, "eq", "stage_one", "stage_two", "P12-CORE-STATE-INVARIANCE"),
        ("route_mask", "blocked_plastic_to_core_routes", blocked_routes, 3.0, 0.0, "eq", "declared", "masked", "P12-FIXED-WIDTH-DEFT"),
        ("source_masks", "core_outgoing_mask_max", core_outgoing_mask_max, 0.0, 0.0, "eq", "declared", "masked", "P12-CORE-STATE-INVARIANCE"),
        ("optimizer_reset", "reused_state_value", cleared_reset_value, 0.0, 0.0, "eq", "stale", "cleared", "P12-FIXED-WIDTH-DEFT"),
        ("masked_optimizer", "maximum_core_drift", step.max_abs_masked_drift, 0.0, 0.0, "eq", "before_step", "after_step", "P12-CORE-STATE-INVARIANCE"),
        ("frozen_bn", "source_buffer_drift", frozen_drift, 0.0, 0.0, "eq", "source_owned", "target_update", "P12-CORE-STATE-INVARIANCE"),
        ("dual_track_bn", "source_track_drift", dual_source_drift, 0.0, 0.0, "eq", "source_track", "target_update", "P12-CORE-STATE-INVARIANCE"),
        ("dual_track_bn", "target_running_mean", target_mean, 2.0, 0.0, "eq", "target_track", "target_update", "P12-FIXED-WIDTH-DEFT"),
        ("dual_track_bn", "target_running_variance", target_variance, 2.5, 0.0, "eq", "target_track", "target_update", "P12-FIXED-WIDTH-DEFT"),
        ("task_selector", "task_conditional_output_gap", selected_output_gap, 0.0, 0.0, "gt", "source_state", "target_state", "P12-FIXED-WIDTH-DEFT"),
    )
    rows = [
        _metric_row(
            case_id=case_id,
            metric=metric,
            value=value,
            reference=reference,
            tolerance=tolerance,
            comparator=comparator,
            state_before=before,
            state_after=after,
            claim_id=claim_id,
        )
        for (
            case_id,
            metric,
            value,
            reference,
            tolerance,
            comparator,
            before,
            after,
            claim_id,
        ) in specs
    ]
    return rows, {
        "masks": masks,
        "parameters": parameters_by_name,
        "gradients": gradients,
        "optimizer_state": optimizer_state,
        "masked_step": step,
        "dual_source_drift": dual_source_drift,
        "selected_output_gap": selected_output_gap,
        "fixed_width_delta": final_width - original_width,
        "stable_slot_ids_preserved": stable_ids_preserved,
    }


def _design_rows() -> list[dict[str, Any]]:
    specs = (
        ("one_measure", "one declared weighted measure per layer", "common.py", "P12-CALIBRATED-COMMON-MEASURE"),
        ("common_psd", "explicit PSD validation of the activation Gram", "common.py", "P12-COMMON-GRAM-PSD"),
        ("kernel_boundary", "finite-epsilon fold and exact zero-gamma limit", "common.py", "P12-KERNEL-BOUNDARY-CONTRACT"),
        ("physical_merge", "measured executed-child-sum residual", "merge.py", "P12-SUM-MAP-MERGE"),
        ("twin_anchor", "exact child direction anchors", "merge.py", "P12-TWIN-PRESERVATION"),
        ("dependency_closure", "complete resolved graph invalidation", "graph.py", "P12-DEPENDENCY-CLOSURE"),
        ("refresh_control", "full-key incremental refresh comparison", "graph.py", "P12-REFRESH-EQUIVALENCE"),
        ("resource_scope", "operation and byte counts without timing", "graph.py", "P12-LIVE-RESOURCE-ACCOUNTING"),
        ("block_gate", "narrow identity-block eligibility", "graph.py", "P12-BLOCK-ELIGIBILITY"),
        ("fixed_width", "stable logical slots and explicit transitions", "deft.py", "P12-FIXED-WIDTH-DEFT"),
        ("core_ownership", "complete masks optimizer reset and BN policy", "deft.py", "P12-CORE-STATE-INVARIANCE"),
        ("local_scope", "local fixtures without trained-network claims", "run.py", "P12-GUARANTEE-SCOPE"),
    )
    return [
        _row(
            claim_id,
            contract_id=contract_id,
            requirement=requirement,
            implementation_source=f"src/hope_repro/proposed_repairs/hope_r/{source}"
            if source != "run.py"
            else "experiments/audits/part12_corrected_hope/run.py",
            scope_value=1.0,
            passed=True,
        )
        for contract_id, requirement, source, claim_id in specs
    ]


def _integration_rows(
    common: Mapping[str, Any],
    merge: Mapping[str, Any],
    graph: Mapping[str, Any],
    deft: Mapping[str, Any],
) -> list[dict[str, Any]]:
    measure = common["measure"]
    coarse = sum_map_optimized_merge(
        merge["noncollinear_first"],
        merge["noncollinear_second"],
        measure,
        angular_samples=4,
    )
    fine = merge["noncollinear_pair"]
    approximation_gap = abs(
        float(common["exact_cross"]) - float(common["zero_bias_approximation"])
    )

    all_trainable = {
        name: np.ones(mask.shape, dtype=np.float64)
        for name, mask in deft["masks"].items()
    }
    unmasked_step = momentumw_step(
        deft["parameters"],
        deft["gradients"],
        deft["optimizer_state"],
        all_trainable,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=0.2,
    )
    core_drift_without_mask = max(
        float(
            np.max(
                np.abs(
                    unmasked_step.parameters[name][0]
                    - deft["parameters"][name][0]
                )
            )
        )
        for name in deft["parameters"]
    )
    specs = (
        (
            "exact_cross_vs_zero_bias",
            "calibrated_exact_cross",
            "zero_bias_angular_approximation",
            float(common["exact_cross"]),
            float(common["zero_bias_approximation"]),
            approximation_gap,
            "nonzero_declared_boundary_gap",
            approximation_gap > 1e-6,
            "P12-KERNEL-BOUNDARY-CONTRACT",
        ),
        (
            "merge_search_resolution",
            "anchored_128_angle_search",
            "anchored_4_angle_search",
            fine.absolute_residual,
            coarse.absolute_residual,
            abs(fine.absolute_residual - coarse.absolute_residual),
            "fine_residual_not_greater",
            fine.absolute_residual <= coarse.absolute_residual + 1e-14,
            "P12-SUM-MAP-MERGE",
        ),
        (
            "dependency_refresh_scope",
            "complete_dependency_closure",
            "current_layer_only",
            0.0,
            float(graph["local_only_stale_values"]),
            float(graph["local_only_stale_values"]),
            "local_only_leaves_stale_values",
            graph["local_only_stale_values"] > 0,
            "P12-REFRESH-EQUIVALENCE",
        ),
        (
            "payoff_state",
            "live_physical_payoff",
            "initial_static_payoff",
            float(graph["live_payoff"]),
            float(graph["static_payoff"]),
            abs(float(graph["live_payoff"] - graph["static_payoff"])),
            "static_and_live_are_distinct",
            graph["live_payoff"] != graph["static_payoff"],
            "P12-LIVE-RESOURCE-ACCOUNTING",
        ),
        (
            "core_update_mask",
            "source_owned_mask",
            "all_parameters_trainable",
            float(deft["masked_step"].max_abs_masked_drift),
            core_drift_without_mask,
            abs(core_drift_without_mask - deft["masked_step"].max_abs_masked_drift),
            "mask_prevents_observed_core_drift",
            deft["masked_step"].max_abs_masked_drift == 0.0
            and core_drift_without_mask > 0.0,
            "P12-CORE-STATE-INVARIANCE",
        ),
    )
    return [
        _row(
            claim_id,
            ablation_id=ablation_id,
            baseline_method=baseline_method,
            ablated_method=ablated_method,
            baseline_metric=baseline_metric,
            ablated_metric=ablated_metric,
            absolute_delta=delta,
            criterion=criterion,
            passed=passed,
        )
        for (
            ablation_id,
            baseline_method,
            ablated_method,
            baseline_metric,
            ablated_metric,
            delta,
            criterion,
            passed,
            claim_id,
        ) in specs
    ]


def _source_rows() -> list[dict[str, Any]]:
    specs = (
        ("P12-CALIBRATED-COMMON-MEASURE", "src/hope_repro/proposed_repairs/hope_r/common.py", "common_gram_cases.csv", "PDF pages 7-9 and 35", "calibrated local geometry"),
        ("P12-COMMON-GRAM-PSD", "src/hope_repro/proposed_repairs/hope_r/common.py", "common_gram_cases.csv", "PDF pages 35 and 52", "declared deterministic fixture"),
        ("P12-KERNEL-BOUNDARY-CONTRACT", "src/hope_repro/proposed_repairs/hope_r/common.py", "common_gram_cases.csv", "PDF pages 5 and 52-56", "repair and comparison kept separate"),
        ("P12-SUM-MAP-MERGE", "src/hope_repro/proposed_repairs/hope_r/merge.py", "merge_cases.csv", "PDF pages 13-16 and 57", "measured one-parent residual"),
        ("P12-TWIN-PRESERVATION", "src/hope_repro/proposed_repairs/hope_r/merge.py", "merge_cases.csv", "PDF pages 13 and 57", "declared twin fixtures only"),
        ("P12-DEPENDENCY-CLOSURE", "src/hope_repro/proposed_repairs/hope_r/graph.py", "invalidation_cases.csv", "PDF pages 19-20 and 68", "resolved supported graph"),
        ("P12-REFRESH-EQUIVALENCE", "src/hope_repro/proposed_repairs/hope_r/graph.py", "invalidation_cases.csv", "PDF page 68", "declared cache-key fixture"),
        ("P12-LIVE-RESOURCE-ACCOUNTING", "src/hope_repro/proposed_repairs/hope_r/graph.py", "resource_accounting.csv", "PDF pages 18-20 and 38-40", "counts and bytes not latency"),
        ("P12-BLOCK-ELIGIBILITY", "src/hope_repro/proposed_repairs/hope_r/graph.py", "block_eligibility.csv", "PDF pages 16-17 and 58", "narrow identity block domain"),
        ("P12-FIXED-WIDTH-DEFT", "src/hope_repro/proposed_repairs/hope_r/deft.py", "deft_transitions.csv", "PDF pages 21-23 and 62-70", "fixed deterministic state fixture"),
        ("P12-CORE-STATE-INVARIANCE", "src/hope_repro/proposed_repairs/hope_r/deft.py", "deft_transitions.csv", "PDF pages 5 23 65-66 and 70", "local parameter and buffer state"),
        ("P12-GUARANTEE-SCOPE", "experiments/audits/part12_corrected_hope/run.py", "design_contract.csv", "PDF pages 18-24 and 68-70", "no trained-network outcome claim"),
    )
    rows = []
    for claim_id, source, output, anchor, scope in specs:
        exists = int((ROOT / source).is_file())
        rows.append(
            _row(
                claim_id,
                implementation_source=source,
                audit_output=output,
                paper_anchor=anchor,
                scope=scope,
                source_exists=exists,
                passed=exists == 1,
            )
        )
    return rows


def _validate_tables(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    if set(tables) != set(CSV_CONTRACTS):
        raise AssertionError("audit tables do not match the exact CSV output contract")
    for filename, fields in CSV_CONTRACTS.items():
        rows = tuple(tables[filename])
        if not rows:
            raise AssertionError(f"audit table must be nonempty: {filename}")
        for index, row in enumerate(rows):
            if set(row) != set(fields):
                raise AssertionError(
                    f"schema mismatch for {filename} row {index}: {tuple(row)}"
                )
            claim_id = str(row["claim_id"])
            if claim_id not in CLAIM_CLASSES:
                raise AssertionError(f"unknown claim ID in {filename}: {claim_id}")
            if row["evidence_label"] != CLAIM_CLASSES[claim_id]:
                raise AssertionError(f"evidence label mismatch in {filename}: {claim_id}")
            if row["passed"] is not True:
                raise AssertionError(f"failing row in {filename}: {row}")
            for field in NUMERIC_FIELDS[filename]:
                value = float(row[field])
                if not math.isfinite(value):
                    raise AssertionError(
                        f"non-finite {field} in {filename} row {index}"
                    )


def _paper_identity(*, pdf_supplied: bool = False) -> dict[str, Any]:
    payload = json.loads(PAPER_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "arxiv_id": payload["arxiv_id"],
        "version": payload["version"],
        "pdf_filename": payload["pdf_filename"],
        "pdf_sha256": payload["pdf_sha256"],
        "pdf_bytes": PAPER_BYTES,
        "pdf_supplied": bool(pdf_supplied),
        "pages": payload["pages"],
    }


def _authenticate_paper(paper_pdf: Path | None, *, full: bool) -> bool:
    if full and paper_pdf is None:
        raise ValueError("--full requires --paper-pdf")
    if paper_pdf is None:
        return False
    if not paper_pdf.is_file():
        raise FileNotFoundError(f"paper PDF does not exist: {paper_pdf}")
    expected = _paper_identity()["pdf_sha256"]
    observed = hashlib.sha256(paper_pdf.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError("paper PDF has the wrong identity")
    return True


def _walk_numbers(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    if isinstance(value, Mapping):
        return [
            number
            for child in value.values()
            for number in _walk_numbers(child)
        ]
    if isinstance(value, (list, tuple)):
        return [number for child in value for number in _walk_numbers(child)]
    return []


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def build_config(*, full: bool = False) -> dict[str, Any]:
    """Return the complete deterministic fixture configuration."""

    return {
        "full_mode": bool(full),
        "fixture_version": FIXTURE_VERSION,
        "numeric_engine": "NumPy float64 and Python finite scalar checks",
        "angular_samples": ANGULAR_SAMPLES,
        "calibration_measure": "part12_weighted_fixture",
        "calibration_example_count": 9,
        "calibration_raw_weights": [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 2.0, 3.0],
        "graph_fixture": "stem_to_mid_to_left_and_right_complete_fanout",
        "deft_fixture": "two_fixed_width_layers_with_stable_slot_ids",
        "random_sampling": False,
    }


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def build_audit(*, full: bool = False) -> dict[str, Any]:
    """Build every deterministic table and its summary without writing files."""

    common_rows, common = _common_geometry()
    merge_rows, merge = _merge_geometry(common["measure"])
    invalidation_rows, resource_rows, graph = _graph_geometry()
    block_rows = _block_rows(graph["graph"])
    deft_rows, deft = _deft_geometry()
    tables: dict[str, list[dict[str, Any]]] = {
        "design_contract.csv": _design_rows(),
        "common_gram_cases.csv": common_rows,
        "merge_cases.csv": merge_rows,
        "invalidation_cases.csv": invalidation_rows,
        "resource_accounting.csv": resource_rows,
        "block_eligibility.csv": block_rows,
        "deft_transitions.csv": deft_rows,
        "integration_ablations.csv": _integration_rows(common, merge, graph, deft),
        "source_map.csv": _source_rows(),
    }
    _validate_tables(tables)
    row_counts = {filename: len(rows) for filename, rows in tables.items()}
    passing_counts = {
        filename: sum(row["passed"] is True for row in rows)
        for filename, rows in tables.items()
    }
    all_rows = [row for rows in tables.values() for row in rows]
    evidence_mismatches = sum(
        row["evidence_label"] != CLAIM_CLASSES[row["claim_id"]]
        for row in all_rows
    ) + sum(
        record["evidence_label"] != CLAIM_CLASSES[record["claim_id"]]
        for record in METHOD_ORIGINS.values()
    )
    observed_values = {
        "nonfinite_count": float(
            sum(
                not math.isfinite(number)
                for number in _walk_numbers(all_rows)
            )
        ),
        "row_failure_count": float(sum(row["passed"] is not True for row in all_rows)),
        "evidence_class_mismatch_count": float(evidence_mismatches),
        "common_gram_minimum_eigenvalue": float(common["minimum_gram_eigenvalue"]),
        "common_gram_symmetry_error": float(common["gram_symmetry_error"]),
        "bn_fold_max_error": float(common["bn_fold_max_error"]),
        "kernel_zero_scale_error": float(common["kernel_zero_scale_error"]),
        "maximum_twin_residual": max(
            merge["zero_bias_twins"].absolute_residual,
            merge["biased_twins"].absolute_residual,
        ),
        "noncollinear_residual": float(merge["noncollinear_pair"].absolute_residual),
        "dependency_key_count": float(len(graph["prune_plan"].cache_keys)),
        "complete_refresh_value_mismatch_count": float(
            graph["complete_refresh_value_mismatches"]
        ),
        "local_only_stale_value_count": float(graph["local_only_stale_values"]),
        "oracle_missing_key_count": float(graph["oracle_missing_keys"]),
        "oracle_extra_key_count": float(graph["oracle_extra_keys"]),
        "independent_cache_universe_count": float(
            graph["independent_cache_universe_count"]
        ),
        "live_parameter_payoff": float(graph["live_payoff"]),
        "initial_static_parameter_payoff": float(graph["static_payoff"]),
        "current_candidate_count": float(
            graph["resource_report"].current_candidate_count
        ),
        "trajectory_candidate_evaluations": float(
            graph["resource_report"].trajectory_candidate_evaluations
        ),
        "eligible_block_count": float(
            sum(check_identity_block(block).eligible for block in graph["graph"].blocks)
        ),
        "block_rejection_case_count": float(
            sum(row["expected_eligible"] == 0 for row in block_rows)
        ),
        "fixed_width_delta": float(deft["fixed_width_delta"]),
        "stable_slot_identity_count": float(deft["stable_slot_ids_preserved"]),
        "maximum_masked_parameter_drift": float(
            deft["masked_step"].max_abs_masked_drift
        ),
        "dual_track_source_buffer_drift": float(deft["dual_source_drift"]),
        "integration_ablation_pass_count": float(
            sum(row["passed"] is True for row in tables["integration_ablations.csv"])
        ),
    }
    if tuple(observed_values) != tuple(CHECK_SPECS):
        raise AuditFailure("observed check registry changed")
    checks = {
        name: {
            "value": value,
            "minimum": CHECK_SPECS[name]["minimum"],
            "maximum": CHECK_SPECS[name]["maximum"],
            "units": CHECK_SPECS[name]["units"],
            "passed": CHECK_SPECS[name]["minimum"]
            <= value
            <= CHECK_SPECS[name]["maximum"],
        }
        for name, value in observed_values.items()
    }
    failed_checks = [name for name, record in checks.items() if not record["passed"]]
    if failed_checks:
        raise AuditFailure(f"audit checks failed: {failed_checks}")
    config = build_config(full=full)
    stem_row_counts = {
        Path(filename).stem: count for filename, count in row_counts.items()
    }
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "passed": True,
        "fixture_version": FIXTURE_VERSION,
        "paper": _paper_identity(pdf_supplied=full),
        "config": config,
        "config_sha256": _config_hash(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": REPRODUCTION_SCOPE,
        "output_filenames": list(OUTPUT_FILENAMES),
        "row_counts": stem_row_counts,
        "csv_passing_row_counts": {
            Path(filename).stem: count
            for filename, count in passing_counts.items()
        },
        "check_count": len(CHECK_SPECS),
        "passing_check_count": len(CHECK_SPECS),
        "claim_ids": list(CLAIM_IDS),
        "claim_classes": CLAIM_CLASSES,
        "headline_metrics": {
            "common_gram_minimum_eigenvalue": common["minimum_gram_eigenvalue"],
            "zero_bias_twin_absolute_residual": merge["zero_bias_twins"].absolute_residual,
            "biased_twin_absolute_residual": merge["biased_twins"].absolute_residual,
            "noncollinear_absolute_residual": merge["noncollinear_pair"].absolute_residual,
            "dependency_key_count": len(graph["prune_plan"].cache_keys),
            "complete_refresh_value_mismatch_count": graph[
                "complete_refresh_value_mismatches"
            ],
            "local_only_stale_value_count": graph["local_only_stale_values"],
            "oracle_missing_key_count": graph["oracle_missing_keys"],
            "oracle_extra_key_count": graph["oracle_extra_keys"],
            "independent_cache_universe_count": graph[
                "independent_cache_universe_count"
            ],
            "live_parameter_payoff": graph["live_payoff"],
            "initial_static_parameter_payoff": graph["static_payoff"],
            "maximum_masked_parameter_drift": deft["masked_step"].max_abs_masked_drift,
            "dual_track_source_buffer_drift": deft["dual_source_drift"],
        },
        "checks": checks,
    }
    validate_summary(summary)
    return {"summary": summary, **tables, "tables": tables}


def validate_summary(summary: Mapping[str, Any]) -> None:
    """Fail closed unless a summary matches every declared audit contract."""

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
    if summary.get("method_origins") != METHOD_ORIGINS:
        raise AuditFailure("summary method origins changed")
    if summary.get("reproduction_scope") != REPRODUCTION_SCOPE:
        raise AuditFailure("summary reproduction scope changed")
    if summary.get("paper") != _paper_identity(
        pdf_supplied=bool(summary.get("config", {}).get("full_mode", False))
    ):
        raise AuditFailure("summary paper identity changed")
    config = summary.get("config")
    if not isinstance(config, Mapping) or summary.get("config_sha256") != _config_hash(config):
        raise AuditFailure("summary configuration hash changed")
    expected_row_keys = {Path(filename).stem for filename in CSV_CONTRACTS}
    row_counts = summary.get("row_counts")
    if not isinstance(row_counts, Mapping) or set(row_counts) != expected_row_keys:
        raise AuditFailure("summary row-count registry changed")
    checks = summary.get("checks")
    if not isinstance(checks, Mapping) or tuple(checks) != tuple(CHECK_SPECS):
        raise AuditFailure("summary check registry changed")
    expected_count = len(CHECK_SPECS)
    if (
        summary.get("check_count") != expected_count
        or summary.get("passing_check_count") != expected_count
    ):
        raise AuditFailure(f"summary must contain {expected_count} passing checks")
    for name, specification in CHECK_SPECS.items():
        record = checks[name]
        if not isinstance(record, Mapping) or record.get("passed") is not True:
            raise AuditFailure(f"check is not passing: {name}")
        value = float(record.get("value"))
        if (
            not math.isfinite(value)
            or not specification["minimum"] <= value <= specification["maximum"]
        ):
            raise AuditFailure(f"check is outside declared range: {name}")
    if any(not math.isfinite(number) for number in _walk_numbers(summary)):
        raise AuditFailure("summary contains a non-finite number")


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if any(not math.isfinite(number) for number in _walk_numbers(payload)):
        raise AuditFailure(f"refusing to write non-finite JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity() -> tuple[str, str, bool]:
    """Bind the declared sources to HEAD, independent of unrelated changes."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", "unavailable", False

    matches = True
    for relative in SOURCE_RELATIVE_PATHS:
        path = ROOT / relative
        try:
            captured = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            if (
                captured.returncode != 0
                or not path.is_file()
                or path.is_symlink()
                or captured.stdout != path.read_bytes()
            ):
                matches = False
                break
        except OSError:
            matches = False
            break
    return commit, "exact_commit" if matches else "working_tree", matches


def _manifest(
    result_dir: Path,
    outputs: Sequence[Path],
    summary: Mapping[str, Any],
    paper: Mapping[str, Any],
    full: bool,
) -> dict[str, Any]:
    """Build the external provenance manifest for the ten result files."""

    commit, state, matches = _git_identity()
    source_hashes: list[dict[str, Any]] = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise AuditFailure(f"required source is missing or symlinked: {relative}")
        source_hashes.append(
            {
                "path": relative,
                "sha256": _sha256_path(path),
                "bytes": path.stat().st_size,
            }
        )

    output_rows: list[dict[str, Any]] = []
    expected_outputs = tuple(result_dir / filename for filename in OUTPUT_FILENAMES)
    if tuple(outputs) != expected_outputs:
        raise AuditFailure("manifest output order changed")
    for path in outputs:
        if not path.is_file() or path.is_symlink():
            raise AuditFailure(f"required output is missing or symlinked: {path}")
        relative = (
            path.relative_to(ROOT).as_posix()
            if path.is_relative_to(ROOT)
            else path.name
        )
        output_rows.append(
            {
                "path": relative,
                "sha256": _sha256_path(path),
                "bytes": path.stat().st_size,
                "media_type": (
                    "application/json" if path.suffix == ".json" else "text/csv"
                ),
            }
        )

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
        "input_hashes": [
            {
                "kind": "paper_manifest",
                "path": "paper/manifest.json",
                "sha256": _sha256_path(PAPER_MANIFEST_PATH),
                "bytes": PAPER_MANIFEST_PATH.stat().st_size,
            }
        ],
        "outputs": output_rows,
        "manifest_hash_scope": "manifest excludes its own hash to avoid self-reference",
        "canonical_invocation": {
            "entrypoint": "experiments/audits/part12_corrected_hope/run.py",
            "result_dir": "<RESULT_DIR>",
            "manifest_path": "<MANIFEST_PATH>",
            "paper_pdf": "<PAPER_PDF>" if paper["pdf_supplied"] else "not supplied",
            "full": bool(full),
        },
    }


def run_audit(
    *,
    result_dir: Path,
    manifest_path: Path | None = None,
    paper_pdf: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Write ten results and, optionally, one external provenance manifest."""

    # Normalize before provenance is assembled.  A repository-relative CLI path
    # must remain repository-qualified in the external manifest, while an
    # output rooted elsewhere intentionally falls back to its basename.
    destination = Path(result_dir).resolve()
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"result path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite nonempty result directory: {destination}"
            )
    external_manifest = Path(manifest_path) if manifest_path is not None else None
    if external_manifest is not None:
        if external_manifest.exists() or external_manifest.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite manifest: {external_manifest}"
            )
        if external_manifest.resolve().is_relative_to(destination.resolve()):
            raise ValueError("--manifest-path must be outside --result-dir")
    _authenticate_paper(paper_pdf, full=full)
    audit = build_audit(full=full)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, fields in CSV_CONTRACTS.items():
        _write_csv(destination / filename, fields, audit[filename])
    _write_json(destination / "summary.json", audit["summary"])
    observed = {path.name for path in destination.iterdir()}
    if observed != set(OUTPUT_FILENAMES):
        raise AssertionError("generator violated the exact ten-output contract")
    if external_manifest is not None:
        outputs = [destination / filename for filename in OUTPUT_FILENAMES]
        _write_json(
            external_manifest,
            _manifest(
                destination,
                outputs,
                audit["summary"],
                audit["summary"]["paper"],
                full,
            ),
        )
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        required=True,
        type=Path,
        help="new or empty directory for the exact ten-output audit package",
    )
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        help="optional exact HOPE v1 PDF to authenticate",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="optional external provenance manifest; the path must not exist",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="require and authenticate --paper-pdf before generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        audit = run_audit(
            result_dir=args.result_dir,
            manifest_path=args.manifest_path,
            paper_pdf=args.paper_pdf,
            full=args.full,
        )
    except (
        AssertionError,
        AuditFailure,
        FileExistsError,
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    summary = audit["summary"]
    print(
        f"PASS: {summary['passing_check_count']}/{summary['check_count']} "
        f"checks; wrote {len(OUTPUT_FILENAMES)} outputs to {args.result_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
