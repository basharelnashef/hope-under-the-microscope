#!/usr/bin/env python3
"""Fail-closed deterministic audit of HOPE v1 DEFT core-state semantics."""

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
AUDIT_ID = "part10_deft_core_audit"
AUDIT_SCHEMA_VERSION = 1
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
PAPER_BYTES = 1_089_944
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_10" / "manual"

PERCENTILE = 50.0
TRAJECTORY_THRESHOLD = 0.75
PARTITION_FIXTURE = {
    "A": (
        {"neuron": "A0", "capacity": 1.0, "N": 3.0, "E_b": 10.0, "step": 1},
        {"neuron": "A1", "capacity": 2.0, "N": 2.0, "E_b": 8.0, "step": 2},
        {"neuron": "A2", "capacity": 8.0, "N": None, "E_b": None, "step": None},
    ),
    "B": (
        {"neuron": "B0", "capacity": 3.0, "N": 3.0, "E_b": 9.0, "step": 3},
        {"neuron": "B1", "capacity": 4.0, "N": 2.0, "E_b": 5.0, "step": 4},
        {"neuron": "B2", "capacity": 5.0, "N": None, "E_b": None, "step": None},
    ),
}
LAYER_THRESHOLDS = {"A": 2.0, "B": 4.0}
GLOBAL_CAPACITY_THRESHOLD = 3.5

ROUTING_SOURCE = np.asarray([[2.0, 3.0], [5.0, 7.0]], dtype=np.float64)
ROUTING_MASK = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
ROUTING_GRADIENT = np.asarray([[99.0, 99.0], [-2.0, -2.0]], dtype=np.float64)
DESTINATION_ELASTICITY = np.asarray([0.0, 1.0], dtype=np.float64)
LEARNING_RATE = 0.5
EXPECTED_ROUTING_AFTER = np.asarray([[2.0, 0.0], [6.0, 8.0]], dtype=np.float64)
SOURCE_ACTIVATION = np.asarray([1.0, 1.0], dtype=np.float64)
TARGET_ACTIVATION = np.asarray([1.0, 4.0], dtype=np.float64)

BN_UPDATE_RATE = 0.5
BN_EPSILON = 1e-5
BN_INITIAL_MEAN = 0.0
BN_INITIAL_VARIANCE = 1.0
BN_TARGET_BATCHES = (
    np.asarray([2.0, 6.0], dtype=np.float64),
    np.asarray([3.0, 9.0], dtype=np.float64),
)
BN_SOURCE_PROBES = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)

REPAIR_INPUT = 1.0
REPAIR_TARGET = 1.0
REPAIR_LEARNING_RATE = 0.25

CLAIM_IDS = (
    "P10-TRAJECTORY-PARTITION",
    "P10-LAYER-THRESHOLD-PARTITION",
    "P10-CURRENT-CAPACITY-PARTITION",
    "DEFT-PARTITION-MISMATCH",
    "P10-NARROW-ROUTING-GUARANTEE",
    "P10-BN-BUFFER-DRIFT",
    "P10-BN-STATE-REPAIR",
    "P10-OUTGOING-OPERATOR-DRIFT",
    "DEFT-CORE-DRIFT",
    "P10-PURGED-RELEASED-CONFLICT",
    "P10-DEAD-ZERO-SLOT",
    "P10-FIXED-WIDTH-REPAIR",
    "P10-SOURCE-STATE-RESTORATION",
)
CLAIM_CLASSES = {
    "P10-TRAJECTORY-PARTITION": "paper_claim",
    "P10-LAYER-THRESHOLD-PARTITION": "paper_claim",
    "P10-CURRENT-CAPACITY-PARTITION": "paper_claim",
    "DEFT-PARTITION-MISMATCH": "verified_fact",
    "P10-NARROW-ROUTING-GUARANTEE": "verified_fact",
    "P10-BN-BUFFER-DRIFT": "counterexample",
    "P10-BN-STATE-REPAIR": "proposed_repair",
    "P10-OUTGOING-OPERATOR-DRIFT": "counterexample",
    "DEFT-CORE-DRIFT": "verified_fact",
    "P10-PURGED-RELEASED-CONFLICT": "verified_fact",
    "P10-DEAD-ZERO-SLOT": "counterexample",
    "P10-FIXED-WIDTH-REPAIR": "proposed_repair",
    "P10-SOURCE-STATE-RESTORATION": "paper_claim",
}

METHOD_ORIGINS = {
    "trajectory_prune_cost_partition": {
        "evidence_label": "paper_claim",
        "claim_id": "P10-TRAJECTORY-PARTITION",
        "origin": "HOPE v1 Eqs. 24-27 on PDF pages 21-22",
        "scope": "global percentile of recorded trajectory pruning costs",
    },
    "per_layer_capacity_partition": {
        "evidence_label": "paper_claim",
        "claim_id": "P10-LAYER-THRESHOLD-PARTITION",
        "origin": "HOPE v1 Definition H.1 and Eqs. 103-104 on PDF page 63",
        "scope": "declared per-layer capacity thresholds",
    },
    "global_current_capacity_partition": {
        "evidence_label": "paper_claim",
        "claim_id": "P10-CURRENT-CAPACITY-PARTITION",
        "origin": "HOPE v1 Algorithm 3 lines 1-9 on PDF page 70",
        "scope": "global current-capacity percentile after merging",
    },
    "destination_elasticity_routing": {
        "evidence_label": "verified_fact",
        "claim_id": "P10-NARROW-ROUTING-GUARANTEE",
        "origin": "independent execution of Eqs. 28-29 on a fixed 2 by 2 routing matrix",
        "scope": "parameterized feed-forward edges in the declared fixture",
    },
    "batchnorm_state_counterexample": {
        "evidence_label": "counterexample",
        "claim_id": "P10-BN-BUFFER-DRIFT",
        "origin": "independent NumPy training-mode running-state fixture",
        "scope": "fixed trainable parameters and one BatchNorm channel",
    },
    "batchnorm_state_repair": {
        "evidence_label": "proposed_repair",
        "claim_id": "P10-BN-STATE-REPAIR",
        "origin": "independent frozen-buffer control on the declared source probes",
        "scope": "source BN-output RMSE only; no target-domain or trained-network claim",
    },
    "complete_operator_counterexample": {
        "evidence_label": "counterexample",
        "claim_id": "P10-OUTGOING-OPERATOR-DRIFT",
        "origin": "independent destination-gated outgoing-vector fixture",
        "scope": "paper Eq. 2 complete operator with a core-to-slack route",
    },
    "fixed_width_released_slot": {
        "evidence_label": "proposed_repair",
        "claim_id": "P10-FIXED-WIDTH-REPAIR",
        "origin": "independent fixed-width reinitialization control",
        "scope": "one scalar ReLU slot with zero initial outgoing contribution",
    },
}

REPRODUCTION_SCOPE = {
    "reproduction_scope": "independent_deft_state_semantics_audit",
    "paper_experiment_reproduced": False,
    "checkpoint_or_dataset_used": False,
    "random_sampling_used": False,
    "torch_used": False,
    "wall_clock_benchmark_used": False,
    "includes": [
        "source map for Section 11.2.2, Appendix G.5, Appendix H, and Algorithms 2-3",
        "three separately named paper partition rules on one fixed spectrum",
        "structural-mask and destination-elasticity routing transition",
        "BatchNorm running-buffer drift with frozen trainable parameters",
        "complete core-operator drift through a core-to-slack outgoing edge",
        "purged-versus-released lifecycle comparison",
        "dead zero-slot gradient and fixed-width proposed-repair control",
        "live, restored, and buffer-frozen source-state controls",
    ],
    "excludes": [
        "the paper CIFAR-100 to SVHN DEFT training protocol",
        "dataset, checkpoint, trained-network prevalence, or optimizer convergence",
        "trained-network final logits, end-task loss, accuracy, wall-clock, or resource claims",
        "a complete corrected DEFT implementation",
        "Parts 11 and 12",
    ],
    "claim_boundary": (
        "This audit makes specification conflicts and local state transitions executable. "
        "It does not reproduce the paper's DEFT experiment or estimate downstream performance."
    ),
}

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part10_deft_core/README.md",
    "experiments/audits/part10_deft_core/run.py",
    "experiments/audits/part10_deft_core/test_run.py",
)

SOURCE_MAP_FIELDS = (
    "step_id", "sequence", "paper_anchor", "object", "formula_or_action",
    "result", "evidence_label", "claim_id", "passed",
)
PARTITION_ASSIGNMENT_FIELDS = (
    "rule_id", "paper_anchor", "layer", "neuron", "capacity",
    "trajectory_n", "trajectory_e_b", "trajectory_cost", "trajectory_step",
    "terminal_survivor", "threshold_kind", "threshold", "assignment",
    "evidence_label", "claim_id", "passed",
)
PARTITION_DISAGREEMENT_FIELDS = (
    "rule_a", "rule_b", "comparison_scope", "disagreement_count",
    "disagreement_neurons", "evidence_label", "claim_id", "passed",
)
ROUTING_TRANSITION_FIELDS = (
    "upstream_index", "upstream_role", "downstream_index", "downstream_role",
    "source_weight", "mask", "masked_weight", "destination_elasticity",
    "raw_gradient", "effective_gradient", "weight_after", "expected_weight_after",
    "dynamic_change", "enters_core", "source_activation", "target_activation",
    "source_core_input_before_mask", "source_core_input_after_mask",
    "target_core_input_after_update", "initialization_shock", "core_dynamic_drift",
    "evidence_label", "claim_id", "passed",
)
BN_BUFFER_FIELDS = (
    "mode", "batch_index", "batch_mean", "batch_variance", "running_mean_before",
    "running_mean_after", "running_variance_before", "running_variance_after",
    "trainable_parameter_change_norm", "source_probe_rmse_before",
    "source_probe_rmse_after", "buffers_updated",
    "evidence_label", "claim_id", "passed",
)
OPERATOR_DRIFT_FIELDS = (
    "upstream_neuron", "role", "activation_second_moment",
    "outgoing_to_core_before", "outgoing_to_core_after",
    "outgoing_to_slack_before", "outgoing_to_slack_after", "outgoing_vector_before",
    "outgoing_vector_after", "operator_drift_norm", "capacity_before",
    "capacity_after", "relative_capacity_change", "incoming_parameter_change",
    "bn_affine_parameter_change", "bn_buffer_change", "evidence_label", "claim_id",
    "passed",
)
RELEASED_SLOT_FIELDS = (
    "variant", "origin", "active_layer_status", "physical_storage_status",
    "reentry_transition_specified", "role", "input_value", "target_value",
    "incoming_u", "scale_gamma", "shift_beta", "outgoing_v", "initial_hidden",
    "initial_output", "initial_residual", "first_gradient_u",
    "first_gradient_gamma", "first_gradient_beta", "first_gradient_v",
    "outgoing_v_after_first_step", "second_output", "second_residual",
    "second_gradient_u", "second_gradient_gamma", "second_gradient_beta",
    "second_gradient_v", "second_internal_gradient_min", "reusable",
    "evidence_label", "claim_id", "passed",
)
SOURCE_STATE_FIELDS = (
    "state_id", "bn_state_origin", "buffers_frozen_during_target",
    "task_specific_restore", "running_mean", "running_variance", "probe_values",
    "reference_outputs", "observed_outputs", "trainable_parameter_change_norm",
    "source_rmse", "evidence_label", "claim_id", "passed",
)

CSV_CONTRACTS = {
    "source_map.csv": SOURCE_MAP_FIELDS,
    "partition_assignments.csv": PARTITION_ASSIGNMENT_FIELDS,
    "partition_disagreements.csv": PARTITION_DISAGREEMENT_FIELDS,
    "routing_transitions.csv": ROUTING_TRANSITION_FIELDS,
    "bn_buffer_drift.csv": BN_BUFFER_FIELDS,
    "operator_drift.csv": OPERATOR_DRIFT_FIELDS,
    "released_slot_gradients.csv": RELEASED_SLOT_FIELDS,
    "source_state_controls.csv": SOURCE_STATE_FIELDS,
}
OUTPUT_FILENAMES = ("summary.json", *CSV_CONTRACTS.keys())

CHECK_SPECS = {
    "nonfinite_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "row_failure_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "evidence_class_mismatch_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
    "trajectory_threshold": {"minimum": 0.75, "maximum": 0.75, "units": "cost"},
    "global_capacity_threshold": {"minimum": 3.5, "maximum": 3.5, "units": "capacity"},
    "trajectory_unassigned_count": {"minimum": 2.0, "maximum": 2.0, "units": "neurons"},
    "trajectory_vs_layer_disagreement_count": {"minimum": 2.0, "maximum": 2.0, "units": "neurons"},
    "trajectory_vs_global_disagreement_count": {"minimum": 1.0, "maximum": 1.0, "units": "neurons"},
    "layer_vs_global_disagreement_count": {"minimum": 1.0, "maximum": 1.0, "units": "neurons"},
    "partition_signature_count": {"minimum": 3.0, "maximum": 3.0, "units": "signatures"},
    "masked_slack_to_core_weight": {"minimum": 0.0, "maximum": 0.0, "units": "weight"},
    "core_dynamic_drift": {"minimum": 0.0, "maximum": 0.0, "units": "activation"},
    "initialization_shock": {"minimum": 3.0, "maximum": 3.0, "units": "activation"},
    "core_to_slack_weight_change": {"minimum": 1.0, "maximum": 1.0, "units": "weight"},
    "operator_drift_norm": {"minimum": 1.0, "maximum": 1.0, "units": "l2"},
    "relative_capacity_change": {
        "minimum": 0.17444043902940712,
        "maximum": 0.17444043902940712,
        "units": "fraction",
    },
    "trainable_parameter_change_norm": {"minimum": 0.0, "maximum": 0.0, "units": "l2"},
    "final_running_mean": {"minimum": 4.0, "maximum": 4.0, "units": "value"},
    "final_running_variance": {"minimum": 5.75, "maximum": 5.75, "units": "value"},
    "live_source_rmse": {
        "minimum": 1.7346960857596274,
        "maximum": 1.7346960857596274,
        "units": "rmse",
    },
    "zero_slot_gradient_norm": {"minimum": 0.0, "maximum": 0.0, "units": "l2"},
    "repair_initial_output": {"minimum": 0.0, "maximum": 0.0, "units": "output"},
    "repair_first_outgoing_gradient_magnitude": {"minimum": 1.0, "maximum": 1.0, "units": "absolute"},
    "repair_second_internal_gradient_min": {"minimum": 0.1875, "maximum": 0.1875, "units": "absolute"},
    "restored_source_rmse": {"minimum": 0.0, "maximum": 0.0, "units": "rmse"},
    "frozen_source_rmse": {"minimum": 0.0, "maximum": 0.0, "units": "rmse"},
}


class AuditFailure(RuntimeError):
    """Raised when a declared Part 10 invariant fails."""


def _trajectory_cost(record: Mapping[str, Any]) -> float | None:
    numerator = record.get("N")
    energy_after = record.get("E_b")
    if numerator is None and energy_after is None:
        return None
    if numerator is None or energy_after is None:
        raise AuditFailure("trajectory N and E_b must both be recorded or both be open")
    numerator = float(numerator)
    energy_after = float(energy_after)
    capacity = float(record["capacity"])
    if numerator <= 0.0 or energy_after <= 0.0 or capacity <= 0.0:
        raise AuditFailure("recorded trajectory factors must be positive")
    return numerator * capacity / energy_after


def _trajectory_costs_by_neuron() -> dict[str, float | None]:
    costs: dict[str, float | None] = {}
    observed_steps: list[int] = []
    for layer, records in PARTITION_FIXTURE.items():
        remaining_capacity = float(sum(float(record["capacity"]) for record in records))
        recorded = sorted(
            (record for record in records if record["step"] is not None),
            key=lambda record: int(record["step"]),
        )
        for removal_index, record in enumerate(recorded):
            observed_steps.append(int(record["step"]))
            expected_n = len(records) - removal_index
            if float(record["N"]) != float(expected_n):
                raise AuditFailure(
                    f"trajectory N is not the active layer width: {record['neuron']}"
                )
            remaining_capacity -= float(record["capacity"])
            if float(record["E_b"]) != remaining_capacity:
                raise AuditFailure(
                    f"trajectory E_b is not post-removal layer capacity: {record['neuron']}"
                )
            costs[str(record["neuron"])] = _trajectory_cost(record)
        terminal = [record for record in records if record["step"] is None]
        if len(terminal) != 1 or remaining_capacity != float(terminal[0]["capacity"]):
            raise AuditFailure(f"layer {layer} must end with one open survivor")
        costs[str(terminal[0]["neuron"])] = _trajectory_cost(terminal[0])
    if sorted(observed_steps) != [1, 2, 3, 4]:
        raise AuditFailure("trajectory removal steps must be exactly 1 through 4")
    return costs


def _derived_global_thresholds() -> tuple[float, float]:
    trajectory_costs = [
        float(cost) for cost in _trajectory_costs_by_neuron().values()
        if cost is not None
    ]
    capacities = [
        float(record["capacity"])
        for records in PARTITION_FIXTURE.values()
        for record in records
    ]
    trajectory = float(np.median(np.asarray(trajectory_costs, dtype=np.float64)))
    capacity = float(np.median(np.asarray(capacities, dtype=np.float64)))
    if trajectory != TRAJECTORY_THRESHOLD or capacity != GLOBAL_CAPACITY_THRESHOLD:
        raise AuditFailure("derived partition medians differ from the fixed fixture contract")
    return trajectory, capacity


def build_config(*, full: bool) -> dict[str, Any]:
    trajectory_threshold, global_capacity_threshold = _derived_global_thresholds()
    return {
        "float_dtype": "float64",
        "full_mode": bool(full),
        "random_sampling": False,
        "torch_used": False,
        "percentile": PERCENTILE,
        "percentile_method": (
            "float64 median; midpoint of the two central values for even counts"
        ),
        "partition_fixture": {
            layer: [dict(record) for record in records]
            for layer, records in PARTITION_FIXTURE.items()
        },
        "partition_thresholds": {
            "trajectory_cost_global": trajectory_threshold,
            "definition_h1_per_layer": dict(LAYER_THRESHOLDS),
            "algorithm3_current_capacity_global": global_capacity_threshold,
            "global_threshold_derivation": "float64 median of the declared fixture values",
        },
        "routing_fixture": {
            "source_weights": ROUTING_SOURCE.tolist(),
            "structural_mask": ROUTING_MASK.tolist(),
            "raw_gradient": ROUTING_GRADIENT.tolist(),
            "destination_elasticity": DESTINATION_ELASTICITY.tolist(),
            "learning_rate": LEARNING_RATE,
            "source_activation": SOURCE_ACTIVATION.tolist(),
            "target_activation": TARGET_ACTIVATION.tolist(),
        },
        "batchnorm_fixture": {
            "update_rate": BN_UPDATE_RATE,
            "epsilon": BN_EPSILON,
            "initial_mean": BN_INITIAL_MEAN,
            "initial_variance": BN_INITIAL_VARIANCE,
            "target_batches": [batch.tolist() for batch in BN_TARGET_BATCHES],
            "source_probes": BN_SOURCE_PROBES.tolist(),
            "variance_convention": "population variance",
        },
        "released_slot_fixture": {
            "input": REPAIR_INPUT,
            "target": REPAIR_TARGET,
            "learning_rate": REPAIR_LEARNING_RATE,
            "initial_state": {"u": 1.0, "gamma": 1.0, "beta": 0.0, "v": 0.0},
            "loss": "half squared error",
            "relu_derivative_at_zero": 0.0,
        },
    }


def _source_map_rows() -> list[dict[str, Any]]:
    specifications = (
        (
            "trajectory_cost", "Eqs. 24-27, PDF pages 21-22", "main-text partition",
            "compute J=N*c/E_b at each recorded removal, then take one global percentile",
            "the rule makes a partial assignment on four recorded removals; terminal survivors stay open",
            "paper_claim", "P10-TRAJECTORY-PARTITION",
        ),
        (
            "layer_threshold", "Definition H.1 and Eqs. 103-104, PDF page 63",
            "theorem partition", "compare current capacity with tau^(l) inside each layer",
            "Appendix H assumes a layer-indexed threshold rather than Eq. 24 costs",
            "paper_claim", "P10-LAYER-THRESHOLD-PARTITION",
        ),
        (
            "current_capacity", "Algorithm 3 lines 1-9, PDF page 70",
            "pseudocode partition", "global percentile of current capacities after merging",
            "the pseudocode does not reconstruct the trajectory costs in Eqs. 24-27",
            "paper_claim", "P10-CURRENT-CAPACITY-PARTITION",
        ),
        (
            "partition_comparison", "Eqs. 24-27; Definition H.1; Algorithm 3",
            "shared fixture", "execute all three printed rules on one declared spectrum",
            "the three signatures differ on their common four-neuron domain",
            "verified_fact", "DEFT-PARTITION-MISMATCH",
        ),
        (
            "structural_mask", "Eq. 28, PDF page 22", "slack-to-core route",
            "zero a weight from upstream slack to downstream core",
            "the fixed routing fixture preserves that zero during the gated update",
            "verified_fact", "P10-NARROW-ROUTING-GUARANTEE",
        ),
        (
            "destination_gate", "Eq. 29, PDF page 23", "gradient route",
            "scale each weight gradient by the destination neuron's elasticity",
            "a core-to-slack weight remains plastic because its destination is slack",
            "counterexample", "P10-OUTGOING-OPERATOR-DRIFT",
        ),
        (
            "bn_state", "Theorem H.2, PDF pages 65-66", "claimed core state",
            "freeze incoming weights, bias, gamma, and beta through zero elasticity",
            "gradient masking does not freeze forward-pass BatchNorm running buffers",
            "counterexample", "P10-BN-BUFFER-DRIFT",
        ),
        (
            "bn_state_repair", "Independent proposed repair", "source BN buffers",
            "freeze source buffers during target adaptation or maintain explicit source and target buffer tracks",
            "the frozen-buffer control has zero BN-output RMSE on declared source probes only",
            "proposed_repair", "P10-BN-STATE-REPAIR",
        ),
        (
            "complete_operator", "Eq. 2; Eq. 29; Theorem H.2", "functional neuron",
            "include the full outgoing vector in the operator f_i",
            "BN buffers and core-to-slack outgoing weights provide two drift channels",
            "verified_fact", "DEFT-CORE-DRIFT",
        ),
        (
            "purged_child", "Algorithm 2 line 14, PDF page 69", "merged child j",
            "remove or purge active child j while child i holds the parent",
            "retention, recreation, physical storage, and reentry are left unspecified",
            "verified_fact", "P10-PURGED-RELEASED-CONFLICT",
        ),
        (
            "released_child", "Axiom 3, PDF page 64; Proposition H.4, pages 66-67",
            "merged child vessel", "assign full plasticity to structurally released children",
            "this requires a slot lifecycle different from Algorithm 2's purge",
            "verified_fact", "P10-PURGED-RELEASED-CONFLICT",
        ),
        (
            "dead_zero_slot", "Independent released-slot fixture", "zeroed ReLU vessel",
            "set incoming, bias, and outgoing values to zero",
            "both sides receive zero gradient at the declared point",
            "counterexample", "P10-DEAD-ZERO-SLOT",
        ),
        (
            "fixed_width_repair", "Independent proposed repair", "retained vessel",
            "initialize a live incoming feature with zero outgoing source contribution",
            "the outgoing slack edge receives a nonzero first-step gradient",
            "proposed_repair", "P10-FIXED-WIDTH-REPAIR",
        ),
        (
            "source_restoration", "Appendix G.5, PDF page 62; Algorithm 3 lines 27-30",
            "source evaluation state", "restore source BN state and mask slack drift",
            "the reported source path uses task-specific stored state",
            "paper_claim", "P10-SOURCE-STATE-RESTORATION",
        ),
    )
    return [
        {
            "step_id": step_id,
            "sequence": sequence,
            "paper_anchor": anchor,
            "object": object_name,
            "formula_or_action": action,
            "result": result,
            "evidence_label": evidence_label,
            "claim_id": claim_id,
            "passed": True,
        }
        for sequence, (
            step_id, anchor, object_name, action, result, evidence_label, claim_id
        ) in enumerate(specifications, start=1)
    ]


def _partition_assignments() -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    trajectory_threshold, global_capacity_threshold = _derived_global_thresholds()
    trajectory_costs = _trajectory_costs_by_neuron()
    rule_specs = (
        (
            "eq24_27_trajectory_cost", "Eqs. 24-27, PDF pages 21-22",
            "global trajectory cost", "P10-TRAJECTORY-PARTITION",
        ),
        (
            "definition_h1_layer_capacity", "Definition H.1, PDF page 63",
            "per-layer current capacity", "P10-LAYER-THRESHOLD-PARTITION",
        ),
        (
            "algorithm3_global_capacity", "Algorithm 3 lines 1-9, PDF page 70",
            "global current capacity", "P10-CURRENT-CAPACITY-PARTITION",
        ),
    )
    rows: list[dict[str, Any]] = []
    assignments: dict[str, dict[str, str]] = {rule[0]: {} for rule in rule_specs}
    for rule_id, anchor, threshold_kind, claim_id in rule_specs:
        for layer, records in PARTITION_FIXTURE.items():
            for record in records:
                cost = trajectory_costs[record["neuron"]]
                terminal = cost is None
                if rule_id == "eq24_27_trajectory_cost":
                    threshold = trajectory_threshold
                    assignment = (
                        "unassigned" if terminal
                        else "slack" if float(cost) < threshold else "core"
                    )
                    expected = {
                        "A0": "slack", "A1": "slack", "A2": "unassigned",
                        "B0": "core", "B1": "core", "B2": "unassigned",
                    }[record["neuron"]]
                elif rule_id == "definition_h1_layer_capacity":
                    threshold = LAYER_THRESHOLDS[layer]
                    assignment = "core" if record["capacity"] > threshold else "slack"
                    expected = {
                        "A0": "slack", "A1": "slack", "A2": "core",
                        "B0": "slack", "B1": "slack", "B2": "core",
                    }[record["neuron"]]
                else:
                    threshold = global_capacity_threshold
                    assignment = "core" if record["capacity"] > threshold else "slack"
                    expected = {
                        "A0": "slack", "A1": "slack", "A2": "core",
                        "B0": "slack", "B1": "core", "B2": "core",
                    }[record["neuron"]]
                assignments[rule_id][record["neuron"]] = assignment
                rows.append({
                    "rule_id": rule_id,
                    "paper_anchor": anchor,
                    "layer": layer,
                    "neuron": record["neuron"],
                    "capacity": record["capacity"],
                    "trajectory_n": "not_recorded" if terminal else record["N"],
                    "trajectory_e_b": "not_recorded" if terminal else record["E_b"],
                    "trajectory_cost": "not_recorded" if terminal else cost,
                    "trajectory_step": "terminal" if terminal else record["step"],
                    "terminal_survivor": terminal,
                    "threshold_kind": threshold_kind,
                    "threshold": threshold,
                    "assignment": assignment,
                    "evidence_label": CLAIM_CLASSES[claim_id],
                    "claim_id": claim_id,
                    "passed": assignment == expected,
                })
    return rows, assignments


def _partition_disagreements(
    assignments: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    comparisons = (
        (
            "eq24_27_trajectory_cost", "definition_h1_layer_capacity",
            ("A0", "A1", "B0", "B1"), ("B0", "B1"),
        ),
        (
            "eq24_27_trajectory_cost", "algorithm3_global_capacity",
            ("A0", "A1", "B0", "B1"), ("B0",),
        ),
        (
            "definition_h1_layer_capacity", "algorithm3_global_capacity",
            ("A0", "A1", "A2", "B0", "B1", "B2"), ("B1",),
        ),
    )
    rows: list[dict[str, Any]] = []
    for rule_a, rule_b, domain, expected in comparisons:
        observed = tuple(
            neuron for neuron in domain
            if assignments[rule_a][neuron] != assignments[rule_b][neuron]
        )
        rows.append({
            "rule_a": rule_a,
            "rule_b": rule_b,
            "comparison_scope": (
                "common four-neuron recorded-trajectory domain"
                if rule_a == "eq24_27_trajectory_cost"
                else "all six neurons assigned by both current-capacity rules"
            ),
            "disagreement_count": len(observed),
            "disagreement_neurons": "|".join(observed),
            "evidence_label": "verified_fact",
            "claim_id": "DEFT-PARTITION-MISMATCH",
            "passed": observed == expected,
        })
    return rows


def _routing_rows() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    masked = ROUTING_SOURCE * ROUTING_MASK
    effective_gradient = ROUTING_GRADIENT * DESTINATION_ELASTICITY[:, None]
    after = masked - LEARNING_RATE * effective_gradient
    source_core_before_mask = float(ROUTING_SOURCE[0] @ SOURCE_ACTIVATION)
    source_core_after_mask = float(masked[0] @ SOURCE_ACTIVATION)
    target_core_after_update = float(after[0] @ TARGET_ACTIVATION)
    initialization_shock = abs(source_core_before_mask - source_core_after_mask)
    core_dynamic_drift = abs(target_core_after_update - source_core_after_mask)
    roles = ("core", "slack")
    rows: list[dict[str, Any]] = []
    for downstream in range(2):
        for upstream in range(2):
            expected = EXPECTED_ROUTING_AFTER[downstream, upstream]
            enters_core = downstream == 0
            rows.append({
                "upstream_index": upstream,
                "upstream_role": roles[upstream],
                "downstream_index": downstream,
                "downstream_role": roles[downstream],
                "source_weight": ROUTING_SOURCE[downstream, upstream],
                "mask": ROUTING_MASK[downstream, upstream],
                "masked_weight": masked[downstream, upstream],
                "destination_elasticity": DESTINATION_ELASTICITY[downstream],
                "raw_gradient": ROUTING_GRADIENT[downstream, upstream],
                "effective_gradient": effective_gradient[downstream, upstream],
                "weight_after": after[downstream, upstream],
                "expected_weight_after": expected,
                "dynamic_change": after[downstream, upstream] - masked[downstream, upstream],
                "enters_core": enters_core,
                "source_activation": SOURCE_ACTIVATION[upstream],
                "target_activation": TARGET_ACTIVATION[upstream],
                "source_core_input_before_mask": source_core_before_mask,
                "source_core_input_after_mask": source_core_after_mask,
                "target_core_input_after_update": target_core_after_update,
                "initialization_shock": initialization_shock,
                "core_dynamic_drift": core_dynamic_drift,
                "evidence_label": "verified_fact",
                "claim_id": "P10-NARROW-ROUTING-GUARANTEE",
                "passed": bool(
                    abs(after[downstream, upstream] - expected) <= 1e-15
                    and (not (upstream == 1 and downstream == 0) or after[downstream, upstream] == 0.0)
                ),
            })
    return rows, masked, after


def _bn_output(values: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return (values - mean) / math.sqrt(variance + BN_EPSILON)


def _rmse(observed: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(observed - reference))))


def _bn_rows() -> tuple[list[dict[str, Any]], float, float]:
    rows: list[dict[str, Any]] = []
    reference_outputs = _bn_output(
        BN_SOURCE_PROBES, BN_INITIAL_MEAN, BN_INITIAL_VARIANCE
    )
    for mode in ("live_training_buffers", "frozen_buffer_control"):
        mean = BN_INITIAL_MEAN
        variance = BN_INITIAL_VARIANCE
        for index, batch in enumerate(BN_TARGET_BATCHES, start=1):
            batch_mean = float(np.mean(batch))
            batch_variance = float(np.var(batch))
            output_before = _bn_output(BN_SOURCE_PROBES, mean, variance)
            rmse_before = _rmse(output_before, reference_outputs)
            before_mean, before_variance = mean, variance
            if mode == "live_training_buffers":
                mean = (1.0 - BN_UPDATE_RATE) * mean + BN_UPDATE_RATE * batch_mean
                variance = (
                    (1.0 - BN_UPDATE_RATE) * variance
                    + BN_UPDATE_RATE * batch_variance
                )
            output_after = _bn_output(BN_SOURCE_PROBES, mean, variance)
            rmse_after = _rmse(output_after, reference_outputs)
            buffer_change = abs(mean - before_mean) + abs(variance - before_variance)
            expected_update = mode == "live_training_buffers"
            claim_id = (
                "P10-BN-BUFFER-DRIFT"
                if expected_update else "P10-BN-STATE-REPAIR"
            )
            rows.append({
                "mode": mode,
                "batch_index": index,
                "batch_mean": batch_mean,
                "batch_variance": batch_variance,
                "running_mean_before": before_mean,
                "running_mean_after": mean,
                "running_variance_before": before_variance,
                "running_variance_after": variance,
                "trainable_parameter_change_norm": 0.0,
                "source_probe_rmse_before": rmse_before,
                "source_probe_rmse_after": rmse_after,
                "buffers_updated": buffer_change > 1e-15,
                "evidence_label": CLAIM_CLASSES[claim_id],
                "claim_id": claim_id,
                "passed": (
                    (buffer_change > 1e-15 and rmse_after > rmse_before)
                    if expected_update else (buffer_change == 0.0 and rmse_after == 0.0)
                ),
            })
        if mode == "live_training_buffers":
            final_live_mean, final_live_variance = mean, variance
    return rows, final_live_mean, final_live_variance


def _operator_rows(masked: np.ndarray, after: np.ndarray) -> list[dict[str, Any]]:
    del masked
    before_vector = ROUTING_SOURCE[:, 0]
    after_vector = after[:, 0]
    capacity_before = float(np.linalg.norm(before_vector))
    capacity_after = float(np.linalg.norm(after_vector))
    drift_norm = float(np.linalg.norm(after_vector - before_vector))
    relative_capacity_change = (capacity_after - capacity_before) / capacity_before
    return [{
        "upstream_neuron": "upstream_0",
        "role": "core",
        "activation_second_moment": 1.0,
        "outgoing_to_core_before": before_vector[0],
        "outgoing_to_core_after": after_vector[0],
        "outgoing_to_slack_before": before_vector[1],
        "outgoing_to_slack_after": after_vector[1],
        "outgoing_vector_before": "2|5",
        "outgoing_vector_after": "2|6",
        "operator_drift_norm": drift_norm,
        "capacity_before": capacity_before,
        "capacity_after": capacity_after,
        "relative_capacity_change": relative_capacity_change,
        "incoming_parameter_change": 0.0,
        "bn_affine_parameter_change": 0.0,
        "bn_buffer_change": 0.0,
        "evidence_label": "counterexample",
        "claim_id": "P10-OUTGOING-OPERATOR-DRIFT",
        "passed": bool(
            after_vector[0] == before_vector[0]
            and after_vector[1] != before_vector[1]
            and drift_norm == 1.0
            and relative_capacity_change > 0.0
        ),
    }]


def _slot_gradient(
    *, u: float, gamma: float, beta: float, v: float,
    input_value: float, target_value: float,
) -> dict[str, float]:
    internal = u * input_value
    preactivation = gamma * internal + beta
    hidden = max(0.0, preactivation)
    output = v * hidden
    residual = output - target_value
    relu_derivative = 1.0 if preactivation > 0.0 else 0.0
    def clean(value: float) -> float:
        return 0.0 if value == 0.0 else float(value)

    return {
        "hidden": clean(hidden),
        "output": clean(output),
        "residual": clean(residual),
        "gradient_u": clean(residual * v * relu_derivative * gamma * input_value),
        "gradient_gamma": clean(residual * v * relu_derivative * internal),
        "gradient_beta": clean(residual * v * relu_derivative),
        "gradient_v": clean(residual * hidden),
    }


def _released_slot_rows() -> list[dict[str, Any]]:
    input_value, target_value = REPAIR_INPUT, REPAIR_TARGET
    rows: list[dict[str, Any]] = []

    rows.append({
        "variant": "algorithm2_purged_child",
        "origin": "Algorithm 2 line 14, PDF page 69",
        "active_layer_status": "removed_or_purged",
        "physical_storage_status": "unspecified_by_algorithm",
        "reentry_transition_specified": False,
        "role": "paper_pseudocode_child",
        "input_value": "not_applicable", "target_value": "not_applicable",
        "incoming_u": "unspecified", "scale_gamma": "unspecified",
        "shift_beta": "unspecified", "outgoing_v": "unspecified",
        "initial_hidden": "not_evaluated", "initial_output": "not_evaluated",
        "initial_residual": "not_evaluated", "first_gradient_u": "not_evaluated",
        "first_gradient_gamma": "not_evaluated",
        "first_gradient_beta": "not_evaluated", "first_gradient_v": "not_evaluated",
        "outgoing_v_after_first_step": "not_evaluated",
        "second_output": "not_evaluated", "second_residual": "not_evaluated",
        "second_gradient_u": "not_evaluated",
        "second_gradient_gamma": "not_evaluated",
        "second_gradient_beta": "not_evaluated",
        "second_gradient_v": "not_evaluated",
        "second_internal_gradient_min": "not_evaluated",
        "reusable": False,
        "evidence_label": "verified_fact",
        "claim_id": "P10-PURGED-RELEASED-CONFLICT",
        "passed": True,
    })
    rows.append({
        "variant": "axiom3_released_child",
        "origin": "Axiom 3, PDF page 64",
        "active_layer_status": "plastic_vessel_assumed",
        "physical_storage_status": "unspecified_by_axiom",
        "reentry_transition_specified": False,
        "role": "paper_released_child_claim",
        "input_value": "not_applicable", "target_value": "not_applicable",
        "incoming_u": "unspecified", "scale_gamma": "unspecified",
        "shift_beta": "unspecified", "outgoing_v": "unspecified",
        "initial_hidden": "not_evaluated", "initial_output": "not_evaluated",
        "initial_residual": "not_evaluated", "first_gradient_u": "not_evaluated",
        "first_gradient_gamma": "not_evaluated",
        "first_gradient_beta": "not_evaluated", "first_gradient_v": "not_evaluated",
        "outgoing_v_after_first_step": "not_evaluated",
        "second_output": "not_evaluated", "second_residual": "not_evaluated",
        "second_gradient_u": "not_evaluated",
        "second_gradient_gamma": "not_evaluated",
        "second_gradient_beta": "not_evaluated",
        "second_gradient_v": "not_evaluated",
        "second_internal_gradient_min": "not_evaluated",
        "reusable": True,
        "evidence_label": "verified_fact",
        "claim_id": "P10-PURGED-RELEASED-CONFLICT",
        "passed": True,
    })

    zero = _slot_gradient(
        u=0.0, gamma=0.0, beta=0.0, v=0.0,
        input_value=input_value, target_value=target_value,
    )
    rows.append({
        "variant": "all_zero_retained_slot",
        "origin": "independent dead-slot counterexample",
        "active_layer_status": "retained",
        "physical_storage_status": "retained_by_fixture",
        "reentry_transition_specified": True,
        "role": "plastic_but_dead",
        "input_value": input_value,
        "target_value": target_value,
        "incoming_u": 0.0, "scale_gamma": 0.0, "shift_beta": 0.0,
        "outgoing_v": 0.0, "initial_hidden": zero["hidden"],
        "initial_output": zero["output"], "initial_residual": zero["residual"],
        "first_gradient_u": zero["gradient_u"],
        "first_gradient_gamma": zero["gradient_gamma"],
        "first_gradient_beta": zero["gradient_beta"],
        "first_gradient_v": zero["gradient_v"],
        "outgoing_v_after_first_step": 0.0, "second_output": zero["output"],
        "second_residual": zero["residual"],
        "second_gradient_u": zero["gradient_u"],
        "second_gradient_gamma": zero["gradient_gamma"],
        "second_gradient_beta": zero["gradient_beta"],
        "second_gradient_v": zero["gradient_v"],
        "second_internal_gradient_min": 0.0,
        "reusable": False,
        "evidence_label": "counterexample",
        "claim_id": "P10-DEAD-ZERO-SLOT",
        "passed": max(
            abs(value) for key, value in zero.items() if key.startswith("gradient_")
        ) == 0.0,
    })

    repair = _slot_gradient(
        u=1.0, gamma=1.0, beta=0.0, v=0.0,
        input_value=input_value, target_value=target_value,
    )
    post_outgoing = 0.0 - REPAIR_LEARNING_RATE * repair["gradient_v"]
    second = _slot_gradient(
        u=1.0, gamma=1.0, beta=0.0, v=post_outgoing,
        input_value=input_value, target_value=target_value,
    )
    second_internal_min = min(
        abs(second[name])
        for name in ("gradient_u", "gradient_gamma", "gradient_beta")
    )
    rows.append({
        "variant": "proposed_fixed_width_reinitialized_slot",
        "origin": "independent proposed repair",
        "active_layer_status": "retained",
        "physical_storage_status": "retained_by_fixture",
        "reentry_transition_specified": True,
        "role": "plastic_fixed_width",
        "input_value": input_value,
        "target_value": target_value,
        "incoming_u": 1.0, "scale_gamma": 1.0, "shift_beta": 0.0,
        "outgoing_v": 0.0, "initial_hidden": repair["hidden"],
        "initial_output": repair["output"], "initial_residual": repair["residual"],
        "first_gradient_u": repair["gradient_u"],
        "first_gradient_gamma": repair["gradient_gamma"],
        "first_gradient_beta": repair["gradient_beta"],
        "first_gradient_v": repair["gradient_v"],
        "outgoing_v_after_first_step": post_outgoing,
        "second_output": second["output"], "second_residual": second["residual"],
        "second_gradient_u": second["gradient_u"],
        "second_gradient_gamma": second["gradient_gamma"],
        "second_gradient_beta": second["gradient_beta"],
        "second_gradient_v": second["gradient_v"],
        "second_internal_gradient_min": second_internal_min,
        "reusable": second_internal_min > 0.0,
        "evidence_label": "proposed_repair",
        "claim_id": "P10-FIXED-WIDTH-REPAIR",
        "passed": (
            repair["output"] == 0.0
            and abs(repair["gradient_v"]) == 1.0
            and post_outgoing == 0.25
            and second_internal_min == 0.1875
        ),
    })
    return rows


def _source_state_rows(final_mean: float, final_variance: float) -> list[dict[str, Any]]:
    reference = _bn_output(BN_SOURCE_PROBES, BN_INITIAL_MEAN, BN_INITIAL_VARIANCE)
    specifications = (
        (
            "source_reference", "source", False, False,
            BN_INITIAL_MEAN, BN_INITIAL_VARIANCE,
            "P10-SOURCE-STATE-RESTORATION", "paper_claim",
        ),
        (
            "live_target_state", "target_adapted", False, False,
            final_mean, final_variance,
            "P10-BN-BUFFER-DRIFT", "counterexample",
        ),
        (
            "trainable_freeze_only", "target_adapted", False, False,
            final_mean, final_variance,
            "P10-BN-BUFFER-DRIFT", "counterexample",
        ),
        (
            "restored_source_state", "source_restored", False, True,
            BN_INITIAL_MEAN, BN_INITIAL_VARIANCE,
            "P10-SOURCE-STATE-RESTORATION", "paper_claim",
        ),
        (
            "frozen_buffer_control", "source_frozen", True, False,
            BN_INITIAL_MEAN, BN_INITIAL_VARIANCE,
            "P10-BN-STATE-REPAIR", "proposed_repair",
        ),
    )
    rows: list[dict[str, Any]] = []
    for (
        state_id, origin, frozen, restored, mean, variance, claim_id, evidence_label,
    ) in specifications:
        observed = _bn_output(BN_SOURCE_PROBES, mean, variance)
        source_rmse = _rmse(observed, reference)
        if state_id in {"source_reference", "restored_source_state", "frozen_buffer_control"}:
            passed = source_rmse == 0.0
        else:
            passed = source_rmse > 0.0
        rows.append({
            "state_id": state_id,
            "bn_state_origin": origin,
            "buffers_frozen_during_target": frozen,
            "task_specific_restore": restored,
            "running_mean": mean,
            "running_variance": variance,
            "probe_values": "-1|0|1",
            "reference_outputs": "|".join(f"{float(value):.17g}" for value in reference),
            "observed_outputs": "|".join(f"{float(value):.17g}" for value in observed),
            "trainable_parameter_change_norm": 0.0,
            "source_rmse": source_rmse,
            "evidence_label": evidence_label,
            "claim_id": claim_id,
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


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def build_audit(*, full: bool = False) -> dict[str, Any]:
    trajectory_threshold, global_capacity_threshold = _derived_global_thresholds()
    config = build_config(full=full)
    source_rows = _source_map_rows()
    partition_rows, assignments = _partition_assignments()
    disagreement_rows = _partition_disagreements(assignments)
    routing_rows, masked, routing_after = _routing_rows()
    bn_rows, final_mean, final_variance = _bn_rows()
    operator_rows = _operator_rows(masked, routing_after)
    slot_rows = _released_slot_rows()
    source_state_rows = _source_state_rows(final_mean, final_variance)
    row_groups = (
        source_rows, partition_rows, disagreement_rows, routing_rows, bn_rows,
        operator_rows, slot_rows, source_state_rows,
    )
    all_rows = [row for group in row_groups for row in group]
    evidence_mismatches = sum(
        row["evidence_label"] != CLAIM_CLASSES[row["claim_id"]] for row in all_rows
    ) + sum(
        record["evidence_label"] != CLAIM_CLASSES[record["claim_id"]]
        for record in METHOD_ORIGINS.values()
    )

    disagreements = {
        (row["rule_a"], row["rule_b"]): row["disagreement_count"]
        for row in disagreement_rows
    }
    slack_to_core = next(
        row for row in routing_rows
        if row["upstream_role"] == "slack" and row["downstream_role"] == "core"
    )
    core_to_slack = next(
        row for row in routing_rows
        if row["upstream_role"] == "core" and row["downstream_role"] == "slack"
    )
    zero_slot = next(row for row in slot_rows if row["variant"] == "all_zero_retained_slot")
    repair_slot = next(
        row for row in slot_rows if row["variant"] == "proposed_fixed_width_reinitialized_slot"
    )
    state_by_id = {row["state_id"]: row for row in source_state_rows}
    neuron_order = tuple(sorted(next(iter(assignments.values()))))
    partition_signatures = {
        tuple(rule_assignments[neuron] for neuron in neuron_order)
        for rule_assignments in assignments.values()
    }
    zero_gradients = np.asarray(
        [
            zero_slot["first_gradient_u"],
            zero_slot["first_gradient_gamma"],
            zero_slot["first_gradient_beta"],
            zero_slot["first_gradient_v"],
        ],
        dtype=np.float64,
    )

    checks = {
        "nonfinite_count": float(sum(
            not math.isfinite(number) for number in _walk_numbers(all_rows)
        )),
        "row_failure_count": float(sum(not row["passed"] for row in all_rows)),
        "evidence_class_mismatch_count": float(evidence_mismatches),
        "trajectory_threshold": trajectory_threshold,
        "global_capacity_threshold": global_capacity_threshold,
        "trajectory_unassigned_count": float(sum(
            assignment == "unassigned"
            for assignment in assignments["eq24_27_trajectory_cost"].values()
        )),
        "trajectory_vs_layer_disagreement_count": float(disagreements[
            ("eq24_27_trajectory_cost", "definition_h1_layer_capacity")
        ]),
        "trajectory_vs_global_disagreement_count": float(disagreements[
            ("eq24_27_trajectory_cost", "algorithm3_global_capacity")
        ]),
        "layer_vs_global_disagreement_count": float(disagreements[
            ("definition_h1_layer_capacity", "algorithm3_global_capacity")
        ]),
        "partition_signature_count": float(len(partition_signatures)),
        "masked_slack_to_core_weight": float(abs(slack_to_core["masked_weight"])),
        "core_dynamic_drift": routing_rows[0]["core_dynamic_drift"],
        "initialization_shock": routing_rows[0]["initialization_shock"],
        "core_to_slack_weight_change": float(abs(core_to_slack["dynamic_change"])),
        "operator_drift_norm": operator_rows[0]["operator_drift_norm"],
        "relative_capacity_change": operator_rows[0]["relative_capacity_change"],
        "trainable_parameter_change_norm": float(np.linalg.norm([
            row["trainable_parameter_change_norm"] for row in bn_rows
        ])),
        "final_running_mean": final_mean,
        "final_running_variance": final_variance,
        "live_source_rmse": state_by_id["live_target_state"]["source_rmse"],
        "zero_slot_gradient_norm": float(np.linalg.norm(zero_gradients)),
        "repair_initial_output": abs(repair_slot["initial_output"]),
        "repair_first_outgoing_gradient_magnitude": abs(repair_slot["first_gradient_v"]),
        "repair_second_internal_gradient_min": repair_slot["second_internal_gradient_min"],
        "restored_source_rmse": state_by_id["restored_source_state"]["source_rmse"],
        "frozen_source_rmse": state_by_id["frozen_buffer_control"]["source_rmse"],
    }
    if tuple(checks) != tuple(CHECK_SPECS):
        raise AuditFailure("internal check order does not match the fixed Part 10 registry")
    check_rows = {
        name: {
            "value": value,
            "minimum": CHECK_SPECS[name]["minimum"],
            "maximum": CHECK_SPECS[name]["maximum"],
            "units": CHECK_SPECS[name]["units"],
            "passed": bool(
                CHECK_SPECS[name]["minimum"] <= value <= CHECK_SPECS[name]["maximum"]
            ),
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
                "Section 11.2.2 and Eqs. 24-29, PDF pages 21-23",
                "Appendix G.5, PDF page 62",
                "Appendix H and Eqs. 102-118, PDF pages 62-67",
                "Algorithms 2-3, PDF pages 69-70",
            ],
        },
        "config": config,
        "config_sha256": _config_hash(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": REPRODUCTION_SCOPE,
        "check_count": len(check_rows),
        "passing_check_count": sum(row["passed"] for row in check_rows.values()),
        "checks": check_rows,
        "partition_fixture": {
            "trajectory_formula": "J=N*c/E_b",
            "trajectory_costs": _trajectory_costs_by_neuron(),
            "trajectory_threshold": trajectory_threshold,
            "layer_thresholds": LAYER_THRESHOLDS,
            "global_capacity_threshold": global_capacity_threshold,
            "assignments": assignments,
        },
        "routing_fixture": {
            "source_weights": ROUTING_SOURCE.tolist(),
            "masked_weights": masked.tolist(),
            "updated_weights": routing_after.tolist(),
            "source_activation": SOURCE_ACTIVATION.tolist(),
            "target_activation": TARGET_ACTIVATION.tolist(),
            "slack_to_core_weight_after": slack_to_core["weight_after"],
            "core_to_slack_weight_after": core_to_slack["weight_after"],
            "initialization_shock": checks["initialization_shock"],
            "core_dynamic_drift": checks["core_dynamic_drift"],
            "operator_drift_norm": checks["operator_drift_norm"],
            "relative_capacity_change": checks["relative_capacity_change"],
        },
        "bn_fixture": {
            "source_probes": BN_SOURCE_PROBES.tolist(),
            "final_running_mean": final_mean,
            "final_running_variance": final_variance,
            "trainable_parameter_change_norm": checks["trainable_parameter_change_norm"],
            "live_source_rmse": checks["live_source_rmse"],
        },
        "released_slot_fixture": {
            "dead_slot_gradient_norm": checks["zero_slot_gradient_norm"],
            "repair_initial_output": checks["repair_initial_output"],
            "repair_first_outgoing_gradient": repair_slot["first_gradient_v"],
            "repair_outgoing_after_first_step": repair_slot["outgoing_v_after_first_step"],
            "repair_second_internal_gradient_min": checks[
                "repair_second_internal_gradient_min"
            ],
        },
        "row_counts": {
            "source_map": len(source_rows),
            "partition_assignments": len(partition_rows),
            "partition_disagreements": len(disagreement_rows),
            "routing_transitions": len(routing_rows),
            "bn_buffer_drift": len(bn_rows),
            "operator_drift": len(operator_rows),
            "released_slot_gradients": len(slot_rows),
            "source_state_controls": len(source_state_rows),
        },
    }
    if not passed:
        failed_checks = [name for name, row in check_rows.items() if not row["passed"]]
        failed_rows = [
            row.get("step_id", row.get("neuron", row.get("variant", row.get("state_id", "unknown"))))
            for row in all_rows if not row["passed"]
        ]
        raise AuditFailure(f"audit checks failed: checks={failed_checks}, rows={failed_rows}")
    return {
        "summary": summary,
        "source_map.csv": source_rows,
        "partition_assignments.csv": partition_rows,
        "partition_disagreements.csv": disagreement_rows,
        "routing_transitions.csv": routing_rows,
        "bn_buffer_drift.csv": bn_rows,
        "operator_drift.csv": operator_rows,
        "released_slot_gradients.csv": slot_rows,
        "source_state_controls.csv": source_state_rows,
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
    if summary.get("check_count") != 26 or summary.get("passing_check_count") != 26:
        raise AuditFailure("summary must contain 26 passing checks")
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paper_identity(paper_pdf: Path | None, full: bool) -> dict[str, Any]:
    manifest = json.loads(PAPER_MANIFEST_PATH.read_text(encoding="utf-8"))
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
            "entrypoint": "experiments/audits/part10_deft_core/run.py",
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
