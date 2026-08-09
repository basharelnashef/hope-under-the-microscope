"""Physical sum-map optimized merge for the proposed HOPE-R design.

The executed object is the child contribution sum. This module searches a
deterministic set of ReLU directions in the children’s at-most-two-dimensional
augmented affine span and fits the outgoing vector by weighted least squares.
The result is sum-map optimized on the declared calibration measure. It is not
universally sum preserving.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from hope_repro.proposed_repairs.hope_r.common import (
    CalibrationMeasure,
    FunctionalNeuron,
    _readonly,
)


DEFAULT_ANGULAR_SAMPLES = 720


def _stable_norm(vector: np.ndarray) -> float:
    """Return a scale-safe Euclidean norm for one finite float64 vector."""

    maximum = float(np.max(np.abs(vector)))
    if maximum == 0.0:
        return 0.0
    scaled = vector / maximum
    value = maximum * math.sqrt(float(scaled @ scaled))
    if not math.isfinite(value):
        raise FloatingPointError("vector norm was non-finite")
    return value


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = _stable_norm(vector)
    if norm == 0.0:
        return None
    result = np.asarray(vector / norm, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("direction normalization was non-finite")
    return result


def _span_basis(first: np.ndarray, second: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Build a deterministic Gram-Schmidt basis anchored at the first child."""

    first_unit = _normalize(first)
    second_unit = _normalize(second)
    if first_unit is None and second_unit is None:
        return [], 0
    if first_unit is None:
        return [second_unit], 1  # type: ignore[list-item]

    basis = [first_unit]
    if second_unit is None:
        return basis, 1
    projection = float(first_unit @ second_unit)
    residual = second_unit - projection * first_unit
    residual_norm = _stable_norm(residual)
    dependence_tolerance = 512.0 * np.finfo(np.float64).eps
    if residual_norm <= dependence_tolerance:
        return basis, 1
    second_basis = residual / residual_norm
    if not np.all(np.isfinite(second_basis)):
        raise FloatingPointError("second span basis direction was non-finite")
    basis.append(np.asarray(second_basis, dtype=np.float64))
    return basis, 2


def _candidate_directions(
    first: np.ndarray,
    second: np.ndarray,
    angular_samples: int,
) -> tuple[list[np.ndarray], int]:
    """Return child anchors plus a deterministic full-circle span search."""

    basis, rank = _span_basis(first, second)
    if rank == 0:
        return [np.zeros_like(first, dtype=np.float64)], 0

    candidates: list[np.ndarray] = []
    duplicate_tolerance = 2048.0 * np.finfo(np.float64).eps

    def append_unique(value: np.ndarray | None) -> None:
        if value is None:
            return
        direction = _normalize(value)
        if direction is None:
            return
        for existing in candidates:
            if _stable_norm(direction - existing) <= duplicate_tolerance:
                return
        candidates.append(direction)

    first_anchor = _normalize(first)
    second_anchor = _normalize(second)
    append_unique(first_anchor)
    append_unique(second_anchor)
    append_unique(None if first_anchor is None else -first_anchor)
    append_unique(None if second_anchor is None else -second_anchor)

    if rank == 1:
        append_unique(basis[0])
        append_unique(-basis[0])
    else:
        for index in range(angular_samples):
            angle = 2.0 * math.pi * index / angular_samples
            direction = math.cos(angle) * basis[0] + math.sin(angle) * basis[1]
            append_unique(direction)
    return candidates, rank


def _weighted_squared_norm(values: np.ndarray, weights: np.ndarray) -> float:
    with np.errstate(over="raise", invalid="raise"):
        try:
            row_squared = np.einsum("ij,ij->i", values, values)
            result = float(weights @ row_squared)
        except FloatingPointError as error:
            raise FloatingPointError("weighted squared norm overflowed float64") from error
    if not math.isfinite(result):
        raise FloatingPointError("weighted squared norm was non-finite")
    roundoff = 128.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(row_squared))
    )
    if result < 0.0 and result >= -roundoff:
        return 0.0
    if result < 0.0:
        raise FloatingPointError("weighted squared norm was materially negative")
    return result


def _fit_outgoing(
    activation: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit one outgoing vector and return it, residual squared, parent squared."""

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            denominator = float(weights @ (activation * activation))
            if denominator == 0.0:
                outgoing = np.zeros(target.shape[1], dtype=np.float64)
                parent_values = np.zeros_like(target, dtype=np.float64)
            else:
                numerator = (weights * activation) @ target
                outgoing = np.asarray(numerator / denominator, dtype=np.float64)
                parent_values = activation[:, np.newaxis] * outgoing
        except FloatingPointError as error:
            raise FloatingPointError("least-squares outgoing fit overflowed float64") from error
    if denominator < 0.0 or not math.isfinite(denominator):
        raise FloatingPointError("least-squares activation energy was invalid")
    if not np.all(np.isfinite(outgoing)) or not np.all(np.isfinite(parent_values)):
        raise FloatingPointError("least-squares outgoing fit was non-finite")
    residual = target - parent_values
    return (
        outgoing,
        _weighted_squared_norm(residual, weights),
        _weighted_squared_norm(parent_values, weights),
    )


@dataclass(frozen=True)
class SumMapMergeResult:
    """Measured result of one physical sum-map optimized merge."""

    parent: FunctionalNeuron
    selected_augmented_direction: np.ndarray
    selected_outgoing_weight: np.ndarray
    source_child_sum_norm: float
    parent_norm: float
    absolute_residual: float
    relative_residual: float
    candidate_count: int
    span_rank: int
    measure_label: str
    method: str = "sum_map_optimized"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_augmented_direction",
            _readonly(
                np.array(
                    self.selected_augmented_direction,
                    dtype=np.float64,
                    copy=True,
                )
            ),
        )
        object.__setattr__(
            self,
            "selected_outgoing_weight",
            _readonly(
                np.array(self.selected_outgoing_weight, dtype=np.float64, copy=True)
            ),
        )

    @property
    def measured_residual(self) -> float:
        """Alias for the absolute executed-sum residual."""

        return self.absolute_residual


def sum_map_optimized_merge(
    first: FunctionalNeuron,
    second: FunctionalNeuron,
    measure: CalibrationMeasure,
    *,
    angular_samples: int = DEFAULT_ANGULAR_SAMPLES,
) -> SumMapMergeResult:
    """Fit one physical ReLU neuron to the executed two-child sum.

    Candidate incoming affine directions are restricted to the augmented span
    of ``[w_first, b_first]`` and ``[w_second, b_second]``. The two signed
    child directions are exact anchors, and a rank-two span receives a
    deterministic full-circle angular search. For each candidate, the complete
    outgoing vector is the exact weighted least-squares solution under
    ``measure``.

    The reported residual is measured on that same declared sample measure.
    A zero residual proves realizability only there. Noncollinear ReLU sums are
    generally not representable by one ReLU neuron, so this API intentionally
    says ``optimized`` rather than claiming universal sum preservation.
    """

    if not isinstance(first, FunctionalNeuron) or not isinstance(
        second, FunctionalNeuron
    ):
        raise TypeError("first and second must be FunctionalNeuron instances")
    if not isinstance(measure, CalibrationMeasure):
        raise TypeError("measure must be a CalibrationMeasure")
    if isinstance(angular_samples, bool) or not isinstance(
        angular_samples, (int, np.integer)
    ):
        raise ValueError("angular_samples must be an integer")
    angular_count = int(angular_samples)
    if angular_count < 4:
        raise ValueError("angular_samples must be at least four")
    if first.incoming_weight.shape != second.incoming_weight.shape:
        raise ValueError("child incoming dimensions must match")
    if first.outgoing_weight.shape != second.outgoing_weight.shape:
        raise ValueError("child outgoing dimensions must match")
    if measure.samples.shape[1] != first.incoming_weight.size:
        raise ValueError("calibration sample width must match child input width")

    first_values = first.contributions(measure.samples)
    second_values = second.contributions(measure.samples)
    with np.errstate(over="raise", invalid="raise"):
        try:
            target = first_values + second_values
        except FloatingPointError as error:
            raise FloatingPointError("child contribution sum overflowed float64") from error
    if not np.all(np.isfinite(target)):
        raise FloatingPointError("child contribution sum was non-finite")
    source_squared = _weighted_squared_norm(target, measure.weights)

    directions, span_rank = _candidate_directions(
        first.augmented_weight,
        second.augmented_weight,
        angular_count,
    )
    augmented_samples = np.column_stack(
        [measure.samples, np.ones(measure.samples.shape[0], dtype=np.float64)]
    )

    best_direction: np.ndarray | None = None
    best_outgoing: np.ndarray | None = None
    best_residual_squared = math.inf
    best_parent_squared = math.inf
    comparison_tolerance = 128.0 * np.finfo(np.float64).eps * max(
        1.0, source_squared
    )
    for direction in directions:
        with np.errstate(over="raise", invalid="raise"):
            try:
                activation = np.maximum(augmented_samples @ direction, 0.0)
            except FloatingPointError as error:
                raise FloatingPointError("parent activation evaluation overflowed") from error
        if not np.all(np.isfinite(activation)):
            raise FloatingPointError("parent activation evaluation was non-finite")
        outgoing, residual_squared, parent_squared = _fit_outgoing(
            activation,
            target,
            measure.weights,
        )
        if residual_squared < best_residual_squared - comparison_tolerance:
            best_direction = direction
            best_outgoing = outgoing
            best_residual_squared = residual_squared
            best_parent_squared = parent_squared

    if best_direction is None or best_outgoing is None:
        raise FloatingPointError("sum-map optimized search produced no candidate")

    source_norm = math.sqrt(source_squared)
    parent_norm = math.sqrt(best_parent_squared)
    residual = math.sqrt(best_residual_squared)
    relative = residual / source_norm if source_norm > 0.0 else 0.0
    if not all(
        math.isfinite(value)
        for value in (source_norm, parent_norm, residual, relative)
    ):
        raise FloatingPointError("sum-map optimized metrics were non-finite")

    parent = FunctionalNeuron(
        incoming_weight=best_direction[:-1],
        bias=float(best_direction[-1]),
        outgoing_weight=best_outgoing,
    )
    return SumMapMergeResult(
        parent=parent,
        selected_augmented_direction=best_direction,
        selected_outgoing_weight=best_outgoing,
        source_child_sum_norm=source_norm,
        parent_norm=parent_norm,
        absolute_residual=residual,
        relative_residual=relative,
        candidate_count=len(directions),
        span_rank=span_rank,
        measure_label=measure.label,
    )
