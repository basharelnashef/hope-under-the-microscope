"""Framework-independent fixed-width DEFT repair primitives.

This module defines a proposed repair, not paper-literal HOPE behavior. Its
claims are local to the deterministic state transitions and NumPy fixtures
implemented here. It does not claim benchmark accuracy, retention, H-score,
or optimizer convergence.

The route convention is ``[source_slot, destination_slot]``. A complete
source-owned operator contains its incoming parameters, affine parameters,
and every element of its outgoing row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


class SlotState(str, Enum):
    """Lifecycle and adaptation state of one fixed physical slot."""

    CORE = "core"
    SLACK = "slack"
    RELEASED = "released"
    REINITIALIZED = "reinitialized"


class BatchNormBufferPolicy(str, Enum):
    """Ownership policy for BatchNorm running buffers during adaptation."""

    FROZEN = "frozen"
    DUAL_TRACK = "dual_track"


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty, trimmed string")
    return value


def _positive_width(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    width = int(value)
    if width <= 0:
        raise ValueError(f"{name} must be positive")
    return width


def _finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a Boolean")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _finite_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    nonempty: bool = True,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be convertible to float64") from error
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if nonempty and array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite float64 values")
    return np.array(array, dtype=np.float64, copy=True)


def _binary_mask(
    value: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    mask = _finite_array(value, name=name, shape=shape)
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return mask


def _readonly_vector(
    value: ArrayLike,
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray:
    shape = None if length is None else (length,)
    vector = _finite_array(value, name=name, ndim=1, shape=shape)
    vector.setflags(write=False)
    return vector


def _slot_state(value: SlotState | str, *, name: str) -> SlotState:
    if isinstance(value, SlotState):
        return value
    try:
        return SlotState(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(state.value for state in SlotState)
        raise ValueError(f"{name} must be one of: {allowed}") from error


def _buffer_policy(
    value: BatchNormBufferPolicy | str,
    *,
    name: str,
) -> BatchNormBufferPolicy:
    if isinstance(value, BatchNormBufferPolicy):
        return value
    try:
        return BatchNormBufferPolicy(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(policy.value for policy in BatchNormBufferPolicy)
        raise ValueError(f"{name} must be one of: {allowed}") from error


def _slot_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of stable slot IDs")
    identifiers = tuple(
        _identifier(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not identifiers:
        raise ValueError(f"{name} must be nonempty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must contain unique stable slot IDs")
    return identifiers


@dataclass(frozen=True)
class LayerDefinition:
    """Fixed physical width and explicit core count for one layer."""

    layer_id: str
    slot_ids: tuple[str, ...]
    core_count: int

    def __post_init__(self) -> None:
        layer_id = _identifier(self.layer_id, name="layer_id")
        slot_ids = _slot_ids(self.slot_ids, name=f"{layer_id}.slot_ids")
        if isinstance(self.core_count, (bool, np.bool_)) or not isinstance(
            self.core_count, (int, np.integer)
        ):
            raise TypeError("core_count must be an integer")
        core_count = int(self.core_count)
        if not 0 <= core_count <= len(slot_ids):
            raise ValueError("core_count must be between zero and fixed layer width")
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(self, "slot_ids", slot_ids)
        object.__setattr__(self, "core_count", core_count)

    @property
    def width(self) -> int:
        return len(self.slot_ids)


@dataclass(frozen=True)
class LayerPartition:
    """Result of the one declared current-capacity partition rule."""

    layer_id: str
    slot_ids: tuple[str, ...]
    capacities: tuple[float, ...]
    core_count: int
    ranked_slot_ids: tuple[str, ...]
    core_slot_ids: tuple[str, ...]
    slack_slot_ids: tuple[str, ...]
    tie_break_rule: str = "capacity descending, then stable slot ID ascending"

    def state_by_slot(self) -> dict[str, SlotState]:
        core = set(self.core_slot_ids)
        return {
            slot_id: SlotState.CORE if slot_id in core else SlotState.SLACK
            for slot_id in self.slot_ids
        }


def top_k_current_capacity_partition(
    layer: LayerDefinition,
    current_capacity: Mapping[str, float],
) -> LayerPartition:
    """Select exactly ``core_count`` slots by current capacity in one layer.

    Larger capacity ranks first. Exact ties are resolved by ascending stable
    slot ID, independent of mapping insertion order and physical slot index.
    The capacity mapping must name exactly the layer's stable slots.
    """

    if not isinstance(layer, LayerDefinition):
        raise TypeError("layer must be a LayerDefinition")
    if not isinstance(current_capacity, Mapping):
        raise TypeError("current_capacity must be a mapping keyed by slot ID")
    if any(not isinstance(slot_id, str) for slot_id in current_capacity):
        raise ValueError("current_capacity keys must be stable string slot IDs")
    observed_ids = set(current_capacity)
    expected_ids = set(layer.slot_ids)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise ValueError(
            f"current_capacity keys must equal layer slot IDs; "
            f"missing={missing}, extra={extra}"
        )
    capacities_by_id = {
        slot_id: _finite_scalar(
            current_capacity[slot_id],
            name=f"current_capacity[{slot_id!r}]",
        )
        for slot_id in layer.slot_ids
    }
    if any(value < 0.0 for value in capacities_by_id.values()):
        raise ValueError("current capacities must be nonnegative")
    ranked = tuple(
        sorted(
            layer.slot_ids,
            key=lambda slot_id: (-capacities_by_id[slot_id], slot_id),
        )
    )
    core = ranked[: layer.core_count]
    slack = ranked[layer.core_count :]
    return LayerPartition(
        layer_id=layer.layer_id,
        slot_ids=layer.slot_ids,
        capacities=tuple(capacities_by_id[slot_id] for slot_id in layer.slot_ids),
        core_count=layer.core_count,
        ranked_slot_ids=ranked,
        core_slot_ids=core,
        slack_slot_ids=slack,
    )


def layerwise_top_k_current_capacity_partition(
    layers: Sequence[LayerDefinition],
    current_capacity: Mapping[str, float],
) -> tuple[LayerPartition, ...]:
    """Apply the same explicit top-k rule independently to every layer."""

    definitions = tuple(layers)
    if not definitions:
        raise ValueError("layers must be nonempty")
    if not all(isinstance(layer, LayerDefinition) for layer in definitions):
        raise TypeError("layers must contain only LayerDefinition values")
    layer_ids = tuple(layer.layer_id for layer in definitions)
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("layer IDs must be unique")
    all_slot_ids = tuple(
        slot_id for layer in definitions for slot_id in layer.slot_ids
    )
    if len(set(all_slot_ids)) != len(all_slot_ids):
        raise ValueError("stable slot IDs must be unique across all layers")
    if not isinstance(current_capacity, Mapping):
        raise TypeError("current_capacity must be a mapping keyed by slot ID")
    if any(not isinstance(slot_id, str) for slot_id in current_capacity):
        raise ValueError("current_capacity keys must be stable string slot IDs")
    if set(current_capacity) != set(all_slot_ids):
        missing = sorted(set(all_slot_ids) - set(current_capacity))
        extra = sorted(set(current_capacity) - set(all_slot_ids))
        raise ValueError(
            f"current_capacity keys must equal all stable slot IDs; "
            f"missing={missing}, extra={extra}"
        )
    return tuple(
        top_k_current_capacity_partition(
            layer,
            {slot_id: current_capacity[slot_id] for slot_id in layer.slot_ids},
        )
        for layer in definitions
    )


@dataclass(frozen=True)
class ReinitializedSlotParameters:
    """Live incoming side and initially silent outgoing side for one slot."""

    slot_id: str
    incoming_weight: np.ndarray
    affine_scale: float
    affine_shift: float
    outgoing_weight: np.ndarray

    def __post_init__(self) -> None:
        slot_id = _identifier(self.slot_id, name="slot_id")
        incoming = _readonly_vector(self.incoming_weight, name="incoming_weight")
        outgoing = _readonly_vector(self.outgoing_weight, name="outgoing_weight")
        scale = _finite_scalar(self.affine_scale, name="affine_scale")
        shift = _finite_scalar(self.affine_shift, name="affine_shift")
        if not np.any(incoming != 0.0):
            raise ValueError("incoming_weight must contain a live nonzero path")
        if np.any(outgoing != 0.0):
            raise ValueError("outgoing_weight must be zero at reinitialization")
        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "incoming_weight", incoming)
        object.__setattr__(self, "outgoing_weight", outgoing)
        object.__setattr__(self, "affine_scale", scale)
        object.__setattr__(self, "affine_shift", shift)

    def forward(self, inputs: ArrayLike) -> tuple[float, np.ndarray]:
        """Return the ReLU activation and outgoing contribution."""

        vector = _finite_array(
            inputs,
            name="inputs",
            ndim=1,
            shape=self.incoming_weight.shape,
        )
        with np.errstate(over="raise", invalid="raise"):
            preactivation = self.affine_scale * float(self.incoming_weight @ vector)
            preactivation += self.affine_shift
            activation = max(preactivation, 0.0)
            contribution = self.outgoing_weight * activation
        if not math.isfinite(activation) or not np.all(np.isfinite(contribution)):
            raise FloatingPointError("released-slot forward result was non-finite")
        return activation, np.array(contribution, copy=True)


def initialize_released_slot(
    slot_id: str,
    *,
    input_width: int,
    output_width: int,
    incoming_scale: float = 1.0,
) -> ReinitializedSlotParameters:
    """Create a deterministic live-incoming, zero-outgoing slot.

    The incoming vector is positive and L2-normalized up to
    ``incoming_scale``. It is live on an all-ones probe. All outgoing routes
    start at zero, so the slot initially contributes exactly zero.
    """

    input_width = _positive_width(input_width, name="input_width")
    output_width = _positive_width(output_width, name="output_width")
    incoming_scale = _finite_scalar(incoming_scale, name="incoming_scale")
    if incoming_scale <= 0.0:
        raise ValueError("incoming_scale must be positive")
    coefficient = incoming_scale / math.sqrt(input_width)
    if not math.isfinite(coefficient) or coefficient == 0.0:
        raise ValueError("incoming_scale must be representable at the requested width")
    return ReinitializedSlotParameters(
        slot_id=slot_id,
        incoming_weight=np.full(input_width, coefficient, dtype=np.float64),
        affine_scale=1.0,
        affine_shift=0.0,
        outgoing_weight=np.zeros(output_width, dtype=np.float64),
    )


@dataclass(frozen=True)
class GradientLivenessTrace:
    """Two backward passes for the declared scalar released-slot fixture."""

    initial_activation: float
    initial_output: float
    first_incoming_gradient: float
    first_scale_gradient: float
    first_shift_gradient: float
    first_outgoing_gradient: float
    outgoing_after_first_step: float
    second_output: float
    second_incoming_gradient: float
    second_scale_gradient: float
    second_shift_gradient: float
    second_outgoing_gradient: float

    @property
    def first_stage_outgoing_live(self) -> bool:
        return self.first_outgoing_gradient != 0.0

    @property
    def second_stage_incoming_live(self) -> bool:
        return (
            self.second_incoming_gradient != 0.0
            and self.second_scale_gradient != 0.0
            and self.second_shift_gradient != 0.0
        )


def two_stage_gradient_liveness(
    *,
    input_value: float = 1.0,
    target_value: float = 1.0,
    learning_rate: float = 0.25,
) -> GradientLivenessTrace:
    """Execute two squared-loss backward passes on the scalar repair fixture."""

    x = _finite_scalar(input_value, name="input_value")
    target = _finite_scalar(target_value, name="target_value")
    rate = _finite_scalar(learning_rate, name="learning_rate")
    if rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    incoming = 1.0
    scale = 1.0
    shift = 0.0
    outgoing = 0.0
    with np.errstate(over="raise", invalid="raise"):
        preactivation = scale * incoming * x + shift
        if preactivation <= 0.0:
            raise ValueError("fixture requires a strictly positive initial preactivation")
        activation = preactivation
        output = outgoing * activation
        residual = output - target
        first_outgoing = residual * activation
        first_shift = residual * outgoing
        first_scale = first_shift * incoming * x
        first_incoming = first_shift * scale * x

        outgoing_after = outgoing - rate * first_outgoing
        incoming_after = incoming - rate * first_incoming
        scale_after = scale - rate * first_scale
        shift_after = shift - rate * first_shift

        second_preactivation = scale_after * incoming_after * x + shift_after
        if second_preactivation <= 0.0:
            raise ValueError("fixture became ReLU-inactive after the first step")
        second_activation = second_preactivation
        second_output = outgoing_after * second_activation
        second_residual = second_output - target
        second_outgoing = second_residual * second_activation
        second_shift = second_residual * outgoing_after
        second_scale = second_shift * incoming_after * x
        second_incoming = second_shift * scale_after * x

    values = (
        activation,
        output,
        first_incoming,
        first_scale,
        first_shift,
        first_outgoing,
        outgoing_after,
        second_output,
        second_incoming,
        second_scale,
        second_shift,
        second_outgoing,
    )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("gradient-liveness fixture produced a non-finite value")
    return GradientLivenessTrace(*values)


class FixedWidthDeftStateMachine:
    """Fixed-width slot lifecycle with stable IDs and explicit transitions."""

    _ALLOWED_TRANSITIONS = {
        SlotState.CORE: frozenset(),
        SlotState.SLACK: frozenset({SlotState.RELEASED}),
        SlotState.RELEASED: frozenset({SlotState.REINITIALIZED}),
        SlotState.REINITIALIZED: frozenset({SlotState.SLACK, SlotState.RELEASED}),
    }

    def __init__(
        self,
        layers: Sequence[LayerDefinition],
        current_capacity: Mapping[str, float],
    ) -> None:
        self._layers = tuple(layers)
        self._partitions = layerwise_top_k_current_capacity_partition(
            self._layers, current_capacity
        )
        self._layer_by_id = {layer.layer_id: layer for layer in self._layers}
        self._layer_by_slot = {
            slot_id: layer.layer_id
            for layer in self._layers
            for slot_id in layer.slot_ids
        }
        self._states = {
            slot_id: state
            for partition in self._partitions
            for slot_id, state in partition.state_by_slot().items()
        }
        self._reinitialized_parameters: dict[str, ReinitializedSlotParameters] = {}

    @property
    def partitions(self) -> tuple[LayerPartition, ...]:
        return self._partitions

    @property
    def fixed_widths(self) -> dict[str, int]:
        return {layer.layer_id: layer.width for layer in self._layers}

    @property
    def stable_slot_ids(self) -> tuple[str, ...]:
        return tuple(slot_id for layer in self._layers for slot_id in layer.slot_ids)

    @property
    def state_by_slot(self) -> dict[str, SlotState]:
        return dict(self._states)

    def layer_slot_ids(self, layer_id: str) -> tuple[str, ...]:
        identifier = _identifier(layer_id, name="layer_id")
        try:
            return self._layer_by_id[identifier].slot_ids
        except KeyError as error:
            raise KeyError(f"unknown layer ID: {identifier}") from error

    def state(self, slot_id: str) -> SlotState:
        identifier = _identifier(slot_id, name="slot_id")
        try:
            return self._states[identifier]
        except KeyError as error:
            raise KeyError(f"unknown stable slot ID: {identifier}") from error

    def states_for_layer(self, layer_id: str) -> tuple[SlotState, ...]:
        return tuple(self._states[slot_id] for slot_id in self.layer_slot_ids(layer_id))

    def transition(self, slot_id: str, target: SlotState | str) -> None:
        identifier = _identifier(slot_id, name="slot_id")
        current = self.state(identifier)
        destination = _slot_state(target, name="target")
        if destination is SlotState.REINITIALIZED:
            raise ValueError(
                "use reinitialize() so the reinitialized state has declared parameters"
            )
        if destination not in self._ALLOWED_TRANSITIONS[current]:
            raise ValueError(
                f"invalid slot transition for {identifier}: "
                f"{current.value} -> {destination.value}"
            )
        self._states[identifier] = destination
        if destination is SlotState.RELEASED:
            self._reinitialized_parameters.pop(identifier, None)

    def release(self, slot_id: str) -> None:
        """Release a slack or reinitialized slot without changing width."""

        self.transition(slot_id, SlotState.RELEASED)

    def reinitialize(
        self,
        slot_id: str,
        *,
        input_width: int,
        output_width: int,
        incoming_scale: float = 1.0,
    ) -> ReinitializedSlotParameters:
        """Reinitialize a released slot and retain its stable ID."""

        identifier = _identifier(slot_id, name="slot_id")
        if self.state(identifier) is not SlotState.RELEASED:
            raise ValueError("only a released slot can be reinitialized")
        parameters = initialize_released_slot(
            identifier,
            input_width=input_width,
            output_width=output_width,
            incoming_scale=incoming_scale,
        )
        self._states[identifier] = SlotState.REINITIALIZED
        self._reinitialized_parameters[identifier] = parameters
        return parameters

    def activate_reinitialized(self, slot_id: str) -> None:
        """Move a reinitialized plastic slot into the ordinary slack state."""

        self.transition(slot_id, SlotState.SLACK)

    def reinitialized_parameters(self, slot_id: str) -> ReinitializedSlotParameters:
        identifier = _identifier(slot_id, name="slot_id")
        if self.state(identifier) is not SlotState.REINITIALIZED:
            raise ValueError("slot is not in the reinitialized state")
        return self._reinitialized_parameters[identifier]

    def structural_mask(
        self,
        source_layer_id: str,
        destination_layer_id: str,
    ) -> np.ndarray:
        """Return the fixed structural route mask for two declared layers."""

        return structural_slack_to_core_mask(
            self.states_for_layer(source_layer_id),
            self.states_for_layer(destination_layer_id),
        )

    def source_owned_masks(
        self,
        layer_id: str,
        *,
        input_width: int,
        output_width: int,
    ) -> SourceOwnedParameterMasks:
        """Return complete source-owned masks in physical slot order."""

        slot_ids = self.layer_slot_ids(layer_id)
        return source_owned_parameter_masks(
            slot_ids,
            self.states_for_layer(layer_id),
            input_width=input_width,
            output_width=output_width,
        )


def structural_slack_to_core_mask(
    source_states: Sequence[SlotState | str],
    destination_states: Sequence[SlotState | str],
) -> np.ndarray:
    """Return a binary route mask with every plastic-to-core route removed.

    Slack, released, and reinitialized sources are all treated as plastic-side
    slots. This makes routes into downstream core slots structurally zero
    throughout release and reinitialization, not only after activation.
    """

    sources = tuple(
        _slot_state(value, name=f"source_states[{index}]")
        for index, value in enumerate(source_states)
    )
    destinations = tuple(
        _slot_state(value, name=f"destination_states[{index}]")
        for index, value in enumerate(destination_states)
    )
    if not sources or not destinations:
        raise ValueError("source_states and destination_states must be nonempty")
    mask = np.ones((len(sources), len(destinations)), dtype=np.float64)
    for source_index, source_state in enumerate(sources):
        if source_state is SlotState.CORE:
            continue
        for destination_index, destination_state in enumerate(destinations):
            if destination_state is SlotState.CORE:
                mask[source_index, destination_index] = 0.0
    return mask


@dataclass(frozen=True)
class SourceOwnedParameterMasks:
    """Update masks for every parameter owned by each source slot."""

    slot_ids: tuple[str, ...]
    trainable_by_slot: np.ndarray
    incoming_weight: np.ndarray
    incoming_bias: np.ndarray
    bn_scale: np.ndarray
    bn_shift: np.ndarray
    outgoing_weight: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return writable copies keyed by canonical parameter name."""

        return {
            "incoming_weight": np.array(self.incoming_weight, copy=True),
            "incoming_bias": np.array(self.incoming_bias, copy=True),
            "bn_scale": np.array(self.bn_scale, copy=True),
            "bn_shift": np.array(self.bn_shift, copy=True),
            "outgoing_weight": np.array(self.outgoing_weight, copy=True),
        }


def source_owned_parameter_masks(
    slot_ids: Sequence[str],
    states: Sequence[SlotState | str],
    *,
    input_width: int,
    output_width: int,
) -> SourceOwnedParameterMasks:
    """Mask each complete core operator, including all outgoing routes.

    Core and released slots have update mask zero. Slack and reinitialized
    slots have update mask one. Outgoing ownership follows the source axis, so
    a core source's entire outgoing row is zero even for slack destinations.
    """

    identifiers = _slot_ids(slot_ids, name="slot_ids")
    parsed_states = tuple(
        _slot_state(value, name=f"states[{index}]")
        for index, value in enumerate(states)
    )
    if len(parsed_states) != len(identifiers):
        raise ValueError("states length must equal slot_ids length")
    input_width = _positive_width(input_width, name="input_width")
    output_width = _positive_width(output_width, name="output_width")
    trainable = np.asarray(
        [
            1.0
            if state in (SlotState.SLACK, SlotState.REINITIALIZED)
            else 0.0
            for state in parsed_states
        ],
        dtype=np.float64,
    )
    incoming = np.repeat(trainable[:, np.newaxis], input_width, axis=1)
    outgoing = np.repeat(trainable[:, np.newaxis], output_width, axis=1)
    arrays = (trainable, incoming, outgoing)
    for array in arrays:
        array.setflags(write=False)
    return SourceOwnedParameterMasks(
        slot_ids=identifiers,
        trainable_by_slot=trainable,
        incoming_weight=incoming,
        incoming_bias=trainable,
        bn_scale=trainable,
        bn_shift=trainable,
        outgoing_weight=outgoing,
    )


def _validate_tensor_mappings(
    reference: Mapping[str, ArrayLike],
    *others: tuple[str, Mapping[str, ArrayLike]],
) -> tuple[dict[str, np.ndarray], ...]:
    if not isinstance(reference, Mapping) or not reference:
        raise ValueError("parameters must be a nonempty mapping")
    keys = tuple(reference)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("parameter names must be nonempty strings")
    converted: list[dict[str, np.ndarray]] = [
        {
            key: _finite_array(
                reference[key], name=f"parameters[{key!r}]", nonempty=True
            )
            for key in keys
        }
    ]
    for mapping_name, mapping in others:
        if not isinstance(mapping, Mapping) or set(mapping) != set(keys):
            raise ValueError(f"{mapping_name} keys must equal parameter keys")
        converted.append(
            {
                key: _finite_array(
                    mapping[key],
                    name=f"{mapping_name}[{key!r}]",
                    shape=converted[0][key].shape,
                    nonempty=True,
                )
                for key in keys
            }
        )
    return tuple(converted)


def clear_optimizer_state(
    optimizer_state: Mapping[str, ArrayLike],
    update_masks: Mapping[str, ArrayLike],
    *,
    reset_masks: Mapping[str, ArrayLike] | None = None,
) -> dict[str, np.ndarray]:
    """Clear masked state and optional stale state for reused parameters.

    ``update_masks`` uses one for trainable entries and zero for frozen or
    inactive entries. ``reset_masks``, when supplied, uses one for entries
    whose history must be cleared during release or reinitialization.
    Inputs are not mutated.
    """

    if not isinstance(optimizer_state, Mapping) or not optimizer_state:
        raise ValueError("optimizer_state must be a nonempty mapping")
    if not isinstance(update_masks, Mapping) or set(update_masks) != set(optimizer_state):
        raise ValueError("update_masks keys must equal optimizer_state keys")
    if reset_masks is not None and (
        not isinstance(reset_masks, Mapping)
        or set(reset_masks) != set(optimizer_state)
    ):
        raise ValueError("reset_masks keys must equal optimizer_state keys")

    cleared: dict[str, np.ndarray] = {}
    for name, raw_state in optimizer_state.items():
        state = _finite_array(raw_state, name=f"optimizer_state[{name!r}]")
        update = _binary_mask(
            update_masks[name],
            name=f"update_masks[{name!r}]",
            shape=state.shape,
        )
        clear_where = update == 0.0
        if reset_masks is not None:
            reset = _binary_mask(
                reset_masks[name],
                name=f"reset_masks[{name!r}]",
                shape=state.shape,
            )
            clear_where = clear_where | (reset == 1.0)
        cleared[name] = np.where(clear_where, 0.0, state)
    return cleared


@dataclass(frozen=True)
class ParameterDriftDiagnostic:
    """Observed drift for one masked optimizer tensor."""

    parameter_name: str
    masked_element_count: int
    trainable_element_count: int
    max_abs_masked_drift: float
    max_abs_trainable_drift: float
    max_abs_masked_optimizer_state: float
    masked_parameters_unchanged: bool


@dataclass(frozen=True)
class MomentumWStepResult:
    """Updated tensors, cleared momentum state, and drift diagnostics."""

    parameters: dict[str, np.ndarray]
    optimizer_state: dict[str, np.ndarray]
    diagnostics: tuple[ParameterDriftDiagnostic, ...]

    @property
    def max_abs_masked_drift(self) -> float:
        return max(
            (diagnostic.max_abs_masked_drift for diagnostic in self.diagnostics),
            default=0.0,
        )

    @property
    def all_masked_parameters_unchanged(self) -> bool:
        return all(
            diagnostic.masked_parameters_unchanged
            for diagnostic in self.diagnostics
        )


def momentumw_step(
    parameters: Mapping[str, ArrayLike],
    gradients: Mapping[str, ArrayLike],
    optimizer_state: Mapping[str, ArrayLike],
    update_masks: Mapping[str, ArrayLike],
    *,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> MomentumWStepResult:
    """Apply deterministic momentum and masked decoupled weight decay.

    Masked entries are selected from the original parameters with
    ``np.where``. They therefore receive neither gradient, stale momentum, nor
    decoupled weight decay. Their returned momentum is exactly zero.
    """

    rate = _finite_scalar(learning_rate, name="learning_rate")
    coefficient = _finite_scalar(momentum, name="momentum")
    decay = _finite_scalar(weight_decay, name="weight_decay")
    if rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= coefficient < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if decay < 0.0:
        raise ValueError("weight_decay must be nonnegative")
    if not math.isfinite(rate * decay):
        raise ValueError("learning_rate times weight_decay must be finite")

    converted = _validate_tensor_mappings(
        parameters,
        ("gradients", gradients),
        ("optimizer_state", optimizer_state),
        ("update_masks", update_masks),
    )
    parameter_arrays, gradient_arrays, state_arrays, raw_masks = converted
    masks = {
        name: _binary_mask(
            raw_masks[name],
            name=f"update_masks[{name!r}]",
            shape=parameter_arrays[name].shape,
        )
        for name in parameter_arrays
    }
    cleared_state = clear_optimizer_state(state_arrays, masks)

    updated_parameters: dict[str, np.ndarray] = {}
    updated_state: dict[str, np.ndarray] = {}
    diagnostics: list[ParameterDriftDiagnostic] = []
    for name, parameter in parameter_arrays.items():
        mask = masks[name]
        trainable = mask == 1.0
        masked = ~trainable
        with np.errstate(over="raise", invalid="raise"):
            velocity = coefficient * cleared_state[name] + gradient_arrays[name] * mask
            decay_parameter = np.where(trainable, parameter, 0.0)
            candidate = parameter - rate * velocity - rate * decay * decay_parameter
            updated = np.where(trainable, candidate, parameter)
            velocity = np.where(trainable, velocity, 0.0)
        if not np.all(np.isfinite(updated)) or not np.all(np.isfinite(velocity)):
            raise FloatingPointError(f"optimizer step for {name!r} was non-finite")
        drift = np.abs(updated - parameter)
        masked_drift = float(np.max(drift[masked])) if np.any(masked) else 0.0
        trainable_drift = (
            float(np.max(drift[trainable])) if np.any(trainable) else 0.0
        )
        masked_state = (
            float(np.max(np.abs(velocity[masked]))) if np.any(masked) else 0.0
        )
        unchanged = bool(np.array_equal(updated[masked], parameter[masked]))
        updated_parameters[name] = updated
        updated_state[name] = velocity
        diagnostics.append(
            ParameterDriftDiagnostic(
                parameter_name=name,
                masked_element_count=int(np.count_nonzero(masked)),
                trainable_element_count=int(np.count_nonzero(trainable)),
                max_abs_masked_drift=masked_drift,
                max_abs_trainable_drift=trainable_drift,
                max_abs_masked_optimizer_state=masked_state,
                masked_parameters_unchanged=unchanged,
            )
        )
    return MomentumWStepResult(
        parameters=updated_parameters,
        optimizer_state=updated_state,
        diagnostics=tuple(diagnostics),
    )


@dataclass(frozen=True)
class BatchNormOwnership:
    """Trainable and running-buffer ownership for one BatchNorm layer.

    Scale and shift channel ``i`` are owned by ``channel_slot_ids[i]``. Under
    ``frozen``, both tasks select the source buffer track and no adaptation
    update is applied. Under ``dual_track``, each task owns a distinct track.
    """

    channel_slot_ids: tuple[str, ...]
    buffer_policy: BatchNormBufferPolicy
    source_task_id: str
    target_task_id: str
    source_track_id: str
    target_track_id: str | None = None

    def __post_init__(self) -> None:
        slots = _slot_ids(self.channel_slot_ids, name="channel_slot_ids")
        policy = _buffer_policy(self.buffer_policy, name="buffer_policy")
        source_task = _identifier(self.source_task_id, name="source_task_id")
        target_task = _identifier(self.target_task_id, name="target_task_id")
        source_track = _identifier(self.source_track_id, name="source_track_id")
        if source_task == target_task:
            raise ValueError("source_task_id and target_task_id must differ")
        target_track = self.target_track_id
        if policy is BatchNormBufferPolicy.FROZEN:
            if target_track is not None:
                raise ValueError("frozen policy must not declare a target track")
        else:
            if target_track is None:
                raise ValueError("dual_track policy requires a target track")
            target_track = _identifier(target_track, name="target_track_id")
            if target_track == source_track:
                raise ValueError("dual tracks must have distinct IDs")
        object.__setattr__(self, "channel_slot_ids", slots)
        object.__setattr__(self, "buffer_policy", policy)
        object.__setattr__(self, "source_task_id", source_task)
        object.__setattr__(self, "target_task_id", target_task)
        object.__setattr__(self, "source_track_id", source_track)
        object.__setattr__(self, "target_track_id", target_track)

    @property
    def channel_count(self) -> int:
        return len(self.channel_slot_ids)

    @property
    def trainable_owners(self) -> dict[str, tuple[str, ...]]:
        return {
            "scale": self.channel_slot_ids,
            "shift": self.channel_slot_ids,
        }

    @property
    def buffer_owners(self) -> dict[str, dict[str, str]]:
        owners = {self.source_track_id: self.source_task_id}
        if self.buffer_policy is BatchNormBufferPolicy.DUAL_TRACK:
            assert self.target_track_id is not None
            owners[self.target_track_id] = self.target_task_id
        return {
            "running_mean": dict(owners),
            "running_variance": dict(owners),
        }

    @property
    def track_ids(self) -> tuple[str, ...]:
        if self.buffer_policy is BatchNormBufferPolicy.FROZEN:
            return (self.source_track_id,)
        assert self.target_track_id is not None
        return (self.source_track_id, self.target_track_id)

    def track_for_task(self, task_id: str) -> str:
        task = _identifier(task_id, name="task_id")
        if task == self.source_task_id:
            return self.source_track_id
        if task == self.target_task_id:
            if self.buffer_policy is BatchNormBufferPolicy.FROZEN:
                return self.source_track_id
            assert self.target_track_id is not None
            return self.target_track_id
        raise KeyError(f"unknown task ID for BatchNorm ownership: {task}")


@dataclass(frozen=True)
class BatchNormBuffers:
    """One immutable running-mean and running-variance track."""

    running_mean: np.ndarray
    running_variance: np.ndarray

    def __post_init__(self) -> None:
        mean = _readonly_vector(self.running_mean, name="running_mean")
        variance = _readonly_vector(
            self.running_variance,
            name="running_variance",
            length=len(mean),
        )
        if np.any(variance < 0.0):
            raise ValueError("running_variance must be nonnegative")
        object.__setattr__(self, "running_mean", mean)
        object.__setattr__(self, "running_variance", variance)


@dataclass(frozen=True)
class StateDeclaration:
    """Complete task-conditional inference state declaration."""

    state_id: str
    task_id: str
    head_id: str
    batch_norm_track_id: str
    slot_ids: tuple[str, ...]
    slot_mask: tuple[float, ...]

    def __post_init__(self) -> None:
        state_id = _identifier(self.state_id, name="state_id")
        task_id = _identifier(self.task_id, name="task_id")
        head_id = _identifier(self.head_id, name="head_id")
        track_id = _identifier(self.batch_norm_track_id, name="batch_norm_track_id")
        slots = _slot_ids(self.slot_ids, name="slot_ids")
        mask = _binary_mask(
            self.slot_mask,
            name="slot_mask",
            shape=(len(slots),),
        )
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "head_id", head_id)
        object.__setattr__(self, "batch_norm_track_id", track_id)
        object.__setattr__(self, "slot_ids", slots)
        object.__setattr__(self, "slot_mask", tuple(float(value) for value in mask))


@dataclass(frozen=True)
class EvaluationDeclaration:
    """Dataset, metric, aggregation, checkpoint, and state used in evaluation."""

    evaluation_id: str
    state_id: str
    task_id: str
    dataset_id: str
    metric_name: str
    aggregation: str
    checkpoint_rule: str

    def __post_init__(self) -> None:
        for field_name in (
            "evaluation_id",
            "state_id",
            "task_id",
            "dataset_id",
            "metric_name",
            "aggregation",
            "checkpoint_rule",
        ):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), name=field_name),
            )


class TaskStateSelector:
    """Resolve evaluations to declared task state and BatchNorm track."""

    def __init__(
        self,
        states: Sequence[StateDeclaration],
        *,
        batch_norm_ownership: BatchNormOwnership | None = None,
    ) -> None:
        declarations = tuple(states)
        if not declarations:
            raise ValueError("states must be nonempty")
        if not all(isinstance(state, StateDeclaration) for state in declarations):
            raise TypeError("states must contain only StateDeclaration values")
        state_ids = tuple(state.state_id for state in declarations)
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("state IDs must be unique")
        if batch_norm_ownership is not None:
            if not isinstance(batch_norm_ownership, BatchNormOwnership):
                raise TypeError("batch_norm_ownership must be BatchNormOwnership")
            for state in declarations:
                expected_track = batch_norm_ownership.track_for_task(state.task_id)
                if state.batch_norm_track_id != expected_track:
                    raise ValueError(
                        f"state {state.state_id!r} declares track "
                        f"{state.batch_norm_track_id!r}; policy requires "
                        f"{expected_track!r}"
                    )
        self._states = {state.state_id: state for state in declarations}
        self._batch_norm_ownership = batch_norm_ownership

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(self._states)

    def state(self, state_id: str) -> StateDeclaration:
        identifier = _identifier(state_id, name="state_id")
        try:
            return self._states[identifier]
        except KeyError as error:
            raise KeyError(f"unknown state ID: {identifier}") from error

    def select(self, evaluation: EvaluationDeclaration) -> StateDeclaration:
        if not isinstance(evaluation, EvaluationDeclaration):
            raise TypeError("evaluation must be an EvaluationDeclaration")
        state = self.state(evaluation.state_id)
        if state.task_id != evaluation.task_id:
            raise ValueError(
                f"evaluation task {evaluation.task_id!r} does not match "
                f"state task {state.task_id!r}"
            )
        return state

    def batch_norm_track(self, evaluation: EvaluationDeclaration) -> str:
        return self.select(evaluation).batch_norm_track_id


@dataclass(frozen=True)
class BatchNormState:
    """Immutable affine values and explicitly owned running-buffer tracks."""

    ownership: BatchNormOwnership
    scale: np.ndarray
    shift: np.ndarray
    buffers: Mapping[str, BatchNormBuffers]
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.ownership, BatchNormOwnership):
            raise TypeError("ownership must be BatchNormOwnership")
        channel_count = self.ownership.channel_count
        scale = _readonly_vector(self.scale, name="scale", length=channel_count)
        shift = _readonly_vector(self.shift, name="shift", length=channel_count)
        epsilon = _finite_scalar(self.epsilon, name="epsilon")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not isinstance(self.buffers, Mapping):
            raise TypeError("buffers must be a mapping keyed by track ID")
        if set(self.buffers) != set(self.ownership.track_ids):
            raise ValueError("buffer keys must equal the declared owned track IDs")
        copied_buffers: dict[str, BatchNormBuffers] = {}
        for track_id in self.ownership.track_ids:
            buffer = self.buffers[track_id]
            if not isinstance(buffer, BatchNormBuffers):
                raise TypeError("buffers must contain BatchNormBuffers values")
            if len(buffer.running_mean) != channel_count:
                raise ValueError("buffer channel count must equal ownership channel count")
            copied_buffers[track_id] = BatchNormBuffers(
                buffer.running_mean,
                buffer.running_variance,
            )
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "buffers", copied_buffers)

    def updated_for_task(
        self,
        batch: ArrayLike,
        *,
        task_id: str,
        update_rate: float,
    ) -> BatchNormState:
        """Return a state after one declared task-buffer update.

        Frozen ownership returns an equal independent state. Dual-track
        ownership updates only the selected task track.
        """

        values = _finite_array(batch, name="batch", ndim=2)
        if values.shape[1] != self.ownership.channel_count:
            raise ValueError("batch channel count must equal BatchNorm channel count")
        rate = _finite_scalar(update_rate, name="update_rate")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("update_rate must be in [0, 1]")
        track_id = self.ownership.track_for_task(task_id)
        copied = {
            name: BatchNormBuffers(buffer.running_mean, buffer.running_variance)
            for name, buffer in self.buffers.items()
        }
        if self.ownership.buffer_policy is BatchNormBufferPolicy.FROZEN:
            return BatchNormState(
                ownership=self.ownership,
                scale=self.scale,
                shift=self.shift,
                buffers=copied,
                epsilon=self.epsilon,
            )

        batch_mean = np.mean(values, axis=0, dtype=np.float64)
        batch_variance = np.var(values, axis=0, dtype=np.float64)
        old = self.buffers[track_id]
        with np.errstate(over="raise", invalid="raise"):
            new_mean = (1.0 - rate) * old.running_mean + rate * batch_mean
            new_variance = (
                (1.0 - rate) * old.running_variance + rate * batch_variance
            )
        if not np.all(np.isfinite(new_mean)) or not np.all(np.isfinite(new_variance)):
            raise FloatingPointError("BatchNorm buffer update was non-finite")
        copied[track_id] = BatchNormBuffers(new_mean, new_variance)
        return BatchNormState(
            ownership=self.ownership,
            scale=self.scale,
            shift=self.shift,
            buffers=copied,
            epsilon=self.epsilon,
        )

    def infer(self, values: ArrayLike, *, track_id: str) -> np.ndarray:
        """Normalize with one explicitly selected owned track."""

        inputs = _finite_array(values, name="values")
        if inputs.ndim < 1 or inputs.shape[-1] != self.ownership.channel_count:
            raise ValueError("values last dimension must equal BatchNorm channel count")
        track = _identifier(track_id, name="track_id")
        if track not in self.buffers:
            raise KeyError(f"unknown BatchNorm track ID: {track}")
        buffer = self.buffers[track]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            denominator = np.sqrt(buffer.running_variance + self.epsilon)
            outputs = (
                (inputs - buffer.running_mean) / denominator * self.scale + self.shift
            )
        if not np.all(np.isfinite(outputs)):
            raise FloatingPointError("BatchNorm inference result was non-finite")
        return np.array(outputs, copy=True)

    def infer_for_evaluation(
        self,
        values: ArrayLike,
        evaluation: EvaluationDeclaration,
        selector: TaskStateSelector,
    ) -> np.ndarray:
        """Infer using the task state named by an evaluation declaration."""

        if not isinstance(selector, TaskStateSelector):
            raise TypeError("selector must be a TaskStateSelector")
        track = selector.batch_norm_track(evaluation)
        expected = self.ownership.track_for_task(evaluation.task_id)
        if track != expected:
            raise ValueError("selected BatchNorm track conflicts with ownership policy")
        return self.infer(values, track_id=track)


__all__ = [
    "BatchNormBufferPolicy",
    "BatchNormBuffers",
    "BatchNormOwnership",
    "BatchNormState",
    "EvaluationDeclaration",
    "FixedWidthDeftStateMachine",
    "GradientLivenessTrace",
    "LayerDefinition",
    "LayerPartition",
    "MomentumWStepResult",
    "ParameterDriftDiagnostic",
    "ReinitializedSlotParameters",
    "SlotState",
    "SourceOwnedParameterMasks",
    "StateDeclaration",
    "TaskStateSelector",
    "clear_optimizer_state",
    "initialize_released_slot",
    "layerwise_top_k_current_capacity_partition",
    "momentumw_step",
    "source_owned_parameter_masks",
    "structural_slack_to_core_mask",
    "top_k_current_capacity_partition",
    "two_stage_gradient_liveness",
]
