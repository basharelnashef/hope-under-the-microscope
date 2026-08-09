"""Fail-closed graph and accounting primitives for the HOPE-R repair.

This module is an independent proposed repair.  It does not implement a
framework graph tracer and it does not infer topology from tensor names.  A
caller supplies every active channel, every input slot, and every consumer.
That explicit boundary lets physical channel surgery either produce a complete
dependency plan or stop before changing a model.

The resource counts below are exact counts for declared abstract records and
array elements.  They are not wall-clock, allocator, or device-memory claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np


SUPPORTED_OPERATORS = frozenset({"dense", "conv1d", "conv2d", "conv3d"})
SUPPORTED_ACTIONS = frozenset({"prune", "merge"})


class GraphValidationError(ValueError):
    """Raised when an explicit graph declaration is internally inconsistent."""


class UnsupportedGraphError(GraphValidationError):
    """Raised when safe dependency or payoff semantics are not implemented."""


@dataclass(frozen=True, order=True, slots=True)
class ChannelRef:
    """Stable identity of one physical output channel.

    ``index`` is an identity token, not a promise that active channels remain
    densely numbered after surgery.  Both fields are immutable.
    """

    layer_id: str
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id:
            raise GraphValidationError("channel layer_id must be a non-empty string")
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise GraphValidationError("channel index must be an integer")
        if self.index < 0:
            raise GraphValidationError("channel index must be non-negative")

    @property
    def token(self) -> str:
        """Return an unambiguous deterministic text token."""

        return f"{len(self.layer_id)}:{self.layer_id}:{self.index}"

    def as_record(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible record."""

        return {"layer_id": self.layer_id, "index": self.index}


@dataclass(frozen=True, order=True, slots=True)
class ConsumerRef:
    """One direct input coordinate consumed by a downstream layer."""

    layer_id: str
    input_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id:
            raise GraphValidationError("consumer layer_id must be a non-empty string")
        if isinstance(self.input_index, bool) or not isinstance(self.input_index, int):
            raise GraphValidationError("consumer input_index must be an integer")
        if self.input_index < 0:
            raise GraphValidationError("consumer input_index must be non-negative")

    def as_record(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible record."""

        return {"layer_id": self.layer_id, "input_index": self.input_index}


def _positive_shape(shape: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    values = tuple(shape)
    if not values:
        raise GraphValidationError(f"{field_name} must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GraphValidationError(f"{field_name} must contain positive integers")
    return values


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """Explicit live physical layer declaration.

    ``input_sources`` has one entry for every live input coordinate.  ``None``
    denotes an external graph input; a :class:`ChannelRef` denotes a graph
    edge that must also appear in ``GraphModel.consumer_map``.  Initial widths
    preserve the paper's static-payoff baseline while the tuple lengths are
    the live physical widths.
    """

    layer_id: str
    channels: tuple[ChannelRef, ...]
    input_sources: tuple[ChannelRef | None, ...]
    initial_input_count: int | None = None
    initial_output_count: int | None = None
    operator: str = "dense"
    kernel_shape: tuple[int, ...] = (1,)
    use_bias: bool = False
    bn_values_per_channel: int = 4
    terminal: bool = False
    groups: int = 1
    depthwise: bool = False
    shared_parameters: bool = False
    parameter_owner: str | None = None
    consumers_resolved: bool = True
    supported: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id:
            raise GraphValidationError("layer_id must be a non-empty string")

        channels = tuple(self.channels)
        inputs = tuple(self.input_sources)
        kernel = _positive_shape(self.kernel_shape, field_name="kernel_shape")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "input_sources", inputs)
        object.__setattr__(self, "kernel_shape", kernel)

        if not channels:
            raise GraphValidationError(f"layer {self.layer_id!r} has no live channels")
        if len(set(channels)) != len(channels):
            raise GraphValidationError(f"layer {self.layer_id!r} repeats a channel")
        if tuple(sorted(channels)) != channels:
            raise GraphValidationError(
                f"layer {self.layer_id!r} channels must be in deterministic order"
            )
        if any(channel.layer_id != self.layer_id for channel in channels):
            raise GraphValidationError(
                f"layer {self.layer_id!r} owns a channel with a different layer_id"
            )
        if not inputs:
            raise GraphValidationError(f"layer {self.layer_id!r} has no live inputs")
        if any(source is not None and not isinstance(source, ChannelRef) for source in inputs):
            raise GraphValidationError(
                f"layer {self.layer_id!r} input_sources must contain ChannelRef or None"
            )

        initial_in = len(inputs) if self.initial_input_count is None else self.initial_input_count
        initial_out = len(channels) if self.initial_output_count is None else self.initial_output_count
        for name, value, live in (
            ("initial_input_count", initial_in, len(inputs)),
            ("initial_output_count", initial_out, len(channels)),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < live:
                raise GraphValidationError(
                    f"layer {self.layer_id!r} {name} must be an integer at least {live}"
                )
        object.__setattr__(self, "initial_input_count", initial_in)
        object.__setattr__(self, "initial_output_count", initial_out)

        for name, value in (
            ("groups", self.groups),
            ("bn_values_per_channel", self.bn_values_per_channel),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise GraphValidationError(f"{name} must be an integer")
        if self.groups <= 0:
            raise GraphValidationError("groups must be positive")
        if self.bn_values_per_channel < 0:
            raise GraphValidationError("bn_values_per_channel must be non-negative")

        owner = self.layer_id if self.parameter_owner is None else self.parameter_owner
        if not isinstance(owner, str) or not owner:
            raise GraphValidationError("parameter_owner must be a non-empty string")
        object.__setattr__(self, "parameter_owner", owner)

    @property
    def live_input_count(self) -> int:
        return len(self.input_sources)

    @property
    def live_output_count(self) -> int:
        return len(self.channels)

    @property
    def kernel_volume(self) -> int:
        return math.prod(self.kernel_shape)


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """Structural declaration for one candidate residual-block eviction."""

    block_id: str
    spanned_layers: tuple[str, ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    stride: int | tuple[int, ...] = 1
    architecture: str = "resnet_v1"
    activation_order: str = "post_activation"
    shortcut_kind: str = "identity"

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise GraphValidationError("block_id must be a non-empty string")
        layers = tuple(self.spanned_layers)
        if not layers or any(not isinstance(layer, str) or not layer for layer in layers):
            raise GraphValidationError("spanned_layers must contain layer identifiers")
        if len(set(layers)) != len(layers):
            raise GraphValidationError("spanned_layers must not repeat a layer")
        object.__setattr__(self, "spanned_layers", layers)
        object.__setattr__(
            self,
            "input_shape",
            _positive_shape(self.input_shape, field_name="input_shape"),
        )
        object.__setattr__(
            self,
            "output_shape",
            _positive_shape(self.output_shape, field_name="output_shape"),
        )
        raw_stride = (self.stride,) if isinstance(self.stride, int) else tuple(self.stride)
        stride = _positive_shape(raw_stride, field_name="stride")
        object.__setattr__(self, "stride", stride)


@dataclass(frozen=True, slots=True)
class BlockEligibility:
    """Machine-readable result of the deliberately narrow block gate."""

    eligible: bool
    reasons: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.eligible

    def as_record(self) -> dict[str, object]:
        return {"eligible": self.eligible, "reasons": list(self.reasons)}


def check_identity_block(spec: BlockSpec) -> BlockEligibility:
    """Accept only same-shape stride-1 post-activation ResNet V1 identity blocks."""

    if not isinstance(spec, BlockSpec):
        raise TypeError("spec must be a BlockSpec")
    reasons: list[str] = []
    if spec.architecture != "resnet_v1":
        reasons.append("architecture_not_resnet_v1")
    if spec.activation_order != "post_activation":
        reasons.append("not_post_activation")
    if spec.shortcut_kind == "none":
        reasons.append("missing_skip")
    elif spec.shortcut_kind != "identity":
        reasons.append("non_identity_shortcut")
    if any(value != 1 for value in spec.stride):
        reasons.append("non_unit_stride")
    if spec.input_shape != spec.output_shape:
        reasons.append("shape_mismatch")
    return BlockEligibility(eligible=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True, order=True, slots=True)
class CacheKey:
    """Canonical identity of one dependency-sensitive cached quantity."""

    kind: str
    owner_id: str
    channels: tuple[ChannelRef, ...] = ()
    detail: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise GraphValidationError("cache-key kind must be a non-empty string")
        if not isinstance(self.owner_id, str) or not self.owner_id:
            raise GraphValidationError("cache-key owner_id must be a non-empty string")
        channels = tuple(self.channels)
        detail = tuple(self.detail)
        if tuple(sorted(channels)) != channels or len(set(channels)) != len(channels):
            raise GraphValidationError("cache-key channels must be unique and sorted")
        if any(not isinstance(value, str) for value in detail):
            raise GraphValidationError("cache-key detail values must be strings")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "detail", detail)

    @classmethod
    def pair(cls, kind: str, layer_id: str, first: ChannelRef, second: ChannelRef) -> "CacheKey":
        if first == second:
            raise GraphValidationError("a pair cache key requires distinct channels")
        return cls(kind=kind, owner_id=layer_id, channels=tuple(sorted((first, second))))

    def as_record(self) -> dict[str, object]:
        """Return a stable JSON record with a fixed schema."""

        return {
            "kind": self.kind,
            "owner_id": self.owner_id,
            "channels": [channel.as_record() for channel in self.channels],
            "detail": list(self.detail),
        }

    def canonical_bytes(self) -> bytes:
        """Serialize the fixed record deterministically."""

        return json.dumps(
            self.as_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class GraphModel:
    """Validated, explicit graph topology for physical channel surgery.

    Grouped, depthwise, shared-parameter, unresolved, and unknown operators are
    rejected.  Explicit fan-out is supported: every branch must appear both in
    the producer's consumer tuple and in the consumer's ``input_sources``.
    """

    layers: tuple[LayerSpec, ...]
    consumer_map: Mapping[ChannelRef, Sequence[ConsumerRef]]
    blocks: tuple[BlockSpec, ...] = ()
    consumer_map_complete: bool = True
    _layers_by_id: Mapping[str, LayerSpec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        blocks = tuple(self.blocks)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "blocks", blocks)
        if not layers:
            raise GraphValidationError("graph must contain at least one layer")
        if not self.consumer_map_complete:
            raise UnsupportedGraphError("consumer map is not declared complete")

        layer_ids = [layer.layer_id for layer in layers]
        if len(set(layer_ids)) != len(layer_ids):
            raise GraphValidationError("graph repeats a layer_id")
        if tuple(sorted(layer_ids)) != tuple(layer_ids):
            raise GraphValidationError("layers must be in deterministic layer_id order")
        by_id = {layer.layer_id: layer for layer in layers}

        parameter_owners: dict[str, str] = {}
        for layer in layers:
            if not layer.supported or layer.operator not in SUPPORTED_OPERATORS:
                raise UnsupportedGraphError(
                    f"unsupported operator for layer {layer.layer_id!r}: {layer.operator!r}"
                )
            if layer.groups != 1:
                raise UnsupportedGraphError(
                    f"grouped operators are unresolved for layer {layer.layer_id!r}"
                )
            if layer.depthwise:
                raise UnsupportedGraphError(
                    f"depthwise operators are unresolved for layer {layer.layer_id!r}"
                )
            if layer.shared_parameters:
                raise UnsupportedGraphError(
                    f"shared parameters are unresolved for layer {layer.layer_id!r}"
                )
            if not layer.consumers_resolved:
                raise UnsupportedGraphError(
                    f"fan-out is unresolved for layer {layer.layer_id!r}"
                )
            previous = parameter_owners.get(layer.parameter_owner)
            if previous is not None:
                raise UnsupportedGraphError(
                    f"layers {previous!r} and {layer.layer_id!r} share parameters"
                )
            parameter_owners[layer.parameter_owner] = layer.layer_id

        active_channels = {
            channel for layer in layers for channel in layer.channels
        }
        supplied_keys = set(self.consumer_map)
        if supplied_keys != active_channels:
            missing = sorted(active_channels - supplied_keys)
            extra = sorted(supplied_keys - active_channels)
            raise GraphValidationError(
                f"consumer map must name every active channel exactly; missing={missing}, extra={extra}"
            )

        canonical_map: dict[ChannelRef, tuple[ConsumerRef, ...]] = {}
        seen_slots: dict[ConsumerRef, ChannelRef] = {}
        for channel in sorted(active_channels):
            raw_consumers = tuple(self.consumer_map[channel])
            if any(not isinstance(item, ConsumerRef) for item in raw_consumers):
                raise GraphValidationError("consumer map values must contain ConsumerRef")
            consumers = tuple(sorted(raw_consumers))
            if len(set(consumers)) != len(consumers):
                raise GraphValidationError(f"consumer map repeats an edge for {channel}")
            owner = by_id[channel.layer_id]
            if owner.terminal and consumers:
                raise GraphValidationError(
                    f"terminal layer {owner.layer_id!r} declares downstream consumers"
                )
            if not owner.terminal and not consumers:
                raise UnsupportedGraphError(
                    f"nonterminal channel {channel.token} has unresolved or empty fan-out"
                )
            for consumer in consumers:
                target = by_id.get(consumer.layer_id)
                if target is None:
                    raise GraphValidationError(
                        f"consumer refers to unknown layer {consumer.layer_id!r}"
                    )
                if consumer.input_index >= target.live_input_count:
                    raise GraphValidationError(
                        f"consumer input index is outside layer {consumer.layer_id!r}"
                    )
                if target.input_sources[consumer.input_index] != channel:
                    raise GraphValidationError(
                        f"consumer edge {channel.token} -> {consumer} disagrees with input_sources"
                    )
                previous_source = seen_slots.get(consumer)
                if previous_source is not None:
                    raise GraphValidationError(
                        f"input slot {consumer} has multiple direct producers"
                    )
                seen_slots[consumer] = channel
            canonical_map[channel] = consumers

        for target in layers:
            for input_index, source in enumerate(target.input_sources):
                if source is None:
                    continue
                if source not in active_channels:
                    raise GraphValidationError(
                        f"layer {target.layer_id!r} input references an inactive channel"
                    )
                slot = ConsumerRef(target.layer_id, input_index)
                if seen_slots.get(slot) != source:
                    raise UnsupportedGraphError(
                        f"consumer map omits the declared edge {source.token} -> {slot}"
                    )

        block_ids: set[str] = set()
        for block in blocks:
            if block.block_id in block_ids:
                raise GraphValidationError("graph repeats a block_id")
            block_ids.add(block.block_id)
            unknown_layers = set(block.spanned_layers) - set(by_id)
            if unknown_layers:
                raise GraphValidationError(
                    f"block {block.block_id!r} spans unknown layers {sorted(unknown_layers)}"
                )

        object.__setattr__(self, "consumer_map", MappingProxyType(canonical_map))
        object.__setattr__(self, "_layers_by_id", MappingProxyType(by_id))

    def layer(self, layer_id: str) -> LayerSpec:
        try:
            return self._layers_by_id[layer_id]
        except KeyError as error:
            raise GraphValidationError(f"unknown layer_id {layer_id!r}") from error

    def has_channel(self, channel: ChannelRef) -> bool:
        layer = self._layers_by_id.get(channel.layer_id)
        return layer is not None and channel in layer.channels

    @property
    def live_widths(self) -> tuple[tuple[str, int], ...]:
        return tuple((layer.layer_id, layer.live_output_count) for layer in self.layers)

    def pair_cache_keys(self, *, kind: str = "pair_geometry") -> tuple[CacheKey, ...]:
        keys: list[CacheKey] = []
        for layer in self.layers:
            for position, first in enumerate(layer.channels):
                for second in layer.channels[position + 1 :]:
                    keys.append(CacheKey.pair(kind, layer.layer_id, first, second))
        return tuple(sorted(keys))


def _channel_key(kind: str, channel: ChannelRef) -> CacheKey:
    return CacheKey(kind=kind, owner_id=channel.layer_id, channels=(channel,))


def _pair_keys(kind: str, layer: LayerSpec) -> tuple[CacheKey, ...]:
    return tuple(
        CacheKey.pair(kind, layer.layer_id, first, second)
        for position, first in enumerate(layer.channels)
        for second in layer.channels[position + 1 :]
    )


def cache_universe(graph: GraphModel) -> tuple[CacheKey, ...]:
    """Enumerate the declared cache independently of any mutation plan.

    The universe is intentionally explicit: every live channel has capacity,
    outgoing, incoming, and BatchNorm-assumption records; every live unordered
    pair has one geometry record; and every declared residual block has one
    score.  An invalidation plan is auditable only if its keys belong to this
    independently constructed set.
    """

    if not isinstance(graph, GraphModel):
        raise TypeError("graph must be a GraphModel")
    keys: set[CacheKey] = set()
    for layer in graph.layers:
        for channel in layer.channels:
            keys.update(
                {
                    _channel_key("channel_capacity", channel),
                    _channel_key("channel_outgoing", channel),
                    _channel_key("channel_incoming", channel),
                    _channel_key("bn_assumption", channel),
                }
            )
        keys.update(_pair_keys("pair_geometry", layer))
    keys.update(
        CacheKey(
            kind="block_score",
            owner_id=block.block_id,
            detail=tuple(block.spanned_layers),
        )
        for block in graph.blocks
    )
    return tuple(sorted(keys))


@dataclass(frozen=True, slots=True)
class DependencyPlan:
    """Complete fail-closed cache invalidation closure for one physical action."""

    action: str
    mutated_channels: tuple[ChannelRef, ...]
    replacement_channel: ChannelRef | None
    current_layer_capacities: tuple[CacheKey, ...]
    current_layer_outgoing: tuple[CacheKey, ...]
    current_layer_incoming: tuple[CacheKey, ...]
    current_layer_bn_assumptions: tuple[CacheKey, ...]
    current_layer_pairs: tuple[CacheKey, ...]
    predecessor_outgoing: tuple[CacheKey, ...]
    predecessor_capacities: tuple[CacheKey, ...]
    predecessor_pairs: tuple[CacheKey, ...]
    successor_incoming: tuple[CacheKey, ...]
    successor_capacities: tuple[CacheKey, ...]
    successor_pairs: tuple[CacheKey, ...]
    successor_bn_assumptions: tuple[CacheKey, ...]
    spanning_block_scores: tuple[CacheKey, ...]

    def __post_init__(self) -> None:
        if self.action not in SUPPORTED_ACTIONS:
            raise GraphValidationError(f"unknown action {self.action!r}")
        if self.action == "prune" and self.replacement_channel is not None:
            raise GraphValidationError("a prune plan cannot name a replacement channel")
        if self.action == "merge" and self.replacement_channel not in self.mutated_channels:
            raise GraphValidationError(
                "a merge plan requires one mutated channel as the replacement vessel"
            )
        for field_name in (
            "mutated_channels",
            "current_layer_capacities",
            "current_layer_outgoing",
            "current_layer_incoming",
            "current_layer_bn_assumptions",
            "current_layer_pairs",
            "predecessor_outgoing",
            "predecessor_capacities",
            "predecessor_pairs",
            "successor_incoming",
            "successor_capacities",
            "successor_pairs",
            "successor_bn_assumptions",
            "spanning_block_scores",
        ):
            values = tuple(getattr(self, field_name))
            if tuple(sorted(values)) != values or len(set(values)) != len(values):
                raise GraphValidationError(f"{field_name} must be unique and sorted")
            object.__setattr__(self, field_name, values)

    @property
    def cache_keys(self) -> tuple[CacheKey, ...]:
        """Return the deterministic union across every dependency category."""

        keys = {
            key
            for group in (
                self.current_layer_pairs,
                self.current_layer_capacities,
                self.current_layer_outgoing,
                self.current_layer_incoming,
                self.current_layer_bn_assumptions,
                self.predecessor_outgoing,
                self.predecessor_capacities,
                self.predecessor_pairs,
                self.successor_incoming,
                self.successor_capacities,
                self.successor_pairs,
                self.successor_bn_assumptions,
                self.spanning_block_scores,
            )
            for key in group
        }
        return tuple(sorted(keys))

    @property
    def current_layer_keys(self) -> tuple[CacheKey, ...]:
        """Return every invalidated record owned by the physically changed layer."""

        return tuple(
            sorted(
                {
                    key
                    for group in (
                        self.current_layer_capacities,
                        self.current_layer_outgoing,
                        self.current_layer_incoming,
                        self.current_layer_bn_assumptions,
                        self.current_layer_pairs,
                    )
                    for key in group
                }
            )
        )

    def records(self) -> tuple[dict[str, object], ...]:
        """Return canonical records in cache-key order."""

        return tuple(key.as_record() for key in self.cache_keys)

    def category_counts(self) -> tuple[tuple[str, int], ...]:
        return (
            ("current_layer_capacities", len(self.current_layer_capacities)),
            ("current_layer_outgoing", len(self.current_layer_outgoing)),
            ("current_layer_incoming", len(self.current_layer_incoming)),
            ("current_layer_bn_assumptions", len(self.current_layer_bn_assumptions)),
            ("current_layer_pairs", len(self.current_layer_pairs)),
            ("predecessor_outgoing", len(self.predecessor_outgoing)),
            ("predecessor_capacities", len(self.predecessor_capacities)),
            ("predecessor_pairs", len(self.predecessor_pairs)),
            ("successor_incoming", len(self.successor_incoming)),
            ("successor_capacities", len(self.successor_capacities)),
            ("successor_pairs", len(self.successor_pairs)),
            ("successor_bn_assumptions", len(self.successor_bn_assumptions)),
            ("spanning_block_scores", len(self.spanning_block_scores)),
        )


def dependency_invalidation(
    graph: GraphModel,
    mutated: ChannelRef | Iterable[ChannelRef],
    *,
    action: str | None = None,
    replacement_channel: ChannelRef | None = None,
) -> DependencyPlan:
    """Build the complete dependency closure after a prune or merge.

    A one-channel mutation defaults to ``prune`` and a two-channel mutation to
    ``merge``.  Passing ``action`` makes that declaration explicit.  A merge
    names its two live children and must identify which stable channel token
    will hold the synthesized parent.  A caller may compute the plan immediately
    before physical surgery and then discard every returned key.
    """

    if not isinstance(graph, GraphModel):
        raise TypeError("graph must be a GraphModel")
    if isinstance(mutated, ChannelRef):
        channels = (mutated,)
    else:
        channels = tuple(mutated)
    channels = tuple(sorted(channels))
    if not channels or len(set(channels)) != len(channels):
        raise GraphValidationError("mutated channels must be a non-empty unique set")
    if any(not graph.has_channel(channel) for channel in channels):
        raise GraphValidationError("mutation names a channel outside the live graph")
    layer_ids = {channel.layer_id for channel in channels}
    if len(layer_ids) != 1:
        raise GraphValidationError("one action cannot mutate channels from different layers")
    inferred_action = "prune" if len(channels) == 1 else "merge" if len(channels) == 2 else ""
    selected_action = inferred_action if action is None else action
    if selected_action not in SUPPORTED_ACTIONS:
        raise GraphValidationError("action must be 'prune' or 'merge'")
    if selected_action == "prune" and len(channels) != 1:
        raise GraphValidationError("a prune plan requires exactly one channel")
    if selected_action == "merge" and len(channels) != 2:
        raise GraphValidationError("a merge plan requires exactly two channels")
    if selected_action == "prune" and replacement_channel is not None:
        raise GraphValidationError("a prune action cannot name a replacement channel")
    if selected_action == "merge" and replacement_channel not in channels:
        raise GraphValidationError(
            "a merge action requires one mutated channel as replacement_channel"
        )

    current = graph.layer(channels[0].layer_id)
    predecessor_ids = sorted(
        {source.layer_id for source in current.input_sources if source is not None}
    )
    successor_ids = sorted(
        {
            consumer.layer_id
            for channel in channels
            for consumer in graph.consumer_map[channel]
        }
    )
    predecessors = tuple(graph.layer(layer_id) for layer_id in predecessor_ids)
    successors = tuple(graph.layer(layer_id) for layer_id in successor_ids)

    predecessor_channels = tuple(
        sorted(
            {
                source
                for source in current.input_sources
                if source is not None
            }
        )
    )
    predecessor_outgoing = tuple(
        sorted(_channel_key("channel_outgoing", channel) for channel in predecessor_channels)
    )
    predecessor_capacities = tuple(
        sorted(_channel_key("channel_capacity", channel) for channel in predecessor_channels)
    )
    predecessor_pairs = tuple(
        sorted(key for layer in predecessors for key in _pair_keys("pair_geometry", layer))
    )

    successor_incoming = tuple(
        sorted(
            _channel_key("channel_incoming", channel)
            for layer in successors
            for channel in layer.channels
        )
    )
    successor_capacities = tuple(
        sorted(
            _channel_key("channel_capacity", channel)
            for layer in successors
            for channel in layer.channels
        )
    )
    successor_pairs = tuple(
        sorted(key for layer in successors for key in _pair_keys("pair_geometry", layer))
    )
    successor_bn = tuple(
        sorted(
            _channel_key("bn_assumption", channel)
            for layer in successors
            for channel in layer.channels
        )
    )

    affected_layers = {current.layer_id, *predecessor_ids, *successor_ids}
    block_keys = tuple(
        sorted(
            CacheKey(
                kind="block_score",
                owner_id=block.block_id,
                detail=tuple(block.spanned_layers),
            )
            for block in graph.blocks
            if affected_layers.intersection(block.spanned_layers)
        )
    )
    return DependencyPlan(
        action=selected_action,
        mutated_channels=channels,
        replacement_channel=replacement_channel,
        current_layer_capacities=tuple(
            sorted(_channel_key("channel_capacity", channel) for channel in channels)
        ),
        current_layer_outgoing=tuple(
            sorted(_channel_key("channel_outgoing", channel) for channel in channels)
        ),
        current_layer_incoming=tuple(
            sorted(_channel_key("channel_incoming", channel) for channel in channels)
        ),
        current_layer_bn_assumptions=tuple(
            sorted(_channel_key("bn_assumption", channel) for channel in channels)
        ),
        current_layer_pairs=tuple(
            sorted(
                key
                for key in _pair_keys("pair_geometry", current)
                if any(channel in key.channels for channel in channels)
            )
        ),
        predecessor_outgoing=predecessor_outgoing,
        predecessor_capacities=predecessor_capacities,
        predecessor_pairs=predecessor_pairs,
        successor_incoming=successor_incoming,
        successor_capacities=successor_capacities,
        successor_pairs=successor_pairs,
        successor_bn_assumptions=successor_bn,
        spanning_block_scores=block_keys,
    )


def graph_after_channel_surgery(graph: GraphModel, plan: DependencyPlan) -> GraphModel:
    """Return the declared post-surgery topology without mutating ``graph``.

    A merge retains the explicitly selected replacement channel as the stable
    physical vessel and removes the other child.  This helper changes topology
    only; callers remain responsible for writing the fitted parent parameters.
    """

    if not isinstance(graph, GraphModel):
        raise TypeError("graph must be a GraphModel")
    if not isinstance(plan, DependencyPlan):
        raise TypeError("plan must be a DependencyPlan")
    if any(not graph.has_channel(channel) for channel in plan.mutated_channels):
        raise GraphValidationError("plan names a channel outside the supplied graph")

    removed = set(plan.mutated_channels)
    if plan.action == "merge":
        removed.remove(plan.replacement_channel)

    updated_layers: list[LayerSpec] = []
    for layer in graph.layers:
        live_channels = tuple(channel for channel in layer.channels if channel not in removed)
        live_inputs = tuple(source for source in layer.input_sources if source not in removed)
        if not live_channels:
            raise UnsupportedGraphError(
                f"surgery would remove every channel from layer {layer.layer_id!r}"
            )
        if not live_inputs:
            raise UnsupportedGraphError(
                f"surgery would remove every input from layer {layer.layer_id!r}"
            )
        updated_layers.append(
            replace(layer, channels=live_channels, input_sources=live_inputs)
        )

    consumers: dict[ChannelRef, list[ConsumerRef]] = {
        channel: [] for layer in updated_layers for channel in layer.channels
    }
    for layer in updated_layers:
        for input_index, source in enumerate(layer.input_sources):
            if source is not None:
                consumers[source].append(ConsumerRef(layer.layer_id, input_index))
    canonical_consumers = {
        channel: tuple(sorted(channel_consumers))
        for channel, channel_consumers in consumers.items()
    }
    return GraphModel(
        layers=tuple(updated_layers),
        consumer_map=canonical_consumers,
        blocks=graph.blocks,
    )


def _parameter_payoff(graph: GraphModel, channel: ChannelRef, *, initial: bool) -> int:
    if not graph.has_channel(channel):
        raise GraphValidationError("parameter payoff requires a live graph channel")
    layer = graph.layer(channel.layer_id)
    input_count = layer.initial_input_count if initial else layer.live_input_count
    payoff = input_count * layer.kernel_volume
    payoff += int(layer.use_bias) + layer.bn_values_per_channel

    for consumer in graph.consumer_map[channel]:
        successor = graph.layer(consumer.layer_id)
        output_count = (
            successor.initial_output_count if initial else successor.live_output_count
        )
        payoff += output_count * successor.kernel_volume
    return int(payoff)


def live_parameter_payoff(graph: GraphModel, channel: ChannelRef) -> int:
    """Return parameters physically released by removing one live channel.

    The count includes the channel's generating filter, optional bias, declared
    BN values, and one incoming weight coordinate for every output channel of
    every explicit consumer.  Fan-out is therefore counted branch by branch.
    """

    return _parameter_payoff(graph, channel, initial=False)


def initial_static_parameter_payoff(graph: GraphModel, channel: ChannelRef) -> int:
    """Return the separate initial-architecture payoff baseline."""

    return _parameter_payoff(graph, channel, initial=True)


def calibrated_branch_cost(
    residual_output: np.ndarray | Sequence[float],
    shortcut_input: np.ndarray | Sequence[float],
    *,
    spec: BlockSpec | None = None,
    sample_weights: np.ndarray | Sequence[float] | None = None,
    activation: str = "post_relu",
) -> float:
    """Measure direct calibrated boundary displacement from branch eviction.

    For the default post-activation ResNet V1 boundary, the retained block is
    ``relu(shortcut + residual)`` and the evicted block is ``relu(shortcut)``.
    The result is the weighted mean, over calibration examples, of the mean
    squared displacement across all remaining coordinates.  ``identity`` is
    available for a direct preactivation diagnostic.  Neither mode sums local
    internal-layer scores.
    """

    if spec is not None:
        eligibility = check_identity_block(spec)
        if not eligibility.eligible:
            raise UnsupportedGraphError(
                "direct block cost requires an eligible identity block: "
                + ",".join(eligibility.reasons)
            )
    residual = np.asarray(residual_output, dtype=np.float64)
    shortcut = np.asarray(shortcut_input, dtype=np.float64)
    if residual.shape != shortcut.shape:
        raise ValueError("residual_output and shortcut_input must have the same shape")
    if residual.ndim == 0 or residual.shape[0] == 0 or residual.size == 0:
        raise ValueError("calibration arrays must contain at least one example")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(shortcut)):
        raise ValueError("calibration arrays must be finite")

    if activation == "post_relu":
        retained = np.maximum(shortcut + residual, 0.0)
        evicted = np.maximum(shortcut, 0.0)
    elif activation == "identity":
        retained = shortcut + residual
        evicted = shortcut
    else:
        raise ValueError("activation must be 'post_relu' or 'identity'")

    squared = np.square(retained - evicted)
    per_example = squared.reshape(squared.shape[0], -1).mean(axis=1)
    if sample_weights is None:
        result = float(per_example.mean())
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != (residual.shape[0],):
            raise ValueError("sample_weights must have one value per example")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weights must be finite and non-negative")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("sample_weights must have positive total weight")
        result = float(np.dot(weights, per_example) / total)
    if not math.isfinite(result):
        raise ValueError("calibrated branch cost is not finite")
    return result


def _choose2(width: int) -> int:
    return width * (width - 1) // 2


@dataclass(frozen=True, slots=True)
class ResourceReport:
    """Exact declared workload and byte counts, with no timing interpretation."""

    live_widths: tuple[tuple[str, int], ...]
    current_prune_candidates: int
    current_merge_candidates: int
    current_block_candidates: int
    current_candidate_count: int
    current_scan_evaluations: int
    trajectory_states: tuple[tuple[tuple[str, int], ...], ...]
    trajectory_scan_count: int
    trajectory_prune_evaluations: int
    trajectory_merge_evaluations: int
    trajectory_block_evaluations: int
    trajectory_candidate_evaluations: int
    pair_cache_entries: int
    cache_scalars_per_pair: int
    cache_scalar_payload_bytes: int
    cache_indices_per_pair: int
    cache_index_payload_bytes: int
    cache_key_record_bytes: int
    cache_total_accounted_bytes: int
    cache_container_overhead_included: bool
    refresh_plan_count: int
    refresh_key_evaluations: int
    refresh_unique_key_count: int
    refresh_counts_by_kind: tuple[tuple[str, int], ...]
    calibration_examples: int
    calibration_output_elements: int
    calibration_input_values: int
    calibration_branch_cost_evaluations: int
    wall_clock_claim: bool = False

    def __post_init__(self) -> None:
        if self.wall_clock_claim:
            raise ValueError("ResourceReport cannot make a wall-clock claim")
        numeric_fields = (
            self.current_prune_candidates,
            self.current_merge_candidates,
            self.current_block_candidates,
            self.current_candidate_count,
            self.current_scan_evaluations,
            self.trajectory_scan_count,
            self.trajectory_prune_evaluations,
            self.trajectory_merge_evaluations,
            self.trajectory_block_evaluations,
            self.trajectory_candidate_evaluations,
            self.pair_cache_entries,
            self.cache_scalars_per_pair,
            self.cache_scalar_payload_bytes,
            self.cache_indices_per_pair,
            self.cache_index_payload_bytes,
            self.cache_key_record_bytes,
            self.cache_total_accounted_bytes,
            self.refresh_plan_count,
            self.refresh_key_evaluations,
            self.refresh_unique_key_count,
            self.calibration_examples,
            self.calibration_output_elements,
            self.calibration_input_values,
            self.calibration_branch_cost_evaluations,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in numeric_fields):
            raise ValueError("resource counts must be non-negative integers")

    def as_record(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible accounting record."""

        return {
            "live_widths": [list(item) for item in self.live_widths],
            "candidates": {
                "prune": self.current_prune_candidates,
                "merge": self.current_merge_candidates,
                "block": self.current_block_candidates,
                "total": self.current_candidate_count,
            },
            "scan": {"candidate_evaluations": self.current_scan_evaluations},
            "trajectory": {
                "states": [[list(item) for item in state] for state in self.trajectory_states],
                "scan_count": self.trajectory_scan_count,
                "prune_evaluations": self.trajectory_prune_evaluations,
                "merge_evaluations": self.trajectory_merge_evaluations,
                "block_evaluations": self.trajectory_block_evaluations,
                "candidate_evaluations": self.trajectory_candidate_evaluations,
            },
            "cache": {
                "pair_entries": self.pair_cache_entries,
                "scalars_per_pair": self.cache_scalars_per_pair,
                "scalar_payload_bytes": self.cache_scalar_payload_bytes,
                "indices_per_pair": self.cache_indices_per_pair,
                "index_payload_bytes": self.cache_index_payload_bytes,
                "key_record_bytes": self.cache_key_record_bytes,
                "total_accounted_bytes": self.cache_total_accounted_bytes,
                "container_overhead_included": self.cache_container_overhead_included,
            },
            "refresh": {
                "plan_count": self.refresh_plan_count,
                "key_evaluations": self.refresh_key_evaluations,
                "unique_key_count": self.refresh_unique_key_count,
                "counts_by_kind": [list(item) for item in self.refresh_counts_by_kind],
            },
            "calibration": {
                "examples": self.calibration_examples,
                "output_elements": self.calibration_output_elements,
                "input_values": self.calibration_input_values,
                "branch_cost_evaluations": self.calibration_branch_cost_evaluations,
            },
            "wall_clock_claim": False,
        }


def _normalize_trajectory_states(
    graph: GraphModel,
    trajectory_states: Sequence[Mapping[str, int]] | None,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    live = dict(graph.live_widths)
    raw_states: Sequence[Mapping[str, int]] = (
        (live,) if trajectory_states is None else trajectory_states
    )
    if not raw_states:
        raise ValueError("trajectory_states must contain at least one scan state")
    normalized: list[tuple[tuple[str, int], ...]] = []
    expected_ids = set(live)
    initial_widths = {layer.layer_id: layer.initial_output_count for layer in graph.layers}
    for state in raw_states:
        if set(state) != expected_ids:
            raise ValueError("each trajectory state must name every graph layer exactly")
        items: list[tuple[str, int]] = []
        for layer_id in sorted(expected_ids):
            width = state[layer_id]
            if isinstance(width, bool) or not isinstance(width, int):
                raise ValueError("trajectory widths must be integers")
            if width < 0 or width > initial_widths[layer_id]:
                raise ValueError("trajectory width is outside the declared initial width")
            items.append((layer_id, width))
        normalized.append(tuple(items))
    return tuple(normalized)


def _normalize_calibration_shapes(
    shapes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    normalized: list[tuple[int, ...]] = []
    for shape in shapes:
        values = tuple(shape)
        if not values:
            raise ValueError("calibration shapes must include an example axis")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("calibration shapes must contain non-negative integers")
        normalized.append(values)
    return tuple(normalized)


def account_resources(
    graph: GraphModel,
    *,
    trajectory_states: Sequence[Mapping[str, int]] | None = None,
    dependency_plans: Sequence[DependencyPlan] = (),
    cache_scalars_per_pair: int = 3,
    scalar_dtype: np.dtype[object] | type | str = np.float64,
    cache_indices_per_pair: int = 2,
    index_dtype: np.dtype[object] | type | str = np.uint32,
    calibration_shapes: Sequence[Sequence[int]] = (),
) -> ResourceReport:
    """Account exact candidate, scan, trajectory, cache, refresh, and calibration work.

    ``trajectory_states`` is the complete declared sequence of states that are
    scanned.  No unobserved action schedule is inferred.  Cache bytes describe
    fixed NumPy scalar and index payloads plus newline-delimited canonical
    ``CacheKey`` JSON records.  Python/container overhead is explicitly absent.
    Calibration counts cover pairs of residual and shortcut arrays of the
    declared shapes.
    """

    if not isinstance(graph, GraphModel):
        raise TypeError("graph must be a GraphModel")
    for name, value in (
        ("cache_scalars_per_pair", cache_scalars_per_pair),
        ("cache_indices_per_pair", cache_indices_per_pair),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    scalar = np.dtype(scalar_dtype)
    index = np.dtype(index_dtype)
    if scalar.hasobject or scalar.itemsize <= 0:
        raise ValueError("scalar_dtype must have a fixed non-object item size")
    if index.hasobject or index.itemsize <= 0 or index.kind not in "iu":
        raise ValueError("index_dtype must be a fixed-width integer dtype")

    widths = dict(graph.live_widths)
    current_prunes = sum(widths.values())
    current_merges = sum(_choose2(width) for width in widths.values())
    eligible_blocks = sum(check_identity_block(block).eligible for block in graph.blocks)
    current_total = current_prunes + current_merges + eligible_blocks

    states = _normalize_trajectory_states(graph, trajectory_states)
    trajectory_prunes = 0
    trajectory_merges = 0
    trajectory_blocks = 0
    for state in states:
        state_widths = dict(state)
        trajectory_prunes += sum(state_widths.values())
        trajectory_merges += sum(_choose2(width) for width in state_widths.values())
        trajectory_blocks += eligible_blocks
    trajectory_total = trajectory_prunes + trajectory_merges + trajectory_blocks

    pair_keys = graph.pair_cache_keys()
    pair_entries = len(pair_keys)
    scalar_bytes = pair_entries * cache_scalars_per_pair * scalar.itemsize
    index_bytes = pair_entries * cache_indices_per_pair * index.itemsize
    record_bytes = sum(len(key.canonical_bytes()) + 1 for key in pair_keys)

    plans = tuple(dependency_plans)
    if any(not isinstance(plan, DependencyPlan) for plan in plans):
        raise TypeError("dependency_plans must contain DependencyPlan")
    refresh_keys = [key for plan in plans for key in plan.cache_keys]
    unique_refresh = set(refresh_keys)
    refresh_by_kind: dict[str, int] = {}
    for key in refresh_keys:
        refresh_by_kind[key.kind] = refresh_by_kind.get(key.kind, 0) + 1

    shapes = _normalize_calibration_shapes(calibration_shapes)
    examples = sum(shape[0] for shape in shapes)
    elements = sum(math.prod(shape) for shape in shapes)

    return ResourceReport(
        live_widths=graph.live_widths,
        current_prune_candidates=current_prunes,
        current_merge_candidates=current_merges,
        current_block_candidates=eligible_blocks,
        current_candidate_count=current_total,
        current_scan_evaluations=current_total,
        trajectory_states=states,
        trajectory_scan_count=len(states),
        trajectory_prune_evaluations=trajectory_prunes,
        trajectory_merge_evaluations=trajectory_merges,
        trajectory_block_evaluations=trajectory_blocks,
        trajectory_candidate_evaluations=trajectory_total,
        pair_cache_entries=pair_entries,
        cache_scalars_per_pair=cache_scalars_per_pair,
        cache_scalar_payload_bytes=scalar_bytes,
        cache_indices_per_pair=cache_indices_per_pair,
        cache_index_payload_bytes=index_bytes,
        cache_key_record_bytes=record_bytes,
        cache_total_accounted_bytes=scalar_bytes + index_bytes + record_bytes,
        cache_container_overhead_included=False,
        refresh_plan_count=len(plans),
        refresh_key_evaluations=len(refresh_keys),
        refresh_unique_key_count=len(unique_refresh),
        refresh_counts_by_kind=tuple(sorted(refresh_by_kind.items())),
        calibration_examples=examples,
        calibration_output_elements=elements,
        calibration_input_values=2 * elements,
        calibration_branch_cost_evaluations=elements,
        wall_clock_claim=False,
    )


__all__ = [
    "BlockEligibility",
    "BlockSpec",
    "CacheKey",
    "ChannelRef",
    "ConsumerRef",
    "DependencyPlan",
    "GraphModel",
    "GraphValidationError",
    "LayerSpec",
    "ResourceReport",
    "UnsupportedGraphError",
    "account_resources",
    "cache_universe",
    "calibrated_branch_cost",
    "check_identity_block",
    "dependency_invalidation",
    "graph_after_channel_surgery",
    "initial_static_parameter_payoff",
    "live_parameter_payoff",
]
