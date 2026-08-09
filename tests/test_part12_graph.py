"""Focused contracts for the framework-independent HOPE-R graph repair."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hope_repro.proposed_repairs.hope_r.graph import (  # noqa: E402
    BlockSpec,
    CacheKey,
    ChannelRef,
    ConsumerRef,
    GraphModel,
    GraphValidationError,
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


def channels(layer_id: str, width: int) -> tuple[ChannelRef, ...]:
    return tuple(ChannelRef(layer_id, index) for index in range(width))


def branched_graph() -> GraphModel:
    """Return stem -> mid -> {left, right} with complete direct fan-out."""

    stem_channels = channels("stem", 2)
    mid_channels = channels("mid", 2)
    left_channels = channels("left", 2)
    right_channels = channels("right", 2)
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


class IdentifierAndTopologyTests(unittest.TestCase):
    def test_channel_and_layer_identity_is_immutable(self) -> None:
        channel = ChannelRef("mid", 7)
        layer = LayerSpec("mid", (channel,), (None,))
        with self.assertRaises(FrozenInstanceError):
            channel.index = 8  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            layer.layer_id = "changed"  # type: ignore[misc]
        self.assertEqual(channel.as_record(), {"layer_id": "mid", "index": 7})

    def test_explicit_branched_consumer_map_is_complete_and_read_only(self) -> None:
        graph = branched_graph()
        mid = ChannelRef("mid", 0)
        self.assertEqual(
            graph.consumer_map[mid],
            (ConsumerRef("left", 0), ConsumerRef("right", 0)),
        )
        with self.assertRaises(TypeError):
            graph.consumer_map[mid] = ()  # type: ignore[index]

    def test_missing_branched_consumer_fails_closed(self) -> None:
        graph = branched_graph()
        consumers = dict(graph.consumer_map)
        consumers[ChannelRef("mid", 0)] = (ConsumerRef("left", 0),)
        with self.assertRaisesRegex(UnsupportedGraphError, "omits the declared edge"):
            GraphModel(graph.layers, consumers, graph.blocks)

    def test_incomplete_or_unresolved_consumer_declaration_fails_closed(self) -> None:
        graph = branched_graph()
        with self.assertRaisesRegex(UnsupportedGraphError, "not declared complete"):
            GraphModel(
                graph.layers,
                graph.consumer_map,
                graph.blocks,
                consumer_map_complete=False,
            )
        altered = tuple(
            replace(layer, consumers_resolved=False) if layer.layer_id == "mid" else layer
            for layer in graph.layers
        )
        with self.assertRaisesRegex(UnsupportedGraphError, "fan-out is unresolved"):
            GraphModel(altered, graph.consumer_map, graph.blocks)

    def test_grouped_depthwise_shared_and_unknown_operators_fail_closed(self) -> None:
        graph = branched_graph()
        mutations = (
            ({"groups": 2}, "grouped operators"),
            ({"depthwise": True}, "depthwise operators"),
            ({"shared_parameters": True}, "shared parameters"),
            ({"supported": False}, "unsupported operator"),
            ({"operator": "attention"}, "unsupported operator"),
        )
        for changes, message in mutations:
            with self.subTest(changes=changes):
                altered = tuple(
                    replace(layer, **changes) if layer.layer_id == "mid" else layer
                    for layer in graph.layers
                )
                with self.assertRaisesRegex(UnsupportedGraphError, message):
                    GraphModel(altered, graph.consumer_map, graph.blocks)


class DependencyTests(unittest.TestCase):
    def test_prune_closure_covers_both_successor_branches_and_spanning_block(self) -> None:
        graph = branched_graph()
        plan = dependency_invalidation(graph, ChannelRef("mid", 0))
        self.assertEqual(plan.action, "prune")
        self.assertIsNone(plan.replacement_channel)
        self.assertEqual(len(plan.current_layer_capacities), 1)
        self.assertEqual(len(plan.current_layer_outgoing), 1)
        self.assertEqual(len(plan.current_layer_incoming), 1)
        self.assertEqual(len(plan.current_layer_bn_assumptions), 1)
        self.assertEqual(len(plan.current_layer_pairs), 1)
        self.assertEqual(len(plan.predecessor_outgoing), 2)
        self.assertEqual(len(plan.predecessor_capacities), 2)
        self.assertEqual(len(plan.predecessor_pairs), 1)
        self.assertEqual(len(plan.successor_incoming), 4)
        self.assertEqual(len(plan.successor_capacities), 4)
        self.assertEqual(len(plan.successor_pairs), 2)
        self.assertEqual(len(plan.successor_bn_assumptions), 4)
        self.assertEqual(len(plan.spanning_block_scores), 1)
        self.assertEqual(len(plan.cache_keys), 25)
        self.assertEqual(set(plan.cache_keys) - set(cache_universe(graph)), set())
        removed_capacity = CacheKey(
            "channel_capacity", "mid", (ChannelRef("mid", 0),)
        )
        self.assertIn(removed_capacity, plan.current_layer_capacities)
        successor_owners = {
            key.owner_id for key in plan.successor_incoming
        }
        self.assertEqual(successor_owners, {"left", "right"})
        self.assertEqual(plan.spanning_block_scores[0].owner_id, "identity_0")

    def test_merge_uses_same_complete_dependency_closure(self) -> None:
        graph = branched_graph()
        plan = dependency_invalidation(
            graph,
            (ChannelRef("mid", 1), ChannelRef("mid", 0)),
            action="merge",
            replacement_channel=ChannelRef("mid", 0),
        )
        self.assertEqual(plan.mutated_channels, channels("mid", 2))
        self.assertEqual(plan.replacement_channel, ChannelRef("mid", 0))
        self.assertEqual(len(plan.current_layer_capacities), 2)
        self.assertEqual(len(plan.cache_keys), 29)
        after = graph_after_channel_surgery(graph, plan)
        self.assertEqual(after.layer("mid").channels, (ChannelRef("mid", 0),))
        self.assertEqual(after.layer("left").input_sources, (ChannelRef("mid", 0),))
        self.assertEqual(len(cache_universe(graph)), 37)
        self.assertEqual(len(cache_universe(after)), 32)
        with self.assertRaisesRegex(GraphValidationError, "replacement_channel"):
            dependency_invalidation(
                graph,
                (ChannelRef("mid", 1), ChannelRef("mid", 0)),
                action="merge",
            )
        with self.assertRaisesRegex(GraphValidationError, "exactly two"):
            dependency_invalidation(graph, ChannelRef("mid", 0), action="merge")

    def test_unknown_or_cross_layer_mutation_fails_closed(self) -> None:
        graph = branched_graph()
        with self.assertRaisesRegex(GraphValidationError, "outside the live graph"):
            dependency_invalidation(graph, ChannelRef("mid", 99))
        with self.assertRaisesRegex(GraphValidationError, "different layers"):
            dependency_invalidation(
                graph,
                (ChannelRef("mid", 0), ChannelRef("stem", 0)),
                action="merge",
            )

    def test_cache_records_are_canonical_and_deterministic(self) -> None:
        first = ChannelRef("mid", 0)
        second = ChannelRef("mid", 1)
        key = CacheKey.pair("pair_geometry", "mid", second, first)
        self.assertEqual(key.channels, (first, second))
        decoded = json.loads(key.canonical_bytes().decode("utf-8"))
        self.assertEqual(decoded, key.as_record())
        self.assertEqual(key.sha256, key.sha256)
        plan = dependency_invalidation(branched_graph(), first)
        self.assertEqual(plan.records(), tuple(key.as_record() for key in plan.cache_keys))


class PayoffAndResourceTests(unittest.TestCase):
    def test_live_physical_payoff_is_not_the_initial_static_payoff(self) -> None:
        graph = branched_graph()
        channel = ChannelRef("mid", 0)
        self.assertEqual(live_parameter_payoff(graph, channel), 10)
        self.assertEqual(initial_static_parameter_payoff(graph, channel), 22)
        self.assertNotEqual(
            live_parameter_payoff(graph, channel),
            initial_static_parameter_payoff(graph, channel),
        )

    def test_resource_report_accounts_every_declared_scope_exactly(self) -> None:
        graph = branched_graph()
        plan = dependency_invalidation(graph, ChannelRef("mid", 0))
        current = dict(graph.live_widths)
        after = dict(current)
        after["mid"] = 1
        report = account_resources(
            graph,
            trajectory_states=(current, after),
            dependency_plans=(plan,),
            cache_scalars_per_pair=3,
            scalar_dtype=np.float64,
            cache_indices_per_pair=2,
            index_dtype=np.uint32,
            calibration_shapes=((3, 2),),
        )
        self.assertEqual(report.current_prune_candidates, 8)
        self.assertEqual(report.current_merge_candidates, 4)
        self.assertEqual(report.current_block_candidates, 1)
        self.assertEqual(report.current_candidate_count, 13)
        self.assertEqual(report.current_scan_evaluations, 13)
        self.assertEqual(report.trajectory_scan_count, 2)
        self.assertEqual(report.trajectory_prune_evaluations, 15)
        self.assertEqual(report.trajectory_merge_evaluations, 7)
        self.assertEqual(report.trajectory_block_evaluations, 2)
        self.assertEqual(report.trajectory_candidate_evaluations, 24)
        self.assertEqual(report.pair_cache_entries, 4)
        self.assertEqual(report.cache_scalar_payload_bytes, 96)
        self.assertEqual(report.cache_index_payload_bytes, 32)
        self.assertGreater(report.cache_key_record_bytes, 0)
        self.assertEqual(
            report.cache_total_accounted_bytes,
            96 + 32 + report.cache_key_record_bytes,
        )
        self.assertFalse(report.cache_container_overhead_included)
        self.assertEqual(report.refresh_plan_count, 1)
        self.assertEqual(report.refresh_key_evaluations, 25)
        self.assertEqual(report.refresh_unique_key_count, 25)
        self.assertEqual(report.calibration_examples, 3)
        self.assertEqual(report.calibration_output_elements, 6)
        self.assertEqual(report.calibration_input_values, 12)
        self.assertEqual(report.calibration_branch_cost_evaluations, 6)
        self.assertFalse(report.wall_clock_claim)
        self.assertFalse(report.as_record()["wall_clock_claim"])

    def test_resource_accounting_requires_explicit_complete_states(self) -> None:
        graph = branched_graph()
        with self.assertRaisesRegex(ValueError, "every graph layer"):
            account_resources(graph, trajectory_states=({"mid": 2},))
        with self.assertRaisesRegex(ValueError, "fixed-width integer"):
            account_resources(graph, index_dtype=np.float64)


class BlockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = BlockSpec(
            block_id="block",
            spanned_layers=("mid",),
            input_shape=(4, 8, 8),
            output_shape=(4, 8, 8),
        )

    def test_only_identity_stride1_same_shape_post_activation_v1_is_eligible(self) -> None:
        result = check_identity_block(self.valid)
        self.assertTrue(result)
        self.assertEqual(result.reasons, ())

    def test_projection_stride_v2_and_no_skip_cases_are_rejected(self) -> None:
        cases = (
            (replace(self.valid, shortcut_kind="projection"), "non_identity_shortcut"),
            (replace(self.valid, stride=2), "non_unit_stride"),
            (replace(self.valid, architecture="resnet_v2"), "architecture_not_resnet_v1"),
            (replace(self.valid, activation_order="pre_activation"), "not_post_activation"),
            (replace(self.valid, shortcut_kind="none"), "missing_skip"),
            (replace(self.valid, output_shape=(8, 4, 4)), "shape_mismatch"),
        )
        for spec, reason in cases:
            with self.subTest(reason=reason):
                result = check_identity_block(spec)
                self.assertFalse(result)
                self.assertIn(reason, result.reasons)

    def test_calibrated_cost_prices_direct_boundary_displacement(self) -> None:
        shortcut = np.asarray([[-1.0, 1.0], [2.0, -2.0]])
        residual = np.asarray([[2.0, -0.5], [-3.0, 4.0]])
        self.assertEqual(
            calibrated_branch_cost(residual, shortcut, spec=self.valid),
            2.3125,
        )
        self.assertEqual(
            calibrated_branch_cost(
                residual,
                shortcut,
                spec=self.valid,
                sample_weights=np.asarray([1.0, 3.0]),
            ),
            3.15625,
        )
        self.assertEqual(
            calibrated_branch_cost(residual, shortcut, activation="identity"),
            7.3125,
        )
        with self.assertRaisesRegex(UnsupportedGraphError, "eligible identity block"):
            calibrated_branch_cost(
                residual,
                shortcut,
                spec=replace(self.valid, shortcut_kind="projection"),
            )


if __name__ == "__main__":
    unittest.main()
