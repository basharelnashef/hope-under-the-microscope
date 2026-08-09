"""Focused tests for the proposed fixed-width DEFT repair state machine."""

from __future__ import annotations

import unittest

import numpy as np

from hope_repro.proposed_repairs.hope_r.deft import (
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
    layerwise_top_k_current_capacity_partition,
    momentumw_step,
    source_owned_parameter_masks,
    structural_slack_to_core_mask,
    two_stage_gradient_liveness,
)


class Part12DeftRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layers = (
            LayerDefinition("hidden-1", ("h1:2", "h1:0", "h1:1"), 2),
            LayerDefinition("hidden-2", ("h2:0", "h2:1"), 1),
        )
        self.capacities = {
            "h1:0": 4.0,
            "h1:1": 4.0,
            "h1:2": 1.0,
            "h2:0": 7.0,
            "h2:1": 3.0,
        }

    def test_layerwise_top_k_has_explicit_counts_and_slot_id_tie_break(self) -> None:
        partitions = layerwise_top_k_current_capacity_partition(
            self.layers,
            dict(reversed(tuple(self.capacities.items()))),
        )
        first, second = partitions
        self.assertEqual(first.core_count, 2)
        self.assertEqual(first.ranked_slot_ids, ("h1:0", "h1:1", "h1:2"))
        self.assertEqual(first.core_slot_ids, ("h1:0", "h1:1"))
        self.assertEqual(first.slack_slot_ids, ("h1:2",))
        self.assertEqual(second.core_count, 1)
        self.assertEqual(second.core_slot_ids, ("h2:0",))
        self.assertIn("stable slot ID", first.tie_break_rule)

    def test_fixed_width_lifecycle_never_renumbers_a_slot(self) -> None:
        machine = FixedWidthDeftStateMachine(self.layers, self.capacities)
        original_ids = machine.stable_slot_ids
        original_widths = machine.fixed_widths

        self.assertEqual(machine.state("h1:0"), SlotState.CORE)
        with self.assertRaises(ValueError):
            machine.release("h1:0")

        self.assertEqual(machine.state("h1:2"), SlotState.SLACK)
        machine.release("h1:2")
        self.assertEqual(machine.state("h1:2"), SlotState.RELEASED)
        with self.assertRaises(ValueError):
            machine.transition("h1:2", SlotState.REINITIALIZED)
        parameters = machine.reinitialize(
            "h1:2", input_width=4, output_width=2
        )
        self.assertEqual(machine.state("h1:2"), SlotState.REINITIALIZED)
        self.assertEqual(parameters.slot_id, "h1:2")
        activation, contribution = parameters.forward(np.ones(4))
        self.assertGreater(activation, 0.0)
        np.testing.assert_array_equal(contribution, np.zeros(2))
        machine.activate_reinitialized("h1:2")
        self.assertEqual(machine.state("h1:2"), SlotState.SLACK)
        self.assertEqual(machine.stable_slot_ids, original_ids)
        self.assertEqual(machine.fixed_widths, original_widths)

        with self.assertRaises(ValueError):
            machine.activate_reinitialized("h1:2")

    def test_reinitialized_slot_has_two_stage_gradient_liveness(self) -> None:
        trace = two_stage_gradient_liveness()
        self.assertEqual(trace.initial_activation, 1.0)
        self.assertEqual(trace.initial_output, 0.0)
        self.assertEqual(trace.first_incoming_gradient, 0.0)
        self.assertEqual(trace.first_scale_gradient, 0.0)
        self.assertEqual(trace.first_shift_gradient, 0.0)
        self.assertEqual(trace.first_outgoing_gradient, -1.0)
        self.assertEqual(trace.outgoing_after_first_step, 0.25)
        self.assertEqual(trace.second_output, 0.25)
        self.assertEqual(trace.second_incoming_gradient, -0.1875)
        self.assertEqual(trace.second_scale_gradient, -0.1875)
        self.assertEqual(trace.second_shift_gradient, -0.1875)
        self.assertEqual(trace.second_outgoing_gradient, -0.75)
        self.assertTrue(trace.first_stage_outgoing_live)
        self.assertTrue(trace.second_stage_incoming_live)

    def test_structural_mask_blocks_every_plastic_to_core_route(self) -> None:
        mask = structural_slack_to_core_mask(
            (
                SlotState.CORE,
                SlotState.SLACK,
                SlotState.RELEASED,
                SlotState.REINITIALIZED,
            ),
            (SlotState.CORE, SlotState.SLACK),
        )
        np.testing.assert_array_equal(
            mask,
            np.asarray(
                [[1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            ),
        )

    def test_source_owned_masks_freeze_complete_core_outgoing_row(self) -> None:
        masks = source_owned_parameter_masks(
            ("core-slot", "slack-slot", "released-slot", "new-slot"),
            (
                SlotState.CORE,
                SlotState.SLACK,
                SlotState.RELEASED,
                SlotState.REINITIALIZED,
            ),
            input_width=3,
            output_width=4,
        )
        np.testing.assert_array_equal(
            masks.trainable_by_slot, np.asarray([0.0, 1.0, 0.0, 1.0])
        )
        np.testing.assert_array_equal(masks.incoming_weight[0], np.zeros(3))
        np.testing.assert_array_equal(masks.bn_scale, masks.trainable_by_slot)
        np.testing.assert_array_equal(masks.bn_shift, masks.trainable_by_slot)
        np.testing.assert_array_equal(masks.outgoing_weight[0], np.zeros(4))
        np.testing.assert_array_equal(masks.outgoing_weight[1], np.ones(4))
        np.testing.assert_array_equal(masks.outgoing_weight[2], np.zeros(4))

    def test_clear_optimizer_state_clears_masked_and_reused_entries(self) -> None:
        state = {"weight": np.asarray([[8.0, 7.0], [6.0, 5.0]])}
        update = {"weight": np.asarray([[0.0, 0.0], [1.0, 1.0]])}
        reset = {"weight": np.asarray([[0.0, 0.0], [1.0, 0.0]])}
        cleared = clear_optimizer_state(state, update, reset_masks=reset)
        np.testing.assert_array_equal(
            cleared["weight"], np.asarray([[0.0, 0.0], [0.0, 5.0]])
        )
        np.testing.assert_array_equal(
            state["weight"], np.asarray([[8.0, 7.0], [6.0, 5.0]])
        )

    def test_momentumw_has_no_core_drift_with_stale_momentum_and_decay(self) -> None:
        states = (SlotState.CORE, SlotState.SLACK)
        mask_bundle = source_owned_parameter_masks(
            ("core", "slack"), states, input_width=2, output_width=3
        )
        masks = mask_bundle.as_dict()
        parameters = {
            name: np.arange(mask.size, dtype=np.float64).reshape(mask.shape) + 1.0
            for name, mask in masks.items()
        }
        gradients = {
            name: np.full(mask.shape, 2.0, dtype=np.float64)
            for name, mask in masks.items()
        }
        stale_momentum = {
            name: np.full(mask.shape, 100.0, dtype=np.float64)
            for name, mask in masks.items()
        }
        result = momentumw_step(
            parameters,
            gradients,
            stale_momentum,
            masks,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=0.2,
        )
        self.assertTrue(result.all_masked_parameters_unchanged)
        self.assertEqual(result.max_abs_masked_drift, 0.0)
        for name, mask in masks.items():
            frozen = mask == 0.0
            trainable = mask == 1.0
            np.testing.assert_array_equal(
                result.parameters[name][frozen], parameters[name][frozen]
            )
            np.testing.assert_array_equal(
                result.optimizer_state[name][frozen], np.zeros(np.count_nonzero(frozen))
            )
            self.assertTrue(
                np.any(result.parameters[name][trainable] != parameters[name][trainable])
            )
        outgoing = result.parameters["outgoing_weight"]
        np.testing.assert_array_equal(outgoing[0], parameters["outgoing_weight"][0])
        self.assertTrue(np.any(outgoing[1] != parameters["outgoing_weight"][1]))

    def test_frozen_batchnorm_owns_trainables_and_preserves_buffers(self) -> None:
        ownership = BatchNormOwnership(
            channel_slot_ids=("slot-0", "slot-1"),
            buffer_policy=BatchNormBufferPolicy.FROZEN,
            source_task_id="source-task",
            target_task_id="target-task",
            source_track_id="source-bn",
        )
        state = BatchNormState(
            ownership=ownership,
            scale=np.asarray([1.0, 2.0]),
            shift=np.asarray([0.0, 0.5]),
            buffers={
                "source-bn": BatchNormBuffers(
                    np.asarray([0.0, 1.0]), np.asarray([1.0, 4.0])
                )
            },
        )
        updated = state.updated_for_task(
            np.asarray([[10.0, 20.0], [14.0, 28.0]]),
            task_id="target-task",
            update_rate=0.5,
        )
        self.assertEqual(
            ownership.trainable_owners,
            {
                "scale": ("slot-0", "slot-1"),
                "shift": ("slot-0", "slot-1"),
            },
        )
        self.assertEqual(
            ownership.buffer_owners["running_mean"],
            {"source-bn": "source-task"},
        )
        np.testing.assert_array_equal(
            updated.buffers["source-bn"].running_mean,
            state.buffers["source-bn"].running_mean,
        )
        np.testing.assert_array_equal(
            updated.buffers["source-bn"].running_variance,
            state.buffers["source-bn"].running_variance,
        )

    def test_dual_track_updates_only_target_and_selector_controls_evaluation(self) -> None:
        ownership = BatchNormOwnership(
            channel_slot_ids=("slot-0",),
            buffer_policy="dual_track",
            source_task_id="source-task",
            target_task_id="target-task",
            source_track_id="source-bn",
            target_track_id="target-bn",
        )
        state = BatchNormState(
            ownership=ownership,
            scale=np.asarray([1.0]),
            shift=np.asarray([0.0]),
            buffers={
                "source-bn": BatchNormBuffers(np.asarray([0.0]), np.asarray([1.0])),
                "target-bn": BatchNormBuffers(np.asarray([0.0]), np.asarray([1.0])),
            },
        )
        updated = state.updated_for_task(
            np.asarray([[2.0], [6.0]]),
            task_id="target-task",
            update_rate=0.5,
        )
        np.testing.assert_array_equal(
            updated.buffers["source-bn"].running_mean, np.asarray([0.0])
        )
        np.testing.assert_array_equal(
            updated.buffers["source-bn"].running_variance, np.asarray([1.0])
        )
        np.testing.assert_array_equal(
            updated.buffers["target-bn"].running_mean, np.asarray([2.0])
        )
        np.testing.assert_array_equal(
            updated.buffers["target-bn"].running_variance, np.asarray([2.5])
        )

        source_state = StateDeclaration(
            state_id="source-state",
            task_id="source-task",
            head_id="source-head",
            batch_norm_track_id="source-bn",
            slot_ids=("slot-0",),
            slot_mask=(1.0,),
        )
        target_state = StateDeclaration(
            state_id="target-state",
            task_id="target-task",
            head_id="target-head",
            batch_norm_track_id="target-bn",
            slot_ids=("slot-0",),
            slot_mask=(1.0,),
        )
        selector = TaskStateSelector(
            (source_state, target_state), batch_norm_ownership=ownership
        )
        source_evaluation = EvaluationDeclaration(
            evaluation_id="source-absolute-accuracy",
            state_id="source-state",
            task_id="source-task",
            dataset_id="source-validation",
            metric_name="absolute_accuracy",
            aggregation="example_mean",
            checkpoint_rule="declared-checkpoint",
        )
        target_evaluation = EvaluationDeclaration(
            evaluation_id="target-absolute-accuracy",
            state_id="target-state",
            task_id="target-task",
            dataset_id="target-validation",
            metric_name="absolute_accuracy",
            aggregation="example_mean",
            checkpoint_rule="declared-checkpoint",
        )
        probe = np.asarray([[0.0]])
        source_output = updated.infer_for_evaluation(
            probe, source_evaluation, selector
        )
        target_output = updated.infer_for_evaluation(
            probe, target_evaluation, selector
        )
        self.assertEqual(selector.select(source_evaluation).head_id, "source-head")
        self.assertEqual(selector.batch_norm_track(target_evaluation), "target-bn")
        self.assertNotEqual(float(source_output[0, 0]), float(target_output[0, 0]))

    def test_selector_rejects_task_or_policy_state_mismatch(self) -> None:
        ownership = BatchNormOwnership(
            channel_slot_ids=("slot-0",),
            buffer_policy="frozen",
            source_task_id="source-task",
            target_task_id="target-task",
            source_track_id="source-bn",
        )
        wrong_track = StateDeclaration(
            state_id="target-state",
            task_id="target-task",
            head_id="target-head",
            batch_norm_track_id="target-bn",
            slot_ids=("slot-0",),
            slot_mask=(1.0,),
        )
        with self.assertRaises(ValueError):
            TaskStateSelector((wrong_track,), batch_norm_ownership=ownership)

        source_state = StateDeclaration(
            state_id="source-state",
            task_id="source-task",
            head_id="source-head",
            batch_norm_track_id="source-bn",
            slot_ids=("slot-0",),
            slot_mask=(1.0,),
        )
        selector = TaskStateSelector((source_state,), batch_norm_ownership=ownership)
        mismatched = EvaluationDeclaration(
            evaluation_id="bad-evaluation",
            state_id="source-state",
            task_id="target-task",
            dataset_id="validation",
            metric_name="accuracy",
            aggregation="example_mean",
            checkpoint_rule="fixed",
        )
        with self.assertRaises(ValueError):
            selector.select(mismatched)

    def test_nonfinite_inputs_and_nonbinary_masks_fail_closed(self) -> None:
        bad_capacity = dict(self.capacities)
        bad_capacity["h1:0"] = np.nan
        with self.assertRaises(ValueError):
            FixedWidthDeftStateMachine(self.layers, bad_capacity)
        negative_capacity = dict(self.capacities)
        negative_capacity["h1:0"] = -1.0
        with self.assertRaises(ValueError):
            FixedWidthDeftStateMachine(self.layers, negative_capacity)
        with self.assertRaises(ValueError):
            source_owned_parameter_masks(
                ("slot",), (SlotState.CORE,), input_width=0, output_width=1
            )
        with self.assertRaises(ValueError):
            clear_optimizer_state(
                {"p": np.ones(1)}, {"p": np.asarray([0.5])}
            )
        with self.assertRaises(ValueError):
            momentumw_step(
                {"p": np.ones(1)},
                {"p": np.asarray([np.inf])},
                {"p": np.zeros(1)},
                {"p": np.ones(1)},
                learning_rate=0.1,
                momentum=0.9,
                weight_decay=0.1,
            )
        with self.assertRaises(ValueError):
            BatchNormBuffers(np.asarray([0.0]), np.asarray([-1.0]))


if __name__ == "__main__":
    unittest.main()
