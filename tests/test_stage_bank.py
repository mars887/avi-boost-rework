import unittest
import threading
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

from runner import (
    ActivePlanState,
    DEFAULT_STAGE_BANK,
    PlanExecution,
    SessionController,
    StageBankConfig,
    StageBankStage,
    StageState,
    STAGE_ATTACHMENTS,
    STAGE_AUDIO,
    STAGE_AUTOBOOST_SCENE,
    STAGE_FASTPASS,
    STAGE_HDR_PATCH,
    STAGE_MAINPASS,
    STAGE_MUX,
    STAGE_SSIMU2,
    STAGE_VERIFY,
    STAGE_ZONE_EDIT,
    effective_stage_dependencies,
    downstream_stage_names,
    load_stage_bank_config,
)


class StageBankConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_stage_bank_config()

    def test_parallelism_limits_are_loaded(self) -> None:
        self.assertEqual(self.config.capacity, 10)
        self.assertEqual(self.config.max_active_plans, 3)
        self.assertEqual(self.config.max_running_stages, 5)

    def test_stage_priorities_are_loaded(self) -> None:
        self.assertEqual(self.config.stage_priority(STAGE_FASTPASS), 3)
        self.assertEqual(self.config.stage_priority(STAGE_MAINPASS), 3)
        self.assertEqual(self.config.stage_priority(STAGE_SSIMU2), 2)
        self.assertEqual(self.config.stage_priority(STAGE_VERIFY), 1)
        self.assertEqual(self.config.stage_priority(STAGE_MUX), 1)

    def test_verify_waits_for_attachments_audio_and_mainpass(self) -> None:
        self.assertEqual(
            set(self.config.stage_requires(STAGE_VERIFY)),
            {STAGE_ATTACHMENTS, STAGE_AUDIO, STAGE_MAINPASS},
        )

    def test_hdr_patch_depends_on_zone_edit_only(self) -> None:
        self.assertEqual(self.config.stage_requires(STAGE_HDR_PATCH), (STAGE_ZONE_EDIT,))

    def test_missing_dependencies_are_ignored_for_plan_stage_set(self) -> None:
        stage_names = [STAGE_FASTPASS, STAGE_SSIMU2, STAGE_ZONE_EDIT, STAGE_HDR_PATCH]
        self.assertEqual(
            effective_stage_dependencies(STAGE_FASTPASS, stage_names, self.config),
            [],
        )
        self.assertEqual(
            effective_stage_dependencies(STAGE_ZONE_EDIT, stage_names, self.config),
            [STAGE_SSIMU2],
        )

    def test_downstream_uses_configured_dag(self) -> None:
        stage_names = [
            STAGE_AUTOBOOST_SCENE,
            STAGE_FASTPASS,
            STAGE_SSIMU2,
            STAGE_ZONE_EDIT,
            STAGE_HDR_PATCH,
            STAGE_MAINPASS,
        ]
        self.assertEqual(
            downstream_stage_names(STAGE_ZONE_EDIT, stage_names, self.config),
            [STAGE_HDR_PATCH, STAGE_MAINPASS],
        )

    def test_default_stage_costs_are_validated_against_custom_capacity(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "StagesBankTree.toml"
            path.write_text(
                "[bank]\n"
                "capacity = 5\n"
                "max_active_plans = 3\n"
                "max_running_stages = 5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "stage cost exceeds bank capacity: Fastpass"):
                load_stage_bank_config(path)

    def test_custom_stage_priority_is_validated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "StagesBankTree.toml"
            path.write_text(
                "[bank]\n"
                "capacity = 10\n"
                "max_active_plans = 3\n"
                "max_running_stages = 5\n"
                "\n"
                "[stages.Fastpass]\n"
                "cost = 10\n"
                "priority = 0\n"
                "requires = []\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "stage priority must be greater than zero: Fastpass"):
                load_stage_bank_config(path)

    def test_stage_dependency_cycles_are_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "StagesBankTree.toml"
            path.write_text(
                "[bank]\n"
                "capacity = 10\n"
                "max_active_plans = 3\n"
                "max_running_stages = 5\n"
                "\n"
                "[stages.Fastpass]\n"
                "cost = 10\n"
                "priority = 2\n"
                'requires = ["SSIMU2 Metrics"]\n'
                "\n"
                '[stages."SSIMU2 Metrics"]\n'
                "cost = 5\n"
                "priority = 2\n"
                'requires = ["Fastpass"]\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "stage bank dependency cycle"):
                load_stage_bank_config(path)

    def test_ready_stage_candidates_use_priority_before_cost(self) -> None:
        stages = dict(DEFAULT_STAGE_BANK.stages)
        stages[STAGE_FASTPASS] = StageBankStage(cost=10, priority=2)
        stages[STAGE_AUDIO] = StageBankStage(cost=2, priority=1)
        controller = object.__new__(SessionController)
        controller.stage_bank = StageBankConfig(
            capacity=10,
            max_active_plans=3,
            max_running_stages=5,
            stages=stages,
        )
        controller.lock = threading.Lock()
        controller.stop_requested = False
        controller.paused_by_source = {}
        controller.pause_after_current_by_source = {}
        item = type("DummyItem", (), {"source_dir": "source-a"})()
        execution = PlanExecution(
            plan_run_id="plan-a",
            item=item,
            commands=[
                (STAGE_FASTPASS, ["fastpass"]),
                (STAGE_AUDIO, ["audio"]),
            ],
        )
        controller.executions = {"plan-a": execution}
        controller.active = {"plan-a": ActivePlanState(plan_run_id="plan-a", item=item, started_at=1.0)}

        candidates = controller._ready_stage_candidates()

        self.assertEqual([candidate[-1] for candidate in candidates], [STAGE_AUDIO, STAGE_FASTPASS])

    def test_queued_item_with_active_workdir_is_not_activated(self) -> None:
        controller = object.__new__(SessionController)
        controller.stage_bank = DEFAULT_STAGE_BANK
        controller.lock = threading.Lock()
        controller.stop_requested = False
        controller.running_stage_tasks = {}
        controller.paused_by_source = {}
        controller.pause_after_current_by_source = {}

        active_item = type("DummyItem", (), {"source_dir": "source-a", "workdir": Path("same-workdir")})()
        same_workdir = type("DummyItem", (), {"source_dir": "source-b", "workdir": Path("same-workdir")})()
        other_workdir = type("DummyItem", (), {"source_dir": "source-c", "workdir": Path("other-workdir")})()
        controller.active = {
            "plan-a": ActivePlanState(plan_run_id="plan-a", item=active_item, started_at=1.0),
        }
        controller.queue = deque([same_workdir, other_workdir])
        activated = []

        def activate(item):
            activated.append(item)
            return True

        controller._activate_item = activate

        self.assertTrue(controller._activate_next_queued_item())
        self.assertEqual(activated, [other_workdir])
        self.assertEqual(list(controller.queue), [same_workdir])

    def test_stop_cancels_pending_execution_as_failed(self) -> None:
        controller = object.__new__(SessionController)
        controller.lock = threading.Lock()
        controller.stop_requested = True
        controller.running_stage_tasks = {}
        item = type("DummyItem", (), {})()
        execution = PlanExecution(
            plan_run_id="plan-a",
            item=item,
            commands=[(STAGE_FASTPASS, ["fastpass"])],
        )
        controller.executions = {"plan-a": execution}
        controller.active = {
            "plan-a": ActivePlanState(
                plan_run_id="plan-a",
                item=item,
                started_at=1.0,
                stages=[StageState(STAGE_FASTPASS)],
            ),
        }
        calls = []

        def finish(plan_run_id, *, final_status, failed_stage="", message=""):
            calls.append((plan_run_id, final_status, failed_stage, message))

        controller._finish_plan_execution = finish

        self.assertTrue(controller._cancel_stopped_pending_executions())
        self.assertEqual(calls, [("plan-a", "failed", "Item", "stop_requested")])


if __name__ == "__main__":
    unittest.main()
