import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.pipeline_runtime import Toolchain
from utils.plan_model import resolve_file_plan
from utils.runner.models import QueueItem
from utils.runner.session import SessionController
from utils.runner_state import STAGE_AUTOBOOST_PSD_SCENE, STAGE_AUTOBOOST_SCENE, STAGE_FASTPASS


def write_video_plan(path: Path, extra_primary: str = "", extra_pipeline: str = "") -> None:
    path.write_text(
        "\n".join(
            [
                'source = "source.mkv"',
                "quality = 30",
                "",
                "[paths]",
                'workdir = "work"',
                "",
                "[video.primary]",
                extra_primary.strip(),
                "",
                "[video.pipeline]",
                extra_pipeline.strip(),
                "",
                "[video]",
                "track_id = 0",
                'action = "edit"',
                "",
                "[meta]",
                'name = "sample"',
                'plan_type = "file"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_commands(plan_path: Path, *, mode: str = "fastpass") -> list[tuple[str, list[str]]]:
    resolved = resolve_file_plan(plan_path)
    controller = object.__new__(SessionController)
    controller.toolchain = Toolchain(
        python_exe=sys.executable,
        vs_python_exe=sys.executable,
        av1an_exe="av1an",
        psd_script="Progressive-Scene-Detection.py",
    )
    controller.av1an_fork_enabled = False
    controller.av1an_progress_jsonl_enabled = False

    item = QueueItem(resolved=resolved, mode=mode)
    return controller._build_item_commands(item)


def build_fastpass_command(plan_path: Path) -> list[str]:
    commands = build_commands(plan_path)
    for stage, cmd in commands:
        if stage == STAGE_FASTPASS:
            return cmd
    raise AssertionError("Fastpass command was not built")


class AutoBoostPlanDefaultsTest(unittest.TestCase):
    def test_omitted_ab_tuning_does_not_emit_auto_boost_flags(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "source.plan"
            write_video_plan(plan_path)

            resolved = resolve_file_plan(plan_path)
            primary = resolved.plan.video.primary
            self.assertIsNone(primary.ab_multiplier)
            self.assertIsNone(primary.ab_pos_dev)
            self.assertIsNone(primary.ab_neg_dev)

            cmd = build_fastpass_command(plan_path)
            self.assertNotIn("-a", cmd)
            self.assertNotIn("--max-positive-dev", cmd)
            self.assertNotIn("--max-negative-dev", cmd)

    def test_explicit_ab_split_multipliers_are_not_shadowed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "source.plan"
            write_video_plan(
                plan_path,
                """
ab_pos_multiplier = "1.25"
ab_neg_multiplier = "0.85"
""",
            )

            cmd = build_fastpass_command(plan_path)
            self.assertNotIn("-a", cmd)
            self.assertIn("--pos-dev-multiplier", cmd)
            self.assertIn("--neg-dev-multiplier", cmd)
            self.assertEqual(cmd[cmd.index("--pos-dev-multiplier") + 1], "1.25")
            self.assertEqual(cmd[cmd.index("--neg-dev-multiplier") + 1], "0.85")

    def test_explicit_ab_multiplier_keeps_auto_boost_precedence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "source.plan"
            write_video_plan(
                plan_path,
                """
ab_multiplier = 0.7
ab_pos_multiplier = "1.25"
ab_neg_multiplier = "0.85"
ab_pos_dev = 5
ab_neg_dev = 4
""",
            )

            cmd = build_fastpass_command(plan_path)
            self.assertIn("-a", cmd)
            self.assertEqual(cmd[cmd.index("-a") + 1], "0.7")
            self.assertNotIn("--pos-dev-multiplier", cmd)
            self.assertNotIn("--neg-dev-multiplier", cmd)
            self.assertIn("--max-positive-dev", cmd)
            self.assertIn("--max-negative-dev", cmd)

    def test_av1an_scene_detection_knobs_are_emitted(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "source.plan"
            write_video_plan(
                plan_path,
                extra_pipeline="""
scene_detection = "av1an"
av1an_extra_split_sec = 22
av1an_min_scene_len = 48
""",
            )

            cmd = build_fastpass_command(plan_path)

            self.assertEqual(cmd[cmd.index("--sdm") + 1], "av1an")
            self.assertEqual(cmd[cmd.index("--av1an-extra-split-sec") + 1], "22")
            self.assertEqual(cmd[cmd.index("--av1an-min-scene-len") + 1], "48")

    def test_none_scene_detection_builds_single_scene_stage_for_no_fastpass(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "source.plan"
            write_video_plan(
                plan_path,
                extra_pipeline="""
scene_detection = "none"
no_fastpass = true
""",
            )

            commands = build_commands(plan_path)

            stage, cmd = next((item for item in commands if item[0] == STAGE_AUTOBOOST_SCENE), (None, []))
            self.assertEqual(stage, STAGE_AUTOBOOST_SCENE)
            self.assertEqual(cmd[cmd.index("--sdm") + 1], "none")
            self.assertEqual(cmd[cmd.index("--run-stages") + 1], "none,base-scenes")

    def test_psd_scene_detection_knobs_are_forwarded_as_psd_args(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "source.plan"
            write_video_plan(
                plan_path,
                extra_pipeline="""
scene_detection = "psd"
psd_scene_detection_extra_split = 333
psd_scene_detection_min_scene_len = 21
psd_scene_detection_18_target_split = 35
psd_scene_detection_12_target_split = 101
psd_scene_detection_27_extra_target_split = 165
""",
            )

            commands = build_commands(plan_path)

            stage, cmd = next((item for item in commands if item[0] == STAGE_AUTOBOOST_PSD_SCENE), (None, []))
            self.assertEqual(stage, STAGE_AUTOBOOST_PSD_SCENE)
            self.assertEqual(cmd[cmd.index("--sdm") + 1], "psd")
            psd_args = cmd[cmd.index("--psd-args") + 1]
            self.assertIn("--scene-detection-extra-split 333", psd_args)
            self.assertIn("--scene-detection-min-scene-len 21", psd_args)
            self.assertIn("--scene-detection-18-target-split 35", psd_args)
            self.assertNotIn("--scene-detection-vapoursynth-range", psd_args)


if __name__ == "__main__":
    unittest.main()
