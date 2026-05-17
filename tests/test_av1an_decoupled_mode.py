import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.pipeline_runtime import Toolchain
from utils.plan_model import load_file_plan, resolve_file_plan, save_plan
from utils.runner.models import QueueItem
from utils.runner.session import SessionController
from utils.runner_state import STAGE_MAINPASS


def write_decoupled_plan(path: Path, *, enabled: bool = True) -> None:
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
                "mainpass_workers = 11",
                "",
                "[video.experimental]",
                f"av1an_decoupled_enabled = {'true' if enabled else 'false'}",
                "source_workers = 1",
                "encoder_workers = 3",
                'raw_spool_limit = "30G"',
                'raw_spool_dir = "spool"',
                'raw_spool_min_free_ram = "10G"',
                "use_disk_cache = true",
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


def build_mainpass_command(plan_path: Path, *, fork_enabled: bool) -> list[str]:
    resolved = resolve_file_plan(plan_path)
    controller = object.__new__(SessionController)
    controller.toolchain = Toolchain(
        python_exe=sys.executable,
        vs_python_exe=sys.executable,
        av1an_exe="av1an",
        psd_script="Progressive-Scene-Detection.py",
    )
    controller.av1an_fork_enabled = fork_enabled
    controller.av1an_progress_jsonl_enabled = False
    controller.add_source_bitrate = False

    item = QueueItem(resolved=resolved, mode="full")
    commands = controller._build_item_commands(item)
    for stage, cmd in commands:
        if stage == STAGE_MAINPASS:
            return cmd
    raise AssertionError("Mainpass command was not built")


class Av1anDecoupledModeTest(unittest.TestCase):
    def test_plan_round_trips_decoupled_settings(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "source.plan"
            write_decoupled_plan(plan_path)

            plan = load_file_plan(plan_path)
            experimental = plan.video.experimental
            self.assertTrue(experimental.av1an_decoupled_enabled)
            self.assertEqual(experimental.source_workers, "1")
            self.assertEqual(experimental.encoder_workers, "3")
            self.assertEqual(experimental.raw_spool_limit, "30G")
            self.assertEqual(experimental.raw_spool_dir, "spool")
            self.assertEqual(experimental.raw_spool_min_free_ram, "10G")
            self.assertTrue(experimental.use_disk_cache)

            saved_path = root / "saved.plan"
            save_plan(plan, saved_path)
            saved = saved_path.read_text(encoding="utf-8")
            self.assertIn("av1an_decoupled_enabled = true", saved)
            self.assertIn("source_workers = 1", saved)
            self.assertIn("encoder_workers = 3", saved)
            self.assertIn('raw_spool_limit = "30G"', saved)
            self.assertIn('raw_spool_dir = "spool"', saved)
            self.assertIn('raw_spool_min_free_ram = "10G"', saved)
            self.assertIn("use_disk_cache = true", saved)

    def test_runner_emits_decoupled_mainpass_flags(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "source.plan"
            write_decoupled_plan(plan_path)

            cmd = build_mainpass_command(plan_path, fork_enabled=True)

            self.assertNotIn("--workers", cmd)
            self.assertEqual(cmd[cmd.index("--source-workers") + 1], "1")
            self.assertEqual(cmd[cmd.index("--encoder-workers") + 1], "3")
            self.assertEqual(cmd[cmd.index("--raw-spool-limit") + 1], "30G")
            self.assertEqual(cmd[cmd.index("--raw-spool-dir") + 1], str((root / "spool").resolve()))
            self.assertEqual(cmd[cmd.index("--raw-spool-min-free-ram") + 1], "10G")
            self.assertEqual(cmd[cmd.index("--use-disk-cache") + 1], "true")

    def test_runner_keeps_mainpass_workers_when_fork_flags_are_unavailable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "source.plan"
            write_decoupled_plan(plan_path)

            cmd = build_mainpass_command(plan_path, fork_enabled=False)

            self.assertNotIn("--source-workers", cmd)
            self.assertIn("--workers", cmd)
            self.assertEqual(cmd[cmd.index("--workers") + 1], "11")


if __name__ == "__main__":
    unittest.main()
