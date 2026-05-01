import argparse
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.discord_bridge import DiscordBridge
from utils.runner.api import RunnerLaunchConfig, RunnerRuntime
from utils.runner.cli import build_arg_parser, run_headless
from utils.runner.integrations import DISCORD_SECRET_HEADER, HttpRunnerIntegrationBridge
from utils.runner.logs import RunnerLogLine
from utils.runner.models import ActivePlanState, StageState
from utils.runner.session import SessionController, StageOutputFilter
from utils.runner_state import (
    STAGE_AUTOBOOST_PSD_SCENE,
    STAGE_AUTOBOOST_SCENE,
    STAGE_FASTPASS,
    STAGE_SSIMU2,
)
from utils.runner.terminal import TerminalScreen, has_terminal_repaint, strip_ansi


def write_plan(path: Path, source_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                f'source = "{source_name}"',
                "quality = 30",
                "",
                "[video]",
                "track_id = 0",
                'action = "copy"',
                "",
                "[meta]",
                'name = "sample"',
                'plan_type = "file"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class RunnerCliTest(unittest.TestCase):
    def test_gui_is_default_and_no_gui_is_explicit(self) -> None:
        parser = build_arg_parser()
        self.assertTrue(parser.parse_args(["one.plan"]).gui)
        self.assertFalse(parser.parse_args(["--no-gui", "one.plan"]).gui)

    def test_headless_requires_plans(self) -> None:
        args = argparse.Namespace(plans=[], mode="", no_gui=True)
        self.assertEqual(run_headless(args), 2)


class RunnerRuntimeTest(unittest.TestCase):
    def test_runtime_builds_queue_and_multi_source_session_ids(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_a = root / "a" / "a.mkv"
            source_b = root / "b" / "b.mkv"
            source_a.parent.mkdir()
            source_b.parent.mkdir()
            plan_a = source_a.with_suffix(".plan")
            plan_b = source_b.with_suffix(".plan")
            write_plan(plan_a, source_a.name)
            write_plan(plan_b, source_b.name)

            runtime = RunnerRuntime(RunnerLaunchConfig(plans=[str(plan_a), str(plan_b)]))

            self.assertEqual(len(runtime.queue), 2)
            self.assertEqual(len(runtime.source_dirs), 2)
            session_ids = [runtime.integration_session_id_for_source(source_dir) for source_dir in runtime.source_dirs]
            self.assertTrue(all(session_id.startswith(runtime.session_id) for session_id in session_ids))
            self.assertEqual(len(set(session_ids)), 2)

    def test_runtime_accepts_per_plan_mode_overrides(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_a = root / "a.mkv"
            source_b = root / "b.mkv"
            plan_a = source_a.with_suffix(".plan")
            plan_b = source_b.with_suffix(".plan")
            write_plan(plan_a, source_a.name)
            write_plan(plan_b, source_b.name)

            runtime = RunnerRuntime(
                RunnerLaunchConfig(
                    plans=[str(plan_a), str(plan_b)],
                    plan_modes={str(plan_a): "fastpass", str(plan_b): "full"},
                )
            )

            self.assertEqual([item.mode for item in runtime.queue], ["fastpass", "full"])


class HttpIntegrationBridgeTest(unittest.TestCase):
    def test_notify_event_enqueues_matching_source_payload(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "source.mkv"
            bridge = HttpRunnerIntegrationBridge(
                service_url="http://127.0.0.1:1",
                session_id="session-a",
                source_dir=str(root),
                enabled=True,
            )
            event = {"event": "runner", "source": str(source), "stage": "Demux"}
            snapshot = {"session_id": "session-a"}

            bridge.notify_event(event, snapshot)

            item = bridge.outbox.get_nowait()
            self.assertEqual(item["path"], "/api/sessions/session-a/events")
            self.assertEqual(item["payload"]["event"], event)
            self.assertEqual(item["payload"]["snapshot"], snapshot)

    def test_discord_bridge_uses_discord_secret_header(self) -> None:
        bridge = DiscordBridge(
            service_url="http://127.0.0.1:1",
            session_id="session-a",
            enabled=True,
            shared_secret="secret",
        )
        self.assertEqual(bridge._auth_headers(), {DISCORD_SECRET_HEADER: "secret"})


class RunnerLogCaptureTest(unittest.TestCase):
    def test_run_stage_captures_stdout_to_sink_and_file(self) -> None:
        import utils.runner.session as session_module

        class FakeStdout:
            def __init__(self) -> None:
                self.lines = ["hello\n", "done\n", ""]

            def readline(self) -> str:
                return self.lines.pop(0)

        class FakeProc:
            def __init__(self, *_args, **_kwargs) -> None:
                self.stdout = FakeStdout()
                self.returncode = None
                self.poll_count = 0
                self.terminated = False

            def poll(self):
                self.poll_count += 1
                if self.poll_count >= 2:
                    self.returncode = 0
                    return 0
                return None

            def terminate(self) -> None:
                self.terminated = True
                self.returncode = 1

            def kill(self) -> None:
                self.returncode = 1

        with TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            item = type(
                "Item",
                (),
                {
                    "name": "sample",
                    "source": workdir / "source.mkv",
                    "plan_path": workdir / "source.plan",
                    "workdir": workdir,
                },
            )()
            controller = object.__new__(SessionController)
            controller.session_id = "session-a"
            controller.lock = threading.Lock()
            controller.current_stage = ""
            controller.current_plan_run_id = "run-a"
            controller.running_stage_processes = {}
            controller.stop_requested = False
            controller.log_sinks = []
            lines: list[RunnerLogLine] = []
            controller.add_log_sink(lines.append)
            controller._stage_cached_message = lambda _item, _stage: ""
            controller._active_state_for_item = lambda _item: type("Active", (), {"plan_run_id": "run-a"})()
            controller._emit = lambda *_args, **_kwargs: None
            controller._forward_child_events = lambda _path, offset, _item: offset
            controller._refresh_running_stage_progress = lambda _item, **_kwargs: None
            controller._notify_event_sinks = lambda _payload: None

            original_popen = session_module.subprocess.Popen
            original_valid = session_module.stage_completion_artifacts_valid
            original_write_marker = session_module.write_stage_marker
            try:
                session_module.subprocess.Popen = FakeProc
                session_module.stage_completion_artifacts_valid = lambda _item, _stage: True
                session_module.write_stage_marker = lambda _item, _stage: None
                controller._run_stage(item, "Fake Stage", ["fake", "command"])
            finally:
                session_module.subprocess.Popen = original_popen
                session_module.stage_completion_artifacts_valid = original_valid
                session_module.write_stage_marker = original_write_marker

            self.assertTrue(any(line.text == "hello" for line in lines))
            capture_logs = list((workdir / "00_logs").glob("runner_capture_Fake_Stage_run-a.log"))
            self.assertEqual(len(capture_logs), 1)
            self.assertIn("done", capture_logs[0].read_text(encoding="utf-8"))


class TerminalScreenTest(unittest.TestCase):
    def test_decodes_visible_ansi_and_detects_repaint(self) -> None:
        self.assertEqual(strip_ansi(r"\x1b[31mred\x1b[0m"), "red")
        self.assertTrue(has_terminal_repaint("\x1b[2Kprogress\r"))

    def test_cursor_rewrite_updates_current_progress_line(self) -> None:
        screen = TerminalScreen(max_lines=20)
        screen.feed(
            "Queue 13\r\n"
            "[Chunk 10]\r\n"
            "00:00:00 [9/22 Chunks] 59%"
            "\x1b[2A\x1b[2K[Chunk 11]\r\n"
            "\x1b[2K00:00:01 [10/22 Chunks] 60%"
        )
        self.assertEqual(screen.current_nonempty_plain(), "00:00:01 [10/22 Chunks] 60%")

    def test_windows_progress_shade_glyphs_are_preserved(self) -> None:
        screen = TerminalScreen(max_lines=20)
        screen.feed("█▓▒░ ▁▂▃▄▅▆▇")

        self.assertEqual(screen.latest_nonempty_plain(), "█▓▒░ ▏▎▍▌▋▊▉")


class Av1anProgressParseTest(unittest.TestCase):
    def test_parse_scene_detection_progress_jsonl(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scene.progress.jsonl"
            path.write_text(
                '{"event":"progress","percent":38.0,"pos":773,"total":2047,'
                '"fps":134.42,"eta":"9s"}\n',
                encoding="utf-8",
            )
            controller = object.__new__(SessionController)

            payload = controller._last_av1an_progress_from_jsonl(path)

        self.assertEqual(payload.get("progress"), 38.0)
        self.assertEqual(payload.get("pos"), 773)
        self.assertEqual(payload.get("total"), 2047)
        self.assertEqual(payload.get("fps"), 134.42)
        self.assertEqual(payload.get("eta"), "9s")

    def test_parse_estimated_size_from_console_progress(self) -> None:
        payload = SessionController._parse_av1an_progress_text(
            "[9/22 Chunks] 00:00:00 59% 1282/2181 "
            "(0 fps, eta unknown, 794.8 Kbps, est. 8.62 MiB)"
        )

        self.assertEqual(payload.get("progress"), 59.0)
        self.assertEqual(payload.get("chunks_done"), 9)
        self.assertEqual(payload.get("chunks_total"), 22)
        self.assertEqual(payload.get("kbps"), 794.8)
        self.assertEqual(payload.get("estimated_size"), "8.62 MiB")

    def test_parse_current_progress_jsonl_schema(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.jsonl"
            path.write_text(
                '{"event":"progress","ts":1710000000.0,"percent":39.0,"pos":28115,'
                '"total":72469,"fps":11.89,"eta":"62m","kbps":1624.3,'
                '"est_size":"1.42 GiB","est_size_bytes":1524713390,"elapsed":"39m 42s",'
                '"chunks_done":127,"chunks_total":343}\n',
                encoding="utf-8",
            )
            controller = object.__new__(SessionController)

            payload = controller._last_av1an_progress_from_jsonl(path)

        self.assertEqual(payload.get("progress"), 39.0)
        self.assertEqual(payload.get("pos"), 28115)
        self.assertEqual(payload.get("total"), 72469)
        self.assertEqual(payload.get("fps"), 11.89)
        self.assertEqual(payload.get("kbps"), 1624.3)
        self.assertEqual(payload.get("eta"), "62m")
        self.assertEqual(payload.get("estimated_size"), "1.42 GiB")
        self.assertEqual(payload.get("estimated_size_bytes"), 1524713390)
        self.assertEqual(payload.get("progress_elapsed"), "39m 42s")
        self.assertEqual(payload.get("chunks_done"), 127)
        self.assertEqual(payload.get("chunks_total"), 343)


class RunnerChildProgressTest(unittest.TestCase):
    def test_final_progress_jsonl_refresh_updates_stage_details(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            log_dir = workdir / "00_logs"
            log_dir.mkdir()
            (log_dir / "03.1_fastpass.progress.jsonl").write_text(
                '{"event":"progress","ts":1777387384.2337778,"percent":100.0,'
                '"pos":36612,"total":36612,"fps":56.73,"eta":"0s","kbps":1232.0,'
                '"est_size":"224.27 MiB","est_size_bytes":235167413,"elapsed":"11m",'
                '"chunks_done":383,"chunks_total":383}\n',
                encoding="utf-8",
            )
            item = type(
                "Item",
                (),
                {
                    "name": "sample",
                    "source": workdir / "source.mkv",
                    "plan_path": workdir / "source.plan",
                    "workdir": workdir,
                    "mode": "fastpass",
                },
            )()
            active = ActivePlanState(
                plan_run_id="run-a",
                item=item,
                stages=[StageState(name=STAGE_FASTPASS, status="started", started_at=10.0)],
            )
            controller = object.__new__(SessionController)
            controller.lock = threading.Lock()
            controller.active = {"run-a": active}

            controller._refresh_running_stage_progress(item, stage_names=[STAGE_FASTPASS])

            stage = active.stage(STAGE_FASTPASS)
            self.assertEqual(stage.progress, 100.0)
            self.assertEqual(stage.details.get("fps"), 56.73)
            self.assertEqual(stage.details.get("kbps"), 1232.0)
            self.assertEqual(stage.details.get("eta"), "0s")
            self.assertEqual(stage.details.get("estimated_size"), "224.27 MiB")
            self.assertEqual(stage.details.get("estimated_size_bytes"), 235167413)
            self.assertEqual(stage.details.get("chunks_done"), 383)
            self.assertEqual(stage.details.get("chunks_total"), 383)

    def test_child_progress_updates_stage_details_without_changing_status(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            item = type(
                "Item",
                (),
                {
                    "name": "sample",
                    "source": workdir / "source.mkv",
                    "plan_path": workdir / "source.plan",
                    "workdir": workdir,
                    "mode": "fastpass",
                },
            )()
            active = ActivePlanState(
                plan_run_id="run-a",
                item=item,
                stages=[StageState(name=STAGE_SSIMU2, status="started", started_at=10.0)],
            )
            controller = object.__new__(SessionController)
            controller.session_id = "session-a"
            controller.lock = threading.Lock()
            controller.event_io_lock = threading.Lock()
            controller.active = {"run-a": active}
            controller.event_sinks = []
            controller._write_item_state = lambda *_args, **_kwargs: None

            controller._ingest_child_event(
                item,
                {
                    "event": "runner_child",
                    "plan_run_id": "run-a",
                    "stage": STAGE_SSIMU2,
                    "status": "progress",
                    "progress": 42.0,
                    "details": {"fps": 77.12, "eta": "9s", "ssimu2": 84.79},
                    "timestamp": 20.0,
                },
            )

            stage = active.stage(STAGE_SSIMU2)
            self.assertEqual(stage.status, "started")
            self.assertEqual(stage.progress, 42.0)
            self.assertEqual(stage.details.get("fps"), 77.12)
            self.assertEqual(stage.details.get("eta"), "9s")
            self.assertEqual(stage.details.get("ssimu2"), 84.79)


class StageOutputFilterTest(unittest.TestCase):
    def test_psd_filter_throttles_frames_and_keeps_leaf_scenes(self) -> None:
        output_filter = StageOutputFilter(STAGE_AUTOBOOST_PSD_SCENE)
        text = "\n".join(
            [
                "Frame 0 / Detecting scenes / 0.00 fps",
                "Frame 1 / Detecting scenes / 53.65 fps",
                "Scene [    0:34456] / Creating scenes",
                "Scene [    0:   61] / Creating scenes",
                "Scene [   61:   97] / Creating scenes",
                "Scene [   97: 2470] / Creating scenes",
                "Scene [   97:  133] / Creating scenes",
            ]
        ) + "\n"

        ui_text, capture_text = output_filter.process("stdout", text, 100.0)
        flush_ui, flush_capture = output_filter.flush(110.0, final=True)

        self.assertIn("Frame 0 / Detecting scenes", ui_text)
        self.assertNotIn("Frame 1 / Detecting scenes", ui_text)
        self.assertNotIn("Scene [    0:34456]", flush_capture)
        self.assertIn("Scene [    0:   61] / Creating scenes", flush_capture)
        self.assertIn("Scene [   61:   97] / Creating scenes", flush_capture)
        self.assertIn("Scene [   97:  133] / Creating scenes", flush_ui)

    def test_psd_progress_repaints_one_console_line(self) -> None:
        output_filter = StageOutputFilter(STAGE_AUTOBOOST_PSD_SCENE)
        first_ui, _first_capture = output_filter.process(
            "stdout",
            "Frame 1 / Detecting scenes / 53.65 fps\r",
            100.0,
        )
        second_ui, _second_capture = output_filter.process(
            "stdout",
            "Frame 2 / Detecting scenes / 54.40 fps\r",
            103.0,
        )
        screen = TerminalScreen(max_lines=20)

        screen.feed(first_ui)
        screen.feed(second_ui)

        lines = [line for line in screen.plain_lines() if line.strip()]
        self.assertEqual(lines, ["[progress] Frame 2 / Detecting scenes / 54.40 fps"])
        self.assertTrue(first_ui.endswith("\r"))
        self.assertTrue(second_ui.endswith("\r"))

    def test_fastpass_capture_is_snapshot_throttled(self) -> None:
        output_filter = StageOutputFilter(STAGE_FASTPASS)

        first_ui, first_capture = output_filter.process("pty", "00:00 10%\r", 100.0)
        second_ui, second_capture = output_filter.process("pty", "00:01 11%\r", 101.0)
        final_ui, final_capture = output_filter.flush(102.0, final=True)

        self.assertTrue(first_ui)
        self.assertIn("[progress]", first_capture)
        self.assertTrue(second_ui)
        self.assertEqual(second_capture, "")
        self.assertEqual(final_ui, "")
        self.assertIn("11%", final_capture)

    def test_av1an_scene_capture_is_snapshot_throttled(self) -> None:
        output_filter = StageOutputFilter(STAGE_AUTOBOOST_SCENE)

        first_ui, first_capture = output_filter.process(
            "pty",
            "00:00:05 38% 773/2047 (134.42 fps, eta 9s)\r",
            100.0,
        )
        second_ui, second_capture = output_filter.process(
            "pty",
            "00:00:06 39% 798/2047 (135.00 fps, eta 8s)\r",
            101.0,
        )

        self.assertTrue(first_ui)
        self.assertIn("[progress]", first_capture)
        self.assertTrue(second_ui)
        self.assertEqual(second_capture, "")


if __name__ == "__main__":
    unittest.main()
