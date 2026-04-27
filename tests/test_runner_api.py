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
from utils.runner.session import SessionController
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
            controller._refresh_running_stage_progress = lambda _item: None
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


if __name__ == "__main__":
    unittest.main()
