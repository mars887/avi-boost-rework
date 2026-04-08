from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional

from utils.pipeline_runtime import AUTOBOOST_DIR, ROOT_DIR, UTILS_DIR, ensure_dir, is_mars_av1an_fork, load_toolchain
from utils.plan_model import BatchPlan, FilePlan, ResolvedFilePlan, RunnerEvent, load_plan, resolve_file_plan

FAST_INTERRUPT = False


@dataclass(frozen=True)
class QueueItem:
    resolved: ResolvedFilePlan
    mode: str

    @property
    def plan_path(self) -> Path:
        return self.resolved.paths.plan_path

    @property
    def source(self) -> Path:
        return self.resolved.paths.source

    @property
    def workdir(self) -> Path:
        return self.resolved.paths.workdir

    @property
    def name(self) -> str:
        return self.resolved.plan.meta.name or self.source.stem


def normalize_mode(value: str) -> str:
    return "fastpass" if str(value or "").strip().lower() == "fastpass" else "full"


def av1an_encoder_name(value: str) -> str:
    return "x265" if str(value or "").strip().lower() in ("libx265", "x265") else "svt-av1"


def resolve_optional_path(raw_value: str, plan_path: Path) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (plan_path.parent / path).resolve()
    return str(path)


def build_queue(plan_args: List[str], cli_mode: str) -> List[QueueItem]:
    queue: List[QueueItem] = []
    seen: set[str] = set()

    def visit(path: Path, inherited_mode: str) -> None:
        plan_path = path.expanduser().resolve()
        plan = load_plan(plan_path)
        if isinstance(plan, FilePlan):
            key = str(plan_path).lower()
            if key in seen:
                return
            seen.add(key)
            mode = normalize_mode(cli_mode or inherited_mode or plan.meta.mode or "full")
            queue.append(QueueItem(resolved=resolve_file_plan(plan_path), mode=mode))
            return

        batch_mode = normalize_mode(cli_mode or plan.meta.mode or inherited_mode)
        for item in plan.items:
            nested = Path(item.plan).expanduser()
            if not nested.is_absolute():
                nested = (plan_path.parent / nested).resolve()
            visit(nested, batch_mode)

    for raw in plan_args:
        visit(Path(raw), cli_mode or "")
    return queue


class SessionController:
    def __init__(
        self,
        *,
        items: List[QueueItem],
        events_jsonl: str,
        no_source_bitrate: bool,
        exit_when_idle: bool,
    ) -> None:
        self.toolchain = load_toolchain()
        self.av1an_fork_enabled = is_mars_av1an_fork(self.toolchain.av1an_exe)
        self.queue: Deque[QueueItem] = deque(items)
        self.failed: List[QueueItem] = []
        self.completed: List[QueueItem] = []
        self.current: Optional[QueueItem] = None
        self.current_stage = ""
        self.last_item: Optional[QueueItem] = None
        self.events_jsonl = Path(events_jsonl).expanduser().resolve() if events_jsonl else None
        self.no_source_bitrate = bool(no_source_bitrate)
        self.pause_after_current = False
        self.paused = False
        self.exit_when_idle = bool(exit_when_idle)
        self.rerun_after_current = False
        self.stop_requested = False
        self.worker_done = False
        self.lock = threading.Lock()
        self.wake_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_main, name="runner-worker", daemon=True)

    def start(self) -> None:
        self.worker.start()
        self.wake_event.set()

    def join(self) -> None:
        self.worker.join()

    def is_idle(self) -> bool:
        with self.lock:
            return self.current is None and not self.queue

    def is_busy(self) -> bool:
        with self.lock:
            return self.current is not None

    def is_finished(self) -> bool:
        with self.lock:
            return self.worker_done

    def status_text(self) -> str:
        with self.lock:
            current = f"{self.current.name} [{self.current_stage or 'pending'}]" if self.current is not None else "-"
            queued = len(self.queue)
            failed = len(self.failed)
            completed = len(self.completed)
            paused = "yes" if self.paused else "no"
            pause_after = "yes" if self.pause_after_current else "no"
            exit_idle = "yes" if self.exit_when_idle else "no"
        return (
            f"current: {current}\n"
            f"queued: {queued}\n"
            f"completed: {completed}\n"
            f"failed: {failed}\n"
            f"paused: {paused}\n"
            f"pause_after_current: {pause_after}\n"
            f"exit_when_idle: {exit_idle}"
        )

    def request_pause_after_current(self) -> None:
        with self.lock:
            self.pause_after_current = True

    def resume(self) -> None:
        with self.lock:
            self.paused = False
            self.pause_after_current = False
        self.wake_event.set()

    def retry_failed(self) -> None:
        with self.lock:
            if not self.failed:
                return
            for item in self.failed:
                self.queue.append(item)
            self.failed.clear()
            self.paused = False
        self.wake_event.set()

    def rerun_current_item(self) -> None:
        with self.lock:
            if self.current is not None:
                self.rerun_after_current = True
            elif self.last_item is not None:
                self.queue.appendleft(self.last_item)
                self.paused = False
                self.wake_event.set()

    def request_exit_when_idle(self) -> None:
        with self.lock:
            self.exit_when_idle = True
        self.wake_event.set()

    def request_stop(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.exit_when_idle = True
            self.paused = False
            self.pause_after_current = False
            self.rerun_after_current = False
            self.queue.clear()
        self.wake_event.set()

    def _write_item_state(self, item: QueueItem, event: RunnerEvent) -> None:
        meta_dir = item.workdir / "00_meta"
        ensure_dir(meta_dir)
        state = {
            "plan": str(item.plan_path),
            "source": str(item.source),
            "mode": item.mode,
            "stage": event.stage,
            "status": event.status,
            "message": event.message,
            "timestamp": event.timestamp,
        }
        (meta_dir / "runner_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def _emit(self, item: QueueItem, stage: str, status: str, message: str = "") -> None:
        event = RunnerEvent(
            event="runner",
            plan=str(item.plan_path),
            mode=item.mode,
            stage=stage,
            status=status,
            message=message,
            timestamp=time.time(),
        )
        text = f"[runner] {item.name} | {stage} | {status}"
        if message:
            text += f" | {message}"
        print(text, flush=True)

        meta_dir = item.workdir / "00_meta"
        ensure_dir(meta_dir)
        event_line = json.dumps(event.__dict__, ensure_ascii=False)
        with (meta_dir / "runner_events.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(event_line + "\n")
        if self.events_jsonl is not None:
            self.events_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.events_jsonl.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(event_line + "\n")
        self._write_item_state(item, event)

    def _run_stage(self, item: QueueItem, stage: str, cmd: List[str]) -> None:
        with self.lock:
            self.current_stage = stage
        self._emit(item, stage, "started")
        print("[cmd]", subprocess.list2cmdline(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT_DIR)).returncode
        if rc != 0:
            self._emit(item, stage, "failed", f"exit_code={rc}")
            raise RuntimeError(f"{stage}_failed_rc_{rc}")
        self._emit(item, stage, "completed")

    def _build_item_commands(self, item: QueueItem) -> List[tuple[str, List[str]]]:
        plan = item.resolved.plan
        paths = item.resolved.paths
        primary = plan.video.primary
        details = plan.video.details
        plan_path = str(paths.plan_path)
        workdir = paths.workdir
        log_dir = workdir / "00_logs"
        ensure_dir(log_dir)
        ensure_dir(workdir / "00_meta")
        ensure_dir(workdir / "audio")
        ensure_dir(workdir / "video")
        ensure_dir(workdir / "sub")
        ensure_dir(workdir / "attachments")
        ensure_dir(workdir / "chapters")
        ensure_dir(paths.zone_file.parent)
        if not paths.zone_file.exists():
            paths.zone_file.write_text("", encoding="utf-8", newline="\n")

        commands: List[tuple[str, List[str]]] = [
            (
                "demux",
                [
                    self.toolchain.python_exe,
                    str(UTILS_DIR / "demux.py"),
                    "--plan",
                    plan_path,
                    "--log",
                    str(log_dir / "01_demux.log"),
                ],
            ),
            (
                "attachments-cleaner",
                [
                    self.toolchain.python_exe,
                    str(UTILS_DIR / "attachments-cleaner.py"),
                    "--subs",
                    str(workdir / "sub"),
                    "--attachments",
                    str(workdir / "attachments"),
                    "--log",
                    str(log_dir / "02_att_clean.log"),
                ],
            ),
        ]

        if item.mode == "fastpass" and not item.resolved.has_video_edit():
            return commands

        if item.resolved.has_video_edit():
            fast_vpy = resolve_optional_path(details.fast_vpy, paths.plan_path)
            main_vpy = resolve_optional_path(details.main_vpy, paths.plan_path)
            proxy_vpy = resolve_optional_path(details.proxy_vpy, paths.plan_path)
            auto_boost_cmd = [
                self.toolchain.vs_python_exe,
                str(AUTOBOOST_DIR / "auto_boost.py"),
                "--av1an",
                self.toolchain.av1an_exe,
                "--input",
                str(paths.source),
                "--out-scenes",
                str(workdir / "video" / "scenes.json"),
                "--temp",
                str(workdir / "video"),
                "--log",
                str(log_dir / "03_autoboost.log"),
                "--sdm",
                str(primary.scene_detection or "av1an"),
                "--encoder",
                str(primary.encoder),
                "--workers",
                str(int(primary.fastpass_workers)),
                "--av1an-log-file",
                str(log_dir / "03.1_fastpass.log"),
                "--av1an-log-level",
                "info",
                "--quality",
                str(primary.quality),
                "-v",
                item.resolved.build_fastpass_params_text(),
                "--final-override",
                item.resolved.build_mainpass_params_text(),
                "--keep",
                "--verbose",
            ]
            if self.av1an_fork_enabled:
                auto_boost_cmd.extend(["--chunk-order", str(primary.chunk_order or "")])
                if str(primary.encoder_path or "").strip():
                    auto_boost_cmd.extend(["--encoder-path", str(primary.encoder_path)])
                if FAST_INTERRUPT:
                    auto_boost_cmd.append("--fast-interrupt")
            if str(primary.scene_detection or "").strip().lower() == "psd":
                auto_boost_cmd.extend(["--psd-script", self.toolchain.psd_script])
            if primary.no_fastpass:
                auto_boost_cmd.append("--no-fastpass")
            if primary.fastpass_hdr:
                auto_boost_cmd.append("--fastpass-hdr")
            if fast_vpy:
                auto_boost_cmd.extend(["--fastpass-vpy", fast_vpy])
            if proxy_vpy:
                auto_boost_cmd.extend(["--fastpass-proxy", proxy_vpy])
            if primary.fastpass_preset:
                auto_boost_cmd.extend(["--fast-preset", str(primary.fastpass_preset)])
            if primary.preset:
                auto_boost_cmd.extend(["--preset", str(primary.preset)])
            if str(primary.ab_multiplier).strip():
                auto_boost_cmd.extend(["-a", str(primary.ab_multiplier)])
            elif str(primary.ab_pos_multiplier).strip() and str(primary.ab_neg_multiplier).strip():
                auto_boost_cmd.extend(["--pos-dev-multiplier", str(primary.ab_pos_multiplier)])
                auto_boost_cmd.extend(["--neg-dev-multiplier", str(primary.ab_neg_multiplier)])
            if str(primary.ab_pos_dev).strip():
                auto_boost_cmd.extend(["--max-positive-dev", str(primary.ab_pos_dev)])
            if str(primary.ab_neg_dev).strip():
                auto_boost_cmd.extend(["--max-negative-dev", str(primary.ab_neg_dev)])
            if details.fastpass_filter:
                auto_boost_cmd.extend(["-f", str(details.fastpass_filter)])
            if item.mode == "fastpass":
                auto_boost_cmd.append("--stop-before-stage4")
            commands.append(("auto-boost", auto_boost_cmd))

            if item.mode == "full":
                hdr_cmd = [
                    self.toolchain.vs_python_exe,
                    str(UTILS_DIR / "av1an_hdr_metadata_patch_v2.py"),
                    "--source",
                    str(paths.source),
                    "--scenes",
                    str(workdir / "video" / "scenes.json"),
                    "--output",
                    str(workdir / "video" / "scenes-hdr.json"),
                    "--workdir",
                    str(workdir / "video" / "hdr_tmp"),
                    "--encoder",
                    str(primary.encoder),
                    "--log",
                    str(log_dir / "04_hdr_patch.log"),
                ]
                if primary.strict_sdr_8bit:
                    hdr_cmd.append("--no-hdr10")
                if primary.no_hdr10plus or primary.strict_sdr_8bit:
                    hdr_cmd.append("--no-hdr10plus")
                if primary.no_dolby_vision or primary.strict_sdr_8bit:
                    hdr_cmd.append("--no-dv")
                commands.append(("hdr-patch", hdr_cmd))

                commands.append(
                    (
                        "zone-editor",
                        [
                            self.toolchain.vs_python_exe,
                            str(UTILS_DIR / "zone-editor.py"),
                            "--source",
                            str(paths.source),
                            "--scenes",
                            str(workdir / "video" / "scenes-hdr.json"),
                            "--out",
                            str(workdir / "video" / "scenes-final.json"),
                            "--command",
                            str(paths.zone_file),
                            "--log",
                            str(log_dir / "05_zone_edit.log"),
                        ],
                    )
                )

                main_input = main_vpy or str(paths.source)
                mainpass_cmd = [
                    self.toolchain.av1an_exe,
                    "-i",
                    main_input,
                    "-o",
                    str(workdir / "video" / "video-final.mkv"),
                    "--scenes",
                    str(workdir / "video" / "scenes-final.json"),
                    "--workers",
                    str(int(primary.mainpass_workers)),
                    "--temp",
                    str(workdir / "video" / "mainpass"),
                    "-n",
                    "--keep",
                    "--verbose",
                    "--resume",
                    "--cache-mode",
                    "temp",
                    "--log-file",
                    str(log_dir / "06_av1an_mainpass.log"),
                    "--log-level",
                    "info",
                    "--chunk-method",
                    "ffms2",
                    "-e",
                    av1an_encoder_name(primary.encoder),
                    "--pix-format",
                    "yuv420p" if primary.strict_sdr_8bit else "yuv420p10le",
                    "--no-defaults",
                    "-a=-an -sn",
                ]
                if self.av1an_fork_enabled:
                    if str(primary.chunk_order or "").strip():
                        mainpass_cmd.extend(["--chunk-order", str(primary.chunk_order)])
                    if str(primary.encoder_path or "").strip():
                        mainpass_cmd.extend(["--encoder-path", str(primary.encoder_path)])
                    if FAST_INTERRUPT:
                        mainpass_cmd.append("--fast-interrupt")
                if proxy_vpy:
                    mainpass_cmd.extend(["--proxy", proxy_vpy])
                if main_vpy or proxy_vpy:
                    mainpass_cmd.extend(["--vspipe-args", f"src={paths.source}"])
                if details.mainpass_filter:
                    mainpass_cmd.extend(["-f", f"-vf {details.mainpass_filter}"])
                commands.append(("mainpass", mainpass_cmd))

        if item.mode == "full":
            commands.append(
                (
                    "audio",
                    [
                        self.toolchain.python_exe,
                        str(UTILS_DIR / "audio-tool-v2.py"),
                        "--plan",
                        plan_path,
                        "--copy-container",
                        "mka",
                        "--no-preserve-special",
                        "--log",
                        str(log_dir / "07_audio.log"),
                    ],
                )
            )
            commands.append(
                (
                    "verify",
                    [
                        self.toolchain.python_exe,
                        str(UTILS_DIR / "verify.py"),
                        "--plan",
                        plan_path,
                        "--log",
                        str(log_dir / "08_verify.log"),
                    ],
                )
            )
            mux_cmd = [
                self.toolchain.python_exe,
                str(UTILS_DIR / "mux.py"),
                "--plan",
                plan_path,
                "--log",
                str(log_dir / "09_mux.log"),
            ]
            if self.no_source_bitrate:
                mux_cmd.append("--no-source-bitrate")
            commands.append(("mux", mux_cmd))
        return commands

    def _process_item(self, item: QueueItem) -> None:
        output_path = item.source.parent / f"{item.source.stem}-av1.mkv"
        if item.mode == "full" and output_path.exists():
            self._emit(item, "item", "skipped", f"output_exists={output_path.name}")
            return
        if item.mode == "fastpass" and not item.resolved.has_video_edit():
            self._emit(item, "item", "skipped", "fastpass_mode_without_video_edit")
            return

        self._emit(item, "item", "started")
        for stage, cmd in self._build_item_commands(item):
            self._run_stage(item, stage, cmd)
            if item.mode == "fastpass" and stage == "auto-boost":
                break
        self._emit(item, "item", "completed")

    def _worker_main(self) -> None:
        while True:
            with self.lock:
                if self.stop_requested and self.current is None and not self.queue:
                    self.worker_done = True
                    return
                if self.current is None and not self.queue:
                    if self.exit_when_idle:
                        self.worker_done = True
                        return
                    should_wait = True
                elif self.paused:
                    should_wait = True
                else:
                    should_wait = False
                    item = self.queue.popleft()
                    self.current = item
                    self.current_stage = ""
            if should_wait:
                self.wake_event.wait(0.25)
                self.wake_event.clear()
                continue

            assert self.current is not None
            item = self.current
            try:
                self._process_item(item)
            except Exception as exc:
                self._emit(item, "item", "failed", str(exc))
                with self.lock:
                    self.failed.append(item)
            else:
                with self.lock:
                    self.completed.append(item)
            finally:
                with self.lock:
                    self.last_item = item
                    if self.rerun_after_current:
                        self.queue.appendleft(item)
                        self.rerun_after_current = False
                    if self.pause_after_current:
                        self.paused = True
                        self.pause_after_current = False
                    self.current = None
                    self.current_stage = ""
                self.wake_event.set()


def print_help() -> None:
    print(
        "Commands:\n"
        "  status\n"
        "  pause after current\n"
        "  resume\n"
        "  retry failed\n"
        "  rerun current item\n"
        "  exit when idle\n"
        "  quit",
        flush=True,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Session runner for file and batch .plan files.")
    parser.add_argument("--mode", choices=["full", "fastpass"], default="")
    parser.add_argument("--events-jsonl", default="")
    parser.add_argument("--no-source-bitrate", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--exit-when-idle", action="store_true")
    parser.add_argument("plans", nargs="+")
    args = parser.parse_args(argv)

    queue = build_queue(args.plans, args.mode)
    if not queue:
        print("[runner] no plans resolved", file=sys.stderr)
        return 1

    print(f"[runner] queue size: {len(queue)}", flush=True)
    for index, item in enumerate(queue, start=1):
        print(f"  {index}. {item.mode} | {item.plan_path}", flush=True)

    controller = SessionController(
        items=queue,
        events_jsonl=args.events_jsonl,
        no_source_bitrate=args.no_source_bitrate,
        exit_when_idle=(args.exit_when_idle or args.no_interactive),
    )
    controller.start()

    if args.no_interactive:
        controller.join()
        return 1 if controller.failed else 0

    print_help()
    while not controller.is_finished():
        try:
            raw = input("runner> ").strip().lower()
        except EOFError:
            controller.request_exit_when_idle()
            break
        if raw in ("", "status"):
            print(controller.status_text(), flush=True)
            continue
        if raw == "pause after current":
            controller.request_pause_after_current()
            print("[runner] will pause after current item", flush=True)
            continue
        if raw == "resume":
            controller.resume()
            print("[runner] resumed", flush=True)
            continue
        if raw == "retry failed":
            controller.retry_failed()
            print("[runner] failed items re-queued", flush=True)
            continue
        if raw == "rerun current item":
            controller.rerun_current_item()
            print("[runner] rerun requested", flush=True)
            continue
        if raw == "exit when idle":
            controller.request_exit_when_idle()
            print("[runner] will exit when idle", flush=True)
            continue
        if raw in ("quit", "exit"):
            if controller.is_busy():
                controller.request_exit_when_idle()
                print("[runner] busy; will exit when idle", flush=True)
            else:
                controller.request_stop()
            continue
        if raw == "help":
            print_help()
            continue
        print("[runner] unknown command", flush=True)

    controller.join()
    return 1 if controller.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
