from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from utils.discord_config import discord_config_value

from .api import RunnerLaunchConfig, RunnerRuntime
from .integrations import attach_discord_integrations


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Session runner for file and batch .plan files.")
    parser.add_argument("--mode", choices=["full", "fastpass"], default="")
    parser.add_argument("--events-jsonl", default="")
    parser.add_argument("--add-source-bitrate", action="store_true", help="Include source bitrate metadata in mux output.")
    parser.add_argument(
        "--no-source-bitrate",
        dest="add_source_bitrate",
        action="store_false",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--exit-when-idle", action="store_true")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--gui", dest="gui", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="Run the legacy console runner.")
    discord_group = parser.add_mutually_exclusive_group()
    discord_group.add_argument(
        "--discord",
        dest="discord_verbose",
        action="store_true",
        default=False,
        help="Enable Discord integration and print Discord connection errors.",
    )
    discord_group.add_argument(
        "--no-discord",
        dest="discord_enabled",
        action="store_false",
        default=True,
        help="Disable registration in the local Discord bot service.",
    )
    parser.add_argument(
        "--discord-service-url",
        default=discord_config_value("PBBATCH_DISCORD_SERVICE_URL", "http://127.0.0.1:8794"),
    )
    parser.add_argument(
        "--discord-shared-secret",
        default=discord_config_value("PBBATCH_DISCORD_SHARED_SECRET", ""),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("plans", nargs="*")
    return parser


def runtime_config_from_args(args: argparse.Namespace) -> RunnerLaunchConfig:
    return RunnerLaunchConfig(
        plans=list(args.plans or []),
        mode=str(args.mode or ""),
        events_jsonl=str(args.events_jsonl or ""),
        add_source_bitrate=bool(args.add_source_bitrate),
        exit_when_idle=bool(args.exit_when_idle),
        no_interactive=bool(args.no_interactive),
        session_id=str(args.session_id or ""),
    )


def attach_cli_integrations(runtime: RunnerRuntime, args: argparse.Namespace) -> None:
    if bool(getattr(args, "discord_enabled", True)) and len(runtime.source_dirs) > 1:
        print(f"[discord] multi-folder runner: {len(runtime.source_dirs)} folder sessions will be published", flush=True)
    bridges = attach_discord_integrations(
        runtime,
        service_url=str(getattr(args, "discord_service_url", "") or ""),
        shared_secret=str(getattr(args, "discord_shared_secret", "") or ""),
        enabled=bool(getattr(args, "discord_enabled", True)),
        verbose=bool(getattr(args, "discord_verbose", False)),
        logger=lambda message: print(message, flush=True),
    )
    if bool(getattr(args, "discord_enabled", True)):
        print(f"[discord] enabled: session_id={runtime.session_id}", flush=True)
        print(f"[discord] service_url={getattr(args, 'discord_service_url', '')}", flush=True)
        for bridge in bridges:
            print(f"[discord] source_dir={bridge.source_dir} | discord_session_id={bridge.session_id}", flush=True)
    else:
        print("[discord] disabled", flush=True)


def run_headless(args: argparse.Namespace) -> int:
    if not args.plans:
        print("[runner] no plans provided; use GUI mode or pass at least one .plan with --no-gui", file=sys.stderr)
        return 2
    try:
        runtime = RunnerRuntime(runtime_config_from_args(args))
    except Exception as exc:
        print(f"[runner] {exc}", file=sys.stderr)
        return 1
    if not runtime.queue:
        print("[runner] no plans resolved", file=sys.stderr)
        return 1

    print(f"[runner] queue size: {len(runtime.queue)}", flush=True)
    for index, item in enumerate(runtime.queue, start=1):
        print(f"  {index}. {item.mode} | {item.plan_path}", flush=True)

    attach_cli_integrations(runtime, args)
    try:
        runtime.start()
    except Exception as exc:
        runtime.close()
        print(f"[runner] {exc}", file=sys.stderr)
        return 2
    for folder_lock in runtime.folder_locks:
        print(f"[runner] folder lock acquired: {folder_lock.path}", flush=True)
    if bool(getattr(args, "discord_enabled", True)):
        connected_count = sum(1 for integration in runtime.integrations if getattr(integration, "connected", False))
        if connected_count == len(runtime.integrations):
            print("[discord] runner registered in bot service", flush=True)
        elif bool(getattr(args, "discord_verbose", False)):
            print(
                f"[discord] registered {connected_count}/{len(runtime.integrations)} folder sessions; "
                "runner will keep working locally",
                flush=True,
            )

    if args.no_interactive:
        code = runtime.join()
        if bool(getattr(args, "discord_enabled", True)):
            print("[discord] bridge stopped", flush=True)
        print("[runner] folder lock released", flush=True)
        return code

    controller = runtime.controller

    def request_interrupt_shutdown() -> None:
        if controller.is_busy():
            runtime.stop()
            print("[runner] interrupt received; stopping active work", flush=True)
        else:
            runtime.stop()
            print("[runner] interrupt received; stopping", flush=True)

    print_help()
    try:
        try:
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
                        runtime.stop()
                    continue
                if raw == "help":
                    print_help()
                    continue
                print("[runner] unknown command", flush=True)
        except KeyboardInterrupt:
            request_interrupt_shutdown()
        try:
            code = runtime.join()
        except KeyboardInterrupt:
            request_interrupt_shutdown()
            code = runtime.join()
        return code
    finally:
        runtime.close()
        if bool(getattr(args, "discord_enabled", True)):
            print("[discord] bridge stopped", flush=True)
        print("[runner] folder lock released", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.gui:
        try:
            from .gui import run_runner_gui
        except ImportError as exc:
            print(f"[runner-gui] PySide6 GUI is unavailable: {exc}", file=sys.stderr)
            print("[runner-gui] Install it with: python -m pip install \"PySide6>=6.10,<7\"", file=sys.stderr)
            return 2
        return run_runner_gui(args)
    return run_headless(args)

