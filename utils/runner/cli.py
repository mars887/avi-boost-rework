from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.discord_bridge import DiscordBridge
from utils.discord_config import discord_config_value
from utils.runner_lock import SourceDirLock
from utils.runner_source_info import prepare_source_info

from .helpers import build_queue
from .session import SessionController

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
    parser.add_argument("plans", nargs="+")
    args = parser.parse_args(argv)

    queue = build_queue(args.plans, args.mode)
    if not queue:
        print("[runner] no plans resolved", file=sys.stderr)
        return 1
    source_dirs = sorted({str(item.source.parent.resolve()) for item in queue}, key=str.lower)
    if args.discord_enabled and len(source_dirs) > 1:
        print(f"[discord] multi-folder runner: {len(source_dirs)} folder sessions will be published", flush=True)

    print(f"[runner] queue size: {len(queue)}", flush=True)
    for index, item in enumerate(queue, start=1):
        print(f"  {index}. {item.mode} | {item.plan_path}", flush=True)

    controller = SessionController(
        items=queue,
        events_jsonl=args.events_jsonl,
        add_source_bitrate=args.add_source_bitrate,
        exit_when_idle=(args.exit_when_idle or args.no_interactive),
        session_id=args.session_id,
    )
    def discord_session_id_for_source(source_dir: str) -> str:
        if len(source_dirs) <= 1:
            return controller.session_id
        suffix = hashlib.sha1(source_dir.lower().encode("utf-8", errors="ignore")).hexdigest()[:8]
        return f"{controller.session_id}-{suffix}"

    bridges: List[tuple[str, DiscordBridge]] = []
    if args.discord_enabled:
        for source_dir in source_dirs:
            discord_session_id = discord_session_id_for_source(source_dir)
            bridge = DiscordBridge(
                service_url=args.discord_service_url,
                session_id=discord_session_id,
                enabled=True,
                shared_secret=args.discord_shared_secret,
            )
            bridge.attach(
                snapshot_provider=lambda sd=source_dir, sid=discord_session_id: controller.snapshot(sd, session_id=sid),
                command_handler=lambda command, sd=source_dir: controller.handle_command(command, source_dir=sd),
            )
            if args.discord_verbose:
                bridge.set_error_callback(
                    lambda message, sd=source_dir: print(f"[discord] bridge unavailable for {sd}: {message}", flush=True)
                )
            bridges.append((source_dir, bridge))

        bridge_by_source = {source_dir: bridge for source_dir, bridge in bridges}

        def notify_discord_event(payload: Dict[str, Any], _snapshot: Dict[str, Any]) -> None:
            source = str(payload.get("source") or "")
            source_dir = str(Path(source).parent.resolve()) if source else ""
            bridge = bridge_by_source.get(source_dir)
            if bridge is None:
                return
            bridge.notify_event(payload, controller.snapshot(source_dir, session_id=bridge.session_id))

        controller.add_event_sink(notify_discord_event)
    folder_locks = [SourceDirLock(source_dir=source_dir, session_id=controller.session_id, enabled=True) for source_dir in source_dirs]
    if args.discord_enabled:
        print(f"[discord] enabled: session_id={controller.session_id}", flush=True)
        print(f"[discord] service_url={args.discord_service_url}", flush=True)
        for source_dir, bridge in bridges:
            print(f"[discord] source_dir={source_dir} | discord_session_id={bridge.session_id}", flush=True)
    else:
        print("[discord] disabled", flush=True)
    acquired_locks: List[SourceDirLock] = []
    try:
        for folder_lock in folder_locks:
            folder_lock.acquire()
            acquired_locks.append(folder_lock)
    except RuntimeError as exc:
        for folder_lock in acquired_locks:
            folder_lock.release()
        print(f"[runner] {exc}", file=sys.stderr)
        return 2
    for folder_lock in folder_locks:
        print(f"[runner] folder lock acquired: {folder_lock.path}", flush=True)
    try:
        for item in queue:
            prepare_source_info(item)
    except Exception as exc:
        for folder_lock in folder_locks:
            folder_lock.release()
        print(f"[runner] source info preparation failed: {exc}", file=sys.stderr)
        return 2
    controller.start()
    for _, bridge in bridges:
        bridge.start()
    if args.discord_enabled:
        connected_count = sum(1 for _, bridge in bridges if bridge.connected)
        if connected_count == len(bridges):
            print("[discord] runner registered in bot service", flush=True)
        elif args.discord_verbose:
            print(f"[discord] registered {connected_count}/{len(bridges)} folder sessions; runner will keep working locally", flush=True)

    if args.no_interactive:
        try:
            controller.join()
            return 1 if controller.failed else 0
        finally:
            for _, bridge in bridges:
                bridge.stop()
            for folder_lock in folder_locks:
                folder_lock.release()
            if args.discord_enabled:
                print("[discord] bridge stopped", flush=True)
            print("[runner] folder lock released", flush=True)

    def request_interrupt_shutdown() -> None:
        if controller.is_busy():
            controller.request_stop()
            print("[runner] interrupt received; stopping active work", flush=True)
        else:
            controller.request_stop()
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
                        controller.request_stop()
                    continue
                if raw == "help":
                    print_help()
                    continue
                print("[runner] unknown command", flush=True)
        except KeyboardInterrupt:
            request_interrupt_shutdown()
        try:
            controller.join()
        except KeyboardInterrupt:
            request_interrupt_shutdown()
            controller.join()
        return 1 if controller.failed else 0
    finally:
        for _, bridge in bridges:
            bridge.stop()
        for folder_lock in folder_locks:
            folder_lock.release()
        if args.discord_enabled:
            print("[discord] bridge stopped", flush=True)
        print("[runner] folder lock released", flush=True)
